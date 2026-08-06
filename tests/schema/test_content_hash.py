"""Content hash stability and immutability.

Every schema object is content-addressed.  If a hash moves when nothing
semantic changed, lineage grouping (R10) and ABI checks become noise.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scwbd.schema import (
    ClockId,
    FrameId,
    NormalPrior,
    TemporalSupport,
    UncertaintyLedger,
    Unit,
    canonical_json,
)
from scwbd.schema.examples import build_three_region_schema


def test_hash_is_stable_across_identical_constructions():
    a = TemporalSupport(clock=ClockId("eeg_amp"), dt=1e-3, group_delay=1.5e-3)
    b = TemporalSupport(clock="eeg_amp", dt=1e-3, group_delay=1.5e-3)
    assert a.content_hash() == b.content_hash()


def test_hash_is_stable_across_rebuilds_of_the_whole_schema():
    first = build_three_region_schema().content_hash()
    second = build_three_region_schema().content_hash()
    assert first == second
    assert len(first) == 64


def test_hash_changes_when_any_field_changes():
    base = TemporalSupport(clock="eeg_amp", dt=1e-3)
    assert base.content_hash() != TemporalSupport(clock="eeg_amp", dt=2e-3).content_hash()
    assert base.content_hash() != TemporalSupport(clock="meg_amp", dt=1e-3).content_hash()


def test_hash_ignores_keyword_ordering():
    a = UncertaintyLedger(
        variance={"measurement": 1.0, "numerical": 2.0},
        bias_interval=(-1.0, 1.0),
        bias_status="externally_bounded",
        external_bound_source="phantom",
    )
    b = UncertaintyLedger(
        bias_status="externally_bounded",
        external_bound_source="phantom",
        bias_interval=(-1.0, 1.0),
        variance={"numerical": 2.0, "measurement": 1.0},
    )
    assert a.content_hash() == b.content_hash()


def test_canonical_json_is_sorted_and_compact():
    text = canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
    assert text == '{"a":{"c":3,"d":2},"b":1}'
    assert json.loads(text) == {"b": 1, "a": {"c": 3, "d": 2}}


def test_models_are_frozen():
    ledger = UncertaintyLedger(bias_interval=(-1.0, 1.0))
    with pytest.raises(ValidationError):
        ledger.bias_status = "design_estimable"  # type: ignore[misc]


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        TemporalSupport(clock="eeg_amp", dt=1e-3, sneaky=True)  # type: ignore[call-arg]


def test_nested_hash_composes():
    """Changing a nested prior changes the parent hash."""
    a = NormalPrior(loc=0.0, scale=1.0, units=Unit("V"))
    b = NormalPrior(loc=0.0, scale=1.0, units=Unit("mV"))
    assert a.content_hash() != b.content_hash()


def test_frame_and_clock_ids_do_not_collide():
    assert FrameId("eeg_cap") == "eeg_cap"
    with pytest.raises(ValueError):
        ClockId("9lives")  # must start with a letter
    with pytest.raises(ValueError):
        FrameId("has space")
