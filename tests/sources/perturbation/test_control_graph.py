"""Control-graph refusals, each exercised by breaking something on purpose.

House discipline: a guard nobody has watched fire is indistinguishable from one
that cannot fire.  Every refusal below is paired with an input under which the
same guard reads the *other* way, so none of these assertions is decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.sources.lineage import Lineage
from scwbd.sources.perturbation.control_graph import (
    ControlGraph,
    ControlGraphError,
    ControlVariable,
    ExposureInterval,
    Provenance,
)


def _exposures(n: int = 4, *, known_duration: bool = False) -> ExposureInterval:
    return ExposureInterval(
        onset_sample=np.arange(n) * 1000,
        onset_s=np.arange(n) * 0.05,
        duration_s=(np.full(n, 1e-4) if known_duration else np.full(n, np.nan)),
        clock_id="ds004024.brainvision_amp",
        duration_provenance=Provenance.RECORDED if known_duration else Provenance.UNKNOWN,
    )


def _graph(manipulated, *, participant="sub-CON001", record_id="r") -> ControlGraph:
    return ControlGraph(
        source_id="ds004024",
        record_id=record_id,
        lineage=Lineage(participant=participant, family=f"singleton:{participant}"),
        manipulated=tuple(manipulated),
        exposures=_exposures(),
    )


# -- absence must never acquire a stand-in value ------------------------
def test_unknown_variable_may_not_carry_a_value():
    with pytest.raises(ValueError, match="unknown must not be given a stand-in value"):
        ControlVariable(name="coil_pose", provenance=Provenance.UNKNOWN, value=(0, 0, 0))
    # the same constructor accepts the same field when it is genuinely known,
    # so the guard discriminates rather than banning the field outright
    ok = ControlVariable(
        name="coil_pose", provenance=Provenance.RECORDED, value=(0, 0, 0)
    )
    assert ok.value == (0, 0, 0)


def test_derived_variable_must_name_its_method():
    with pytest.raises(ValueError, match="names no method"):
        ControlVariable(
            name="stimulated_hemisphere",
            provenance=Provenance.DERIVED,
            value="left",
            levels_present=("left",),
        )
    ok = ControlVariable(
        name="stimulated_hemisphere",
        provenance=Provenance.DERIVED,
        value="left",
        levels_present=("left",),
        method="lateralised MEP log ratio",
    )
    assert ok.method


def test_unknown_exposure_width_may_not_be_stored_as_a_number():
    with pytest.raises(ValueError, match="must not be stored as a number"):
        ExposureInterval(
            onset_sample=np.arange(3),
            onset_s=np.arange(3) / 10.0,
            duration_s=np.zeros(3),  # a zero-width exposure asserts it was never on
            clock_id="c",
            duration_provenance=Provenance.UNKNOWN,
        )
    ok = _exposures(known_duration=True)
    assert np.all(np.isfinite(ok.duration_s))


# -- supports() must be able to answer both ways -----------------------
def test_state_dependence_refused_when_declared_but_supported_when_recorded():
    declared = _graph(
        [
            ControlVariable(
                name="block_timepoint",
                provenance=Provenance.DECLARED,
                levels_present=(),
            )
        ]
    )
    ok, reason = declared.supports("state_dependence")
    assert ok is False
    assert "not attached to any run" in reason

    # the SAME check reads True once a per-record label exists with >= 2 levels
    recorded = _graph(
        [
            ControlVariable(
                name="block_timepoint",
                provenance=Provenance.RECORDED,
                value="before",
                levels_present=("before", "after10", "after60"),
            )
        ]
    )
    assert recorded.supports("state_dependence")[0] is True


def test_dose_refused_for_one_level_and_for_unknown_absolute_dose():
    one_level = _graph(
        [
            ControlVariable(
                name="intensity_pct_rmt",
                provenance=Provenance.DECLARED,
                value=100.0,
                levels_present=(100.0,),
            ),
            ControlVariable(name="realised_dose", provenance=Provenance.UNKNOWN),
        ]
    )
    ok, reason = one_level.supports("dose")
    assert ok is False and "1 level present" in reason

    # two levels is enough for a GAIN slope ...
    two_levels = _graph(
        [
            ControlVariable(
                name="intensity_pct_rmt",
                provenance=Provenance.DECLARED,
                value=None,
                levels_present=(100.0, 110.0),
            ),
            ControlVariable(name="realised_dose", provenance=Provenance.UNKNOWN),
        ]
    )
    assert two_levels.supports("gain")[0] is True
    # ... but NOT for absolute dose while the rMT is undistributed
    ok, reason = two_levels.supports("dose")
    assert ok is False and "resting motor threshold" in reason

    with_abs = _graph(
        [
            ControlVariable(
                name="intensity_pct_rmt",
                provenance=Provenance.DECLARED,
                levels_present=(100.0, 110.0),
            ),
            ControlVariable(
                name="realised_dose",
                provenance=Provenance.RECORDED,
                value=42.0,
                units="V/m",
                levels_present=(42.0, 46.0),
            ),
        ]
    )
    assert with_abs.supports("dose")[0] is True


def test_direction_needs_two_levels():
    one = _graph(
        [
            ControlVariable(
                name="stimulated_hemisphere",
                provenance=Provenance.DERIVED,
                value="left",
                levels_present=("left",),
                method="MEP",
            )
        ]
    )
    assert one.supports("direction")[0] is False
    two = _graph(
        [
            ControlVariable(
                name="stimulated_hemisphere",
                provenance=Provenance.DERIVED,
                levels_present=("left", "right"),
                method="MEP",
            )
        ]
    )
    assert two.supports("direction")[0] is True


def test_supports_refuses_unknown_quantities_by_default():
    g = _graph([])
    ok, reason = g.supports("efficacy")
    assert ok is False and "refuses by default" in reason


def test_require_raises_and_names_the_missing_fields():
    g = _graph([ControlVariable(name="coil_pose", provenance=Provenance.UNKNOWN)])
    with pytest.raises(ControlGraphError) as exc:
        g.require("dose")
    assert exc.value.quantity == "dose"
    assert "coil_pose" in exc.value.missing


# -- combining ---------------------------------------------------------
def test_combine_refuses_to_cross_participants():
    a = _graph([], participant="sub-CON001", record_id="a")
    b = _graph([], participant="sub-CON006", record_id="b")
    with pytest.raises(ControlGraphError, match="across participants"):
        ControlGraph.combine([a, b], record_id="design")
    # within a participant it succeeds
    c = _graph([], participant="sub-CON001", record_id="c")
    assert ControlGraph.combine([a, c], record_id="design").record_id == "design"


def test_combine_unions_levels_and_counts_unlabelled_records():
    def hemi(level):
        return ControlVariable(
            name="stimulated_hemisphere",
            provenance=Provenance.DERIVED if level else Provenance.UNKNOWN,
            value=level,
            levels_present=((level,) if level else ()),
            method="MEP" if level else None,
        )

    graphs = [
        _graph([hemi("left")], record_id="r1"),
        _graph([hemi("right")], record_id="r2"),
        _graph([hemi(None)], record_id="r3"),  # unrecoverable on this run
    ]
    merged = ControlGraph.combine(graphs, record_id="design")
    v = merged.variable("stimulated_hemisphere")
    # the unrecovered run does not downgrade the factor ...
    assert set(v.levels_present) == {"left", "right"}
    assert v.provenance is Provenance.DERIVED
    assert merged.supports("direction")[0] is True
    # ... but it is counted, not silently dropped
    assert v.evidence["n_records_unlabelled"] == 1
    assert "unlabelled on this factor" in v.note


def test_recovery_report_covers_every_quantity_g4_enumerates():
    from scwbd.sources.perturbation.control_graph import G4_RECOVERY_QUANTITIES

    rep = _graph([]).recovery_report()
    assert set(rep) == set(G4_RECOVERY_QUANTITIES)
    assert all("reason" in v and v["reason"] for v in rep.values())
