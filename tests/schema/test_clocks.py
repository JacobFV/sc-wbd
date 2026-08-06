"""Clock specs and the update-trigger IR (thesis sec. 4.5).

Synchronization is forced "when a fast transition changes the support of a
slower process".  A bare period cannot express that, so triggers are a small
declarative expression language the compiler can inspect statically.
"""

from __future__ import annotations

import pytest

from scwbd.schema import (
    AllOf,
    AnyOf,
    BoundaryTrigger,
    ClockSpec,
    EventTrigger,
    NotTrigger,
    PeriodicTrigger,
    ScheduleContext,
)


def test_periodic_trigger_fires_on_its_period():
    trig = PeriodicTrigger(period=0.01)
    assert trig.fires(ScheduleContext(t=0.0))  # never fired before
    assert not trig.fires(ScheduleContext(t=0.005, last_fired=0.0))
    assert trig.fires(ScheduleContext(t=0.010, last_fired=0.0))
    assert trig.periods() == (0.01,)
    assert not trig.is_event_driven()


def test_event_trigger_needs_a_threshold():
    with pytest.raises(ValueError, match="needs gt or lt"):
        EventTrigger(signal="metabolic_demand")


def test_event_trigger_fires_on_threshold_crossing():
    trig = EventTrigger(signal="prediction_error", gt=1.0, min_interval=0.1)
    ctx = ScheduleContext(t=1.0, signals={"prediction_error": 2.0}, last_fired=0.0)
    assert trig.fires(ctx)
    assert trig.is_event_driven()
    # below threshold
    assert not trig.fires(ctx.model_copy(update={"signals": {"prediction_error": 0.5}}))
    # inside the cooldown
    assert not trig.fires(ctx.model_copy(update={"last_fired": 0.95}))
    # unknown signal is never a firing reason
    assert not trig.fires(ScheduleContext(t=1.0, last_fired=0.0))


def test_boundary_trigger():
    trig = BoundaryTrigger(name="stimulus_onset")
    assert trig.fires(ScheduleContext(t=0.0, boundary="stimulus_onset"))
    assert not trig.fires(ScheduleContext(t=0.0, boundary="volume_trigger"))
    assert trig.is_event_driven()


def test_combinators():
    fast = PeriodicTrigger(period=0.001)
    salient = EventTrigger(signal="salience", gt=0.5)
    both = AllOf(children=(fast, salient))
    either = AnyOf(children=(fast, salient))
    ctx = ScheduleContext(t=0.002, signals={"salience": 0.9}, last_fired=0.0)
    assert both.fires(ctx)
    assert either.fires(ctx)
    assert both.is_event_driven() and either.is_event_driven()
    assert set(either.periods()) == {0.001}
    quiet = ctx.model_copy(update={"signals": {"salience": 0.1}})
    assert not both.fires(quiet)
    assert either.fires(quiet)
    assert NotTrigger(child=salient).fires(quiet)


def test_triggers_are_content_hashable_and_serializable():
    trig = AnyOf(
        children=(PeriodicTrigger(period=1.0), BoundaryTrigger(name="volume_trigger"))
    )
    dumped = trig.model_dump(mode="json")
    assert dumped["kind"] == "any_of"
    assert dumped["children"][0]["kind"] == "periodic"
    assert len(trig.content_hash()) == 64


def test_clock_spec_defaults_to_a_periodic_trigger():
    clock = ClockSpec(id="eeg_amp", dt=1e-3, reference="sim", sync_evidence="physical_trigger")
    trig = clock.effective_trigger()
    assert isinstance(trig, PeriodicTrigger)
    assert trig.period == pytest.approx(1e-3)
    assert clock.rate_hz == pytest.approx(1000.0)
    assert not clock.is_master


def test_clock_cannot_reference_itself():
    with pytest.raises(ValueError, match="itself"):
        ClockSpec(id="loop", dt=1e-3, reference="loop")


def test_piecewise_drift_requires_drift():
    with pytest.raises(ValueError, match="piecewise"):
        ClockSpec(id="c", dt=1e-3, piecewise_segments=3, drift=0.0)


def test_unverified_sync_is_representable_but_flagged():
    """R01 refuses it at compile time; the type still lets it be declared."""
    from scwbd.schema import UNVERIFIED_SYNC

    clock = ClockSpec(id="wearable", dt=0.02, reference="sim", sync_evidence="assumed")
    assert clock.sync_evidence in UNVERIFIED_SYNC
