"""Multirate co-simulation: integer ticks, lazy updates, forced sync, budgets."""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.dynamics import (
    FieldPolicy,
    MultirateScheduler,
    prediction_error_trigger,
    sustained_activity_trigger,
)


def _const(value):
    return lambda t, dt, s: value


# ---------------------------------------------------------------------------
# Clock arithmetic
# ---------------------------------------------------------------------------


def test_integer_tick_periods_and_update_counts():
    s = MultirateScheduler()
    s.register(FieldPolicy("neural", dt=1e-3), _const(1.0))
    s.register(FieldPolicy("vascular", dt=1e-2), _const(2.0))
    s.register(FieldPolicy("gain", dt=1e-1), _const(3.0))
    s.compile()
    assert s.periods == {"neural": 1, "vascular": 10, "gain": 100}
    rep = s.run(t_end=1.0)
    assert rep.n_ticks == 1000
    assert rep.updates == {"neural": 1000, "vascular": 100, "gain": 10}


def test_incommensurable_clock_is_refused_not_rounded():
    s = MultirateScheduler()
    s.register(FieldPolicy("a", dt=1e-3), _const(0.0))
    s.register(FieldPolicy("b", dt=2.5e-3 * 1.13), _const(0.0))
    with pytest.raises(ValueError, match="not an integer multiple"):
        s.compile()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(kind="whenever"), "unknown update kind"),
        (dict(interpolation="spline"), "unknown interpolation"),
        (dict(dt=0.0), "dt must be"),
    ],
)
def test_policy_enumerations_are_enforced(kwargs, match):
    base = dict(name="x", dt=1e-3)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FieldPolicy(**base)


def test_lazy_field_must_declare_its_inputs():
    with pytest.raises(ValueError, match="must declare the inputs"):
        FieldPolicy("lazy_one", dt=1e-2, kind="lazy")


def test_schedule_is_deterministic():
    def build():
        s = MultirateScheduler()
        s.register(FieldPolicy("fast", dt=1e-3), _const(1.0))
        s.register(FieldPolicy("slow", dt=5e-3), _const(1.0))
        return s.run(t_end=0.1)

    a, b = build(), build()
    assert a.updates == b.updates and a.n_ticks == b.n_ticks


# ---------------------------------------------------------------------------
# Lazy updates
# ---------------------------------------------------------------------------


def test_lazy_field_skips_when_inputs_do_not_change_materially(device):
    s = MultirateScheduler()
    driver = {"v": torch.zeros(1, device=device)}

    def drive(t, dt, sched):
        # constant for the first half of the run, then a ramp
        driver["v"] = torch.zeros(1, device=device) if t < 0.05 else torch.full((1,), t, device=device)
        return driver["v"]

    s.register(FieldPolicy("input", dt=1e-3), drive)
    s.register(
        FieldPolicy("lazy", dt=1e-3, kind="lazy", inputs=("input",), materiality=0.01),
        lambda t, dt, sched: sched.read("input"),
    )
    rep = s.run(t_end=0.1)
    assert rep.skipped["lazy"] > 30, "a lazy field must skip while its input is static"
    assert rep.updates["lazy"] > 10, "a lazy field must still update once its input moves"
    assert rep.updates["lazy"] + rep.skipped["lazy"] == 100


# ---------------------------------------------------------------------------
# Forced synchronisation (§4.5)
# ---------------------------------------------------------------------------


def test_sustained_activity_forces_the_metabolic_field(device):
    """A fast transition that changes the support of a slower process forces a sync."""
    s = MultirateScheduler()
    n_calls = {"vascular": 0}

    def neural(t, dt, sched):
        # quiet, then sustained high activity from t = 20 ms
        level = 0.01 if t < 0.02 else 1.0
        return torch.full((3,), level, device=device)

    def vascular(t, dt, sched):
        n_calls["vascular"] += 1
        return torch.zeros(3, device=device)

    s.register(FieldPolicy("neural", dt=1e-3), neural)
    s.register(FieldPolicy("vascular", dt=5e-2, interpolation="linear"), vascular)
    s.add_trigger(
        sustained_activity_trigger("neural", ["vascular"], threshold=0.5, window=5, cooldown_ticks=5)
    )
    rep = s.run(t_end=0.1)
    scheduled = 2  # 100 ms / 50 ms
    assert rep.forced_updates["vascular"] > 0, "sustained activity must force a metabolic update"
    assert rep.updates["vascular"] > scheduled
    assert any(e.trigger.startswith("sustained_activity") for e in rep.events)
    # the forced updates happen off the vascular cadence
    forced_times = [e.time for e in rep.events]
    assert min(forced_times) < 0.05


def test_prediction_error_opens_a_plasticity_window(device):
    s = MultirateScheduler()
    s.register(
        FieldPolicy("error", dt=1e-3),
        lambda t, dt, sched: torch.full((2,), 0.0 if t < 0.03 else 5.0, device=device),
    )
    s.register(
        FieldPolicy("plasticity", dt=1.0, kind="event_driven"),
        lambda t, dt, sched: torch.ones(2, device=device),
    )
    s.add_trigger(prediction_error_trigger("error", ["plasticity"], threshold=1.0, cooldown_ticks=10))
    rep = s.run(t_end=0.1)
    assert rep.updates["plasticity"] > 0, "an event-driven field only runs when triggered"
    assert rep.forced_updates["plasticity"] == rep.updates["plasticity"]
    assert all(e.time >= 0.03 - 1e-9 for e in rep.events)


def test_event_driven_field_never_runs_without_a_trigger(device):
    s = MultirateScheduler()
    s.register(FieldPolicy("fast", dt=1e-3), _const(torch.zeros(1, device=device)))
    s.register(FieldPolicy("structure", dt=1.0, kind="event_driven"), _const(torch.ones(1, device=device)))
    rep = s.run(t_end=0.5)
    assert rep.updates["structure"] == 0


def test_trigger_with_dangling_reference_is_refused():
    s = MultirateScheduler()
    s.register(FieldPolicy("a", dt=1e-3), _const(0.0))
    with pytest.raises(KeyError, match="unregistered"):
        s.add_trigger(prediction_error_trigger("a", ["nonexistent"], threshold=1.0))
    with pytest.raises(KeyError, match="unregistered"):
        s.add_trigger(prediction_error_trigger("nonexistent", ["a"], threshold=1.0))


# ---------------------------------------------------------------------------
# Interpolation contracts and the coarsening budget
# ---------------------------------------------------------------------------


def test_staleness_is_reported_explicitly(device):
    s = MultirateScheduler()
    s.register(FieldPolicy("fast", dt=1e-3), _const(torch.zeros(1, device=device)))
    s.register(FieldPolicy("slow", dt=1e-2), _const(torch.zeros(1, device=device)))
    s.compile()
    for _ in range(5):
        s.step()
    # Observed between steps the clock already points at the next tick, so the
    # fast field (updated at tick 4) is one tick old and the slow field
    # (updated at tick 0) is five. During a step a just-updated field reads 0.
    assert s.staleness("fast") == pytest.approx(1e-3)
    assert s.staleness("slow") == pytest.approx(5e-3)


def test_linear_contract_beats_zoh_on_a_smooth_field(device):
    """The interpolation contract is a real error claim, tested as a relative one."""

    def build(interp):
        s = MultirateScheduler()
        s.register(
            FieldPolicy("slow", dt=1e-2, interpolation=interp),
            lambda t, dt, sched: torch.sin(torch.tensor([2 * math.pi * t], device=device)),
        )
        return s.run(t_end=1.0)

    zoh = build("zoh").interpolation_error["slow"]
    lin = build("linear").interpolation_error["slow"]
    assert lin < zoh, f"linear contract error {lin:.4g} should beat ZOH {zoh:.4g}"


def test_temporal_coarsening_enters_the_reported_budget(device):
    s = MultirateScheduler()
    s.register(
        FieldPolicy("slow", dt=1e-2, interpolation="zoh"),
        lambda t, dt, sched: torch.sin(torch.tensor([2 * math.pi * t], device=device)),
    )
    rep = s.run(t_end=1.0)
    assert "temporal_coarsening.slow" in rep.budget.entries
    assert rep.budget.total_variance > 0
    assert rep.as_dict()["budget"]["total_variance"] > 0


def test_error_budget_violation_is_reported_not_silently_accepted(device):
    s = MultirateScheduler()
    s.register(
        FieldPolicy("slow", dt=5e-2, interpolation="zoh", error_budget=1e-6),
        lambda t, dt, sched: torch.sin(torch.tensor([2 * math.pi * t], device=device)),
    )
    rep = s.run(t_end=1.0)
    assert rep.budget_violations, "an exceeded coarsening budget must be reported"
    assert "exceeds declared budget" in rep.budget_violations[0]


def test_none_interpolation_refuses_a_stale_read(device):
    s = MultirateScheduler()
    s.register(FieldPolicy("fast", dt=1e-3), _const(torch.zeros(1, device=device)))
    s.register(
        FieldPolicy("strict", dt=1e-2, interpolation="none"), _const(torch.zeros(1, device=device))
    )
    s.compile()
    for _ in range(3):
        s.step()
    with pytest.raises(RuntimeError, match="may not be read stale"):
        s.read("strict", allow_stale=False)


def test_summary_renders(device):
    s = MultirateScheduler()
    s.register(FieldPolicy("neural", dt=1e-3), _const(0.0))
    s.register(FieldPolicy("vascular", dt=5e-2), _const(0.0))
    s.add_trigger(sustained_activity_trigger("neural", ["vascular"], threshold=1.0))
    text = s.summary()
    assert "neural" in text and "vascular" in text and "trigger" in text
