"""Multirate co-simulation scheduler (thesis §4.5).

Electrical, cognitive, vascular, endocrine, learning and structural states cannot
be advanced efficiently at one clock.  Every field declares:

1. an **update policy** — ``continuous`` (fixed cadence), ``lazy`` (update only
   when its inputs materially change), or ``event_driven`` (only when a trigger
   fires);
2. an **interpolation contract** — what a consumer gets between updates
   (``zoh``, ``linear``, or ``none`` meaning consumers must not read stale
   values at all);
3. an **error budget** — the coarsening error it is allowed to contribute.

Two mechanisms are non-negotiable:

* **Forced synchronization.**  When a fast transition changes the *support* of a
  slower process — sustained activity altering metabolic demand, a salient
  prediction error opening a plasticity window — the slow field is updated
  immediately, off its nominal cadence.  This is not an optimisation; running
  the slow field on its own clock through such a transition integrates the
  wrong process.
* **Temporal coarsening contributes to reported posterior uncertainty.**  The
  scheduler measures the interpolation error each lazy/slow field actually
  incurred and writes it into a :class:`NumericalBudget`, which is returned with
  the run.  Coarsening is never an invisible implementation choice.

All time bookkeeping is in **integer ticks** on a base clock, so the schedule is
exactly reproducible and two fields at 1 ms and 10 ms never drift apart by
floating-point accumulation.  Clocks incommensurable with the base tick are
refused rather than rounded (that is a declaration error, cf. R01).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .types import NumericalBudget

__all__ = [
    "UpdateKind",
    "Interpolation",
    "FieldPolicy",
    "FieldState",
    "SyncTrigger",
    "SyncEvent",
    "MultirateScheduler",
    "ScheduleReport",
    "sustained_activity_trigger",
    "prediction_error_trigger",
]

UpdateKind = Literal["continuous", "lazy", "event_driven"]
Interpolation = Literal["zoh", "linear", "none"]


@dataclass(frozen=True)
class FieldPolicy:
    """Update policy + interpolation contract + error budget for one field."""

    name: str
    dt: float  # nominal update interval, seconds
    kind: UpdateKind = "continuous"
    interpolation: Interpolation = "zoh"
    error_budget: float = math.inf  # allowed coarsening error (state units)
    materiality: float = 0.0  # lazy: relative input change that forces an update
    inputs: tuple[str, ...] = ()
    units: str = "dimensionless"
    clock: str = ""  # ClockId (agent A) when available
    description: str = ""

    #: enumerations are enforced at construction. An unknown policy string must
    #: never fall through to a "safe default" — a typo that makes a field update
    #: every tick is a silent modelling change.
    VALID_KINDS: ClassVar[frozenset[str]] = frozenset({"continuous", "lazy", "event_driven"})
    VALID_INTERPOLATION: ClassVar[frozenset[str]] = frozenset({"zoh", "linear", "none"})

    def __post_init__(self) -> None:
        if self.kind not in self.VALID_KINDS:
            raise ValueError(
                f"field {self.name!r}: unknown update kind {self.kind!r}; "
                f"valid: {sorted(self.VALID_KINDS)}"
            )
        if self.interpolation not in self.VALID_INTERPOLATION:
            raise ValueError(
                f"field {self.name!r}: unknown interpolation contract {self.interpolation!r}; "
                f"valid: {sorted(self.VALID_INTERPOLATION)}"
            )
        if self.dt <= 0 and self.kind != "event_driven":
            raise ValueError(f"field {self.name!r}: dt must be > 0 unless kind='event_driven'")
        if self.materiality < 0:
            raise ValueError(f"field {self.name!r}: materiality must be >= 0")
        if self.kind == "lazy" and not self.inputs:
            raise ValueError(
                f"field {self.name!r}: a lazy field must declare the inputs whose material change "
                "triggers it — 'update when something changes' is not a policy"
            )


@dataclass
class FieldState:
    """Live state of one scheduled field."""

    policy: FieldPolicy
    value: Any = None
    prev_value: Any = None
    derivative: Any = None  # for the linear interpolation contract
    last_tick: int = -1
    prev_tick: int = -1
    n_updates: int = 0
    n_forced: int = 0
    n_skipped: int = 0
    max_interp_error: float = 0.0
    interp_error_sq_sum: float = 0.0
    n_interp: int = 0

    def age_ticks(self, tick: int) -> int:
        return tick - self.last_tick if self.last_tick >= 0 else 0


@dataclass
class SyncTrigger:
    """Forced synchronization: a fast transition changing a slow field's support.

    ``predicate`` is evaluated after every update of ``watch`` and receives
    ``(value, context)``.  It may return a bool or a bool tensor; a tensor is
    reduced with ``any`` (if *any* parameter set in the batch crosses the
    transition, the batched slow field must be advanced — running half the batch
    on a stale schedule is not an option).
    """

    name: str
    watch: str
    force: tuple[str, ...]
    predicate: Callable[[Any, "MultirateScheduler"], Any]
    reason: str = ""
    cooldown_ticks: int = 0
    _last_fired: int = field(default=-(10**9), repr=False)

    def evaluate(self, value: Any, sched: "MultirateScheduler", tick: int) -> bool:
        if tick - self._last_fired < self.cooldown_ticks:
            return False
        out = self.predicate(value, sched)
        if isinstance(out, Tensor):
            out = bool(out.any())
        if out:
            self._last_fired = tick
        return bool(out)


@dataclass
class SyncEvent:
    tick: int
    time: float
    trigger: str
    forced: tuple[str, ...]
    reason: str


@dataclass
class ScheduleReport:
    """What the schedule actually did — returned with every run."""

    t_end: float
    tick_dt: float
    n_ticks: int
    updates: dict[str, int]
    forced_updates: dict[str, int]
    skipped: dict[str, int]
    events: list[SyncEvent]
    budget: NumericalBudget
    budget_violations: list[str]
    interpolation_error: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_end": self.t_end,
            "tick_dt": self.tick_dt,
            "n_ticks": self.n_ticks,
            "updates": dict(self.updates),
            "forced_updates": dict(self.forced_updates),
            "skipped": dict(self.skipped),
            "n_sync_events": len(self.events),
            "events": [e.__dict__ for e in self.events[:64]],
            "interpolation_error": dict(self.interpolation_error),
            "budget": self.budget.as_dict(),
            "budget_violations": list(self.budget_violations),
        }


class MultirateScheduler:
    """Deterministic integer-tick multirate scheduler."""

    def __init__(self, tick_dt: float | None = None, *, tick_tolerance: float = 1e-9):
        self.tick_dt = tick_dt
        self.tick_tolerance = tick_tolerance
        self.fields: dict[str, FieldState] = {}
        self.updaters: dict[str, Callable[[float, float, "MultirateScheduler"], Any]] = {}
        self.periods: dict[str, int] = {}
        self.triggers: list[SyncTrigger] = []
        self.events: list[SyncEvent] = []
        self.tick = 0
        self.budget = NumericalBudget()
        self._pending_force: set[str] = set()

    # -- registration ------------------------------------------------------
    def register(
        self,
        policy: FieldPolicy,
        updater: Callable[[float, float, "MultirateScheduler"], Any],
        *,
        initial: Any = None,
    ) -> None:
        if policy.name in self.fields:
            raise ValueError(f"field {policy.name!r} already registered")
        self.fields[policy.name] = FieldState(policy=policy, value=initial, prev_value=initial)
        self.updaters[policy.name] = updater

    def add_trigger(self, trigger: SyncTrigger) -> None:
        if trigger.watch not in self.fields:
            raise KeyError(f"trigger {trigger.name!r} watches unregistered field {trigger.watch!r}")
        missing = [f for f in trigger.force if f not in self.fields]
        if missing:
            raise KeyError(f"trigger {trigger.name!r} forces unregistered field(s) {missing}")
        self.triggers.append(trigger)

    # -- clock -------------------------------------------------------------
    def _resolve_tick(self) -> float:
        dts = [f.policy.dt for f in self.fields.values() if f.policy.dt > 0]
        if not dts:
            raise RuntimeError("no field declares a positive dt; nothing to schedule")
        base = self.tick_dt if self.tick_dt is not None else min(dts)
        for f in self.fields.values():
            if f.policy.dt <= 0:
                continue
            ratio = f.policy.dt / base
            if abs(ratio - round(ratio)) > self.tick_tolerance * max(ratio, 1.0):
                raise ValueError(
                    f"field {f.policy.name!r} has dt={f.policy.dt:g} s which is not an integer "
                    f"multiple of the base tick {base:g} s. Incommensurable clocks are refused, "
                    "not rounded: declare a compatible dt or an explicit tick_dt."
                )
        return base

    def compile(self) -> None:
        base = self._resolve_tick()
        self.tick_dt = base
        self.periods = {
            name: (0 if f.policy.dt <= 0 else int(round(f.policy.dt / base)))
            for name, f in self.fields.items()
        }

    # -- reading with the interpolation contract ---------------------------
    def read(self, name: str, *, allow_stale: bool = True) -> Any:
        st = self.fields[name]
        age = st.age_ticks(self.tick)
        if age == 0 or st.policy.interpolation == "zoh":
            if age > 0 and st.policy.interpolation == "none" and not allow_stale:
                raise RuntimeError(
                    f"field {name!r} declares interpolation='none' but is {age} ticks stale; "
                    "its consumer must be synchronised with it"
                )
            return st.value
        if st.policy.interpolation == "none":
            if not allow_stale:
                raise RuntimeError(f"field {name!r} may not be read stale (interpolation='none')")
            return st.value
        # linear extrapolation from the last two updates
        if st.derivative is None or st.prev_value is None or st.prev_tick < 0:
            return st.value
        dt = age * self.tick_dt
        return st.value + st.derivative * dt

    def value(self, name: str) -> Any:
        return self.fields[name].value

    def staleness(self, name: str) -> float:
        """Seconds since ``name`` was last updated.

        Consumers get this explicitly rather than having to infer it: the delay
        and the coarsening age are known exactly, so they are passed, not
        re-derived.
        """
        st = self.fields[name]
        return st.age_ticks(self.tick) * float(self.tick_dt or 0.0)

    def time(self) -> float:
        return self.tick * float(self.tick_dt)

    def summary(self) -> str:
        """Human-readable schedule — the first thing you want when it misbehaves."""
        if not self.periods:
            self.compile()
        lines = [f"MultirateScheduler(tick_dt={self.tick_dt:g}s, t={self.time():g}s)"]
        for name, st in self.fields.items():
            p = self.periods[name]
            lines.append(
                f"  {name:<16} dt={st.policy.dt:<8g} period={p:<6d} {st.policy.kind:<12}"
                f" interp={st.policy.interpolation:<7} updates={st.n_updates}"
                f" forced={st.n_forced} skipped={st.n_skipped}"
            )
        for tr in self.triggers:
            lines.append(f"  trigger {tr.name}: {tr.watch} -> {', '.join(tr.force)}")
        return "\n".join(lines)

    # -- the loop ----------------------------------------------------------
    def _material_change(self, st: FieldState) -> bool:
        """Lazy policy: has any declared input changed materially since last update?"""
        thresh = st.policy.materiality
        for src in st.policy.inputs:
            s = self.fields[src]
            if s.value is None or s.prev_value is None:
                return True
            if isinstance(s.value, Tensor):
                num = float((s.value - s.prev_value).abs().max())
                den = float(s.value.abs().max()) + 1e-12
            else:
                num = abs(float(s.value) - float(s.prev_value))
                den = abs(float(s.value)) + 1e-12
            if num / den > thresh:
                return True
        return False

    def _do_update(self, name: str, forced: bool) -> None:
        st = self.fields[name]
        t = self.time()
        dt = max(st.age_ticks(self.tick), 1) * float(self.tick_dt)
        new = self.updaters[name](t, dt, self)
        if new is not None:
            if isinstance(new, Tensor) and isinstance(st.value, Tensor) and st.last_tick >= 0:
                d_ticks = max(self.tick - st.last_tick, 1)
                deriv = (new - st.value) / (d_ticks * float(self.tick_dt))
                # coarsening error of the *contract* actually offered
                if st.policy.interpolation == "zoh":
                    err = float((new - st.value).abs().max())
                else:
                    err = float(
                        (new - (st.value + (st.derivative if st.derivative is not None else deriv) * (d_ticks * float(self.tick_dt)))).abs().max()
                    )
                st.max_interp_error = max(st.max_interp_error, err)
                st.interp_error_sq_sum += err * err
                st.n_interp += 1
                st.derivative = deriv
            st.prev_value = st.value
            st.value = new
        st.prev_tick = st.last_tick
        st.last_tick = self.tick
        st.n_updates += 1
        if forced:
            st.n_forced += 1
        # triggers fire on the field that just changed
        for trig in self.triggers:
            if trig.watch != name:
                continue
            if trig.evaluate(st.value, self, self.tick):
                self.events.append(
                    SyncEvent(self.tick, self.time(), trig.name, tuple(trig.force), trig.reason)
                )
                self._pending_force.update(trig.force)

    def step(self) -> None:
        """Advance one base tick, updating whichever fields are due."""
        if not self.periods:
            self.compile()
        due: list[str] = []
        for name, st in self.fields.items():
            p = self.periods[name]
            if st.policy.kind == "event_driven":
                continue
            if p and self.tick % p == 0:
                if st.policy.kind == "lazy" and st.last_tick >= 0 and not self._material_change(st):
                    st.n_skipped += 1
                    continue
                due.append(name)
        for name in due:
            self._do_update(name, forced=False)
        # forced synchronisation, including cascades within the same tick
        guard = 0
        while self._pending_force:
            pending = sorted(self._pending_force)
            self._pending_force.clear()
            for name in pending:
                if name in due:
                    continue
                self._do_update(name, forced=True)
                due.append(name)
            guard += 1
            if guard > 8:
                raise RuntimeError("forced-synchronisation cascade did not settle within 8 rounds")
        self.tick += 1

    def run(self, t_end: float, *, t_start: float = 0.0) -> ScheduleReport:
        self.compile()
        n = int(round((t_end - t_start) / float(self.tick_dt)))
        self.tick = int(round(t_start / float(self.tick_dt)))
        for _ in range(n):
            self.step()
        return self.report(t_end)

    # -- reporting ---------------------------------------------------------
    def report(self, t_end: float) -> ScheduleReport:
        budget = NumericalBudget()
        violations: list[str] = []
        interp: dict[str, float] = {}
        for name, st in self.fields.items():
            rms = math.sqrt(st.interp_error_sq_sum / st.n_interp) if st.n_interp else 0.0
            interp[name] = rms
            if rms > 0:
                # temporal coarsening enters the reported posterior uncertainty
                budget.add(
                    f"temporal_coarsening.{name}",
                    rms**2,
                    f"{name}: {st.policy.interpolation} contract at dt={st.policy.dt:g}s",
                )
            if st.max_interp_error > st.policy.error_budget:
                violations.append(
                    f"{name}: coarsening error {st.max_interp_error:.4g} exceeds declared budget "
                    f"{st.policy.error_budget:.4g} — refine its clock or widen the declared budget"
                )
        self.budget = budget
        return ScheduleReport(
            t_end=t_end,
            tick_dt=float(self.tick_dt),
            n_ticks=self.tick,
            updates={k: v.n_updates for k, v in self.fields.items()},
            forced_updates={k: v.n_forced for k, v in self.fields.items()},
            skipped={k: v.n_skipped for k, v in self.fields.items()},
            events=list(self.events),
            budget=budget,
            budget_violations=violations,
            interpolation_error=interp,
        )


# ---------------------------------------------------------------------------
# Canonical triggers from §4.5
# ---------------------------------------------------------------------------


def sustained_activity_trigger(
    watch: str,
    force: Sequence[str],
    *,
    threshold: float,
    window: int = 10,
    name: str = "sustained_activity->metabolic",
    cooldown_ticks: int = 0,
) -> SyncTrigger:
    """Sustained activity alters metabolic demand -> force the vascular field.

    Keeps a running mean of the watched activity over ``window`` updates; fires
    when it crosses ``threshold``.  The running state lives in the closure so the
    trigger is self-contained and the scheduler stays generic.
    """
    hist: list[Tensor] = []

    def predicate(value: Any, sched: MultirateScheduler) -> Any:
        if value is None:
            return False
        v = value if isinstance(value, Tensor) else torch.as_tensor(value)
        hist.append(v.detach().abs().mean(dim=tuple(range(1, v.ndim))) if v.ndim > 1 else v.detach().abs())
        if len(hist) > window:
            hist.pop(0)
        if len(hist) < window:
            return False
        return torch.stack(hist).mean(dim=0) > threshold

    return SyncTrigger(
        name=name,
        watch=watch,
        force=tuple(force),
        predicate=predicate,
        reason="sustained neural activity changes the support of the metabolic process (§4.5)",
        cooldown_ticks=cooldown_ticks,
    )


def prediction_error_trigger(
    watch: str,
    force: Sequence[str],
    *,
    threshold: float,
    name: str = "prediction_error->plasticity",
    cooldown_ticks: int = 0,
) -> SyncTrigger:
    """A salient prediction error opens a plasticity window -> force plasticity."""

    def predicate(value: Any, sched: MultirateScheduler) -> Any:
        if value is None:
            return False
        v = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return v.detach().abs().amax() > threshold

    return SyncTrigger(
        name=name,
        watch=watch,
        force=tuple(force),
        predicate=predicate,
        reason="salient prediction error opens a plasticity window (§4.5)",
        cooldown_ticks=cooldown_ticks,
    )
