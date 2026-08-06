"""The multirate plan (thesis sec. 4.5, ARCHITECTURE.md sec. 2).

"A multirate scheduler gives every field an update policy, interpolation
contract, and error budget."  This module emits that plan; it does not run it
(agent E owns execution).

Sync points are computed exactly, in integer nanoseconds, over the hyperperiod
of the periodic fields.  Floating-point accumulation of ``t += dt`` would drift
against a 1 ms EEG clock inside a 1 s fMRI volume, and a schedule that drifts
is a schedule that silently mislabels which samples are simultaneous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Iterator

from ..schema.clocks import ClockSpec, PeriodicTrigger, ScheduleContext, UpdateTrigger
from ..schema.schema import BrainSchema
from .layout import StateLayout

__all__ = [
    "FieldPolicy",
    "InterpolationContract",
    "ScheduleEvent",
    "MultirateSchedule",
    "build_schedule",
    "NS",
]

#: Time is discretized to nanoseconds for exact period arithmetic.
NS = 1_000_000_000


@dataclass(frozen=True)
class FieldPolicy:
    """Update policy for one state field (a region/component block)."""

    region: str
    component: str
    clock: str
    #: Update period in seconds (native ``dt`` of the field's clock).
    period: float
    period_ns: int
    trigger: UpdateTrigger
    #: True when the field can also be forced to update by an event.
    event_driven: bool
    #: Interpolation used when a faster/slower consumer reads this field.
    interpolation: str
    #: Temporal coarsening error budget; enters reported posterior uncertainty
    #: rather than remaining an invisible implementation choice (sec. 4.5).
    error_budget: float | None

    @property
    def key(self) -> str:
        return f"{self.region}.{self.component}"

    def fires_at(self, t: float, last_fired: float = float("-inf")) -> bool:
        return self.trigger.fires(ScheduleContext(t=t, last_fired=last_fired))


@dataclass(frozen=True)
class InterpolationContract:
    """How a value crossing two clocks is resampled, and what that costs."""

    from_clock: str
    to_clock: str
    method: str
    #: Ratio of periods; >1 means the consumer is slower than the producer.
    period_ratio: float
    #: Group delay plus half the integration window of the producer, seconds.
    producer_latency: float
    #: Whether the clock relation itself is backed by evidence (R01 checks it).
    sync_evidence: str

    @property
    def is_downsampling(self) -> bool:
        return self.period_ratio > 1.0


@dataclass(frozen=True)
class ScheduleEvent:
    """One tick of the plan."""

    t_ns: int
    fields: tuple[str, ...]
    clocks: tuple[str, ...]
    is_sync: bool

    @property
    def t(self) -> float:
        return self.t_ns / NS


@dataclass(frozen=True)
class MultirateSchedule:
    """Per-field update periods, sync points, and interpolation contracts."""

    policies: tuple[FieldPolicy, ...]
    #: Times (seconds) at which more than one clock fires simultaneously.
    sync_points: tuple[float, ...]
    interpolation: tuple[InterpolationContract, ...]
    #: Least common period of all periodic fields, seconds.
    hyperperiod: float
    base_dt: float
    #: Fields whose update can be forced by an event rather than a period.
    event_driven_fields: tuple[str, ...] = ()
    _by_key: dict[str, FieldPolicy] = field(repr=False, default_factory=dict)

    # -- accessors ---------------------------------------------------------
    def policy(self, region: str, component: str) -> FieldPolicy:
        return self._by_key[f"{region}.{component}"]

    def periods(self) -> dict[str, float]:
        return {p.key: p.period for p in self.policies}

    def clocks(self) -> tuple[str, ...]:
        return tuple(sorted({p.clock for p in self.policies}))

    def fields_on_clock(self, clock: str) -> tuple[str, ...]:
        return tuple(p.key for p in self.policies if p.clock == str(clock))

    def contract(self, from_clock: str, to_clock: str) -> InterpolationContract | None:
        for c in self.interpolation:
            if c.from_clock == str(from_clock) and c.to_clock == str(to_clock):
                return c
        return None

    # -- plan --------------------------------------------------------------
    def step_plan(self, t0: float, t1: float) -> tuple[ScheduleEvent, ...]:
        """Ordered periodic events in ``(t0, t1]``.

        The interval is half-open at the *start*: a runtime already holds state
        at ``t0``, so the plan lists the updates that carry it forward.

        Event- and boundary-triggered updates are deliberately absent: they
        depend on runtime signals, and this object is a static plan.  Fields
        that can be event-forced are listed in ``event_driven_fields`` so the
        runtime knows which policies to re-evaluate each step.
        """
        start_ns = int(round(t0 * NS))
        stop_ns = int(round(t1 * NS))
        ticks: dict[int, list[FieldPolicy]] = {}
        for p in self.policies:
            if p.period_ns <= 0:
                continue
            first = (start_ns // p.period_ns + 1) * p.period_ns
            t = first
            while t <= stop_ns:
                ticks.setdefault(t, []).append(p)
                t += p.period_ns
        events: list[ScheduleEvent] = []
        for t_ns in sorted(ticks):
            group = ticks[t_ns]
            clocks = tuple(sorted({p.clock for p in group}))
            events.append(
                ScheduleEvent(
                    t_ns=t_ns,
                    fields=tuple(sorted(p.key for p in group)),
                    clocks=clocks,
                    is_sync=len(clocks) > 1,
                )
            )
        return tuple(events)

    def __iter__(self) -> Iterator[FieldPolicy]:
        return iter(self.policies)

    def summary(self) -> str:
        return (
            f"MultirateSchedule({len(self.policies)} fields, "
            f"{len(self.clocks())} clocks, base_dt={self.base_dt}s, "
            f"hyperperiod={self.hyperperiod}s, {len(self.sync_points)} sync points)"
        )


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else max(a, b)


def build_schedule(
    schema: BrainSchema,
    layout: StateLayout,
    *,
    max_hyperperiod_s: float = 60.0,
) -> MultirateSchedule:
    clocks: dict[str, ClockSpec] = {str(c.id): c for c in schema.clocks}

    def clock_or_fallback(cid: str, dt: float) -> ClockSpec:
        """Undeclared clocks are an R01 refusal.

        The compiler only reaches this function with an undeclared clock when
        R01 was explicitly overridden, in which case the artifact is already
        demoted.  Synthesizing a period-only stand-in keeps the emitted plan
        honest about what it does not know (``sync_evidence="unknown"``)
        instead of crashing on a program the author chose to admit.
        """
        spec = clocks.get(str(cid))
        if spec is not None:
            return spec
        return ClockSpec(
            id=cid,  # type: ignore[arg-type]
            label="undeclared clock (R01 overridden)",
            dt=dt,
            sync_evidence="unknown",
            interpolation="none",
        )

    policies: list[FieldPolicy] = []
    for entry in layout.entries:
        clock = clock_or_fallback(
            entry.clock,
            schema.region(entry.region).state.components[entry.component].temporal.dt,
        )
        comp = schema.region(entry.region).state.components[entry.component]
        period = comp.temporal.dt
        trigger: UpdateTrigger = (
            clock.trigger if clock.trigger is not None else PeriodicTrigger(period=period)
        )
        budget = None
        if comp.ledger is not None:
            budget = comp.ledger.variance.get("numerical")
        policies.append(
            FieldPolicy(
                region=entry.region,
                component=entry.component,
                clock=entry.clock,
                period=period,
                period_ns=int(round(period * NS)),
                trigger=trigger,
                event_driven=trigger.is_event_driven(),
                interpolation=clock.interpolation,
                error_budget=budget,
            )
        )

    # Hyperperiod over distinct periods, in exact integer nanoseconds.
    periods_ns = sorted({p.period_ns for p in policies if p.period_ns > 0})
    hyper_ns = 0
    for p_ns in periods_ns:
        hyper_ns = _lcm(hyper_ns, p_ns) if hyper_ns else p_ns
        if hyper_ns > max_hyperperiod_s * NS:
            hyper_ns = int(max_hyperperiod_s * NS)
            break

    sync_points: list[float] = []
    if len(periods_ns) > 1 and hyper_ns:
        step = periods_ns[0]
        t = step
        while t <= hyper_ns:
            firing = {p_ns for p_ns in periods_ns if t % p_ns == 0}
            if len(firing) > 1:
                sync_points.append(t / NS)
            t += step

    # Interpolation contracts for every ordered pair of clocks that an operator
    # actually crosses, plus every pair a port connects.
    dt_of: dict[str, float] = {}
    for region in schema.regions:
        for comp in region.state.components.values():
            dt_of.setdefault(str(comp.temporal.clock), comp.temporal.dt)
        for port in region.ports:
            dt_of.setdefault(str(port.temporal.clock), port.temporal.dt)

    pairs: set[tuple[str, str]] = set()
    for op in schema.operators:
        src_clocks = {
            str(c.temporal.clock) for c in schema.region(op.src).state.components.values()
        }
        dst_clocks = {
            str(c.temporal.clock) for c in schema.region(op.dst).state.components.values()
        }
        for a in src_clocks:
            for b in dst_clocks:
                if a != b:
                    pairs.add((a, b))
    for region, port in schema.iter_ports():
        port_clock = str(port.temporal.clock)
        for comp in region.state.components.values():
            other = str(comp.temporal.clock)
            if other != port_clock:
                pairs.add((other, port_clock) if port.direction != "in" else (port_clock, other))

    contracts: list[InterpolationContract] = []
    for a, b in sorted(pairs):
        ca = clock_or_fallback(a, dt_of.get(a, 1.0))
        cb = clock_or_fallback(b, dt_of.get(b, 1.0))
        contracts.append(
            InterpolationContract(
                from_clock=a,
                to_clock=b,
                method=cb.interpolation,
                period_ratio=cb.dt / ca.dt,
                producer_latency=ca.group_delay + 0.5 * ca.integration_window,
                sync_evidence=cb.sync_evidence if not cb.is_master else "shared_hardware",
            )
        )

    base_dt = min((p.period for p in policies), default=0.0)
    return MultirateSchedule(
        policies=tuple(policies),
        sync_points=tuple(sync_points),
        interpolation=tuple(contracts),
        hyperperiod=hyper_ns / NS if hyper_ns else 0.0,
        base_dt=base_dt,
        event_driven_fields=tuple(sorted(p.key for p in policies if p.event_driven)),
        _by_key={p.key: p for p in policies},
    )
