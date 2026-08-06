"""Clocks, update triggers, and the clock graph (thesis sec. 4.5, Appendix C).

Multirate co-simulation gives "every field an update policy, interpolation
contract, and error budget", and forces synchronization "when a fast transition
changes the support of a slower process".  A pure period cannot express that
second requirement, so update policy is a small declarative expression IR
(``PeriodicTrigger``, ``EventTrigger``, ``BoundaryTrigger`` and combinators)
evaluated against a :class:`ScheduleContext`.

The engineering merit of an expression IR over an ad-hoc callback is that it is
serializable, content-hashable, statically inspectable by the compiler (which
must know *whether* a field can be event-triggered at all in order to emit sync
points), and comparable across model variants.  No dynamics live here: the
scheduler that consumes this belongs to agent E.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import ClockId
from .ledger import UncertaintyLedger

__all__ = [
    "ScheduleContext",
    "UpdateTrigger",
    "PeriodicTrigger",
    "EventTrigger",
    "BoundaryTrigger",
    "AllOf",
    "AnyOf",
    "NotTrigger",
    "ClockSpec",
    "ClockEdge",
    "SyncEvidence",
]

#: How a clock's relation to its reference is established.  "unknown" and
#: "assumed" are R01 conditions: Appendix C requires "physical synchronization
#: event or independent cross-correlation target".
SyncEvidence = Literal[
    "physical_trigger",
    "shared_hardware",
    "cross_correlation",
    "assumed",
    "unknown",
]

UNVERIFIED_SYNC: frozenset[str] = frozenset({"assumed", "unknown"})


class ScheduleContext(SchemaModel):
    """Evaluation context for an update trigger."""

    #: Absolute time in seconds on the reference clock.
    t: float = 0.0
    #: Named scalar summaries the trigger may test (e.g. residual energy).
    signals: dict[str, float] = Field(default_factory=dict)
    #: Named boundary event that just occurred, if any.
    boundary: str | None = None
    #: Time of this field's last update, seconds; -inf when never updated.
    last_fired: float = float("-inf")


class _TriggerBase(SchemaModel):
    def fires(self, ctx: ScheduleContext) -> bool:
        raise NotImplementedError

    def periods(self) -> tuple[float, ...]:
        """Every periodic component in the expression, in seconds."""
        return ()

    def is_event_driven(self) -> bool:
        return False


class PeriodicTrigger(_TriggerBase):
    """Fires every ``period`` seconds on the reference clock."""

    kind: Literal["periodic"] = "periodic"
    period: float = Field(gt=0.0)
    phase: float = 0.0

    def fires(self, ctx: ScheduleContext) -> bool:
        if ctx.last_fired == float("-inf"):
            return ctx.t + 1e-12 >= self.phase
        return (ctx.t - ctx.last_fired) + 1e-12 >= self.period

    def periods(self) -> tuple[float, ...]:
        return (self.period,)


class EventTrigger(_TriggerBase):
    """Fires when a named summary crosses a threshold.

    This is the "sustained activity altering metabolic demand" / "salient
    prediction error opening a plasticity window" case in thesis sec. 4.5.
    """

    kind: Literal["event"] = "event"
    signal: str
    gt: float | None = None
    lt: float | None = None
    #: Minimum seconds between firings; prevents a noisy summary from forcing
    #: synchronization every step.
    min_interval: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "EventTrigger":
        if self.gt is None and self.lt is None:
            raise ValueError(f"event trigger on {self.signal!r} needs gt or lt")
        return self

    def fires(self, ctx: ScheduleContext) -> bool:
        if ctx.t - ctx.last_fired < self.min_interval:
            return False
        value = ctx.signals.get(self.signal)
        if value is None:
            return False
        if self.gt is not None and value > self.gt:
            return True
        if self.lt is not None and value < self.lt:
            return True
        return False

    def is_event_driven(self) -> bool:
        return True


class BoundaryTrigger(_TriggerBase):
    """Fires on a named boundary event (stimulus onset, volume trigger, ...)."""

    kind: Literal["boundary"] = "boundary"
    name: str

    def fires(self, ctx: ScheduleContext) -> bool:
        return ctx.boundary == self.name

    def is_event_driven(self) -> bool:
        return True


class AllOf(_TriggerBase):
    kind: Literal["all_of"] = "all_of"
    children: tuple["UpdateTrigger", ...]

    def fires(self, ctx: ScheduleContext) -> bool:
        return bool(self.children) and all(c.fires(ctx) for c in self.children)

    def periods(self) -> tuple[float, ...]:
        return tuple(p for c in self.children for p in c.periods())

    def is_event_driven(self) -> bool:
        return any(c.is_event_driven() for c in self.children)


class AnyOf(_TriggerBase):
    kind: Literal["any_of"] = "any_of"
    children: tuple["UpdateTrigger", ...]

    def fires(self, ctx: ScheduleContext) -> bool:
        return any(c.fires(ctx) for c in self.children)

    def periods(self) -> tuple[float, ...]:
        return tuple(p for c in self.children for p in c.periods())

    def is_event_driven(self) -> bool:
        return any(c.is_event_driven() for c in self.children)


class NotTrigger(_TriggerBase):
    kind: Literal["not"] = "not"
    child: "UpdateTrigger"

    def fires(self, ctx: ScheduleContext) -> bool:
        return not self.child.fires(ctx)

    def periods(self) -> tuple[float, ...]:
        return self.child.periods()

    def is_event_driven(self) -> bool:
        return self.child.is_event_driven()


UpdateTrigger = Annotated[
    Union[PeriodicTrigger, EventTrigger, BoundaryTrigger, AllOf, AnyOf, NotTrigger],
    Field(discriminator="kind"),
]

AllOf.model_rebuild()
AnyOf.model_rebuild()
NotTrigger.model_rebuild()


class ClockSpec(SchemaModel):
    """A named measurement or simulation clock."""

    id: ClockId
    label: str = ""
    #: Nominal sampling period in seconds.
    dt: float = Field(gt=0.0)
    #: Epoch label the clock's zero refers to.
    epoch: str = "session_start"
    #: Reference clock this one is expressed against; None marks the master.
    reference: ClockId | None = None
    #: Constant offset in seconds relative to ``reference``.
    offset: float = 0.0
    #: Linear drift in seconds per second relative to ``reference``.
    drift: float = 0.0
    jitter_sd: float = Field(default=0.0, ge=0.0)
    integration_window: float = Field(default=0.0, ge=0.0)
    group_delay: float = 0.0
    #: How the relation to ``reference`` was established (R01).
    sync_evidence: SyncEvidence = "unknown"
    #: Whether the scheduler may vary the step for operators on this clock.
    #: Enabling this without semigroup testing is R06.
    adaptive_stepping: bool = False
    dropped_sample_policy: Literal[
        "marginalize", "mask", "refuse", "interpolate_with_uncertainty"
    ] = "mask"
    #: Piecewise drift segments when one affine clock map is insufficient.
    piecewise_segments: int = Field(default=1, ge=1)
    #: When this clock's fields update; defaults to periodic at ``dt``.
    trigger: UpdateTrigger | None = None
    #: Interpolation contract used when a consumer runs at a different rate.
    interpolation: Literal[
        "zero_order_hold", "linear", "band_limited", "event_exact", "none"
    ] = "zero_order_hold"
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "ClockSpec":
        if self.reference is not None and self.reference == self.id:
            raise ValueError(f"clock {self.id} cannot reference itself")
        if self.piecewise_segments > 1 and self.drift == 0.0:
            raise ValueError(
                f"clock {self.id} declares piecewise drift segments but zero drift"
            )
        return self

    @property
    def is_master(self) -> bool:
        return self.reference is None

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    def effective_trigger(self) -> "UpdateTrigger":
        return self.trigger if self.trigger is not None else PeriodicTrigger(period=self.dt)

    def to_reference(self, t: float) -> float:
        """Map a timestamp on this clock onto its reference clock."""
        return self.offset + t * (1.0 + self.drift)

    def from_reference(self, t: float) -> float:
        return (t - self.offset) / (1.0 + self.drift)


class ClockEdge(SchemaModel):
    """An explicit synchronization relation between two clocks."""

    src: ClockId
    dst: ClockId
    offset: float = 0.0
    drift: float = 0.0
    sync_evidence: SyncEvidence = "unknown"
    #: Number of synchronization observations backing the estimate.
    n_observations: int = Field(default=0, ge=0)
    residual: float | None = None
    validity_interval: tuple[float, float] | None = None
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "ClockEdge":
        if self.validity_interval is not None:
            lo, hi = self.validity_interval
            if hi <= lo:
                raise ValueError(
                    f"clock edge {self.src}->{self.dst} validity interval must be ordered"
                )
        return self
