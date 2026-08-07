"""The control graph: what was manipulated, when, where, and what is *not* known.

This is the object whose absence blocks G4.  The simulated corpus carries
``control_graph`` as a bare string label naming which *anatomical* control
connectivity a shard was generated with (``none`` / ``randomized`` /
``distance_matched`` / ``dense`` / ``local_only``; see
``scwbd.foundation.simulate``).  That is a different object from this one and
the two must not be conflated: a shard's label says which graph the simulator
used, whereas a :class:`ControlGraph` here says which variables a *real*
experiment manipulated, over which physical exposure intervals, and — the part
that does the work — which variables were **not** recorded.

Design rules, all of them load-bearing:

* **Absence writes something.**  Every manipulated variable carries a
  :class:`Provenance`.  A quantity the upstream release does not distribute is
  recorded as :attr:`Provenance.UNKNOWN` and is never imputed, defaulted, or
  inferred from silence.  An intervention with an unknown coil pose is still an
  intervention — it simply constrains different claims than one with a measured
  pose, and :meth:`ControlGraph.supports` is where that difference becomes
  executable rather than editorial.
* **A derived level is not a recorded level.**  A factor recovered from the
  signal (here: stimulated hemisphere, recovered from the lateralised MEP)
  is :attr:`Provenance.DERIVED` and must carry the method and the statistic
  that supports it.  It is usable, but a consumer can tell it apart from a
  label the provider actually shipped.
* **Prose is not a per-record label.**  A design described in
  ``dataset_description.json`` but not attached to any run is
  :attr:`Provenance.DECLARED`.  Knowing that a study *ran* a pre/post design
  does not tell you which run is "pre".

:meth:`ControlGraph.supports` refuses by default: a quantity must be explicitly
supported, and every refusal names the field that is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from ..lineage import Lineage


class ControlGraphError(RuntimeError):
    """Raised when a consumer asks a control graph for a quantity it cannot support."""

    def __init__(self, message: str, *, quantity: str = "", missing: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.quantity = quantity
        self.missing = tuple(missing)


class Provenance(str, Enum):
    """How a control variable's value came to be known.

    The ordering is a strength ordering: ``RECORDED`` is the only value a
    consumer may treat as a provider-supplied ground-truth label.
    """

    RECORDED = "recorded"
    """Distributed per-record by the provider, machine-readable (e.g. the pulse
    onset sample in ``events.tsv``)."""

    DERIVED = "derived"
    """Recovered by us from the distributed signal, with a named method and a
    reported statistic.  Carries :attr:`ControlVariable.evidence`."""

    DECLARED = "declared"
    """Stated in provider prose about the study as a whole, but not attached to
    any individual record.  Cannot be used as a per-record label."""

    UNKNOWN = "unknown"
    """Not distributed.  Never imputed."""


@dataclass(frozen=True)
class ControlVariable:
    """One manipulated variable, with its provenance and its levels."""

    name: str
    provenance: Provenance
    value: Any = None
    units: str | None = None
    levels_present: tuple[Any, ...] = ()
    """The levels actually present *in the loaded records* — not the levels the
    study ran.  A one-level factor cannot support a slope."""
    method: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance is Provenance.DERIVED and not self.method:
            raise ValueError(
                f"control variable {self.name!r} is DERIVED but names no method; "
                "a derived level without a stated method is indistinguishable "
                "from an assumption"
            )
        if self.provenance is Provenance.UNKNOWN and self.value is not None:
            raise ValueError(
                f"control variable {self.name!r} is UNKNOWN but carries value "
                f"{self.value!r}; unknown must not be given a stand-in value"
            )

    @property
    def n_levels(self) -> int:
        return len(self.levels_present)

    @property
    def is_contrastable(self) -> bool:
        """True when this variable has >= 2 levels present and a usable provenance."""
        return self.n_levels >= 2 and self.provenance in (
            Provenance.RECORDED,
            Provenance.DERIVED,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance.value,
            "value": self.value,
            "units": self.units,
            "levels_present": list(self.levels_present),
            "n_levels": self.n_levels,
            "is_contrastable": self.is_contrastable,
            "method": self.method,
            "evidence": dict(self.evidence),
            "note": self.note,
        }


@dataclass(frozen=True)
class ExposureInterval:
    """A controlled input over its physical exposure interval (``body.tex`` §2.4).

    The intervention is *not* an instantaneous event flag.  It is an input that
    is on over ``[onset, onset + duration)`` on a named clock.  Where the
    provider distributes only a trigger sample and not a pulse width, the
    duration is recorded with :attr:`duration_provenance` ``UNKNOWN`` and the
    stored duration is ``nan`` rather than zero — a zero-width exposure is a
    claim that the input was never on.
    """

    onset_sample: np.ndarray
    onset_s: np.ndarray
    duration_s: np.ndarray
    clock_id: str
    duration_provenance: Provenance = Provenance.UNKNOWN
    note: str = ""

    def __post_init__(self) -> None:
        n = len(self.onset_sample)
        if not (len(self.onset_s) == len(self.duration_s) == n):
            raise ValueError("onset_sample, onset_s and duration_s must be the same length")
        if self.duration_provenance is Provenance.UNKNOWN and not np.all(
            np.isnan(self.duration_s)
        ):
            raise ValueError(
                "duration_provenance is UNKNOWN but durations are not nan; "
                "an unknown exposure width must not be stored as a number"
            )

    def __len__(self) -> int:
        return int(len(self.onset_sample))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_exposures": len(self),
            "clock_id": self.clock_id,
            "onset_sample_first": int(self.onset_sample[0]) if len(self) else None,
            "onset_sample_last": int(self.onset_sample[-1]) if len(self) else None,
            "duration_provenance": self.duration_provenance.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExcludedWindow:
    """A span of samples that must not be read as signal, and why.

    Recorded rather than silently dropped: a consumer that averages over an
    epoch has to be able to see that the first N samples after each exposure
    were excluded, and on what measured grounds.
    """

    name: str
    t_start_s: float
    t_stop_s: float
    reason: str
    measured: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "t_start_s": self.t_start_s,
            "t_stop_s": self.t_stop_s,
            "reason": self.reason,
            "measured": dict(self.measured),
        }


#: Quantities G4's ``prospective_recovery`` sub-check enumerates.  A control
#: graph is asked whether it can support each of these by name.
G4_RECOVERY_QUANTITIES: tuple[str, ...] = (
    "direction",
    "delay",
    "gain",
    "dose",
    "state_dependence",
)


@dataclass(frozen=True)
class ControlGraph:
    """What was manipulated, when, where, and what is not known."""

    source_id: str
    record_id: str
    lineage: Lineage
    manipulated: tuple[ControlVariable, ...]
    exposures: ExposureInterval
    observed_only: tuple[str, ...] = ()
    excluded_windows: tuple[ExcludedWindow, ...] = ()
    target_site: str | None = None
    target_site_provenance: Provenance = Provenance.UNKNOWN
    note: str = ""

    def variable(self, name: str) -> ControlVariable:
        for v in self.manipulated:
            if v.name == name:
                return v
        raise KeyError(f"no control variable {name!r} in {self.record_id}")

    @property
    def unknowns(self) -> tuple[str, ...]:
        """Names of manipulated variables whose value was not distributed."""
        out = [v.name for v in self.manipulated if v.provenance is Provenance.UNKNOWN]
        if self.target_site_provenance is Provenance.UNKNOWN:
            out.append("target_site")
        return tuple(sorted(out))

    @property
    def contrastable(self) -> tuple[str, ...]:
        return tuple(sorted(v.name for v in self.manipulated if v.is_contrastable))

    # -- the executable part -------------------------------------------
    def supports(self, quantity: str) -> tuple[bool, str]:
        """Can this graph support ``quantity``?  Returns ``(ok, reason)``.

        Refuses by default.  Every ``False`` names the field responsible, so a
        gate can report *which* absence blocked it rather than a bare
        ``COULD_NOT_RUN``.
        """
        if quantity == "delay":
            ok = self.exposures.clock_id != "unknown" and len(self.exposures) > 0
            return (
                ok,
                "exposure onsets are on an explicit integer sample clock"
                if ok
                else "no exposure onsets on a named clock",
            )
        if quantity == "direction":
            v = next(
                (v for v in self.manipulated if v.name == "stimulated_hemisphere"), None
            )
            if v is None:
                return False, "no stimulated_hemisphere variable"
            if not v.is_contrastable:
                return (
                    False,
                    f"stimulated_hemisphere has {v.n_levels} level(s) present "
                    f"with provenance {v.provenance.value}; a direction contrast "
                    "needs >= 2 usable levels",
                )
            return True, f"stimulated_hemisphere is {v.provenance.value} with {v.n_levels} levels"
        if quantity in ("gain", "dose"):
            v = next((v for v in self.manipulated if v.name == "intensity_pct_rmt"), None)
            if v is None:
                return False, "no intensity_pct_rmt variable"
            if v.n_levels < 2:
                return (
                    False,
                    f"intensity_pct_rmt has {v.n_levels} level present "
                    f"({v.levels_present}); a {quantity} estimate needs >= 2 levels",
                )
            if quantity == "dose":
                abs_dose = next(
                    (v2 for v2 in self.manipulated if v2.name == "realised_dose"), None
                )
                if abs_dose is None or abs_dose.provenance is Provenance.UNKNOWN:
                    return (
                        False,
                        "intensity is expressed relative to each participant's "
                        "resting motor threshold and the rMT in stimulator output "
                        "units is not distributed, so no absolute dose exists",
                    )
            return True, f"intensity_pct_rmt has {v.n_levels} levels"
        if quantity == "state_dependence":
            v = next((v for v in self.manipulated if v.name == "block_timepoint"), None)
            if v is None:
                return False, "no block_timepoint variable"
            if v.provenance in (Provenance.UNKNOWN, Provenance.DECLARED):
                return (
                    False,
                    f"block_timepoint provenance is {v.provenance.value}: the "
                    "pre/post design is described in provider prose but is not "
                    "attached to any run, so no record carries a timepoint label",
                )
            if not v.is_contrastable:
                return False, f"block_timepoint has {v.n_levels} usable level(s)"
            return True, f"block_timepoint is {v.provenance.value} with {v.n_levels} levels"
        return False, f"unknown quantity {quantity!r}; this graph refuses by default"

    @classmethod
    def combine(
        cls, graphs: Sequence["ControlGraph"], *, record_id: str, note: str = ""
    ) -> "ControlGraph":
        """Union several per-run graphs into the design they jointly form.

        A single run holds one level of each between-run factor, so a contrast
        such as left-versus-right M1 only exists across runs.  This unions
        ``levels_present`` per variable and concatenates the exposure intervals,
        which is what makes :meth:`supports` answer about a *design* rather than
        about one recording.

        Provenance is combined pessimistically: a variable is only as strong as
        the weakest contributing record, so one run whose hemisphere could not
        be recovered does not get upgraded by five that could.  Records must
        share a participant — combining across people would silently create a
        design no single person was exposed to.
        """
        if not graphs:
            raise ValueError("no graphs to combine")
        participants = {g.lineage.participant for g in graphs}
        if len(participants) > 1:
            raise ControlGraphError(
                "refusing to combine control graphs across participants "
                f"({sorted(participants)}): the union would describe a design no "
                "single participant was exposed to. Combine within a participant, "
                "then compare participants."
            )
        order = [
            Provenance.RECORDED,
            Provenance.DERIVED,
            Provenance.DECLARED,
            Provenance.UNKNOWN,
        ]
        names: list[str] = []
        for g in graphs:
            for v in g.manipulated:
                if v.name not in names:
                    names.append(v.name)
        merged: list[ControlVariable] = []
        for name in names:
            vs = [g.variable(name) for g in graphs if any(m.name == name for m in g.manipulated)]
            # Provenance is the weakest among the records that actually
            # contributed a level.  A record whose level could not be recovered
            # contributes no level, so it must not silently downgrade a factor
            # that other records did recover -- but it is counted below, because
            # "one of six runs is unlabelled" is part of what this factor is.
            contributing = [v for v in vs if v.levels_present]
            prov = max(
                (v.provenance for v in (contributing or vs)), key=order.index
            )
            levels: list[Any] = []
            for v in vs:
                for lv in v.levels_present:
                    if lv not in levels:
                        levels.append(lv)
            if name == "pulse_onset":
                levels = list(range(sum(len(v.levels_present) for v in vs)))
            n_unlabelled = len(vs) - len(contributing)
            merged.append(
                ControlVariable(
                    name=name,
                    provenance=prov,
                    value=(vs[0].value if len({str(v.value) for v in vs}) == 1 else None),
                    units=vs[0].units,
                    levels_present=tuple(levels),
                    method=next((v.method for v in vs if v.method), None),
                    evidence={
                        "per_record": {
                            g.record_id: g.variable(name).evidence
                            for g in graphs
                            if any(m.name == name for m in g.manipulated)
                            and g.variable(name).evidence
                        },
                        "n_records": len(vs),
                        "n_records_contributing_a_level": len(contributing),
                        "n_records_unlabelled": n_unlabelled,
                    },
                    note=(
                        vs[0].note
                        + (
                            f" COMBINED: {n_unlabelled} of {len(vs)} records "
                            "contributed no level for this variable and are "
                            "unlabelled on this factor; they are excluded from any "
                            "contrast over it rather than assigned a level."
                            if n_unlabelled
                            else ""
                        )
                    ),
                )
            )
        ex = ExposureInterval(
            onset_sample=np.concatenate([g.exposures.onset_sample for g in graphs]),
            onset_s=np.concatenate([g.exposures.onset_s for g in graphs]),
            duration_s=np.concatenate([g.exposures.duration_s for g in graphs]),
            clock_id=graphs[0].exposures.clock_id,
            duration_provenance=max(
                (g.exposures.duration_provenance for g in graphs), key=order.index
            ),
            note=(
                "concatenated across runs; onsets are per-run indices on the "
                "amplifier clock and are NOT comparable between runs, because the "
                "release distributes no wall clock (scans.tsv acq_time is 'n/a' "
                "and pybv wrote empty marker dates)"
            ),
        )
        sites = sorted({g.target_site for g in graphs if g.target_site})
        return cls(
            source_id=graphs[0].source_id,
            record_id=record_id,
            lineage=graphs[0].lineage.with_(run=None),
            manipulated=tuple(merged),
            exposures=ex,
            observed_only=graphs[0].observed_only,
            excluded_windows=tuple(w for g in graphs for w in g.excluded_windows),
            target_site=("+".join(sites) if sites else None),
            target_site_provenance=max(
                (g.target_site_provenance for g in graphs), key=order.index
            ),
            note=note or graphs[0].note,
        )

    def recovery_report(self) -> dict[str, dict[str, Any]]:
        """Per-quantity support for the five quantities G4 enumerates."""
        out = {}
        for q in G4_RECOVERY_QUANTITIES:
            ok, reason = self.supports(q)
            out[q] = {"supported": ok, "reason": reason}
        return out

    def require(self, quantity: str) -> None:
        ok, reason = self.supports(quantity)
        if not ok:
            raise ControlGraphError(
                f"{self.record_id}: cannot support {quantity!r} — {reason}",
                quantity=quantity,
                missing=self.unknowns,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "scwbd-control-graph/1.0.0",
            "source_id": self.source_id,
            "record_id": self.record_id,
            "lineage": self.lineage.as_dict(),
            "manipulated": [v.as_dict() for v in self.manipulated],
            "observed_only": list(self.observed_only),
            "exposures": self.exposures.as_dict(),
            "excluded_windows": [w.as_dict() for w in self.excluded_windows],
            "target_site": self.target_site,
            "target_site_provenance": self.target_site_provenance.value,
            "unknowns": list(self.unknowns),
            "contrastable": list(self.contrastable),
            "recovery": self.recovery_report(),
            "note": self.note,
        }
