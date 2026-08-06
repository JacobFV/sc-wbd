"""The clock graph -- Appendix C layer 4.

    "Clock identity, sample rate, epoch, trigger path, offset, drift, jitter,
     dropped samples and integration/filter delay ... Physical synchronization
     event or independent cross-correlation target; piecewise drift when one
     affine clock map is insufficient."

Three commitments make this module different from a dictionary of sample rates.

1. **A sample index is not a time.**  ``index_to_time`` walks the dropped-sample
   record and the filter group delay.  ``i / fs`` is wrong the moment one packet
   is lost, and the error is cumulative and silent.
2. **A clock relation needs evidence.**  :meth:`ClockGraph.relate` refuses an
   edge whose ``evidence`` is not a physical trigger, a shared hardware clock,
   an independent cross-correlation target, or a declared identity.  Two clocks
   with no path do not get a guessed offset -- :meth:`ClockGraph.align` raises
   (refusal R01).
3. **Every alignment carries timing uncertainty.**  ``align`` returns
   ``(t, sigma_t)``; the variance accumulates offset/drift parameter covariance,
   fit residual, jitter and the group-delay ledger along the path.

The firing-rule AST (:class:`ClockExpr` and friends) is adapted from the clock
IR in CommandAGI/canvas-engineering, which composes ``Periodic``/``OnEvent``/
``Boundary`` leaves with And/Or/Not and Cooldown/MaxSilence decorators.  The
composition idea travels well; the semantics do not.  There, a clock fires when
``external_t % period == 0`` -- an integer step index, unitless and exact.  Here
every leaf resolves in **physical seconds on a named clock**, through the clock
graph, with an uncertainty and a validity interval, because a scanner volume
trigger and an EEG amplifier sample index are not the same quantity even when
both are called ``t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from .calibration import CalibrationRecord, ExpiryPolicy, ValidityCheck
from .errors import (
    ClockRelationUnknownError,
    NonInvertibleTransformError,
    TransformError,
)
from .se3 import DTYPE, ValidityInterval, as_tensor

# --------------------------------------------------------------------------
# clock specification
# --------------------------------------------------------------------------

InterpolationPolicy = Literal["none", "zoh", "linear", "sinc"]

#: Evidence kinds that may license a clock relation (Appendix C layer 4).
ADMISSIBLE_CLOCK_EVIDENCE = frozenset(
    {
        "physical_trigger",  # shared TTL / scanner trigger / photodiode
        "shared_hardware_clock",  # same oscillator, e.g. one amplifier
        "cross_correlation",  # independent common signal, e.g. audio in both streams
        "declared_identity",  # the two ids name the same physical clock
    }
)


@dataclass(frozen=True)
class DropSpec:
    """A run of dropped samples on a clock.

    ``start_index`` is the index in the *acquisition tick* numbering of the
    first lost tick; ``count`` is how many consecutive ticks were lost.  Stored
    sample ``i`` therefore corresponds to acquisition tick
    ``i + (number of ticks dropped at or before that point)``.
    """

    start_index: int
    count: int
    detected_by: str = "undeclared"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.count <= 0 or self.start_index < 0:
            raise TransformError(
                f"invalid drop specification start={self.start_index} count={self.count}",
                remedy="Drops need a non-negative start and a positive count.",
                offending_object=self,
            )


@dataclass(frozen=True)
class ClockSpec:
    """One clock in the graph.

    Attributes
    ----------
    id:
        e.g. ``"eeg_amp"``, ``"scanner_volume"``, ``"eyetracker"``, ``"stim_pc"``.
    rate_hz:
        Nominal sample rate as an exact :class:`~fractions.Fraction` (so
        hyperperiods of 1000 Hz EEG and 1/0.72 Hz TR volumes are exact, not
        floating-point near-misses).  ``None`` for a pure event clock.
    epoch:
        Time of sample 0 on the clock's own timeline, in seconds.  ``None``
        means the epoch is unknown, which is a legitimate state -- it just means
        this clock cannot be aligned without an edge that supplies it.
    trigger_path:
        Provenance of the synchronization signal, e.g.
        ``("scanner_ttl", "parallel_port", "amplifier_din")``.  Empty means the
        clock is free-running.
    group_delay_s:
        Filter/integration delay: a stored sample labelled ``t`` reflects the
        physical process at ``t - group_delay_s``.
    jitter_sd_s:
        Standard deviation of per-sample timing jitter.
    dropped:
        Known dropped-sample runs.
    integration_window_s:
        Width of the acquisition's integration kernel (0 = instantaneous).
        A BOLD volume is not a point in time.
    interpolation_policy:
        What resampling this clock's data admits.  ``"none"`` is correct for
        event/point-process clocks: interpolating a spike time is a category
        error, and the scheduler refuses it rather than producing a number.
    """

    id: str
    rate_hz: Fraction | float | None = None
    epoch: float | None = None
    trigger_path: tuple[str, ...] = ()
    group_delay_s: float = 0.0
    jitter_sd_s: float = 0.0
    dropped: tuple[DropSpec, ...] = ()
    integration_window_s: float = 0.0
    interpolation_policy: InterpolationPolicy = "linear"
    max_interpolation_gap_s: float | None = None
    domain: Literal["acquisition", "stimulus", "solver", "event"] = "acquisition"
    units: str = "s"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.rate_hz is not None:
            r = Fraction(self.rate_hz).limit_denominator(10**9)
            if r <= 0:
                raise TransformError(
                    f"clock {self.id!r} has non-positive rate {self.rate_hz}",
                    remedy="Declare a positive sample rate, or None for an event clock.",
                    offending_object=self.id,
                )
            object.__setattr__(self, "rate_hz", r)
        if self.units != "s":
            raise TransformError(
                f"clock {self.id!r} must express time in seconds, got {self.units!r}",
                remedy="Convert explicitly with scwbd.transforms.units.convert.",
                offending_object=self.units,
            )
        if self.jitter_sd_s < 0 or self.group_delay_s < 0:
            raise TransformError(
                f"clock {self.id!r} has negative jitter or group delay",
                remedy="Both are non-negative by definition.",
                offending_object=self.id,
            )
        drops = tuple(sorted(self.dropped, key=lambda d: d.start_index))
        for a, b in zip(drops, drops[1:]):
            if b.start_index < a.start_index + a.count:
                raise TransformError(
                    f"clock {self.id!r} has overlapping dropped-sample runs "
                    f"{a} and {b}",
                    remedy="Merge overlapping drop records.",
                    offending_object=self.id,
                )
        object.__setattr__(self, "dropped", drops)

    # -- rate helpers ------------------------------------------------------

    @property
    def dt(self) -> float:
        if self.rate_hz is None:
            raise TransformError(
                f"clock {self.id!r} is an event clock and has no fixed dt",
                remedy="Use the event times directly.",
                offending_object=self.id,
            )
        return float(1 / Fraction(self.rate_hz))

    @property
    def period(self) -> Fraction:
        if self.rate_hz is None:
            raise TransformError(
                f"clock {self.id!r} is an event clock and has no period",
                remedy="Use the event times directly.",
                offending_object=self.id,
            )
        return 1 / Fraction(self.rate_hz)

    # -- index <-> time ----------------------------------------------------

    def ticks_dropped_before(self, stored_index: int) -> int:
        """Acquisition ticks lost at or before stored sample ``stored_index``."""
        lost = 0
        for d in self.dropped:
            # d.start_index is in acquisition-tick numbering; a stored sample
            # index shifts by every completed drop that precedes it.
            if d.start_index <= stored_index + lost:
                lost += d.count
            else:
                break
        return lost

    def is_dropped_tick(self, tick: int) -> bool:
        return any(d.start_index <= tick < d.start_index + d.count for d in self.dropped)

    def index_to_time(
        self, stored_index: int, *, compensate_group_delay: bool = False
    ) -> float:
        """Time of a *stored* sample, honouring dropped ticks and the epoch.

        ``compensate_group_delay=True`` returns the time of the physical event
        the sample reflects (``t - group_delay``) rather than the time the
        sample was written.  The distinction is the whole reason the field
        exists, so the caller must choose.
        """
        if self.rate_hz is None:
            raise TransformError(
                f"clock {self.id!r} is an event clock; index_to_time is undefined",
                remedy="Supply explicit event times.",
                offending_object=self.id,
            )
        if stored_index < 0:
            raise TransformError(
                f"negative sample index {stored_index} on clock {self.id!r}",
                remedy="Indices start at 0.",
                offending_object=stored_index,
            )
        if self.epoch is None:
            raise TransformError(
                f"clock {self.id!r} has no declared epoch; a sample index cannot "
                "be turned into a time",
                remedy=(
                    "Declare the epoch from the trigger record, or align through "
                    "a clock edge that supplies it."
                ),
                offending_object=self.id,
            )
        tick = stored_index + self.ticks_dropped_before(stored_index)
        t = self.epoch + tick * self.dt
        return t - self.group_delay_s if compensate_group_delay else t

    def tick_to_time(self, tick: int, *, compensate_group_delay: bool = False) -> float:
        """Time of an acquisition tick, which may have been dropped."""
        if self.epoch is None or self.rate_hz is None:
            raise TransformError(
                f"clock {self.id!r} lacks an epoch or a rate",
                remedy="Declare both before converting ticks to times.",
                offending_object=self.id,
            )
        t = self.epoch + tick * self.dt
        return t - self.group_delay_s if compensate_group_delay else t

    def time_to_index(self, t: float) -> int:
        """Nearest stored sample index at or before ``t``; refuses inside a gap."""
        if self.epoch is None or self.rate_hz is None:
            raise TransformError(
                f"clock {self.id!r} lacks an epoch or a rate",
                remedy="Declare both before converting times to indices.",
                offending_object=self.id,
            )
        tick = int(math.floor((t - self.epoch) / self.dt + 1e-9))
        if tick < 0:
            raise TransformError(
                f"time {t:.9g} precedes the epoch of clock {self.id!r}",
                remedy="Query inside the recording.",
                offending_object=t,
            )
        if self.is_dropped_tick(tick):
            raise TransformError(
                f"time {t:.9g} on clock {self.id!r} falls in a dropped-sample gap "
                f"(tick {tick})",
                remedy=(
                    "Missing data is never imputed (ARCHITECTURE.md §7.1). "
                    "Either mark the read Unresolved or interpolate explicitly "
                    "under the clock's declared interpolation policy."
                ),
                offending_object=(self.id, tick),
            )
        lost = sum(d.count for d in self.dropped if d.start_index + d.count <= tick)
        return tick - lost

    def sample_times(
        self, t0: float, t1: float, *, compensate_group_delay: bool = False
    ) -> list[float]:
        """Times of stored samples in ``[t0, t1]``, gaps excluded."""
        if self.epoch is None or self.rate_hz is None:
            return []
        first = max(0, int(math.ceil((t0 - self.epoch) / self.dt - 1e-9)))
        last = int(math.floor((t1 - self.epoch) / self.dt + 1e-9))
        out = []
        for tick in range(first, last + 1):
            if self.is_dropped_tick(tick):
                continue
            out.append(self.tick_to_time(tick, compensate_group_delay=compensate_group_delay))
        return out


def detect_dropped_samples(
    timestamps: Sequence[float], dt: float, *, tolerance: float = 0.25
) -> tuple[DropSpec, ...]:
    """Recover a drop record from observed timestamps.

    A gap of ``k`` nominal periods (within ``tolerance`` of an integer) is
    reported as ``k - 1`` dropped ticks.  Non-integer gaps are refused rather
    than rounded: a 1.5-period gap means the rate or the timestamps are wrong,
    and quietly calling it "one dropped sample" would bury that.
    """
    drops: list[DropSpec] = []
    lost = 0
    for i, (a, b) in enumerate(zip(timestamps, timestamps[1:])):
        gap = (b - a) / dt
        k = round(gap)
        if abs(gap - k) > tolerance:
            raise TransformError(
                f"sample gap {gap:.4f} periods at index {i} is not an integer "
                f"number of ticks (tolerance {tolerance})",
                remedy=(
                    "The declared rate disagrees with the timestamps. Fit a "
                    "clock map (fit_clock_map) instead of assuming drops."
                ),
                offending_object=(i, gap),
            )
        if k > 1:
            drops.append(DropSpec(start_index=i + 1 + lost, count=k - 1, detected_by="gap_scan"))
            lost += k - 1
    return tuple(drops)


# --------------------------------------------------------------------------
# clock maps: affine and piecewise affine
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockMap:
    """Continuous piecewise-affine map ``t_b = t_a + delta(t_a)``.

    ``delta(t) = a0 + a1 (t - t_ref) + sum_j c_j max(0, t - b_j)``

    The hinge basis makes the map continuous by construction and collapses to a
    plain affine map when ``breakpoints`` is empty -- which is the point:
    Appendix C asks for "piecewise drift when one affine clock map is
    insufficient", and the same object serves both cases so the fit can choose.

    ``params`` = ``[a0, a1, c_1, ..., c_k]`` (offset in seconds, drift and hinge
    slopes dimensionless), ``param_cov`` its covariance, ``residual_sd`` the
    unexplained per-observation scatter of the fit.
    """

    params: torch.Tensor
    breakpoints: tuple[float, ...] = ()
    t_ref: float = 0.0
    param_cov: torch.Tensor | None = None
    residual_sd: float = 0.0
    validity: ValidityInterval = field(default_factory=ValidityInterval.unbounded)
    fit_method: str = "declared"
    n_observations: int | None = None

    def __post_init__(self) -> None:
        p = as_tensor(self.params).reshape(-1)
        k = len(self.breakpoints)
        if p.numel() != 2 + k:
            raise TransformError(
                f"clock map with {k} breakpoints needs {2 + k} parameters, got {p.numel()}",
                remedy="params = [offset, drift, hinge_1..hinge_k].",
                offending_object=tuple(p.shape),
            )
        object.__setattr__(self, "params", p)
        object.__setattr__(self, "breakpoints", tuple(float(b) for b in self.breakpoints))
        if self.param_cov is not None:
            C = as_tensor(self.param_cov)
            if C.shape != (p.numel(), p.numel()):
                raise TransformError(
                    f"param_cov must be {p.numel()}x{p.numel()}, got {tuple(C.shape)}",
                    remedy="Match the parameter covariance to the parameter vector.",
                    offending_object=tuple(C.shape),
                )
            object.__setattr__(self, "param_cov", 0.5 * (C + C.T))

    # -- construction ------------------------------------------------------

    @staticmethod
    def affine(
        offset: float, drift: float = 0.0, *, t_ref: float = 0.0, **kw: Any
    ) -> "ClockMap":
        return ClockMap(
            torch.tensor([offset, drift], dtype=DTYPE), (), t_ref=t_ref, **kw
        )

    @staticmethod
    def identity() -> "ClockMap":
        return ClockMap(torch.zeros(2, dtype=DTYPE), (), fit_method="declared_identity")

    # -- evaluation --------------------------------------------------------

    def design(self, t: float) -> torch.Tensor:
        row = [1.0, t - self.t_ref] + [max(0.0, t - b) for b in self.breakpoints]
        return torch.tensor(row, dtype=DTYPE)

    def slope(self, t: float) -> float:
        """``d t_b / d t_a`` = 1 + drift(t)."""
        s = 1.0 + float(self.params[1])
        for j, b in enumerate(self.breakpoints):
            if t > b:
                s += float(self.params[2 + j])
        return s

    def drift_at(self, t: float) -> float:
        """Fractional rate error at ``t`` (dimensionless, e.g. 3e-5 = 30 ppm)."""
        return self.slope(t) - 1.0

    def offset_at(self, t: float) -> float:
        return float(self.design(t) @ self.params)

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Return ``(t_out, d t_out / d t_in, var_contributed)``."""
        d = self.design(t)
        delta = float(d @ self.params)
        var = self.residual_sd**2
        if self.param_cov is not None:
            var += float(d @ self.param_cov @ d)
        return t + delta, self.slope(t), var

    def invert(self, t_out: float, *, tol: float = 1e-12, max_iter: int = 64) -> float:
        """Numeric inverse.  Refuses non-monotone maps (no unique inverse)."""
        # a strictly increasing map is required for a unique inverse
        probes = [self.t_ref, t_out] + list(self.breakpoints)
        if any(self.slope(p) <= 0 for p in probes):
            raise NonInvertibleTransformError(
                "clock map is not strictly increasing; time ordering would be "
                "reversed and the inverse is not unique",
                remedy="A clock that runs backwards is a fit failure, not a map.",
                offending_object=self.params.tolist(),
            )
        t = t_out
        for _ in range(max_iter):
            f, s, _ = self.evaluate(t)
            step = (f - t_out) / s
            t -= step
            if abs(step) < tol:
                return t
        raise NonInvertibleTransformError(  # pragma: no cover - Newton on a PL map converges
            f"clock map inversion did not converge for t_out={t_out}",
            remedy="Check the map parameters.",
            offending_object=t_out,
        )

    def summary(self) -> dict[str, Any]:
        """Human-facing summary (ppm units).  Not the serialization format."""
        return {
            "offset_s": float(self.params[0]),
            "drift_ppm": float(self.params[1]) * 1e6,
            "breakpoints": list(self.breakpoints),
            "hinge_slopes_ppm": [float(v) * 1e6 for v in self.params[2:]],
            "residual_sd_s": self.residual_sd,
            "fit_method": self.fit_method,
            "n_observations": self.n_observations,
            "validity": str(self.validity),
        }

    def to_dict(self) -> dict[str, Any]:
        """Lossless serialization (piecewise maps included)."""
        return {
            "params": self.params.tolist(),
            "breakpoints": list(self.breakpoints),
            "t_ref": self.t_ref,
            "param_cov": None if self.param_cov is None else self.param_cov.tolist(),
            "residual_sd": self.residual_sd,
            "fit_method": self.fit_method,
            "n_observations": self.n_observations,
            "validity": {
                "start": self.validity.start,
                "end": self.validity.end,
                "clock": self.validity.clock,
            },
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ClockMap":
        v = d.get("validity") or {}
        return ClockMap(
            torch.tensor(list(d["params"]), dtype=DTYPE),
            tuple(d.get("breakpoints", ())),
            t_ref=float(d.get("t_ref", 0.0)),
            param_cov=(
                None if d.get("param_cov") is None else torch.tensor(d["param_cov"], dtype=DTYPE)
            ),
            residual_sd=float(d.get("residual_sd", 0.0)),
            fit_method=str(d.get("fit_method", "declared")),
            n_observations=d.get("n_observations"),
            validity=ValidityInterval(
                v.get("start"), v.get("end"), str(v.get("clock", "wall"))
            ),
        )


def fit_clock_map(
    t_a: Sequence[float],
    t_b: Sequence[float],
    *,
    breakpoints: Sequence[float] | None = None,
    max_breakpoints: int = 2,
    n_candidates: int = 24,
    criterion: str = "bic",
    fit_method: str = "least_squares",
    validity: ValidityInterval | None = None,
    min_segment_fraction: float = 0.15,
) -> ClockMap:
    """Fit ``t_b = t_a + delta(t_a)`` by least squares on the hinge basis.

    With ``breakpoints=None`` the breakpoints are *selected*: candidates are
    scanned greedily and accepted only when they improve BIC, so a genuinely
    affine relation stays affine and a genuine drift change is recovered
    ("piecewise drift when one affine clock map is insufficient").

    Two guards keep the selection from manufacturing drift changes out of noise,
    which would show up downstream as a confidently wrong alignment:

    * each breakpoint costs **two** parameters in the criterion (its slope *and*
      its location, which the search chose by looking at the data);
    * a breakpoint must leave at least ``min_segment_fraction`` of the
      observations on each side.  A drift change inferred from the last five
      sync pulses is not a drift change.
    """
    ta = as_tensor(list(t_a)).reshape(-1)
    tb = as_tensor(list(t_b)).reshape(-1)
    if ta.numel() != tb.numel():
        raise TransformError(
            f"clock observation vectors differ in length: {ta.numel()} vs {tb.numel()}",
            remedy="Pair each synchronization observation.",
            offending_object=(ta.numel(), tb.numel()),
        )
    n = int(ta.numel())
    if n < 3:
        raise TransformError(
            f"a clock map needs at least 3 synchronization observations, got {n}",
            remedy=(
                "Two points fit an affine map exactly and leave no residual, "
                "so the timing uncertainty would be reported as zero."
            ),
            offending_object=n,
        )
    t_ref = float(ta[0])
    y = tb - ta

    def _fit(bps: tuple[float, ...]) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        cols = [torch.ones(n, dtype=DTYPE), ta - t_ref]
        for b in bps:
            cols.append(torch.clamp(ta - b, min=0.0))
        X = torch.stack(cols, dim=1)
        XtX = X.T @ X
        if float(torch.linalg.matrix_rank(XtX)) < XtX.shape[0]:
            raise TransformError(
                "clock-map design matrix is rank deficient; the proposed "
                "breakpoints are not supported by the observations",
                remedy="Reduce max_breakpoints or supply more sync events.",
                offending_object=bps,
            )
        XtX_inv = torch.linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta
        dof = max(1, n - X.shape[1])
        sigma2 = float(resid @ resid) / dof
        cov = XtX_inv * sigma2
        rss = float(resid @ resid)
        return beta, cov, math.sqrt(max(sigma2, 0.0)), rss

    def _ic(rss: float, k: int) -> float:
        sigma2 = max(rss / n, 1e-300)
        ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
        p = 2 + 2 * k  # slope + location for each selected breakpoint
        return (p * math.log(n) - 2 * ll) if criterion == "bic" else (2 * p - 2 * ll)

    min_side = max(3, int(math.ceil(min_segment_fraction * n)))

    def _supported(b: float) -> bool:
        left = int((ta <= b).sum())
        return left >= min_side and (n - left) >= min_side

    if breakpoints is not None:
        bps = tuple(sorted(float(b) for b in breakpoints))
        beta, cov, sd, rss = _fit(bps)
    else:
        bps: tuple[float, ...] = ()
        beta, cov, sd, rss = _fit(bps)
        best_ic = _ic(rss, 0)
        lo, hi = float(ta.min()), float(ta.max())
        span = hi - lo
        for _ in range(max_breakpoints):
            # one breakpoint per round: pick the single best candidate, then
            # re-scan. Accepting every improving candidate inside one sweep
            # would add a hinge at every grid point.
            best_trial = None
            for m in range(1, n_candidates):
                cand = lo + span * m / n_candidates
                if any(abs(cand - b) < span / (2 * n_candidates) for b in bps):
                    continue
                if not _supported(cand):
                    continue
                trial = tuple(sorted(bps + (cand,)))
                try:
                    b2, c2, s2, r2 = _fit(trial)
                except TransformError:
                    continue
                ic = _ic(r2, len(trial))
                if ic < best_ic - 1e-9 and (best_trial is None or ic < best_trial[0]):
                    best_trial = (ic, trial, b2, c2, s2, r2)
            if best_trial is None:
                break
            best_ic, bps, beta, cov, sd, rss = best_trial

    return ClockMap(
        beta,
        bps,
        t_ref=t_ref,
        param_cov=cov,
        residual_sd=sd,
        fit_method=fit_method + (f"+{criterion}_breakpoints" if breakpoints is None else ""),
        n_observations=n,
        validity=validity or ValidityInterval(float(ta.min()), float(ta.max())),
    )


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockEdge:
    """A directed clock relation ``source -> target`` with its evidence."""

    source: str
    target: str
    map: ClockMap
    evidence: str
    calibration: CalibrationRecord = field(default_factory=CalibrationRecord)
    notes: str = ""
    #: True when this edge is the reverse view of a map declared in the other
    #: direction, so traversing it means *inverting* ``map`` rather than
    #: evaluating it.  Without this flag a bidirectional relation silently
    #: applies the offset twice.
    inverted: bool = False

    def __post_init__(self) -> None:
        if self.evidence not in ADMISSIBLE_CLOCK_EVIDENCE:
            raise ClockRelationUnknownError(
                f"clock edge {self.source}->{self.target} declares evidence "
                f"{self.evidence!r}, which is not an admissible synchronization "
                f"basis {sorted(ADMISSIBLE_CLOCK_EVIDENCE)}",
                remedy=(
                    "Appendix C layer 4 requires a physical synchronization event "
                    "or an independent cross-correlation target. Co-occurrence in "
                    "the same session is not a clock relation."
                ),
                offending_object=(self.source, self.target, self.evidence),
            )

    @property
    def label(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class AlignResult:
    """Time on the target clock plus the timing uncertainty that produced it."""

    time: float
    sd: float
    path: tuple[str, ...]
    edges: tuple[str, ...]
    variance_terms: dict[str, float]
    validity_checks: tuple[ValidityCheck, ...] = ()
    warnings: tuple[str, ...] = ()

    def __iter__(self):  # so `t, sd = graph.align(...)` reads naturally
        yield self.time
        yield self.sd

    def as_record(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "sd": self.sd,
            "path": list(self.path),
            "edges": list(self.edges),
            "variance_terms": self.variance_terms,
            "validity": [v.as_record() for v in self.validity_checks],
            "warnings": list(self.warnings),
        }


class ClockGraph:
    """Clocks and the evidenced relations between them."""

    def __init__(self, *, expiry_policy: ExpiryPolicy | str = ExpiryPolicy.REFUSE) -> None:
        self._clocks: dict[str, ClockSpec] = {}
        self._edges: dict[str, list[ClockEdge]] = {}
        self.expiry_policy = ExpiryPolicy.coerce(expiry_policy)

    # -- declaration -------------------------------------------------------

    def add_clock(self, spec: ClockSpec) -> ClockSpec:
        if spec.id in self._clocks:
            raise TransformError(
                f"clock {spec.id!r} is already declared",
                remedy="Clock ids are unique; version the id if the device changed.",
                offending_object=spec.id,
            )
        self._clocks[spec.id] = spec
        self._edges.setdefault(spec.id, [])
        return spec

    def clock(self, cid: str) -> ClockSpec:
        try:
            return self._clocks[cid]
        except KeyError:
            raise ClockRelationUnknownError(
                f"clock {cid!r} is not declared in this graph "
                f"(known: {sorted(self._clocks)})",
                remedy="Declare the clock with its rate, epoch and trigger path.",
                offending_object=cid,
            ) from None

    @property
    def clocks(self) -> dict[str, ClockSpec]:
        return dict(self._clocks)

    def relate(
        self,
        source: str,
        target: str,
        cmap: ClockMap,
        *,
        evidence: str,
        calibration: CalibrationRecord | None = None,
        bidirectional: bool = True,
        notes: str = "",
    ) -> ClockEdge:
        """Declare a clock relation.  Refuses unevidenced synchronization."""
        self.clock(source)
        self.clock(target)
        edge = ClockEdge(
            source,
            target,
            cmap,
            evidence,
            calibration or CalibrationRecord(method=evidence, validity=cmap.validity),
            notes,
        )
        self._edges[source].append(edge)
        if bidirectional:
            self._edges[target].append(
                ClockEdge(
                    target,
                    source,
                    cmap,
                    evidence,
                    edge.calibration,
                    notes=(notes + " (traversed inverted)").strip(),
                    inverted=True,
                )
            )
        return edge

    def edges(self) -> list[ClockEdge]:
        return [e for lst in self._edges.values() for e in lst]

    # -- alignment ---------------------------------------------------------

    def _find_path(self, a: str, b: str) -> list[ClockEdge]:
        self.clock(a)
        self.clock(b)
        if a == b:
            return []
        seen = {a}
        frontier: list[tuple[str, list[ClockEdge]]] = [(a, [])]
        while frontier:
            node, chain = frontier.pop(0)
            for e in self._edges.get(node, []):
                nxt = e.target if e.source == node else e.source
                if nxt in seen:
                    continue
                new_chain = chain + [e]
                if nxt == b:
                    return new_chain
                seen.add(nxt)
                frontier.append((nxt, new_chain))
        raise ClockRelationUnknownError(
            f"no evidenced relation connects clock {a!r} to clock {b!r}",
            remedy=(
                "Record a physical synchronization event (shared trigger) or an "
                "independent cross-correlation target and declare the edge. Until "
                "then these two streams cannot be placed on a common timeline; "
                "the runtime returns Unresolved rather than a number."
            ),
            offending_object=(a, b),
        )

    def align(
        self,
        clock_a: str,
        clock_b: str,
        t: float,
        *,
        expiry_policy: ExpiryPolicy | str | None = None,
        triggers_fired: Sequence[str] = (),
        include_jitter: bool = True,
    ) -> AlignResult:
        """Map time ``t`` on ``clock_a`` to ``clock_b``, with timing uncertainty.

        Refuses (R01) when no evidenced relation exists.  Variance accumulates,
        along the path: clock-map parameter covariance, fit residual, and the
        endpoint clocks' jitter.  Group delay is *reported*, never silently
        applied -- whether a sample's label or its physical instant is wanted is
        the caller's declaration.
        """
        policy = ExpiryPolicy.coerce(expiry_policy or self.expiry_policy)
        chain = self._find_path(clock_a, clock_b)

        cur_t = float(t)
        var = 0.0
        terms: dict[str, float] = {}
        checks: list[ValidityCheck] = []
        warnings: list[str] = []
        path_nodes = [clock_a]
        edge_labels: list[str] = []

        # forward pass: value and per-edge variance; slopes composed afterwards
        per_edge: list[tuple[str, float, float]] = []  # (label, var_i, slope_i)
        node = clock_a
        for e in chain:
            along_edge = e.source == node
            # traverse the *map* forwards only when we walk the edge in its
            # declared direction and the edge is not itself a reverse view
            apply_forward = along_edge != e.inverted
            chk = e.calibration.check(
                cur_t, policy=policy, label=e.label, triggers_fired=triggers_fired
            )
            checks.append(chk)
            if not chk.inside:
                warnings.append(chk.reason)
            if apply_forward:
                t_new, slope, v = e.map.evaluate(cur_t)
            else:
                t_new = e.map.invert(cur_t)
                _, fwd_slope, v = e.map.evaluate(t_new)
                slope = 1.0 / fwd_slope
                v = v / (fwd_slope**2)
            node = e.target if along_edge else e.source
            v = v * chk.inflation_factor
            per_edge.append((e.label, v, slope))
            edge_labels.append(e.label)
            path_nodes.append(node)
            cur_t = t_new

        # variance of the composition: an early edge's error is amplified by the
        # slopes of every downstream edge
        for i, (label, v, _) in enumerate(per_edge):
            gain = 1.0
            for _, _, s in per_edge[i + 1 :]:
                gain *= s
            contrib = v * gain * gain
            terms[f"map:{label}"] = contrib
            var += contrib

        if include_jitter:
            ja = self.clock(clock_a).jitter_sd_s
            jb = self.clock(clock_b).jitter_sd_s
            if not per_edge:
                # aligning a clock to itself: one jitter term, not two
                terms["jitter:source"] = ja**2
                var += terms["jitter:source"]
            else:
                total_slope = 1.0
                for _, _, s in per_edge:
                    total_slope *= s
                terms["jitter:source"] = (ja * total_slope) ** 2
                terms["jitter:target"] = jb**2
                var += terms["jitter:source"] + terms["jitter:target"]

        gd_a = self.clock(clock_a).group_delay_s
        gd_b = self.clock(clock_b).group_delay_s
        if abs(gd_a - gd_b) > 0:
            warnings.append(
                f"group delays differ ({gd_a:.6g} s on {clock_a} vs {gd_b:.6g} s on "
                f"{clock_b}); align() maps sample *labels*. Use "
                "compensate_group_delay=True on index_to_time to reach the "
                "physical instant."
            )
        terms["group_delay_difference_s"] = gd_a - gd_b

        return AlignResult(
            time=cur_t,
            sd=math.sqrt(max(var, 0.0)),
            path=tuple(path_nodes),
            edges=tuple(edge_labels),
            variance_terms=terms,
            validity_checks=tuple(checks),
            warnings=tuple(warnings),
        )


# --------------------------------------------------------------------------
# firing rules (composable clock expressions in physical time)
# --------------------------------------------------------------------------


class ClockExpr:
    """Base of the firing-rule AST.

    Adapted in shape from canvas-engineering's ``clock_ir``; the difference is
    that :meth:`fire_times` produces **physical times on a named clock**, not
    integer step indices, and every leaf declares which clock it is stated on.
    """

    def fire_times(self, ctx: "ScheduleContext") -> list[float]:
        raise NotImplementedError

    def clocks_used(self) -> set[str]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def __and__(self, other: "ClockExpr") -> "ClockExpr":
        return AndExpr(self, other)

    def __or__(self, other: "ClockExpr") -> "ClockExpr":
        return OrExpr(self, other)


@dataclass(frozen=True)
class ScheduleContext:
    """Window over which firing rules are resolved, on a reference clock."""

    graph: ClockGraph
    reference: str
    t0: float
    t1: float
    coincidence_tolerance: float = 1e-3
    events: Mapping[str, Sequence[float]] = field(default_factory=dict)
    boundaries: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def to_reference(self, clock: str, t: float) -> tuple[float, float]:
        r = self.graph.align(clock, self.reference, t)
        return r.time, r.sd


@dataclass(frozen=True)
class Periodic(ClockExpr):
    """Fire every ``period`` seconds on ``clock``, offset by ``phase``."""

    period: float
    clock: str
    phase: float = 0.0

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        if self.period <= 0:
            raise TransformError(
                f"periodic firing rule needs a positive period, got {self.period}",
                remedy="Declare the period in seconds.",
                offending_object=self.period,
            )
        # resolve the window into the leaf's own clock, then emit and map back
        t0 = ctx.graph.align(ctx.reference, self.clock, ctx.t0).time
        t1 = ctx.graph.align(ctx.reference, self.clock, ctx.t1).time
        k0 = math.ceil((t0 - self.phase) / self.period - 1e-9)
        k1 = math.floor((t1 - self.phase) / self.period + 1e-9)
        out = []
        for k in range(k0, k1 + 1):
            local = self.phase + k * self.period
            out.append(ctx.graph.align(self.clock, ctx.reference, local).time)
        return out

    def clocks_used(self) -> set[str]:
        return {self.clock}

    def to_dict(self) -> dict:
        return {"type": "periodic", "period": self.period, "clock": self.clock, "phase": self.phase}


@dataclass(frozen=True)
class OnEvent(ClockExpr):
    """Fire at declared event times of ``source`` (stated on ``clock``)."""

    source: str
    clock: str

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        if self.source not in ctx.events:
            raise TransformError(
                f"firing rule references event source {self.source!r}, which the "
                "schedule context does not supply",
                remedy="Pass the recorded event times; do not synthesize them.",
                offending_object=self.source,
            )
        out = []
        for t in ctx.events[self.source]:
            tr = ctx.graph.align(self.clock, ctx.reference, float(t)).time
            if ctx.t0 - 1e-12 <= tr <= ctx.t1 + 1e-12:
                out.append(tr)
        return out

    def clocks_used(self) -> set[str]:
        return {self.clock}

    def to_dict(self) -> dict:
        return {"type": "on_event", "source": self.source, "clock": self.clock}


@dataclass(frozen=True)
class Boundary(ClockExpr):
    """Fire at a named lifecycle boundary (block start, episode end, ...)."""

    name: str
    clock: str

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        times = ctx.boundaries.get(self.name)
        if times is None:
            raise TransformError(
                f"boundary {self.name!r} is not declared in the schedule context",
                remedy="Supply the recorded boundary times.",
                offending_object=self.name,
            )
        return [
            ctx.graph.align(self.clock, ctx.reference, float(t)).time
            for t in times
            if ctx.t0 - 1e-12 <= ctx.graph.align(self.clock, ctx.reference, float(t)).time <= ctx.t1 + 1e-12
        ]

    def clocks_used(self) -> set[str]:
        return {self.clock}

    def to_dict(self) -> dict:
        return {"type": "boundary", "name": self.name, "clock": self.clock}


def _merge(a: Iterable[float], b: Iterable[float], tol: float) -> list[float]:
    out = sorted(list(a) + list(b))
    merged: list[float] = []
    for t in out:
        if merged and abs(t - merged[-1]) <= tol:
            continue
        merged.append(t)
    return merged


@dataclass(frozen=True)
class AndExpr(ClockExpr):
    left: ClockExpr
    right: ClockExpr

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        lt = self.left.fire_times(ctx)
        rt = self.right.fire_times(ctx)
        return [t for t in lt if any(abs(t - s) <= ctx.coincidence_tolerance for s in rt)]

    def clocks_used(self) -> set[str]:
        return self.left.clocks_used() | self.right.clocks_used()

    def to_dict(self) -> dict:
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass(frozen=True)
class OrExpr(ClockExpr):
    left: ClockExpr
    right: ClockExpr

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        return _merge(
            self.left.fire_times(ctx), self.right.fire_times(ctx), ctx.coincidence_tolerance
        )

    def clocks_used(self) -> set[str]:
        return self.left.clocks_used() | self.right.clocks_used()

    def to_dict(self) -> dict:
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass(frozen=True)
class Cooldown(ClockExpr):
    """Suppress re-firing within ``seconds`` of the previous fire."""

    child: ClockExpr
    seconds: float

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        out: list[float] = []
        for t in sorted(self.child.fire_times(ctx)):
            if out and t - out[-1] < self.seconds:
                continue
            out.append(t)
        return out

    def clocks_used(self) -> set[str]:
        return self.child.clocks_used()

    def to_dict(self) -> dict:
        return {"type": "cooldown", "child": self.child.to_dict(), "seconds": self.seconds}


@dataclass(frozen=True)
class MaxSilence(ClockExpr):
    """Force a fire whenever the child has been silent for ``seconds``."""

    child: ClockExpr
    seconds: float

    def fire_times(self, ctx: ScheduleContext) -> list[float]:
        base = sorted(self.child.fire_times(ctx))
        out: list[float] = []
        last = ctx.t0
        for t in base:
            while t - last > self.seconds:
                last = last + self.seconds
                out.append(last)
            out.append(t)
            last = t
        while ctx.t1 - last > self.seconds:
            last = last + self.seconds
            out.append(last)
        return _merge(out, [], ctx.coincidence_tolerance)

    def clocks_used(self) -> set[str]:
        return self.child.clocks_used()

    def to_dict(self) -> dict:
        return {"type": "max_silence", "child": self.child.to_dict(), "seconds": self.seconds}


# --------------------------------------------------------------------------
# multirate scheduling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpolationContract:
    """What it would take to read ``clock``'s data at a requested time.

    ``admissible=False`` is a first-class outcome: an event clock, or a gap
    wider than the declared maximum, yields a refusal rather than a number.
    """

    clock: str
    target_time: float
    method: InterpolationPolicy
    nearest_sample_time: float | None
    gap_s: float | None
    max_gap_s: float | None
    integration_window_s: float
    group_delay_s: float
    timing_sd_s: float
    admissible: bool
    reason: str = ""

    def require(self) -> "InterpolationContract":
        if not self.admissible:
            raise TransformError(
                f"cannot read clock {self.clock!r} at t={self.target_time:.9g}: {self.reason}",
                remedy=(
                    "Return Unresolved, widen the declared interpolation policy "
                    "with evidence, or schedule the read at a supported time."
                ),
                offending_object=self.clock,
            )
        return self

    def as_record(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "target_time": self.target_time,
            "method": self.method,
            "nearest_sample_time": self.nearest_sample_time,
            "gap_s": self.gap_s,
            "max_gap_s": self.max_gap_s,
            "integration_window_s": self.integration_window_s,
            "group_delay_s": self.group_delay_s,
            "timing_sd_s": self.timing_sd_s,
            "admissible": self.admissible,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SyncPoint:
    """A time (on the reference clock) where several clocks can be read together."""

    time: float
    clocks: tuple[str, ...]
    contracts: tuple[InterpolationContract, ...]
    timing_sd_s: float

    @property
    def admissible(self) -> bool:
        return all(c.admissible for c in self.contracts)


@dataclass(frozen=True)
class MultirateSchedule:
    """Events, sync points and interpolation contracts over a window."""

    reference: str
    t0: float
    t1: float
    events: dict[str, list[float]]
    sync_points: tuple[SyncPoint, ...]
    hyperperiod_s: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def admissible_sync_points(self) -> tuple[SyncPoint, ...]:
        return tuple(s for s in self.sync_points if s.admissible)

    def as_record(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "window": [self.t0, self.t1],
            "hyperperiod_s": self.hyperperiod_s,
            "n_events": {k: len(v) for k, v in self.events.items()},
            "sync_points": [
                {
                    "time": s.time,
                    "clocks": list(s.clocks),
                    "timing_sd_s": s.timing_sd_s,
                    "admissible": s.admissible,
                    "contracts": [c.as_record() for c in s.contracts],
                }
                for s in self.sync_points
            ],
            "provenance": self.provenance,
        }


def hyperperiod(rates: Sequence[Fraction | float]) -> float | None:
    """Exact hyperperiod of a set of rational rates (``None`` if any is absent)."""
    periods = []
    for r in rates:
        if r is None:
            return None
        periods.append(1 / Fraction(r).limit_denominator(10**9))
    if not periods:
        return None
    acc = periods[0]
    for p in periods[1:]:
        # lcm of two rationals = lcm(num)/gcd(den)
        acc = Fraction(
            math.lcm(acc.numerator, p.numerator), math.gcd(acc.denominator, p.denominator)
        )
    return float(acc)


def interpolation_contract(
    graph: ClockGraph,
    clock: str,
    t_reference: float,
    *,
    reference: str,
) -> InterpolationContract:
    """What reading ``clock`` at reference-time ``t_reference`` would require."""
    spec = graph.clock(clock)
    aligned = graph.align(reference, clock, t_reference)
    t_local = aligned.time

    if spec.rate_hz is None or spec.epoch is None:
        return InterpolationContract(
            clock=clock,
            target_time=t_reference,
            method="none",
            nearest_sample_time=None,
            gap_s=None,
            max_gap_s=spec.max_interpolation_gap_s,
            integration_window_s=spec.integration_window_s,
            group_delay_s=spec.group_delay_s,
            timing_sd_s=aligned.sd,
            admissible=False,
            reason=(
                f"clock {clock!r} is an event/undated clock (rate={spec.rate_hz}, "
                f"epoch={spec.epoch}); it has no sample grid to interpolate on"
            ),
        )

    dt = spec.dt
    tick = (t_local - spec.epoch) / dt
    lo_tick, hi_tick = math.floor(tick), math.ceil(tick)
    candidates = [
        spec.tick_to_time(k)
        for k in (lo_tick, hi_tick)
        if k >= 0 and not spec.is_dropped_tick(k)
    ]
    if not candidates:
        return InterpolationContract(
            clock, t_reference, "none", None, None, spec.max_interpolation_gap_s,
            spec.integration_window_s, spec.group_delay_s, aligned.sd, False,
            reason=(
                f"the samples bracketing t={t_local:.9g} on clock {clock!r} were "
                "dropped; missing data is never imputed"
            ),
        )
    nearest = min(candidates, key=lambda x: abs(x - t_local))
    gap = abs(nearest - t_local)
    max_gap = spec.max_interpolation_gap_s if spec.max_interpolation_gap_s is not None else dt

    if spec.interpolation_policy == "none":
        admissible = gap <= 1e-12
        reason = (
            ""
            if admissible
            else (
                f"clock {clock!r} declares interpolation_policy='none' (a "
                "point-process / event stream); resampling it to t="
                f"{t_local:.9g} would invent samples"
            )
        )
    elif gap > max_gap:
        admissible = False
        reason = (
            f"nearest sample on {clock!r} is {gap:.6g} s away, beyond the declared "
            f"maximum interpolation gap {max_gap:.6g} s"
        )
    else:
        admissible = True
        reason = ""

    return InterpolationContract(
        clock=clock,
        target_time=t_reference,
        method=spec.interpolation_policy,
        nearest_sample_time=nearest,
        gap_s=gap,
        max_gap_s=max_gap,
        integration_window_s=spec.integration_window_s,
        group_delay_s=spec.group_delay_s,
        timing_sd_s=aligned.sd,
        admissible=admissible,
        reason=reason,
    )


def schedule_multirate(
    graph: ClockGraph,
    clocks: Sequence[str],
    *,
    reference: str,
    t0: float,
    t1: float,
    tolerance: float = 1e-3,
    min_clocks_per_sync: int = 2,
    rules: Mapping[str, ClockExpr] | None = None,
    context: ScheduleContext | None = None,
) -> MultirateSchedule:
    """Produce the events, synchronization points and interpolation contracts.

    Sample events for each clock are enumerated on that clock's own grid (gaps
    excluded) and mapped into the reference timeline.  A **sync point** is a
    reference time where events of at least ``min_clocks_per_sync`` clocks
    coincide within ``tolerance``; each carries an
    :class:`InterpolationContract` per participating clock so a fusion step can
    see, before it runs, exactly which reads are supported and which are not.

    ``rules`` optionally adds named :class:`ClockExpr` firing rules (periodic,
    event-driven, boundary, with cooldown / max-silence) whose fire times join
    the event table.
    """
    for c in clocks:
        graph.clock(c)
    graph.clock(reference)

    events: dict[str, list[float]] = {}
    for c in clocks:
        spec = graph.clock(c)
        if spec.rate_hz is None or spec.epoch is None:
            events[c] = []
            continue
        lo = graph.align(reference, c, t0).time
        hi = graph.align(reference, c, t1).time
        local = spec.sample_times(lo, hi)
        events[c] = [graph.align(c, reference, t).time for t in local]

    if rules:
        ctx = context or ScheduleContext(graph, reference, t0, t1, tolerance)
        for name, expr in rules.items():
            events[f"rule:{name}"] = expr.fire_times(ctx)

    # candidate sync points: cluster all event times within tolerance
    stamped: list[tuple[float, str]] = [
        (t, c) for c, ts in events.items() for t in ts if not c.startswith("rule:")
    ]
    stamped.sort()
    sync: list[SyncPoint] = []
    i = 0
    while i < len(stamped):
        j = i
        members: dict[str, float] = {}
        while j < len(stamped) and stamped[j][0] - stamped[i][0] <= tolerance:
            members.setdefault(stamped[j][1], stamped[j][0])
            j += 1
        if len(members) >= min_clocks_per_sync:
            t_mid = sum(members.values()) / len(members)
            contracts = tuple(
                interpolation_contract(graph, c, t_mid, reference=reference)
                for c in sorted(members)
            )
            sd = math.sqrt(sum(c.timing_sd_s**2 for c in contracts) / max(len(contracts), 1))
            sync.append(SyncPoint(t_mid, tuple(sorted(members)), contracts, sd))
            i = j
        else:
            i += 1

    hp = hyperperiod([graph.clock(c).rate_hz for c in clocks])
    return MultirateSchedule(
        reference=reference,
        t0=t0,
        t1=t1,
        events=events,
        sync_points=tuple(sync),
        hyperperiod_s=hp,
        provenance={
            "clocks": list(clocks),
            "tolerance_s": tolerance,
            "min_clocks_per_sync": min_clocks_per_sync,
            "rules": sorted(rules) if rules else [],
        },
    )


__all__ = [
    "ADMISSIBLE_CLOCK_EVIDENCE",
    "DropSpec",
    "ClockSpec",
    "ClockMap",
    "ClockEdge",
    "ClockGraph",
    "AlignResult",
    "fit_clock_map",
    "detect_dropped_samples",
    "hyperperiod",
    "ClockExpr",
    "ScheduleContext",
    "Periodic",
    "OnEvent",
    "Boundary",
    "AndExpr",
    "OrExpr",
    "Cooldown",
    "MaxSilence",
    "InterpolationContract",
    "SyncPoint",
    "MultirateSchedule",
    "interpolation_contract",
    "schedule_multirate",
]
