"""Intervention operator interface implementing the thesis Sec. 2.4 controlled SDE.

**SIMULATION ONLY.**  Everything in :mod:`scwbd.intervene` operates on simulated
fields, simulated tissue, and simulated or previously-recorded open-data
responses.  That is a statement about what these objects *are*, and it is the
only scope claim this module makes.

The governing form (thesis Sec. 2.4) is

.. math::

    \\mathrm dX(t) = \\mathcal F(X,t)\\,\\mathrm dt
        + \\mathcal G_k(X,t;A_p,C,\\omega_k)\\,u_k(t)\\,\\mathrm dt
        + Q^{1/2}\\,\\mathrm dW_t, \\qquad t\\in[t_0,t_1].

Five things stay **distinct fields** and are never collapsed into a scalar
"stimulation strength":

===========================  ===========================================
:class:`DeviceGeometry`      where the device is and what shape it is
:class:`WaveformSpec`        the drive :math:`u_k(t)` and burst structure
:class:`ThermalHistory`      cumulative thermal state across the session
:class:`TissueCoupling`      how the physical field couples into tissue
:class:`MechanisticUncertainty`  which coupling story is being assumed
===========================  ===========================================

The instantaneous jump
:math:`X(t_0^+) = \\mathcal I_k(X(t_0^-), \\int u_k\\,\\mathrm dt, \\ldots)` is
available only behind ``impulse_limit=True``, which *runs the test that would
justify it* (:class:`ImpulseLimitReport`) and refuses when the error against the
finite-duration integration exceeds tolerance.

Finally, the four quantities the thesis insists are different are four different
Python types: :class:`PhysicalDose`, :class:`TargetEngagement`,
:class:`NetworkEffect`, and :class:`ClinicalUtility` (which refuses to be
constructed at all in this release).  An induced electric field, an acoustic
pressure, or a presented sentence is never a neural effect.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "SIMULATION_ONLY_NOTICE",
    "simulation_only_notice",
    "InterventionRefusal",
    "Ledger",
    "DeviceGeometry",
    "WaveformSpec",
    "BurstSequence",
    "ThermalHistory",
    "TissueCoupling",
    "MechanisticUncertainty",
    "ExposureWindow",
    "PhysicalDose",
    "TargetEngagement",
    "NetworkEffect",
    "ClinicalUtility",
    "ImpulseLimitReport",
    "InterventionResult",
    "InterventionOperator",
    "LinearFieldIntervention",
]

#: Every clause is a property of *this software*, checkable by looking for the
#: thing it says is absent.  There is no device command surface in this
#: package, so the notice describes what the objects are rather than making a
#: claim about the world that nothing verifies.
SIMULATION_ONLY_NOTICE = (
    "SIMULATION ONLY. This object models a simulated physical field and a "
    "simulated neural state. It is not a device driver, not a dosing "
    "protocol, and not a recommendation for any person. This package emits no "
    "device command and drives no hardware."
)


def simulation_only_notice() -> str:
    """Return the notice every public entry point in this package carries."""
    return SIMULATION_ONLY_NOTICE


class InterventionRefusal(RuntimeError):
    """A refusal raised by the intervention stack.

    Mirrors ``scwbd.compiler.CompilerRefusal``: carries a refusal ``code`` from
    ``thesis_contract.tex`` Table ``tab:compiler-refusals``, a ``remedy``, and
    the ``offending_object``.  ``scwbd.intervene.safety`` re-exports the R11
    flavour as ``CompilerRefusal`` so downstream code can catch either.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        remedy: str = "",
        offending_object: Any = None,
    ) -> None:
        super().__init__(f"[{code}] {message}" + (f"  remedy: {remedy}" if remedy else ""))
        self.code = code
        self.message = message
        self.remedy = remedy
        self.offending_object = offending_object


# ---------------------------------------------------------------------------
# uncertainty ledger (thin local adapter; agent A owns the schema version)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ledger:
    """Bias--variance ledger (thesis Sec. 2.7). Bias and variance never collapse.

    Thin local mirror of ``scwbd.schema.UncertaintyLedger`` so that
    ``scwbd.intervene`` can be developed and tested independently of agent A.
    ``to_schema()`` upcasts once the schema object is importable.
    """

    variance: Mapping[str, float] = field(default_factory=dict)
    bias_interval: tuple[float, float] = (0.0, 0.0)
    bias_status: Literal[
        "design_estimable", "externally_bounded", "prior_specified_sensitivity"
    ] = "prior_specified_sensitivity"
    model_discrepancy: float | None = None
    validity_domain: Mapping[str, Any] = field(default_factory=dict)

    def total_variance(self) -> float:
        return float(sum(self.variance.values()))

    def merged(self, other: "Ledger") -> "Ledger":
        """Combine two ledgers additively in variance and interval-wise in bias."""
        var = dict(self.variance)
        for k, v in other.variance.items():
            var[k] = var.get(k, 0.0) + v
        lo = self.bias_interval[0] + other.bias_interval[0]
        hi = self.bias_interval[1] + other.bias_interval[1]
        rank = {
            "design_estimable": 0,
            "externally_bounded": 1,
            "prior_specified_sensitivity": 2,
        }
        status = max((self.bias_status, other.bias_status), key=lambda s: rank[s])
        md = None
        if self.model_discrepancy is not None or other.model_discrepancy is not None:
            md = (self.model_discrepancy or 0.0) + (other.model_discrepancy or 0.0)
        dom = {**dict(self.validity_domain), **dict(other.validity_domain)}
        return Ledger(var, (lo, hi), status, md, dom)

    def to_schema(self) -> Any:
        try:  # pragma: no cover - depends on agent A landing
            from scwbd.schema import UncertaintyLedger  # type: ignore
        except Exception:
            return self
        return UncertaintyLedger(
            variance=dict(self.variance),
            bias_interval=tuple(self.bias_interval),
            bias_status=self.bias_status,
            model_discrepancy=self.model_discrepancy,
            validity_domain=dict(self.validity_domain),
        )


# ---------------------------------------------------------------------------
# the five distinct fields of an intervention
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceGeometry(abc.ABC):
    """Physical geometry of the device. Never merged into the waveform."""

    device_id: str
    frame: str  # e.g. "coil", "transducer" -- a *device* frame, never a scalp label

    @abc.abstractmethod
    def element_positions(self) -> Tensor:
        """[n_elements, 3] element positions in the device frame, metres."""

    def describe(self) -> str:
        return f"{type(self).__name__}(device_id={self.device_id!r}, frame={self.frame!r})"


@dataclass(frozen=True)
class BurstSequence:
    """Burst / train structure. Distinct from the within-pulse waveform."""

    pulse_period: float  # s, inter-pulse interval within a burst
    n_pulses_per_burst: int
    burst_period: float  # s, inter-burst interval
    n_bursts: int
    units: str = "s"

    def pulse_onsets(self) -> Tensor:
        """Onset times of every pulse in the sequence, seconds."""
        b = torch.arange(self.n_bursts, dtype=torch.float64) * self.burst_period
        p = torch.arange(self.n_pulses_per_burst, dtype=torch.float64) * self.pulse_period
        return (b[:, None] + p[None, :]).reshape(-1)

    @property
    def total_pulses(self) -> int:
        return int(self.n_bursts * self.n_pulses_per_burst)

    @property
    def duration(self) -> float:
        onsets = self.pulse_onsets()
        return float(onsets[-1] - onsets[0]) if onsets.numel() else 0.0


@dataclass(frozen=True)
class WaveformSpec:
    """The drive :math:`u_k(t)` in device-native units, plus its burst structure.

    ``sample`` returns the *normalised* drive shape; physical scaling lives in
    the device stack (coil dI/dt in A/s, transducer surface velocity in m/s).
    """

    name: str
    units: str
    period: float  # s, single-pulse duration
    sample_fn: Callable[[Tensor], Tensor]
    burst: BurstSequence | None = None
    amplitude: float = 1.0

    def sample(self, t: Tensor) -> Tensor:
        """Evaluate the waveform at times ``t`` (seconds, single-pulse phase)."""
        return self.amplitude * self.sample_fn(torch.as_tensor(t, dtype=torch.float64))

    def time_integral(self, n: int = 20001) -> float:
        """:math:`\\int_0^{T} u(t)\\,\\mathrm dt` over one pulse, trapezoid rule."""
        t = torch.linspace(0.0, self.period, n, dtype=torch.float64)
        return float(torch.trapezoid(self.sample(t), t))

    def abs_time_integral(self, n: int = 20001) -> float:
        t = torch.linspace(0.0, self.period, n, dtype=torch.float64)
        return float(torch.trapezoid(self.sample(t).abs(), t))


@dataclass(frozen=True)
class ThermalHistory:
    """Cumulative thermal state. A separate field because it *accumulates*.

    ``cem43_s`` is the cumulative equivalent minutes at 43 degC expressed in
    **seconds** internally (converted on read) so that accumulation across
    heterogeneous sampling stays exact.
    """

    baseline_temp_c: float = 37.0
    cem43_s: float = 0.0
    peak_temp_c: float = 37.0
    elapsed_s: float = 0.0

    @property
    def cem43_minutes(self) -> float:
        return self.cem43_s / 60.0

    def accumulate(self, temp_c: Tensor | float, dt: float) -> "ThermalHistory":
        """Add ``dt`` seconds at ``temp_c`` using the Sapareto--Dewey CEM43 rule.

        :math:`\\mathrm{CEM43} = \\sum R^{43-T}\\Delta t` with :math:`R=0.25`
        for :math:`T<43` degC and :math:`R=0.5` for :math:`T\\ge 43` degC
        (Sapareto & Dewey 1984).  ``temp_c`` may be a tensor field, in which
        case the **maximum** over the field is accumulated (worst-case voxel).
        """
        tt = torch.as_tensor(temp_c, dtype=torch.float64)
        t_max = float(tt.max())
        r = 0.5 if t_max >= 43.0 else 0.25
        add = (r ** (43.0 - t_max)) * dt
        return ThermalHistory(
            baseline_temp_c=self.baseline_temp_c,
            cem43_s=self.cem43_s + add,
            peak_temp_c=max(self.peak_temp_c, t_max),
            elapsed_s=self.elapsed_s + dt,
        )


@dataclass(frozen=True)
class TissueCoupling:
    """How the *physical* field couples into tissue. Not a neural effect.

    ``parameters`` are biophysical (conductivity, sound speed, membrane time
    constant); ``mechanistic_status`` records whether this coupling is claimed
    mechanistic, effective, functional, or a surrogate (schema vocabulary).
    """

    name: str
    parameters: Mapping[str, float]
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"]
    citation: str = ""
    ledger: Ledger = field(default_factory=Ledger)


@dataclass(frozen=True)
class MechanisticUncertainty:
    """Which coupling story is being assumed, and how strongly.

    For TMS and for low-intensity tFUS the mechanism is *unresolved*, so this
    is a distribution over named candidate operators, never a single choice.
    """

    candidates: tuple[str, ...]
    log_weights: Tensor  # unnormalised log model weights, [n_candidates]
    resolved: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.candidates) != int(self.log_weights.numel()):
            raise ValueError("candidates and log_weights length mismatch")

    def posterior(self) -> Tensor:
        return torch.softmax(self.log_weights.to(torch.float64), dim=0)

    def entropy_nats(self) -> float:
        p = self.posterior()
        return float(-(p * torch.log(p.clamp_min(1e-300))).sum())

    def disagreement(self, predictions: Tensor) -> float:
        """Posterior-weighted spread of ``predictions`` [n_candidates, ...].

        This is the :math:`\\mathcal U_{\\rm epi}` ingredient of thesis
        Sec. 7.4: reducible model disagreement, kept separate from aleatoric
        outcome variance.
        """
        p = self.posterior().reshape(-1, *([1] * (predictions.dim() - 1)))
        mean = (p * predictions).sum(0)
        var = (p * (predictions - mean) ** 2).sum(0)
        return float(var.sqrt().mean())


@dataclass(frozen=True)
class ExposureWindow:
    """The physical exposure interval :math:`[t_0, t_1]` of thesis Sec. 2.4."""

    t0: float
    t1: float
    clock: str = "device"

    def __post_init__(self) -> None:
        if not self.t1 > self.t0:
            raise ValueError("exposure window must have t1 > t0")

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


# ---------------------------------------------------------------------------
# the four quantities that must never be equated
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalDose:
    """Level 1: what the *physics* did. V/m, Pa, dB SPL, characters of text.

    Constructing this object says nothing whatsoever about neurons.
    """

    modality: str
    quantity: str  # "E_field", "peak_negative_pressure", "sentence"
    units: str
    value: Tensor
    support: str  # frame / mesh / grid the value lives on
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    def peak(self) -> float:
        return float(torch.as_tensor(self.value).abs().max())

    def as_neural_effect(self) -> "NetworkEffect":  # pragma: no cover - refusal
        raise InterventionRefusal(
            "R04",
            "A physical dose cannot be converted to a neural effect. Route it "
            "through an explicit, named candidate response operator under "
            "model comparison.",
            remedy="use scwbd.intervene.tms.response / tfus.response",
            offending_object=self,
        )


@dataclass(frozen=True)
class TargetEngagement:
    """Level 2: modelled drive delivered to a named target population.

    Produced only by an explicitly named candidate response operator.  Carries
    ``response_model`` so no engagement number is ever anonymous.
    """

    target: str
    response_model: str
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"]
    units: str
    value: Tensor
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class NetworkEffect:
    """Level 3: simulated change in distributed state after propagation."""

    readout: str
    units: str
    value: Tensor
    horizon_s: float
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class ClinicalUtility:
    """Level 4: comparative clinical benefit. **Refuses to be constructed.**

    Clinical utility requires a prospective, causally identified comparison in
    people that has actually been *run* (thesis Sec. 7.2 validation ladder).
    No such comparison exists here -- a fact about the evidence in this
    repository, checkable by looking for the dataset.  This type exists solely
    to keep the name from being silently attached to a simulated network
    effect.
    """

    def __post_init__(self) -> None:  # pragma: no cover - always refuses
        raise InterventionRefusal(
            "R11",
            "Clinical utility is not estimable in SC-WBD-001-beta. Field "
            "accuracy, target engagement, network effect and clinical utility "
            "are four separate validation levels and only the first three are "
            "in scope for this release.",
            remedy="report NetworkEffect and stop; do not label it clinical.",
            offending_object="ClinicalUtility",
        )


# ---------------------------------------------------------------------------
# impulse limit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpulseLimitReport:
    """Evidence for (or against) replacing the exposure integral by a jump.

    ``rel_error`` compares the impulse-limit trajectory endpoint against the
    finite-duration integration of the same SDE with the noise switched off
    (the noise is common to both and would only add variance to the
    comparison).  ``admitted`` is ``False`` when the error exceeds ``tol``.
    """

    exposure_duration_s: float
    system_timescale_s: float
    rel_error: float
    abs_error: float
    tol: float
    admitted: bool
    finite_duration_state: Tensor
    impulse_state: Tensor
    n_steps: int
    notice: str = SIMULATION_ONLY_NOTICE

    def summary(self) -> str:
        verdict = "ADMITTED" if self.admitted else "REFUSED"
        return (
            f"impulse limit {verdict}: exposure {self.exposure_duration_s:.3e} s vs "
            f"system timescale {self.system_timescale_s:.3e} s, "
            f"relative endpoint error {self.rel_error:.3e} (tol {self.tol:.3e})"
        )


@dataclass(frozen=True)
class InterventionResult:
    """Trajectory plus the dose that produced it, kept as separate objects."""

    times: Tensor
    states: Tensor
    dose: PhysicalDose | None
    thermal_history: ThermalHistory
    impulse_report: ImpulseLimitReport | None
    ledger: Ledger
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def final_state(self) -> Tensor:
        return self.states[-1]


# ---------------------------------------------------------------------------
# the operator interface
# ---------------------------------------------------------------------------

DriftFn = Callable[[Tensor, float], Tensor]

#: minimum integration steps per pulse for the impulse-limit reference
_MIN_STEPS_PER_PULSE = 8.0

#: quadrature points per pulse used for the impulse-limit drive integral
_QUAD_POINTS_PER_PULSE = 64.0


class InterventionOperator(abc.ABC):
    """Base class for every write channel: TMS, tFUS, sensory/cognitive input.

    Subclasses supply :meth:`gain` (:math:`\\mathcal G_k`), :meth:`drive`
    (:math:`u_k`), and optionally :meth:`impulse_map` (:math:`\\mathcal I_k`).
    The base class owns the SDE integration, the impulse-limit test, and the
    bookkeeping that keeps geometry, waveform, thermal history, coupling and
    mechanistic uncertainty from being fused.
    """

    #: refusal-visible banner, echoed by ``describe()``
    notice: str = SIMULATION_ONLY_NOTICE

    def __init__(
        self,
        *,
        name: str,
        geometry: DeviceGeometry,
        waveform: WaveformSpec,
        coupling: TissueCoupling,
        mechanistic_uncertainty: MechanisticUncertainty,
        thermal_history: ThermalHistory | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.name = name
        self.geometry = geometry
        self.waveform = waveform
        self.coupling = coupling
        self.mechanistic_uncertainty = mechanistic_uncertainty
        self.thermal_history = thermal_history or ThermalHistory()
        self.ledger = ledger or Ledger()

    # -- required physics ---------------------------------------------------

    @abc.abstractmethod
    def gain(
        self,
        x: Tensor,
        t: float,
        *,
        anatomy: Any = None,
        context: Any = None,
    ) -> Tensor:
        """:math:`\\mathcal G_k(X,t;A_p,C,\\omega_k)`, shape ``[n_state, n_chan]``.

        ``anatomy`` is :math:`A_p` (agent C), ``context`` is :math:`C` (session
        and cognitive state).  State dependence is explicit in ``x``.
        """

    @abc.abstractmethod
    def drive(self, t: float) -> Tensor:
        """:math:`u_k(t)`, shape ``[n_chan]``, in the waveform's declared units."""

    def impulse_map(
        self,
        x: Tensor,
        drive_integral: Tensor,
        *,
        anatomy: Any = None,
        context: Any = None,
        t0: float = 0.0,
    ) -> Tensor:
        """:math:`\\mathcal I_k(X(t_0^-), \\int u_k\\,\\mathrm dt, \\ldots)`.

        Default: the first-order (frozen-gain) jump.  Subclasses may override
        with a device-specific map, but the impulse-limit test still applies.
        """
        g = self.gain(x, t0, anatomy=anatomy, context=context)
        return x + g @ drive_integral.to(x.dtype)

    # -- integration --------------------------------------------------------

    def integrate(
        self,
        x0: Tensor,
        drift: DriftFn,
        window: ExposureWindow,
        *,
        dt: float,
        Q_sqrt: Tensor | None = None,
        seed: int | None = None,
        anatomy: Any = None,
        context: Any = None,
        impulse_limit: bool = False,
        impulse_tol: float = 1e-2,
        dose: PhysicalDose | None = None,
    ) -> InterventionResult:
        """Integrate the Sec. 2.4 SDE over ``window`` with Euler--Maruyama.

        With ``impulse_limit=True`` the operator *first* runs
        :meth:`check_impulse_limit`, reports the error against the
        finite-duration integration, and **refuses** (``InterventionRefusal``
        R06) if that error exceeds ``impulse_tol``.  The flag is never a free
        modelling convenience.
        """
        if seed is None and Q_sqrt is not None:
            raise ValueError("stochastic integration requires an explicit seed")

        report: ImpulseLimitReport | None = None
        if impulse_limit:
            report = self.check_impulse_limit(
                x0,
                drift,
                window,
                dt=dt,
                anatomy=anatomy,
                context=context,
                tol=impulse_tol,
            )
            if not report.admitted:
                raise InterventionRefusal(
                    "R06",
                    "impulse_limit=True refused: " + report.summary(),
                    remedy=(
                        "integrate the finite-duration exposure, or shorten the "
                        "pulse relative to the system timescale"
                    ),
                    offending_object=self.name,
                )

        gen = None
        if Q_sqrt is not None:
            gen = torch.Generator(device=x0.device)
            gen.manual_seed(int(seed))  # type: ignore[arg-type]

        if impulse_limit:
            times, states = self._integrate_impulse(
                x0, drift, window, dt=dt, Q_sqrt=Q_sqrt, gen=gen,
                anatomy=anatomy, context=context,
            )
        else:
            times, states = self._integrate_finite(
                x0, drift, window, dt=dt, Q_sqrt=Q_sqrt, gen=gen,
                anatomy=anatomy, context=context,
            )

        return InterventionResult(
            times=times,
            states=states,
            dose=dose,
            thermal_history=self.thermal_history,
            impulse_report=report,
            ledger=self.ledger,
        )

    # -- internals ----------------------------------------------------------

    def _steps(self, window: ExposureWindow, dt: float) -> tuple[int, float]:
        n = max(1, int(round(window.duration / dt)))
        return n, window.duration / n

    def _integrate_finite(
        self,
        x0: Tensor,
        drift: DriftFn,
        window: ExposureWindow,
        *,
        dt: float,
        Q_sqrt: Tensor | None,
        gen: torch.Generator | None,
        anatomy: Any,
        context: Any,
        with_drive: bool = True,
    ) -> tuple[Tensor, Tensor]:
        n, h = self._steps(window, dt)
        x = x0.clone()
        out = [x.clone()]
        ts = [window.t0]
        for i in range(n):
            t = window.t0 + i * h
            dx = drift(x, t) * h
            if with_drive:
                g = self.gain(x, t, anatomy=anatomy, context=context)
                u = self.drive(t).to(x.dtype)
                dx = dx + (g @ u) * h
            if Q_sqrt is not None:
                dw = torch.randn(
                    Q_sqrt.shape[-1], generator=gen, device=x.device, dtype=x.dtype
                ) * math.sqrt(h)
                dx = dx + Q_sqrt.to(x.dtype) @ dw
            x = x + dx
            out.append(x.clone())
            ts.append(t + h)
        return (
            torch.tensor(ts, dtype=torch.float64),
            torch.stack(out),
        )

    def _integrate_impulse(
        self,
        x0: Tensor,
        drift: DriftFn,
        window: ExposureWindow,
        *,
        dt: float,
        Q_sqrt: Tensor | None,
        gen: torch.Generator | None,
        anatomy: Any,
        context: Any,
    ) -> tuple[Tensor, Tensor]:
        ui = self.drive_integral(window)
        x_plus = self.impulse_map(
            x0, ui, anatomy=anatomy, context=context, t0=window.t0
        )
        times, states = self._integrate_finite(
            x_plus, drift, window, dt=dt, Q_sqrt=Q_sqrt, gen=gen,
            anatomy=anatomy, context=context, with_drive=False,
        )
        # prepend X(t0^-) so the jump is visible in the trajectory
        return (
            torch.cat([times[:1], times]),
            torch.cat([x0.unsqueeze(0), states]),
        )

    def drive_batch(self, ts: Tensor) -> Tensor:
        """Vectorised :meth:`drive`. Override when the waveform is vectorisable."""
        return torch.stack([self.drive(float(t)).to(torch.float64) for t in ts])

    def drive_integral(self, window: ExposureWindow, n: int | None = None) -> Tensor:
        """:math:`\\int_{t_0}^{t_1} u_k(t)\\,\\mathrm dt`, trapezoid rule.

        The sample count is chosen to resolve the *pulse*, not the window. A
        quadrature that steps over a brief pulse would report a zero impulse
        and then "prove" that the impulse limit fails -- the same
        under-resolution trap guarded in :meth:`check_impulse_limit`.
        """
        if n is None:
            per = self.waveform.period
            step = (per / _QUAD_POINTS_PER_PULSE) if per > 0 else window.duration / 4096
            n = int(min(2_000_001, max(4097, round(window.duration / step) + 1)))
        ts = torch.linspace(window.t0, window.t1, n, dtype=torch.float64)
        return torch.trapezoid(self.drive_batch(ts), ts, dim=0)

    # -- the test that justifies the jump -----------------------------------

    def check_impulse_limit(
        self,
        x0: Tensor,
        drift: DriftFn,
        window: ExposureWindow,
        *,
        dt: float,
        anatomy: Any = None,
        context: Any = None,
        tol: float = 1e-2,
    ) -> ImpulseLimitReport:
        """Compare the jump against the finite-duration integration.

        Deterministic (noise off): the diffusion term is common to both
        trajectories and would only add variance to the comparison.

        Refuses when ``dt`` does not resolve the pulse.  An under-resolved
        finite-duration reference would *manufacture* agreement with the jump
        (both would miss the same drive), so the flag would be justified by the
        very approximation it is meant to test.
        """
        n, h = self._steps(window, dt)
        if self.waveform.period > 0 and self.waveform.period / h < _MIN_STEPS_PER_PULSE:
            raise InterventionRefusal(
                "R06",
                f"step dt={h:.3e} s resolves the {self.waveform.period:.3e} s "
                f"pulse with only {self.waveform.period / h:.2f} steps; the "
                "finite-duration reference for the impulse-limit test must be "
                f"resolved with at least {_MIN_STEPS_PER_PULSE} steps per pulse",
                remedy="reduce dt, or integrate the exposure window separately "
                "from the post-exposure relaxation",
                offending_object=self.name,
            )
        _, s_fin = self._integrate_finite(
            x0, drift, window, dt=dt, Q_sqrt=None, gen=None,
            anatomy=anatomy, context=context, with_drive=True,
        )
        _, s_imp = self._integrate_impulse(
            x0, drift, window, dt=dt, Q_sqrt=None, gen=None,
            anatomy=anatomy, context=context,
        )
        xf, xi = s_fin[-1], s_imp[-1]
        delta = (xf - x0)
        abs_err = float((xf - xi).norm())
        scale = float(delta.norm())
        rel = abs_err / scale if scale > 0 else (0.0 if abs_err == 0 else float("inf"))
        tau = self._system_timescale(drift, x0, window.t0)
        return ImpulseLimitReport(
            exposure_duration_s=window.duration,
            system_timescale_s=tau,
            rel_error=rel,
            abs_error=abs_err,
            tol=tol,
            admitted=rel <= tol,
            finite_duration_state=xf,
            impulse_state=xi,
            n_steps=n,
        )

    @staticmethod
    def _system_timescale(drift: DriftFn, x0: Tensor, t0: float) -> float:
        """Slowest relaxation time of the linearised drift at ``x0``."""
        x = x0.detach().to(torch.float64).requires_grad_(False)
        eps = 1e-6 * max(1.0, float(x.abs().max()))
        n = x.numel()
        J = torch.zeros(n, n, dtype=torch.float64)
        f0 = drift(x, t0).to(torch.float64)
        for i in range(n):
            xp = x.clone()
            xp[i] += eps
            J[:, i] = (drift(xp, t0).to(torch.float64) - f0) / eps
        ev = torch.linalg.eigvals(J)
        rates = ev.real.abs()
        rates = rates[rates > 1e-12]
        if rates.numel() == 0:
            return float("inf")
        return float(1.0 / rates.min())

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry": self.geometry.describe(),
            "waveform": self.waveform.name,
            "waveform_units": self.waveform.units,
            "coupling": self.coupling.name,
            "coupling_status": self.coupling.mechanistic_status,
            "mechanism_resolved": self.mechanistic_uncertainty.resolved,
            "mechanism_candidates": list(self.mechanistic_uncertainty.candidates),
            "thermal_cem43_min": self.thermal_history.cem43_minutes,
            "notice": self.notice,
        }


# ---------------------------------------------------------------------------
# a concrete minimal operator used by tests and by the impulse-limit machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PointGeometry(DeviceGeometry):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def element_positions(self) -> Tensor:
        return torch.tensor([self.position], dtype=torch.float64)


class LinearFieldIntervention(InterventionOperator):
    """Fixed spatial write pattern times a scalar waveform.

    The simplest operator satisfying the Sec. 2.4 form.  It exists so the SDE
    machinery, the impulse-limit test, and the safety guard can be exercised
    without dragging in a field solver, and so that field-solver stacks have a
    reference for what "writing into state" is allowed to mean.
    """

    def __init__(
        self,
        *,
        pattern: Tensor,
        waveform: WaveformSpec,
        name: str = "linear_field_intervention",
        geometry: DeviceGeometry | None = None,
        coupling: TissueCoupling | None = None,
        mechanistic_uncertainty: MechanisticUncertainty | None = None,
        state_gain: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            geometry=geometry or _PointGeometry(device_id="point", frame="device"),
            waveform=waveform,
            coupling=coupling
            or TissueCoupling(
                name="identity_linear",
                parameters={},
                mechanistic_status="surrogate",
                citation="none; test scaffold",
            ),
            mechanistic_uncertainty=mechanistic_uncertainty
            or MechanisticUncertainty(
                candidates=("identity_linear",),
                log_weights=torch.zeros(1, dtype=torch.float64),
                resolved=False,
                note="scaffold operator; no mechanistic claim",
            ),
        )
        self.pattern = pattern.reshape(-1, 1).to(torch.float64)
        self._state_gain = state_gain

    def gain(self, x, t, *, anatomy=None, context=None):  # noqa: D102
        g = self.pattern.to(x.dtype)
        if self._state_gain is not None:
            g = g * self._state_gain(x).to(x.dtype).reshape(-1, 1)
        return g

    def drive(self, t):  # noqa: D102
        phase = t - self.waveform_t0
        if phase < 0.0 or phase > self.waveform.period:
            return torch.zeros(1, dtype=torch.float64)
        return self.waveform.sample(torch.tensor(phase, dtype=torch.float64)).reshape(1)

    def drive_batch(self, ts):  # noqa: D102
        phase = ts.to(torch.float64) - self.waveform_t0
        v = self.waveform.sample(phase.clamp_min(0.0))
        return torch.where(
            (phase >= 0.0) & (phase <= self.waveform.period), v, torch.zeros_like(v)
        ).reshape(-1, 1)

    waveform_t0: float = 0.0

    def with_onset(self, t0: float) -> "LinearFieldIntervention":
        self.waveform_t0 = t0
        return self


def monophasic_waveform(period: float = 100e-6, amplitude: float = 1.0) -> WaveformSpec:
    """Half-sine monophasic pulse shape, normalised, unitless drive."""

    def f(t: Tensor) -> Tensor:
        return torch.where(
            (t >= 0) & (t <= period),
            torch.sin(math.pi * t / period),
            torch.zeros_like(t),
        )

    return WaveformSpec(
        name="monophasic_half_sine",
        units="dimensionless",
        period=period,
        sample_fn=f,
        amplitude=amplitude,
    )


def replace_ledger(obj: Any, ledger: Ledger) -> Any:
    """Convenience: replace the ledger on any frozen dataclass that has one."""
    return replace(obj, ledger=ledger)


def as_sequence(x: Sequence[float] | Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)
