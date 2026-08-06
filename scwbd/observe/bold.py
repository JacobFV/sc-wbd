"""fMRI/BOLD observation head implementing thesis contract T3.

.. math::
    y_B[n] = M \\int_0^{T_h} h(s;\\rho)\\,x(n\\Delta_B - s)\\,\\mathrm ds
             + \\epsilon_B[n],\\qquad \\Delta_B = 1~\\mathrm s.

Two routes to ``h(s;rho)`` are provided and are *compared*, not assumed:

``CanonicalHRF``
    SPM double-gamma kernel.  ``rho`` = (peak delay, undershoot delay,
    dispersions, undershoot ratio) and every one of them is a subject/session
    nuisance **with a prior**, never a fixed constant.
``BalloonWindkesselReadout``
    The BOLD signal equation applied to the hemodynamic state ``(v, q)``.  Agent
    E owns the state; this module owns the readout and the sampling.
    :func:`hemodynamic_state_from_dynamics` is the adapter that consumes their
    output when it exists.

Everything else in this file is the part of fMRI that a "voxel time series"
abstraction silently discards: voxel point-spread plus vascular support, slice
timing, cardiac/respiratory physiology aliased by the TR, motion, scanner
drift, and partial-volume dilution.  Per thesis Sec. 7.1, "fMRI voxels need not
be assigned sensor-space electrical precision" -- the returned support says so
explicitly.

Units: BOLD is dimensionless fractional signal change; ``"%"`` is offered as an
alternative display unit with an exact round trip (tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Protocol, Sequence

import torch

from .base import (
    DIMENSIONLESS,
    PERCENT,
    UNKNOWN,
    BiasTerm,
    ObservationOperator,
    ObservationRead,
    ObservationRefusal,
    PSF,
    Prior,
    Provenance,
    Support,
    TemporalSupport,
    UncertaintyLedger,
    Unresolved,
    VarianceDecomposition,
)

__all__ = [
    "HRFParameters",
    "CanonicalHRF",
    "BalloonWindkesselParameters",
    "BalloonWindkesselReadout",
    "HemodynamicState",
    "hemodynamic_state_from_dynamics",
    "SliceTiming",
    "PhysiologicalNoise",
    "MotionModel",
    "DriftModel",
    "PartialVolume",
    "VoxelPSF",
    "BOLDObservationOperator",
    "percent_to_fraction",
    "fraction_to_percent",
]


def _causal_convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``(x * h)[:n]`` along the last axis, evaluated by FFT.

    The HRF kernel is tens of thousands of taps at a 1 ms latent step, so a
    direct ``conv1d`` would materialise an im2col buffer of ``n * n_h`` elements.
    The FFT route keeps the T3 integral on the latent grid without that cost and
    is asserted against ``scipy.signal.convolve`` in ``tests/observe/test_hrf.py``.
    """
    n = int(x.shape[-1])
    n_h = int(h.shape[-1])
    size = n + n_h - 1
    n_fft = 1 << (size - 1).bit_length()
    X = torch.fft.rfft(x.to(torch.float64), n=n_fft, dim=-1)
    H = torch.fft.rfft(h.to(torch.float64), n=n_fft, dim=-1)
    return torch.fft.irfft(X * H, n=n_fft, dim=-1)[..., :n]


def fraction_to_percent(x: torch.Tensor) -> torch.Tensor:
    """Dimensionless fractional signal change -> percent."""
    return x * 100.0


def percent_to_fraction(x: torch.Tensor) -> torch.Tensor:
    """Percent -> dimensionless fractional signal change."""
    return x / 100.0


# ==========================================================================
# rho: the haemodynamic response, as priors
# ==========================================================================


@dataclass(frozen=True)
class HRFParameters:
    """``rho`` for the canonical double-gamma HRF.

    SPM parameterisation: ``h(t) = G(t; a1, b1) - c G(t; a2, b2)`` with
    ``G`` the gamma density.  Defaults are the SPM canonical values; the priors
    encode the *measured* between-subject and between-region variability of the
    response (Handwerker et al. 2004 report peak-latency spreads of several
    seconds across subjects and regions), which is precisely why a fixed HRF is
    a bias source rather than a convenience.
    """

    peak_shape: float = 6.0
    peak_scale: float = 1.0
    undershoot_shape: float = 16.0
    undershoot_scale: float = 1.0
    undershoot_ratio: float = 1.0 / 6.0
    kernel_length_s: float = 32.0

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "peak_shape": Prior(
                "peak_shape",
                "lognormal",
                (math.log(6.0), 0.20),
                source="SPM canonical a1=6; Handwerker et al. 2004 between-subject "
                "and between-region HRF latency variability",
                validity=(3.0, 12.0),
            ),
            "peak_scale": Prior(
                "peak_scale",
                "lognormal",
                (math.log(1.0), 0.15),
                units="s",
                source="SPM canonical b1=1 s; dispersion varies with vasculature",
            ),
            "undershoot_shape": Prior(
                "undershoot_shape",
                "lognormal",
                (math.log(16.0), 0.20),
                source="SPM canonical a2=16",
            ),
            "undershoot_scale": Prior(
                "undershoot_scale", "lognormal", (math.log(1.0), 0.15), units="s"
            ),
            "undershoot_ratio": Prior(
                "undershoot_ratio",
                "lognormal",
                (math.log(1.0 / 6.0), 0.45),
                source="SPM canonical c=1/6; undershoot amplitude is strongly "
                "region- and subject-dependent",
                validity=(0.0, 0.6),
            ),
        }

    def sample(self, *, seed: int) -> "HRFParameters":
        pr = self.priors()
        vals = {k: float(p.sample((), seed=seed + i)) for i, (k, p) in enumerate(pr.items())}
        return replace(self, **vals)


def _gamma_pdf(t: torch.Tensor, shape: float, scale: float) -> torch.Tensor:
    t = t.clamp_min(0.0)
    log_pdf = (
        (shape - 1.0) * torch.log(t.clamp_min(1e-12))
        - t / scale
        - shape * math.log(scale)
        - math.lgamma(shape)
    )
    out = torch.exp(log_pdf)
    return torch.where(t > 0, out, torch.zeros_like(out))


@dataclass(frozen=True)
class CanonicalHRF:
    """``h(s; rho)`` sampled on an arbitrary (fine) time grid."""

    params: HRFParameters = HRFParameters()
    normalise: Literal["none", "unit_area", "unit_peak"] = "unit_area"

    def kernel(self, dt: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """``h(s)`` for ``s = 0, dt, ..., T_h``.  Length ``ceil(T_h/dt)+1``."""
        p = self.params
        n = int(round(p.kernel_length_s / dt)) + 1
        s = dt * torch.arange(n, dtype=dtype)
        h = _gamma_pdf(s, p.peak_shape, p.peak_scale) - p.undershoot_ratio * _gamma_pdf(
            s, p.undershoot_shape, p.undershoot_scale
        )
        if self.normalise == "unit_area":
            a = (h * dt).sum()
            if float(a.abs()) > 1e-30:
                h = h / a
        elif self.normalise == "unit_peak":
            h = h / h.abs().max().clamp_min(1e-30)
        return h

    @property
    def Th(self) -> float:
        return self.params.kernel_length_s


# ==========================================================================
# the Balloon-Windkessel route (agent E owns the state, we own the readout)
# ==========================================================================


class HemodynamicState(Protocol):
    """What this module needs from agent E's ``scwbd.dynamics.hemodynamics``.

    ``v`` normalised blood volume, ``q`` normalised deoxyhaemoglobin content,
    both shaped ``(n_regions, n_time)`` on the latent clock.
    """

    v: torch.Tensor
    q: torch.Tensor


@dataclass(frozen=True)
class _SimpleHemodynamicState:
    v: torch.Tensor
    q: torch.Tensor


def hemodynamic_state_from_dynamics(obj: Any) -> HemodynamicState:
    """Adapter accepting agent E's output in any of its plausible shapes."""
    if hasattr(obj, "v") and hasattr(obj, "q"):
        return obj  # type: ignore[return-value]
    if isinstance(obj, Mapping) and "v" in obj and "q" in obj:
        return _SimpleHemodynamicState(v=obj["v"], q=obj["q"])
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return _SimpleHemodynamicState(v=obj[0], q=obj[1])
    raise ObservationRefusal(
        code="R01",
        message="hemodynamic state does not expose normalised blood volume v "
        "and deoxyhaemoglobin q",
        remedy="pass scwbd.dynamics.hemodynamics output, a mapping with keys "
        "'v' and 'q', or a (v, q) pair",
        offending_object=type(obj).__name__,
    )


@dataclass(frozen=True)
class BalloonWindkesselParameters:
    """BOLD signal-equation parameters (Buxton 1998; Obata 2004; Stephan 2007).

    ``E0`` (resting oxygen extraction fraction), ``V0`` (resting venous volume
    fraction) and ``epsilon`` (intra/extravascular signal ratio) are subject and
    field-strength dependent nuisance, hence priors.
    """

    TE: float = 0.04
    V0: float = 0.02
    E0: float = 0.34
    epsilon: float = 1.43
    nu0: float = 40.3
    r0: float = 25.0

    @property
    def k1(self) -> float:
        return 4.3 * self.nu0 * self.E0 * self.TE

    @property
    def k2(self) -> float:
        return self.epsilon * self.r0 * self.E0 * self.TE

    @property
    def k3(self) -> float:
        return 1.0 - self.epsilon

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "E0": Prior(
                "E0",
                "lognormal",
                (math.log(0.34), 0.20),
                source="Stephan et al. 2007 resting oxygen extraction fraction",
                validity=(0.2, 0.55),
            ),
            "V0": Prior(
                "V0",
                "lognormal",
                (math.log(0.02), 0.30),
                source="Buxton et al. 1998 resting venous volume fraction",
                validity=(0.01, 0.06),
            ),
            "epsilon": Prior(
                "epsilon",
                "lognormal",
                (math.log(1.43), 0.25),
                source="Obata et al. 2004 intra/extravascular ratio; field- and "
                "sequence-dependent",
            ),
            "transit_time_tau": Prior(
                "transit_time_tau",
                "lognormal",
                (math.log(2.0), 0.30),
                units="s",
                source="Friston et al. 2000 mean transit time; owned by the "
                "hemodynamic state (agent E) but carried here for completeness",
            ),
            "grubb_alpha": Prior(
                "grubb_alpha",
                "lognormal",
                (math.log(0.32), 0.15),
                source="Grubb et al. 1974 volume/flow exponent",
            ),
        }


@dataclass(frozen=True)
class BalloonWindkesselReadout:
    """``y = V0 [k1 (1-q) + k2 (1 - q/v) + k3 (1 - v)]`` -- dimensionless."""

    params: BalloonWindkesselParameters = BalloonWindkesselParameters()

    def signal(self, state: HemodynamicState) -> torch.Tensor:
        p = self.params
        v = state.v.to(torch.float64).clamp_min(1e-6)
        q = state.q.to(torch.float64)
        return p.V0 * (
            p.k1 * (1.0 - q) + p.k2 * (1.0 - q / v) + p.k3 * (1.0 - v)
        )


def reference_balloon_windkessel(
    neural: torch.Tensor,
    dt: float,
    *,
    kappa: float = 0.65,
    gamma: float = 0.41,
    tau: float = 0.98,
    alpha: float = 0.32,
    E0: float = 0.34,
) -> _SimpleHemodynamicState:
    """Reference Balloon-Windkessel integrator, **for testing this module only**.

    Agent E owns hemodynamic state at runtime
    (``scwbd.dynamics.hemodynamics``).  This forward-Euler reference exists so
    that the observation side is testable in isolation and so that the readout
    equation can be checked against a known dynamical input.  It is not exported
    as a dynamics API and must not be used in production paths.
    """
    n_r, n_t = neural.shape
    s = torch.zeros(n_r, dtype=torch.float64)
    f = torch.ones(n_r, dtype=torch.float64)
    v = torch.ones(n_r, dtype=torch.float64)
    q = torch.ones(n_r, dtype=torch.float64)
    V = torch.empty((n_r, n_t), dtype=torch.float64)
    Q = torch.empty((n_r, n_t), dtype=torch.float64)
    z = neural.to(torch.float64)
    for t in range(n_t):
        fv = v.clamp_min(1e-6) ** (1.0 / alpha)
        ff = f.clamp_min(1e-6)
        E = 1.0 - (1.0 - E0) ** (1.0 / ff)
        ds = z[:, t] - kappa * s - gamma * (f - 1.0)
        df = s
        dv = (ff - fv) / tau
        dq = (ff * E / E0 - fv * q / v.clamp_min(1e-6)) / tau
        s = s + dt * ds
        f = f + dt * df
        v = v + dt * dv
        q = q + dt * dq
        V[:, t] = v
        Q[:, t] = q
    return _SimpleHemodynamicState(v=V, q=Q)


# ==========================================================================
# acquisition nuisance
# ==========================================================================


@dataclass(frozen=True)
class SliceTiming:
    """Per-element acquisition offset within the TR.

    Slice timing shifts the *sampling instant*, it does not interpolate the
    signal.  Correcting it by interpolation is a modelling choice with a bias;
    modelling it here means no correction is needed.
    """

    offsets_s: torch.Tensor  # (n_elements,)
    order: Literal["ascending", "descending", "interleaved", "multiband", "custom"] = "custom"
    tr: float = 1.0

    @staticmethod
    def interleaved(n_slices: int, tr: float, multiband: int = 1) -> "SliceTiming":
        n_eff = n_slices // multiband
        order = list(range(0, n_eff, 2)) + list(range(1, n_eff, 2))
        off = torch.zeros(n_slices, dtype=torch.float64)
        for k, sl in enumerate(order):
            for b in range(multiband):
                off[sl + b * n_eff] = tr * k / n_eff
        return SliceTiming(offsets_s=off, order="interleaved", tr=tr)

    @staticmethod
    def simultaneous(n_elements: int, tr: float = 1.0) -> "SliceTiming":
        return SliceTiming(
            offsets_s=torch.zeros(n_elements, dtype=torch.float64), order="custom", tr=tr
        )

    def bias_term(self) -> BiasTerm:
        span = float(self.offsets_s.max() - self.offsets_s.min()) if self.offsets_s.numel() else 0.0
        return BiasTerm(
            name="slice_timing_offset_span",
            interval=(-span / 2.0, span / 2.0),
            status="design_estimable",
            units="s",
            estimator="the acquisition's own slice-timing table; measured by the "
            "sequence, not inferred",
            note="modelled as a sampling-instant shift; no temporal interpolation "
            "of the latent is performed",
        )


@dataclass(frozen=True)
class PhysiologicalNoise:
    """Cardiac and respiratory contributions, generated in continuous time.

    RETROICOR-style Fourier expansion in physiological phase.  Because the
    process is synthesised on the *latent* clock and only then sampled at the
    BOLD instants, aliasing of the ~1 Hz cardiac rhythm into the low-frequency
    band emerges from the sampling rather than being added by hand.  That is the
    whole point of not resampling.
    """

    cardiac_hz: float = 1.05
    respiratory_hz: float = 0.28
    cardiac_amplitude: float = 0.004
    respiratory_amplitude: float = 0.006
    n_harmonics: int = 2
    rate_variability: float = 0.05

    def generate(
        self,
        n_elements: int,
        t: torch.Tensor,
        *,
        seed: int,
    ) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        t = t.to(torch.float64)
        out = torch.zeros((n_elements, t.numel()), dtype=torch.float64)
        for base, amp, tag in (
            (self.cardiac_hz, self.cardiac_amplitude, 0),
            (self.respiratory_hz, self.respiratory_amplitude, 1),
        ):
            # slow rate wander -> phase is an integral, not a fixed sinusoid
            wander = self.rate_variability * torch.cumsum(
                torch.randn(t.numel(), generator=g, dtype=torch.float64), 0
            )
            wander = wander / max(math.sqrt(t.numel()), 1.0)
            phase = 2 * math.pi * base * t + wander
            for m in range(1, self.n_harmonics + 1):
                a = amp / m * torch.randn(n_elements, generator=g, dtype=torch.float64)
                b = amp / m * torch.randn(n_elements, generator=g, dtype=torch.float64)
                out += a.unsqueeze(-1) * torch.cos(m * phase).unsqueeze(0)
                out += b.unsqueeze(-1) * torch.sin(m * phase).unsqueeze(0)
        return out


@dataclass(frozen=True)
class MotionModel:
    """Head motion: six rigid parameters plus their signal consequences."""

    translation_sd_m: float = 3e-4
    rotation_sd_rad: float = 3e-3
    spin_history_gain: float = 0.02
    susceptibility_gain: float = 0.03
    drift_rate_per_s: float = 2e-5

    def parameters(self, n: int, dt: float, *, seed: int) -> torch.Tensor:
        """``(6, n)`` motion trace: random walk + slow drift."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        w = torch.randn((6, n), generator=g, dtype=torch.float64)
        walk = torch.cumsum(w, dim=-1) * math.sqrt(dt)
        sd = torch.tensor(
            [self.translation_sd_m] * 3 + [self.rotation_sd_rad] * 3, dtype=torch.float64
        ).unsqueeze(-1)
        walk = walk / walk.std(dim=-1, keepdim=True).clamp_min(1e-30) * sd
        t = dt * torch.arange(n, dtype=torch.float64)
        return walk + self.drift_rate_per_s * t.unsqueeze(0) * sd

    def signal_effect(
        self, motion: torch.Tensor, sensitivity: torch.Tensor
    ) -> torch.Tensor:
        """Element-wise signal change from motion, ``(n_elements, n_time)``.

        ``sensitivity`` is ``(n_elements, 6)``: how strongly each element's
        signal responds to each motion parameter.  Elements at tissue boundaries
        and near susceptibility gradients are the sensitive ones -- which is why
        motion bias is *spatially structured* and never a global regressor.
        """
        direct = sensitivity.to(torch.float64) @ motion
        dm = torch.diff(motion, dim=-1, prepend=motion[:, :1])
        spin = self.spin_history_gain * (sensitivity.to(torch.float64) @ dm.abs())
        return direct + spin


@dataclass(frozen=True)
class DriftModel:
    """Scanner drift as a low-order discrete cosine basis on the BOLD clock."""

    n_terms: int = 3
    amplitude: float = 0.01

    def generate(self, n_elements: int, n_t: int, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        k = torch.arange(n_t, dtype=torch.float64)
        basis = torch.stack(
            [torch.cos(math.pi * (j + 1) * (2 * k + 1) / (2 * n_t)) for j in range(self.n_terms)]
        )
        coef = self.amplitude * torch.randn(
            (n_elements, self.n_terms), generator=g, dtype=torch.float64
        )
        return coef @ basis


@dataclass(frozen=True)
class PartialVolume:
    """Tissue mixing within an element: the dominant *bias* of parcel BOLD.

    ``gm_fraction`` is measured from the subject's own segmentation, which makes
    the resulting dilution bias ``design_estimable``.  ``wm_csf_signal_ratio``
    says how much of the non-grey signal still tracks the neural source (it is
    not zero: draining veins and partial CSF pulsation contribute).
    """

    gm_fraction: torch.Tensor
    wm_csf_signal_ratio: float = 0.15
    segmentation_sd: float = 0.05

    @property
    def gain(self) -> torch.Tensor:
        f = self.gm_fraction.to(torch.float64).clamp(0.0, 1.0)
        return f + (1.0 - f) * self.wm_csf_signal_ratio

    def bias_term(self) -> BiasTerm:
        g = self.gain
        dilution = float((1.0 - g).mean())
        return BiasTerm(
            name="partial_volume_dilution",
            interval=(-dilution - 3 * self.segmentation_sd, -dilution + 3 * self.segmentation_sd)
            if dilution > 0
            else (-3 * self.segmentation_sd, 3 * self.segmentation_sd),
            status="design_estimable",
            units=DIMENSIONLESS,
            estimator="subject tissue segmentation (grey-matter fraction per "
            "element) with its own reproducibility, propagated as the interval "
            "half-width",
            note="multiplicative dilution of fractional signal change; the sign "
            "is negative because mixing can only attenuate a grey-matter source",
        )


@dataclass(frozen=True)
class VoxelPSF:
    """Voxel point-spread **plus** vascular support.

    thesis Sec. 7.1: "fMRI voxels need not be assigned sensor-space electrical
    precision".  The effective spatial support of a BOLD sample is the voxel
    convolved with the haemodynamic point-spread and displaced toward the pial
    surface by draining-vein flow; the displacement is a *bias*, not blur.
    """

    voxel_size_m: tuple[float, float, float] = (0.003, 0.003, 0.003)
    hemodynamic_fwhm_m: float = 0.0035
    draining_vein_displacement_m: float = 0.002
    field_strength_t: float = 3.0
    sequence: str = "GE-EPI"

    @property
    def effective_fwhm_m(self) -> tuple[float, float, float]:
        h2 = self.hemodynamic_fwhm_m**2
        return tuple(math.sqrt(v**2 + h2) for v in self.voxel_size_m)  # type: ignore[return-value]

    def as_psf(self, frame: str) -> PSF:
        return PSF(
            kind="gaussian",
            frame=frame,
            units="m",
            fwhm=self.effective_fwhm_m,
            extent=self.voxel_size_m,
            meta={
                "hemodynamic_fwhm_m": self.hemodynamic_fwhm_m,
                "draining_vein_displacement_m": self.draining_vein_displacement_m,
                "field_strength_T": self.field_strength_t,
                "sequence": self.sequence,
                "note": "spatial support is voxel (*) haemodynamic PSF, displaced "
                "pial-ward; it is not a point and it is not electrically precise",
            },
        )

    def bias_term(self) -> BiasTerm:
        d = self.draining_vein_displacement_m
        return BiasTerm(
            name="vascular_displacement",
            interval=(0.0, d),
            status="externally_bounded",
            units="m",
            external_bound="GE-EPI draining-vein displacement measured against "
            "SE-EPI/VASO and cortical-depth profiles (Turner 2002; Polimeni et "
            "al. 2010): 1-4 mm pial-ward at 3 T",
            note="signed toward the pial surface; a symmetric blur model would "
            "hide it",
        )


# ==========================================================================
# the operator
# ==========================================================================


class BOLDObservationOperator(ObservationOperator):
    """``O_BOLD``: T3 with the acquisition physics that voxel time series omit.

    The latent is consumed on **its own** clock: the convolution integral is
    discretised on the latent grid and evaluated at the BOLD sampling instants
    ``n dt_B + slice_offset``.  The latent is never resampled to the TR, and the
    EEG head simultaneously reads the same trajectory at 1 ms.
    """

    name = "bold_observation_operator"
    version = "0.1.0"

    def __init__(
        self,
        n_elements: int,
        *,
        tr: float = 1.0,
        clock: str = "scanner_volume",
        frame: str = "subject_functional",
        support_kind: Literal["voxel", "parcel", "surface_vertex"] = "parcel",
        mixing: torch.Tensor | None = None,
        hrf: CanonicalHRF | None = None,
        balloon: BalloonWindkesselReadout | None = None,
        slice_timing: SliceTiming | None = None,
        physio: PhysiologicalNoise | None = None,
        motion: MotionModel | None = None,
        motion_sensitivity: torch.Tensor | None = None,
        drift: DriftModel | None = None,
        partial_volume: PartialVolume | None = None,
        voxel_psf: VoxelPSF | None = None,
        thermal_noise_sd: float = 0.005,
        slice_quantisation_tolerance: float = 0.02,
        labels: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if hrf is not None and balloon is not None:
            raise ObservationRefusal(
                code="R01",
                message="both a canonical HRF and a Balloon-Windkessel readout "
                "were supplied; the two are competing model classes and must be "
                "run separately so their disagreement is measurable",
                remedy="construct one operator per route and record the "
                "model_class variance from their disagreement",
            )
        self.n_elements = int(n_elements)
        self.mixing = mixing  # M in T3: (n_elements, n_sources)
        self.hrf = hrf if (hrf is not None or balloon is not None) else CanonicalHRF()
        self.balloon = balloon
        self.slice_timing = slice_timing or SliceTiming.simultaneous(n_elements, tr)
        self.physio = physio
        self.motion = motion
        self.motion_sensitivity = motion_sensitivity
        self.drift = drift
        self.partial_volume = partial_volume
        self.voxel_psf = voxel_psf or VoxelPSF()
        self.thermal_noise_sd = float(thermal_noise_sd)
        self.slice_quantisation_tolerance = float(slice_quantisation_tolerance)
        self._last_slice_residual_s = 0.0
        self.frame = frame
        self.support_kind = support_kind
        self.labels = tuple(labels or ())
        self.dtype = dtype
        self._temporal = TemporalSupport(
            clock=clock,
            dt=float(tr),
            integration_window=float(self.hrf.Th if self.hrf is not None else 32.0),
            group_delay=0.0,
            jitter_sd=0.0,
        )

    # -- descriptors --------------------------------------------------------
    @property
    def support(self) -> Support:
        return Support(
            kind=self.support_kind,
            frame=self.frame,
            units=DIMENSIONLESS,
            psf=self.voxel_psf.as_psf(self.frame),
            extent=self.voxel_psf.voxel_size_m,
            n_elements=self.n_elements,
            labels=self.labels or None,
        )

    @property
    def temporal(self) -> TemporalSupport:
        return self._temporal

    @property
    def units(self) -> str:
        return DIMENSIONLESS

    @property
    def nuisance_priors(self) -> dict[str, Prior]:
        p = dict(HRFParameters.priors())
        p.update(BalloonWindkesselParameters.priors())
        p["baseline_signal"] = Prior(
            "baseline_signal",
            "lognormal",
            (0.0, 0.15),
            source="session/coil-loading dependent receive gain; the denominator "
            "of percent signal change is itself uncertain",
        )
        return p

    # -- T3 -----------------------------------------------------------------
    def convolve_native(
        self,
        latent: torch.Tensor,
        latent_temporal: TemporalSupport,
        *,
        n_samples: int | None = None,
        hrf: CanonicalHRF | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ``int_0^{Th} h(s;rho) x(t - s) ds`` at the BOLD instants.

        The integral is a Riemann sum on the **latent** grid; the result is then
        read at ``t_n = n dt_B + slice_offset_e``.  When the slice offset is an
        integer number of latent steps this involves no interpolation at all;
        otherwise the operator refuses rather than interpolating silently.

        Returns ``(y, t_n)`` with ``y`` shaped ``(n_sources, n_bold)``.
        """
        h_obj = hrf or self.hrf
        if h_obj is None:
            raise ObservationRefusal(
                code="R01",
                message="convolve_native called on a Balloon-Windkessel operator",
                remedy="use observe_hemodynamic_state() for the balloon route",
            )
        dt_l = latent_temporal.dt
        x = latent.to(torch.float64)
        h = h_obj.kernel(dt_l)
        # causal convolution on the latent clock
        conv = _causal_convolve(x, h) * dt_l

        ratio = self._temporal.dt / dt_l
        k = round(ratio)
        if abs(ratio - k) > 1e-9 * max(1.0, ratio):
            raise ObservationRefusal(
                code="R01",
                message=f"TR {self._temporal.dt}s is not an integer multiple of "
                f"the latent dt {dt_l}s",
                remedy="declare a clock-graph edge with its own uncertainty",
            )
        n_lat = conv.shape[-1]
        n_max = 1 + (n_lat - 1) // k
        n_b = n_max if n_samples is None else min(n_samples, n_max)
        idx = k * torch.arange(n_b)
        t_n = self._temporal.dt * torch.arange(n_b, dtype=torch.float64)
        return conv.index_select(-1, idx), t_n

    def _sample_with_slice_timing(
        self,
        conv_full: torch.Tensor,
        latent_temporal: TemporalSupport,
        n_b: int,
    ) -> torch.Tensor:
        """Sample ``(n_elements, n_b)`` from a latent-grid signal at ``n TR + o_e``.

        The slice offset selects the **nearest latent sample**; it never
        interpolates.  The residual between the nominal offset and the selected
        instant is a real timing error, so it is measured and returned by
        :meth:`_slice_quantisation_error` for the ledger.  When the latent clock
        is too coarse to represent the acquisition's slice order at all -- a
        residual above ``slice_quantisation_tolerance`` of the TR -- the operator
        refuses rather than pretending the timing was honoured.
        """
        dt_l = latent_temporal.dt
        k = round(self._temporal.dt / dt_l)
        n_lat = conv_full.shape[-1]
        out = torch.zeros((conv_full.shape[0], n_b), dtype=torch.float64)
        offs = self.slice_timing.offsets_s.to(torch.float64)
        if offs.numel() == 1:
            offs = offs.expand(conv_full.shape[0])
        tol = self.slice_quantisation_tolerance * self._temporal.dt
        worst = 0.0
        for e in range(conv_full.shape[0]):
            o = float(offs[e]) if e < offs.numel() else 0.0
            j = int(round(o / dt_l))
            residual = abs(o - j * dt_l)
            worst = max(worst, residual)
            if residual > tol:
                raise ObservationRefusal(
                    code="R01",
                    message=f"slice offset {o}s for element {e} cannot be "
                    f"represented on the latent clock ({dt_l}s): the nearest "
                    f"latent sample is {residual:.4g}s away, more than "
                    f"{100 * self.slice_quantisation_tolerance:.1f}% of the "
                    f"{self._temporal.dt}s TR. Interpolating the latent to hide "
                    "this is forbidden",
                    remedy="refine the latent clock, or declare an explicit "
                    "interpolation operator with its own bias term",
                )
            idx = (j + k * torch.arange(n_b)).clamp(max=n_lat - 1)
            out[e] = conv_full[e].index_select(-1, idx)
        self._last_slice_residual_s = worst
        return out

    def _slice_quantisation_bias(self) -> BiasTerm:
        r = getattr(self, "_last_slice_residual_s", 0.0)
        return BiasTerm(
            name="slice_timing_grid_quantisation",
            interval=(-r, r),
            status="design_estimable",
            units="s",
            estimator="difference between the sequence's nominal slice offsets "
            "and the nearest latent sample instant, computed per element",
            note="sampling is by selection on the latent grid; this residual is "
            "the price of not interpolating, and it is reported rather than hidden",
        )

    # -- the read -----------------------------------------------------------
    def observe(
        self,
        latent: torch.Tensor,
        latent_temporal: TemporalSupport,
        *,
        seed: int,
        n_samples: int | None = None,
        units: Literal["dimensionless", "%"] = "dimensionless",
        include_noise: bool = True,
        hrf: CanonicalHRF | None = None,
        n_rho_draws: int = 0,
    ) -> ObservationRead | Unresolved:
        """Observe a neural latent through the haemodynamic route.

        ``latent`` is ``(n_sources, n_latent_samples)`` neural drive
        (dimensionless normalised activity).  ``M`` (``mixing``) maps sources to
        acquisition elements; when absent the two must already match.
        """
        x = latent.to(torch.float64)
        h_obj = hrf or self.hrf
        if h_obj is None:
            return Unresolved(
                reason="this operator uses the Balloon-Windkessel route; call "
                "observe_hemodynamic_state()",
                missing=("hemodynamic_state",),
            )

        dt_l = latent_temporal.dt
        h = h_obj.kernel(dt_l)
        conv = _causal_convolve(x, h) * dt_l

        M = self.mixing
        if M is None:
            if conv.shape[0] != self.n_elements:
                return Unresolved(
                    reason=f"latent has {conv.shape[0]} sources but the operator "
                    f"has {self.n_elements} elements and no mixing matrix M",
                    missing=("mixing",),
                )
            mixed = conv
        else:
            mixed = M.to(torch.float64) @ conv

        k = round(self._temporal.dt / dt_l)
        n_lat = mixed.shape[-1]
        n_max = 1 + (n_lat - 1) // k
        n_b = n_max if n_samples is None else min(n_samples, n_max)
        if n_b <= 0:
            return Unresolved(reason="latent too short for one BOLD volume")

        y = self._sample_with_slice_timing(mixed, latent_temporal, n_b)
        components: dict[str, torch.Tensor] = {"neurovascular": y.clone().to(self.dtype)}

        gain = None
        if self.partial_volume is not None:
            gain = self.partial_volume.gain
            y = gain.unsqueeze(-1) * y
            components["after_partial_volume"] = y.clone().to(self.dtype)

        var_meas = 0.0
        var_within = 0.0
        if include_noise:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            thermal = self.thermal_noise_sd * torch.randn(
                (self.n_elements, n_b), generator=g, dtype=torch.float64
            )
            y = y + thermal
            components["thermal"] = thermal.to(self.dtype)
            var_meas += float(thermal.var(dim=-1).mean())

            if self.physio is not None:
                # generated on the LATENT clock, then sampled -> aliasing is real
                t_lat = dt_l * torch.arange(n_lat, dtype=torch.float64)
                ph_full = self.physio.generate(self.n_elements, t_lat, seed=seed + 7)
                ph = self._sample_with_slice_timing(ph_full, latent_temporal, n_b)
                y = y + ph
                components["physiological"] = ph.to(self.dtype)
                var_within += float(ph.var(dim=-1).mean())

            if self.motion is not None:
                mp = self.motion.parameters(n_b, self._temporal.dt, seed=seed + 8)
                sens = self.motion_sensitivity
                if sens is None:
                    gg = torch.Generator(device="cpu").manual_seed(int(seed) + 9)
                    sens = 0.05 * torch.randn(
                        (self.n_elements, 6), generator=gg, dtype=torch.float64
                    )
                mo = self.motion.signal_effect(mp, sens)
                y = y + mo
                components["motion"] = mo.to(self.dtype)
                components["motion_parameters"] = mp.to(self.dtype)
                var_within += float(mo.var(dim=-1).mean())

            if self.drift is not None:
                dr = self.drift.generate(self.n_elements, n_b, seed=seed + 10)
                y = y + dr
                components["drift"] = dr.to(self.dtype)
                var_within += float(dr.var(dim=-1).mean())

        var_param: float | str = UNKNOWN
        if n_rho_draws > 0:
            acc = []
            for j in range(n_rho_draws):
                rho = h_obj.params.sample(seed=seed + 500 + 97 * j)
                yy, _ = self.convolve_native(
                    latent, latent_temporal, n_samples=n_b, hrf=CanonicalHRF(rho, h_obj.normalise)
                )
                acc.append(yy if M is None else M.to(torch.float64) @ yy)
            var_param = float(torch.stack(acc).var(dim=0, unbiased=True).mean())

        out_units = DIMENSIONLESS
        if units == "%":
            y = fraction_to_percent(y)
            components = {kk: fraction_to_percent(vv) for kk, vv in components.items()}
            out_units = PERCENT

        ledger = self._ledger(
            seed=seed,
            var_meas=var_meas,
            var_within=var_within,
            var_param=var_param,
            n_b=n_b,
            units=out_units,
            route="canonical_hrf",
        )
        return ObservationRead(
            prediction=y.to(self.dtype),
            units=out_units,
            support=self.support,
            temporal=self._temporal,
            ledger=ledger,
            components=components,
        )

    def observe_hemodynamic_state(
        self,
        state: Any,
        latent_temporal: TemporalSupport,
        *,
        seed: int,
        n_samples: int | None = None,
        units: Literal["dimensionless", "%"] = "dimensionless",
        include_noise: bool = True,
    ) -> ObservationRead | Unresolved:
        """Balloon-Windkessel route: read agent E's ``(v, q)`` at the BOLD clock."""
        if self.balloon is None:
            return Unresolved(
                reason="operator was not constructed with a BalloonWindkesselReadout",
                missing=("balloon",),
            )
        st = hemodynamic_state_from_dynamics(state)
        sig = self.balloon.signal(st)
        if self.mixing is not None:
            sig = self.mixing.to(torch.float64) @ sig

        dt_l = latent_temporal.dt
        k = round(self._temporal.dt / dt_l)
        n_lat = sig.shape[-1]
        n_max = 1 + (n_lat - 1) // k
        n_b = n_max if n_samples is None else min(n_samples, n_max)
        y = self._sample_with_slice_timing(sig, latent_temporal, n_b)
        components: dict[str, torch.Tensor] = {"balloon_windkessel": y.clone().to(self.dtype)}

        if self.partial_volume is not None:
            y = self.partial_volume.gain.unsqueeze(-1) * y

        var_meas = 0.0
        if include_noise:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            thermal = self.thermal_noise_sd * torch.randn(
                (y.shape[0], n_b), generator=g, dtype=torch.float64
            )
            y = y + thermal
            components["thermal"] = thermal.to(self.dtype)
            var_meas = float(thermal.var(dim=-1).mean())

        out_units = DIMENSIONLESS
        if units == "%":
            y = fraction_to_percent(y)
            out_units = PERCENT

        ledger = self._ledger(
            seed=seed,
            var_meas=var_meas,
            var_within=UNKNOWN if not include_noise else 0.0,
            var_param=UNKNOWN,
            n_b=n_b,
            units=out_units,
            route="balloon_windkessel",
        )
        return ObservationRead(
            prediction=y.to(self.dtype),
            units=out_units,
            support=self.support,
            temporal=self._temporal,
            ledger=ledger,
            components=components,
        )

    # -- ledger -------------------------------------------------------------
    def _ledger(
        self,
        *,
        seed: int,
        var_meas: float | str,
        var_within: float | str,
        var_param: float | str,
        n_b: int,
        units: str,
        route: str,
    ) -> UncertaintyLedger:
        bias: list[BiasTerm] = [
            self.slice_timing.bias_term(),
            self._slice_quantisation_bias(),
            self.voxel_psf.bias_term(),
            BiasTerm(
                name="neurovascular_coupling_model",
                interval=(-0.30, 0.30),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-0.30, -0.15, 0.0, 0.15, 0.30),
                note="the map from neural activity to flow is not measured in the "
                "same session; sweep it, do not assert it",
            ),
            BiasTerm(
                name="baseline_cbf_state",
                interval=(-0.20, 0.20),
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound="breath-hold or hypercapnia CVR calibration scan "
                "(appendix tab:appendix-calibration-sources): bounds the "
                "vascular-reactivity scaling that otherwise masquerades as a "
                "neural amplitude difference",
            ),
        ]
        if self.partial_volume is not None:
            bias.append(self.partial_volume.bias_term())
        if self.motion is not None:
            bias.append(
                BiasTerm(
                    name="motion_residual_after_regression",
                    interval=(-0.02, 0.02),
                    status="design_estimable",
                    units=DIMENSIONLESS,
                    estimator="framewise-displacement-stratified split-half "
                    "reliability within the same session",
                )
            )

        return UncertaintyLedger(
            variance=VarianceDecomposition(
                measurement=var_meas,
                within_session=var_within,
                between_session=UNKNOWN,
                parameter_posterior=var_param,
                model_class=UNKNOWN,
                numerical=0.0,
                units="dimensionless" if units == DIMENSIONLESS else "%^2",
            ),
            bias=tuple(bias),
            model_discrepancy=UNKNOWN,
            model_discrepancy_flag=True,
            validity_domain={
                "units": units,
                "clock": self._temporal.clock,
                "tr_s": self._temporal.dt,
                "integration_window_s": self._temporal.integration_window,
                "route": route,
                "n_volumes": n_b,
                "spatial_support": self.support_kind,
                "psf_fwhm_m": self.voxel_psf.effective_fwhm_m,
                "claim_boundary": "haemodynamic observable; not a measure of "
                "neuronal firing and not electrically precise (thesis Sec. 7.1)",
            },
            provenance=Provenance(
                operator=self.name,
                version=self.version,
                frames=(self.frame,),
                clocks=(self._temporal.clock,),
                inputs=("neural_latent[dimensionless]",),
                references=(
                    "thesis_contract.tex T3",
                    "body.tex tab:modalities (fMRI row)",
                    "Buxton et al. 1998; Friston et al. 2000; Stephan et al. 2007",
                ),
                seed=seed,
            ),
            notes=(
                "model_class variance is 'unknown' until the canonical-HRF and "
                "Balloon-Windkessel routes have both been run on this session; "
                "their disagreement is the estimator.",
            ),
        )
