"""fNIRS observation head: photon path, partial volume, and the scalp problem.

The dominant systematic error in continuous-wave fNIRS is **extracerebral
contamination**: most of the detected photons' path length lies in scalp and
skull, so systemic scalp haemodynamics (blood pressure waves, skin flow,
Mayer waves at ~0.1 Hz, task-evoked sympathetic responses) enter the channel
with a larger weight than the cortical signal.  Per body.tex Sec. 2.4 --- "fMRI
and fNIRS heads introduce neurovascular and metabolic states before sampling"
--- and per Sec. 2.7, that term is a *modelled* bias with a status, not an
omission.

Physics implemented here
------------------------
1. Modified Beer--Lambert law with **layer-resolved partial path lengths**::

       Delta OD(lambda) = sum_layer sum_chromophore
                          eps(lambda, c) * Delta C_layer(c) * L_layer(lambda)

   The two-layer partial path lengths come from the diffusion approximation for
   a semi-infinite two-layer medium as a function of source--detector
   separation and upper-layer thickness, which reproduces the two facts that
   matter: short separations see almost only scalp, and the cerebral partial
   path length is a small fraction of the total even at 3 cm.
2. Extinction coefficients for HbO/HbR at the standard 760/850 nm pair
   (Prahl compilation of Wray et al. 1988 / Zijlstra et al. 1991).
3. Concentration recovery by inverting the 2x2 extinction system, with the
   condition number of that system reported as a real variance source: the
   HbO/HbR separation is ill-conditioned at poorly chosen wavelength pairs.
4. Short-separation regression as an *explicit* estimator of the extracerebral
   term, which is what upgrades the contamination bias from
   ``prior_specified_sensitivity`` to ``design_estimable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .base import (
    DIMENSIONLESS,
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
    "EXTINCTION_COEFF",
    "OpticalProperties",
    "PhotonPathModel",
    "ExtracerebralModel",
    "FNIRSObservationOperator",
]


EXTINCTION_COEFF: dict[float, dict[str, float]] = {
    760.0: {"HbO": 1486.5865, "HbR": 3843.707},
    850.0: {"HbO": 2526.391, "HbR": 1798.643},
    690.0: {"HbO": 812.7, "HbR": 4383.0},
    830.0: {"HbO": 2321.4, "HbR": 1791.7},
}
"""Molar extinction coefficients, ``1/(cm * M)`` (natural-log convention x ln10).

Prahl's compilation of Wray et al. (1988) and Zijlstra et al. (1991), the same
table used by MNE-NIRS and Homer.  Values are per ``cm^-1 / (moles/litre)``.
"""


@dataclass(frozen=True)
class OpticalProperties:
    """Tissue absorption and reduced scattering at a wavelength.

    Defaults are adult-head literature values (Strangman et al. 2003; Custo et
    al. 2006); the priors record that they are population values, not this
    subject's.
    """

    mu_a_scalp: float = 0.017  # 1/mm
    mu_a_skull: float = 0.012
    mu_a_csf: float = 0.004
    mu_a_brain: float = 0.019
    mus_p_scalp: float = 0.85  # 1/mm reduced scattering
    mus_p_skull: float = 0.95
    mus_p_csf: float = 0.03
    mus_p_brain: float = 1.10

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "mu_a_scalp": Prior.lognormal_from_mean_cv(
                "mu_a_scalp", 0.017, 0.30, units="1/mm",
                source="Strangman et al. 2003 adult head optical properties",
            ),
            "mu_a_brain": Prior.lognormal_from_mean_cv(
                "mu_a_brain", 0.019, 0.30, units="1/mm",
                source="Strangman et al. 2003",
            ),
            "extracerebral_thickness": Prior.lognormal_from_mean_cv(
                "extracerebral_thickness", 12.0, 0.25, units="mm",
                source="scalp+skull+CSF thickness from subject MRI; population "
                "mean 10-14 mm with strong regional and individual variation",
                validity=(6.0, 22.0),
            ),
            "differential_pathlength_factor": Prior.lognormal_from_mean_cv(
                "differential_pathlength_factor", 6.0, 0.20,
                source="Duncan et al. 1995 age-dependent DPF; using a fixed DPF "
                "is a known amplitude bias",
                validity=(4.0, 8.0),
            ),
        }


@dataclass(frozen=True)
class PhotonPathModel:
    """Layer-resolved partial path lengths from the diffusion approximation.

    For a semi-infinite homogeneous medium the mean total path length between a
    source and a detector separated by ``rho`` is ``DPF * rho``.  Splitting it
    between an extracerebral slab of thickness ``d`` and the brain underneath
    uses the standard depth-sensitivity result that the cerebral fraction rises
    monotonically with ``rho`` and collapses to zero for ``rho`` below roughly
    ``2 d``: photons simply never reach the cortex at short separations.  That
    single fact is the design principle behind short-separation channels.
    """

    dpf: float = 6.0
    extracerebral_thickness_mm: float = 12.0
    max_cerebral_fraction: float = 0.45
    steepness_mm: float = 8.0

    def total_pathlength_mm(self, separation_mm: torch.Tensor) -> torch.Tensor:
        return self.dpf * separation_mm.to(torch.float64)

    def cerebral_fraction(self, separation_mm: torch.Tensor) -> torch.Tensor:
        """Fraction of the mean path length spent in brain tissue.

        **Exactly** zero below ``2 * thickness`` -- the diffuse penetration depth
        is about half the separation, so a channel shorter than twice the
        extracerebral layer cannot sample cortex at all -- then rising and
        saturating at ``max_cerebral_fraction``.  At ``rho = 30 mm`` with a 12 mm
        extracerebral layer this yields ~0.16, matching the widely reported
        result that only 15-25 % of the fNIRS signal at 3 cm is cerebral.
        """
        rho = separation_mm.to(torch.float64)
        onset = 2.0 * self.extracerebral_thickness_mm
        u = ((rho - onset) / self.steepness_mm).clamp_min(0.0)
        return self.max_cerebral_fraction * (u * u) / (1.0 + u * u)

    def partial_pathlengths_mm(
        self, separation_mm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(L_extracerebral, L_brain)`` in mm, per channel."""
        total = self.total_pathlength_mm(separation_mm)
        fb = self.cerebral_fraction(separation_mm)
        return total * (1.0 - fb), total * fb

    def as_psf(self, separation_mm: torch.Tensor, frame: str) -> PSF:
        """The banana-shaped photon sensitivity volume, not a point."""
        rho = separation_mm.to(torch.float64)
        depth = 0.5 * rho  # classic half-separation penetration depth
        return PSF(
            kind="photon_path",
            frame=frame,
            units="m",
            fwhm=(float(rho.mean()) * 1e-3, float(rho.mean()) * 0.5e-3, float(depth.mean()) * 1e-3),
            extent=(float(rho.max()) * 1e-3,),
            meta={
                "shape": "banana (diffuse reflectance sensitivity volume)",
                "penetration_depth_mm": [float(d) for d in depth],
                "cerebral_fraction": [float(f) for f in self.cerebral_fraction(rho)],
                "note": "an optode pair has no voxel; its support is a depth- and "
                "separation-dependent sensitivity volume dominated by scalp",
            },
        )


@dataclass(frozen=True)
class ExtracerebralModel:
    """Systemic scalp haemodynamics: the dominant fNIRS bias, modelled.

    Includes Mayer waves (~0.1 Hz), respiration, cardiac pulsation, and a
    task-locked systemic response.  The last one is why "the fNIRS response was
    time-locked to the task" is *not* evidence of a cortical origin.
    """

    mayer_hz: float = 0.10
    mayer_amplitude_uM: float = 0.9
    respiratory_hz: float = 0.25
    respiratory_amplitude_uM: float = 0.4
    cardiac_hz: float = 1.1
    cardiac_amplitude_uM: float = 0.3
    task_locked_amplitude_uM: float = 0.5
    drift_uM_per_s: float = 0.002

    def generate(
        self,
        n_channels: int,
        t: torch.Tensor,
        *,
        task: torch.Tensor | None = None,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        """Scalp ``[HbO], [HbR]`` in micromolar, ``(n_channels, n_time)``."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        t = t.to(torch.float64)
        n_t = t.numel()

        def osc(f: float, amp: float) -> torch.Tensor:
            ph = 2 * math.pi * torch.rand(n_channels, generator=g, dtype=torch.float64)
            gain = 1.0 + 0.2 * torch.randn(n_channels, generator=g, dtype=torch.float64)
            return (amp * gain).unsqueeze(-1) * torch.sin(
                2 * math.pi * f * t.unsqueeze(0) + ph.unsqueeze(-1)
            )

        hbo = (
            osc(self.mayer_hz, self.mayer_amplitude_uM)
            + osc(self.respiratory_hz, self.respiratory_amplitude_uM)
            + osc(self.cardiac_hz, self.cardiac_amplitude_uM)
        )
        hbo = hbo + self.drift_uM_per_s * t.unsqueeze(0) * torch.randn(
            (n_channels, 1), generator=g, dtype=torch.float64
        )
        if task is not None:
            gain = self.task_locked_amplitude_uM * (
                1.0 + 0.3 * torch.randn(n_channels, generator=g, dtype=torch.float64)
            )
            hbo = hbo + gain.unsqueeze(-1) * task.to(torch.float64).unsqueeze(0)[:, :n_t]
        # scalp HbR is a small anticorrelated fraction of HbO
        hbr = -0.25 * hbo
        return {"HbO": hbo, "HbR": hbr}


class FNIRSObservationOperator(ObservationOperator):
    """``O_fNIRS``: cortical HbO/HbR -> optical density change per channel.

    ``observe`` returns **optical density change** (dimensionless, the actual
    measurement) for every (channel, wavelength) pair.  :meth:`recover_hb`
    performs the standard MBLL inversion and reports how much of the answer is
    an artefact of the assumed path lengths --- which is the point.
    """

    name = "fnirs_observation_operator"
    version = "0.1.0"

    def __init__(
        self,
        separations_mm: torch.Tensor,
        *,
        wavelengths_nm: Sequence[float] = (760.0, 850.0),
        dt: float = 0.1,
        clock: str = "nirs_device",
        frame: str = "optode_montage",
        path_model: PhotonPathModel | None = None,
        extracerebral: ExtracerebralModel | None = None,
        optical: OpticalProperties | None = None,
        short_separation_mm: float = 8.0,
        instrument_noise_od: float = 2e-4,
        labels: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.separations_mm = separations_mm.to(torch.float64)
        self.wavelengths = tuple(float(w) for w in wavelengths_nm)
        for w in self.wavelengths:
            if w not in EXTINCTION_COEFF:
                raise ObservationRefusal(
                    code="R01",
                    message=f"no extinction coefficients tabulated for {w} nm",
                    remedy="add the wavelength to EXTINCTION_COEFF with its "
                    "literature source, or use a tabulated pair",
                )
        self.path_model = path_model or PhotonPathModel()
        self.extracerebral = extracerebral or ExtracerebralModel()
        self.optical = optical or OpticalProperties()
        self.short_separation_mm = float(short_separation_mm)
        self.instrument_noise_od = float(instrument_noise_od)
        self.frame = frame
        self.labels = tuple(labels or ())
        self.dtype = dtype
        self._temporal = TemporalSupport(
            clock=clock, dt=float(dt), integration_window=0.0, group_delay=0.0
        )

    # -- descriptors --------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return int(self.separations_mm.numel())

    @property
    def support(self) -> Support:
        return Support(
            kind="optode_channel",
            frame=self.frame,
            units=DIMENSIONLESS,
            psf=self.path_model.as_psf(self.separations_mm, self.frame),
            n_elements=self.n_channels,
            labels=self.labels or None,
        )

    @property
    def temporal(self) -> TemporalSupport:
        return self._temporal

    @property
    def units(self) -> str:
        return DIMENSIONLESS  # optical density change

    @property
    def nuisance_priors(self) -> dict[str, Prior]:
        return dict(OpticalProperties.priors())

    @property
    def short_channels(self) -> torch.Tensor:
        return self.separations_mm <= self.short_separation_mm

    # -- extinction system --------------------------------------------------
    def extinction_matrix(self) -> torch.Tensor:
        """``(n_wavelengths, 2)`` in ``1/(mm * uM)``.

        Table values are ``cm^-1/M``; converting to ``mm^-1/uM`` divides by
        ``10 * 1e6`` -- carried through explicitly so the unit round trip is
        checkable rather than a magic constant.
        """
        rows = []
        for w in self.wavelengths:
            e = EXTINCTION_COEFF[w]
            rows.append([e["HbO"], e["HbR"]])
        return torch.tensor(rows, dtype=torch.float64) / (10.0 * 1e6)

    def separation_condition_number(self) -> float:
        """Condition number of the HbO/HbR separation at these wavelengths."""
        return float(torch.linalg.cond(self.extinction_matrix()))

    # -- forward ------------------------------------------------------------
    def observe(
        self,
        cortical_hb: Mapping[str, torch.Tensor],
        latent_temporal: TemporalSupport,
        *,
        seed: int,
        n_samples: int | None = None,
        task: torch.Tensor | None = None,
        include_extracerebral: bool = True,
        include_noise: bool = True,
        channel_mixing: torch.Tensor | None = None,
    ) -> ObservationRead | Unresolved:
        """Forward-model optical density change.

        Parameters
        ----------
        cortical_hb
            ``{"HbO": (n_sources, n_t), "HbR": (n_sources, n_t)}`` in micromolar
            on the latent clock.
        channel_mixing
            ``(n_channels, n_sources)`` sensitivity of each channel to each
            cortical source.  Identity-like when sources are already channels.
        """
        if not {"HbO", "HbR"} <= set(cortical_hb):
            return Unresolved(
                reason="cortical haemoglobin must supply both HbO and HbR",
                missing=tuple(sorted({"HbO", "HbR"} - set(cortical_hb))),
            )
        hbo = cortical_hb["HbO"].to(torch.float64)
        hbr = cortical_hb["HbR"].to(torch.float64)
        if channel_mixing is not None:
            Mx = channel_mixing.to(torch.float64)
            hbo, hbr = Mx @ hbo, Mx @ hbr
        if hbo.shape[0] != self.n_channels:
            return Unresolved(
                reason=f"{hbo.shape[0]} cortical sources but {self.n_channels} "
                "channels and no channel_mixing",
                missing=("channel_mixing",),
            )

        x_c, idx = self.sample_latent_on_native_clock(
            torch.stack([hbo, hbr]).reshape(-1, hbo.shape[-1]),
            latent_temporal,
            n_samples,
        )
        n_t = int(x_c.shape[-1])
        if n_t == 0:
            return Unresolved(reason="latent too short for one NIRS sample")
        hbo_s, hbr_s = x_c.reshape(2, self.n_channels, n_t)

        L_ec, L_br = self.path_model.partial_pathlengths_mm(self.separations_mm)
        eps = self.extinction_matrix()  # (n_wl, 2) in 1/(mm uM)
        t = self._temporal.dt * torch.arange(n_t, dtype=torch.float64)

        scalp = {"HbO": torch.zeros_like(hbo_s), "HbR": torch.zeros_like(hbr_s)}
        if include_extracerebral:
            task_native = None
            if task is not None:
                tt, _ = self.sample_latent_on_native_clock(
                    task.reshape(1, -1).to(torch.float64), latent_temporal, n_t
                )
                task_native = tt[0]
            scalp = self.extracerebral.generate(
                self.n_channels, t, task=task_native, seed=seed + 31
            )

        n_wl = len(self.wavelengths)
        od = torch.zeros((self.n_channels, n_wl, n_t), dtype=torch.float64)
        od_brain = torch.zeros_like(od)
        od_scalp = torch.zeros_like(od)
        for w in range(n_wl):
            e_o, e_r = float(eps[w, 0]), float(eps[w, 1])
            od_brain[:, w] = L_br.unsqueeze(-1) * (e_o * hbo_s + e_r * hbr_s)
            od_scalp[:, w] = L_ec.unsqueeze(-1) * (
                e_o * scalp["HbO"] + e_r * scalp["HbR"]
            )
        od = od_brain + od_scalp

        components: dict[str, torch.Tensor] = {
            "cerebral": od_brain.to(self.dtype),
            "extracerebral": od_scalp.to(self.dtype),
        }
        var_meas = 0.0
        if include_noise:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            # detector shot noise grows with separation (less light returns)
            atten = torch.exp(0.06 * (self.separations_mm - 30.0)).clamp(0.2, 20.0)
            noise = (
                self.instrument_noise_od
                * atten.reshape(-1, 1, 1)
                * torch.randn((self.n_channels, n_wl, n_t), generator=g, dtype=torch.float64)
            )
            od = od + noise
            components["instrument"] = noise.to(self.dtype)
            var_meas = float(noise.var(dim=-1).mean())

        cf = self.path_model.cerebral_fraction(self.separations_mm)
        ledger = self._ledger(
            seed=seed,
            var_meas=var_meas,
            var_within=float(od_scalp.var(dim=-1).mean()),
            n_t=n_t,
            cerebral_fraction=cf,
            has_short=bool(self.short_channels.any()),
        )
        return ObservationRead(
            prediction=od.to(self.dtype),
            units=DIMENSIONLESS,
            support=self.support,
            temporal=self._temporal,
            ledger=ledger,
            components=components,
            residual_channels={
                "cerebral_fraction": cf,
                "partial_pathlength_extracerebral_mm": L_ec,
                "partial_pathlength_brain_mm": L_br,
                "latent_sample_index": idx,
            },
        )

    # -- inversion ----------------------------------------------------------
    def recover_hb(
        self,
        od: torch.Tensor,
        *,
        assumed_pathlength_mm: torch.Tensor | None = None,
        short_separation_regression: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Modified Beer--Lambert inversion, ``(n_channels, n_wl, n_t) -> HbO/HbR``.

        ``assumed_pathlength_mm`` is what the *analyst* assumes, and it is
        usually the total DPF path length rather than the cerebral partial path
        length.  Using the total is exactly the step that turns extracerebral
        contamination into an apparent cortical concentration change, so the
        default reproduces the standard (biased) pipeline and the ledger says so.
        """
        eps = self.extinction_matrix()
        Lp = (
            assumed_pathlength_mm
            if assumed_pathlength_mm is not None
            else self.path_model.total_pathlength_mm(self.separations_mm)
        ).to(torch.float64)
        A = eps.unsqueeze(0) * Lp.reshape(-1, 1, 1)  # (n_ch, n_wl, 2)
        out = torch.linalg.lstsq(A, od.to(torch.float64)).solution  # (n_ch, 2, n_t)
        hbo, hbr = out[:, 0], out[:, 1]

        if short_separation_regression:
            short = self.short_channels
            if not bool(short.any()):
                raise ObservationRefusal(
                    code="R08",
                    message="short-separation regression requested but the montage "
                    "contains no short channel; the extracerebral term would be "
                    "assigned a point estimate with no estimator",
                    remedy="add a short-separation channel (<= ~10 mm) or keep the "
                    "extracerebral bias as prior_specified_sensitivity",
                )
            def _regress_out(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
                num = (src * ref).sum(-1, keepdim=True)
                den = (ref * ref).sum(-1, keepdim=True).clamp_min(1e-30)
                return src - (num / den) * ref

            hbo = _regress_out(hbo, hbo[short].mean(0, keepdim=True))
            hbr = _regress_out(hbr, hbr[short].mean(0, keepdim=True))
        return {"HbO": hbo, "HbR": hbr}

    # -- ledger -------------------------------------------------------------
    def _ledger(
        self,
        *,
        seed: int,
        var_meas: float,
        var_within: float,
        n_t: int,
        cerebral_fraction: torch.Tensor,
        has_short: bool,
    ) -> UncertaintyLedger:
        cf_mean = float(cerebral_fraction.mean())
        contamination = 1.0 - cf_mean

        if has_short:
            extracerebral_bias = BiasTerm(
                name="extracerebral_contamination",
                interval=(-contamination * 0.25, contamination * 0.25),
                status="design_estimable",
                units=DIMENSIONLESS,
                estimator="short-separation channels in the same montage measure "
                "the extracerebral signal directly; the residual after regression "
                "is estimated from the short-channel cross-validation",
                note=f"mean cerebral path fraction {cf_mean:.3f}: most of the "
                "detected path is extracerebral even after regression",
            )
        else:
            extracerebral_bias = BiasTerm(
                name="extracerebral_contamination",
                interval=(-contamination, contamination),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-contamination, -contamination / 2, 0.0,
                                  contamination / 2, contamination),
                note="no short-separation channel: the systemic scalp component "
                "is not identified and must be swept, not subtracted",
            )

        bias = (
            extracerebral_bias,
            BiasTerm(
                name="partial_pathlength_assumption",
                interval=(-0.5, 0.5),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-0.5, -0.25, 0.0, 0.25, 0.5),
                note="the differential pathlength factor is a population value; "
                "concentration amplitudes inherit its error multiplicatively "
                "(Duncan et al. 1995)",
            ),
            BiasTerm(
                name="optode_coupling_and_hair",
                interval=(-0.15, 0.15),
                status="design_estimable",
                units=DIMENSIONLESS,
                estimator="per-channel signal quality index (SCI/PSP) measured at "
                "session start and end",
            ),
            BiasTerm(
                name="chromophore_separation_conditioning",
                interval=(
                    -0.02 * self.separation_condition_number(),
                    0.02 * self.separation_condition_number(),
                ),
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound="condition number of the tabulated extinction "
                "matrix at the chosen wavelength pair; crosstalk between HbO and "
                "HbR scales with it (Uludag et al. 2004)",
            ),
        )
        return UncertaintyLedger(
            variance=VarianceDecomposition(
                measurement=var_meas,
                within_session=var_within,
                between_session=UNKNOWN,
                parameter_posterior=UNKNOWN,
                model_class=UNKNOWN,
                numerical=0.0,
                units="dimensionless",
            ),
            bias=bias,
            model_discrepancy=UNKNOWN,
            model_discrepancy_flag=True,
            validity_domain={
                "units": "optical density change (dimensionless)",
                "wavelengths_nm": self.wavelengths,
                "clock": self._temporal.clock,
                "dt_s": self._temporal.dt,
                "n_samples": n_t,
                "separations_mm": [float(s) for s in self.separations_mm],
                "mean_cerebral_path_fraction": cf_mean,
                "has_short_separation_channel": has_short,
                "claim_boundary": "channel-space optical density; a cortical "
                "concentration claim additionally requires the photon-path model "
                "and an identified extracerebral estimator",
            },
            provenance=Provenance(
                operator=self.name,
                version=self.version,
                frames=(self.frame,),
                clocks=(self._temporal.clock,),
                inputs=("cortical_HbO_uM", "cortical_HbR_uM"),
                references=(
                    "Delpy et al. 1988 (MBLL)",
                    "Prahl / Wray et al. 1988 extinction table",
                    "Strangman et al. 2003 partial path lengths",
                    "body.tex Sec. 2.4 (fNIRS introduces neurovascular state "
                    "before sampling)",
                ),
                seed=seed,
            ),
            notes=(
                "extracerebral contamination is modelled, not ignored: it is the "
                "dominant systematic term for continuous-wave fNIRS.",
            ),
        )
