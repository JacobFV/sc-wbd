"""TMS--EEG artifact model: keep physical dose, instrument, and periphery apart.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Appendix A (``tab:appendix-calibration-sources``, row *TMS--EEG artifact
phantoms, coil probes and manufacturer waveform data*) requires that the
recorded post-pulse EEG be decomposable into components that are **not** a
cortical response:

``pulse``
    the induced-field transient itself -- physical dose coupling directly into
    the electrodes;
``saturation``
    amplifier clipping and its recovery -- an *instrument* state, with a
    recovery time constant and a blanking window;
``auditory``
    the coil click, an N100/P200-shaped evoked response to a **sound**;
``somatosensory``
    scalp/muscle co-stimulation, a response to a **touch**;
``decay``
    slow electrode/electrode-gel polarisation drift.

Only what remains after these are accounted for is even a *candidate* cortical
response, and :meth:`TMSEEGArtifactModel.separate` refuses to label the
residual as cortical.  A sham condition rarely matches all peripheral effects,
so the components are modelled explicitly rather than assumed subtracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, InterventionRefusal, Ledger

__all__ = [
    "AmplifierSpec",
    "EvokedTemplate",
    "ArtifactComponents",
    "TMSEEGArtifactModel",
]

_DT = torch.float64


@dataclass(frozen=True)
class AmplifierSpec:
    """Instrument, not brain. Saturation and recovery are device properties."""

    range_uv: float = 5000.0  # clipping level
    recovery_tau_s: float = 5e-3
    blanking_s: float = 2e-3  # samples the amplifier reports as invalid
    saturation_gain_uv_per_v_per_m: float = 400.0
    sample_rate_hz: float = 5000.0
    device_id: str = "generic_tms_compatible_amplifier"
    citation: str = "TMS-EEG artifact phantom / manufacturer recovery specification"
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class EvokedTemplate:
    """A peripherally evoked template: a sum of Gaussian components in time."""

    name: str
    latencies_s: tuple[float, ...]
    amplitudes_uv: tuple[float, ...]
    widths_s: tuple[float, ...]
    modality: str  # "auditory" | "somatosensory"
    citation: str = ""

    def evaluate(self, t: Tensor) -> Tensor:
        t = t.to(_DT)
        out = torch.zeros_like(t)
        for lat, amp, w in zip(self.latencies_s, self.amplitudes_uv, self.widths_s):
            out = out + amp * torch.exp(-0.5 * ((t - lat) / w) ** 2)
        return out

    @classmethod
    def auditory_click(cls, scale: float = 1.0) -> "EvokedTemplate":
        """Auditory evoked potential to the coil click: N100 then P200."""
        return cls(
            name="auditory_click_aep",
            latencies_s=(0.100, 0.180),
            amplitudes_uv=(-6.0 * scale, 4.0 * scale),
            widths_s=(0.030, 0.045),
            modality="auditory",
            citation="TMS click AEP; masking rarely eliminates it entirely",
        )

    @classmethod
    def somatosensory(cls, scale: float = 1.0) -> "EvokedTemplate":
        """Scalp/muscle co-stimulation: early muscle burst plus a later SEP."""
        return cls(
            name="scalp_somatosensory",
            latencies_s=(0.012, 0.045, 0.120),
            amplitudes_uv=(25.0 * scale, -8.0 * scale, 3.0 * scale),
            widths_s=(0.004, 0.012, 0.035),
            modality="somatosensory",
            citation="scalp muscle and cutaneous co-stimulation templates",
        )


@dataclass(frozen=True)
class ArtifactComponents:
    """The decomposition. ``residual`` is *not* named a cortical response."""

    times_s: Tensor
    pulse: Tensor
    saturation: Tensor
    auditory: Tensor
    somatosensory: Tensor
    decay: Tensor
    recorded: Tensor
    residual: Tensor
    valid_mask: Tensor  # False inside the blanking window
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    def peripheral(self) -> Tensor:
        return self.auditory + self.somatosensory

    def instrumental(self) -> Tensor:
        return self.pulse + self.saturation + self.decay

    def as_cortical_response(self):  # pragma: no cover - refusal
        raise InterventionRefusal(
            "R04",
            "the artifact residual is not a cortical response. Physical dose, "
            "instrument saturation and peripheral co-stimulation have been "
            "removed under a model; what remains is an unexplained residual "
            "that still requires artifact-injection recovery and negative "
            "controls before any cortical claim (Appendix A).",
            remedy="run injection recovery + negative control, then compare "
            "against candidate response operators explicitly",
            offending_object="ArtifactComponents.residual",
        )

    def component_energy(self) -> dict[str, float]:
        m = self.valid_mask
        def e(x: Tensor) -> float:
            return float((x[m] ** 2).sum())
        return {
            "pulse": e(self.pulse),
            "saturation": e(self.saturation),
            "auditory": e(self.auditory),
            "somatosensory": e(self.somatosensory),
            "decay": e(self.decay),
            "residual": e(self.residual),
        }


class TMSEEGArtifactModel:
    """Generate and separate TMS--EEG artifact components.

    ``simulate`` builds a recorded trace from *named* physical causes;
    ``separate`` recovers those components by least squares against the same
    basis.  The round trip is the test: if injected components cannot be
    recovered, no cortical claim from this recording is admissible.
    """

    def __init__(
        self,
        amplifier: AmplifierSpec | None = None,
        auditory: EvokedTemplate | None = None,
        somatosensory: EvokedTemplate | None = None,
        *,
        decay_tau_s: float = 0.20,
    ) -> None:
        self.amplifier = amplifier or AmplifierSpec()
        self.auditory = auditory or EvokedTemplate.auditory_click()
        self.somatosensory = somatosensory or EvokedTemplate.somatosensory()
        self.decay_tau_s = decay_tau_s

    # -- basis ---------------------------------------------------------------

    def times(self, t_end: float = 0.5, t_start: float = -0.05) -> Tensor:
        n = int(round((t_end - t_start) * self.amplifier.sample_rate_hz)) + 1
        return torch.linspace(t_start, t_end, n, dtype=_DT)

    def _pulse_component(self, t: Tensor, e_field_v_per_m: float) -> Tensor:
        """Direct induced transient at the electrode: sub-millisecond, huge."""
        w = 1.0 / self.amplifier.sample_rate_hz
        amp = self.amplifier.saturation_gain_uv_per_v_per_m * e_field_v_per_m
        return amp * torch.exp(-0.5 * (t / (0.5 * w)) ** 2) * (t >= 0)

    def _saturation_component(self, t: Tensor, e_field_v_per_m: float) -> Tensor:
        """Exponential amplifier recovery after the pulse. Instrument state."""
        amp = 0.15 * self.amplifier.saturation_gain_uv_per_v_per_m * e_field_v_per_m
        return torch.where(
            t >= 0,
            amp * torch.exp(-t / self.amplifier.recovery_tau_s),
            torch.zeros_like(t),
        )

    def _decay_component(self, t: Tensor, amplitude_uv: float) -> Tensor:
        return torch.where(
            t >= 0,
            amplitude_uv * (1.0 - torch.exp(-t / self.decay_tau_s)),
            torch.zeros_like(t),
        )

    def basis(self, t: Tensor, e_field_v_per_m: float) -> tuple[Tensor, tuple[str, ...]]:
        cols = [
            self._pulse_component(t, e_field_v_per_m),
            self._saturation_component(t, e_field_v_per_m),
            self.auditory.evaluate(t),
            self.somatosensory.evaluate(t),
            self._decay_component(t, 1.0),
        ]
        return torch.stack(cols, dim=-1), (
            "pulse", "saturation", "auditory", "somatosensory", "decay"
        )

    def valid_mask(self, t: Tensor, recorded: Tensor | None = None) -> Tensor:
        """Samples the amplifier can be believed.

        Two exclusions, both *instrumental*: the declared blanking window, and
        any sample sitting at the clipping rail.  A railed sample carries no
        information about the input, so fitting to it would let the clipping
        non-linearity leak into whatever is called a cortical residual.
        """
        m = ~((t >= 0.0) & (t < self.amplifier.blanking_s))
        if recorded is not None:
            m = m & (recorded.abs() < 0.999 * self.amplifier.range_uv)
        return m

    # -- forward -------------------------------------------------------------

    def simulate(
        self,
        t: Tensor,
        *,
        e_field_v_per_m: float,
        cortical_response: Tensor | None = None,
        decay_uv: float = 5.0,
        noise_uv: float = 0.0,
        seed: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Build a recorded trace from named causes. Returns ``(trace, parts)``."""
        B, names = self.basis(t, e_field_v_per_m)
        gains = torch.tensor([1.0, 1.0, 1.0, 1.0, decay_uv], dtype=_DT)
        parts = {n: B[:, i] * gains[i] for i, n in enumerate(names)}
        y = sum(parts.values())
        if cortical_response is not None:
            parts["cortical_injected"] = cortical_response.to(_DT)
            y = y + cortical_response.to(_DT)
        if noise_uv > 0:
            g = torch.Generator().manual_seed(int(seed))
            y = y + noise_uv * torch.randn(t.numel(), generator=g, dtype=_DT)
        y = y.clamp(-self.amplifier.range_uv, self.amplifier.range_uv)  # clipping
        return y, parts

    # -- separation ----------------------------------------------------------

    def separate(
        self, t: Tensor, recorded: Tensor, *, e_field_v_per_m: float
    ) -> ArtifactComponents:
        """Least-squares decomposition on the valid (non-blanked) samples."""
        B, names = self.basis(t, e_field_v_per_m)
        m = self.valid_mask(t, recorded.to(_DT))
        Bv = B[m]

        # A component with no support on the valid samples is **not
        # identifiable**: the direct induced transient lives entirely inside
        # the blanking window, so nothing in the surviving record constrains
        # it. Fitting it anyway would let noise be amplified without bound and
        # then be subtracted from the residual as if it were known.
        colnorm = Bv.norm(dim=0)
        identifiable = colnorm > 1e-9 * float(colnorm.max())
        coef = torch.zeros(B.shape[1], dtype=_DT)
        coef[identifiable] = torch.linalg.lstsq(
            Bv[:, identifiable], recorded.to(_DT)[m], driver="gelsd"
        ).solution
        unidentifiable = [n for n, ok in zip(names, identifiable.tolist()) if not ok]
        comp = {n: B[:, i] * coef[i] for i, n in enumerate(names)}
        residual = recorded.to(_DT) - sum(comp.values())
        return ArtifactComponents(
            times_s=t,
            pulse=comp["pulse"],
            saturation=comp["saturation"],
            auditory=comp["auditory"],
            somatosensory=comp["somatosensory"],
            decay=comp["decay"],
            recorded=recorded.to(_DT),
            residual=residual,
            valid_mask=m,
            ledger=Ledger(
                variance={"measurement": float(residual[m].var())},
                bias_status="externally_bounded",
                validity_domain={
                    "blanking_s": self.amplifier.blanking_s,
                    "n_invalid_samples": int((~m).sum()),
                    "saturated_samples_excluded": True,
                    "unidentifiable_components": unidentifiable,
                    "recovery_tau_s": self.amplifier.recovery_tau_s,
                    "separated_components": list(names),
                    "residual_is_not_cortical": True,
                },
            ),
        )

    def injection_recovery(
        self,
        t: Tensor,
        injected: Tensor,
        *,
        e_field_v_per_m: float,
        noise_uv: float = 0.5,
        seed: int = 0,
    ) -> dict[str, float]:
        """Artifact-injection recovery test required by Appendix A.

        Inject a known waveform on top of the full artifact stack, separate,
        and report how much of the injected signal survives.  A pipeline that
        cannot pass this cannot support a cortical claim.
        """
        y, _ = self.simulate(
            t,
            e_field_v_per_m=e_field_v_per_m,
            cortical_response=injected,
            noise_uv=noise_uv,
            seed=seed,
        )
        parts = self.separate(t, y, e_field_v_per_m=e_field_v_per_m)
        m = parts.valid_mask
        r, i = parts.residual[m], injected.to(_DT)[m]
        corr = float(
            ((r - r.mean()) * (i - i.mean())).sum()
            / (r.std() * i.std() * (r.numel() - 1)).clamp_min(1e-30)
        )
        gain = float((r @ i) / (i @ i).clamp_min(1e-30))
        return {
            "recovery_correlation": corr,
            "recovery_gain": gain,
            "residual_rms_uv": float(r.std()),
            "injected_rms_uv": float(i.std()),
        }
