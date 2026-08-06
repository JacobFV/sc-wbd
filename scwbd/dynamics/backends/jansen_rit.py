"""Jansen–Rit PSP-based cortical column backend.

Jansen & Rit (1995).  Six states per region: three post-synaptic potentials and
their derivatives.  This is the backend that produces realistic EEG spectra
(alpha rhythm from the pyramidal PSP ``y1 - y2``), which is why it is kept even
though its firing-rate summary is less convenient than Wilson–Cowan's.

    y0'' = A a S(y1 - y2)                       - 2a y0' - a^2 y0   (pyramidal)
    y1'' = A a (p + C2 S(C1 y0) + g c)          - 2a y1' - a^2 y1   (excitatory)
    y2'' = B b C4 S(C3 y0)                      - 2b y2' - b^2 y2   (inhibitory)
    S(v) = 2 e0 / (1 + exp(r (v0 - v)))

Units: potentials in mV, rates in s^-1.  The long-range coupling variable is the
**pyramidal output firing rate** ``S(y1 - y2)``, not the PSP itself: a tract
transmits spikes.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from ..base import BackendInfo, DynamicsBackend, register_backend
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator


@register_backend
class JansenRit(DynamicsBackend):
    info: ClassVar[BackendInfo] = BackendInfo(
        name="jansen_rit",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("y0", "y1", "y2", "y3", "y4", "y5"),
        units=("mV", "mV", "mV", "mV/s", "mV/s", "mV/s"),
        reference="Jansen & Rit 1995; David & Friston 2003",
        falsifier=(
            "Disabled if its PSP-level commitments buy nothing: i.e. if an equal-capacity "
            "surrogate matched on spectra also reproduces the laminar/timing signatures "
            "(alpha-band peak shift under changed inhibitory time constant b) that only the "
            "PSP kinetics are supposed to explain."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1

    defaults: ClassVar[Mapping[str, float]] = {
        "A": 3.25,  # mV, excitatory PSP amplitude
        "B": 22.0,  # mV, inhibitory PSP amplitude
        "a": 100.0,  # 1/s, excitatory rate constant
        "b": 50.0,  # 1/s, inhibitory rate constant
        "C": 135.0,
        "c1_f": 1.0,
        "c2_f": 0.8,
        "c3_f": 0.25,
        "c4_f": 0.25,
        "v0": 6.0,  # mV
        "e0": 2.5,  # 1/s
        "r": 0.56,  # 1/mV
        "p_mean": 120.0,  # 1/s
        "g_coupling": 0.0,
        "sigma": 20.0,  # noise on the excitatory input channel, 1/s
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "A": Prior("A", 3.25, 0.5, "normal", "mV", low=1.0),
        "B": Prior("B", 22.0, 3.0, "normal", "mV", low=5.0),
        "a": Prior("a", 100.0, 15.0, "normal", "1/s", low=40.0),
        "b": Prior("b", 50.0, 8.0, "normal", "1/s", low=20.0),
        "C": Prior("C", 135.0, 20.0, "normal", "dimensionless", low=50.0),
        "p_mean": Prior("p_mean", 150.0, 60.0, "uniform", "1/s", low=60.0, high=320.0),
        "g_coupling": Prior("g_coupling", 5.0, 5.0, "uniform", "dimensionless", low=0.0, high=20.0),
        "sigma": Prior("sigma", 20.0, 15.0, "uniform", "1/s", low=0.0, high=60.0),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("p_mean",)

    #: ``a`` is the excitatory PSP rate constant: the kernel ``A a t exp(-a t)``
    #: peaks at ``1/a`` = 10 ms, the same synaptic time constant Wilson-Cowan
    #: spells ``tau_e`` = 0.010 s.  It is strictly positive and its reciprocal is
    #: genuinely the relaxation time, so the intrinsic-timescale prior applies
    #: here on exactly the grounds it applies to ``tau_e`` -- reciprocally,
    #: because a slower region has a *smaller* rate.
    inverse_timescale_params: ClassVar[tuple[str, ...]] = ("a",)
    #: 1/a in 5-25 ms; below ~5 ms the PSP is faster than the 1 ms solver step,
    #: above ~25 ms Jansen-Rit leaves the regime where it produces alpha.
    param_support: ClassVar[Mapping[str, tuple[float | None, float | None]]] = {
        "a": (40.0, 200.0),
        "b": (20.0, 100.0),
    }
    #: Jansen-Rit has no scalar "inhibitory gain"; the closest candidate is ``B``,
    #: the inhibitory PSP amplitude.  Deliberately NOT mapped: the alpha rhythm
    #: depends jointly on A, B and the C-ratios, so rescaling B alone changes the
    #: spectral peak as much as the E/I balance, and the anatomical E/I contrast
    #: is route-fragile in sign (see map_fragility on ei_proxy).  Recording the
    #: candidate and the refusal is more useful than a mapping we cannot defend.
    ei_not_mapped_reason: ClassVar[str] = (
        "Jansen-Rit has no scalar inhibitory-gain parameter. The candidate is B (inhibitory PSP "
        "amplitude, mV), deliberately not mapped: B co-determines the alpha peak with A and the "
        "C-ratios, so scaling it alone would confound an E/I change with a spectral one, and the "
        "receptor-derived E/I contrast is route-fragile in per-parcel sign."
    )

    @property
    def state_dim(self) -> int:
        return 6

    @staticmethod
    def _sigm(v: Tensor, e0: Tensor, r: Tensor, v0: Tensor) -> Tensor:
        return 2.0 * e0 / (1.0 + torch.exp(r * (v0 - v)))

    def init_state(
        self,
        batch: int,
        n_regions: int,
        *,
        seed: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        theta: ParamPack | None = None,
    ) -> Tensor:
        dev = default_device(device)
        g = make_generator(seed, dev)
        x = torch.zeros((batch, n_regions, 6), device=dev, dtype=dtype)
        x[..., :3] = 0.1 * torch.randn((batch, n_regions, 3), generator=g, device=dev, dtype=dtype)
        return x

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        y0, y1, y2 = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        y3, y4, y5 = x[..., 3:4], x[..., 4:5], x[..., 5:6]
        A, B = theta.get("A"), theta.get("B")
        a, b = theta.get("a"), theta.get("b")
        C = theta.get("C")
        C1, C2 = C * theta.get("c1_f"), C * theta.get("c2_f")
        C3, C4 = C * theta.get("c3_f"), C * theta.get("c4_f")
        e0, r, v0 = theta.get("e0"), theta.get("r"), theta.get("v0")
        p = theta.get("p_mean")
        g = theta.get("g_coupling")
        c = coupling_input[..., 0:1]
        drive = u[..., 0:1] if u is not None else 0.0

        s_pyr = self._sigm(y1 - y2, e0, r, v0)
        dy0 = y3
        dy1 = y4
        dy2 = y5
        dy3 = A * a * s_pyr - 2.0 * a * y3 - a * a * y0
        dy4 = A * a * (p + C2 * self._sigm(C1 * y0, e0, r, v0) + g * c + drive) - 2.0 * a * y4 - a * a * y1
        dy5 = B * b * C4 * self._sigm(C3 * y0, e0, r, v0) - 2.0 * b * y5 - b * b * y2
        return torch.cat([dy0, dy1, dy2, dy3, dy4, dy5], dim=-1)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        """Noise enters only the excitatory input channel (y4), as in the literature."""
        out = torch.zeros_like(x)
        amp = theta.get("sigma") * theta.get("A") * theta.get("a")
        out[..., 4:5] = amp
        return out

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        psp = x[..., 1] - x[..., 2]
        return {
            "activity": psp,  # pyramidal membrane potential -> EEG-facing observable
            "psp": psp,
            "pyramidal_rate": (2.0 * 2.5 / (1.0 + torch.exp(0.56 * (6.0 - psp)))),
        }

    def coupling_variable(self, x: Tensor, theta: ParamPack | None = None) -> Tensor:
        psp = (x[..., 1] - x[..., 2]).unsqueeze(-1)
        if theta is None:
            e0 = torch.as_tensor(self.defaults["e0"], device=x.device, dtype=x.dtype)
            r = torch.as_tensor(self.defaults["r"], device=x.device, dtype=x.dtype)
            v0 = torch.as_tensor(self.defaults["v0"], device=x.device, dtype=x.dtype)
        else:
            e0, r, v0 = theta.get("e0"), theta.get("r"), theta.get("v0")
        return self._sigm(psp, e0, r, v0)
