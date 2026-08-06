"""Wilson–Cowan E/I population-rate backend (batched, GPU, differentiable).

Wilson & Cowan (1972); network form as in Deco et al. (2009) / TVB.  Time is in
**seconds** throughout (SI, per the units contract).

    tau_E dE/dt = -E + (1 - r_E E) S_E(c_ee E - c_ei I + P + g c)
    tau_I dI/dt = -I + (1 - r_I I) S_I(c_ie E - c_ii I + Q)

with ``S(x) = [sigma(a(x-b)) - sigma(-ab)] / (1 - sigma(-ab))`` so that
``S(0) = 0`` and the fixed point at zero input is at the origin.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from ..base import BackendInfo, DynamicsBackend, register_backend, sigmoid_offset
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator


@register_backend
class WilsonCowan(DynamicsBackend):
    """Two-population (E, I) firing-rate model."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="wilson_cowan",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("E", "I"),
        units=("dimensionless", "dimensionless"),
        reference="Wilson & Cowan 1972; Deco et al. 2009",
        falsifier=(
            "Disabled if, at matched capacity, an equal-capacity learned surrogate matches or "
            "beats its E/I-specific predictions (rate-dependent gain, inhibition-stabilised "
            "paradoxical response to inhibitory drive) on held-out perturbations."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1
    coupling_kind: ClassVar[str] = "additive"

    defaults: ClassVar[Mapping[str, float]] = {
        "c_ee": 16.0,
        "c_ei": 12.0,
        "c_ie": 15.0,
        "c_ii": 3.0,
        "tau_e": 0.010,  # s
        "tau_i": 0.020,  # s
        "a_e": 1.3,
        "b_e": 4.0,
        "a_i": 2.0,
        "b_i": 3.7,
        "r_e": 1.0,
        "r_i": 1.0,
        "P": 1.25,
        "Q": 0.0,
        "g_coupling": 1.0,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "c_ee": Prior("c_ee", 16.0, 2.0, "normal", "dimensionless", low=4.0),
        "c_ei": Prior("c_ei", 12.0, 2.0, "normal", "dimensionless", low=2.0),
        "c_ie": Prior("c_ie", 15.0, 2.0, "normal", "dimensionless", low=2.0),
        "c_ii": Prior("c_ii", 3.0, 1.0, "normal", "dimensionless", low=0.0),
        "P": Prior("P", 1.25, 0.4, "normal", "dimensionless"),
        "g_coupling": Prior("g_coupling", 0.5, 0.4, "uniform", "dimensionless", low=0.0, high=2.0),
        "sigma": Prior("sigma", 0.01, 0.005, "uniform", "dimensionless", low=0.0, high=0.05),
        "tau_e": Prior("tau_e", 0.010, 0.002, "normal", "s", low=0.002),
        "tau_i": Prior("tau_i", 0.020, 0.004, "normal", "s", low=0.004),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("P", "c_ei")

    @property
    def state_dim(self) -> int:
        return 2

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
        x = 0.1 + 0.05 * torch.rand((batch, n_regions, 2), generator=g, device=dev, dtype=dtype)
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
        E = x[..., 0:1]
        I = x[..., 1:2]
        c_ee = theta.get("c_ee")
        c_ei = theta.get("c_ei") * theta.get("ei_ratio")
        c_ie = theta.get("c_ie")
        c_ii = theta.get("c_ii")
        tau_e = theta.get("tau_e")
        tau_i = theta.get("tau_i")
        P = theta.get("P")
        Q = theta.get("Q")
        g = theta.get("g_coupling")
        c = coupling_input[..., 0:1]
        drive = u[..., 0:1] if u is not None else 0.0

        x_e = c_ee * E - c_ei * I + P + g * c + drive
        x_i = c_ie * E - c_ii * I + Q
        s_e = sigmoid_offset(x_e, theta.get("a_e"), theta.get("b_e"))
        s_i = sigmoid_offset(x_i, theta.get("a_i"), theta.get("b_i"))
        dE = (-E + (1.0 - theta.get("r_e") * E) * s_e) / tau_e
        dI = (-I + (1.0 - theta.get("r_i") * I) * s_i) / tau_i
        return torch.cat([dE, dI], dim=-1)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        E, I = x[..., 0], x[..., 1]
        return {
            "activity": E,  # excitatory rate drives the hemodynamic head
            "E": E,
            "I": I,
            "ei_balance": E - I,
        }
