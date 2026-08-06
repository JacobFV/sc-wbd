"""Balloon–Windkessel neurovascular model (Buxton/Friston) — neural state -> BOLD.

This is the *neural-to-hemodynamic state* only.  The sampling/observation head
(slice timing, drift, noise model, spatial support) belongs to agent F; what
lives here is the vascular state itself, because it is a **state of the
simulated brain** that must be advanced on the multirate schedule (thesis §4.5:
"sustained activity altering metabolic demand" is the canonical forced
synchronisation).

State per region (4): ``s`` vasodilatory signal, ``f`` inflow, ``v`` blood
volume, ``q`` deoxyhaemoglobin content.  Integrated in log space for ``f, v, q``
(Stephan et al. 2007) so positivity is structural rather than clamped.

    ds/dt      = z - kappa s - gamma (f - 1)
    dln f/dt   = s / f
    dln v/dt   = (f - v^(1/alpha)) / (tau v)
    dln q/dt   = (f E(f,E0)/E0 - v^(1/alpha) q/v) / (tau q)
    E(f,E0)    = 1 - (1 - E0)^(1/f)

    BOLD = V0 [ k1 (1-q) + k2 (1 - q/v) + k3 (1-v) ]

Hemodynamic parameters vary by subject and session; they are supplied as
:class:`~scwbd.dynamics.types.Prior` objects and sampled with an explicit
subject/session decomposition that is **centred** (sum-to-zero) so the additive
decomposition is identified (cf. refusal R07).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import torch
from torch import Tensor, nn

from .scheduler import FieldPolicy
from .types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = ["BalloonWindkessel", "HEMODYNAMIC_PRIORS", "sample_hemodynamic_params", "bold_field_policy"]


#: Literature priors (Friston et al. 2000, 2003; Stephan et al. 2007).
HEMODYNAMIC_PRIORS: dict[str, Prior] = {
    "kappa": Prior("kappa", 0.65, 0.12, "lognormal_like", "1/s", low=0.2, high=1.5, provenance="Friston 2003"),
    "gamma": Prior("gamma", 0.41, 0.08, "normal", "1/s", low=0.1, high=0.9, provenance="Friston 2003"),
    "tau": Prior("tau", 0.98, 0.15, "normal", "s", low=0.4, high=2.5, provenance="Friston 2003"),
    "alpha": Prior("alpha", 0.32, 0.04, "normal", "dimensionless", low=0.15, high=0.6),
    "E0": Prior("E0", 0.34, 0.05, "normal", "dimensionless", low=0.15, high=0.6),
    "epsilon": Prior("epsilon", 1.0, 0.2, "normal", "dimensionless", low=0.3, high=2.5),
    "V0": Prior("V0", 0.04, 0.005, "normal", "dimensionless", low=0.01, high=0.08),
    "neural_gain": Prior("neural_gain", 1.0, 0.2, "normal", "dimensionless", low=0.2),
}


class BalloonWindkessel(nn.Module):
    """Batched Balloon–Windkessel.  State ``(B, N, 4)`` = ``(s, ln f, ln v, ln q)``."""

    state_dim: ClassVar[int] = 4
    state_names: ClassVar[tuple[str, ...]] = ("s", "log_f", "log_v", "log_q")
    defaults: ClassVar[Mapping[str, float]] = {
        "kappa": 0.65,
        "gamma": 0.41,
        "tau": 0.98,
        "alpha": 0.32,
        "E0": 0.34,
        "epsilon": 1.0,
        "V0": 0.04,
        "neural_gain": 1.0,
        "TE": 0.04,  # s, echo time
        "nu0": 40.3,  # 1/s, frequency offset at outer surface of magnetised vessel (3T)
        "r0": 25.0,  # 1/s, intravascular relaxation rate slope
        "sigma": 0.0,  # process noise on the vasodilatory signal
    }

    def __init__(self, *, log_space: bool = True):
        super().__init__()
        self.log_space = bool(log_space)

    # -- state -------------------------------------------------------------
    def init_state(
        self,
        batch: int,
        n_regions: int,
        *,
        seed: int = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
    ) -> Tensor:
        """Resting fixed point: ``s=0, f=v=q=1`` (log 0)."""
        dev = default_device(device)
        return torch.zeros((batch, n_regions, 4), device=dev, dtype=dtype)

    def _unpack(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s = x[..., 0:1]
        if self.log_space:
            f = x[..., 1:2].exp()
            v = x[..., 2:3].exp()
            q = x[..., 3:4].exp()
        else:
            f = x[..., 1:2].clamp_min(1e-6)
            v = x[..., 2:3].clamp_min(1e-6)
            q = x[..., 3:4].clamp_min(1e-6)
        return s, f, v, q

    def drift(self, x: Tensor, neural: Tensor, theta: ParamPack, t: float = 0.0) -> Tensor:
        """``neural`` is the neural drive ``(B, N)`` or ``(B, N, 1)``."""
        if neural.ndim == 2:
            neural = neural.unsqueeze(-1)
        s, f, v, q = self._unpack(x)
        kappa, gamma = theta.get("kappa"), theta.get("gamma")
        tau, alpha = theta.get("tau"), theta.get("alpha")
        E0 = theta.get("E0")
        z = theta.get("neural_gain") * neural
        ds = z - kappa * s - gamma * (f - 1.0)
        v_alpha = v.pow(1.0 / alpha)
        E = 1.0 - (1.0 - E0).pow(1.0 / f.clamp_min(1e-3))
        if self.log_space:
            df = s / f
            dv = (f - v_alpha) / (tau * v)
            dq = (f * E / E0 - v_alpha * q / v) / (tau * q)
        else:
            df = s
            dv = (f - v_alpha) / tau
            dq = (f * E / E0 - v_alpha * q / v) / tau
        return torch.cat([ds, df, dv, dq], dim=-1)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        out = torch.zeros_like(x)
        out[..., 0:1] = theta.get("sigma")
        return out

    # -- observation -------------------------------------------------------
    def bold(self, x: Tensor, theta: ParamPack) -> Tensor:
        """BOLD percent signal change ``(B, N)``.

        ``k1 = 4.3 nu0 E0 TE``, ``k2 = epsilon r0 E0 TE``, ``k3 = 1 - epsilon``
        (Obata et al. 2004 / Stephan et al. 2007 revised coefficients).
        """
        _, _, v, q = self._unpack(x)
        E0, TE = theta.get("E0"), theta.get("TE")
        eps = theta.get("epsilon")
        k1 = 4.3 * theta.get("nu0") * E0 * TE
        k2 = eps * theta.get("r0") * E0 * TE
        k3 = 1.0 - eps
        V0 = theta.get("V0")
        y = V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
        return y.squeeze(-1)

    def observables(self, x: Tensor, theta: ParamPack) -> dict[str, Tensor]:
        s, f, v, q = self._unpack(x)
        return {
            "bold": self.bold(x, theta),
            "cbf": f.squeeze(-1),
            "cbv": v.squeeze(-1),
            "dhb": q.squeeze(-1),
            "vasodilatory_signal": s.squeeze(-1),
        }

    def make_theta(
        self, batch: int, n_regions: int, *, device=None, dtype: torch.dtype = DTYPE, **overrides
    ) -> ParamPack:
        return ParamPack(
            values=dict(overrides),
            batch=batch,
            n_regions=n_regions,
            device=default_device(device),
            dtype=dtype,
            defaults=dict(self.defaults),
        )

    # -- convenience: fixed-step rollout ----------------------------------
    def rollout(
        self,
        neural: Tensor,
        theta: ParamPack,
        *,
        dt: float,
        x0: Tensor | None = None,
        method: str = "heun",
    ) -> tuple[Tensor, Tensor]:
        """Drive the balloon with a neural time series ``(T, B, N)`` -> BOLD ``(T, B, N)``."""
        from .integrators import get_integrator

        step = get_integrator(method)
        T, B, N = neural.shape
        x = self.init_state(B, N, device=neural.device, dtype=neural.dtype) if x0 is None else x0
        outs = []
        for k in range(T):
            zk = neural[k]
            f = lambda xx, tt, z=zk: self.drift(xx, z, theta, tt)
            x = step(f, None, x, k * dt, dt, None)
            outs.append(self.bold(x, theta))
        return torch.stack(outs), x


def sample_hemodynamic_params(
    *,
    n_subjects: int,
    n_sessions: int,
    n_regions: int = 1,
    seed: int,
    device: str | torch.device | None = None,
    subject_sd_frac: float = 0.15,
    session_sd_frac: float = 0.05,
    names: tuple[str, ...] = ("kappa", "gamma", "tau", "alpha", "E0", "epsilon", "neural_gain"),
) -> ParamPack:
    """Subject/session-varying hemodynamic parameters, **centred**.

    ``value = population + subject_effect + session_effect`` with each effect
    forced to sum to zero over its level, so the additive decomposition is
    identified (refusal R07: population/subject/session effects without
    centering or shrinkage are rejected).  Batch axis is ``subject x session``.
    """
    dev = default_device(device)
    B = n_subjects * n_sessions
    vals: dict[str, Tensor | float] = {}
    for i, name in enumerate(names):
        pr = HEMODYNAMIC_PRIORS[name]
        g = make_generator(seed + 7919 * i, dev)
        pop = float(pr.mean())
        subj = pr.scale * subject_sd_frac / max(pr.scale, 1e-9) * (
            torch.randn((n_subjects, 1), generator=g, device=dev, dtype=DTYPE) * abs(pop) * subject_sd_frac
        )
        subj = subj - subj.mean()
        sess = torch.randn((n_subjects, n_sessions), generator=g, device=dev, dtype=DTYPE) * abs(pop) * session_sd_frac
        sess = sess - sess.mean(dim=1, keepdim=True)
        v = (pop + subj + sess).reshape(B, 1)
        lo = pr.low if pr.low is not None else -float("inf")
        hi = pr.high if pr.high is not None else float("inf")
        vals[name] = v.clamp(lo, hi)
    return ParamPack(
        values=vals,
        batch=B,
        n_regions=n_regions,
        device=dev,
        defaults=dict(BalloonWindkessel.defaults),
    )


def bold_field_policy(
    name: str = "hemodynamic",
    *,
    dt: float = 0.05,
    error_budget: float = 1e-3,
    inputs: tuple[str, ...] = ("neural",),
) -> FieldPolicy:
    """The hemodynamic state as a **slow field** in the multirate schedule.

    ``dt = 50 ms`` against a ~0.1–1 ms neural clock: two to three orders of
    magnitude slower, ``linear`` interpolation contract (the vascular response
    is smooth), and an explicit coarsening error budget.
    """
    return FieldPolicy(
        name=name,
        dt=dt,
        kind="continuous",
        interpolation="linear",
        error_budget=error_budget,
        inputs=inputs,
        units="dimensionless",
        clock="hemodynamic",
        description="Balloon-Windkessel vascular state; observation-side, forced-synced by sustained activity",
    )
