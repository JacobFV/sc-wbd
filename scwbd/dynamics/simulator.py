"""The whole-brain rollout: backend + delayed coupling + integrator, batched on device.

This is the throughput-critical path — it generates the simulated trajectories
the foundation model trains on — so the hot loop obeys three rules:

1. **No Python loop over regions or edges.**  Per step there are exactly two
   sparse ops (one gather, one segment-scatter) plus the backend's elementwise
   vector field.
2. **Nothing leaves the device.**  No ``.item()``, no host sync, no CPU
   allocation inside the loop.  Delay indices are computed once, not per step.
3. **Preallocated output.**  Recording writes into a preallocated tensor rather
   than appending to a list of GPU tensors.

Delay/Heun interaction: within one step the delayed long-range input is held
fixed at ``c(t)`` for both the predictor and the corrector stage.  This is the
standard treatment for delay differential equations at ``dt << tau`` and it
costs one order in the *coupling* term only; it is stated here rather than
hidden, and :func:`~scwbd.dynamics.integrators.certify_semigroup` will see it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .backends import DynamicsBackend
from .coupling import DelayBuffer, DelayedConnectome, HybridField, LocalField, deterministic_scatter
from .integrators import get_integrator
from .residual import MechanismDominanceGuard
from .types import DTYPE, NumericalBudget, ParamPack, assert_solver_dtype, default_device, make_generator

__all__ = ["SimConfig", "SimResult", "WholeBrainSimulator", "functional_connectivity", "fc_correlation"]


@dataclass
class SimConfig:
    dt: float = 1e-3
    n_steps: int = 1000
    method: str = "heun"
    seed: int = 0
    warmup_steps: int = 0
    record_every: int = 1
    record: tuple[str, ...] = ("activity",)
    store_state: bool = False
    stochastic: bool = True
    #: bitwise reproducibility (see coupling.deterministic_scatter); ~4x slower
    #: on the sparse scatter. Off by default for bulk trajectory generation.
    deterministic: bool = False


@dataclass
class SimResult:
    """Trajectory plus everything needed to report it honestly."""

    observables: dict[str, Tensor]  # each (T_rec, B, N)
    x_final: Tensor
    states: Tensor | None
    dt: float
    n_steps: int
    record_every: int
    budget: NumericalBudget
    guard_report: dict[str, Any] | None = None
    wall_time_s: float = 0.0

    @property
    def n_recorded(self) -> int:
        k = next(iter(self.observables))
        return int(self.observables[k].shape[0])

    def activity(self) -> Tensor:
        return self.observables["activity"]

    def fc(self, name: str = "activity", *, downsample: int = 1) -> Tensor:
        return functional_connectivity(self.observables[name][::downsample])

    def summary(self) -> str:
        return (
            f"SimResult(T={self.n_recorded}, B={self.observables['activity'].shape[1]}, "
            f"N={self.observables['activity'].shape[2]}, dt={self.dt:g}s, "
            f"sim_time={self.n_steps * self.dt:g}s, wall={self.wall_time_s:.3f}s)"
        )


def functional_connectivity(x: Tensor) -> Tensor:
    """Pearson FC over time.  ``x``: ``(T, B, N)`` -> ``(B, N, N)``."""
    xc = x - x.mean(dim=0, keepdim=True)
    sd = xc.pow(2).mean(dim=0).sqrt().clamp_min(1e-8)
    xn = xc / sd.unsqueeze(0)
    T = x.shape[0]
    return torch.einsum("tbi,tbj->bij", xn, xn) / T


def fc_correlation(fc_a: Tensor, fc_b: Tensor, *, offdiag_only: bool = True) -> Tensor:
    """Correlation between two FC matrices (the standard model-fit statistic)."""
    N = fc_a.shape[-1]
    if offdiag_only:
        mask = ~torch.eye(N, dtype=torch.bool, device=fc_a.device)
        a = fc_a[..., mask]
        b = fc_b[..., mask] if fc_b.ndim == fc_a.ndim else fc_b[mask].unsqueeze(0)
    else:
        a = fc_a.flatten(start_dim=-2)
        b = fc_b.flatten(start_dim=-2)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    num = (a * b).sum(-1)
    den = (a.pow(2).sum(-1).sqrt() * b.pow(2).sum(-1).sqrt()).clamp_min(1e-12)
    return num / den


class WholeBrainSimulator:
    """Composes a backend, the §4.1 field factorization and an integrator."""

    def __init__(
        self,
        backend: DynamicsBackend,
        connectome: DelayedConnectome | None = None,
        *,
        local: LocalField | None = None,
        residual=None,
        guard: MechanismDominanceGuard | None = None,
        clamp: tuple[float, float] | None = None,
        wrap_phase: bool = False,
    ):
        self.backend = backend
        self.field = HybridField(backend, connectome, local, residual, guard)
        self.connectome = connectome
        self.clamp = clamp
        self.wrap_phase = wrap_phase or backend.coupling_kind == "phase_difference"
        if connectome is not None and connectome.C != backend.n_coupling_channels:
            raise ValueError(
                f"coupling channel mismatch: connectome has {connectome.C} channels, "
                f"backend {backend.info.name} consumes {backend.n_coupling_channels}"
            )
        if connectome is not None and connectome.mode != backend.coupling_kind:
            raise ValueError(
                f"coupling mode mismatch: connectome mode={connectome.mode!r} but backend "
                f"{backend.info.name} declares coupling_kind={backend.coupling_kind!r}"
            )

    # -- setup -------------------------------------------------------------
    def prepare(self, theta: ParamPack, cfg: SimConfig, x0: Tensor | None = None):
        B, N = theta.batch, theta.n_regions
        dev = theta.device
        # learned operators may have been constructed on CPU; the parameter batch
        # decides the device, and nothing crosses devices inside the loop.
        self.field.to(dev)
        x = self.backend.init_state(B, N, seed=cfg.seed, device=dev, theta=theta) if x0 is None else x0
        assert_solver_dtype(x)
        buf = None
        delay_steps = None
        if self.connectome is not None:
            delay_steps = self.connectome.delay_steps(theta, cfg.dt)
            buf = DelayBuffer(
                B, N, self.connectome.C, int(math.ceil(float(delay_steps.max()))) + 1, device=dev
            )
            cv = self.backend.coupling_variable(x, theta)
            buf.reset(cv)
        return x, buf, delay_steps

    # -- the hot loop ------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        theta: ParamPack,
        cfg: SimConfig,
        *,
        x0: Tensor | None = None,
        u: Callable[[float], Tensor] | Tensor | None = None,
        grad: bool = False,
    ) -> SimResult:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, deterministic_scatter(cfg.deterministic):
            return self._run(theta, cfg, x0=x0, u=u)

    def _run(
        self,
        theta: ParamPack,
        cfg: SimConfig,
        *,
        x0: Tensor | None = None,
        u: Callable[[float], Tensor] | Tensor | None = None,
    ) -> SimResult:
        import time

        x, buf, delay_steps = self.prepare(theta, cfg, x0)
        B, N, D = x.shape
        dev = x.device
        step_fn = get_integrator(cfg.method)
        dt = float(cfg.dt)
        gen = make_generator(cfg.seed + 977, dev) if cfg.stochastic else None
        sqrt_dt = math.sqrt(dt)

        n_rec = (cfg.n_steps + cfg.record_every - 1) // cfg.record_every
        recs = {name: torch.empty(n_rec, B, N, device=dev, dtype=x.dtype) for name in cfg.record}
        states = torch.empty(n_rec, B, N, D, device=dev, dtype=x.dtype) if cfg.store_state else None

        def drift(xx: Tensor, tt: float) -> Tensor:
            uu = u(tt) if callable(u) else u
            return self.field(xx, buf, theta, dt, uu, tt, delay_steps=delay_steps)

        def diffusion(xx: Tensor, tt: float) -> Tensor:
            return self.backend.diffusion(xx, theta)

        g_fn = diffusion if cfg.stochastic else None
        noise = torch.empty_like(x) if cfg.stochastic else None

        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        t = 0.0
        for k in range(-cfg.warmup_steps, cfg.n_steps):
            dW = None
            if cfg.stochastic:
                noise.normal_(generator=gen)
                dW = noise * sqrt_dt
            x = step_fn(drift, g_fn, x, t, dt, dW)
            if self.clamp is not None:
                x = x.clamp(self.clamp[0], self.clamp[1])
            if self.wrap_phase:
                x = torch.remainder(x, 2.0 * math.pi)
            t += dt
            if buf is not None:
                buf.push(self.backend.coupling_variable(x, theta))
            if k >= 0 and (k % cfg.record_every == 0):
                i = k // cfg.record_every
                obs = self.backend.observables(x)
                for name in cfg.record:
                    recs[name][i] = obs[name]
                if states is not None:
                    states[i] = x

        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t_start

        budget = NumericalBudget()
        # first-order truncation estimate for the coupling term held over the step
        if self.connectome is not None:
            budget.add(
                "delay_coupling_zoh",
                (dt**2),
                "long-range coupling held constant across the Heun stages within a step",
            )
        guard_report = None
        if self.field.guard is not None:
            guard_report = self.field.guard.finalize().as_dict()
        return SimResult(
            observables=recs,
            x_final=x,
            states=states,
            dt=dt,
            n_steps=cfg.n_steps,
            record_every=cfg.record_every,
            budget=budget,
            guard_report=guard_report,
            wall_time_s=wall,
        )

    # -- diagnostics -------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.backend.describe()}
        if self.connectome is not None:
            out["connectome"] = self.connectome.describe()
        out["has_local_field"] = self.field.local is not None
        out["has_residual"] = self.field.residual is not None
        out["has_guard"] = self.field.guard is not None
        return out
