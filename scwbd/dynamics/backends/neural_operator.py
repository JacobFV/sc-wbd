"""``LearnedNeuralOperator`` — the equal-capacity control for every mechanistic claim.

Per thesis §4.4 and Appendix D ("Operator / mechanism claim"), a mechanistic
label is earned only by predictions a generic surrogate misses.  This module is
that generic surrogate: same interface, same state dimension, same coupling
channels, same integrator, *matched capacity* — and no mechanistic commitments
whatsoever.  Its ``mechanistic_status`` is permanently ``"surrogate"``.

The capacity match is enforced, not asserted: :func:`match_capacity` picks the
hidden width that lands within a tolerance of a target parameter count, and
:func:`assert_equal_capacity` raises if a comparison is run at mismatched
capacity.

bf16 is permitted here (ARCHITECTURE.md §3 allows it *inside* learned
operators): set ``compute_dtype=torch.bfloat16``.  The returned drift is always
cast back to the solver dtype, so the solver itself never sees bf16.
"""

from __future__ import annotations

import math
from typing import ClassVar, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..base import BackendInfo, DynamicsBackend, register_backend
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = ["LearnedNeuralOperator", "match_capacity", "assert_equal_capacity"]


class _FiLM(nn.Module):
    """Parameter-set conditioning: theta -> per-feature scale and shift."""

    def __init__(self, n_cond: int, width: int):
        super().__init__()
        self.to_scale = nn.Linear(n_cond, width)
        self.to_shift = nn.Linear(n_cond, width)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return h * (1.0 + self.to_scale(cond)) + self.to_shift(cond)


@register_backend
class LearnedNeuralOperator(DynamicsBackend):
    """A capacity-matched generic drift/diffusion operator.

    Architecture: per-region MLP over ``[x, coupling, u]`` with FiLM
    conditioning on the (named, ordered) parameter vector.  It is deliberately
    *not* given the connectome as an inductive bias — that is the job of the
    coupling module, which both this and the mechanistic backends share, so the
    comparison isolates the regional operator.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="learned_operator",
        family="surrogate",
        mechanistic_status="surrogate",
        state_names=("z",),
        units=("dimensionless",),
        reference="thesis §4.3/§4.4; Appendix D operator/mechanism claim",
        falsifier=(
            "Not falsifiable as a mechanism — it makes no mechanistic claim. Its role is to "
            "falsify *other* backends' mechanistic labels by matching their predictions at "
            "equal capacity."
        ),
    )
    learned: ClassVar[bool] = True
    defaults: ClassVar[Mapping[str, float]] = {"sigma": 0.01, "gain": 1.0}
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "sigma": Prior("sigma", 0.01, 0.01, "uniform", "dimensionless", low=0.0, high=0.05),
        "gain": Prior("gain", 1.0, 0.2, "normal", "dimensionless", low=0.1),
    }

    def __init__(
        self,
        state_dim: int = 2,
        *,
        n_coupling_channels: int = 1,
        cond_names: Sequence[str] = ("gain",),
        width: int = 64,
        depth: int = 2,
        u_dim: int = 0,
        learn_diffusion: bool = True,
        compute_dtype: torch.dtype | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self._state_dim = int(state_dim)
        self.n_coupling_channels = int(n_coupling_channels)  # instance override of ClassVar
        self.cond_names = tuple(cond_names)
        self.u_dim = int(u_dim)
        self.compute_dtype = compute_dtype
        torch.manual_seed(seed)
        d_in = self._state_dim + self.n_coupling_channels + self.u_dim
        layers: list[nn.Module] = [nn.Linear(d_in, width)]
        self.films = nn.ModuleList([_FiLM(len(self.cond_names), width) for _ in range(depth)])
        hidden = [nn.Linear(width, width) for _ in range(depth - 1)]
        self.inp = layers[0]
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(width, self._state_dim)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
        self.log_sigma = (
            nn.Parameter(torch.full((self._state_dim,), math.log(0.01))) if learn_diffusion else None
        )
        self.act = nn.Tanh()

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(f"z{i}" for i in range(self._state_dim))

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
        return 0.1 * torch.randn((batch, n_regions, self._state_dim), generator=g, device=dev, dtype=dtype)

    def _cond(self, theta: ParamPack, x: Tensor) -> Tensor:
        cols = [theta.get(n).expand(x.shape[0], x.shape[1], 1) for n in self.cond_names]
        return torch.cat(cols, dim=-1)

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        parts = [x, coupling_input]
        if self.u_dim:
            parts.append(u if u is not None else x.new_zeros(x.shape[0], x.shape[1], self.u_dim))
        h_in = torch.cat(parts, dim=-1)
        cond = self._cond(theta, x)
        cdt = self.compute_dtype
        if cdt is None:
            out = self._net(h_in, cond)
        else:
            # bf16 is permitted *inside* a learned operator (ARCHITECTURE.md §3)
            with torch.autocast(device_type=x.device.type, dtype=cdt):
                out = self._net(h_in, cond)
        return out.to(x.dtype)  # the solver never sees bf16

    def _net(self, h_in: Tensor, cond: Tensor) -> Tensor:
        h = self.act(self.films[0](self.inp(h_in), cond))
        for i, layer in enumerate(self.hidden):
            h = self.act(self.films[i + 1](layer(h), cond))
        return self.out(h)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        if self.log_sigma is None:
            return theta.get("sigma").expand_as(x)
        return self.log_sigma.exp().to(x.dtype).reshape(1, 1, -1).expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        return {"activity": x[..., 0], "z": x[..., 0]}


def match_capacity(
    target_params: int,
    *,
    state_dim: int,
    n_coupling_channels: int = 1,
    n_cond: int = 1,
    depth: int = 2,
    u_dim: int = 0,
    tol: float = 0.10,
    **kw,
) -> LearnedNeuralOperator:
    """Build a surrogate whose parameter count lands within ``tol`` of ``target_params``.

    Binary search on the hidden width.  Raises if no width lands inside the
    tolerance band — a comparison at mismatched capacity is not a comparison,
    and silently accepting one is exactly the failure mode Appendix D names.
    """
    def build(w: int) -> LearnedNeuralOperator:
        return LearnedNeuralOperator(
            state_dim=state_dim,
            n_coupling_channels=n_coupling_channels,
            cond_names=tuple(f"c{i}" for i in range(n_cond)) if n_cond != 1 else ("gain",),
            width=w,
            depth=depth,
            u_dim=u_dim,
            **kw,
        )

    lo, hi = 1, 4
    while build(hi).capacity() < target_params and hi < 8192:
        hi *= 2
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build(mid)
        c = m.capacity()
        if best is None or abs(c - target_params) < abs(best[1] - target_params):
            best = (m, c)
        if c < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    assert best is not None
    model, cap = best
    if abs(cap - target_params) > tol * max(target_params, 1):
        raise ValueError(
            f"cannot match capacity {target_params} within {tol:.0%} using width search "
            f"(closest = {cap}). Change depth or relax tol explicitly — do not run a "
            "mechanism-vs-surrogate comparison at mismatched capacity."
        )
    return model


def assert_equal_capacity(a: DynamicsBackend, b: DynamicsBackend, tol: float = 0.10) -> None:
    """Refuse a mechanism-vs-surrogate comparison at mismatched capacity."""
    ca, cb = a.capacity(), b.capacity()
    ref = max(ca, cb, 1)
    if abs(ca - cb) > tol * ref:
        raise ValueError(
            f"capacity mismatch: {a.info.name}={ca} vs {b.info.name}={cb} "
            f"(> {tol:.0%} of {ref}). An 'equal-capacity control' must actually be equal capacity."
        )
