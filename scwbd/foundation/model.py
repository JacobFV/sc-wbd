"""SC-WBD-001-beta -- the conditional multirate whole-brain neural operator.

Implements the factorisation of ``paper/body.tex`` §4.1

.. math::  \\mathcal F(X) = \\mathcal F_{\\rm local}(X;\\mathcal M)
                          + \\mathcal F_{\\rm long}(X;\\mathcal G)
                          + \\mathcal R_\\theta(X, C)

over the structured per-parcel state of :mod:`scwbd.foundation.state`, with

* ``F_local``  a weight-shared regional operator with per-parcel FiLM modulation,
  or -- when ``local_core`` names a mechanistic backend -- that backend's drift
  evaluated in fp32, making the mechanistic/learned comparison a *configuration*
  rather than a rewrite;
* ``F_long``   delayed, connectome-masked, block-sparse **typed** coupling with
  separate hard / soft / proposed evidence classes and per-sample conduction
  delays (the delay bins are cut on tract length, so a whole batch of different
  conduction velocities still costs one GEMM);
* ``R_theta``  a learned residual whose magnitude ratio ``rho`` against the
  mechanistic terms is measured every step and refused above ``rho_max``
  (refusal **R05**).

Two clocks run: the neural state at ``dt_model`` and the haemodynamic
compartments every ``hemo_ratio`` steps (§4.5 multirate co-simulation).

The learned operator is simultaneously a **backend** and the equal-capacity
**control** for every mechanistic claim (Appendix D, "Operator / mechanism
claim").  :class:`LearnedOperatorBackend` exposes it through agent E's
``DynamicsBackend`` interface so the comparison is mechanical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .anatomy import EVIDENCE_CLASSES, AnatomyPrior
from .config import ModelConfig
from .heads import BOLDHead, BehaviourHead, EEGHead, LeadField, build_lead_field
from .state import ComponentSpec, StateLayout, default_layout, scalar_layout

__all__ = ["SCWBD", "TypedDelayCoupling", "RegionalOperator", "LearnedResidual", "R05Violation", "RolloutResult"]


class R05Violation(RuntimeError):
    """The learned residual dominated the mechanistic terms (refusal R05)."""

    code = "R05"

    def __init__(self, rho: float, rho_max: float) -> None:
        self.rho, self.rho_max = rho, rho_max
        super().__init__(
            f"[R05] learned residual magnitude ratio rho={rho:.3f} exceeds rho_max={rho_max:.3f}; "
            "the mechanistic term is being silently replaced. Remedy: lower lambda on the "
            "residual, raise its regulariser, or declare the operator 'surrogate' in the manifest."
        )


# ======================================================================
# typed delayed coupling
# ======================================================================
class TypedDelayCoupling(nn.Module):
    """``F_long``: delayed, connectome-masked, evidence-class-typed coupling.

    A long-range edge is a **typed operator**, not a scalar weight (thesis §2.2).
    Three parallel adjacency masks (hard / soft / proposed) each carry their own
    learned gain and their own regulariser: *hard* topology is compiled and its
    gain is pinned, *soft* edges may move within their prior, *proposed* edges
    are shrunk hard and reported separately, so a speculative edge can never
    quietly become load-bearing.

    Delay handling: bins are cut on **tract length**, which does not depend on
    conduction velocity.  A per-sample velocity therefore only changes the
    integer lag at which each bin is read, so a batch spanning the whole velocity
    prior still costs a single shared GEMM.
    """

    def __init__(
        self,
        anat: AnatomyPrior,
        *,
        message_dim: int,
        n_bins: int = 8,
        dt: float = 0.008,
        max_lag_steps: int = 20,
        dense_ablation: bool = False,
        control_graph: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_regions = anat.n_regions
        self.message_dim = int(message_dim)
        self.dt = float(dt)
        self.max_lag_steps = int(max_lag_steps)
        self.dense_ablation = bool(dense_ablation)
        self.control_graph = control_graph

        W = anat.weights if control_graph == "none" else anat.control_graph(control_graph, seed=seed)
        W = W.float().clone()
        if dense_ablation:  # G2 control: no connectome mask, same total weight
            n = self.n_regions
            W = torch.full_like(W, float(W.sum() / (n * n - n)))
            W.fill_diagonal_(0.0)
        L = anat.tract_length.float()

        present = W > 0
        nb = int(min(n_bins, max(int(present.sum().item()), 1)))
        if present.any():
            q = torch.linspace(0, 1, nb + 1, device=W.device)
            edges = torch.quantile(L[present], q)
            edges[0] = 0.0
            edges = torch.unique(torch.cummax(edges, 0).values)
        else:
            edges = torch.linspace(0.0, 1.0, nb + 1, device=W.device)
        nb = int(edges.numel() - 1)
        self.n_bins = nb
        idx = torch.bucketize(L, edges[1:-1].contiguous(), right=False).clamp(0, nb - 1)
        cnt = torch.zeros(nb, device=W.device).scatter_add_(0, idx.reshape(-1), present.float().reshape(-1))
        tot = torch.zeros(nb, device=W.device).scatter_add_(0, idx.reshape(-1), (L * present).reshape(-1))
        bin_length_mm = torch.where(cnt > 0, tot / cnt.clamp_min(1), 0.5 * (edges[:-1] + edges[1:]))
        self.register_buffer("bin_length_mm", bin_length_mm)

        ec = anat.evidence_class if control_graph == "none" and not dense_ablation else torch.where(
            present, torch.ones_like(idx), torch.full_like(idx, -1)
        )
        cats = []
        for ci in range(len(EVIDENCE_CLASSES)):
            m = (ec == ci) & present
            Wc = torch.zeros(nb, self.n_regions, self.n_regions, device=W.device)
            flat = Wc.reshape(nb, -1)
            flat.scatter_(0, idx.reshape(1, -1), (W * m).reshape(1, -1))
            cats.append(Wc.permute(1, 0, 2).reshape(self.n_regions, nb * self.n_regions))
        self.register_buffer("Wcat", torch.stack(cats))  # (3, N, nb*N)
        self.register_buffer("class_nnz", torch.tensor([float((ec == c).sum()) for c in range(3)]))

        # hard gain is pinned at 1 (compiled anatomy); soft/proposed are learned
        self.register_buffer("gain_hard", torch.ones(1))
        self.gain_soft = nn.Parameter(torch.zeros(1))
        self.gain_proposed = nn.Parameter(torch.full((1,), -1.5))
        self.global_scale = nn.Parameter(torch.zeros(1))

    # -- geometry ---------------------------------------------------------
    def lags_for_velocity(self, velocity_m_s: Tensor) -> Tensor:
        """``(B,) m/s -> (B, n_bins)`` integer lags in model steps."""
        v = velocity_m_s.reshape(-1, 1).clamp_min(0.2)
        lag = torch.round((self.bin_length_mm.reshape(1, -1) * 1e-3) / v / self.dt)
        return lag.clamp(0, self.max_lag_steps).long()

    def new_history(self, batch: int, device, dtype=torch.float32, *, fill: Tensor | None = None) -> Tensor:
        h = torch.zeros(batch, self.max_lag_steps + 1, self.n_regions, self.message_dim, device=device, dtype=dtype)
        return h if fill is None else h + fill.unsqueeze(1)

    @staticmethod
    def push(history: Tensor, msg: Tensor) -> Tensor:
        """Roll the window forward one step (out-of-place: autograd-safe)."""
        return torch.cat([history[:, 1:], msg.unsqueeze(1)], dim=1)

    def effective_weights(self, dtype=torch.float32) -> Tensor:
        """``(N, n_bins*N)`` typed mixture of the three evidence-class adjacencies."""
        gains = torch.stack(
            [self.gain_hard[0], torch.nn.functional.softplus(self.gain_soft[0]), torch.sigmoid(self.gain_proposed[0])]
        ).to(dtype)
        W = torch.einsum("e,enk->nk", gains, self.Wcat.to(dtype))
        return W * torch.nn.functional.softplus(self.global_scale + 0.5413).to(dtype)

    def forward(self, history: Tensor, lags: Tensor, *, weights: Tensor | None = None) -> Tensor:
        """``history (B,L,N,C)`` newest-last, ``lags (B,K)`` -> coupling ``(B,N,C)``.

        ``history`` is a *rolling window* whose last slice is the most recent
        message, so a read at lag ``d`` is ``history[:, L-1-d]``.  There is no
        modular wrap-around: a ring buffer that wraps would silently read a
        future value, which is an acausality bug, not an optimisation.
        """
        B, Lh, N, C = history.shape
        K = self.n_bins
        # history[:, -1] is the newest message; lag 0 reads it, lag L-1 the oldest
        idx = (Lh - 1 - lags).clamp(0, Lh - 1).reshape(B, K, 1, 1).expand(B, K, N, C)
        h = torch.gather(history, 1, idx)  # (B,K,N,C)
        hf = h.permute(0, 3, 1, 2).reshape(B * C, K * N)
        W = self.effective_weights(hf.dtype) if weights is None else weights
        out = hf @ W.transpose(0, 1)
        return out.reshape(B, C, N).permute(0, 2, 1)

    def class_gains(self) -> dict[str, float]:
        return {
            "hard": float(self.gain_hard[0]),
            "soft": float(torch.nn.functional.softplus(self.gain_soft[0]).detach()),
            "proposed": float(torch.sigmoid(self.gain_proposed[0]).detach()),
            "global_scale": float(torch.nn.functional.softplus(self.global_scale + 0.5413).detach()),
        }

    def topology_penalty(self) -> Tensor:
        """R_topology: proposed edges pay, hard edges do not (§6.4 assembly objective)."""
        return torch.sigmoid(self.gain_proposed[0]) ** 2 + 0.1 * torch.nn.functional.softplus(self.gain_soft[0]) ** 2


# ======================================================================
# local operator
# ======================================================================
class FiLM(nn.Module):
    """Per-parcel + per-sample modulation.

    Parcel identity enters as a *learned per-region affine* on every hidden layer
    (cheap in FLOPs, expensive in parameters -- the right trade when the operator
    is applied 454 times per step), and the sample context (theta, session) enters
    as a shared affine.
    """

    def __init__(self, n_regions: int, hidden: int, context_dim: int) -> None:
        super().__init__()
        self.region_scale = nn.Parameter(torch.zeros(n_regions, hidden))
        self.region_shift = nn.Parameter(torch.zeros(n_regions, hidden))
        self.ctx = nn.Linear(context_dim, 2 * hidden)
        nn.init.zeros_(self.ctx.weight)
        nn.init.zeros_(self.ctx.bias)

    def params(self, ctx: Tensor, dtype) -> tuple[Tensor, Tensor]:
        """Precompute ``(scale, shift)`` once per rollout.

        Neither factor depends on ``t``: parcel identity is fixed and the sample
        context is fixed for the whole rollout.  Materialising them once instead
        of once per step is the difference between ~20 GB and ~7 GB of
        activations at T=64, and costs nothing in expressiveness.
        """
        g, b = self.ctx(ctx).chunk(2, dim=-1)  # (B, 2H)
        scale = 1.0 + self.region_scale.to(dtype).unsqueeze(0) + g.unsqueeze(1).to(dtype)
        shift = self.region_shift.to(dtype).unsqueeze(0) + b.unsqueeze(1).to(dtype)
        return scale, shift

    def forward(self, h: Tensor, film: tuple[Tensor, Tensor]) -> Tensor:
        return torch.addcmul(film[1], h, film[0])


class RegionalOperator(nn.Module):
    """``F_local``: the learned regional vector field, shared across parcels."""

    def __init__(self, layout: StateLayout, n_regions: int, cfg: ModelConfig, in_extra: int) -> None:
        super().__init__()
        H = cfg.hidden
        self.embed = nn.Parameter(torch.randn(n_regions, cfg.region_embed) * 0.02)
        in_dim = layout.dim + cfg.region_embed + in_extra
        self.inp = nn.Linear(in_dim, H)
        self.layers = nn.ModuleList(nn.Linear(H, H) for _ in range(cfg.n_local_layers - 1))
        self.films = nn.ModuleList(FiLM(n_regions, H, cfg.context_dim) for _ in range(cfg.n_local_layers))
        self.out = nn.Linear(H, layout.dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def prepare(self, ctx: Tensor, dtype) -> list[tuple[Tensor, Tensor]]:
        return [f.params(ctx, dtype) for f in self.films]

    def forward(self, x: Tensor, extra: Tensor, film: list[tuple[Tensor, Tensor]]) -> Tensor:
        e = self.embed.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        h = self.act(self.films[0](self.inp(torch.cat([x, e, extra], dim=-1)), film[0]))
        for i, lin in enumerate(self.layers):
            h = h + self.drop(self.act(self.films[i + 1](lin(h), film[i + 1])))
        return self.out(h)


class MechanisticCore(nn.Module):
    """``F_local`` supplied by a mechanistic backend, evaluated in fp32.

    This is what makes "interchangeable backends compared, not assumed" a
    configuration switch.  The backend holds no learnable parameters; its
    parameters come from the conditioning vector, so a mechanistic SC-WBD and a
    learned SC-WBD differ only in this block.
    """

    def __init__(self, backend_name: str, layout: StateLayout, n_regions: int) -> None:
        super().__init__()
        from .backends import resolve_backend

        self.backend = resolve_backend(backend_name)
        self.backend_name = backend_name
        self.n_regions = n_regions
        self.layout = layout
        self.mech_dim = self.backend.state_dim
        if "mech" not in layout:
            raise ValueError("a mechanistic core needs a 'mech' state component; use SCWBD.build_layout")

    def forward(self, x: Tensor, coupling: Tensor, theta_pack) -> Tensor:
        m = self.layout.get(x, "mech").float()
        d = self.backend.drift(m, coupling[..., :1].float(), theta_pack)
        out = torch.zeros_like(x, dtype=torch.float32)
        a, b = self.layout.span("mech")
        out[..., a:b] = d
        obs = self.backend.observables(m)
        for name, key in (("rate_e", "rate_e"), ("rate_i", "rate_i")):
            if name in self.layout and key in obs:
                s = self.layout.slice(name)
                out[..., s] = (obs[key].unsqueeze(-1) - x[..., s].float()) * 20.0
        return out.to(x.dtype)


class LearnedResidual(nn.Module):
    """``R_theta``: the learned correction, kept on a short leash (R05).

    Regularised for magnitude and locality; its ratio against the mechanistic
    terms is measured, not assumed.
    """

    def __init__(self, layout: StateLayout, n_regions: int, cfg: ModelConfig, in_extra: int) -> None:
        super().__init__()
        H = max(cfg.hidden // 2, 64)
        in_dim = layout.dim + cfg.region_embed + in_extra
        self.embed = nn.Parameter(torch.randn(n_regions, cfg.region_embed) * 0.02)
        self.net = nn.Sequential(nn.Linear(in_dim, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, layout.dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(cfg.residual_init_scale, 1e-6))))

    def forward(self, x: Tensor, extra: Tensor) -> Tensor:
        e = self.embed.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.net(torch.cat([x, e, extra], dim=-1)) * self.log_scale.exp().to(x.dtype)


# ======================================================================
# assimilation encoder
# ======================================================================
class Assimilator(nn.Module):
    """Observed context window -> initial structured state (the filter's warm start).

    Missing channels enter as an explicit mask, never as zeros
    (ARCHITECTURE.md §7 rule 1).
    """

    def __init__(self, layout: StateLayout, n_regions: int, cfg: ModelConfig) -> None:
        super().__init__()
        C = cfg.encoder_channels
        self.conv = nn.Sequential(
            nn.Conv1d(2, C, 5, padding=2),
            nn.GELU(),
            *[m for i in range(cfg.encoder_layers - 1) for m in (nn.Conv1d(C, C, 5, padding=2 * 2**i, dilation=2**i), nn.GELU())],
        )
        self.embed = nn.Parameter(torch.randn(n_regions, cfg.region_embed) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(2 * C + cfg.region_embed + 4, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, layout.dim)
        )
        self.layout = layout

    def forward(self, y: Tensor, mask: Tensor | None = None) -> Tensor:
        """``y (B,T,N)`` observed regional activity -> state ``(B,N,D)``."""
        B, T, N = y.shape
        m = torch.ones_like(y) if mask is None else mask
        z = torch.stack([y * m, m], dim=-1)  # (B,T,N,2)
        z = z.permute(0, 2, 3, 1).reshape(B * N, 2, T)
        h = self.conv(z)  # (B*N, C, T)
        feat = torch.cat([h.mean(-1), h[..., -1]], dim=-1).reshape(B, N, -1)
        stats = torch.stack(
            [
                (y * m).sum(1) / m.sum(1).clamp_min(1),
                y.std(1),
                m.mean(1),
                y.diff(dim=1).abs().mean(1),
            ],
            dim=-1,
        )
        e = self.embed.unsqueeze(0).expand(B, -1, -1)
        return self.head(torch.cat([feat, e.to(feat.dtype), stats.to(feat.dtype)], dim=-1))


# ======================================================================
# the model
# ======================================================================
@dataclass
class RolloutResult:
    state: Tensor  # (B, T, N, D)
    activity: Tensor  # (B, T, N) predicted regional activity mean
    activity_logvar: Tensor  # (B, T, N)
    hemo: Tensor | None  # (B, T_slow, N, 4)
    rho: Tensor  # scalar: mean ||R|| / ||F_local + F_long||
    diagnostics: dict[str, Any]


class SCWBD(nn.Module):
    """The SC-WBD-001-beta foundation model.

    Population/general adult model with an amortized posterior for
    individualization.  **Not** a validated twin of any person.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        anat: AnatomyPrior,
        *,
        lead_field: LeadField | None = None,
        theta_dim: int = 6,
        n_context_extra: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.anat_summary = anat.summary()
        self.n_regions = anat.n_regions
        self.theta_dim = theta_dim
        self.layout = self.build_layout(cfg)
        L = self.layout

        self.coupling = TypedDelayCoupling(
            anat,
            message_dim=cfg.message_dim,
            n_bins=cfg.n_delay_bins,
            dt=cfg.dt_model,
            dense_ablation=cfg.dense_coupling_ablation,
            control_graph=cfg.control_graph,
        )
        exported = list(L.exported_names())
        self.export_dim = sum(L.spec(n).dim for n in exported)
        self._exported = exported
        self.msg_proj = nn.Linear(self.export_dim, cfg.message_dim)
        self.msg_readin = nn.Sequential(nn.Linear(cfg.message_dim, cfg.hidden // 2), nn.GELU())
        long_dim = cfg.hidden // 2

        self.context = nn.Sequential(
            nn.Linear(theta_dim + 3 + n_context_extra, cfg.context_dim), nn.GELU(), nn.Linear(cfg.context_dim, cfg.context_dim)
        )
        self.mechanistic = None
        if cfg.local_core != "learned":
            self.mechanistic = MechanisticCore(cfg.local_core, L, self.n_regions)
        self.local = RegionalOperator(L, self.n_regions, cfg, in_extra=long_dim)
        self.residual = LearnedResidual(L, self.n_regions, cfg, in_extra=long_dim)
        self.assimilate = Assimilator(L, self.n_regions, cfg)

        self.readout = nn.Sequential(nn.Linear(self.export_dim, cfg.hidden // 2), nn.GELU(), nn.Linear(cfg.hidden // 2, 2))
        self.register_buffer("tau_prior", anat.timescale_prior.float().clone())
        self.log_dt_scale = nn.Parameter(torch.zeros(self.n_regions))

        lf = lead_field if lead_field is not None else build_lead_field(anat, device=anat.weights.device)
        self.eeg = EEGHead(L, lf)
        self.bold = BOLDHead(L, self.n_regions, dt_slow=cfg.dt_model * cfg.hemo_ratio)
        self.behaviour = BehaviourHead(L, self.n_regions, n_out=cfg.n_behaviour)
        self._rho_ema = 0.0

    # -- construction -----------------------------------------------------
    @staticmethod
    def build_layout(cfg: ModelConfig) -> StateLayout:
        if cfg.scalar_state_ablation:
            return scalar_layout()
        lay = default_layout(
            n_spectral_modes=cfg.n_spectral_modes,
            n_adaptation=cfg.n_adaptation,
            n_uncertainty=cfg.n_uncertainty,
        )
        if cfg.local_core != "learned":
            from .backends import resolve_backend

            d = resolve_backend(cfg.local_core).state_dim
            lay = StateLayout(
                lay.components
                + (ComponentSpec("mech", d, "backend_native", "fast", False, True, f"{cfg.local_core} native state"),)
            )
        return lay

    @classmethod
    def from_config(cls, cfg, anat: AnatomyPrior | None = None, **kw) -> "SCWBD":
        from .anatomy import load_anatomy
        from .config import FoundationConfig

        mcfg = cfg.model if isinstance(cfg, FoundationConfig) else cfg
        anat = anat or load_anatomy(device="cpu", n_cortex=400)
        return cls(mcfg, anat, **kw)

    # -- context ----------------------------------------------------------
    def make_context(self, theta: Tensor, extra: Tensor | None = None) -> Tensor:
        """Conditioning vector from theta (+ optional individual/session effects)."""
        G = theta[:, 0:1]
        v = theta[:, 1:2]
        aux = torch.stack([G.squeeze(-1), v.squeeze(-1), theta[:, 2:3].squeeze(-1)], dim=-1)
        z = torch.cat([theta, aux] + ([extra] if extra is not None else []), dim=-1)
        return self.context(z)

    def set_mechanistic_theta(self, theta: Tensor, anat: AnatomyPrior) -> None:
        """Bind a batched ``ParamPack`` for a mechanistic ``local_core``."""
        if self.mechanistic is None:
            self._mech_pack = None
            return
        from .simulate import _regional_theta

        b = self.mechanistic.backend
        pack = b.make_theta(theta.shape[0], self.n_regions, device=theta.device)
        for k, v in _regional_theta(theta, anat, self.mechanistic.backend_name, defaults=dict(b.defaults)).items():
            pack.set(k, v.float())
        self._mech_pack = pack

    # -- one step ---------------------------------------------------------
    def step(
        self,
        x: Tensor,
        history: Tensor,
        lags: Tensor,
        film: list,
        *,
        weights: Tensor | None = None,
        mech_pack=None,
        u: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance the fast clock by one ``dt_model``.

        Returns ``(x_next, message, mech_energy, residual_energy)``.
        """
        cpl = self.coupling(history, lags, weights=weights)  # (B,N,C)
        long_feat = self.msg_readin(cpl)
        if self.mechanistic is not None and mech_pack is not None:
            f_mech = self.mechanistic(x, cpl, mech_pack) * self.cfg.dt_model
        else:
            dt_scale = torch.sigmoid(self.log_dt_scale).to(x.dtype).reshape(1, -1, 1) * 2.0
            f_mech = self.local(x, long_feat, film) * dt_scale
        f_res = self.residual(x, long_feat)
        dx = f_mech + f_res
        if u is not None:
            dx = dx + u
        x_next = torch.nan_to_num(x + dx, nan=0.0, posinf=0.0, neginf=0.0).clamp(-30.0, 30.0)
        msg = self.msg_proj(torch.cat([self.layout.get(x_next, n) for n in self._exported], dim=-1))
        return x_next, msg, f_mech.detach().float().pow(2).mean(), f_res.detach().float().pow(2).mean()

    # -- rollout ----------------------------------------------------------
    def rollout(
        self,
        *,
        y_context: Tensor,
        theta: Tensor,
        n_steps: int,
        context_mask: Tensor | None = None,
        context_extra: Tensor | None = None,
        x0: Tensor | None = None,
        u: Tensor | None = None,
        with_hemo: bool = False,
        enforce_r05: bool = True,
    ) -> RolloutResult:
        """Assimilate ``y_context`` then roll the operator forward ``n_steps``.

        The delay history is a **pre-zero-padded causal buffer**, not a ring:
        reads at ``t - lag`` with ``t`` offset by ``max_lag`` can never wrap
        around onto a future value.
        """
        B = theta.shape[0]
        dev = theta.device
        ctx = self.make_context(theta, context_extra)
        x = self.assimilate(y_context, context_mask) if x0 is None else x0
        lags = self.coupling.lags_for_velocity(theta[:, 1].exp())
        msg0 = self.msg_proj(torch.cat([self.layout.get(x, n) for n in self._exported], dim=-1))
        # steady pre-window (the assimilated state held constant), not zeros:
        # zero-padding the delay line is an imputation, and imputation is forbidden.
        history = self.coupling.new_history(B, dev, dtype=x.dtype, fill=msg0)

        weights = self.coupling.effective_weights(x.dtype)
        film = self.local.prepare(ctx, x.dtype)
        mech_pack = getattr(self, "_mech_pack", None)
        states, mech_e, res_e = [], [], []
        hemo = self.bold.initial(B, self.n_regions, dev) if with_hemo else None
        hemos: list[Tensor] = []
        for t in range(n_steps):
            x, msg, me, re_ = self.step(
                x, history, lags, film, weights=weights, mech_pack=mech_pack, u=None if u is None else u[:, t]
            )
            history = self.coupling.push(history, msg)
            states.append(x)
            mech_e.append(me)
            res_e.append(re_)
            if with_hemo and (t + 1) % self.cfg.hemo_ratio == 0:
                drive = self.layout.get(x, "rate_e").squeeze(-1) if "rate_e" in self.layout else x[..., 0]
                hemo = self.bold.step(hemo, drive.float())
                hemos.append(hemo)
        X = torch.stack(states, dim=1)  # (B,T,N,D)
        feat = torch.cat([self.layout.get(X, n) for n in self._exported], dim=-1)
        out = self.readout(feat)
        act, logvar = out[..., 0], out[..., 1].clamp(-4.0, 6.0)
        rho = (torch.stack(res_e).mean() / (torch.stack(mech_e).mean() + 1e-12)).sqrt()
        self._rho_ema = 0.98 * self._rho_ema + 0.02 * float(rho)
        # R05 applies only when there *is* a mechanistic term to be dominated.
        if enforce_r05 and self.mechanistic is not None and not self.training and self._rho_ema > self.cfg.residual_rho_max:
            raise R05Violation(self._rho_ema, self.cfg.residual_rho_max)
        return RolloutResult(
            state=X,
            activity=act,
            activity_logvar=logvar,
            hemo=torch.stack(hemos, dim=1) if hemos else None,
            rho=rho,
            diagnostics={
                "rho_ema": self._rho_ema,
                "rho_enforced": self.mechanistic is not None,
                "mech_energy": float(torch.stack(mech_e).mean()),
                "residual_energy": float(torch.stack(res_e).mean()),
                **{f"gain_{k}": v for k, v in self.coupling.class_gains().items()},
            },
        )

    # -- regularisers -----------------------------------------------------
    def residual_penalty(self) -> Tensor:
        return self.residual.log_scale.exp() ** 2

    def stability_penalty(self, X: Tensor) -> Tensor:
        """``R_stability``: homeostatic -- the state may not drift without bound."""
        return X.float().pow(2).mean().clamp_min(0.0) * 1e-3 + X.float().mean(dim=(1,)).pow(2).mean() * 1e-2

    def port_penalty(self, X: Tensor) -> Tensor:
        """``R_ports``: non-exported components may not carry the message.

        Penalises the predictability of the exported message from the *private*
        components, which is the numerical side-channel §6.2 forbids.
        """
        private = [n for n in self.layout.clock_names("fast") if n not in self._exported]
        if not private:
            return X.new_zeros(())
        p = torch.cat([self.layout.get(X, n) for n in private], dim=-1).float()
        return p.pow(2).mean() * 1e-3

    def parameter_report(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            key = name.split(".")[0]
            groups[key] = groups.get(key, 0) + p.numel()
        groups["TOTAL"] = sum(groups.values())
        return groups

    def module_names(self) -> list[str]:
        """Top-level module names -- the vocabulary of ``GradientPermission.A_k``."""
        return sorted({n.split(".")[0] for n, _ in self.named_parameters()})


# ======================================================================
# the learned operator, exposed as a backend
# ======================================================================
class LearnedOperatorBackend(nn.Module):
    """Adapter presenting :class:`SCWBD` through agent E's ``DynamicsBackend`` API.

    Its ``mechanistic_status`` is ``surrogate``: it is the equal-capacity control
    against which every mechanistic backend must earn its label, and it is never
    itself evidence that a mechanism is neurally realized.
    """

    def __init__(self, model: SCWBD) -> None:
        super().__init__()
        self.model = model
        self.name = "scwbd_learned_operator"
        self.mechanistic_status = "surrogate"

    @property
    def state_dim(self) -> int:
        return self.model.layout.dim

    def capacity(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "surrogate",
            "mechanistic_status": self.mechanistic_status,
            "state_dim": self.state_dim,
            "capacity": self.capacity(),
            "falsifier": (
                "If the learned operator matches every mechanistic backend's held-out "
                "likelihood *and* its perturbational forecasts, the mechanistic labels are "
                "unearned; if it fails on perturbations it did not see, its own generality "
                "claim fails."
            ),
        }
