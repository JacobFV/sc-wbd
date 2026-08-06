"""Delayed, block-sparse connectome coupling and the §4.1 field factorization.

Implements

    F(X) = F_local(X; M) + F_long(X; G) + R_theta(X, C)

where ``M`` is the local cortical mesh (agent C's geometry) and ``G`` is the
sparse long-range graph (agent C's connectome prior).  The three terms are kept
*separately addressable* on purpose: the R05 mechanism-dominance guard needs the
mechanistic terms and the learned residual as distinct quantities, and the G2
gate needs to swap ``G`` for dense/randomised/distance-matched controls without
touching anything else.

Design commitments:

* **Edge-list, never dense.** A 400-region connectome at 15% density is ~24k
  edges; densifying it to 160k entries per parameter set to use a matmul is a
  20x waste and it destroys per-edge delay heterogeneity.  Dense coupling exists
  only behind ``allow_dense=True`` because the dense graph is a *required G2
  control*, not a default.
* **Per-edge integer + fractional delay.** Delays come from tract length /
  conduction velocity, and conduction velocity is a batched parameter, so the
  delay index tensor is ``(B, E)`` and history reads are linearly interpolated
  between two taps.
* **Evidence classes get different treatment** (thesis §2.2): ``hard`` edges are
  a fixed mask, ``soft`` edges carry a shrinkage prior, ``proposed`` edges pay a
  complexity + distance + provenance penalty and are admitted only inside a
  declared model comparison.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = [
    "deterministic_scatter",
    "EvidenceClass",
    "EDGE_CLASS_CODES",
    "EdgeSet",
    "DelayBuffer",
    "DelayedConnectome",
    "LocalField",
    "HybridField",
    "EdgePenalty",
    "randomized_control",
    "distance_matched_control",
]

@contextmanager
def deterministic_scatter(enabled: bool = True):
    """Make the sparse scatter bitwise reproducible.

    ``index_add_`` on CUDA accumulates with atomics, so the *order* of the
    additions into a destination region varies between launches and the fp32
    result differs in the last bits.  Over a long rollout those bits amplify:
    two runs with the same seed agree to ~1e-5, not exactly.

    Determinism is a test, not an aspiration (ARCHITECTURE.md §3) — but it costs
    about 4x on the scatter (measured on GB10: 0.41 ms -> 1.58 ms at
    E=24k, N=400, B=64).  So it is explicit and opt-in: run the *reproducibility
    tests* and any run whose bitwise output matters under this context manager,
    and generate bulk training trajectories without it.
    """
    if not enabled:
        yield
        return
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


EvidenceClass = Literal["hard", "soft", "proposed"]
EDGE_CLASS_CODES: dict[str, int] = {"hard": 0, "soft": 1, "proposed": 2}
EDGE_CLASS_NAMES = {v: k for k, v in EDGE_CLASS_CODES.items()}


# ---------------------------------------------------------------------------
# Edge sets
# ---------------------------------------------------------------------------


@dataclass
class EdgeSet:
    """A typed, directed, sparse edge list on device.

    ``weight`` is the anatomical prior strength (dimensionless, normalised);
    ``distance_mm`` is tract length used for delays and for the distance penalty
    on proposed edges.  ``evidence`` holds the compilation class code.
    """

    src: Tensor  # (E,) long
    dst: Tensor  # (E,) long
    weight: Tensor  # (E,) or (B, E) float
    distance_mm: Tensor  # (E,) float
    evidence: Tensor  # (E,) long, values in EDGE_CLASS_CODES
    n_regions: int
    provenance: tuple[str, ...] = ()
    #: per-edge provenance penalty multiplier (1.0 = fully supported source)
    provenance_penalty: Tensor | None = None

    def __post_init__(self) -> None:
        E = self.src.numel()
        for name in ("dst", "distance_mm", "evidence"):
            t = getattr(self, name)
            if t.numel() != E:
                raise ValueError(f"EdgeSet.{name} has {t.numel()} entries, expected {E}")
        if self.weight.shape[-1] != E:
            raise ValueError(f"EdgeSet.weight last dim {self.weight.shape[-1]} != n_edges {E}")
        if E and (int(self.src.max()) >= self.n_regions or int(self.dst.max()) >= self.n_regions):
            raise ValueError("edge endpoint out of range for n_regions")

    @property
    def n_edges(self) -> int:
        return int(self.src.numel())

    @property
    def device(self) -> torch.device:
        return self.src.device

    def to(self, device: str | torch.device) -> "EdgeSet":
        dev = torch.device(device)
        return EdgeSet(
            self.src.to(dev),
            self.dst.to(dev),
            self.weight.to(dev),
            self.distance_mm.to(dev),
            self.evidence.to(dev),
            self.n_regions,
            self.provenance,
            None if self.provenance_penalty is None else self.provenance_penalty.to(dev),
        )

    def mask_classes(self, keep: Iterable[EvidenceClass]) -> "EdgeSet":
        """Return the sub-graph containing only the named evidence classes.

        This is how the G2 controls are built: ``hard``-only, ``hard+soft``, and
        the full graph are different *models*, compared, not merged.
        """
        codes = torch.tensor(
            [EDGE_CLASS_CODES[k] for k in keep], device=self.evidence.device, dtype=self.evidence.dtype
        )
        keep_mask = (self.evidence.unsqueeze(-1) == codes.unsqueeze(0)).any(dim=-1)
        idx = keep_mask.nonzero(as_tuple=True)[0]
        w = self.weight[..., idx]
        return EdgeSet(
            self.src[idx],
            self.dst[idx],
            w,
            self.distance_mm[idx],
            self.evidence[idx],
            self.n_regions,
            self.provenance,
            None if self.provenance_penalty is None else self.provenance_penalty[idx],
        )

    def class_counts(self) -> dict[str, int]:
        return {
            name: int((self.evidence == code).sum()) for name, code in EDGE_CLASS_CODES.items()
        }

    def to_dense(self, *, allow_dense: bool = False, batch: int | None = None) -> Tensor:
        """Materialise ``(N, N)`` / ``(B, N, N)``.  Requires an explicit opt-in.

        Densifying a whole-brain graph is a declared control (G2), not a
        convenience.  Refusing by default is the point.
        """
        if not allow_dense:
            raise RuntimeError(
                "refusing to densify a connectome. The dense graph is a required G2 control, "
                "not a default: pass allow_dense=True and record it as a control in the claim report."
            )
        w = self.weight if self.weight.ndim == 2 else self.weight.unsqueeze(0)
        B = w.shape[0] if batch is None else batch
        out = torch.zeros(B, self.n_regions, self.n_regions, device=self.device, dtype=w.dtype)
        flat = self.dst * self.n_regions + self.src  # row = destination
        out.view(B, -1).index_add_(1, flat, w.expand(B, -1))
        return out if (self.weight.ndim == 2 or batch is not None) else out[0]

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_dense(
        cls,
        weights: Tensor,
        distances_mm: Tensor | None = None,
        *,
        evidence: Tensor | str = "soft",
        threshold: float = 0.0,
        n_regions: int | None = None,
        device: str | torch.device | None = None,
    ) -> "EdgeSet":
        """Build from a dense ``(N, N)`` matrix indexed ``[dst, src]``."""
        dev = default_device(device)
        W = torch.as_tensor(weights, device=dev, dtype=DTYPE)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("weights must be square (N, N) indexed [dst, src]")
        N = n_regions or W.shape[0]
        mask = W.abs() > threshold
        dst, src = mask.nonzero(as_tuple=True)
        w = W[dst, src]
        if distances_mm is None:
            d = torch.full_like(w, float("nan"))
        else:
            D = torch.as_tensor(distances_mm, device=dev, dtype=DTYPE)
            d = D[dst, src]
        if isinstance(evidence, str):
            ev = torch.full_like(src, EDGE_CLASS_CODES[evidence])
        else:
            E = torch.as_tensor(evidence, device=dev)
            ev = E[dst, src].long() if E.ndim == 2 else E.long()
        return cls(src, dst, w, d, ev, N)

    @classmethod
    def from_prior(cls, brain_prior: Any, *, device: str | torch.device | None = None) -> "EdgeSet":
        """Adapt ``scwbd.anatomy.BrainPrior`` (agent C).

        Recognised attributes: ``weights`` (dense or (src,dst,weight) triplet),
        ``distances_mm`` / ``lengths_mm``, ``evidence_class`` (dense codes or
        per-edge strings), ``n_regions``, ``provenance``.  Missing distances are
        left as NaN and *fail loudly* when a delay is requested rather than
        being imputed as zero (rule 1: missing data is never imputed).
        """
        dev = default_device(device)
        get = lambda *names: next(
            (getattr(brain_prior, n) for n in names if getattr(brain_prior, n, None) is not None), None
        )
        w = get("weights", "connectome", "sc")
        if w is None:
            raise TypeError("BrainPrior exposes no weights/connectome/sc attribute")
        d = get("distances_mm", "lengths_mm", "tract_lengths")
        ev = get("evidence_class", "evidence")
        if ev is not None and not torch.is_tensor(ev):
            ev = torch.tensor(
                [EDGE_CLASS_CODES[str(e)] for e in ev], device=dev, dtype=torch.long
            )
        edges = cls.from_dense(
            torch.as_tensor(w, device=dev, dtype=DTYPE),
            None if d is None else torch.as_tensor(d, device=dev, dtype=DTYPE),
            evidence=ev if ev is not None else "soft",
            device=dev,
        )
        prov = getattr(brain_prior, "provenance", None)
        if prov is not None:
            edges.provenance = tuple(prov) if isinstance(prov, (list, tuple)) else (str(prov),)
        return edges

    @classmethod
    def random(
        cls,
        n_regions: int,
        density: float = 0.1,
        *,
        seed: int = 0,
        device: str | torch.device | None = None,
        max_distance_mm: float = 150.0,
        class_fractions: Mapping[str, float] = {"hard": 0.3, "soft": 0.6, "proposed": 0.1},
    ) -> "EdgeSet":
        """A synthetic connectome for tests and benchmarks (never for science)."""
        dev = default_device(device)
        g = make_generator(seed, dev)
        M = torch.rand((n_regions, n_regions), generator=g, device=dev, dtype=DTYPE)
        keep = M < density
        keep.fill_diagonal_(False)
        dst, src = keep.nonzero(as_tuple=True)
        E = src.numel()
        w = torch.rand(E, generator=g, device=dev, dtype=DTYPE) * 0.9 + 0.1
        d = torch.rand(E, generator=g, device=dev, dtype=DTYPE) * max_distance_mm + 10.0
        u = torch.rand(E, generator=g, device=dev, dtype=DTYPE)
        ev = torch.full((E,), EDGE_CLASS_CODES["proposed"], device=dev, dtype=torch.long)
        ev[u < class_fractions["hard"] + class_fractions["soft"]] = EDGE_CLASS_CODES["soft"]
        ev[u < class_fractions["hard"]] = EDGE_CLASS_CODES["hard"]
        return cls(src, dst, w, d, ev, n_regions, provenance=("synthetic",))


def randomized_control(edges: EdgeSet, *, seed: int) -> EdgeSet:
    """G2 control: rewire while preserving edge count and weight distribution."""
    g = make_generator(seed, edges.device)
    E = edges.n_edges
    src = torch.randint(0, edges.n_regions, (E,), generator=g, device=edges.device)
    dst = torch.randint(0, edges.n_regions, (E,), generator=g, device=edges.device)
    bad = src == dst
    dst = torch.where(bad, (dst + 1) % edges.n_regions, dst)
    return EdgeSet(src, dst, edges.weight.clone(), edges.distance_mm.clone(), edges.evidence.clone(),
                   edges.n_regions, edges.provenance + ("randomized_control",))


def distance_matched_control(edges: EdgeSet, positions: Tensor, *, seed: int, n_bins: int = 10) -> EdgeSet:
    """G2 control: rewire within distance bins, preserving the distance profile."""
    g = make_generator(seed, edges.device)
    pos = positions.to(edges.device)
    D = torch.cdist(pos, pos)
    E = edges.n_edges
    d = edges.distance_mm
    finite = torch.isfinite(d)
    if not bool(finite.all()):
        raise ValueError("distance-matched control needs finite tract lengths on every edge")
    qs = torch.quantile(d, torch.linspace(0, 1, n_bins + 1, device=d.device))
    new_src = edges.src.clone()
    new_dst = edges.dst.clone()
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        sel = (d >= lo) & (d <= hi)
        k = int(sel.sum())
        if k == 0:
            continue
        cand = ((D >= lo) & (D <= hi)).nonzero()
        if cand.shape[0] == 0:
            continue
        pick = torch.randint(0, cand.shape[0], (k,), generator=g, device=d.device)
        chosen = cand[pick]
        idx = sel.nonzero(as_tuple=True)[0]
        new_dst[idx] = chosen[:, 0]
        new_src[idx] = chosen[:, 1]
    return EdgeSet(new_src, new_dst, edges.weight.clone(), edges.distance_mm.clone(),
                   edges.evidence.clone(), edges.n_regions,
                   edges.provenance + ("distance_matched_control",))


# ---------------------------------------------------------------------------
# History / delay buffer
# ---------------------------------------------------------------------------


class DelayBuffer:
    """Ring buffer of the coupling variable, with fractional-delay reads.

    Layout ``(B, T, N, C)``.  ``T`` is chosen from the maximum delay in steps.
    Reads never allocate a dense ``N x N`` object; they gather ``E`` values.

    The buffer is **not** an ``nn.Module``: it holds no parameters and its
    contents are part of the simulation state, not the model.
    """

    def __init__(
        self,
        batch: int,
        n_regions: int,
        n_channels: int,
        max_delay_steps: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        fill: float = 0.0,
    ):
        self.device = default_device(device)
        self.dtype = dtype
        self.B, self.N, self.C = int(batch), int(n_regions), int(n_channels)
        self.T = int(max_delay_steps) + 2
        self.buf = torch.full((self.B, self.T, self.N, self.C), float(fill), device=self.device, dtype=dtype)
        self.head = 0  # index of the most recently written slot
        self.n_written = 0

    def reset(self, value: Tensor | float = 0.0) -> None:
        if isinstance(value, Tensor):
            self.buf[:] = value.reshape(self.B, 1, self.N, self.C)
        else:
            self.buf.fill_(float(value))
        self.head = 0
        self.n_written = 0

    def push(self, value: Tensor) -> None:
        """Write the current coupling variable ``(B, N, C)``."""
        if value.shape != (self.B, self.N, self.C):
            raise ValueError(f"push expects {(self.B, self.N, self.C)}, got {tuple(value.shape)}")
        self.head = (self.head + 1) % self.T
        self.buf[:, self.head] = value
        self.n_written += 1

    def current(self) -> Tensor:
        return self.buf[:, self.head]

    def read(self, node_idx: Tensor, delay_steps: Tensor) -> Tensor:
        """Gather delayed values.

        ``node_idx``: ``(E,)`` source regions.
        ``delay_steps``: ``(E,)`` or ``(B, E)`` non-negative float delays in steps.
        Returns ``(B, E, C)`` with linear interpolation between the two
        neighbouring taps (fractional delay).
        """
        d = delay_steps
        if d.ndim == 1:
            d = d.unsqueeze(0)
        d = d.clamp(0.0, float(self.T - 2))
        d0 = torch.floor(d)
        frac = (d - d0).unsqueeze(-1)
        i0 = d0.long()
        i1 = i0 + 1
        flat = self.buf.reshape(self.B, self.T * self.N, self.C)
        base = node_idx.reshape(1, -1)
        t0 = (self.head - i0) % self.T
        t1 = (self.head - i1) % self.T
        idx0 = (t0 * self.N + base).expand(self.B, -1).unsqueeze(-1).expand(-1, -1, self.C)
        idx1 = (t1 * self.N + base).expand(self.B, -1).unsqueeze(-1).expand(-1, -1, self.C)
        v0 = torch.gather(flat, 1, idx0)
        v1 = torch.gather(flat, 1, idx1)
        return v0 * (1.0 - frac) + v1 * frac

    def memory_bytes(self) -> int:
        return self.buf.numel() * self.buf.element_size()


# ---------------------------------------------------------------------------
# Long-range coupling operator
# ---------------------------------------------------------------------------


class DelayedConnectome(nn.Module):
    """``F_long``: delayed, block-sparse, evidence-typed long-range coupling.

    ``mode`` is taken from the backend's ``coupling_kind``:

    ``additive``          ``c_i = sum_j w_ij v_j(t - tau_ij)``
    ``phase_difference``  ``c_i = sum_j w_ij sin(phi_j(t - tau_ij) - phi_i(t) - alpha)``

    Conduction velocity is a *batched parameter*: ``theta['velocity']`` in m/s
    gives per-parameter-set delays ``tau = distance_mm / (1000 * velocity)``.
    """

    def __init__(
        self,
        edges: EdgeSet,
        *,
        mode: str = "additive",
        n_channels: int = 1,
        normalize: str = "none",  # none | row | max
        learn_soft_weights: bool = False,
        edge_channel_gain: Tensor | None = None,
    ):
        super().__init__()
        self.edges = edges
        self.mode = mode
        self.C = int(n_channels)
        self.normalize = normalize
        self.register_buffer("src", edges.src, persistent=False)
        self.register_buffer("dst", edges.dst, persistent=False)
        self.register_buffer("distance_mm", edges.distance_mm, persistent=False)
        self.register_buffer("evidence", edges.evidence, persistent=False)
        w = edges.weight
        if learn_soft_weights:
            # hard edges keep the anatomical value fixed; only soft/proposed edges
            # get a learnable multiplicative deviation (thesis §2.2)
            self.register_buffer("w_base", w, persistent=False)
            self.log_dev = nn.Parameter(torch.zeros_like(w if w.ndim == 1 else w[0]))
            self.register_buffer(
                "learnable_mask", (edges.evidence != EDGE_CLASS_CODES["hard"]).to(w.dtype), persistent=False
            )
        else:
            self.register_buffer("w_base", w, persistent=False)
            self.log_dev = None
            self.register_buffer("learnable_mask", torch.zeros_like(edges.distance_mm), persistent=False)
        self.register_buffer(
            "edge_channel_gain",
            (
                torch.ones(edges.n_edges, self.C, device=edges.device, dtype=DTYPE)
                if edge_channel_gain is None
                else edge_channel_gain.to(edges.device)
            ),
            persistent=False,
        )
        self.n_regions = edges.n_regions
        self._row_sum_cache: Tensor | None = None

    # -- weights -----------------------------------------------------------
    def effective_weights(self, batch: int = 1) -> Tensor:
        w = self.w_base
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if self.log_dev is not None:
            dev = (self.log_dev * self.learnable_mask).exp()
            w = w * dev.unsqueeze(0)
        if self.normalize == "row":
            rs = self.row_sum(w).clamp_min(1e-8)
            w = w / rs.gather(1, self.dst.reshape(1, -1).expand(w.shape[0], -1))
        elif self.normalize == "max":
            w = w / w.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return w.expand(max(batch, w.shape[0]), -1) if w.shape[0] == 1 else w

    def row_sum(self, w: Tensor | None = None) -> Tensor:
        """``(B, N)`` in-strength per destination — needed by diffusive coupling."""
        w = self.effective_weights() if w is None else w
        out = torch.zeros(w.shape[0], self.n_regions, device=w.device, dtype=w.dtype)
        out.index_add_(1, self.dst, w)
        return out

    # -- delays ------------------------------------------------------------
    def delay_steps(self, theta: ParamPack, dt: float) -> Tensor:
        """``(B, E)`` (or ``(1, E)``) delays in integer+fractional steps."""
        if "velocity" not in theta and "delay_s" not in theta:
            raise KeyError(
                "delayed coupling needs theta['velocity'] (m/s) or theta['delay_s'] (s). "
                "Delays are not defaulted: an unstated conduction delay is a modelling claim."
            )
        if "delay_s" in theta:
            tau = theta.get("delay_s").reshape(-1, theta.get("delay_s").shape[-2])
            tau = tau.expand(tau.shape[0], self.edges.n_edges) if tau.shape[1] == 1 else tau
        else:
            v = theta.get("velocity").reshape(-1, 1)[:, :1]  # (B,1) or (1,1) m/s
            d = self.distance_mm.reshape(1, -1)
            if not bool(torch.isfinite(d).all()):
                raise ValueError(
                    "tract lengths contain NaN: cannot derive conduction delays. "
                    "Missing distances are never imputed as zero (ARCHITECTURE.md §7 rule 1)."
                )
            tau = d / (1000.0 * v.clamp_min(1e-6))  # mm -> m
        return tau / float(dt)

    def max_delay_steps(self, theta: ParamPack, dt: float) -> int:
        return int(math.ceil(float(self.delay_steps(theta, dt).max()))) + 1

    def make_buffer(self, batch: int, theta: ParamPack, dt: float, *, fill: float = 0.0, **kw) -> DelayBuffer:
        return DelayBuffer(
            batch, self.n_regions, self.C, self.max_delay_steps(theta, dt),
            device=self.src.device, fill=fill, **kw
        )

    # -- the operator ------------------------------------------------------
    def forward(
        self,
        buffer: DelayBuffer,
        theta: ParamPack,
        dt: float,
        *,
        current: Tensor | None = None,
        delay_steps: Tensor | None = None,
    ) -> Tensor:
        """Compute ``F_long`` -> ``(B, N, C)``.

        ``current`` (the destination's *present* value) is required for
        pairwise-nonlinear coupling (Kuramoto).  ``delay_steps`` may be passed
        in to avoid recomputing it every step.
        """
        d = self.delay_steps(theta, dt) if delay_steps is None else delay_steps
        vals = buffer.read(self.src, d)  # (B, E, C)
        B = vals.shape[0]
        w = self.effective_weights(B)  # (B, E)
        gain = self.edge_channel_gain.unsqueeze(0)  # (1, E, C)
        if self.mode == "phase_difference":
            if current is None:
                raise ValueError("phase-difference coupling needs the current destination phase")
            phi_i = current.gather(1, self.dst.reshape(1, -1, 1).expand(B, -1, self.C))
            alpha = theta.get("alpha", 0.0).reshape(-1, 1, 1)
            contrib = torch.sin(vals - phi_i - alpha) * w.unsqueeze(-1) * gain
        elif self.mode == "additive":
            contrib = vals * w.unsqueeze(-1) * gain
        else:  # pragma: no cover
            raise ValueError(f"unknown coupling mode {self.mode!r}")
        out = torch.zeros(B, self.n_regions, self.C, device=vals.device, dtype=vals.dtype)
        out.index_add_(1, self.dst, contrib)
        return out

    # -- diagnostics -------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "n_edges": self.edges.n_edges,
            "density": self.edges.n_edges / max(self.n_regions**2 - self.n_regions, 1),
            "class_counts": self.edges.class_counts(),
            "mode": self.mode,
            "channels": self.C,
            "learnable_edges": int(self.learnable_mask.sum()),
            "provenance": list(self.edges.provenance),
        }


# ---------------------------------------------------------------------------
# Local cortical field
# ---------------------------------------------------------------------------


class LocalField(nn.Module):
    """``F_local``: mesh Laplacian diffusion + short-range kernel on the sheet.

    ``F_local(X) = -kappa L X + beta (K X - X)``

    where ``L`` is the (graph or cotangent) Laplacian of agent C's cortical mesh
    and ``K`` is a normalised distance kernel over near neighbours.  Both are
    edge lists; nothing dense is formed.  With the default ``beta=0`` this is
    pure graph diffusion, which is the conservative choice: it preserves the
    mean field and cannot inject energy.
    """

    def __init__(
        self,
        lap_src: Tensor,
        lap_dst: Tensor,
        lap_w: Tensor,
        n_nodes: int,
        *,
        kernel: tuple[Tensor, Tensor, Tensor] | None = None,
    ):
        super().__init__()
        self.register_buffer("lap_src", lap_src, persistent=False)
        self.register_buffer("lap_dst", lap_dst, persistent=False)
        self.register_buffer("lap_w", lap_w, persistent=False)
        self.n_nodes = int(n_nodes)
        if kernel is not None:
            ks, kd, kw = kernel
            self.register_buffer("k_src", ks, persistent=False)
            self.register_buffer("k_dst", kd, persistent=False)
            self.register_buffer("k_w", kw, persistent=False)
            self.has_kernel = True
        else:
            self.has_kernel = False

    @classmethod
    def from_positions(
        cls,
        positions: Tensor,
        *,
        k: int = 6,
        sigma_mm: float = 12.0,
        include_kernel: bool = True,
    ) -> "LocalField":
        """Build a k-NN graph Laplacian from region/vertex coordinates (mm).

        This is the stand-in until agent C's geometry module lands; the same
        class consumes a real cotangent Laplacian via :meth:`from_prior`.
        """
        pos = positions
        N = pos.shape[0]
        D = torch.cdist(pos, pos)
        D.fill_diagonal_(float("inf"))
        kk = min(k, N - 1)
        vals, idx = torch.topk(D, kk, dim=-1, largest=False)
        dst = torch.arange(N, device=pos.device).repeat_interleave(kk)
        src = idx.reshape(-1)
        w = torch.exp(-(vals.reshape(-1) ** 2) / (2.0 * sigma_mm**2))
        # Symmetrise: a k-NN graph is directed, and an asymmetric Laplacian does
        # not conserve the field mean — diffusion would silently inject energy.
        src_s = torch.cat([src, dst])
        dst_s = torch.cat([dst, src])
        w_s = torch.cat([w, w]) * 0.5
        kernel = (src_s, dst_s, w_s / w_s.sum()) if include_kernel else None
        return cls(src_s, dst_s, w_s, N, kernel=kernel)

    @classmethod
    def from_prior(cls, brain_prior: Any, **kw) -> "LocalField":
        """Adapt agent C's geometry: ``laplacian`` (sparse) or ``positions``."""
        lap = getattr(brain_prior, "laplacian", None)
        if lap is not None:
            if lap.is_sparse:
                lap = lap.coalesce()
                idx, val = lap.indices(), lap.values()
                return cls(idx[1], idx[0], val, lap.shape[0])
            dst, src = lap.nonzero(as_tuple=True)
            return cls(src, dst, lap[dst, src], lap.shape[0])
        pos = getattr(brain_prior, "positions", None)
        if pos is None:
            pos = getattr(brain_prior, "centroids", None)
        if pos is None:
            raise TypeError("BrainPrior exposes neither a laplacian nor positions/centroids")
        return cls.from_positions(torch.as_tensor(pos), **kw)

    def laplacian_apply(self, x: Tensor) -> Tensor:
        """``(L x)_i = sum_j w_ij (x_i - x_j)`` — batched over (B, N, D)."""
        B, N, D = x.shape
        w = self.lap_w.reshape(1, -1, 1)
        xj = x.index_select(1, self.lap_src)
        xi = x.index_select(1, self.lap_dst)
        contrib = w * (xi - xj)
        out = torch.zeros_like(x)
        out.index_add_(1, self.lap_dst, contrib)
        return out

    def kernel_apply(self, x: Tensor) -> Tensor:
        xj = x.index_select(1, self.k_src) * self.k_w.reshape(1, -1, 1)
        out = torch.zeros_like(x)
        out.index_add_(1, self.k_dst, xj)
        return out

    def forward(self, x: Tensor, theta: ParamPack) -> Tensor:
        kappa = theta.get("kappa_local", 0.0)
        out = -kappa * self.laplacian_apply(x)
        if self.has_kernel:
            beta = theta.get("beta_local", 0.0)
            if bool((beta != 0).any()):
                out = out + beta * (self.kernel_apply(x) - x)
        return out


# ---------------------------------------------------------------------------
# Evidence-class penalties
# ---------------------------------------------------------------------------


@dataclass
class EdgePenalty:
    """Per-evidence-class regularisation (thesis §2.2).

    * ``hard``: no penalty, no learnable deviation — a fixed mask.
    * ``soft``: shrinkage (Gaussian/Laplace) prior on the *deviation* from the
      anatomical prior, so uncertain edges are pulled back towards the prior.
    * ``proposed``: complexity (per-edge cost) + distance (long edges are
      metabolically expensive and a-priori unlikely) + provenance
      (weakly-sourced edges cost more).  Admitted only inside a declared model
      comparison, which is why :meth:`__call__` requires ``in_model_comparison``.
    """

    soft_shrinkage: float = 1.0
    soft_prior: Literal["gaussian", "laplace"] = "gaussian"
    proposed_complexity: float = 1.0
    proposed_distance_mm_scale: float = 50.0
    provenance_weight: float = 1.0

    def __call__(
        self,
        connectome: DelayedConnectome,
        *,
        in_model_comparison: bool = False,
    ) -> dict[str, Tensor]:
        ev = connectome.evidence
        dev = connectome.log_dev
        zero = torch.zeros((), device=ev.device, dtype=DTYPE)
        soft = zero
        if dev is not None:
            soft_mask = (ev == EDGE_CLASS_CODES["soft"]).to(dev.dtype)
            d = dev * soft_mask
            soft = self.soft_shrinkage * (d.pow(2).sum() if self.soft_prior == "gaussian" else d.abs().sum())
        prop_mask = ev == EDGE_CLASS_CODES["proposed"]
        n_prop = int(prop_mask.sum())
        if n_prop and not in_model_comparison:
            raise RuntimeError(
                f"{n_prop} proposed edges are present but this is not a declared model comparison. "
                "Proposed edges are admitted only inside an explicit comparison and pay a "
                "complexity/distance/provenance penalty (thesis §2.2)."
            )
        if n_prop:
            dist = connectome.distance_mm[prop_mask]
            prov = (
                connectome.edges.provenance_penalty[prop_mask]
                if connectome.edges.provenance_penalty is not None
                else torch.ones_like(dist)
            )
            proposed = (
                self.proposed_complexity * n_prop
                + (dist / self.proposed_distance_mm_scale).sum()
                + self.provenance_weight * prov.sum()
            )
        else:
            proposed = zero
        return {"soft": soft, "proposed": proposed, "total": soft + proposed}


# ---------------------------------------------------------------------------
# The §4.1 factorization
# ---------------------------------------------------------------------------


class HybridField(nn.Module):
    """``F(X) = F_local(X;M) + F_long(X;G) + R_theta(X,C)``.

    The three terms are computed separately and returned separately so that the
    R05 mechanism-dominance guard sees ``||R_theta||`` against ``||F_mech||``
    rather than a fused number.  ``guard`` is a
    :class:`scwbd.dynamics.residual.MechanismDominanceGuard`; if it is present
    it *runs on every call*.
    """

    def __init__(
        self,
        backend,
        connectome: DelayedConnectome | None = None,
        local: LocalField | None = None,
        residual: nn.Module | None = None,
        guard=None,
    ):
        super().__init__()
        self.backend = backend
        self.connectome = connectome
        self.local = local
        self.residual = residual
        self.guard = guard

    def coupling_input(
        self,
        x: Tensor,
        buffer: DelayBuffer | None,
        theta: ParamPack,
        dt: float,
        *,
        delay_steps: Tensor | None = None,
    ) -> Tensor:
        if self.connectome is None or buffer is None:
            return self.backend.zero_coupling(x)
        cur = self.backend.coupling_variable(x, theta)
        return self.connectome(buffer, theta, dt, current=cur, delay_steps=delay_steps)

    def forward(
        self,
        x: Tensor,
        buffer: DelayBuffer | None,
        theta: ParamPack,
        dt: float,
        u: Tensor | None = None,
        t: float = 0.0,
        *,
        delay_steps: Tensor | None = None,
        context: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        c = self.coupling_input(x, buffer, theta, dt, delay_steps=delay_steps)
        f_regional = self.backend.drift(x, c, theta, u, t)
        f_local = self.local(x, theta) if self.local is not None else torch.zeros_like(x)
        f_mech = f_regional + f_local
        if self.residual is not None:
            r = self.residual(x, c, theta, context)
            if self.guard is not None:
                self.guard.observe(f_mech, r)
                r = self.guard.apply(f_mech, r)
        else:
            r = torch.zeros_like(x)
        total = f_mech + r
        if return_parts:
            return {
                "total": total,
                "f_regional": f_regional,
                "f_local": f_local,
                "f_mech": f_mech,
                "residual": r,
                "coupling": c,
            }
        return total
