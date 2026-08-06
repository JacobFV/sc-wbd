"""Anatomical prior adapter for the foundation model (agent I side of agent C).

The real object is ``scwbd.anatomy.BrainPrior`` (agent C): Schaefer-400 cortex +
subcortex + cerebellum, a probabilistic connectome, tract lengths, delay priors,
receptor-derived E/I priors, gradient-derived timescale priors, and the G2
control graphs.  This module **prefers that object whenever it can be imported**
and otherwise builds a clearly-labelled synthetic stand-in so the foundation
model can be built and smoke-tested without blocking.

The stand-in is not anatomy.  Its ``provenance`` is
``"synthetic_fallback"`` and :meth:`AnatomyPrior.is_biological` returns False;
:class:`~scwbd.foundation.manifest.ClaimManifest` refuses to record any
anatomical claim for a checkpoint trained on it.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

__all__ = ["AnatomyPrior", "load_anatomy", "EVIDENCE_CLASSES", "CONTROL_GRAPHS"]

EVIDENCE_CLASSES = ("hard", "soft", "proposed")
CONTROL_GRAPHS = ("randomized", "distance_matched", "dense", "local_only", "none")

_DEF_CORTEX = 400
_DEF_SUBCORTEX = 32
_DEF_CEREBELLUM = 22


@dataclass
class AnatomyPrior:
    """Everything the foundation model needs from anatomy, with provenance.

    Shapes: ``positions (N,3)`` mm in a declared frame; ``weights (N,N)``
    dimensionless structural connectivity (row ``i`` = *incoming* to ``i`` from
    ``j``); ``tract_length (N,N)`` mm; ``ei_prior (N,)`` dimensionless E/I ratio;
    ``timescale_prior (N,)`` seconds.
    """

    n_regions: int
    labels: tuple[str, ...]
    positions: Tensor  # (N,3) mm
    hemisphere: Tensor  # (N,) int: 0=L 1=R 2=midline
    division: tuple[str, ...]  # "cortex" | "subcortex" | "cerebellum"
    system: Tensor  # (N,) int network/system id
    weights: Tensor  # (N,N) structural connectivity, weights[i,j]: j -> i
    tract_length: Tensor  # (N,N) mm
    ei_prior: Tensor  # (N,)
    timescale_prior: Tensor  # (N,) seconds
    gradient: Tensor  # (N,) principal functional gradient, z-scored
    evidence_class: Tensor  # (N,N) int index into EVIDENCE_CLASSES, -1 = absent
    frame: str = "MNI152NLin2009cAsym_RAS"
    units_position: str = "mm"
    provenance: str = "synthetic_fallback"
    source_note: str = ""

    # ------------------------------------------------------------------
    def is_biological(self) -> bool:
        """True only when the prior came from a real atlas/connectome."""
        return self.provenance not in ("synthetic_fallback",)

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def to(self, device: str | torch.device) -> "AnatomyPrior":
        d = torch.device(device)
        kw = {}
        for k, v in self.__dict__.items():
            kw[k] = v.to(d) if isinstance(v, Tensor) else v
        return AnatomyPrior(**kw)

    def density(self) -> float:
        n = self.n_regions
        return float((self.weights > 0).sum().item()) / (n * n - n)

    def class_mask(self, cls: str) -> Tensor:
        """Boolean (N,N) mask of edges in one evidence class."""
        if cls not in EVIDENCE_CLASSES:
            raise KeyError(f"unknown evidence class {cls!r}")
        return self.evidence_class == EVIDENCE_CLASSES.index(cls)

    def delay_matrix(self, velocity_m_s: float | Tensor) -> Tensor:
        """Conduction delay in **seconds** from tract length / conduction velocity.

        ``velocity`` in m/s, ``tract_length`` in mm, so delay = mm/1000 / (m/s).
        Broadcasts over a batch of velocities: ``(B,)`` -> ``(B,N,N)``.
        """
        v = torch.as_tensor(velocity_m_s, device=self.tract_length.device, dtype=self.tract_length.dtype)
        if v.ndim == 0:
            return (self.tract_length * 1e-3) / v.clamp_min(1e-3)
        return (self.tract_length * 1e-3).unsqueeze(0) / v.reshape(-1, 1, 1).clamp_min(1e-3)

    # ------------------------------------------------------------------
    def control_graph(self, kind: str, *, seed: int = 0) -> Tensor:
        """G2 control connectomes at matched density (ARCHITECTURE.md §4, G2).

        ``randomized``       degree-preserving-ish rewire (destroys geometry+topology)
        ``distance_matched`` rewire preserving the edge-length distribution
        ``dense``            all-to-all with the same total weight
        ``local_only``       keep only the shortest edges, matched density
        ``none``             the true prior (identity)
        """
        if kind not in CONTROL_GRAPHS:
            raise KeyError(f"unknown control graph {kind!r}; have {CONTROL_GRAPHS}")
        W = self.weights
        n = self.n_regions
        if kind == "none":
            return W.clone()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        eye = torch.eye(n, dtype=torch.bool, device=W.device)
        if kind == "dense":
            out = torch.full_like(W, float(W.sum() / (n * n - n)))
            out[eye] = 0.0
            return out
        present = (W > 0) & (~eye)
        vals = W[present]
        k = int(present.sum().item())
        if kind == "randomized":
            flat = torch.zeros(n * n, device=W.device, dtype=W.dtype)
            cand = torch.randperm(n * n, generator=g).to(W.device)
            cand = cand[~eye.reshape(-1)[cand]][:k]
            flat[cand] = vals[torch.randperm(k, generator=g).to(W.device)]
            return flat.reshape(n, n)
        if kind == "local_only":
            d = self.tract_length.clone()
            d[eye] = float("inf")
            thresh = torch.kthvalue(d.reshape(-1), k).values
            out = torch.zeros_like(W)
            sel = d <= thresh
            out[sel] = float(vals.mean())
            return out
        # distance_matched: bucket by distance decile, permute weights within bucket
        d = self.tract_length
        out = torch.zeros_like(W)
        dl = d[present]
        qs = torch.quantile(dl, torch.linspace(0, 1, 11, device=W.device))
        idx = present.nonzero(as_tuple=False)
        alld = d[~eye]
        allidx = (~eye).nonzero(as_tuple=False)
        for b in range(10):
            lo, hi = qs[b], qs[b + 1]
            in_b = ((dl >= lo) & (dl <= hi)).nonzero(as_tuple=False).squeeze(-1)
            pool = ((alld >= lo) & (alld <= hi)).nonzero(as_tuple=False).squeeze(-1)
            if in_b.numel() == 0 or pool.numel() == 0:
                continue
            pick = pool[torch.randperm(pool.numel(), generator=g).to(W.device)[: in_b.numel()]]
            tgt = allidx[pick]
            out[tgt[:, 0], tgt[:, 1]] = vals[in_b]
        del idx
        return out

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "n_cortex": sum(d == "cortex" for d in self.division),
            "n_subcortex": sum(d == "subcortex" for d in self.division),
            "n_cerebellum": sum(d == "cerebellum" for d in self.division),
            "density": round(self.density(), 4),
            "mean_tract_length_mm": round(float(self.tract_length[self.weights > 0].mean().item()), 2),
            "edges_by_class": {
                c: int((self.evidence_class == i).sum().item()) for i, c in enumerate(EVIDENCE_CLASSES)
            },
            "frame": self.frame,
            "provenance": self.provenance,
            "is_biological": self.is_biological(),
            "source_note": self.source_note,
        }


# ----------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------
def _from_agent_c(obj: Any, device: torch.device) -> AnatomyPrior:
    """Duck-typed adapter onto ``scwbd.anatomy.BrainPrior``."""

    def T(name: str, *alts: str) -> Tensor | None:
        for nm in (name, *alts):
            v = getattr(obj, nm, None)
            if v is not None:
                return torch.as_tensor(v, device=device, dtype=torch.float32)
        return None

    # Verified against the real ``scwbd.anatomy.BrainPrior`` (414 parcels):
    #   p.structural.weights        (N,N) ndarray -- the connectome
    #   p.structural.distance_mm    (N,N) ndarray -- tract lengths, mm
    #   p.centroids_mni             (N,3)
    #   p.ei_ratio_prior()          list[PriorBase], len N   <- METHOD, list
    #   p.timescale_prior()         list[PriorBase], len N   <- METHOD, list
    #   p.coupling_mask(min_class)  (N,N) ndarray            <- METHOD
    #
    # The previous adapter looked for bare ``weights``/``ei_prior`` attributes,
    # found neither, and its AttributeError was swallowed (see load_anatomy).
    # Note this is **not** a rename: the priors are methods returning *lists of
    # distribution objects*, so aligning names alone would hand
    # ``torch.as_tensor`` a list of pydantic models and raise.
    def sub(owner: Any, name: str, *alts: str) -> Tensor | None:
        for nm in (name, *alts):
            v = getattr(owner, nm, None)
            if v is not None:
                return torch.as_tensor(v, device=device, dtype=torch.float32)
        return None

    def prior_values(name: str) -> Tensor | None:
        """A per-parcel prior exposed as a callable returning distributions."""
        f = getattr(obj, name, None)
        if f is None:
            return None
        seq = f() if callable(f) else f
        vals: list[float] = []
        for p in seq:
            m = getattr(p, "mean", None)
            if m is None:
                m = getattr(p, "median", None)
            if callable(m):  # ``mean`` is a method on PriorBase, not a property
                m = m()
            if m is None:
                m = p
            try:
                vals.append(float(m))
            except (TypeError, ValueError):
                return None
        return torch.as_tensor(vals, device=device, dtype=torch.float32) if vals else None

    structural = getattr(obj, "structural", None)
    W = sub(structural, "weights") if structural is not None else None
    if W is None:
        W = T("weights", "connectome", "sc", "structural_connectivity")
    if W is None:
        raise AttributeError(
            "BrainPrior exposes no connectome: tried structural.weights and "
            "weights/connectome/sc/structural_connectivity"
        )
    n = int(W.shape[-1])
    L = sub(structural, "distance_mm", "euclidean_mm") if structural is not None else None
    if L is None:
        L = T("tract_length", "distances", "lengths", "tract_lengths")
    if L is None:
        raise AttributeError(
            "BrainPrior exposes no tract lengths: tried structural.distance_mm/"
            "euclidean_mm and tract_length/distances/lengths/tract_lengths"
        )
    pos = T("centroids_mni", "positions", "centroids", "coords")
    if pos is None:
        pos = torch.zeros(n, 3, device=device)
    # Explicit None checks: ``or`` on a multi-element tensor raises "Boolean
    # value of Tensor with more than one element is ambiguous".
    ei = prior_values("ei_ratio_prior")
    if ei is None:
        ei = T("ei_prior", "ei_ratio", "excitation_inhibition")
    ts = prior_values("timescale_prior")
    if ts is None:
        ts = T("timescales", "tau_prior")
    # A prior that is absent must not silently become a constant: that is how the
    # connectome defect would have survived a rename-only fix, yielding a real
    # ENIGMA connectome with no receptor E/I and no way to tell.
    if ei is None:
        raise AttributeError(
            "BrainPrior exposes no E/I prior: tried ei_ratio_prior() and "
            "ei_prior/ei_ratio/excitation_inhibition"
        )
    if ts is None:
        raise AttributeError(
            "BrainPrior exposes no timescale prior: tried timescale_prior() and "
            "timescales/tau_prior"
        )
    grad = T("gradient", "gradient_prior", "principal_gradient")
    # NB: `x or default` is unusable here -- these are numpy arrays, and
    # `array or list` raises "truth value of an array ... is ambiguous".
    # The same idiom appears three times in this adapter and was unreachable
    # while the connectome lookup failed first.
    _labels = getattr(obj, "labels", None)
    labels = tuple(str(x) for x in _labels) if _labels is not None else tuple(
        f"region_{i:04d}" for i in range(n)
    )
    _division = getattr(obj, "division", None)
    if _division is None:
        _division = getattr(obj, "structure", None)
    division = tuple(str(x) for x in _division) if _division is not None else ("cortex",) * n
    hemi = getattr(obj, "hemisphere", None)
    hemi_t = (
        torch.as_tensor(hemi, device=device, dtype=torch.long)
        if hemi is not None
        else torch.zeros(n, dtype=torch.long, device=device)
    )
    sysid = getattr(obj, "system", None)
    if sysid is None:
        sysid = getattr(obj, "network", None)
    if sysid is None:
        sys_t = torch.zeros(n, dtype=torch.long, device=device)
    else:
        try:
            sys_t = torch.as_tensor(sysid, device=device, dtype=torch.long)
        except TypeError:
            # `network` is an array of system NAMES (strings). Factorise to
            # stable integer codes rather than dropping the grouping.
            order = {k: i for i, k in enumerate(sorted({str(x) for x in sysid}))}
            sys_t = torch.as_tensor(
                [order[str(x)] for x in sysid], device=device, dtype=torch.long
            )
    ec = getattr(obj, "evidence_class", None)
    if ec is None:
        ec = _classify_edges(W, L)
    else:
        ec = torch.as_tensor(ec, device=device, dtype=torch.long)
    return AnatomyPrior(
        n_regions=n,
        labels=labels,
        positions=pos,
        hemisphere=hemi_t,
        division=division,
        system=sys_t,
        weights=W,
        tract_length=L,
        ei_prior=ei if ei is not None else torch.ones(n, device=device),
        timescale_prior=ts if ts is not None else torch.full((n,), 0.05, device=device),
        gradient=grad if grad is not None else torch.zeros(n, device=device),
        evidence_class=ec,
        frame=str(getattr(obj, "frame", "MNI152NLin2009cAsym_RAS")),
        provenance=str(getattr(obj, "provenance", "scwbd.anatomy.BrainPrior")),
        source_note="adapted from scwbd.anatomy.BrainPrior (agent C)",
    )


def _classify_edges(W: Tensor, L: Tensor) -> Tensor:
    """Assign an evidence class per edge (thesis §2.2: a typed edge, not a scalar).

    Heuristic used only for the fallback / for a BrainPrior that does not declare
    classes: strong short edges are treated as *hard* topology, mid-range as
    *soft*, weak long-range as *proposed*.  A real deployment must take these
    from tractography reliability + tracer evidence, and the checkpoint manifest
    records which was used.
    """
    n = W.shape[-1]
    ec = torch.full((n, n), -1, dtype=torch.long, device=W.device)
    present = W > 0
    if present.sum() == 0:
        return ec
    w = W[present]
    d = L[present]
    wq = torch.quantile(w, torch.tensor([0.5, 0.85], device=W.device))
    dq = torch.quantile(d, torch.tensor([0.6], device=W.device))
    cls = torch.full_like(w, 1, dtype=torch.long)  # soft
    cls[(w >= wq[1]) & (d <= dq[0])] = 0  # hard
    cls[(w < wq[0]) & (d > dq[0])] = 2  # proposed
    ec[present] = cls
    return ec


def load_anatomy(
    *,
    device: str | torch.device = "cpu",
    n_cortex: int = _DEF_CORTEX,
    n_subcortex: int = _DEF_SUBCORTEX,
    n_cerebellum: int = _DEF_CEREBELLUM,
    density: float = 0.10,
    seed: int = 20260805,
    allow_fallback: bool = True,
    force_fallback: bool = False,
) -> AnatomyPrior:
    """Return the best available anatomical prior.

    Tries ``scwbd.anatomy`` first (agent C).  Falls back to a synthetic,
    clearly-labelled stand-in when unavailable and ``allow_fallback``.
    """
    dev = torch.device(device)
    if not force_fallback:
        try:
            mod = importlib.import_module("scwbd.anatomy")
            for ctor in ("load_brain_prior", "default_prior", "BrainPrior"):
                fn = getattr(mod, ctor, None)
                if fn is None:
                    continue
                obj = fn() if not isinstance(fn, type) else fn.load()  # type: ignore[attr-defined]
                return _from_agent_c(obj, dev)
        except ModuleNotFoundError:
            # scwbd.anatomy genuinely absent -> the fallback is legitimate.
            if not allow_fallback:
                raise
        except Exception as exc:  # noqa: BLE001
            # The package IS present and something else went wrong -- an API
            # mismatch, a missing prior, a corrupt asset.  REFUSE.
            #
            # This was previously a bare ``except Exception`` that substituted
            # the synthetic prior silently.  It cost SC-WBD-001-beta its entire
            # anatomical content: the adapter looked for ``weights`` where
            # BrainPrior exposes ``structural.weights``, the AttributeError was
            # swallowed, and the model, the corpus and every downstream claim
            # were built on a synthetic ellipsoid.  Every provenance record said
            # so correctly and nothing read them.
            #
            # A degraded path is only honest if reaching it requires a decision.
            raise RuntimeError(
                "scwbd.anatomy is installed but its prior could not be adapted: "
                f"{type(exc).__name__}: {exc}\n"
                "Refusing to substitute the synthetic prior silently -- that would "
                "produce a model with no biological content and a checkpoint that "
                "says so only in a field nobody reads.\n"
                "Fix the adapter, or pass force_fallback=True to state explicitly "
                "that this run is not intended to carry anatomy."
            ) from exc
    if not allow_fallback:
        raise RuntimeError("scwbd.anatomy unavailable and allow_fallback=False")
    return _synthetic_prior(
        n_cortex=n_cortex,
        n_subcortex=n_subcortex,
        n_cerebellum=n_cerebellum,
        density=density,
        seed=seed,
        device=dev,
    )


def _synthetic_prior(
    *, n_cortex: int, n_subcortex: int, n_cerebellum: int, density: float, seed: int, device: torch.device
) -> AnatomyPrior:
    """A geometry-respecting synthetic connectome. **Not anatomy.**

    Built to have the *statistical* features the model must cope with -- an
    exponential distance-decay of connection probability, a homotopic bump, a
    heavy-tailed weight distribution, hemispheric structure, and a smooth
    unimodal-to-transmodal gradient driving E/I and timescale heterogeneity --
    so that architecture and training code can be exercised honestly.  It
    carries no biological information whatsoever.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = n_cortex + n_subcortex + n_cerebellum
    labels: list[str] = []
    division: list[str] = []
    pos = torch.zeros(n, 3)
    hemi = torch.zeros(n, dtype=torch.long)
    system = torch.zeros(n, dtype=torch.long)

    # cortex: points on two half-ellipsoid shells (Fibonacci sphere per hemisphere)
    per_h = n_cortex // 2
    for h in (0, 1):
        idx = torch.arange(per_h, dtype=torch.float64)
        phi = math.pi * (3.0 - math.sqrt(5.0)) * idx
        z = 1 - 2 * (idx + 0.5) / per_h
        r = torch.sqrt(torch.clamp(1 - z * z, min=0))
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        sx, sy, sz = 65.0, 85.0, 60.0
        base = h * per_h
        pos[base : base + per_h, 0] = (x.abs() * sx * (1 if h else -1)).float()
        pos[base : base + per_h, 1] = (y * sy).float()
        pos[base : base + per_h, 2] = (z * sz).float() + 10.0
        hemi[base : base + per_h] = h
        for i in range(per_h):
            labels.append(f"Schaefer400_{'RH' if h else 'LH'}_{i:03d}")
            division.append("cortex")
    if n_cortex % 2:  # odd tail
        labels.append("Schaefer400_LH_extra")
        division.append("cortex")
    # 7-network system id from a smooth angular function
    ang = torch.atan2(pos[:n_cortex, 1], pos[:n_cortex, 2])
    system[:n_cortex] = ((ang + math.pi) / (2 * math.pi) * 7).long().clamp(0, 6)

    # subcortex: near the midline, deep
    off = n_cortex
    for i in range(n_subcortex):
        h = i % 2
        pos[off + i] = torch.tensor(
            [(1 if h else -1) * (8 + 14 * (i / max(n_subcortex - 1, 1))), 6.0 - 24 * (i / max(n_subcortex - 1, 1)), -6.0]
        )
        hemi[off + i] = h
        system[off + i] = 7
        labels.append(f"subcortex_{'R' if h else 'L'}_{i // 2:02d}")
        division.append("subcortex")
    # cerebellum: posterior-inferior
    off = n_cortex + n_subcortex
    for i in range(n_cerebellum):
        h = i % 2
        pos[off + i] = torch.tensor(
            [
                (1 if h else -1) * (10 + 30 * (i / max(n_cerebellum - 1, 1))),
                -60.0 - 20 * (i / max(n_cerebellum - 1, 1)),
                -45.0,
            ]
        )
        hemi[off + i] = h
        system[off + i] = 8
        labels.append(f"cerebellum_{'R' if h else 'L'}_{i // 2:02d}")
        division.append("cerebellum")

    # euclidean distance -> tract length (fibres are not straight lines)
    d = torch.cdist(pos, pos)
    tortuosity = 1.18
    L = d * tortuosity
    L.fill_diagonal_(0.0)

    # homotopic partner bonus: mirror in x
    mirror = pos.clone()
    mirror[:, 0] *= -1
    d_mirror = torch.cdist(pos, mirror)
    homotopic = torch.exp(-d_mirror / 12.0)

    lam = 35.0  # mm, exponential distance decay of connection probability
    p = torch.exp(-d / lam) + 0.55 * homotopic
    p.fill_diagonal_(0.0)
    # subcortex is a hub: boost its probability
    is_sub = torch.tensor([x == "subcortex" for x in division])
    p[is_sub, :] *= 2.2
    p[:, is_sub] *= 2.2
    # cerebellum couples mostly to subcortex (via thalamus) and motor cortex
    is_cb = torch.tensor([x == "cerebellum" for x in division])
    p[is_cb, :] *= 0.45
    p[:, is_cb] *= 0.45
    p[is_cb.unsqueeze(1) & is_sub.unsqueeze(0)] *= 6.0
    p[is_sub.unsqueeze(1) & is_cb.unsqueeze(0)] *= 6.0

    k = int(round(density * (n * n - n)))
    noise = torch.rand(n, n, generator=g)
    score = p * (noise ** 0.35)
    score.fill_diagonal_(0.0)
    thresh = torch.kthvalue(score.reshape(-1), n * n - k).values
    present = score > thresh
    present.fill_diagonal_(False)

    # lognormal weights, distance-modulated
    z = torch.randn(n, n, generator=g)
    W = torch.zeros(n, n)
    lw = -0.5 + 0.85 * z - d / 90.0
    W[present] = torch.exp(lw[present])
    W = W / W.sum(dim=1, keepdim=True).clamp_min(1e-8)  # row-normalised incoming weight

    # principal gradient: smooth unimodal (sensory) -> transmodal (association)
    grad = (pos[:, 1] / 85.0) * 0.8 - (pos[:, 2] / 60.0) * 0.55
    grad = (grad - grad.mean()) / grad.std().clamp_min(1e-6)
    # receptor-derived E/I proxy: E/I rises along the gradient (association cortex
    # has higher E:I in the standard account); subcortex/cerebellum offset.
    ei = 1.0 + 0.22 * grad
    ei[is_sub] = 0.85
    ei[is_cb] = 0.75
    # intrinsic timescale rises along the gradient
    tau = torch.exp(math.log(0.035) + 0.45 * grad).clamp(0.008, 0.35)
    tau[is_sub] = 0.02
    tau[is_cb] = 0.012

    ec = _classify_edges(W, L)
    return AnatomyPrior(
        n_regions=n,
        labels=tuple(labels[:n]),
        positions=pos.to(device),
        hemisphere=hemi.to(device),
        division=tuple(division[:n]),
        system=system.to(device),
        weights=W.to(device),
        tract_length=L.to(device),
        ei_prior=ei.to(device),
        timescale_prior=tau.to(device),
        gradient=grad.to(device),
        evidence_class=ec.to(device),
        frame="synthetic_ellipsoid_RAS",
        provenance="synthetic_fallback",
        source_note=(
            "GEOMETRY-RESPECTING SYNTHETIC CONNECTOME, NOT ANATOMY. Generated by "
            "scwbd.foundation.anatomy._synthetic_prior so the foundation model can be "
            "built and tested before scwbd.anatomy (agent C) lands. Carries no "
            "biological information; no anatomical claim may cite it."
        ),
    )
