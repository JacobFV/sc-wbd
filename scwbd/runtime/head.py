"""The head model the serving path evaluates a coil pose against.

A head model here is deliberately *small*: the geometry the field solve needs,
the parcellation the network prediction needs, the declared registration chain
that relates the caller's frames to this one, and a ledger.  Agent C owns real
anatomy (``scwbd.anatomy``); this type is the runtime's stable view of it, so
that a downstream consumer's code does not have to change when the anatomy
module lands.

Two things it refuses to do:

* it will not accept cortical geometry without a named frame and a declared
  origin class (``measured`` / ``template`` / ``simulated`` / ``phantom``), and
* it will not silently claim a phantom is a subject.  ``origin="phantom"``
  travels all the way into
  :class:`~scwbd.runtime.provenance.ModelProvenance.notes` and into every
  evaluation's ledger ``validity_domain``.

Claim limits: a head model is not a validated digital twin of a person.  For
SC-WBD-001-beta the only head models that ship are analytic phantoms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from ._compat import UncertaintyLedger
from .frames import FrameChain
from .types import full_ledger

__all__ = ["HeadOrigin", "HeadModel", "spherical_phantom"]

_DT = torch.float64

HeadOrigin = Literal["measured", "template", "simulated", "phantom"]


@dataclass(frozen=True)
class HeadModel:
    """Geometry + parcellation + declared registration chain for one head.

    ``frame`` is the working frame of ``cortex_vertices``, ``cortex_normals``
    and ``centre``.  A coil pose expressed in any other frame must reach this
    one through ``frames``; there is no default identity.
    """

    subject_id: str
    frame: str
    origin: HeadOrigin

    #: Centre of the best-fitting sphere for the conducting volume, metres.
    centre: Tensor
    #: Outer (scalp) radius of that sphere, metres.
    scalp_radius: float
    #: Cortical sample points, ``[N, 3]`` metres in ``frame``.
    cortex_vertices: Tensor
    #: Outward unit normals at those points, ``[N, 3]``.
    cortex_normals: Tensor

    #: Parcel names, and the parcel index of each cortical vertex.
    parcels: tuple[str, ...]
    vertex_parcel: Tensor
    #: Non-negative parcel coupling, ``[P, P]``, zero diagonal.  A *topology
    #: prior*, not a measured effective connectivity (thesis Sec. 2.5).
    connectivity: Tensor

    #: Named target regions -> boolean vertex masks.
    target_regions: Mapping[str, Tensor] = field(default_factory=dict)

    #: The declared registration edges available for this subject/session.
    frames: FrameChain = field(default_factory=FrameChain)

    #: Recorded, and *not consumed* by the analytic spherical backend, whose
    #: primary field is conductivity-independent.  Kept so that a numerical
    #: backend can consume it and so the omission is visible.
    conductivity_prior: Mapping[str, float] = field(default_factory=dict)

    ledger: UncertaintyLedger | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = torch.as_tensor(self.cortex_vertices, dtype=_DT)
        n = torch.as_tensor(self.cortex_normals, dtype=_DT)
        c = torch.as_tensor(self.centre, dtype=_DT).reshape(3)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"cortex_vertices must be [N,3], got {tuple(v.shape)}")
        if n.shape != v.shape:
            raise ValueError("cortex_normals must match cortex_vertices")
        norms = torch.linalg.norm(n, dim=-1)
        if float((norms - 1.0).abs().max()) > 1e-6:
            n = n / norms.clamp_min(1e-30).unsqueeze(-1)
        p = torch.as_tensor(self.vertex_parcel, dtype=torch.long).reshape(-1)
        if p.numel() != v.shape[0]:
            raise ValueError("vertex_parcel must have one entry per vertex")
        if int(p.max()) >= len(self.parcels) or int(p.min()) < 0:
            raise ValueError("vertex_parcel indexes outside the parcel list")
        w = torch.as_tensor(self.connectivity, dtype=_DT)
        if w.shape != (len(self.parcels), len(self.parcels)):
            raise ValueError("connectivity must be [P,P]")
        if float(w.min()) < 0.0:
            raise ValueError("connectivity must be non-negative")
        object.__setattr__(self, "cortex_vertices", v)
        object.__setattr__(self, "cortex_normals", n)
        object.__setattr__(self, "centre", c)
        object.__setattr__(self, "vertex_parcel", p)
        object.__setattr__(self, "connectivity", w - torch.diag(torch.diagonal(w)))
        if self.ledger is None:
            object.__setattr__(self, "ledger", _default_head_ledger(self.origin))

    # -- queries ------------------------------------------------------------
    @property
    def n_vertices(self) -> int:
        return int(self.cortex_vertices.shape[0])

    @property
    def n_parcels(self) -> int:
        return len(self.parcels)

    def region_mask(self, name: str) -> Tensor:
        if name not in self.target_regions:
            raise KeyError(
                f"head model {self.subject_id!r} declares no target region "
                f"{name!r}; declared regions: {sorted(self.target_regions)}"
            )
        return torch.as_tensor(self.target_regions[name], dtype=torch.bool)

    def parcel_matrix(self) -> Tensor:
        """``[P, N]`` row-normalised averaging operator, vertices -> parcels."""
        p = torch.zeros(self.n_parcels, self.n_vertices, dtype=_DT)
        p[self.vertex_parcel, torch.arange(self.n_vertices)] = 1.0
        return p / p.sum(dim=1, keepdim=True).clamp_min(1.0)

    def scalp_distance(self, point: Tensor) -> float:
        """Distance from ``point`` to the fitted scalp sphere, metres.

        Negative inside.  This is a *model* distance on a fitted sphere, not a
        measured coil-to-scalp gap.
        """
        r = float(torch.linalg.norm(torch.as_tensor(point, dtype=_DT).reshape(3) - self.centre))
        return r - self.scalp_radius

    def validity_domain(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "head_origin": self.origin,
            "frame": self.frame,
            "n_vertices": self.n_vertices,
            "n_parcels": self.n_parcels,
            "is_phantom": self.origin in ("phantom", "simulated"),
        }


def _default_head_ledger(origin: HeadOrigin) -> UncertaintyLedger:
    """A ledger that states how little a phantom geometry is worth."""
    geometric_sd_m = 0.005 if origin in ("phantom", "simulated") else 0.002
    return full_ledger(
        units="m",
        measurement=geometric_sd_m**2,
        within_session=0.0,
        between_session=0.0,
        parameter=(0.5 * geometric_sd_m) ** 2,
        model_class=0.0,
        numerical=0.0,
        bias_interval=(-3.0 * geometric_sd_m, 3.0 * geometric_sd_m),
        bias_status="prior_specified_sensitivity",
        validity_domain={
            "head_origin": origin,
            "note": (
                "analytic phantom geometry; the bias interval is a "
                "prior-specified sensitivity range, not a measured bound"
            ),
        },
        notes="default head-geometry ledger",
    )


def spherical_phantom(
    *,
    subject_id: str = "phantom-sphere-001",
    frame: str = "phantom_head_RAS",
    n_vertices: int = 320,
    n_parcels: int = 24,
    scalp_radius: float = 0.092,
    cortex_radius: float = 0.078,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_region: str = "left_dlpfc",
    target_direction: tuple[float, float, float] = (-0.62, 0.72, 0.31),
    target_halfangle_deg: float = 18.0,
    connectivity_decay_m: float = 0.055,
    frames: FrameChain | None = None,
    seed: int = 0,
) -> HeadModel:
    """A concentric-sphere phantom head. **Not a person, and labelled so.**

    Cortical samples are a Fibonacci sphere at ``cortex_radius`` with radial
    normals; parcels are the nearest of ``n_parcels`` Fibonacci seeds; parcel
    coupling is an exponential distance kernel.  This is a fixture for
    exercising the serving path and the refusal paths, and it is the *only*
    head geometry SC-WBD-001-beta ships.
    """
    g = torch.Generator().manual_seed(int(seed))
    c = torch.tensor(centre, dtype=_DT)

    def fibonacci(n: int, radius: float) -> Tensor:
        i = torch.arange(n, dtype=_DT) + 0.5
        phi = torch.acos(1.0 - 2.0 * i / n)
        theta = math.pi * (1.0 + 5.0**0.5) * i
        return radius * torch.stack(
            [
                torch.sin(phi) * torch.cos(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(phi),
            ],
            dim=-1,
        )

    unit = fibonacci(n_vertices, 1.0)
    verts = c + cortex_radius * unit
    normals = unit.clone()

    seeds = fibonacci(n_parcels, 1.0)
    d = torch.cdist(unit, seeds)
    vertex_parcel = torch.argmin(d, dim=1)

    seed_xyz = cortex_radius * seeds
    dist = torch.cdist(seed_xyz, seed_xyz)
    conn = torch.exp(-dist / connectivity_decay_m)
    conn.fill_diagonal_(0.0)
    # a topology prior, not a measurement: jitter it and say so
    conn = conn * (1.0 + 0.05 * torch.randn(conn.shape, generator=g, dtype=_DT)).clamp_min(0.0)
    conn = 0.5 * (conn + conn.T)

    axis = torch.tensor(target_direction, dtype=_DT)
    axis = axis / torch.linalg.norm(axis)
    cos_lim = math.cos(math.radians(target_halfangle_deg))
    mask = (unit @ axis) >= cos_lim
    if not bool(mask.any()):  # pragma: no cover - guard on a degenerate request
        mask[int(torch.argmax(unit @ axis))] = True

    return HeadModel(
        subject_id=subject_id,
        frame=frame,
        origin="phantom",
        centre=c,
        scalp_radius=scalp_radius,
        cortex_vertices=verts,
        cortex_normals=normals,
        parcels=tuple(f"parcel_{i:03d}" for i in range(n_parcels)),
        vertex_parcel=vertex_parcel,
        connectivity=conn,
        target_regions={target_region: mask},
        frames=frames or FrameChain(),
        conductivity_prior={
            "brain_S_per_m": 0.33,
            "csf_S_per_m": 1.79,
            "skull_S_per_m": 0.01,
            "scalp_S_per_m": 0.43,
        },
        notes={
            "geometry": "concentric spheres",
            "claim": (
                "analytic phantom; no subject, no imaging, no registration to "
                "any person"
            ),
        },
    )
