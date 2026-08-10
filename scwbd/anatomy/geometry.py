"""Cortical geometry: meshes, geodesics, Laplacians, local kernels, adjacency.

This module supplies the ``M`` of thesis §4.1 -- the local mesh on which
``F_local`` acts -- and the inter-parcel distances that set conduction delays
in :mod:`scwbd.anatomy.connectome`.

What is exact and what is not
-----------------------------
* **Vertex areas, adjacency, cotangent Laplacian** are exact functions of the
  supplied mesh.
* **Geodesic distance** is computed with the heat method (Crane et al. 2013)
  when ``potpourri3d`` is importable and by Dijkstra over mesh edges otherwise.
  Dijkstra on a triangulation *overestimates* true geodesic distance (paths are
  constrained to edges); on these meshes the bias is a few per cent and is
  recorded in the returned ledger rather than corrected away.
* **Cross-hemispheric distance is not geodesic.** The two surfaces are separate
  manifolds; there is no path between them on the mesh. Those entries fall back
  to Euclidean and are flagged in ``method_matrix``, never silently mixed.
* **Euclidean centroid distance is not tract length.** A callosal fibre does
  not travel in a straight line. The tortuosity prior in
  :mod:`scwbd.anatomy.connectome` carries that gap explicitly.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from ._compat import UncertaintyLedger, externally_bounded_ledger, group_average_ledger
from .atlases import Parcellation, _enigma_dir, _read_gii_surface, _vertex_areas
from .paths import derived_dir

__all__ = [
    "Surface",
    "load_surface",
    "SurfaceGeometry",
    "ParcelGeometry",
    "parcel_geometry",
    "ParcelOrientation",
    "parcel_orientation",
    "geodesic_from_sources",
    "GEODESIC_BACKEND",
]


def _geodesic_backend() -> str:
    try:
        import potpourri3d  # noqa: F401

        return "potpourri3d_heat_method"
    except Exception:  # noqa: BLE001
        return "dijkstra_mesh_edges"


GEODESIC_BACKEND = _geodesic_backend()


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Surface:
    """A triangulated cortical surface of one hemisphere."""

    hemi: str
    space: str
    density: str
    kind: str  # "midthickness", "inflated", "sphere", "white", "pial"
    coords: np.ndarray  # (n_vert, 3) float64, millimetres
    faces: np.ndarray  # (n_face, 3) int64

    @property
    def n_vertices(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    # -- differential operators -----------------------------------------
    def vertex_areas(self) -> np.ndarray:
        """Barycentric vertex areas, mm^2."""
        return _vertex_areas(self.coords, self.faces)

    def adjacency(self) -> sp.csr_matrix:
        """Binary vertex adjacency from mesh edges (symmetric, zero diagonal)."""
        f = self.faces
        i = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        j = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        n = self.n_vertices
        a = sp.coo_matrix((np.ones(i.size), (i, j)), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(np.float64)
        a.setdiag(0.0)
        a.eliminate_zeros()
        return a.tocsr()

    def edge_length_graph(self) -> sp.csr_matrix:
        """Sparse graph whose weights are Euclidean edge lengths, mm."""
        a = self.adjacency().tocoo()
        d = np.linalg.norm(self.coords[a.row] - self.coords[a.col], axis=1)
        return sp.coo_matrix((d, (a.row, a.col)), shape=a.shape).tocsr()

    def cotangent_laplacian(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Cotangent Laplacian ``L`` and lumped mass ``M`` (vertex areas).

        Returns ``(L, M)`` with the convention that the Laplace-Beltrami
        operator is ``M^{-1} L`` acting on a vertex signal, ``L`` positive
        semi-definite with zero row sums.  This is the operator agent E's
        cortical-field terms diffuse with.
        """
        v, f = self.coords, self.faces
        n = v.shape[0]
        i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
        e0 = v[i2] - v[i1]
        e1 = v[i0] - v[i2]
        e2 = v[i1] - v[i0]
        cross = np.cross(e1, -e2)
        area2 = np.linalg.norm(cross, axis=1)
        area2 = np.maximum(area2, 1e-12)
        cot0 = -np.einsum("ij,ij->i", e1, e2) / area2
        cot1 = -np.einsum("ij,ij->i", e2, e0) / area2
        cot2 = -np.einsum("ij,ij->i", e0, e1) / area2
        ii = np.concatenate([i1, i2, i2, i0, i0, i1])
        jj = np.concatenate([i2, i1, i0, i2, i1, i0])
        ww = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
        w = sp.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
        lap = sp.diags(np.asarray(w.sum(axis=1)).ravel()) - w
        return lap.tocsr(), self.vertex_areas()

    def local_kernel(
        self,
        sigma_mm: float,
        *,
        truncate: float = 3.0,
        sources: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """Geodesic Gaussian neighbourhood kernel, row-normalised.

        The local operator of thesis §4.1: a sparse ``(n_src, n_vert)`` matrix
        whose row *i* is ``exp(-d_g(i, .)^2 / 2 sigma^2)`` truncated at
        ``truncate * sigma`` and normalised to sum to one.  Locality is what
        preserves retinotopy, cortical distance and travelling waves; a dense
        kernel would not.
        """
        if sigma_mm <= 0:
            raise ValueError("sigma_mm must be positive")
        src = np.arange(self.n_vertices) if sources is None else np.asarray(sources)
        cutoff = truncate * sigma_mm
        graph = self.edge_length_graph()
        from scipy.sparse.csgraph import dijkstra

        rows, cols, vals = [], [], []
        chunk = 512
        for s in range(0, src.size, chunk):
            idx = src[s : s + chunk]
            d = dijkstra(graph, directed=False, indices=idx, limit=cutoff)
            m = np.isfinite(d)
            r, c = np.nonzero(m)
            rows.append(r + s)
            cols.append(c)
            vals.append(np.exp(-0.5 * (d[r, c] / sigma_mm) ** 2))
        k = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(src.size, self.n_vertices),
        ).tocsr()
        rs = np.asarray(k.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        return sp.diags(1.0 / rs) @ k


_SURFACE_FILES: dict[tuple[str, str, str], dict[str, str]] = {
    ("fsLR", "32k", "midthickness"): {"L": "conte69_32k_lh.gii", "R": "conte69_32k_rh.gii"},
    ("fsLR", "32k", "sphere"): {"L": "conte69_32k_lh_sphere.gii", "R": "conte69_32k_rh_sphere.gii"},
    ("fsaverage5", "10k", "midthickness"): {"L": "fsa5_lh.surf.gii", "R": "fsa5_rh.surf.gii"},
    ("fsaverage5", "10k", "sphere"): {"L": "fsa5_sphere_lh.gii", "R": "fsa5_sphere_rh.gii"},
}


@lru_cache(maxsize=16)
def load_surface(space: str, density: str, kind: str = "midthickness", hemi: str = "L") -> Surface:
    """Load a hemisphere surface mesh from the bundled templates."""
    key = (space, density, kind)
    if key not in _SURFACE_FILES:
        raise ValueError(
            f"no {kind} surface for {space}/{density}; have "
            f"{sorted({(a, b, c) for a, b, c in _SURFACE_FILES})}"
        )
    if hemi not in ("L", "R"):
        raise ValueError("hemi must be 'L' or 'R'")
    coords, faces = _read_gii_surface(_enigma_dir() / "surfaces" / _SURFACE_FILES[key][hemi])
    return Surface(hemi=hemi, space=space, density=density, kind=kind, coords=coords, faces=faces)


# ---------------------------------------------------------------------------
# geodesics
# ---------------------------------------------------------------------------
def geodesic_from_sources(surf: Surface, sources: np.ndarray) -> np.ndarray:
    """Geodesic distance (mm) from each source vertex to every vertex.

    Returns ``(n_sources, n_vertices)``.  Uses the heat method when available,
    else Dijkstra over mesh edges (an upper bound on true geodesic distance).
    """
    sources = np.asarray(sources, dtype=np.int64)
    if GEODESIC_BACKEND == "potpourri3d_heat_method":
        import potpourri3d as pp3d

        solver = pp3d.MeshHeatMethodDistanceSolver(surf.coords, surf.faces)
        return np.stack([solver.compute_distance(int(s)) for s in sources])
    from scipy.sparse.csgraph import dijkstra

    return dijkstra(surf.edge_length_graph(), directed=False, indices=sources)


# ---------------------------------------------------------------------------
# parcel-level geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParcelGeometry:
    """Inter-parcel distances and adjacency for one parcellation.

    Attributes
    ----------
    euclidean_mm
        ``(n, n)`` straight-line distance between parcel centroids in MNI.
        Symmetric, zero diagonal, obeys the triangle inequality by construction.
    geodesic_mm
        ``(n, n)`` distance along the cortical surface within a hemisphere;
        Euclidean fallback where no surface path exists.  **Not** a metric on
        the whole matrix, because two different metrics are spliced -- see
        ``method_matrix``.
    method_matrix
        ``(n, n)`` uint8: 0 = geodesic on the mesh, 1 = Euclidean fallback
        (cross-hemisphere, or an endpoint that has no surface support such as a
        subcortical or cerebellar parcel).
    adjacency
        ``(n, n)`` bool: parcels sharing a surface boundary (surface atlases) or
        a voxel face (volumetric atlases).
    ledger
        Uncertainty ledger for the distances.
    """

    parcellation_name: str
    space: str
    density: str
    labels: np.ndarray
    euclidean_mm: np.ndarray
    geodesic_mm: np.ndarray
    method_matrix: np.ndarray
    adjacency: np.ndarray
    ledger: UncertaintyLedger
    backend: str
    notes: str = ""

    @property
    def n_parcels(self) -> int:
        return int(self.labels.shape[0])

    def fraction_geodesic(self) -> float:
        n = self.n_parcels
        off = ~np.eye(n, dtype=bool)
        return float((self.method_matrix[off] == 0).mean())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=self.labels,
            euclidean_mm=self.euclidean_mm,
            geodesic_mm=self.geodesic_mm,
            method_matrix=self.method_matrix,
            adjacency=self.adjacency,
            _meta=np.array(
                json.dumps(
                    {
                        "parcellation_name": self.parcellation_name,
                        "space": self.space,
                        "density": self.density,
                        "backend": self.backend,
                        "notes": self.notes,
                        "ledger": self.ledger.model_dump(mode="json"),
                    }
                )
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ParcelGeometry":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        return cls(
            parcellation_name=meta["parcellation_name"],
            space=meta["space"],
            density=meta["density"],
            labels=z["labels"],
            euclidean_mm=z["euclidean_mm"],
            geodesic_mm=z["geodesic_mm"],
            method_matrix=z["method_matrix"],
            adjacency=z["adjacency"],
            ledger=UncertaintyLedger.model_validate(meta["ledger"]),
            backend=meta["backend"],
            notes=meta.get("notes", ""),
        )


def _euclidean(centroids: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return 0.5 * (d + d.T)  # kill float asymmetry


def _surface_parcel_geodesic(
    parc: Parcellation, euclid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-over-target-vertices geodesic distance between parcels."""
    n = parc.n_parcels
    geo = euclid.copy()
    method = np.ones((n, n), dtype=np.uint8)
    for hemi in ("L", "R"):
        surf = load_surface(parc.space, parc.density, "midthickness", hemi)
        vl = parc.vertex_labels[hemi]  # type: ignore[index]
        present = np.unique(vl)
        present = present[present >= 0]
        if present.size == 0:
            continue
        # representative vertex per parcel: the one closest to the centroid
        reps = np.empty(present.size, dtype=np.int64)
        for a, k in enumerate(present):
            idx = np.flatnonzero(vl == k)
            reps[a] = idx[
                int(np.argmin(np.linalg.norm(surf.coords[idx] - parc.centroids_mni[k], axis=1)))
            ]
        d = geodesic_from_sources(surf, reps)  # (n_present, n_vert)
        va = surf.vertex_areas()
        block = np.zeros((present.size, present.size))
        for b, k in enumerate(present):
            m = vl == k
            w = va[m]
            block[:, b] = (d[:, m] * w).sum(axis=1) / w.sum()
        block = 0.5 * (block + block.T)
        np.fill_diagonal(block, 0.0)
        ii = np.asarray(present)
        geo[np.ix_(ii, ii)] = block
        method[np.ix_(ii, ii)] = 0
    np.fill_diagonal(method, 0)
    np.fill_diagonal(geo, 0.0)
    return geo, method


def _surface_parcel_adjacency(parc: Parcellation) -> np.ndarray:
    n = parc.n_parcels
    adj = np.zeros((n, n), dtype=bool)
    for hemi in ("L", "R"):
        surf = load_surface(parc.space, parc.density, "midthickness", hemi)
        vl = parc.vertex_labels[hemi]  # type: ignore[index]
        f = surf.faces
        for a, b in ((0, 1), (1, 2), (2, 0)):
            la, lb = vl[f[:, a]], vl[f[:, b]]
            m = (la >= 0) & (lb >= 0) & (la != lb)
            adj[la[m], lb[m]] = True
            adj[lb[m], la[m]] = True
    return adj


def _volume_parcel_adjacency(parc: Parcellation) -> np.ndarray:
    n = parc.n_parcels
    lab = parc.voxel_labels
    assert lab is not None
    adj = np.zeros((n, n), dtype=bool)
    for ax in range(3):
        a = np.take(lab, np.arange(0, lab.shape[ax] - 1), axis=ax)
        b = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        m = (a >= 0) & (b >= 0) & (a != b)
        adj[a[m], b[m]] = True
        adj[b[m], a[m]] = True
    return adj


def parcel_geometry(parc: Parcellation, *, rebuild: bool = False) -> ParcelGeometry:
    """Distances and adjacency for a parcellation, cached under ``assets/derived``."""
    cache = derived_dir("geometry") / f"{parc.name}__{parc.space}-{parc.density}__geom.npz"
    if cache.exists() and not rebuild:
        return ParcelGeometry.load(cache)

    euclid = _euclidean(parc.centroids_mni)
    if parc.is_surface():
        geo, method = _surface_parcel_geodesic(parc, euclid)
        adj = _surface_parcel_adjacency(parc)
        notes = (
            "Geodesic within hemisphere on the midthickness mesh; Euclidean "
            "across the midline because the two hemispheres are separate "
            "manifolds. Vertex coordinates are a group-average template."
        )
    else:
        geo, method = euclid.copy(), np.ones(euclid.shape, dtype=np.uint8)
        np.fill_diagonal(method, 0)
        adj = _volume_parcel_adjacency(parc)
        notes = (
            "Volumetric atlas: no surface exists, so every entry is Euclidean "
            "between centres of gravity. Subcortical and cerebellar 'distance' "
            "is therefore a crude stand-in for the length of a polysynaptic path."
        )

    vox = 1.0 if parc.space != "MNI152" else float(parc.density.rstrip("m").rstrip("m") or 1)
    ledger = externally_bounded_ledger(
        units="mm",
        # template resolution bounds the discretisation error on a centroid;
        # the group-average template itself adds an unbounded registration term
        # which is why the interval is asymmetric and wide on the positive side.
        bias_interval=(-2.0, 8.0),
        external_bound_source=(
            f"template mesh/voxel resolution ({parc.space}/{parc.density}); "
            "inter-subject surface registration residual reported by "
            "Van Essen et al. (2012) for fsLR"
        ),
        variance={"measurement": 1.0, "between_session": 4.0},
        validity_domain={
            "space": parc.space,
            "density": parc.density,
            "geodesic_backend": GEODESIC_BACKEND,
            "forbidden_inference": (
                "This is a template distance. It is not this subject's tract "
                "length and must not be used as one without individual imaging."
            ),
        },
        notes=notes,
    )
    pg = ParcelGeometry(
        parcellation_name=parc.name,
        space=parc.space,
        density=parc.density,
        labels=parc.labels,
        euclidean_mm=euclid.astype(np.float64),
        geodesic_mm=geo.astype(np.float64),
        method_matrix=method,
        adjacency=adj,
        ledger=ledger,
        backend=GEODESIC_BACKEND,
        notes=notes,
    )
    pg.save(cache)
    return pg


@dataclass(frozen=True)
class SurfaceGeometry:
    """Bundle of the local-field operators agent E needs for one hemisphere."""

    surface: Surface
    laplacian: sp.csr_matrix
    mass: np.ndarray
    adjacency: sp.csr_matrix

    @classmethod
    def build(cls, space: str, density: str, hemi: str) -> "SurfaceGeometry":
        s = load_surface(space, density, "midthickness", hemi)
        lap, mass = s.cotangent_laplacian()
        return cls(surface=s, laplacian=lap, mass=mass, adjacency=s.adjacency())


# ---------------------------------------------------------------------------
# parcel orientation: the net dipole, and how much of it folding destroys
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParcelOrientation:
    """Net dipole direction per parcel, with the folding cancellation factor.

    Why this exists
    ---------------
    On **Schaefer400x7**, the 400 cortical parcels of the model's 414 regions, a
    **scalar** per parcel carries 32.1% of the whitened lead field while **three
    numbers per parcel** -- the net dipole moment -- carries 83.4%.  Orientation
    is the largest single win available, by a factor of 2.6, and it is the
    cheapest: 1200 oriented numbers over 400 parcels carry more than 3154 scalars
    over subdivided ones (70.8%).  Every per-parcel field the anatomy prior
    shipped before this was orientation-free, so the prior was supplying the
    representation that measurement says is the weak one.
    (`reports/transforms/resolution_pair_schaefer400.md` §3.1.  The same
    comparison on the 68-parcel Desikan-Killiany atlas gives 5.6% against 51.7%,
    a factor of 9.2; DK-68 is not the parcellation this model runs on.)

    The physics, and why ``coherence`` is the load-bearing number
    ------------------------------------------------------------
    Cortical pyramidal cells are oriented normal to the cortical sheet, so a
    parcel's contribution to the EEG/MEG lead field is the **vector sum** of its
    surface normals weighted by area -- not the sum of their magnitudes.  Cortex
    is folded, so opposing banks of a sulcus inside one parcel point in opposite
    directions and **cancel**.

    ``coherence`` is that cancellation, measured: ``|Σ a_f n_f| / Σ a_f`` over
    the parcel's faces, in ``[0, 1]``.  A parcel of 1 means a flat patch whose
    whole area projects; a parcel of 0.1 has a net dipole ten times weaker than
    its area suggests and is nearly invisible to a sensor array regardless of
    how active it is.  ``effective_area_mm2 = coherence * area_mm2`` is what
    actually reaches a sensor, and it is the quantity a lead field should be
    built from.  Reporting orientation without coherence would be worse than
    reporting neither: a unit vector always looks equally informative.

    Sign convention
    ---------------
    ``sign_convention`` records which way "positive" points, resolved from the
    mesh rather than assumed, and R01 refuses an undeclared handedness.  The
    convention is fixed by majority vote against the hemisphere centroid and the
    achieved fraction is recorded in ``sign_agreement`` -- deep sulcal faces
    genuinely point inward, so that fraction is well below 1 and a value near
    0.5 would mean the mesh winding is inconsistent and the sign is meaningless.

    What this is not
    ----------------
    * Not a subject's orientation.  It is the template mesh's, and the ledger
      says so.
    * Not defined off the surface.  Subcortical and cerebellar parcels have no
      cortical sheet; they are ``nan`` and ``covered=False``, never filled in.
    * Not a *family* property.  Families here are bilateral, so their mean
      normal very nearly cancels by symmetry; a family-level dipole direction
      would be an artefact of that cancellation, not a fact about the family.
      Orientation is per parcel and is deliberately not aggregated.
    """

    parcellation_name: str
    space: str
    density: str
    labels: np.ndarray
    normal: np.ndarray  # (n,3) unit vector; nan where not covered
    coherence: np.ndarray  # (n,) |Σ a n| / Σ a  in [0,1]; nan where not covered
    area_mm2: np.ndarray  # (n,) total surface area of the parcel
    effective_area_mm2: np.ndarray  # (n,) coherence * area_mm2
    covered: np.ndarray  # (n,) bool
    frame: str
    handedness: str
    sign_convention: str
    sign_agreement: float
    ledger: UncertaintyLedger
    notes: str = ""

    @property
    def n_parcels(self) -> int:
        return int(self.labels.shape[0])

    def n_covered(self) -> int:
        return int(self.covered.sum())

    def summary(self) -> dict[str, Any]:
        c = self.coherence[self.covered]
        e = self.effective_area_mm2[self.covered]
        a = self.area_mm2[self.covered]
        return {
            "parcellation": self.parcellation_name,
            "n_parcels": self.n_parcels,
            "n_covered": self.n_covered(),
            "frame": self.frame,
            "handedness": self.handedness,
            "sign_convention": self.sign_convention,
            "sign_agreement": round(float(self.sign_agreement), 4),
            "coherence_median": round(float(np.median(c)), 4) if c.size else None,
            "coherence_min": round(float(c.min()), 4) if c.size else None,
            "coherence_max": round(float(c.max()), 4) if c.size else None,
            "total_area_mm2": round(float(a.sum()), 1) if a.size else None,
            "total_effective_area_mm2": round(float(e.sum()), 1) if e.size else None,
            "area_surviving_folding": (
                round(float(e.sum() / a.sum()), 4) if a.size and a.sum() > 0 else None
            ),
        }


def parcel_orientation(parc: Parcellation, *, kind: str = "midthickness") -> ParcelOrientation:
    """Net dipole orientation and folding coherence for each parcel.

    Computed at **face** level, which is where the geometry actually lives: each
    triangle contributes a vector of magnitude equal to its area, pointing along
    its normal.  A face is assigned to the modal parcel of its three vertices,
    so boundary faces are kept rather than dropped -- discarding them would bias
    ``area_mm2`` low for exactly the small parcels where it matters most.
    """
    n = parc.n_parcels
    normal = np.full((n, 3), np.nan)
    coherence = np.full(n, np.nan)
    area = np.zeros(n)
    covered = np.zeros(n, dtype=bool)

    if not parc.is_surface() or parc.vertex_labels is None:
        raise ValueError(
            f"parcel_orientation requires a surface parcellation; {parc.name!r} is "
            "volumetric. A dipole orientation is a property of the cortical sheet "
            "and does not exist for a volume parcel -- there is no honest value to "
            "return, so this raises rather than emitting nan for every parcel."
        )

    net = np.zeros((n, 3))
    agree_num = 0.0
    agree_den = 0.0
    for hemi in ("L", "R"):
        surf = load_surface(parc.space, parc.density, kind, hemi)
        vl = parc.vertex_labels[hemi]
        v, f = surf.coords, surf.faces
        # Face normal scaled by face area: 0.5 * (b-a) x (c-a).
        fn = 0.5 * np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        fa = np.linalg.norm(fn, axis=1)

        # Sign: orient outward from the hemisphere centroid, by majority vote.
        # Deep sulcal faces genuinely point inward, so agreement is well under 1;
        # near 0.5 would mean inconsistent winding and a meaningless sign.
        centre = v.mean(axis=0)
        fc = v[f].mean(axis=1) - centre
        dot = np.einsum("ij,ij->i", fn, fc)
        agree_num += float((dot > 0).sum())
        agree_den += float(dot.size)
        if (dot > 0).mean() < 0.5:
            fn = -fn

        # Face -> parcel by modal vertex label; faces touching unlabelled
        # vertices (medial wall) are dropped only if the modal label is -1.
        lab3 = vl[f]
        flab = np.where(
            (lab3[:, 0] == lab3[:, 1]) | (lab3[:, 0] == lab3[:, 2]),
            lab3[:, 0],
            np.where(lab3[:, 1] == lab3[:, 2], lab3[:, 1], lab3[:, 0]),
        )
        ok = flab >= 0
        np.add.at(net, flab[ok], fn[ok])
        np.add.at(area, flab[ok], fa[ok])

    covered = area > 0
    mag = np.linalg.norm(net, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal[covered] = net[covered] / np.maximum(mag[covered, None], 1e-12)
        coherence[covered] = mag[covered] / np.maximum(area[covered], 1e-12)
    area_out = np.where(covered, area, np.nan)
    eff = np.where(covered, coherence * np.where(covered, area, 0.0), np.nan)
    sign_agreement = agree_num / max(agree_den, 1.0)

    led = externally_bounded_ledger(
        units="dimensionless",
        bias_interval=(-0.05, 0.05),
        external_bound_source=(
            f"{parc.space}/{parc.density} template mesh vertex spacing; the "
            "orientation of a face is exact given the mesh, so the only bias is "
            "the template's own discretisation of the cortical sheet"
        ),
        variance={"numerical": 1e-12},
        notes=(
            "Group template geometry, not any subject's folding pattern. "
            "Individual sulcal geometry differs and coherence is the quantity "
            "most sensitive to it."
        ),
    )
    return ParcelOrientation(
        parcellation_name=parc.name,
        space=parc.space,
        density=parc.density,
        labels=parc.labels,
        normal=normal,
        coherence=coherence,
        area_mm2=area_out,
        effective_area_mm2=eff,
        covered=covered,
        frame=f"{parc.space}_{parc.density}_surface_RAS",
        handedness="RAS (right-anterior-superior), right-handed",
        sign_convention="outward from the hemisphere centroid (majority vote over faces)",
        sign_agreement=sign_agreement,
        ledger=led,
        notes=(
            f"Net dipole per parcel on the {kind} surface. coherence = "
            "|sum a_f n_f| / sum a_f measures cancellation from cortical folding; "
            "effective_area_mm2 = coherence * area_mm2 is what reaches a sensor."
        ),
    )
