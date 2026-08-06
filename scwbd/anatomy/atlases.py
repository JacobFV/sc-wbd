"""Unified parcellation loader.

``load_parcellation(name, space, density) -> Parcellation``

A :class:`Parcellation` is a *labelled support*, not a picture.  It carries the
vertex or voxel assignment, per-parcel centroids in MNI, per-parcel surface
area or volume, hemisphere and network membership, and a full
:class:`Provenance` record naming the template, the version, the software that
produced it, the license and the citation.

Design commitments (thesis §3.1)
--------------------------------
* No atlas is ground truth.  Several parcellations of the same cortex are
  supported side by side and a crosswalk between them is computed rather than
  asserted.
* Medial wall / unassigned vertices are ``-1``.  They are never folded into a
  neighbouring parcel and never imputed (rule §7.1).
* Surface parcellations report ``areas_mm2``; volumetric parcellations report
  ``volumes_mm3``.  The other field is ``nan``, not zero.

Supported names
---------------
Cortex (surface + volume)
    ``Schaefer{N}x{7,17}`` for N in 100..1000 (step 100, plus 800/1000),
    ``DesikanKilliany`` (68), ``Destrieux`` (148), ``Glasser360`` (360),
    ``EconomoKoskinas`` (5 cytoarchitectonic classes).
Subcortex (volume)
    ``TianS1``..``TianS4`` (16/32/50/54), ``Aseg14`` (the 7-per-hemisphere
    FreeSurfer aseg structures that the ENIGMA/HCP connectome actually covers).
Cerebellum (volume)
    ``Buckner7``, ``Buckner17`` (functional), ``SUITAnatom`` (lobular).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import sources as S
from .paths import cache_dir, derived_dir, src_dir

__all__ = [
    "Parcellation",
    "Provenance",
    "load_parcellation",
    "available_parcellations",
    "crosswalk",
    "ATLAS_SPECS",
]

Space = Literal["fsLR", "fsaverage5", "MNI152"]


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Provenance:
    """Where an object came from and under what terms it may be used."""

    atlas_id: str
    version: str
    space: str
    density: str
    template_id: str
    software: str
    source_url: str
    license: str
    citation: str
    notes: str = ""
    built_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        return cls(**d)


# ---------------------------------------------------------------------------
# parcellation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parcellation:
    """A labelled anatomical support.

    Attributes
    ----------
    name
        Canonical atlas name, e.g. ``"Schaefer400x7"``.
    space, density
        Template space and mesh/voxel density, e.g. ``("fsLR", "32k")`` or
        ``("MNI152", "1mm")``.
    labels
        ``(n_parcels,)`` string labels in canonical order.
    hemi
        ``(n_parcels,)`` of ``"L"``/``"R"``/``"M"`` (midline / bilateral).
    network
        ``(n_parcels,)`` network or system membership; ``""`` when the atlas
        does not define one.
    structure
        ``(n_parcels,)`` of ``"cortex"``/``"subcortex"``/``"cerebellum"``.
    centroids_mni
        ``(n_parcels, 3)`` centre of gravity in MNI152 millimetres.  For
        surface atlases this is the mean midthickness vertex coordinate of the
        fsLR-32k mesh, which is aligned to but not identical with MNI152; the
        residual misalignment is recorded in ``provenance.notes``.
    areas_mm2
        ``(n_parcels,)`` midthickness surface area; ``nan`` for volumetric.
    volumes_mm3
        ``(n_parcels,)`` volume; ``nan`` for surface-only.
    vertex_labels
        ``{"L": (n_vert,), "R": (n_vert,)}`` int32 parcel indices, ``-1`` for
        medial wall / unassigned.  ``None`` for volumetric atlases.
    voxel_labels, affine
        ``(i, j, k)`` int32 parcel indices (``-1`` unassigned) and the 4x4
        voxel-to-MNI affine.  ``None`` for surface-only atlases.
    """

    name: str
    space: str
    density: str
    labels: np.ndarray
    hemi: np.ndarray
    network: np.ndarray
    structure: np.ndarray
    centroids_mni: np.ndarray
    areas_mm2: np.ndarray
    volumes_mm3: np.ndarray
    provenance: Provenance
    vertex_labels: dict[str, np.ndarray] | None = None
    voxel_labels: np.ndarray | None = None
    affine: np.ndarray | None = None

    # -- basics ----------------------------------------------------------
    @property
    def n_parcels(self) -> int:
        return int(self.labels.shape[0])

    def __len__(self) -> int:
        return self.n_parcels

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Parcellation({self.name!r}, space={self.space}/{self.density}, "
            f"n={self.n_parcels}, cortex={int((self.structure == 'cortex').sum())}, "
            f"subcortex={int((self.structure == 'subcortex').sum())}, "
            f"cerebellum={int((self.structure == 'cerebellum').sum())})"
        )

    def index(self, label: str) -> int:
        hits = np.flatnonzero(self.labels == label)
        if hits.size != 1:
            raise KeyError(f"{label!r} not a unique label of {self.name}")
        return int(hits[0])

    def is_surface(self) -> bool:
        return self.vertex_labels is not None

    def is_volume(self) -> bool:
        return self.voxel_labels is not None

    def validate(self) -> None:
        """Internal consistency check; raises on violation."""
        n = self.n_parcels
        for nm, arr, shape in (
            ("hemi", self.hemi, (n,)),
            ("network", self.network, (n,)),
            ("structure", self.structure, (n,)),
            ("centroids_mni", self.centroids_mni, (n, 3)),
            ("areas_mm2", self.areas_mm2, (n,)),
            ("volumes_mm3", self.volumes_mm3, (n,)),
        ):
            if tuple(arr.shape) != shape:
                raise ValueError(f"{self.name}.{nm} has shape {arr.shape}, expected {shape}")
        if len(set(self.labels.tolist())) != n:
            raise ValueError(f"{self.name}: duplicate labels")
        bad = set(self.hemi.tolist()) - {"L", "R", "M"}
        if bad:
            raise ValueError(f"{self.name}: bad hemi codes {bad}")
        bad = set(self.structure.tolist()) - {"cortex", "subcortex", "cerebellum"}
        if bad:
            raise ValueError(f"{self.name}: bad structure codes {bad}")
        if not np.isfinite(self.centroids_mni).all():
            raise ValueError(f"{self.name}: non-finite centroid")
        if self.vertex_labels is not None:
            for h, v in self.vertex_labels.items():
                seen = np.unique(v)
                seen = seen[seen >= 0]
                if seen.size and (seen.max() >= n or seen.min() < 0):
                    raise ValueError(f"{self.name}: vertex label out of range on {h}")
        if self.voxel_labels is not None:
            seen = np.unique(self.voxel_labels)
            seen = seen[seen >= 0]
            if seen.size and seen.max() >= n:
                raise ValueError(f"{self.name}: voxel label out of range")

    # -- serialisation ---------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "labels": self.labels,
            "hemi": self.hemi,
            "network": self.network,
            "structure": self.structure,
            "centroids_mni": self.centroids_mni,
            "areas_mm2": self.areas_mm2,
            "volumes_mm3": self.volumes_mm3,
            "_meta": np.array(
                json.dumps(
                    {
                        "name": self.name,
                        "space": self.space,
                        "density": self.density,
                        "provenance": self.provenance.to_dict(),
                        "hemis": sorted(self.vertex_labels) if self.vertex_labels else [],
                    }
                )
            ),
        }
        if self.vertex_labels is not None:
            for h, v in self.vertex_labels.items():
                payload[f"vertex_labels_{h}"] = v
        if self.voxel_labels is not None:
            payload["voxel_labels"] = self.voxel_labels
            payload["affine"] = self.affine
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Parcellation":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        vl = None
        if meta["hemis"]:
            vl = {h: z[f"vertex_labels_{h}"] for h in meta["hemis"]}
        return cls(
            name=meta["name"],
            space=meta["space"],
            density=meta["density"],
            labels=z["labels"],
            hemi=z["hemi"],
            network=z["network"],
            structure=z["structure"],
            centroids_mni=z["centroids_mni"],
            areas_mm2=z["areas_mm2"],
            volumes_mm3=z["volumes_mm3"],
            provenance=Provenance.from_dict(meta["provenance"]),
            vertex_labels=vl,
            voxel_labels=z["voxel_labels"] if "voxel_labels" in z.files else None,
            affine=z["affine"] if "affine" in z.files else None,
        )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
#: name -> (structure, default space, default density, builder key)
ATLAS_SPECS: dict[str, dict[str, Any]] = {}


def _register(name: str, **kw: Any) -> None:
    ATLAS_SPECS[name] = kw


for _n in (100, 200, 300, 400, 500, 600, 800, 1000):
    for _net in (7, 17):
        _register(
            f"Schaefer{_n}x{_net}",
            builder="schaefer",
            n=_n,
            networks=_net,
            structure="cortex",
            spaces={
                # ENIGMA ships fsLR/fsaverage5 vertex labels only for 7-network
                # solutions at 100/200/300/400 (+1000 on fsLR).
                # NOTE: ENIGMA also ships a fsLR-32k vertex map for Schaefer-1000,
                # but parcel 7Networks_RH_Vis_33 has zero vertices in it. A
                # parcellation with an empty parcel is broken, and silently
                # dropping the parcel would renumber every downstream matrix, so
                # that combination is deliberately not offered.
                **(
                    {"fsLR": ["32k"], "fsaverage5": ["10k"]}
                    if (_net == 7 and _n in (100, 200, 300, 400))
                    else {}
                ),
                "MNI152": ["1mm", "2mm"],
            },
        )

_register("DesikanKilliany", builder="dk", structure="cortex",
          spaces={"fsLR": ["32k"], "fsaverage5": ["10k"]})
_register("Glasser360", builder="glasser", structure="cortex",
          spaces={"fsLR": ["32k"], "fsaverage5": ["10k"]})
_register("EconomoKoskinas", builder="economo", structure="cortex",
          spaces={"fsLR": ["32k"], "fsaverage5": ["10k"]})
_register("Destrieux", builder="destrieux", structure="cortex",
          spaces={"fsaverage5": ["10k"]})
for _s in (1, 2, 3, 4):
    _register(f"TianS{_s}", builder="tian", scale=_s, structure="subcortex",
              spaces={"MNI152": ["1mm", "2mm"]})
_register("Aseg14", builder="aseg14", structure="subcortex",
          spaces={"MNI152": ["1mm", "2mm"]})
_register("Buckner7", builder="buckner", k=7, structure="cerebellum",
          spaces={"MNI152": ["1mm", "2mm"]})
_register("Buckner17", builder="buckner", k=17, structure="cerebellum",
          spaces={"MNI152": ["1mm", "2mm"]})
_register("SUITAnatom", builder="suit", structure="cerebellum",
          spaces={"MNI152": ["1mm", "2mm"]})


def available_parcellations() -> dict[str, dict[str, Any]]:
    """Return the registry: name -> {structure, spaces, ...}."""
    return {k: dict(v) for k, v in ATLAS_SPECS.items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _enigma_dir() -> Path:
    """Locate the ENIGMA Toolbox bundled ``datasets`` directory.

    We resolve the path without importing ``enigmatoolbox.datasets``: that
    module imports ``vtk`` at module scope purely for its plotting helpers, and
    we only ever read CSV and GIFTI files off disk.  Requiring a VTK build in
    order to read a text file would be a needless dependency.
    """
    import importlib.util

    spec = importlib.util.find_spec("enigmatoolbox")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "enigmatoolbox is not installed; install it from "
            "https://github.com/MICA-MNI/ENIGMA (pip install <clone>)"
        )
    d = Path(list(spec.submodule_search_locations)[0]) / "datasets"
    if not d.is_dir():  # pragma: no cover
        raise FileNotFoundError(f"enigmatoolbox datasets directory missing: {d}")
    return d


def _enigma_csv(sub: str, name: str) -> np.ndarray:
    import pandas as pd

    return pd.read_csv(_enigma_dir() / sub / name, header=None).values


def _read_gii_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    g = nib.load(str(path))
    coords = np.asarray(g.darrays[0].data, dtype=np.float64)
    faces = np.asarray(g.darrays[1].data, dtype=np.int64)
    return coords, faces


def _vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric (one-third) vertex area from a triangle mesh, in mm^2."""
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    va = np.zeros(coords.shape[0], dtype=np.float64)
    for c in range(3):
        np.add.at(va, faces[:, c], tri / 3.0)
    return va


def _hemi_split(vertex_labels_flat: np.ndarray) -> dict[str, np.ndarray]:
    n = vertex_labels_flat.shape[0]
    if n % 2:
        raise ValueError("expected an even, hemisphere-concatenated label vector")
    half = n // 2
    return {"L": vertex_labels_flat[:half], "R": vertex_labels_flat[half:]}


def _surface_geometry(space: str, density: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Midthickness meshes for the supported surface spaces."""
    d = _enigma_dir() / "surfaces"
    if (space, density) == ("fsLR", "32k"):
        return {
            "L": _read_gii_surface(d / "conte69_32k_lh.gii"),
            "R": _read_gii_surface(d / "conte69_32k_rh.gii"),
        }
    if (space, density) == ("fsaverage5", "10k"):
        return {
            "L": _read_gii_surface(d / "fsa5_lh.surf.gii"),
            "R": _read_gii_surface(d / "fsa5_rh.surf.gii"),
        }
    raise ValueError(f"no midthickness mesh for {space}/{density}")


def _from_vertex_labels(
    *,
    name: str,
    space: str,
    density: str,
    labels: np.ndarray,
    hemi: np.ndarray,
    network: np.ndarray,
    structure: np.ndarray,
    vertex_labels: dict[str, np.ndarray],
    provenance: Provenance,
) -> Parcellation:
    """Compute centroids and areas from a vertex assignment."""
    geom = _surface_geometry(space, density)
    n = labels.shape[0]

    # Upstream vertex maps occasionally leak a few vertices of a left-hemisphere
    # parcel onto the right mesh (and vice versa) as a residual of surface
    # registration.  A parcel whose declared hemisphere is L must not own right
    # -hemisphere vertices, so those vertices are returned to "unassigned"
    # (-1) rather than absorbed.  The count is recorded in the provenance note;
    # it is never silently dropped.
    vertex_labels = {h: np.asarray(v).copy() for h, v in vertex_labels.items()}
    leaked = 0
    for h, vl in vertex_labels.items():
        if h not in ("L", "R"):
            continue
        wrong = np.isin(vl, np.flatnonzero((hemi != h) & (hemi != "M")))
        leaked += int(wrong.sum())
        vl[wrong] = -1
    if leaked:
        provenance = Provenance(
            **{
                **provenance.to_dict(),
                "notes": (
                    provenance.notes
                    + f" Cross-hemispheric label leakage: {leaked} vertices carrying "
                    "a parcel label from the opposite hemisphere were returned to "
                    "unassigned."
                ).strip(),
            }
        )

    cent = np.full((n, 3), np.nan)
    area = np.zeros(n)
    count = np.zeros(n)
    for h, (coords, faces) in geom.items():
        vl = vertex_labels[h]
        if vl.shape[0] != coords.shape[0]:
            raise ValueError(
                f"{name}: {h} vertex labels {vl.shape[0]} != mesh {coords.shape[0]}"
            )
        va = _vertex_areas(coords, faces)
        for k in np.unique(vl):
            if k < 0:
                continue
            m = vl == k
            w = va[m]
            c = np.average(coords[m], axis=0, weights=w)
            if count[k] == 0:
                cent[k] = c * w.sum()
            else:
                cent[k] = cent[k] + c * w.sum()
            area[k] += float(w.sum())
            count[k] += float(w.sum())
    good = count > 0
    cent[good] = cent[good] / count[good, None]
    if not good.all():
        missing = [labels[i] for i in np.flatnonzero(~good)]
        raise ValueError(f"{name}: parcels with no vertices: {missing[:5]}")
    p = Parcellation(
        name=name,
        space=space,
        density=density,
        labels=labels,
        hemi=hemi,
        network=network,
        structure=structure,
        centroids_mni=cent,
        areas_mm2=area,
        volumes_mm3=np.full(n, np.nan),
        provenance=provenance,
        vertex_labels=vertex_labels,
    )
    p.validate()
    return p


def _from_volume(
    *,
    name: str,
    space: str,
    density: str,
    img: Any,
    label_values: np.ndarray,
    labels: np.ndarray,
    hemi: np.ndarray,
    network: np.ndarray,
    structure: np.ndarray,
    provenance: Provenance,
) -> Parcellation:
    """Compute centroids and volumes from a volumetric label image."""
    import nibabel as nib
    from nibabel.processing import resample_to_output

    if density in ("1mm", "2mm"):
        target = 1.0 if density == "1mm" else 2.0
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        if not np.allclose(zooms, target, atol=1e-3):
            img = resample_to_output(img, [target] * 3, order=0)
    data = np.asarray(img.dataobj)
    data = np.rint(data).astype(np.int32)
    affine = np.asarray(img.affine, dtype=np.float64)
    vox_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    n = labels.shape[0]
    # Single pass: build a lookup from raw label value to parcel index, then
    # accumulate counts and coordinate sums with bincount.  The obvious
    # per-parcel `data == value` loop is O(n_parcels * n_voxels) and takes
    # minutes for a 1000-parcel atlas on a 1 mm grid.
    lv = np.asarray(label_values, dtype=np.int64)
    lut = np.full(int(max(lv.max(), data.max())) + 2, -1, dtype=np.int32)
    lut[lv] = np.arange(n, dtype=np.int32)
    flat = data.ravel()
    ok = flat > 0
    idx = np.full(flat.shape, -1, dtype=np.int32)
    idx[ok] = lut[flat[ok]]
    out = idx.reshape(data.shape)

    sel = idx >= 0
    parcel = idx[sel].astype(np.int64)
    counts = np.bincount(parcel, minlength=n)
    empty = [str(labels[k]) for k in np.flatnonzero(counts == 0)]
    if empty:
        raise ValueError(f"{name}: labels absent from the volume: {empty[:5]}")
    ijk = np.stack(np.unravel_index(np.flatnonzero(sel), data.shape), axis=1).astype(np.float64)
    xyz = nib.affines.apply_affine(affine, ijk)
    cent = np.stack(
        [np.bincount(parcel, weights=xyz[:, a], minlength=n) / counts for a in range(3)],
        axis=1,
    )
    vol = counts.astype(np.float64) * vox_mm3
    p = Parcellation(
        name=name,
        space=space,
        density=density,
        labels=labels,
        hemi=hemi,
        network=network,
        structure=structure,
        centroids_mni=cent,
        areas_mm2=np.full(n, np.nan),
        volumes_mm3=vol,
        provenance=provenance,
        voxel_labels=out,
        affine=affine,
    )
    p.validate()
    return p


def _schaefer_meta(raw: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``7Networks_LH_Vis_1`` style labels -> (label, hemi, network)."""
    labels, hemi, net = [], [], []
    for s in raw:
        s = s.strip()
        m = re.match(r"^(\d+Networks)_(LH|RH)_([A-Za-z]+)_?(.*)$", s)
        if m:
            hemi.append("L" if m.group(2) == "LH" else "R")
            net.append(m.group(3))
        else:
            hemi.append("M")
            net.append("")
        labels.append(s)
    return np.array(labels), np.array(hemi), np.array(net)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _build_schaefer(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    n, net = spec["n"], spec["networks"]
    name = f"Schaefer{n}x{net}"
    if space in ("fsLR", "fsaverage5"):
        suffix = "conte69" if space == "fsLR" else "fsa5"
        vl_flat = _enigma_csv("parcellations", f"schaefer_{n}_{suffix}.csv").ravel().astype(np.int32)
        # ENIGMA coding: 0 = medial wall, parcel k <-> label value k+1
        vl_flat = vl_flat - 1
        lab_csv = (
            _enigma_dir() / "matrices/hcp_connectivity" / f"strucLabels_ctx_schaefer_{n}.csv"
        )
        if lab_csv.exists():
            raw = _enigma_csv(
                "matrices/hcp_connectivity", f"strucLabels_ctx_schaefer_{n}.csv"
            ).ravel().tolist()
            raw = [str(x) for x in raw]
        else:
            # ENIGMA ships vertex labels but not the label list at this
            # resolution; the parcel *order* is the CBIG canonical order, so the
            # names from nilearn's volumetric release apply unchanged.
            from nilearn import datasets as _nd

            _a = _nd.fetch_atlas_schaefer_2018(
                n_rois=n, yeo_networks=net, resolution_mm=1,
                data_dir=str(cache_dir() / "nilearn"),
            )
            raw = [x.decode() if isinstance(x, bytes) else str(x) for x in _a["labels"]]
            raw = [s for s in raw if s.strip().lower() not in ("background", "unknown")]
        labels, hemi, network = _schaefer_meta(raw)
        if labels.size != n:
            raise ValueError(f"{name}: {labels.size} labels for n={n}")
        prov = Provenance(
            atlas_id=name,
            version=S.SRC["schaefer2018"]["version"],
            space=space,
            density=density,
            template_id="conte69_32k" if space == "fsLR" else "fsaverage5",
            software=f"enigmatoolbox {S.SRC['enigmatoolbox']['version']}",
            source_url=S.SRC["enigmatoolbox"]["url"],
            license=S.SRC["enigmatoolbox"]["license"],
            citation=S.SRC["schaefer2018"]["citation"],
            notes=(
                "Vertex assignment redistributed by the ENIGMA Toolbox. fsLR "
                "midthickness coordinates are aligned to but not identical with "
                "MNI152; treat centroids as approximate to a few millimetres."
            ),
        )
        return _from_vertex_labels(
            name=name, space=space, density=density, labels=labels, hemi=hemi,
            network=network, structure=np.full(n, "cortex"),
            vertex_labels=_hemi_split(vl_flat), provenance=prov,
        )

    # MNI152 volumetric, from nilearn
    from nilearn import datasets as nd

    res = 1 if density == "1mm" else 2
    a = nd.fetch_atlas_schaefer_2018(
        n_rois=n, yeo_networks=net, resolution_mm=res, data_dir=str(cache_dir() / "nilearn")
    )
    raw = [x.decode() if isinstance(x, bytes) else str(x) for x in a["labels"]]
    # nilearn >= 0.12 prepends a "Background" entry to atlas label lists.
    raw = [s for s in raw if s.strip().lower() not in ("background", "unknown")]
    labels, hemi, network = _schaefer_meta(raw)
    if labels.size != n:
        raise ValueError(f"{name}: nilearn returned {labels.size} labels for n={n}")
    import nibabel as nib

    img = nib.load(a["maps"])
    prov = Provenance(
        atlas_id=name,
        version=S.SRC["schaefer2018"]["version"],
        space="MNI152",
        density=density,
        template_id="MNI152NLin6Asym",
        software=f"nilearn {_nilearn_version()}",
        source_url=S.SRC["schaefer2018"]["url"],
        license=S.SRC["schaefer2018"]["license"],
        citation=S.SRC["schaefer2018"]["citation"],
        notes=(
            "Volumetric projection of a surface-defined parcellation; cortical "
            "ribbon voxels carry partial-volume mixing with white matter and CSF."
        ),
    )
    return _from_volume(
        name=name, space="MNI152", density=density, img=img,
        label_values=np.arange(1, n + 1), labels=labels, hemi=hemi, network=network,
        structure=np.full(n, "cortex"), provenance=prov,
    )


#: ENIGMA maps the 68 Desikan-Killiany parcels onto conte69/fsa5 label values
#: 1..3, 5..38, 40..70 (0 = unknown, 4 = L corpus callosum, 39 = R unknown).
_DK_LABEL_VALUES = np.array(list(range(1, 4)) + list(range(5, 39)) + list(range(40, 71)))


def _build_dk(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    suffix = "conte69" if space == "fsLR" else "fsa5"
    raw_vl = _enigma_csv("parcellations", f"aparc_{suffix}.csv").ravel().astype(np.int32)
    remap = np.full(int(raw_vl.max()) + 1, -1, dtype=np.int32)
    remap[_DK_LABEL_VALUES] = np.arange(_DK_LABEL_VALUES.size, dtype=np.int32)
    vl_flat = remap[raw_vl]
    raw = [str(x) for x in _enigma_csv("matrices/hcp_connectivity", "strucLabels_ctx.csv").ravel()]
    labels = np.array(raw)
    hemi = np.array(["L" if s.startswith("L_") else "R" for s in raw])
    prov = Provenance(
        atlas_id="DesikanKilliany",
        version=S.SRC["desikan2006"]["version"],
        space=space,
        density=density,
        template_id="conte69_32k" if space == "fsLR" else "fsaverage5",
        software=f"enigmatoolbox {S.SRC['enigmatoolbox']['version']}",
        source_url=S.SRC["enigmatoolbox"]["url"],
        license=S.SRC["enigmatoolbox"]["license"],
        citation=S.SRC["desikan2006"]["citation"],
        notes="Gyral-based parcellation; boundaries follow sulcal landmarks, not architecture.",
    )
    return _from_vertex_labels(
        name="DesikanKilliany", space=space, density=density, labels=labels, hemi=hemi,
        network=np.full(68, ""), structure=np.full(68, "cortex"),
        vertex_labels=_hemi_split(vl_flat), provenance=prov,
    )


def _build_glasser(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    suffix = "conte69" if space == "fsLR" else "fsa5"
    vl_flat = _enigma_csv("parcellations", f"glasser_360_{suffix}.csv").ravel().astype(np.int32) - 1
    raw = [
        str(x)
        for x in _enigma_csv(
            "matrices/hcp_connectivity", "strucLabels_ctx_glasser_360.csv"
        ).ravel()
    ]
    labels = np.array(raw)
    hemi = np.array(["L" if s.startswith("L_") else "R" for s in raw])
    prov = Provenance(
        atlas_id="Glasser360",
        version=S.SRC["glasser2016"]["version"],
        space=space,
        density=density,
        template_id="conte69_32k" if space == "fsLR" else "fsaverage5",
        software=f"enigmatoolbox {S.SRC['enigmatoolbox']['version']}",
        source_url=S.SRC["enigmatoolbox"]["url"],
        license=S.SRC["glasser2016"]["license"],
        citation=S.SRC["glasser2016"]["citation"],
        notes=(
            "Multimodal group parcellation. Areal boundaries were defined on a "
            "group average; individual boundaries vary and the atlas does not "
            "resolve that variation."
        ),
    )
    return _from_vertex_labels(
        name="Glasser360", space=space, density=density, labels=labels, hemi=hemi,
        network=np.full(360, ""), structure=np.full(360, "cortex"),
        vertex_labels=_hemi_split(vl_flat), provenance=prov,
    )


def _build_economo(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    suffix = "conte69" if space == "fsLR" else "fsa5"
    raw_vl = _enigma_csv("parcellations", f"economo_koskinas_{suffix}.csv").ravel().astype(np.int32)
    vl_flat = raw_vl - 1  # 0 = unassigned
    labels = np.array(
        ["agranular", "frontal", "parietal", "polar", "granular"], dtype=object
    ).astype(str)
    prov = Provenance(
        atlas_id="EconomoKoskinas",
        version=S.SRC["voneconomo"]["version"],
        space=space,
        density=density,
        template_id="conte69_32k" if space == "fsLR" else "fsaverage5",
        software=f"enigmatoolbox {S.SRC['enigmatoolbox']['version']}",
        source_url=S.SRC["voneconomo"]["url"],
        license=S.SRC["enigmatoolbox"]["license"],
        citation=S.SRC["voneconomo"]["citation"],
        notes=(
            "Five cytoarchitectonic classes of laminar differentiation, digitised "
            "from a single historical atlas; bilateral classes, not parcels."
        ),
    )
    return _from_vertex_labels(
        name="EconomoKoskinas", space=space, density=density, labels=labels,
        hemi=np.full(5, "M"), network=np.full(5, ""), structure=np.full(5, "cortex"),
        vertex_labels=_hemi_split(vl_flat), provenance=prov,
    )


def _build_destrieux(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    from nilearn import datasets as nd

    a = nd.fetch_atlas_surf_destrieux(data_dir=str(cache_dir() / "nilearn"))
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in a["labels"]]
    lh = np.asarray(a["map_left"], dtype=np.int32)
    rh = np.asarray(a["map_right"], dtype=np.int32)
    # index 0 = "Unknown", 42 = "Medial_wall" in the FreeSurfer a2009s LUT
    drop = {i for i, s in enumerate(names) if s.lower() in ("unknown", "medial_wall")}
    keep = [i for i in range(len(names)) if i not in drop]
    labels, hemi = [], []
    remap_l = np.full(len(names), -1, dtype=np.int32)
    remap_r = np.full(len(names), -1, dtype=np.int32)
    k = 0
    for i in keep:
        labels.append(f"L_{names[i]}")
        hemi.append("L")
        remap_l[i] = k
        k += 1
    for i in keep:
        labels.append(f"R_{names[i]}")
        hemi.append("R")
        remap_r[i] = k
        k += 1
    n = k
    vl = {"L": remap_l[lh], "R": remap_r[rh]}
    prov = Provenance(
        atlas_id="Destrieux",
        version=S.SRC["destrieux2010"]["version"],
        space="fsaverage5",
        density="10k",
        template_id="fsaverage5",
        software=f"nilearn {_nilearn_version()}",
        source_url=S.SRC["destrieux2010"]["url"],
        license=S.SRC["destrieux2010"]["license"],
        citation=S.SRC["destrieux2010"]["citation"],
        notes="Sulco-gyral parcellation; boundary definition is anatomical, not functional.",
    )
    return _from_vertex_labels(
        name="Destrieux", space="fsaverage5", density="10k",
        labels=np.array(labels), hemi=np.array(hemi), network=np.full(n, ""),
        structure=np.full(n, "cortex"), vertex_labels=vl, provenance=prov,
    )


def _build_tian(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    import nibabel as nib

    s = spec["scale"]
    base = src_dir() / "tian_subcortex" / "Group-Parcellation" / "3T" / "Subcortex-Only"
    img = nib.load(str(base / f"Tian_Subcortex_S{s}_3T_1mm.nii.gz"))
    names = [
        ln.strip()
        for ln in (base / f"Tian_Subcortex_S{s}_3T_label.txt").read_text().splitlines()
        if ln.strip()
    ]
    n = len(names)
    hemi = np.array(["L" if x.endswith("-lh") else ("R" if x.endswith("-rh") else "M") for x in names])
    prov = Provenance(
        atlas_id=f"TianS{s}",
        version=S.SRC["tian2020"]["version"],
        space="MNI152",
        density=density,
        template_id="MNI152NLin6Asym",
        software="scwbd.anatomy (direct nifti read)",
        source_url=S.SRC["tian2020"]["url"],
        license=S.SRC["tian2020"]["license"],
        citation=S.SRC["tian2020"]["citation"],
        notes=(
            "Gradient-derived functional subdivision of subcortex from 3T rs-fMRI. "
            "No structural-connectome coverage in the ENIGMA/HCP matrices, which "
            "use the coarser FreeSurfer aseg structures."
        ),
    )
    return _from_volume(
        name=f"TianS{s}", space="MNI152", density=density, img=img,
        label_values=np.arange(1, n + 1), labels=np.array(names), hemi=hemi,
        network=np.full(n, ""), structure=np.full(n, "subcortex"), provenance=prov,
    )


#: The 14 FreeSurfer aseg structures the HCP/ENIGMA connectome actually covers,
#: paired with their Harvard-Oxford subcortical atlas region names.
_ASEG14 = [
    ("Laccumb", "Left Accumbens"),
    ("Lamyg", "Left Amygdala"),
    ("Lcaud", "Left Caudate"),
    ("Lhippo", "Left Hippocampus"),
    ("Lpal", "Left Pallidum"),
    ("Lput", "Left Putamen"),
    ("Lthal", "Left Thalamus"),
    ("Raccumb", "Right Accumbens"),
    ("Ramyg", "Right Amygdala"),
    ("Rcaud", "Right Caudate"),
    ("Rhippo", "Right Hippocampus"),
    ("Rpal", "Right Pallidum"),
    ("Rput", "Right Putamen"),
    ("Rthal", "Right Thalamus"),
]


def _build_aseg14(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    from nilearn import datasets as nd

    res = "1mm" if density == "1mm" else "2mm"
    a = nd.fetch_atlas_harvard_oxford(
        f"sub-maxprob-thr25-{res}", data_dir=str(cache_dir() / "nilearn")
    )
    ho_names = [x.decode() if isinstance(x, bytes) else str(x) for x in a["labels"]]
    values, labels, hemi = [], [], []
    for short, ho in _ASEG14:
        cand = [i for i, s in enumerate(ho_names) if s.strip().lower().startswith(ho.lower())]
        if not cand:
            raise ValueError(f"Aseg14: {ho!r} not in Harvard-Oxford subcortical LUT")
        values.append(cand[0])
        labels.append(short)
        hemi.append(short[0])
    prov = Provenance(
        atlas_id="Aseg14",
        version=S.SRC["harvardoxford"]["version"],
        space="MNI152",
        density=density,
        template_id="MNI152NLin6Asym",
        software=f"nilearn {_nilearn_version()}",
        source_url=S.SRC["harvardoxford"]["url"],
        license=S.SRC["harvardoxford"]["license"],
        citation=S.SRC["harvardoxford"]["citation"],
        notes=(
            "Label *names* follow the FreeSurfer aseg structures used by the "
            "ENIGMA/HCP connectome; the *geometry* here is the Harvard-Oxford "
            "maximum-probability subcortical atlas thresholded at 25%. The two "
            "delineations differ; centroids are therefore approximate stand-ins "
            "used only for distance and delay computation."
        ),
    )
    return _from_volume(
        name="Aseg14", space="MNI152", density=density, img=a["maps"] if hasattr(a["maps"], "affine")
        else __import__("nibabel").load(a["maps"]),
        label_values=np.array(values), labels=np.array(labels), hemi=np.array(hemi),
        network=np.full(14, ""), structure=np.full(14, "subcortex"), provenance=prov,
    )


def _build_buckner(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    import nibabel as nib
    import pandas as pd

    k = spec["k"]
    base = src_dir() / "cerebellar_atlases" / "Buckner_2011"
    img = nib.load(str(base / f"atl-Buckner{k}_space-MNI_dseg.nii"))
    tsv = pd.read_csv(base / f"atl-Buckner{k}.tsv", sep="\t")
    idx_col = "index" if "index" in tsv.columns else tsv.columns[0]
    name_col = "name" if "name" in tsv.columns else tsv.columns[1]
    tsv = tsv[tsv[idx_col] > 0]
    prov = Provenance(
        atlas_id=f"Buckner{k}",
        version=S.SRC["buckner2011"]["version"],
        space="MNI152",
        density=density,
        template_id="MNI152NLin6Asym (FNIRT)",
        software="scwbd.anatomy (direct nifti read)",
        source_url=S.SRC["buckner2011"]["url"],
        license=S.SRC["buckner2011"]["license"],
        citation=S.SRC["buckner2011"]["citation"],
        notes=(
            "A functional-connectivity parcellation of cerebellar cortex: parcels "
            "are named for the cerebral network they correlate with, which is not "
            "evidence of a monosynaptic connection. No structural connectome here."
        ),
    )
    names = np.array([f"Cb_{k}net_{str(x).strip()}" for x in tsv[name_col].tolist()])
    return _from_volume(
        name=f"Buckner{k}", space="MNI152", density=density, img=img,
        label_values=tsv[idx_col].to_numpy(), labels=names,
        hemi=np.full(len(names), "M"), network=np.array([str(x).strip() for x in tsv[name_col]]),
        structure=np.full(len(names), "cerebellum"), provenance=prov,
    )


def _build_suit(spec: dict[str, Any], space: str, density: str) -> Parcellation:
    import nibabel as nib
    import pandas as pd

    base = src_dir() / "cerebellar_atlases" / "Diedrichsen_2009"
    img = nib.load(str(base / "atl-Anatom_space-MNI_dseg.nii"))
    tsv = pd.read_csv(base / "atl-Anatom.tsv", sep="\t")
    idx_col = "index" if "index" in tsv.columns else tsv.columns[0]
    name_col = "name" if "name" in tsv.columns else tsv.columns[1]
    tsv = tsv[tsv[idx_col] > 0]
    names = [str(x).strip() for x in tsv[name_col].tolist()]
    hemi = np.array(
        ["L" if n.endswith("_L") or "Left" in n else ("R" if n.endswith("_R") or "Right" in n else "M")
         for n in names]
    )
    prov = Provenance(
        atlas_id="SUITAnatom",
        version=S.SRC["diedrichsen2009"]["version"],
        space="MNI152",
        density=density,
        template_id="MNI152NLin6Asym (FNIRT)",
        software="scwbd.anatomy (direct nifti read)",
        source_url=S.SRC["diedrichsen2009"]["url"],
        license=S.SRC["diedrichsen2009"]["license"],
        citation=S.SRC["diedrichsen2009"]["citation"],
        notes="Probabilistic lobular anatomy of the cerebellum, maximum-probability labelling.",
    )
    return _from_volume(
        name="SUITAnatom", space="MNI152", density=density, img=img,
        label_values=tsv[idx_col].to_numpy(), labels=np.array(names), hemi=hemi,
        network=np.full(len(names), ""), structure=np.full(len(names), "cerebellum"),
        provenance=prov,
    )


_BUILDERS = {
    "schaefer": _build_schaefer,
    "dk": _build_dk,
    "glasser": _build_glasser,
    "economo": _build_economo,
    "destrieux": _build_destrieux,
    "tian": _build_tian,
    "aseg14": _build_aseg14,
    "buckner": _build_buckner,
    "suit": _build_suit,
}


def _nilearn_version() -> str:
    try:
        import nilearn

        return nilearn.__version__
    except Exception:  # noqa: BLE001  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def _default_space(spec: dict[str, Any]) -> tuple[str, str]:
    spaces: dict[str, list[str]] = spec["spaces"]
    for pref in ("fsLR", "fsaverage5", "MNI152"):
        if pref in spaces:
            return pref, spaces[pref][0]
    raise ValueError("atlas has no declared space")


@lru_cache(maxsize=64)
def _load_cached(name: str, space: str, density: str, rebuild: bool) -> Parcellation:
    cache = derived_dir("parcellations") / f"{name}__{space}-{density}.npz"
    if cache.exists() and not rebuild:
        return Parcellation.load(cache)
    spec = ATLAS_SPECS[name]
    p = _BUILDERS[spec["builder"]](spec, space, density)
    p.save(cache)
    return p


def load_parcellation(
    name: str,
    space: str | None = None,
    density: str | None = None,
    *,
    rebuild: bool = False,
) -> Parcellation:
    """Load a parcellation by canonical name.

    Parameters
    ----------
    name
        One of :func:`available_parcellations`.
    space
        ``"fsLR"``, ``"fsaverage5"`` or ``"MNI152"``.  Defaults to the finest
        surface space the atlas supports, else MNI152.
    density
        Mesh density (``"32k"``, ``"10k"``) or voxel size (``"1mm"``, ``"2mm"``).
    rebuild
        Ignore the cached ``.npz`` and rebuild from upstream.

    Raises
    ------
    KeyError
        Unknown atlas name.
    ValueError
        The atlas is not defined in the requested space/density.  We refuse to
        resample a parcellation into a space its authors never released it in.
    """
    if name not in ATLAS_SPECS:
        raise KeyError(
            f"unknown parcellation {name!r}; available: {sorted(ATLAS_SPECS)}"
        )
    spec = ATLAS_SPECS[name]
    if space is None:
        space, dflt = _default_space(spec)
        density = density or dflt
    if space not in spec["spaces"]:
        raise ValueError(
            f"{name} is not distributed in space {space!r}; available: "
            f"{ {k: v for k, v in spec['spaces'].items()} }. "
            "Refusing to resample an atlas into a space its authors did not release."
        )
    if density is None:
        density = spec["spaces"][space][0]
    if density not in spec["spaces"][space]:
        raise ValueError(
            f"{name} in {space} is available at densities {spec['spaces'][space]}, "
            f"not {density!r}"
        )
    return _load_cached(name, space, density, rebuild)


# ---------------------------------------------------------------------------
# crosswalk
# ---------------------------------------------------------------------------
def crosswalk(a: Parcellation, b: Parcellation) -> np.ndarray:
    """Overlap matrix between two parcellations of the same support.

    Returns ``(n_a, n_b)`` of shared area (mm^2) for surface parcellations or
    shared volume (mm^3) for volumetric ones.  Rows do not sum to the parcel's
    total when the two atlases cover different extents; that shortfall is
    information, so it is not normalised away here.

    This is the honest form of "atlas A region X corresponds to atlas B region
    Y": a distribution, not an identity (thesis §3.1, multilayer-partition
    caveat).
    """
    if a.is_surface() and b.is_surface():
        if (a.space, a.density) != (b.space, b.density):
            raise ValueError("surface crosswalk requires a common space/density")
        geom = _surface_geometry(a.space, a.density)
        out = np.zeros((a.n_parcels, b.n_parcels))
        for h, (coords, faces) in geom.items():
            va = _vertex_areas(coords, faces)
            la, lb = a.vertex_labels[h], b.vertex_labels[h]  # type: ignore[index]
            m = (la >= 0) & (lb >= 0)
            np.add.at(out, (la[m], lb[m]), va[m])
        return out
    if a.is_volume() and b.is_volume():
        if a.affine is None or b.affine is None or not np.allclose(a.affine, b.affine):
            raise ValueError("volumetric crosswalk requires a common affine/grid")
        vox = float(abs(np.linalg.det(a.affine[:3, :3])))
        out = np.zeros((a.n_parcels, b.n_parcels))
        la, lb = a.voxel_labels.ravel(), b.voxel_labels.ravel()  # type: ignore[union-attr]
        m = (la >= 0) & (lb >= 0)
        np.add.at(out, (la[m], lb[m]), vox)
        return out
    raise ValueError("crosswalk needs two surface or two volumetric parcellations")
