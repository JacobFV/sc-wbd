"""Regional maps: receptors, myelin, thickness, gradients, timescale, metabolism.

Every map here is a **group average over a population the model will later claim
to individualise**.  That is the single most important fact about this module,
and it is enforced structurally: every :class:`RegionalMap` carries an
:class:`~scwbd.schema.UncertaintyLedger` with
``bias_status="prior_specified_sensitivity"``, a non-degenerate swept interval,
and a ``validity_domain["forbidden_inference"]`` sentence.

The thesis appendix is explicit for the receptor atlas: *never infer a
subject's receptor density from an atlas value alone*.  That sentence travels
with the numbers.

Coverage is honest
------------------
A cortical gradient has no value in the thalamus.  Rather than imputing one
(rule §7.1: missing data is never imputed as zero or as an average-brain
label), maps carry a ``coverage`` mask and ``nan`` outside it.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import sources as S
from ._compat import UncertaintyLedger, group_average_ledger
from .atlases import Parcellation, load_parcellation
from .paths import cache_dir, derived_dir, set_download_env, src_dir

__all__ = [
    "RegionalMap",
    "MapSet",
    "available_maps",
    "load_maps",
    "regional_map",
    "receptor_matrix",
    "SURFACE_MAPS",
    "RECEPTOR_GROUPS",
]


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionalMap:
    """One scalar quantity per parcel, with its uncertainty ledger."""

    name: str
    parcellation_name: str
    values: np.ndarray  # (n_parcels,), nan outside coverage
    coverage: np.ndarray  # (n_parcels,) bool
    units: str
    source_key: str
    ledger: UncertaintyLedger
    mechanistic_status: str = "functional"
    notes: str = ""

    @property
    def n_covered(self) -> int:
        return int(self.coverage.sum())

    def zscored(self) -> np.ndarray:
        """Z-score within the covered parcels.

        Note what this destroys: absolute density.  Two PET tracers z-scored
        onto the same axis are comparable in *pattern* and incomparable in
        *magnitude*.  Downstream code that needs magnitude must not use this.
        """
        v = self.values.copy()
        m = self.coverage & np.isfinite(v)
        if m.sum() < 2:
            return v
        v[m] = (v[m] - v[m].mean()) / v[m].std(ddof=1)
        return v

    def rank(self) -> np.ndarray:
        """Rank-normalise to ``[0, 1]`` within coverage; ``nan`` outside."""
        v = np.full(self.values.shape, np.nan)
        m = self.coverage & np.isfinite(self.values)
        if m.sum() == 0:
            return v
        v[m] = np.argsort(np.argsort(self.values[m])) / max(int(m.sum()) - 1, 1)
        return v

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RegionalMap({self.name!r}, {self.parcellation_name}, "
            f"covered={self.n_covered}/{self.values.size}, units={self.units!r}, "
            f"bias_status={self.ledger.bias_status})"
        )


@dataclass(frozen=True)
class MapSet:
    """All maps available for one parcellation."""

    parcellation_name: str
    space: str
    density: str
    maps: dict[str, RegionalMap]
    receptor_names: tuple[str, ...] = ()
    #: ``{map_name: reason}`` for every map that was *expected* here and could
    #: not be built.  A map that silently vanishes from the artifact is a map
    #: nobody can audit: the Schaefer300/400 receptor panels shipped without
    #: ``receptor_5HT4`` for exactly that reason, and nothing in the file said
    #: so.  An empty dict therefore means "nothing was dropped", not "nobody
    #: looked".
    unavailable: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, k: str) -> RegionalMap:
        if k not in self.maps:
            raise KeyError(
                f"{k!r} is not available for {self.parcellation_name}; have "
                f"{sorted(self.maps)}"
            )
        return self.maps[k]

    def __contains__(self, k: str) -> bool:
        return k in self.maps

    def keys(self) -> Iterable[str]:
        return self.maps.keys()

    def matrix(self, names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Stack several maps; returns ``(values (n, k), coverage (n, k))``."""
        names = list(names)
        v = np.stack([self.maps[n].values for n in names], axis=1)
        c = np.stack([self.maps[n].coverage for n in names], axis=1)
        return v, c

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        meta: dict[str, Any] = {
            "parcellation_name": self.parcellation_name,
            "space": self.space,
            "density": self.density,
            "receptor_names": list(self.receptor_names),
            "unavailable": dict(self.unavailable),
            "maps": {},
        }
        for k, m in self.maps.items():
            payload[f"v__{k}"] = m.values
            payload[f"c__{k}"] = m.coverage
            meta["maps"][k] = {
                "units": m.units,
                "source_key": m.source_key,
                "mechanistic_status": m.mechanistic_status,
                "notes": m.notes,
                "ledger": m.ledger.model_dump(mode="json"),
            }
        payload["_meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MapSet":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        maps = {}
        for k, md in meta["maps"].items():
            maps[k] = RegionalMap(
                name=k,
                parcellation_name=meta["parcellation_name"],
                values=z[f"v__{k}"],
                coverage=z[f"c__{k}"],
                units=md["units"],
                source_key=md["source_key"],
                ledger=UncertaintyLedger.model_validate(md["ledger"]),
                mechanistic_status=md["mechanistic_status"],
                notes=md["notes"],
            )
        return cls(
            parcellation_name=meta["parcellation_name"],
            space=meta["space"],
            density=meta["density"],
            maps=maps,
            receptor_names=tuple(meta["receptor_names"]),
            # artifacts written before the field existed carry no record, which
            # is itself the thing being fixed; they load as "unknown", not "none"
            unavailable=dict(meta.get("unavailable", {})),
        )


# ---------------------------------------------------------------------------
# surface annotation registry (neuromaps)
# ---------------------------------------------------------------------------
#: name -> (source, desc, den, units, source_key, mechanistic_status, forbidden)
SURFACE_MAPS: dict[str, dict[str, Any]] = {
    "fc_gradient1": dict(
        source="margulies2016", desc="fcgradient01", den="32k", units="au",
        key="margulies2016", status="functional",
        forbidden="A diffusion-embedding coordinate of a group FC matrix. It is not a "
                  "measured anatomical hierarchy and not this subject's gradient.",
    ),
    "fc_gradient2": dict(
        source="margulies2016", desc="fcgradient02", den="32k", units="au",
        key="margulies2016", status="functional",
        forbidden="Second embedding component; sign and scale are arbitrary.",
    ),
    "fc_gradient3": dict(
        source="margulies2016", desc="fcgradient03", den="32k", units="au",
        key="margulies2016", status="functional",
        forbidden="Third embedding component; sign and scale are arbitrary.",
    ),
    "myelin_t1t2": dict(
        source="hcps1200", desc="myelinmap", den="32k", units="dimensionless",
        key="hcps1200_maps", status="effective",
        forbidden="T1w/T2w ratio is a myelin *proxy* confounded by iron and water "
                  "content; it is not a myelin concentration and not a subject value.",
    ),
    "cortical_thickness": dict(
        source="hcps1200", desc="thickness", den="32k", units="mm",
        key="hcps1200_maps", status="effective",
        forbidden="Group-average FreeSurfer thickness in young adults; not this "
                  "subject's thickness and not valid outside 22-37 years.",
    ),
    "sa_axis": dict(
        source="sydnor2021", desc="SAaxis", den="32k", units="index",
        key="sydnor2021", status="functional",
        forbidden="A rank average of ten heterogeneous maps. A coordinate along which "
                  "properties vary, not a mechanism and not a measurement.",
    ),
    "evolutionary_expansion": dict(
        source="xu2020", desc="evoexp", den="32k", units="dimensionless",
        key="hill2010", status="functional",
        forbidden="Human/macaque surface expansion under a registration that assumes "
                  "areal homology; homology is contested for association cortex.",
    ),
    "fc_homology": dict(
        source="xu2020", desc="FChomology", den="32k", units="dimensionless",
        key="hill2010", status="functional",
        forbidden="Cross-species functional similarity; not evidence of a shared circuit.",
    ),
    "intrinsic_timescale_meg": dict(
        source="hcps1200", desc="megtimescale", den="4k", units="ms",
        key="hcps1200_maps", status="effective",
        forbidden="MEG source-space timescale in 33 subjects. The inverse problem is "
                  "ill-posed and the estimate is smoothed over centimetres; it bounds "
                  "a regional timescale prior, it does not measure one.",
    ),
    "cbf": dict(
        source="raichle", desc="cbf", den="164k", units="au",
        key="raichle_metabolism", status="effective",
        forbidden="Resting PET group average in 33 adults; metabolism is state dependent.",
    ),
    "cmrglc": dict(
        source="raichle", desc="cmrglc", den="164k", units="au",
        key="raichle_metabolism", status="effective",
        forbidden="Resting PET group average in 33 adults; metabolism is state dependent.",
    ),
    "developmental_expansion": dict(
        source="hill2010", desc="devexp", den="164k", units="dimensionless",
        key="hill2010", status="functional",
        forbidden="Infant-to-adult surface expansion; a developmental summary, not an "
                  "adult mechanism.",
    ),
}

#: Excitatory / inhibitory grouping of the PET targets, used for the E/I proxy.
#: This grouping is a *modelling choice about receptor pharmacology*, not a
#: measurement, and it is coarse: NMDA and mGluR5 are glutamatergic markers of
#: very different kinetics, and GABA-A benzodiazepine binding indexes only one
#: inhibitory receptor family.
RECEPTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "excitatory": ("NMDA", "mGluR5"),
    "inhibitory": ("GABAa",),
    "modulatory_monoamine": ("D1", "D2", "DAT", "NAT", "5HT1a", "5HT1b", "5HT2a",
                             "5HT4", "5HT6", "5HTT"),
    "modulatory_cholinergic": ("A4B2", "M1", "VAChT"),
    "other": ("CB1", "H3", "MOR"),
}


# ---------------------------------------------------------------------------
# fsLR resampling
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _fslr_sphere(den: str, hemi: str) -> np.ndarray:
    set_download_env()
    from neuromaps import datasets as nmd

    a = nmd.fetch_atlas("fsLR", den, data_dir=str(cache_dir() / "neuromaps"))
    paths = {"L": a["sphere"][0], "R": a["sphere"][1]}
    import nibabel as nib

    g = nib.load(str(paths[hemi]))
    v = np.asarray(g.darrays[0].data, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _resample_fslr(values: np.ndarray, src_den: str, dst_den: str, hemi: str) -> np.ndarray:
    """Nearest-neighbour resampling between fsLR densities on the sphere.

    The fsLR meshes at 4k / 32k / 164k share a spherical registration but not a
    vertex correspondence, so a value is carried to its nearest registered
    neighbour.  For the smooth group maps used here that is adequate; for a
    sharp map it would not be, and the substitution is recorded in the ledger
    notes of every map that needed it.
    """
    if src_den == dst_den:
        return values
    from scipy.spatial import cKDTree

    src = _fslr_sphere(src_den, hemi)
    dst = _fslr_sphere(dst_den, hemi)
    _, idx = cKDTree(src).query(dst, k=1)
    return values[idx]


def _fetch_surface_annotation(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    set_download_env()
    import nibabel as nib
    from neuromaps import datasets as nmd

    files = nmd.fetch_annotation(
        source=spec["source"], desc=spec["desc"], space="fsLR", den=spec["den"],
        data_dir=str(cache_dir() / "neuromaps"),
    )
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    out: dict[str, np.ndarray] = {}
    for p in files:
        h = "L" if "hemi-L" in str(p) else "R"
        out[h] = np.asarray(nib.load(str(p)).darrays[0].data, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# parcellating
# ---------------------------------------------------------------------------
def _parcellate_surface(
    parc: Parcellation, per_hemi: dict[str, np.ndarray], src_den: str
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean of a fsLR vertex map within each parcel.

    A hemisphere absent from ``per_hemi`` leaves that hemisphere's parcels
    *uncovered* rather than aborting the map.  Some upstream annotations really
    are one-sided -- neuromaps redistributes the Hill 2010 expansion maps as
    right hemisphere only -- and dropping the whole map on that account throws
    away the hemisphere that does exist.  Per the module docstring, missing data
    is carried in ``coverage``, never imputed.
    """
    from .geometry import load_surface

    if parc.space != "fsLR":
        raise ValueError("surface annotations are parcellated on fsLR only")
    if not per_hemi:
        raise ValueError("annotation has no hemispheres")
    n = parc.n_parcels
    num = np.zeros(n)
    den = np.zeros(n)
    for h in ("L", "R"):
        if h not in per_hemi:
            continue
        surf = load_surface(parc.space, parc.density, "midthickness", h)
        v = _resample_fslr(per_hemi[h], src_den, parc.density, h)
        if v.shape[0] != surf.n_vertices:
            raise ValueError(f"annotation has {v.shape[0]} vertices, mesh has {surf.n_vertices}")
        va = surf.vertex_areas()
        vl = parc.vertex_labels[h]  # type: ignore[index]
        m = (vl >= 0) & np.isfinite(v)
        np.add.at(num, vl[m], v[m] * va[m])
        np.add.at(den, vl[m], va[m])
    vals = np.full(n, np.nan)
    cov = den > 0
    vals[cov] = num[cov] / den[cov]
    return vals, cov


def _parcellate_volume(parc_vol: Parcellation, img_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a volumetric map within each parcel of a volumetric atlas."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    src = nib.load(str(img_path))
    if src.ndim > 3:
        # Several of the PET volumes are stored 4D with a singleton time axis.
        arr = np.asarray(src.dataobj)
        while arr.ndim > 3:
            if arr.shape[-1] != 1:
                raise ValueError(f"{img_path}: expected a scalar map, got shape {arr.shape}")
            arr = arr[..., 0]
        src = nib.Nifti1Image(arr, src.affine, src.header)
    lab = parc_vol.voxel_labels
    assert lab is not None and parc_vol.affine is not None
    ref = nib.Nifti1Image(np.zeros(lab.shape, dtype=np.float32), parc_vol.affine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = resample_to_img(src, ref, interpolation="linear", force_resample=True,
                              copy_header=True)
    data = np.asarray(res.dataobj, dtype=np.float64)
    n = parc_vol.n_parcels
    vals = np.full(n, np.nan)
    cov = np.zeros(n, dtype=bool)
    for k in range(n):
        m = (lab == k) & np.isfinite(data) & (data != 0)
        if m.sum() >= 5:
            vals[k] = float(data[m].mean())
            cov[k] = True
    return vals, cov


def _sample_volume_on_surface(parc: Parcellation, img_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Sample an MNI152 volume at midthickness vertices, then average per parcel.

    This is the volume-to-surface step that ``wb_command
    -volume-to-surface-mapping`` would otherwise do.  Each fsLR vertex is looked
    up in the volume by trilinear interpolation at its own coordinate, and the
    per-parcel value is the area-weighted mean over covered vertices.

    Two errors are incurred and neither is corrected here:

    * the fsLR group-average midthickness surface is *aligned to* MNI152 but not
      identical with it, so a vertex may sample a voxel a millimetre or two off;
    * sampling a single midthickness depth mixes the cortical ribbon with
      neighbouring white matter and CSF wherever the PET volume is smooth on the
      scale of cortical thickness -- which, for 6-8 mm-resolution PET, is
      everywhere.

    Both are recorded in the ledger of every map built this way.
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    from .geometry import load_surface

    src = nib.load(str(img_path))
    arr = np.asarray(src.dataobj, dtype=np.float64)
    while arr.ndim > 3:
        if arr.shape[-1] != 1:
            raise ValueError(f"{img_path}: expected a scalar map, got shape {arr.shape}")
        arr = arr[..., 0]
    inv = np.linalg.inv(np.asarray(src.affine, dtype=np.float64))

    n = parc.n_parcels
    num = np.zeros(n)
    den = np.zeros(n)
    for h in ("L", "R"):
        surf = load_surface(parc.space, parc.density, "midthickness", h)
        vl = parc.vertex_labels[h]  # type: ignore[index]
        ijk = nib.affines.apply_affine(inv, surf.coords)
        v = map_coordinates(arr, ijk.T, order=1, mode="constant", cval=np.nan)
        va = surf.vertex_areas()
        # PET volumes are zero outside the brain mask; a zero there is missing
        # data, not a measurement of zero binding.
        m = (vl >= 0) & np.isfinite(v) & (v != 0.0)
        np.add.at(num, vl[m], v[m] * va[m])
        np.add.at(den, vl[m], va[m])
    vals = np.full(n, np.nan)
    cov = den > 0
    vals[cov] = num[cov] / den[cov]
    return vals, cov


def _volumetric_twin(parc: Parcellation) -> Parcellation | None:
    """The MNI152 release of the same atlas, for parcellating volumetric maps.

    Surface parcels and volumetric parcels of the same atlas share label
    *names*, and joining on those names is how a PET volume reaches a surface
    parcellation here.  That join incurs partial-volume mixing at the ribbon
    and a volume-to-surface correspondence error, both recorded in the ledger.
    """
    if parc.is_volume():
        return parc
    try:
        return load_parcellation(parc.name, "MNI152", "1mm")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# receptors
# ---------------------------------------------------------------------------
#: PET targets in the Hansen collection that are *not* receptors or
#: transporters.  FDOPA indexes presynaptic dopamine synthesis capacity; keeping
#: it in the receptor matrix would misname what it measures.
_NON_RECEPTOR_TARGETS: frozenset[str] = frozenset({"FDOPA"})


def _hansen_nifti_files() -> dict[str, list[Path]]:
    """Group Hansen PET volumes by receptor/transporter target."""
    d = src_dir() / "hansen_receptors" / "data" / "PET_nifti_images"
    if not d.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    for p in sorted(d.iterdir()):
        if not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        target = p.name.split("_")[0]
        target = {"GABAa-bz": "GABAa", "MU": "MOR", "SERT": "5HTT"}.get(target, target)
        out.setdefault(target, []).append(p)
    return out


def _receptor_maps(
    parc: Parcellation,
) -> tuple[dict[str, RegionalMap], tuple[str, ...], dict[str, str]]:
    # Surface parcellations sample the PET volume directly at their own
    # vertices; volumetric parcellations average it over their own voxels.
    # Neither route reindexes one atlas through another.
    if parc.is_surface():
        if parc.space != "fsLR":
            return {}, (), {"receptors": f"no PET route for surface space {parc.space!r}"}
        route = "surface_sampling"
        sampler = lambda p: _sample_volume_on_surface(parc, p)  # noqa: E731
    else:
        route = "voxel_average"
        sampler = lambda p: _parcellate_volume(parc, p)  # noqa: E731
    files = _hansen_nifti_files()
    if not files:
        return {}, (), {"receptors": "the Hansen PET volumes are not on disk"}
    # How well does the shipped fast route agree with the volumetric join?
    # ``None`` means nobody has measured it for this parcellation, which is
    # recorded as unmeasured rather than passed off as agreement.
    from .route_check import FRAGILE_BELOW, load_route_agreement

    _rep = load_route_agreement(parc.name, parc.space, parc.density)
    route_report = (_rep or {}).get("per_target", {})
    maps: dict[str, RegionalMap] = {}
    unavailable: dict[str, str] = {}
    for target, paths in sorted(files.items()):
        per_tracer, cov_any = [], None
        n_hc = 0
        dropped_tracers: list[str] = []
        for p in paths:
            try:
                v, c = sampler(p)
            except Exception as exc:  # noqa: BLE001
                # A silently dropped tracer would quietly shrink the receptor
                # panel, so say so.
                warnings.warn(f"PET volume {p.name} could not be parcellated: {exc!r}",
                              stacklevel=2)
                dropped_tracers.append(f"{p.name}: {exc!r}")
                continue
            m = c & np.isfinite(v)
            if m.sum() < 2:
                dropped_tracers.append(
                    f"{p.name}: only {int(m.sum())} parcels covered, need >= 2")
                continue
            z = np.full(v.shape, np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std(ddof=1)
            per_tracer.append(z)
            cov_any = m if cov_any is None else (cov_any | m)
            for tok in p.name.split("_"):
                if tok.startswith("hc") and tok[2:].isdigit():
                    n_hc += int(tok[2:])
        if not per_tracer:
            # Targets with a single tracer -- 5HT4, NMDA, A4B2, CB1, M1, H3 --
            # are lost entirely when that one file fails, so the loss must be
            # recorded rather than inferred later from a short panel.
            unavailable[f"receptor_{target}"] = (
                "every tracer failed: " + "; ".join(dropped_tracers)
                if dropped_tracers
                else "no usable tracer volume"
            )
            continue
        if dropped_tracers:
            unavailable[f"receptor_{target}__partial"] = (
                f"built from {len(per_tracer)}/{len(paths)} tracers; dropped: "
                + "; ".join(dropped_tracers)
            )
        stack = np.stack(per_tracer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanmean(stack, axis=0)
            spread = np.nanstd(stack, axis=0, ddof=0) if stack.shape[0] > 1 else np.zeros(vals.shape)
        cov = np.asarray(cov_any, dtype=bool)
        vals[~cov] = np.nan
        # Between-route disagreement is a *second* empirical handle on sampling
        # bias, independent of tracer-to-tracer spread: a map that changes when
        # you change how the volume is read into parcels is a map whose value
        # is partly a property of the reader.  It widens the swept interval.
        rr = route_report.get(target)
        route_r = rr["r_route"] if rr else None
        route_fragile = bool(rr["route_fragile"]) if rr else None
        half = max(float(np.nanmax(spread)), 0.5)
        if route_r is not None:
            # 1 - r is the fraction of between-parcel variance the two routes
            # do not share; charge it to the interval rather than hiding it.
            half = max(half, 0.5 + (1.0 - float(route_r)))
        led = group_average_ledger(
            units="zscore",
            # tracer-to-tracer disagreement is the only empirical handle we have
            # on this bias; it is a lower bound, not the bias itself
            bias_interval=(-half, half),
            variance={
                "measurement": float(np.nanmean(spread**2)) if stack.shape[0] > 1 else 0.0,
                "model_class": 0.25,
            },
            forbidden_inference=(
                "A subject's receptor density is NOT inferable from this atlas value "
                "(thesis Appendix A). These are group averages from separate small "
                "donor cohorts, z-scored per tracer, so absolute density is gone."
            ),
            n_donors=n_hc or None,
            validity_domain={
                "n_tracers": int(stack.shape[0]),
                "target": target,
                "route": route,
                "space": (
                    "MNI152 PET volume sampled at fsLR midthickness vertices "
                    "(trilinear), then area-weighted per parcel"
                    if route == "surface_sampling"
                    else "MNI152 PET volume averaged over the parcel's own voxels"
                ),
                "license": S.SRC["hansen_receptors"]["license"],
                # None => nobody measured it for this parcellation. Not "fine".
                "route_agreement_r": route_r,
                "route_fragile": route_fragile,
                "route_agreement_threshold": FRAGILE_BELOW,
            },
            notes=(
                S.SRC["hansen_receptors"]["bias"]
                + (
                    f" ROUTE-FRAGILE: surface sampling and volumetric averaging "
                    f"of this tracer agree at only r={route_r:.2f} across cortical "
                    "parcels, so this map's parcel values depend materially on how "
                    "the PET volume was read into parcels. See "
                    "scwbd.anatomy.route_check and reports/anatomy_prior.md S1.6."
                    if route_fragile
                    else ""
                )
                + (
                    " Volume-to-surface sampling incurs fsLR/MNI152 alignment "
                    "error of a millimetre or two and mixes the cortical ribbon "
                    "with adjacent white matter and CSF, because PET resolution "
                    "(6-8 mm) exceeds cortical thickness."
                    if route == "surface_sampling"
                    else ""
                )
            ),
        )
        maps[f"receptor_{target}"] = RegionalMap(
            name=f"receptor_{target}",
            parcellation_name=parc.name,
            values=vals,
            coverage=cov,
            units="zscore",
            source_key="hansen_receptors",
            ledger=led,
            mechanistic_status="effective",
            notes=(
                f"{stack.shape[0]} PET tracer map(s) for {target}, each z-scored "
                "across parcels before averaging because the tracers are on "
                "mutually incommensurate scales. Absolute density is not recoverable."
                + (
                    f" NOTE: {target} is not a receptor or transporter -- it indexes "
                    "presynaptic dopamine synthesis capacity -- and is excluded from "
                    "the receptor matrix."
                    if target in _NON_RECEPTOR_TARGETS
                    else ""
                )
            ),
        )
    names = tuple(
        sorted(
            k.replace("receptor_", "")
            for k in maps
            if k.replace("receptor_", "") not in _NON_RECEPTOR_TARGETS
        )
    )
    return maps, names, unavailable


def receptor_matrix(parc: Parcellation) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """``(values (n, k), names, coverage (n, k))`` for all receptor maps."""
    ms = load_maps(parc)
    names = tuple(f"receptor_{r}" for r in ms.receptor_names)
    if not names:
        n = parc.n_parcels
        return np.zeros((n, 0)), (), np.zeros((n, 0), dtype=bool)
    v, c = ms.matrix(names)
    return v, ms.receptor_names, c


# ---------------------------------------------------------------------------
# derived maps
# ---------------------------------------------------------------------------
def _ei_proxy(maps: dict[str, RegionalMap], parc: Parcellation) -> RegionalMap | None:
    """Excitation/inhibition proxy from receptor z-scores.

    ``mean(z[excitatory]) - mean(z[inhibitory])``.  This is a *proxy*: receptor
    density is not synaptic conductance, PET binding is not receptor count, and
    the excitatory group here is two glutamatergic markers against one
    GABA-A benzodiazepine-site marker.  It is admitted because §5 of
    ARCHITECTURE requires per-parcel regional heterogeneity and the alternative
    -- one identical neural mass per parcel -- is the failure mode the thesis
    names explicitly.

    A measured caveat, not a hypothetical one -- and note that it *changed* when
    the receptor maps were rebuilt through the surface-sampling route.  The
    cached artifacts this claim was first measured on had been produced by the
    older volumetric-join route, and reported NMDA-GABA-A r = 0.73 with the NMDA
    contribution cancelling out (r(E/I, NMDA) ~ 0.00).  Neither number survives
    the route that is actually shipped.

    Measured on the surface-sampled maps (Schaefer-100 / Schaefer-400):

    ==========================  ==============  ==============
    quantity                    Schaefer-100    Schaefer-400
    ==========================  ==============  ==============
    NMDA - GABA-A                       +0.264          +0.346
    mGluR5 - GABA-A                     +0.275          +0.500
    r(E/I, NMDA)                        +0.510          +0.536
    r(E/I, GABA-A)                      -0.617          -0.516
    r(E/I, S-A axis)                    +0.404          +0.303
    sd(E/I) / mean sd(ingredients)        1.10            0.95
    ==========================  ==============  ==============

    So the ingredients co-vary only *moderately*, the contrast keeps essentially
    the full spread of its ingredients rather than two thirds of it, and NMDA is
    now a substantial contributor rather than a cancelling one.  The proxy is
    less degenerate than previously documented.

    That is not straightforwardly good news.  The contrast's largest single
    contributor is NMDA, and NMDA is the *least route-stable* map in the panel
    (surface vs volumetric r = 0.59; see :mod:`scwbd.anatomy.route_check`).  The
    E/I proxy is therefore not a weak contrast built on strong ingredients -- it
    is a reasonably strong contrast built on the two ingredients whose values
    depend most on how the PET volume was read into parcels.  Both facts travel
    in the ledger: ``validity_domain["route_fragile_ingredients"]`` names them
    and ``forbidden_inference`` spells out the consequence.
    """
    exc = [maps[f"receptor_{r}"] for r in RECEPTOR_GROUPS["excitatory"] if f"receptor_{r}" in maps]
    inh = [maps[f"receptor_{r}"] for r in RECEPTOR_GROUPS["inhibitory"] if f"receptor_{r}" in maps]
    if not exc or not inh:
        return None
    # Which ingredients are route-fragile?  The E/I contrast is the one place
    # where NMDA and GABA-A are load-bearing, and those are precisely the two
    # maps whose value depends most on how the PET volume was read.  That fact
    # travels with the number instead of living only in a report.
    fragile = {
        m.name.replace("receptor_", ""): m.ledger.validity_domain.get("route_agreement_r")
        for m in exc + inh
        if m.ledger.validity_domain.get("route_fragile")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = np.nanmean(np.stack([m.values for m in exc]), axis=0)
        i = np.nanmean(np.stack([m.values for m in inh]), axis=0)
    cov = np.all([m.coverage for m in exc + inh], axis=0)
    v = np.full(parc.n_parcels, np.nan)
    v[cov] = (e - i)[cov]
    led = group_average_ledger(
        units="zscore",
        # widened from +/-1.5 when an ingredient is route-fragile: the contrast
        # inherits its ingredients' sampling bias and cannot be tighter than them
        bias_interval=(-1.5 - 0.5 * len(fragile), 1.5 + 0.5 * len(fragile)),
        variance={"measurement": 0.5 + 0.25 * len(fragile), "model_class": 1.0},
        forbidden_inference=(
            "Not a measurement of excitation/inhibition balance. PET binding "
            "potential is not receptor count, receptor count is not synaptic "
            "conductance, and this contrast uses two glutamatergic markers "
            "against one GABA-A benzodiazepine-site marker."
            + (
                " Additionally, "
                + ", ".join(f"{k} (route r={v:.2f})" for k, v in sorted(fragile.items()))
                + " disagree between the surface-sampling and volumetric-join "
                "routes, so the sign of this contrast in any given parcel is not "
                "robust to a defensible change in how the PET volume is read."
                if fragile
                else ""
            )
        ),
        validity_domain={
            "excitatory_markers": list(RECEPTOR_GROUPS["excitatory"]),
            "inhibitory_markers": list(RECEPTOR_GROUPS["inhibitory"]),
            "interpretation": "relative, rank-meaningful only; zero is the cortical mean",
            "route_fragile_ingredients": {k: v for k, v in sorted(fragile.items())},
        },
        notes=(
            "Model-class variance is set as large as the measurement variance "
            "because the pharmacological grouping is itself a modelling choice. "
            "The excitatory and inhibitory ingredient maps co-vary strongly "
            "across cortex, so this difference is a weak second-order contrast: "
            "it discards the dominant shared gradient in receptor density and "
            "retains only the part on which the tracers disagree."
        ),
    )
    return RegionalMap(
        name="ei_proxy",
        parcellation_name=parc.name,
        values=v,
        coverage=cov,
        units="zscore",
        source_key="hansen_receptors",
        ledger=led,
        mechanistic_status="surrogate",
        notes="Derived contrast, not an upstream measurement.",
    )


# ---------------------------------------------------------------------------
# public loaders
# ---------------------------------------------------------------------------
def available_maps() -> dict[str, dict[str, Any]]:
    out = {k: dict(v) for k, v in SURFACE_MAPS.items()}
    out["ei_proxy"] = {"derived": True, "units": "zscore", "key": "hansen_receptors"}
    for t in sorted(_hansen_nifti_files()):
        out[f"receptor_{t}"] = {"units": "zscore", "key": "hansen_receptors", "volumetric": True}
    return out


def load_maps(parc: Parcellation, *, rebuild: bool = False) -> MapSet:
    """All maps parcellated to ``parc``, cached under ``assets/derived/maps``."""
    cache = derived_dir("maps") / f"{parc.name}__{parc.space}-{parc.density}__maps.npz"
    if cache.exists() and not rebuild:
        return MapSet.load(cache)

    maps: dict[str, RegionalMap] = {}
    unavailable: dict[str, str] = {}

    # -- surface annotations ------------------------------------------
    if parc.is_surface() and parc.space == "fsLR":
        for name, spec in SURFACE_MAPS.items():
            try:
                per_hemi = _fetch_surface_annotation(spec)
                vals, cov = _parcellate_surface(parc, per_hemi, spec["den"])
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"map {name!r} unavailable for {parc.name}: {exc!r}", stacklevel=2)
                unavailable[name] = repr(exc)
                continue
            src = S.SRC[spec["key"]]
            resampled = spec["den"] != parc.density
            led = group_average_ledger(
                units=spec["units"],
                bias_interval=(-1.0, 1.0) if spec["units"] in ("au", "index", "zscore",
                                                               "dimensionless")
                else (-0.2, 0.2),
                variance={"measurement": 0.1, "between_session": 0.1, "model_class": 0.1},
                forbidden_inference=spec["forbidden"],
                validity_domain={
                    "native_density": spec["den"],
                    "resampled_to": parc.density if resampled else None,
                    "population": "healthy adults, group average",
                    "hemispheres": sorted(per_hemi),
                },
                notes=(
                    src["bias"]
                    + (
                        f" Nearest-neighbour resampled fsLR-{spec['den']} -> "
                        f"fsLR-{parc.density} on the spherical registration."
                        if resampled
                        else ""
                    )
                    + (
                        " SINGLE-HEMISPHERE: upstream redistributes only "
                        f"{'/'.join(sorted(per_hemi))}, so the opposite hemisphere's "
                        "parcels are uncovered. Do not mirror this across the "
                        "midline -- cortical asymmetry is exactly what that would "
                        "fabricate."
                        if len(per_hemi) < 2
                        else ""
                    )
                ),
            )
            maps[name] = RegionalMap(
                name=name,
                parcellation_name=parc.name,
                values=vals,
                coverage=cov,
                units=spec["units"],
                source_key=spec["key"],
                ledger=led,
                mechanistic_status=spec["status"],
            )

    # -- receptors (volumetric PET) -------------------------------------
    rec, rec_names, rec_unavailable = _receptor_maps(parc)
    maps.update(rec)
    unavailable.update(rec_unavailable)

    ei = _ei_proxy(maps, parc)
    if ei is not None:
        maps["ei_proxy"] = ei
    else:
        unavailable["ei_proxy"] = (
            "the E/I proxy needs receptor_NMDA, receptor_mGluR5 and receptor_GABAa; "
            f"available receptors: {list(rec_names)}"
        )

    ms = MapSet(
        parcellation_name=parc.name,
        space=parc.space,
        density=parc.density,
        maps=maps,
        receptor_names=rec_names,
        unavailable=unavailable,
    )
    ms.save(cache)
    return ms


def regional_map(parc: Parcellation, name: str) -> np.ndarray:
    """Convenience accessor returning just the values of one map."""
    return load_maps(parc)[name].values
