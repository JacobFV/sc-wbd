"""Do the two PET-to-parcel routes agree, and where they don't, why?

A receptor map reaches a *surface* parcellation by one of two routes:

``surface_sampling``
    Trilinear sample of the MNI152 PET volume at each fsLR midthickness
    vertex, then an area-weighted mean per parcel.  0.16 s/map.
``voxel_average``
    Mean of the PET volume over the voxels of the *volumetric* release of the
    same atlas, joined to the surface parcels by label name.  22.3 s/map.

The fast route is 139x cheaper, so the question of whether it may be used is
worth answering rather than assuming.  On Schaefer400x7 the two routes agree at
r > 0.95 for most tracers but at only r ~ 0.54-0.59 for GABA-A and NMDA -- and
those are exactly the maps the E/I prior is built from.

What this module measures
-------------------------
For each tracer it reports the between-route correlation, and two diagnostics
that discriminate *why* a tracer disagrees:

``depth_stability``
    Correlation between the parcel map sampled at midthickness and the same
    map sampled 3 mm inward along the vertex normal.  Low values mean the
    tracer's parcel-level map reorganises across the cortical ribbon, so a
    single-depth sample and a depth-averaged volume cannot agree in principle.
``d_best``
    The depth offset (mm, positive = outward toward pial) whose surface sample
    best reproduces the volumetric value.  A negative ``d_best`` means the
    volumetric parcel average behaves like a *deeper* sample than
    midthickness, i.e. it is diluted with white matter.

Measured on Schaefer400x7 (n = 39 tracer volumes), ``depth_stability`` predicts
route agreement at Spearman rho = +0.57 (p = 1e-4); the absolute ribbon
contrast predicts it more weakly (rho = -0.35, p = 0.03).  Route disagreement
is therefore mostly a statement about the tracer's radial profile, not about
either route being buggy.

This is a validation artifact, not a build dependency: maps are still built by
the surface route, and the numbers computed here are attached to their ledgers
so a consumer can see how route-fragile each map is.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .atlases import Parcellation, load_parcellation
from .geometry import load_surface
from .paths import derived_dir

__all__ = ["DEPTHS_MM", "route_agreement", "load_route_agreement", "main"]

#: Depth offsets sampled along the vertex normal, mm; positive is outward.
DEPTHS_MM: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)

#: Below this between-route correlation a map is called route-fragile and its
#: sampling bias is widened in the ledger.  0.8 is a judgement call, recorded
#: as such rather than tuned.
FRAGILE_BELOW: float = 0.80


def _vertex_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(coords[faces[:, 1]] - coords[faces[:, 0]],
                  coords[faces[:, 2]] - coords[faces[:, 0]])
    vn = np.zeros_like(coords)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    nrm = np.linalg.norm(vn, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return vn / nrm


def _zscore(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    z = np.full(v.shape, np.nan)
    if m.sum() >= 2:
        z[m] = (v[m] - v[m].mean()) / v[m].std(ddof=1)
    return z


def _surface_at_depths(parc: Parcellation, arr: np.ndarray, affine: np.ndarray,
                       depths: tuple[float, ...]) -> np.ndarray:
    """``(n_depths, n_parcels)`` area-weighted parcel means at each depth."""
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    n = parc.n_parcels
    num = np.zeros((len(depths), n))
    den = np.zeros((len(depths), n))
    for h in ("L", "R"):
        surf = load_surface(parc.space, parc.density, "midthickness", h)
        vl = parc.vertex_labels[h]  # type: ignore[index]
        nrm = _vertex_normals(surf.coords, surf.faces)
        va = surf.vertex_areas()
        for di, d in enumerate(depths):
            ijk = nib.affines.apply_affine(inv, surf.coords + d * nrm)
            v = map_coordinates(arr, ijk.T, order=1, mode="constant", cval=np.nan)
            # PET volumes are zero outside the brain mask; a zero there is
            # missing data, not a measurement of zero binding.
            m = (vl >= 0) & np.isfinite(v) & (v != 0.0)
            np.add.at(num[di], vl[m], v[m] * va[m])
            np.add.at(den[di], vl[m], va[m])
    out = np.full((len(depths), n), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _volume_parcel_means(labels: np.ndarray, n_parcels: int,
                         data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.full(n_parcels, np.nan)
    cov = np.zeros(n_parcels, dtype=bool)
    for k in range(n_parcels):
        m = (labels == k) & np.isfinite(data) & (data != 0)
        if m.sum() >= 5:
            vals[k] = float(data[m].mean())
            cov[k] = True
    return vals, cov


def route_agreement(atlas: str = "Schaefer400x7", *,
                    space: str = "fsLR", density: str = "32k") -> dict[str, Any]:
    """Compare the two PET routes for every Hansen tracer on ``atlas``.

    Returns a JSON-serialisable report with a ``per_target`` section (the
    quantity actually shipped -- the mean of z-scored tracers) and a
    ``per_tracer`` section (which discriminates the mechanism).
    """
    import nibabel as nib
    from nilearn.image import resample_to_img

    from .maps import _NON_RECEPTOR_TARGETS, _hansen_nifti_files

    ps = load_parcellation(atlas, space, density)
    pv = load_parcellation(atlas, "MNI152", "1mm")
    if ps.vertex_labels is None or pv.voxel_labels is None:
        raise ValueError(f"{atlas} lacks a surface/volume pair to compare")

    # Join surface parcels to volumetric parcels by label NAME.  Joining by
    # index would silently compare different cortex if the two releases order
    # their labels differently.
    vidx = {str(l): i for i, l in enumerate(pv.labels)}
    pairs = [(i, vidx[str(l)]) for i, l in enumerate(ps.labels) if str(l) in vidx]
    si = np.array([a for a, _ in pairs], dtype=np.int64)
    vi = np.array([b for _, b in pairs], dtype=np.int64)
    cortex = np.array([str(s) == "cortex" for s in ps.structure])[si]

    lab = np.asarray(pv.voxel_labels)
    ref = nib.Nifti1Image(np.zeros(lab.shape, dtype=np.float32), pv.affine)
    d0 = DEPTHS_MM.index(0.0)
    d_in = DEPTHS_MM.index(-3.0)

    files = _hansen_nifti_files()
    per_tracer: list[dict[str, Any]] = []
    z_surf: dict[str, list[np.ndarray]] = {}
    z_vol: dict[str, list[np.ndarray]] = {}

    for target, paths in sorted(files.items()):
        for p in paths:
            rec: dict[str, Any] = {"target": target, "file": p.name}
            try:
                src = nib.load(str(p))
                arr = np.asarray(src.dataobj, dtype=np.float64)
                while arr.ndim > 3:
                    if arr.shape[-1] != 1:
                        raise ValueError(f"expected a scalar map, got {arr.shape}")
                    arr = arr[..., 0]
                    src = nib.Nifti1Image(arr, src.affine, src.header)
                sv = _surface_at_depths(ps, arr, src.affine, DEPTHS_MM)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = resample_to_img(src, ref, interpolation="linear",
                                          force_resample=True, copy_header=True)
                vv, vc = _volume_parcel_means(lab, pv.n_parcels,
                                              np.asarray(res.dataobj, dtype=np.float64))
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)
                per_tracer.append(rec)
                warnings.warn(f"route check failed for {p.name}: {exc!r}", stacklevel=2)
                continue

            vol = vv[vi]
            base = vc[vi] & cortex & np.isfinite(vol)
            best_r, best_d = -2.0, None
            by_depth: dict[str, float] = {}
            for di, d in enumerate(DEPTHS_MM):
                s = sv[di][si]
                m = base & np.isfinite(s)
                if m.sum() < 10:
                    continue
                r = float(np.corrcoef(s[m], vol[m])[0, 1])
                by_depth[f"{d:+.0f}"] = round(r, 4)
                if r > best_r:
                    best_r, best_d = r, float(d)
            mid, inw = sv[d0][si], sv[d_in][si]
            m2 = cortex & np.isfinite(mid) & np.isfinite(inw)
            rec.update(
                r_route=by_depth.get("+0"),
                r_best=round(best_r, 4) if best_d is not None else None,
                d_best=best_d,
                by_depth=by_depth,
                depth_stability=(round(float(np.corrcoef(mid[m2], inw[m2])[0, 1]), 4)
                                 if m2.sum() > 10 else None),
                ribbon_contrast=(round(float((mid[m2].mean() - inw[m2].mean())
                                             / (mid[m2].std() + 1e-12)), 4)
                                 if m2.sum() > 10 else None),
                n_parcels_compared=int(base.sum()),
            )
            per_tracer.append(rec)
            # accumulate the z-scored per-tracer maps so the *target* level
            # comparison matches what load_maps actually ships
            ms = np.isfinite(mid) & cortex
            z_surf.setdefault(target, []).append(_zscore(mid, ms))
            z_vol.setdefault(target, []).append(_zscore(vol, base))

    per_target: dict[str, Any] = {}
    for target in sorted(z_surf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmean(np.stack(z_surf[target]), axis=0)
            v = np.nanmean(np.stack(z_vol[target]), axis=0)
        m = np.isfinite(s) & np.isfinite(v)
        if m.sum() < 10:
            continue
        r = float(np.corrcoef(s[m], v[m])[0, 1])
        per_target[target] = {
            "r_route": round(r, 4),
            "n_tracers": len(z_surf[target]),
            "n_parcels": int(m.sum()),
            "route_fragile": bool(r < FRAGILE_BELOW),
            "is_receptor": target not in _NON_RECEPTOR_TARGETS,
        }

    return {
        "atlas": atlas,
        "space": space,
        "density": density,
        "depths_mm": list(DEPTHS_MM),
        "fragile_below": FRAGILE_BELOW,
        "shipped_route": "surface_sampling",
        "interpretation": (
            "r_route is the correlation between the shipped surface-sampled map "
            "and the volumetric-join map over cortical parcels joined by label "
            "name. depth_stability is the correlation between the midthickness "
            "sample and a sample 3 mm inward: it is the strongest predictor of "
            "route agreement, so disagreement is primarily a property of the "
            "tracer's radial profile rather than a defect of either route. A "
            "negative d_best means the volumetric average behaves like a deeper "
            "(more white-matter-diluted) sample than midthickness."
        ),
        "per_target": per_target,
        "per_tracer": per_tracer,
    }


def _cache_path(atlas: str, space: str, density: str) -> Path:
    return derived_dir("route_check") / f"{atlas}__{space}-{density}__route.json"


def load_route_agreement(atlas: str, space: str = "fsLR",
                         density: str = "32k") -> dict[str, Any] | None:
    """The cached route report for ``atlas``, or ``None`` if never computed.

    ``None`` means *unmeasured*, not *agreeing*; callers must not read a
    missing report as a clean bill of health.
    """
    p = _cache_path(atlas, space, density)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", action="append", default=None,
                    help="repeatable; default Schaefer400x7")
    ap.add_argument("--space", default="fsLR")
    ap.add_argument("--density", default="32k")
    args = ap.parse_args(argv)
    atlases = args.atlas or ["Schaefer400x7"]

    for atlas in atlases:
        rep = route_agreement(atlas, space=args.space, density=args.density)
        out = _cache_path(atlas, args.space, args.density)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(f"\n== {atlas} ==")
        for t, d in sorted(rep["per_target"].items(), key=lambda kv: kv[1]["r_route"]):
            flag = "  <-- ROUTE-FRAGILE" if d["route_fragile"] else ""
            print(f"  {t:10s} r={d['r_route']:+.3f}  "
                  f"tracers={d['n_tracers']}{flag}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
