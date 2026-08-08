"""Parcellate ds000113's localizer BOLD into Schaefer400x7 parcel space.

The 7T localizer runs are an axial **slab**, not a whole brain, so most parcels
are outside the field of view. That is why `min_coverage` is not applied here:
this script's job is to build the cache and *report* the coverage each run
actually has, and the decision about which runs are usable belongs to the
dataset that reads them, where it is one number in a config rather than a
threshold buried in a preprocessing script.

An uncovered parcel is `NaN` in the timeseries and `False` in the mask, never
0.0 -- a zero is a measurement of zero and the likelihood cannot tell it apart.

Run: PYTHONPATH=. .venv/bin/python scripts/parcellate_ds000113.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scwbd.foundation.bolddata import ParcelBOLDConfig, discover_bold_runs

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/sources/ds000113_parcellation.json"


def main() -> int:
    from scwbd.sources.parcellate_bold import parcellate_run

    cfg = ParcelBOLDConfig(
        root="data/ds000113",
        source="ds000113_real",
        cache_dir="data/foundation_cache/parcel_bold",
    )
    runs = discover_bold_runs(cfg)
    print(f"{len(runs)} runs, {len({r.subject for r in runs})} subjects", flush=True)

    chains: dict[str, object] = {}
    rows = []
    t_all = time.time()
    for i, r in enumerate(runs, 1):
        if r.cache_npy.exists() and r.cache_json.exists():
            meta = json.loads(r.cache_json.read_text())
            cov = float(np.asarray(meta["covered"], dtype=bool).mean())
            rows.append({"subject": r.subject, "run": r.task, "coverage": round(cov, 4),
                         "n_frames": int(meta["n_frames"]), "cached": "reused"})
            print(f"[{i}/{len(runs)}] {r.subject} {r.task}: reused, coverage {cov:.3f}", flush=True)
            continue
        t0 = time.time()
        try:
            pb, chain = parcellate_run(
                r.bold, r.t1w, atlas=cfg.atlas, assets=cfg.assets,
                subject=r.subject, run=r.task, nonlinear=cfg.nonlinear,
                chain=chains.get(r.subject),
            )
        except Exception as exc:  # noqa: BLE001 - a failed run is a fact, not a crash
            print(f"[{i}/{len(runs)}] {r.subject} {r.task}: FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"subject": r.subject, "run": r.task, "error": f"{type(exc).__name__}: {exc}"})
            continue
        chains[r.subject] = chain
        np.save(r.cache_npy, pb.timeseries.astype(np.float32))
        r.cache_json.write_text(json.dumps(
            {**pb.describe(), "covered": pb.covered.astype(bool).tolist(),
             "n_voxels": pb.n_voxels.astype(int).tolist()}, indent=1, default=str))
        cov = float(pb.covered.mean())
        dt = time.time() - t0
        rows.append({"subject": r.subject, "run": r.task, "coverage": round(cov, 4),
                     "n_frames": int(pb.timeseries.shape[1]), "seconds": round(dt, 1),
                     "cached": "built"})
        print(f"[{i}/{len(runs)}] {r.subject} {r.task}: coverage {cov:.3f} in {dt:.0f}s", flush=True)

    ok = [r for r in rows if "coverage" in r]
    payload = {
        "source": "ds000113_real",
        "atlas": cfg.atlas,
        "n_runs": len(runs),
        "n_parcellated": len(ok),
        "n_failed": len(rows) - len(ok),
        "coverage_min": min((r["coverage"] for r in ok), default=None),
        "coverage_max": max((r["coverage"] for r in ok), default=None),
        "coverage_median": float(np.median([r["coverage"] for r in ok])) if ok else None,
        "total_seconds": round(time.time() - t_all, 1),
        "note": (
            "7T localizer runs are an axial slab, not a whole brain. Low coverage "
            "here is the acquisition's field of view, not a registration failure; "
            "an uncovered parcel is NaN in the data and False in the mask."
        ),
        "runs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}", flush=True)
    if ok:
        print(f"coverage {payload['coverage_min']:.3f}-{payload['coverage_max']:.3f}, "
              f"median {payload['coverage_median']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
