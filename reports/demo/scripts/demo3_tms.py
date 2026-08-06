"""Demo 3 - induced TMS E-field on a cortical surface, figure-eight coil.

Computes the field with the analytic layered-sphere solver and cross-checks it
against the charge-BEM solver, then extracts the stored N6/N8 gate numbers.

Run from the main repo with PYTHONPATH=<main repo>.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

from scwbd.intervene.tms.coil import FigureEightCoil, biphasic
from scwbd.intervene.tms.efield import SphericalHeadModel, efield_from_coil
from scwbd.intervene.tms.pose import coil_pose_on_sphere

REPO = pathlib.Path("/home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics")
OUT = pathlib.Path(sys.argv[1])
t0 = time.time()

head = SphericalHeadModel()
coil = FigureEightCoil()
pulse = biphasic()
DIRECTION = [-0.55, 0.68, 0.48]
STANDOFF_M = 0.004
AZIMUTH_DEG = 45.0

pose = coil_pose_on_sphere(
    head,
    DIRECTION,
    standoff_m=STANDOFF_M,
    handle_azimuth_rad=math.radians(AZIMUTH_DEG),
    target_label="simulated left-dorsolateral scalp contact",
)
pts, normals = head.cortical_shell(2562)
print(f"cortical shell: {tuple(pts.shape)} at r={head.cortex_radius} m; scalp r={head.radius} m")

# ---- analytic layered-sphere solver ---------------------------------------
t_a = time.time()
dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head, solver="analytic")
t_analytic = time.time() - t_a
E = dose.value  # [N,3] V/m, float64
mag = E.norm(dim=-1)

# ---- charge-BEM cross-check ------------------------------------------------
t_b = time.time()
try:
    dose_bem = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head, solver="bem")
    mag_bem = dose_bem.value.norm(dim=-1)
    bem_ok, bem_err = True, None
except Exception as exc:  # noqa: BLE001 - a solver that cannot run is reported, not hidden
    dose_bem, mag_bem, bem_ok, bem_err = None, None, False, f"{type(exc).__name__}: {exc}"
t_bem = time.time() - t_b

m = mag.numpy()
pct = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
percentiles = {f"p{p}": float(np.percentile(m, p)) for p in pct}

# component along / across the cortical normal
E_n = (E * normals).sum(-1)
E_t = (E - E_n.unsqueeze(-1) * normals).norm(dim=-1)

counts, edges = np.histogram(m, bins=60)

dist = {
    "peak_magnitude_V_per_m": float(m.max()),
    "peak_vertex_index": int(m.argmax()),
    "mean_magnitude_V_per_m": float(m.mean()),
    "median_magnitude_V_per_m": float(np.median(m)),
    "min_magnitude_V_per_m": float(m.min()),
    "sd_magnitude_V_per_m": float(m.std()),
    "percentiles_V_per_m": percentiles,
    "focality_area_above_half_peak_fraction": float((m > 0.5 * m.max()).mean()),
    "n_vertices_above_half_peak": int((m > 0.5 * m.max()).sum()),
    "peak_abs_component_V_per_m": float(dose.peak()),
    "peak_normal_component_V_per_m": float(E_n.abs().max()),
    "peak_tangential_component_V_per_m": float(E_t.max()),
    "mean_abs_normal_over_mean_tangential": float(E_n.abs().mean() / E_t.mean()),
}
print(json.dumps({k: v for k, v in dist.items() if not isinstance(v, dict)}, indent=1))

peak_note = (
    "TWO DIFFERENT QUANTITIES. `peak_magnitude_V_per_m` is max||E|| over vertices "
    "= %.4f V/m, which is the ~134.5 V/m figure in circulation. `PhysicalDose.peak()` "
    "is `value.abs().max()`, the largest single CARTESIAN COMPONENT = %.4f V/m. The "
    "assertion in tests/intervene/test_tms_efield.py:350 reads dose.peak() but its "
    "comment records the max-norm value; both are reported here so the difference is "
    "not silently inherited."
) % (float(m.max()), float(dose.peak()))
print(peak_note)

if bem_ok:
    mb = mag_bem.numpy()
    rel = float(np.abs(mb - m).mean() / np.abs(m).mean())
    cross = {
        "available": True,
        "solver": "charge BEM (LayeredSphereBEM, graded icosphere)",
        "peak_magnitude_V_per_m": float(mb.max()),
        "mean_magnitude_V_per_m": float(mb.mean()),
        "mean_relative_difference_vs_analytic": rel,
        "max_relative_difference_vs_analytic": float(
            (np.abs(mb - m) / np.maximum(np.abs(m), 1e-12)).max()
        ),
        "wall_seconds": round(t_bem, 2),
    }
    print("BEM cross-check:", json.dumps({k: v for k, v in cross.items() if k != "solver"}))
else:
    cross = {"available": False, "reason": bem_err, "wall_seconds": round(t_bem, 2)}
    print("BEM cross-check FAILED:", bem_err)


# ---- stored gate results ---------------------------------------------------
def load_gate(*cands):
    for c in cands:
        p = REPO / c
        if p.exists():
            raw = p.read_bytes()
            return {
                "path": c,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": json.loads(raw),
            }
    return {"path": None, "error": "not found: " + ", ".join(cands)}


n6 = load_gate("reports/intervene/N6_induced_efield.json", "reports/gates/numerics/N6_induced_efield.json")
n8 = load_gate(
    "reports/intervene/N8_induced_efield_contact.json",
    "reports/gates/numerics/N8_induced_efield_contact.json",
)


def metrics_of(gate: dict) -> dict:
    """Flatten subchecks[].metrics[] into {dotted_metric_name: {...}}."""
    out = {}
    for sc in gate.get("content", {}).get("subchecks", []):
        for mt in sc.get("metrics", []):
            out[mt["name"]] = {
                "value": mt.get("value"),
                "units": mt.get("units"),
                "threshold": mt.get("threshold"),
                "direction": mt.get("direction"),
                "interval": mt.get("interval"),
                "passed": mt.get("passed"),
                "subcheck": sc.get("name"),
                "subcheck_status": sc.get("status"),
            }
    return out


gates = {
    "N6_standoff": {
        "source": n6.get("path"),
        "source_sha256": n6.get("sha256"),
        "claim_id": n6.get("content", {}).get("claim_id"),
        "status": n6.get("content", {}).get("status"),
        "claim_text": n6.get("content", {}).get("manifest", {}).get("claim_text"),
        "acceptance_thresholds": n6.get("content", {}).get("manifest", {}).get("acceptance_thresholds"),
        "metrics": metrics_of(n6),
        "note": "validates the charge-BEM solver against an independently derived spectral reference, STANDOFF geometry only",
    },
    "N8_contact": {
        "source": n8.get("path"),
        "source_sha256": n8.get("sha256"),
        "claim_id": n8.get("content", {}).get("claim_id"),
        "status": n8.get("content", {}).get("status"),
        "claim_text": n8.get("content", {}).get("manifest", {}).get("claim_text"),
        "acceptance_thresholds": n8.get("content", {}).get("manifest", {}).get("acceptance_thresholds"),
        "metrics": metrics_of(n8),
        "note": "near-contact geometry, graded mesh; validated against AxialInductionReference",
    },
}
for gname, g in gates.items():
    print(f"{gname}: status={g['status']}")
    for mn, mv in g["metrics"].items():
        print(f"   {mn} = {mv['value']}  (thr {mv['threshold']}, passed {mv['passed']})")

R = 6


def rnd(x):
    return [float("%.6g" % v) for v in np.asarray(x).ravel()]


block = {
    "title": "TMS induced E-field on a cortical surface, figure-eight coil at a realistic pose",
    "status": "ran",
    "what_it_is": (
        "Quasi-static induced E-field from a 70 mm figure-eight coil, biphasic pulse, "
        "4 mm standoff, evaluated on a 2562-vertex cortical shell of a layered spherical "
        "head. Field physics only - no neural response, no target engagement."
    ),
    "geometry": {
        "head_model": "SphericalHeadModel (layered analytic sphere)",
        "scalp_radius_m": head.radius,
        "layer_radii_m": list(head.radii),
        "conductivities_S_per_m": list(head.conductivities),
        "cortex_radius_m": head.cortex_radius,
        "n_vertices": int(pts.shape[0]),
        "coil": "FigureEightCoil (device_id=%s, wing_separation=%.3f m)" % (coil.device_id, coil.wing_separation),
        "n_coil_dipoles": int(coil.dipole_elements()[0].shape[0]),
        "pulse": "biphasic, peak_didt=%.3g A/s" % pulse.peak_didt,
        "scalp_contact_direction": DIRECTION,
        "standoff_m": STANDOFF_M,
        "handle_azimuth_deg": AZIMUTH_DEG,
        "is_subject_anatomy": False,
        "anatomy_note": (
            "This is an ANALYTIC SPHERE, not a head. run_field_gates.GEOMETRY_PROVENANCE "
            "records uses_subject_anatomy=False and uses_scwbd_anatomy_load_anatomy=False. "
            "The repo does own real cortical surfaces (scwbd/anatomy/geometry.py) but they "
            "are not wired to the field solver."
        ),
    },
    "distribution": dist,
    "peak_definition_note": peak_note,
    "series": {
        "efield_magnitude_V_per_m": {
            "units": "V/m",
            "shape": [int(m.shape[0])],
            "note": "one value per cortical-shell vertex, same order as vertex_positions_m",
            "values": rnd(m),
        },
        "vertex_positions_m": {
            "units": "m (head frame, origin at head centre)",
            "shape": list(pts.shape),
            "values": [rnd(r) for r in pts.numpy()],
        },
        "efield_normal_component_V_per_m": {
            "units": "V/m",
            "note": "E dot outward cortical normal",
            "values": rnd(E_n.numpy()),
        },
        "histogram": {
            "units": "V/m",
            "bin_edges": rnd(edges),
            "counts": [int(c) for c in counts],
        },
    },
    "bem_cross_check": cross,
    "validation_gates_as_stored": gates,
    "provenance": {
        "produced_by": "scratchpad/demo3_tms.py",
        "device": "cpu",
        "dtype": "float64",
        "torch": torch.__version__,
        "solver": "analytic layered-sphere (scwbd.intervene.tms.efield.efield_from_coil, solver='analytic')",
        "checkpoint": None,
        "loaded_key_count": None,
        "checkpoint_note": "no trained checkpoint is involved in this demo - this is field physics, not the neural model",
        "code_lineage": "main repo working tree (scwbd/intervene is ahead of the turing worktree and carries the geometry validation the worktree lacks)",
        "repo_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
        ).stdout.strip(),
        "analytic_wall_seconds": round(t_analytic, 3),
        "wall_seconds": round(time.time() - t0, 2),
    },
}

OUT.write_text(json.dumps(block, indent=1))
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.2f} MB, {time.time()-t0:.1f}s)")
