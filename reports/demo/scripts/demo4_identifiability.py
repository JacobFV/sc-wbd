"""Demo 4 - extract (do not recompute) the identifiability numbers.

Source of truth: reports/identifiability/results.json on master.
Writes: scratchpad/part_demo4.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path("/home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics")
SRC = REPO / "reports/identifiability/results.json"
OUT = pathlib.Path(sys.argv[1])

raw = SRC.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
d = json.loads(raw)

dec = d["decision"]
regimes = d["results"]["regimes"]

designs_order = [
    "eeg_only",
    "fmri_only",
    "joint_native",
    "joint_resampled",
    "joint_resampled_exactmodel",
    "joint_native_impulse",
    "joint_native_impulse_matched",
]

lam = {}
delay = {}
for rname, rv in dec["per_regime"].items():
    lam[rname] = {k: v for k, v in rv["theta_profile_min_eigenvalue_nonprior"].items()}
    delay[rname] = {
        k: (None if v is None else v * 1e3)
        for k, v in rv["delay_rmse_seconds"].items()
    }

# standard errors on the delay RMSE come from the per-design recovery blocks
delay_se = {}
true_delay_ms = {}
for rname, rv in regimes.items():
    true_delay_ms[rname] = rv["eta_true_natural"]["tau"] * 1e3
    delay_se[rname] = {}
    for dname, dv in rv["designs"].items():
        de = dv.get("recovery", {}).get("delay_error")
        if de is None:
            continue
        se = de.get("rmse_seconds_se")
        delay_se[rname][dname] = None if se is None else se * 1e3

git_sha = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()

# The headline "native vs naive-resampling" contrast, stated per regime rather
# than as a single pair, because the value differs by regime and one regime is
# explicitly flagged degenerate by the benchmark itself.
contrast = {}
for rname in delay:
    rv_deg = bool(dec["per_regime"][rname]["delay_comparison_degenerate"])
    contrast[rname] = {
        "joint_native_ms": delay[rname]["joint_native"],
        "joint_native_ms_se": delay_se.get(rname, {}).get("joint_native"),
        "joint_resampled_ms": delay[rname]["joint_resampled"],
        "joint_resampled_ms_se": delay_se.get(rname, {}).get("joint_resampled"),
        "true_delay_ms": true_delay_ms.get(rname),
        "degenerate": rv_deg,
        "note": (
            "DEGENERATE: the true delay equals the prior mean, so a design that learns "
            "nothing about the delay scores a perfect delay error. Not discriminating."
            if rv_deg
            else "discriminating: true delay is placed away from the prior mean"
        ),
    }

block = {
    "title": "Identifiability: theta-profile information and delay recovery by design",
    "status": "extracted",
    "what_it_is": (
        "Fisher/profile-likelihood identifiability benchmark on a 3-region synthetic "
        "generator, comparing observation designs. Extracted verbatim from a committed "
        "artifact; nothing here was recomputed for the demo."
    ),
    "series": {
        "theta_profile_min_eigenvalue_nonprior": {
            "units": "dimensionless (min eigenvalue of the likelihood-only profile information for the preregistered theta subset)",
            "designs": designs_order,
            "by_regime": lam,
            "higher_is_better": True,
        },
        "delay_rmse": {
            "units": "milliseconds",
            "designs": designs_order,
            "by_regime": delay,
            "standard_error_ms_by_regime": delay_se,
            "true_delay_ms_by_regime": true_delay_ms,
            "lower_is_better": True,
        },
        "native_vs_resampled_delay_contrast": contrast,
    },
    "verdict_as_recorded": {
        "criteria_all_regimes": dec["criteria_all_regimes"],
        "C2_native_beats_resampled": dec["criteria_all_regimes"]["C2_native_beats_resampled"],
        "delay_degenerate_regimes": dec["delay_degenerate_regimes"],
        "convergence_gated_regimes": dec["convergence_gated_regimes"],
        "consequence": dec["consequence"],
    },
    "provenance": {
        "produced_by": "extraction only - json.load of a committed artifact",
        "source_file": "reports/identifiability/results.json",
        "source_sha256": sha,
        "repo_git_sha": git_sha,
        "device": "n/a (no compute performed)",
        "checkpoint": "n/a - this benchmark does not use the trained checkpoint",
        "loaded_key_count": None,
        "benchmark_environment_as_recorded": d.get("environment"),
    },
}

OUT.write_text(json.dumps(block, indent=1))
print("wrote", OUT)
print("C2_native_beats_resampled:", dec["criteria_all_regimes"]["C2_native_beats_resampled"])
for r, c in contrast.items():
    print(
        f"  {r}: native={c['joint_native_ms']:.4f} ms  resampled={c['joint_resampled_ms']:.4f} ms  "
        f"degenerate={c['degenerate']}"
    )
