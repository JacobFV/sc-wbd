"""Assemble reports/demo/demo_data.json from the four part files."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess

SP = pathlib.Path(__file__).parent
REPO = pathlib.Path(
    "/home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics"
)
OUT = REPO / "reports/demo/demo_data.json"

parts = {}
for n in (1, 2, 3, 4):
    p = SP / f"part_demo{n}.json"
    if p.exists():
        parts[f"demo{n}"] = json.loads(p.read_text())
    else:
        parts[f"demo{n}"] = {
            "status": "could_not_run",
            "reason": f"part_demo{n}.json was never produced; see DEMO.md",
        }

doc = {
    "artifact": "SC-WBD-001-beta live demonstration",
    "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "produced_by": "agent Ramachandran",
    "repo_git_sha": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip(),
    "read_this_first": {
        "claim_boundary": "reports/CLAIM_BOUNDARY.md",
        "summary": (
            "Every number in this file was regenerated or extracted from a committed "
            "artifact on this machine. Nothing is simulated-for-illustration and nothing "
            "is hand-tuned. The model has SYNTHETIC anatomy and a SYNTHETIC lead field, "
            "its amortised posterior does not recover its own parameters, and it has no "
            "validated held-out performance. All five claim gates G1-G5 are COULD_NOT_RUN. "
            "These demos show machinery working; none of them is evidence about brains."
        ),
        "demos": {
            "demo1": "ran - trained checkpoint, forward multirate rollout",
            "demo2": "regenerated - dynamics module, no trained checkpoint involved",
            "demo3": "ran - field physics, no trained checkpoint involved",
            "demo4": "extracted - no computation performed",
        },
    },
    "corrections_to_the_brief": [
        {
            "claim": "EEG at its native millisecond rate",
            "finding": (
                "The model has exactly TWO clocks: fast (dt_model=0.008 s, 125 Hz) and "
                "slow (dt_model*hemo_ratio=0.2 s, 5 Hz). The EEG head is a memoryless "
                "per-timestep map over the state and therefore emits at 125 Hz. There is "
                "no millisecond EEG clock in scwbd.foundation. The demonstrated contrast "
                "is 125 Hz vs 5 Hz."
            ),
            "where": "demo1.clock_correction",
        },
        {
            "claim": "80.2% of parameter mass in the _orig_mod keys",
            "finding": (
                "Correct against TRAINABLE parameters (1,410,297 / 1,757,613 = 80.24%). "
                "Against all state-dict tensor elements (6,733,924, including the "
                "anatomy-derived coupling buffers) the same 29 keys are 20.94%. Both "
                "denominators are recorded."
            ),
            "where": "demo1.checkpoint_load_trap",
        },
        {
            "claim": "delay RMSE native-clock 0.376 ms vs naive-resampling 5.000 ms",
            "finding": (
                "0.376 ms does not appear anywhere in reports/identifiability/. The "
                "5.000 ms figure is real: it is joint_resampled in the "
                "weak_coupling_long_delay regime, where joint_native scores 1.2346 ms "
                "(not 0.376). The per-regime table is reported instead of a single pair, "
                "and the preregistered criterion C2_native_beats_resampled is recorded "
                "as FAILED in the artifact."
            ),
            "where": "demo4.series.native_vs_resampled_delay_contrast",
        },
        {
            "claim": "peak ~134.5 V/m at baseline",
            "finding": (
                "Reproduced exactly as max||E|| = 134.5164 V/m. Note that "
                "PhysicalDose.peak() returns the largest single Cartesian COMPONENT, "
                "106.7257 V/m - a different quantity. The test that pins this reads "
                "dose.peak() but its comment records the max-norm value."
            ),
            "where": "demo3.peak_definition_note",
        },
        {
            "claim": "FC-SC peak 0.250 at G=2.0",
            "finding": (
                "Reproduced exactly on CUDA. But the source test's own docstring states "
                "the peak LOCATION is not a claim: the supracritical curve is a flat "
                "plateau whose argmax moves to G=4.0 on CPU from the same seed. What is "
                "asserted is onset at the transition and degradation under over-coupling."
            ),
            "where": "demo2.caveat_from_the_test_itself",
        },
    ],
    "demo1_multirate_forward_run": parts["demo1"],
    "demo2_criticality": parts["demo2"],
    "demo3_tms_efield": parts["demo3"],
    "demo4_identifiability": parts["demo4"],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
for k in ("demo1", "demo2", "demo3", "demo4"):
    print(f"  {k}: {parts[k].get('status')}")
