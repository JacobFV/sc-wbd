"""Write ds000117's digitised EEG montage to `configs/montages/ds000117_eeg.json`.

The Wakeman-Henson EEG channels are named ``EEG001``..``EEG074``. No standard
table resolves those names, so the electrode geometry has to come from the
recordings, where it is digitised per participant in the ``fif`` header.

This project uses **one** head model for every participant (there are no
individual forward solutions; see ``LeadField.is_individual``), so a single
montage is what the operator can consume. The one written here is the mean over
the fetched participants' digitisations, and the file records the spread it was
taken over so a reader can see how much was averaged away rather than being
handed a number with no dispersion.

Run: PYTHONPATH=. .venv/bin/python scripts/extract_ds000117_montage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/ds000117/1.1.0"
OUT = REPO / "configs/montages/ds000117_eeg.json"

#: Channels the release marks bad in every run of every fetched participant.
#: ``EEG061``..``EEG064`` are the EOG/ECG block on this cap and are not scalp
#: potentials; they are excluded from the observation montage and picked up as
#: boundary outputs instead.
NON_SCALP = ("EEG061", "EEG062", "EEG063", "EEG064")


def main() -> int:
    import mne

    runs = sorted(ROOT.glob("sub-*/ses-meg/meg/*_task-facerecognition_run-*_meg.fif"))
    if not runs:
        raise SystemExit(f"no MEG runs under {ROOT}")

    per_subject: dict[str, list[np.ndarray]] = {}
    names: tuple[str, ...] | None = None
    for f in runs:
        sub = f.parts[-4]
        raw = mne.io.read_raw_fif(f, preload=False, verbose="ERROR")
        chs = [c for c in raw.info["chs"] if c["kind"] == 2]  # FIFFV_EEG_CH
        keep = [c for c in chs if c["ch_name"] not in NON_SCALP]
        n = tuple(c["ch_name"] for c in keep)
        if names is None:
            names = n
        elif n != names:
            raise SystemExit(f"{f}: channel order differs from the first run")
        loc = np.array([c["loc"][:3] for c in keep], dtype=np.float64) * 1000.0  # m -> mm
        if not np.isfinite(loc).all() or not (np.abs(loc).sum(axis=1) > 0).all():
            print(f"  skip {f.name}: incomplete digitisation")
            continue
        per_subject.setdefault(sub, []).append(loc)

    assert names is not None
    # One digitisation per participant first (runs of a session share a cap
    # placement), then the mean across participants -- so a participant with
    # more runs does not weigh more.
    by_sub = {s: np.mean(np.stack(v), axis=0) for s, v in sorted(per_subject.items())}
    stack = np.stack([by_sub[s] for s in sorted(by_sub)])
    mean = stack.mean(axis=0)
    spread = float(np.linalg.norm(stack - mean, axis=-1).mean()) if len(stack) > 1 else 0.0

    payload = {
        "source_id": "ds000117_real",
        "dataset": "ds000117 v1.1.0 (Wakeman-Henson)",
        "units": "mm",
        "frame": "MEG device / head coordinates as digitised in the fif header",
        "channels": list(names),
        "n_channels": len(names),
        "positions_mm": [[round(float(x), 4) for x in row] for row in mean],
        "derivation": (
            "Mean across participants of the per-participant mean over runs of the "
            "digitised EEG channel positions in the fif headers. One montage, not "
            "one per participant, because this project fits a single head model "
            "for every participant and a per-participant operator would claim an "
            "individual forward solution it does not have."
        ),
        "participants_averaged": sorted(by_sub),
        "n_participants_averaged": len(by_sub),
        "n_runs_read": sum(len(v) for v in per_subject.values()),
        "mean_across_participant_deviation_mm": round(spread, 3),
        "excluded_non_scalp": list(NON_SCALP),
        "note": (
            "EEG001..EEG074 are cap positions, not 10-10 labels; no standard table "
            "resolves them, so the geometry is the recorded digitisation. "
            f"Averaged over {len(by_sub)} participant(s); the mean electrode "
            f"deviates {spread:.2f} mm from that average, which bounds how much "
            "individual head geometry this single operator discards."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    r = np.linalg.norm(mean, axis=1)
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  {len(names)} channels over {len(by_sub)} participants, "
          f"{payload['n_runs_read']} runs")
    print(f"  radius {r.mean():.1f} +/- {r.std():.1f} mm, "
          f"between-participant deviation {spread:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
