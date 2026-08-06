"""Loader for the PhysioNet EEG Motor Movement/Imagery Database (eegmmidb).

109 volunteers, BCI2000, 64 EEG channels in the international 10-10 subset of
the Sharbrough montage, 160 Hz, 14 runs each.  Signals are stored in EDF+ with
physical dimension ``uV`` and a 1 uV quantisation step.

Run semantics (from the PhysioNet record description):

===== ==========================================================
run   task
===== ==========================================================
R01   baseline, eyes open
R02   baseline, eyes closed
R03/07/11  open and close **left or right fist** (executed)
R04/08/12  **imagine** opening and closing left or right fist
R05/09/13  open and close **both fists or both feet** (executed)
R06/10/14  **imagine** opening/closing both fists or both feet
===== ==========================================================

Annotation codes are context dependent: ``T0`` is rest everywhere, while
``T1``/``T2`` mean left/right fist in the fist runs and both-fists/both-feet in
the feet runs.  The loader resolves them into explicit labels so that a
downstream stimulus-level split never groups "T1" across incompatible runs.

Nothing here filters, re-references or resamples.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..lineage import Lineage, Record
from .base import Events, NativeRecording
from .edf import normalise_unit, rate_groups, read_signals

SOURCE_ID = "eegmmidb"
DATASET_VERSION = "1.0.0"

BASELINE_RUNS = {1: "baseline_eyes_open", 2: "baseline_eyes_closed"}
FIST_EXEC_RUNS = {3, 7, 11}
FIST_IMAG_RUNS = {4, 8, 12}
FEET_EXEC_RUNS = {5, 9, 13}
FEET_IMAG_RUNS = {6, 10, 14}

RUN_TASK = {}
for _r in FIST_EXEC_RUNS:
    RUN_TASK[_r] = "left_or_right_fist_executed"
for _r in FIST_IMAG_RUNS:
    RUN_TASK[_r] = "left_or_right_fist_imagined"
for _r in FEET_EXEC_RUNS:
    RUN_TASK[_r] = "fists_or_feet_executed"
for _r in FEET_IMAG_RUNS:
    RUN_TASK[_r] = "fists_or_feet_imagined"
for _r, _n in BASELINE_RUNS.items():
    RUN_TASK[_r] = _n

ANNOT_MEANING = {
    "fist": {"T0": "rest", "T1": "left_fist", "T2": "right_fist"},
    "feet": {"T0": "rest", "T1": "both_fists", "T2": "both_feet"},
    "baseline": {"T0": "rest"},
}


def _run_family(run: int) -> str:
    if run in BASELINE_RUNS:
        return "baseline"
    if run in FIST_EXEC_RUNS | FIST_IMAG_RUNS:
        return "fist"
    return "feet"


def _canonical_channel(name: str) -> str:
    """``'Fc5.'`` -> ``'FC5'``; the EDF labels are dot-padded and mixed case."""
    n = name.strip().rstrip(".").replace(".", "")
    return n.upper().replace("Z", "z") if n.upper().endswith("Z") else n.upper()


@lru_cache(maxsize=1)
def _template_positions() -> dict[str, tuple[float, float, float]]:
    """Template 10-05 electrode positions in the MNE head frame, metres.

    These are **template** coordinates, not this subject's digitised
    electrodes.  The source card therefore marks
    ``calibration.electrode_positions: unknown`` and the lead-field gradient
    path stays disabled for this source.
    """
    try:
        import mne
    except Exception:
        return {}
    mont = mne.channels.make_standard_montage("standard_1005")
    pos = mont.get_positions()["ch_pos"]
    return {k.upper(): tuple(float(x) for x in v) for k, v in pos.items()}


def root(data_root: str | Path | None = None) -> Path:
    from ..registry import get

    if data_root is not None:
        return Path(data_root)
    return get(SOURCE_ID).local_path


def run_path(subject: int | str, run: int, data_root: str | Path | None = None) -> Path:
    sub = subject if isinstance(subject, str) else f"S{subject:03d}"
    return root(data_root) / sub / f"{sub}R{run:02d}.edf"


def iter_run_paths(data_root: str | Path | None = None) -> Iterator[Path]:
    base = root(data_root)
    for sub in sorted(p for p in base.glob("S[0-9][0-9][0-9]") if p.is_dir()):
        yield from sorted(sub.glob("*.edf"))


def load_run(
    subject: int | str,
    run: int,
    *,
    data_root: str | Path | None = None,
    path: str | Path | None = None,
) -> NativeRecording:
    """Load one run at its native 160 Hz, in volts, with resolved event labels."""
    p = Path(path) if path is not None else run_path(subject, run, data_root)
    if not p.exists():
        raise FileNotFoundError(f"eegmmidb run not on disk: {p}")
    sub = p.parent.name
    run_no = int(p.stem[-2:])
    header, signals, annots = read_signals(p)
    groups = rate_groups(header)
    if len(groups) != 1:
        raise ValueError(f"{p}: expected one native rate, found {sorted(groups)}")
    (sfreq, idx), = groups.items()

    data_uv = np.stack([signals[i] for i in idx], axis=0)
    dims = {normalise_unit(header.signals[i].physical_dim) for i in idx}
    if dims != {"uV"}:
        raise ValueError(f"{p}: unexpected physical dimension(s) {dims}")
    # EDF stores microvolts; SI volts are the declared port unit.
    data = data_uv * 1e-6
    names = tuple(_canonical_channel(header.signals[i].label) for i in idx)
    tpos = _template_positions()
    positions = {n: v for n, v in ((n, tpos.get(n.upper())) for n in names) if v}

    fam = _run_family(run_no)
    mapping = ANNOT_MEANING[fam]
    onsets, durs, labels = [], [], []
    for onset, dur, lab in annots:
        lab = lab.strip()
        if not lab:
            continue
        onsets.append(onset)
        durs.append(dur)
        labels.append(mapping.get(lab, f"{lab}_unmapped"))
    events = Events(
        onset=np.asarray(onsets, dtype=np.float64),
        duration=np.asarray(durs, dtype=np.float64),
        label=np.asarray(labels, dtype=object),
        clock_id="eegmmidb.bci2000_amp",  # annotations share the amplifier clock
        sample=np.round(np.asarray(onsets, dtype=np.float64) * sfreq).astype(np.int64)
        if onsets
        else None,
        meta={"raw_codes": sorted({a[2] for a in annots}), "run_family": fam},
    )

    return NativeRecording(
        source_id=SOURCE_ID,
        data=data,
        units=("V",) * len(names),
        sfreq=float(sfreq),
        channel_names=names,
        channel_types=("eeg",) * len(names),
        frame_id="template_10_05_head_RAS",
        clock_id="eegmmidb.bci2000_amp",
        lineage=Lineage(
            participant=sub,
            family=f"singleton:{sub}",
            site="physionet:eegmmidb_single_site",
            device="BCI2000_64ch",
            session=f"{sub}_ses-01",
            run=f"R{run_no:02d}",
        ),
        channel_positions=positions or None,
        montage="standard_1005_template (NOT digitised for this subject)",
        events=events,
        t0=0.0,
        meta={
            "path": str(p),
            "task": RUN_TASK.get(run_no, "unknown"),
            "run_family": fam,
            "edf_physical_dim": "uV",
            "quantisation_volts": float(header.signals[idx[0]].gain) * 1e-6,
            "reference": "unknown (BCI2000 acquisition reference not documented per-run)",
            "record_start_utc": str(header.start),
            "n_records": header.n_records,
            "prefilter": header.signals[idx[0]].prefilter,
        },
    )


def records(
    data_root: str | Path | None = None, *, subjects: Sequence[str] | None = None
) -> list[Record]:
    """One :class:`Record` per run, for :mod:`scwbd.sources.splits`."""
    out: list[Record] = []
    for p in iter_run_paths(data_root):
        sub = p.parent.name
        if subjects is not None and sub not in subjects:
            continue
        run_no = int(p.stem[-2:])
        fam = _run_family(run_no)
        stim = tuple(f"{RUN_TASK.get(run_no,'unknown')}:{v}" for v in ANNOT_MEANING[fam].values())
        out.append(
            Record(
                id=p.stem,
                source_id=SOURCE_ID,
                lineage=Lineage(
                    participant=sub,
                    family=f"singleton:{sub}",
                    site="physionet:eegmmidb_single_site",
                    device="BCI2000_64ch",
                    session=f"{sub}_ses-01",
                    run=f"R{run_no:02d}",
                ),
                stimulus_ids=stim,
                path=str(p),
                n_bytes=p.stat().st_size,
                meta={"task": RUN_TASK.get(run_no, "unknown")},
            )
        )
    return out


__all__ = ("SOURCE_ID", "load_run", "records", "iter_run_paths", "run_path", "RUN_TASK")
