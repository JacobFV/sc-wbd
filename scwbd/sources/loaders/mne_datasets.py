"""Loaders for the MNE-Python sample / somato / spm_face datasets.

These three are in the register for one reason: they ship real MEG/EEG
recordings *together with* the subject's FreeSurfer reconstruction, BEM
surfaces, the head<->device transform and (for ``sample``) a precomputed
forward solution.  That combination is what lets agent F validate a lead field
against an independently computed one instead of trusting its own arithmetic.

Their role in the mixture is therefore ``calibration`` (sample) or a small
``likelihood`` (somato, spm_face) - never population prior: n = 1 subject each.

:func:`anatomy_paths` is the interface agent F should use; it returns the
absolute paths that exist on disk, and reports ``None`` for anything the
dataset does not ship (no invention, no substitution of a template).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..lineage import Lineage, Record
from .base import Events, NativeRecording
from .bids_meg import FIFF_UNITS

DATASETS = {
    "mne-sample": {
        "mne_name": "sample",
        "raw": "MEG/sample/sample_audvis_raw.fif",
        "subject": "sample",
        "device": "Elekta_Neuromag_Vectorview_306",
        "task": "audvis (auditory/visual, left/right + smiley)",
    },
    "mne-somato": {
        "mne_name": "somato",
        "raw": "sub-01/meg/sub-01_task-somato_meg.fif",
        "subject": "01",
        "device": "Elekta_Neuromag_Vectorview_306",
        "task": "median nerve stimulation",
    },
    "mne-spm-face": {
        "mne_name": "spm_face",
        "raw": "MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds",
        "subject": "spm",
        "device": "CTF_275",
        "task": "faces vs scrambled",
    },
}


def root(dataset_id: str, data_root: str | Path | None = None) -> Path:
    """The dataset directory as ``mne.datasets`` laid it out."""
    from ..registry import get

    base = Path(data_root) if data_root is not None else get(dataset_id).local_path
    mne_name = DATASETS[dataset_id]["mne_name"]
    # mne.datasets unpacks into a well-known folder name under `path`
    for cand in (
        base / {"sample": "MNE-sample-data", "somato": "MNE-somato-data", "spm_face": "MNE-spm-face"}[mne_name],
        base,
    ):
        if cand.exists():
            return cand
    return base


def anatomy_paths(dataset_id: str, data_root: str | Path | None = None) -> dict[str, Any]:
    """Absolute paths to MRI / BEM / transform / forward artefacts.

    Missing artefacts are reported as ``None``.  A ``None`` here disables the
    corresponding gradient path in the source card - it does not authorise
    substituting a template head.
    """
    base = root(dataset_id, data_root)
    subject = DATASETS[dataset_id]["subject"]
    subjects_dir = base / "subjects"
    sdir = subjects_dir / subject
    if not sdir.exists():
        # somato ships FreeSurfer under derivatives/freesurfer
        alt = base / "derivatives" / "freesurfer" / "subjects"
        if (alt / f"sub-{subject}").exists():
            subjects_dir, sdir = alt, alt / f"sub-{subject}"
        elif (alt / subject).exists():
            subjects_dir, sdir = alt, alt / subject

    def first(pattern: str, where: Path) -> Path | None:
        if not where.exists():
            return None
        hits = sorted(where.rglob(pattern))
        return hits[0] if hits else None

    return {
        "dataset_id": dataset_id,
        "root": str(base),
        "subjects_dir": str(subjects_dir) if subjects_dir.exists() else None,
        "subject": sdir.name if sdir.exists() else None,
        "t1_mgz": str(p) if (p := first("T1.mgz", sdir)) else None,
        "bem_surfaces": [str(q) for q in sorted(sdir.glob("bem/*.fif"))] if sdir.exists() else [],
        "bem_watershed": bool((sdir / "bem" / "watershed").exists()) if sdir.exists() else False,
        "src": str(p) if (p := first("*-src.fif", sdir)) else None,
        "trans": str(p) if (p := first("*-trans.fif", base)) else None,
        "forward": str(p) if (p := first("*-fwd.fif", base)) else None,
        "cov": str(p) if (p := first("*-cov.fif", base)) else None,
        "surf_white": [
            str(q) for q in sorted(sdir.glob("surf/*h.white"))
        ] if sdir.exists() else [],
    }


def _load_raw(dataset_id: str, data_root: str | Path | None, preload: bool) -> NativeRecording:
    import mne

    spec = DATASETS[dataset_id]
    base = root(dataset_id, data_root)
    raw_path = base / str(spec["raw"])
    if not raw_path.exists():
        hits = sorted(base.rglob("*_raw.fif")) or sorted(base.rglob("*.ds"))
        if not hits:
            raise FileNotFoundError(f"{dataset_id}: no raw file under {base}")
        raw_path = hits[0]
    if raw_path.suffix == ".ds":
        raw = mne.io.read_raw_ctf(raw_path, preload=preload, verbose="error")
    else:
        raw = mne.io.read_raw_fif(raw_path, preload=preload, verbose="error")

    types = tuple(raw.get_channel_types())
    units = tuple(FIFF_UNITS.get(t, "unknown") for t in types)
    data = raw.get_data() if preload else np.zeros((len(raw.ch_names), 0))
    positions = {}
    for ch in raw.info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=float)
        if np.any(np.isfinite(loc)) and np.linalg.norm(loc) > 0:
            positions[ch["ch_name"]] = tuple(float(x) for x in loc)

    events = None
    try:
        ev = mne.find_events(raw, verbose="error") if preload else None
    except Exception:
        ev = None
    if ev is not None and len(ev):
        events = Events(
            onset=(ev[:, 0] - raw.first_samp) / float(raw.info["sfreq"]),
            duration=np.zeros(len(ev)),
            label=np.array([str(x) for x in ev[:, 2]], dtype=object),
            clock_id=f"{dataset_id}.acq",
            sample=(ev[:, 0] - raw.first_samp).astype(np.int64),
            meta={"source": "stim channel trigger codes"},
        )

    anat = anatomy_paths(dataset_id, data_root)
    subject = str(spec["subject"])
    dev_head_t = raw.info.get("dev_head_t")
    return NativeRecording(
        source_id=dataset_id,
        data=data,
        units=units,
        sfreq=float(raw.info["sfreq"]),
        channel_names=tuple(raw.ch_names),
        channel_types=types,
        frame_id=f"{dataset_id}:device (dev_head_t + -trans.fif give head and MRI frames)",
        clock_id=f"{dataset_id}.acq",
        lineage=Lineage(
            participant=f"{dataset_id}:{subject}",
            family=f"singleton:{dataset_id}:{subject}",
            site=f"{dataset_id}:single_site",
            device=str(spec["device"]),
            session="ses-01",
            run=raw_path.stem,
        ),
        channel_positions=positions or None,
        montage=str(spec["device"]),
        events=events,
        meta={
            "path": str(raw_path),
            "task": spec["task"],
            "highpass_hz": float(raw.info["highpass"]),
            "lowpass_hz": float(raw.info["lowpass"]),
            "line_freq_hz": raw.info.get("line_freq"),
            "dev_head_t": None if dev_head_t is None else np.asarray(dev_head_t["trans"]).tolist(),
            "n_digitisation_points": 0 if raw.info["dig"] is None else len(raw.info["dig"]),
            "anatomy": anat,
            "meas_date": str(raw.info.get("meas_date")),
        },
    )


def load_sample(data_root: str | Path | None = None, *, preload: bool = True) -> NativeRecording:
    return _load_raw("mne-sample", data_root, preload)


def load_somato(data_root: str | Path | None = None, *, preload: bool = True) -> NativeRecording:
    return _load_raw("mne-somato", data_root, preload)


def load_spm_face(data_root: str | Path | None = None, *, preload: bool = True) -> NativeRecording:
    return _load_raw("mne-spm-face", data_root, preload)


def records(dataset_id: str, data_root: str | Path | None = None) -> list[Record]:
    base = root(dataset_id, data_root)
    if not base.exists():
        return []
    subject = str(DATASETS[dataset_id]["subject"])
    out: list[Record] = []
    for p in sorted(base.rglob("*_raw.fif")) + sorted(base.rglob("*_meg.fif")):
        out.append(
            Record(
                id=f"{dataset_id}:{p.stem}",
                source_id=dataset_id,
                lineage=Lineage(
                    participant=f"{dataset_id}:{subject}",
                    family=f"singleton:{dataset_id}:{subject}",
                    site=f"{dataset_id}:single_site",
                    device=str(DATASETS[dataset_id]["device"]),
                    session="ses-01",
                    run=p.stem,
                ),
                path=str(p),
                n_bytes=p.stat().st_size,
            )
        )
    return out


__all__ = (
    "DATASETS",
    "anatomy_paths",
    "load_sample",
    "load_somato",
    "load_spm_face",
    "records",
    "root",
)
