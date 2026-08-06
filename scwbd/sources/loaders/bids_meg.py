"""BIDS-MEG loader (Neuromag/Elekta FIFF), used for OpenNeuro ds000117.

The Wakeman-Henson dataset is the cross-modal benchmark: 306 MEG channels
(204 planar gradiometers + 102 magnetometers) and 70 EEG channels recorded
**simultaneously** on the same subjects, plus fMRI, dMRI and structural MRI in
a second session.  A single fif file therefore contains three different
physical units (T, T/m, V) on one clock - which is exactly why
:class:`~scwbd.sources.loaders.base.NativeRecording` carries a *per-channel*
unit tuple instead of one unit for the file.

Head-position information (device->head transform and the digitised head
shape) is preserved in ``meta`` so that agent F can build a lead field without
re-deriving coregistration from scratch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..lineage import Lineage, Record
from .base import Events, NativeRecording

SOURCE_ID = "ds000117"

#: FIFF channel kind -> (scwbd channel type, SI unit)
FIFF_UNITS = {
    "mag": "T",
    "grad": "T/m",
    "eeg": "V",
    "eog": "V",
    "ecg": "V",
    "stim": "dimensionless",
    "misc": "dimensionless",
    "chpi": "dimensionless",
    "syst": "dimensionless",
    "ias": "dimensionless",
}


def root(data_root: str | Path | None = None) -> Path:
    from ..registry import get

    if data_root is not None:
        return Path(data_root)
    return get(SOURCE_ID).local_path


def iter_fif(data_root: str | Path | None = None) -> Iterator[Path]:
    base = root(data_root)
    if not base.exists():
        return
    for p in sorted(base.rglob("*_meg.fif")):
        if "derivatives" in p.parts:
            continue
        yield p


def _entities(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in path.name.split("_"):
        if "-" in token:
            k, _, v = token.partition("-")
            out[k] = v
    return out


def _read_tsv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def load_fif_run(
    path: str | Path,
    *,
    preload: bool = True,
    picks: str | None = None,
) -> NativeRecording:
    """Load one MEG(+EEG) run at its native rate with per-channel units.

    ``picks`` may be ``"meg"``, ``"eeg"`` or ``None`` (everything).  Picking is
    channel selection, not resampling: the rate is untouched either way.
    """
    import mne

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    raw = mne.io.read_raw_fif(p, preload=False, verbose="error", allow_maxshield=True)
    if picks is not None:
        raw = raw.pick(picks)
    if preload:
        raw.load_data(verbose="error")
    ent = _entities(p)

    types = tuple(raw.get_channel_types())
    units = tuple(FIFF_UNITS.get(t, "unknown") for t in types)
    data = raw.get_data() if preload else np.zeros((len(raw.ch_names), 0))

    positions: dict[str, tuple[float, float, float]] = {}
    for ch in raw.info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=float)
        if np.any(np.isfinite(loc)) and np.linalg.norm(loc) > 0:
            positions[ch["ch_name"]] = tuple(float(x) for x in loc)  # type: ignore[assignment]

    dev_head_t = raw.info.get("dev_head_t")
    events = None
    ev_tsv = p.with_name(p.name.replace("_meg.fif", "_events.tsv"))
    if ev_tsv.exists():
        rows = _read_tsv(ev_tsv)
        if rows:
            onset = np.array([float(r["onset"]) for r in rows])
            dur = np.array([float(r.get("duration", 0) or 0) for r in rows])
            key = "trial_type" if "trial_type" in rows[0] else "value"
            lab = np.array([r.get(key, "n/a") for r in rows], dtype=object)
            samp = None
            if "sample" in rows[0]:
                try:
                    samp = np.array([int(float(r["sample"])) for r in rows], dtype=np.int64)
                except ValueError:
                    samp = None
            events = Events(
                onset=onset,
                duration=dur,
                label=lab,
                clock_id=f"{SOURCE_ID}.neuromag_acq",
                sample=samp,
                meta={"events_tsv": str(ev_tsv)},
            )

    sub = f"sub-{ent.get('sub', 'unknown')}"
    return NativeRecording(
        source_id=SOURCE_ID,
        data=data,
        units=units,
        sfreq=float(raw.info["sfreq"]),
        channel_names=tuple(raw.ch_names),
        channel_types=types,
        frame_id="neuromag_device (dev_head_t to head_CTF-like RAS in meta)",
        clock_id=f"{SOURCE_ID}.neuromag_acq",
        lineage=Lineage(
            participant=sub,
            family=f"singleton:{sub}",
            site="openneuro:ds000117:mrc_cbu_cambridge",
            device="Elekta_Neuromag_Vectorview_306",
            session=f"ses-{ent.get('ses', 'meg')}",
            run=ent.get("run", "unknown"),
        ),
        channel_positions=positions or None,
        montage="Neuromag 306 helmet + 70-channel EasyCap EEG (simultaneous)",
        events=events,
        meta={
            "path": str(p),
            "task": ent.get("task", "unknown"),
            "n_mag": sum(1 for t in types if t == "mag"),
            "n_grad": sum(1 for t in types if t == "grad"),
            "n_eeg": sum(1 for t in types if t == "eeg"),
            "highpass_hz": float(raw.info["highpass"]),
            "lowpass_hz": float(raw.info["lowpass"]),
            "line_freq_hz": raw.info.get("line_freq"),
            "dev_head_t": None if dev_head_t is None else np.asarray(dev_head_t["trans"]).tolist(),
            "n_digitisation_points": 0 if raw.info["dig"] is None else len(raw.info["dig"]),
            "meas_date": str(raw.info.get("meas_date")),
        },
    )


def stimulus_ids_for(meg_or_bold: Path) -> tuple[str, ...]:
    """Individual stimulus identities presented in this run.

    Appendix D's "Stimulus memorization" control needs the *image*, not the
    condition: the same face photograph recurs within and across runs, so
    holding out "famous" while training on the same famous faces proves
    nothing. ds000117's events.tsv carries a ``stim_file`` column, and the
    basename of that file is the identity we hold out.  When no stim_file
    column exists the function returns ``("unknown",)`` so a stimulus-level
    split refuses rather than silently grouping everything together.
    """
    ev = meg_or_bold.with_name(
        meg_or_bold.name.replace("_meg.fif", "_events.tsv").replace(
            "_bold.nii.gz", "_events.tsv"
        )
    )
    if not ev.exists():
        return ("unknown",)
    rows = _read_tsv(ev)
    if not rows or "stim_file" not in rows[0]:
        return ("unknown",)
    return tuple(sorted({Path(r["stim_file"]).name for r in rows if r.get("stim_file")}))


def records(data_root: str | Path | None = None) -> list[Record]:
    """Records for MEG runs plus the MRI/fMRI derivatives of the same session.

    fMRI runs of the same participant are *not* independent samples: they are
    emitted with the same participant/family key so that a participant-level
    split keeps a person's MEG and BOLD on the same side.
    """
    out: list[Record] = []
    base = root(data_root)
    for p in iter_fif(data_root):
        ent = _entities(p)
        sub = f"sub-{ent.get('sub', 'unknown')}"
        out.append(
            Record(
                id=p.name.replace("_meg.fif", ""),
                source_id=SOURCE_ID,
                lineage=Lineage(
                    participant=sub,
                    family=f"singleton:{sub}",
                    site="openneuro:ds000117:mrc_cbu_cambridge",
                    device="Elekta_Neuromag_Vectorview_306",
                    session=f"ses-{ent.get('ses', 'meg')}",
                    run=ent.get("run", "unknown"),
                ),
                stimulus_ids=stimulus_ids_for(p),
                path=str(p),
                n_bytes=p.stat().st_size,
                meta={"modality": "meg+eeg", "task": ent.get("task", "unknown")},
            )
        )
    if base.exists():
        for p in sorted(base.rglob("*_bold.nii.gz")):
            ent = _entities(p)
            sub = f"sub-{ent.get('sub', 'unknown')}"
            out.append(
                Record(
                    id=p.name.replace("_bold.nii.gz", ""),
                    source_id=SOURCE_ID,
                    lineage=Lineage(
                        participant=sub,
                        family=f"singleton:{sub}",
                        site="openneuro:ds000117:mrc_cbu_cambridge",
                        device="Siemens_TrioTim_3T",
                        session=f"ses-{ent.get('ses', 'mri')}",
                        run=ent.get("run", "unknown"),
                    ),
                    stimulus_ids=stimulus_ids_for(p),
                    path=str(p),
                    n_bytes=p.stat().st_size,
                    meta={"modality": "bold", "task": ent.get("task", "unknown")},
                )
            )
    return out


__all__ = ("SOURCE_ID", "load_fif_run", "records", "iter_fif")
