"""BIDS-EEG loader (BrainVision), used for OpenNeuro ds004024 (TMS-EEG).

The BIDS sidecars are the point of this loader: ``channels.tsv`` gives the unit
and per-channel filter settings, ``electrodes.tsv`` gives *digitised* electrode
positions for this participant, and ``coordsystem.json`` names the frame those
positions live in (CapTrak, metres) together with the anatomical landmarks.
That is what makes a subject-specific lead field possible, so all of it is
carried through into :class:`~scwbd.sources.loaders.base.NativeRecording`.

The native rate of ds004024 is 20 kHz - far above anything the model will run
at.  It is returned unchanged: downsampling is the consumer's declared
operation with a declared group delay, not a silent property of loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..lineage import Lineage, Record
from .base import Events, NativeRecording

SOURCE_ID = "ds004024"

UNIT_TO_SI = {"µV": (1e-6, "V"), "uV": (1e-6, "V"), "mV": (1e-3, "V"), "V": (1.0, "V")}
TYPE_MAP = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc"}


def root(data_root: str | Path | None = None) -> Path:
    from ..registry import get

    if data_root is not None:
        return Path(data_root)
    return get(SOURCE_ID).local_path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def _sidecar(vhdr: Path, suffix: str, *, entity_strip: Sequence[str] = ()) -> Path | None:
    """Find a BIDS sidecar by progressively dropping entities (inheritance)."""
    stem = vhdr.name
    for drop in ("_eeg.vhdr",):
        if stem.endswith(drop):
            stem = stem[: -len(drop)]
    parts = stem.split("_")
    for i in range(len(parts), 0, -1):
        cand = vhdr.parent / ("_".join(parts[:i]) + suffix)
        if cand.exists():
            return cand
    return None


def iter_vhdr(data_root: str | Path | None = None, *, task: str | None = None) -> Iterator[Path]:
    base = root(data_root)
    if not base.exists():
        return
    for p in sorted(base.rglob("*_eeg.vhdr")):
        if task is None or f"task-{task}_" in p.name:
            yield p


def _entities(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in path.name.split("_"):
        if "-" in token:
            k, _, v = token.partition("-")
            out[k] = v
    return out


def load_brainvision_run(
    path: str | Path,
    *,
    preload: bool = True,
    source_id: str = SOURCE_ID,
) -> NativeRecording:
    """Load one BIDS BrainVision run with digitised electrodes and events."""
    import mne

    vhdr = Path(path)
    if not vhdr.exists():
        raise FileNotFoundError(vhdr)
    raw = mne.io.read_raw_brainvision(vhdr, preload=preload, verbose="error")
    ent = _entities(vhdr)

    ch_tsv = _sidecar(vhdr, "_channels.tsv")
    chan_meta = {r["name"]: r for r in _read_tsv(ch_tsv)} if ch_tsv else {}
    elec_tsv = _sidecar(vhdr, "_electrodes.tsv")
    coord_json = _sidecar(vhdr, "_coordsystem.json")
    ev_tsv = _sidecar(vhdr, "_events.tsv")
    side_json = _sidecar(vhdr, "_eeg.json")

    names = tuple(raw.ch_names)
    # MNE returns volts for EEG/EOG/ECG/EMG regardless of the file's uV storage.
    data = raw.get_data() if preload else raw.get_data(start=0, stop=0)
    units: list[str] = []
    types: list[str] = []
    file_units: dict[str, str] = {}
    for n, t in zip(names, raw.get_channel_types()):
        row = chan_meta.get(n, {})
        file_units[n] = row.get("units", "unknown")
        units.append(UNIT_TO_SI.get(row.get("units", ""), (1.0, "V"))[1])
        types.append(TYPE_MAP.get(str(row.get("type", "")).upper(), t))

    positions: dict[str, tuple[float, float, float]] = {}
    frame_id = "unknown"
    landmarks: dict[str, Any] = {}
    if elec_tsv is not None:
        for r in _read_tsv(elec_tsv):
            try:
                positions[r["name"]] = (float(r["x"]), float(r["y"]), float(r["z"]))
            except (KeyError, ValueError):
                continue
    if coord_json is not None:
        cs = json.loads(coord_json.read_text())
        sysname = cs.get("EEGCoordinateSystem", "unknown")
        cunits = cs.get("EEGCoordinateUnits", "unknown")
        frame_id = f"{sysname}_{cunits}" if sysname != "unknown" else "unknown"
        landmarks = cs.get("AnatomicalLandmarkCoordinates", {})

    events = None
    if ev_tsv is not None:
        rows = _read_tsv(ev_tsv)
        if rows:
            onset = np.array([float(r["onset"]) for r in rows])
            dur = np.array([float(r.get("duration", 0) or 0) for r in rows])
            lab = np.array([r.get("trial_type", r.get("value", "n/a")) for r in rows], dtype=object)
            samp = None
            if "sample" in rows[0]:
                samp = np.array([int(float(r["sample"])) for r in rows], dtype=np.int64)
            events = Events(
                onset=onset,
                duration=dur,
                label=lab,
                clock_id=f"{source_id}.brainvision_amp",
                sample=samp,
                meta={"events_tsv": str(ev_tsv)},
            )

    sidecar: dict[str, Any] = json.loads(side_json.read_text()) if side_json else {}
    sub = ent.get("sub", "unknown")
    ses = ent.get("ses", "unknown")
    return NativeRecording(
        source_id=source_id,
        data=data,
        units=tuple(units),
        sfreq=float(raw.info["sfreq"]),
        channel_names=names,
        channel_types=tuple(types),
        frame_id=frame_id,
        clock_id=f"{source_id}.brainvision_amp",
        lineage=Lineage(
            participant=f"sub-{sub}",
            family=f"singleton:sub-{sub}",
            site="openneuro:ds004024:shirley_ryan_abilitylab",
            device=f"BrainProducts:{sidecar.get('Manufacturer', 'unknown')}",
            session=f"ses-{ses}",
            run=ent.get("run", "unknown"),
        ),
        channel_positions=positions or None,
        montage=str(sidecar.get("EEGPlacementScheme", "unknown")),
        events=events,
        meta={
            "path": str(vhdr),
            "task": ent.get("task", "unknown"),
            "file_units": file_units,
            "sidecar": sidecar,
            "anatomical_landmarks": landmarks,
            "highpass_hz": float(raw.info["highpass"]),
            "lowpass_hz": float(raw.info["lowpass"]),
            "line_freq_hz": sidecar.get("PowerLineFrequency", "unknown"),
            "reference": sidecar.get("EEGReference", "unknown"),
            "electrodes_tsv": str(elec_tsv) if elec_tsv else None,
            "coordsystem_json": str(coord_json) if coord_json else None,
        },
    )


def records(data_root: str | Path | None = None) -> list[Record]:
    out: list[Record] = []
    for p in iter_vhdr(data_root):
        ent = _entities(p)
        sub = f"sub-{ent.get('sub', 'unknown')}"
        task = ent.get("task", "unknown")
        eeg_bin = p.with_name(p.name.replace(".vhdr", ".eeg"))
        out.append(
            Record(
                id=p.name.replace("_eeg.vhdr", ""),
                source_id=SOURCE_ID,
                lineage=Lineage(
                    participant=sub,
                    family=f"singleton:{sub}",
                    site="openneuro:ds004024:shirley_ryan_abilitylab",
                    device="BrainProducts_actiCHamp_64ch",
                    session=f"ses-{ent.get('ses', 'unknown')}",
                    run=ent.get("run", "unknown"),
                ),
                stimulus_ids=(f"task-{task}",),
                path=str(p),
                n_bytes=eeg_bin.stat().st_size if eeg_bin.exists() else None,
                meta={"task": task},
            )
        )
    return out


__all__ = ("SOURCE_ID", "load_brainvision_run", "records", "iter_vhdr")
