"""Loader for Sleep-EDF Database Expanded, sleep-cassette subset.

Whole-night polysomnograms recorded at home in a 1987-1991 age-effect study.
Each recording is ``SC4<ss><n><X>-PSG.edf`` with a matching
``SC4<ss><n><Y>-Hypnogram.edf``, where ``ss`` is the subject number, ``n`` the
night (1 or 2) and the trailing letter identifies the file version/scorer.
Two nights of the *same* person therefore share a participant and must never
be split apart - which is precisely why the lineage carries
``participant='SC4<ss>'`` and ``session='night<n>'``.

The recording is genuinely multirate: EEG Fpz-Cz, EEG Pz-Oz and horizontal EOG
at 100 Hz; submental EMG, oro-nasal respiration, rectal temperature and the
event marker at 1 Hz.  :func:`load_recording` returns a
:class:`~scwbd.sources.loaders.base.MultiRateRecording` with one group per
native rate.  It never upsamples the 1 Hz channels.

Hypnogram annotations (``Sleep stage W/1/2/3/4/R/?``, ``Movement time``) are
returned on the annotation clock with their true 30 s (or longer) durations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..lineage import Lineage, Record
from .base import Events, MultiRateRecording, NativeRecording
from .edf import normalise_unit, rate_groups, read_annotations, read_header, read_signals

SOURCE_ID = "sleep-edfx"
SUBSET = "sleep-cassette"

PSG_RE = re.compile(r"^SC4(?P<ss>\d\d)(?P<night>\d)(?P<ver>[A-Z0-9]+)-PSG\.edf$")

CHANNEL_TYPE = {
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
    "EOG horizontal": "eog",
    "EMG submental": "emg",
    "Resp oro-nasal": "resp",
    "Temp rectal": "temp",
    "Event marker": "misc",
}

#: Bipolar derivations: the "reference" is the second electrode of the pair.
CHANNEL_REFERENCE = {
    "EEG Fpz-Cz": "Cz (bipolar derivation Fpz-Cz)",
    "EEG Pz-Oz": "Oz (bipolar derivation Pz-Oz)",
    "EOG horizontal": "bipolar outer canthi",
    "EMG submental": "bipolar submental",
}

UNIT_TO_SI = {"uV": (1e-6, "V"), "mV": (1e-3, "V"), "V": (1.0, "V"), "degC": (1.0, "degC")}


def root(data_root: str | Path | None = None) -> Path:
    from ..registry import get

    base = Path(data_root) if data_root is not None else get(SOURCE_ID).local_path
    return base / SUBSET


def iter_psg_paths(data_root: str | Path | None = None) -> Iterator[Path]:
    base = root(data_root)
    if not base.exists():
        return
    for p in sorted(base.glob("SC4*-PSG.edf")):
        if PSG_RE.match(p.name):
            yield p


def hypnogram_for(psg: Path) -> Path | None:
    m = PSG_RE.match(psg.name)
    if not m:
        return None
    stem = f"SC4{m['ss']}{m['night']}"
    matches = sorted(psg.parent.glob(f"{stem}*-Hypnogram.edf"))
    return matches[0] if matches else None


def load_recording(
    subject: str | int | None = None,
    night: int | None = None,
    *,
    data_root: str | Path | None = None,
    path: str | Path | None = None,
    with_hypnogram: bool = True,
) -> MultiRateRecording:
    """Load one night as a multirate recording (no channel is resampled)."""
    if path is not None:
        p = Path(path)
    else:
        ss = subject if isinstance(subject, str) else f"{int(subject):02d}"
        ss = ss.replace("SC4", "")[:2]
        cands = sorted(root(data_root).glob(f"SC4{ss}{night}*-PSG.edf"))
        if not cands:
            raise FileNotFoundError(f"no sleep-cassette PSG for subject {ss} night {night}")
        p = cands[0]
    m = PSG_RE.match(p.name)
    if m is None:
        raise ValueError(f"not a sleep-cassette PSG filename: {p.name}")
    participant = f"SC4{m['ss']}"
    lineage = Lineage(
        participant=participant,
        family=f"singleton:{participant}",
        site="sleep-edfx:home_recording",
        device="TelemetryCR_cassette",
        session=f"night{m['night']}",
        run="whole_night",
    )

    header, signals, inline = read_signals(p)
    groups: dict[float, NativeRecording] = {}
    for rate, idx in rate_groups(header).items():
        names = tuple(header.signals[i].label for i in idx)
        raw_units = tuple(normalise_unit(header.signals[i].physical_dim) for i in idx)
        data = np.stack([signals[i] for i in idx], axis=0)
        units: list[str] = []
        for j, u in enumerate(raw_units):
            scale, si = UNIT_TO_SI.get(u, (1.0, u))
            if scale != 1.0:
                data[j] = data[j] * scale
            units.append(si)
        groups[float(rate)] = NativeRecording(
            source_id=f"{SOURCE_ID}/{p.stem}@{rate:g}Hz",
            data=data,
            units=tuple(units),
            sfreq=float(rate),
            channel_names=names,
            channel_types=tuple(CHANNEL_TYPE.get(n, "misc") for n in names),
            frame_id="unknown",  # no electrode digitisation exists for this corpus
            clock_id=f"{SOURCE_ID}.recorder",
            lineage=lineage,
            channel_positions=None,
            montage="bipolar Fpz-Cz / Pz-Oz (no digitised positions)",
            events=None,
            meta={
                "path": str(p),
                "edf_physical_dim": dict(zip(names, raw_units)),
                "reference": {n: CHANNEL_REFERENCE.get(n, "unknown") for n in names},
                "quantisation": {
                    n: float(header.signals[i].gain)
                    * UNIT_TO_SI.get(normalise_unit(header.signals[i].physical_dim), (1.0, ""))[0]
                    for n, i in zip(names, idx)
                },
                "record_start_utc": str(header.start),
                "record_duration_s": header.record_duration,
                "prefilter": {n: header.signals[i].prefilter for n, i in zip(names, idx)},
            },
        )

    events = None
    hyp = hypnogram_for(p) if with_hypnogram else None
    if hyp is not None:
        ann = read_annotations(hyp)
        if ann:
            events = Events(
                onset=np.asarray([a[0] for a in ann], dtype=np.float64),
                duration=np.asarray([a[1] for a in ann], dtype=np.float64),
                label=np.asarray([a[2] for a in ann], dtype=object),
                clock_id=f"{SOURCE_ID}.recorder",
                sample=None,
                meta={"scorer_file": hyp.name, "epoch_length_s": 30.0},
            )
    elif inline:
        events = Events(
            onset=np.asarray([a[0] for a in inline], dtype=np.float64),
            duration=np.asarray([a[1] for a in inline], dtype=np.float64),
            label=np.asarray([a[2] for a in inline], dtype=object),
            clock_id=f"{SOURCE_ID}.recorder",
        )

    for rec in groups.values():
        if events is not None:
            rec.events = Events(
                onset=events.onset,
                duration=events.duration,
                label=events.label,
                clock_id=events.clock_id,
                sample=np.round(events.onset * rec.sfreq).astype(np.int64),
                meta=dict(events.meta),
            )

    return MultiRateRecording(
        source_id=f"{SOURCE_ID}/{p.stem}",
        groups=groups,
        lineage=lineage,
        clock_id=f"{SOURCE_ID}.recorder",
        events=events,
        meta={
            "psg_path": str(p),
            "hypnogram_path": str(hyp) if hyp else None,
            "subject": participant,
            "night": int(m["night"]),
            "native_rates_hz": sorted(groups),
        },
    )


def records(
    data_root: str | Path | None = None, *, subjects: Sequence[str] | None = None
) -> list[Record]:
    out: list[Record] = []
    for p in iter_psg_paths(data_root):
        m = PSG_RE.match(p.name)
        assert m is not None
        participant = f"SC4{m['ss']}"
        if subjects is not None and participant not in subjects:
            continue
        out.append(
            Record(
                id=p.stem,
                source_id=SOURCE_ID,
                lineage=Lineage(
                    participant=participant,
                    family=f"singleton:{participant}",
                    site="sleep-edfx:home_recording",
                    device="TelemetryCR_cassette",
                    session=f"night{m['night']}",
                    run="whole_night",
                ),
                stimulus_ids=(),  # no stimuli: this is spontaneous sleep
                path=str(p),
                n_bytes=p.stat().st_size,
                meta={"night": int(m["night"]), "hypnogram": str(hypnogram_for(p))},
            )
        )
    return out


def stage_durations(rec: MultiRateRecording) -> dict[str, float]:
    """Total seconds per scored stage, for sanity checks and reports."""
    if rec.events is None:
        return {}
    out: dict[str, float] = {}
    for lab, dur in zip(rec.events.label, rec.events.duration):
        out[str(lab)] = out.get(str(lab), 0.0) + float(dur)
    return dict(sorted(out.items()))


__all__ = (
    "SOURCE_ID",
    "load_recording",
    "records",
    "iter_psg_paths",
    "hypnogram_for",
    "stage_durations",
)
