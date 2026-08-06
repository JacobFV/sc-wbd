"""A minimal, native-rate-preserving EDF/EDF+ reader.

Why not just call ``mne.io.read_raw_edf``?  Because MNE returns a single
``sfreq`` for the whole file and upsamples slower channels onto the fastest
channel's raster.  For Sleep-EDF that silently turns a 1 Hz rectal-temperature
channel into a 100 Hz signal - exactly the "forced common raster" the thesis
refuses (§2.6).  This reader keeps every signal at the rate written in the
header and hands back one group per distinct rate.

The reader implements EDF and EDF+ (including the ``EDF Annotations`` TAL
channel).  It is deliberately small and dependency-free (numpy only): its job
is to be *honest about support*, not to be a full I/O stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ANNOT_LABEL = "EDF Annotations"


@dataclass
class EdfSignalHeader:
    label: str
    transducer: str
    physical_dim: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    prefilter: str
    n_samples_per_record: int

    @property
    def gain(self) -> float:
        dspan = self.digital_max - self.digital_min
        if dspan == 0:
            return 1.0
        return (self.physical_max - self.physical_min) / dspan

    @property
    def offset(self) -> float:
        return self.physical_min - self.digital_min * self.gain

    @property
    def is_annotation(self) -> bool:
        return self.label.strip() == ANNOT_LABEL


@dataclass
class EdfHeader:
    version: str
    patient_id: str
    recording_id: str
    start: datetime | None
    n_header_bytes: int
    reserved: str
    n_records: int
    record_duration: float
    signals: list[EdfSignalHeader]

    @property
    def is_edfplus(self) -> bool:
        return self.reserved.strip().startswith("EDF+")

    def rate(self, i: int) -> float:
        return self.signals[i].n_samples_per_record / self.record_duration


def _s(b: bytes) -> str:
    return b.decode("latin-1").strip()


def read_header(path: str | Path) -> EdfHeader:
    with open(path, "rb") as fh:
        raw = fh.read(256)
        version = _s(raw[0:8])
        patient = _s(raw[8:88])
        recording = _s(raw[88:168])
        startdate = _s(raw[168:176])
        starttime = _s(raw[176:184])
        n_header = int(_s(raw[184:192]))
        reserved = _s(raw[192:236])
        n_records = int(_s(raw[236:244]))
        rec_dur = float(_s(raw[244:252]))
        ns = int(_s(raw[252:256]))

        def block(width: int) -> list[str]:
            return [_s(fh.read(width)) for _ in range(ns)]

        labels = block(16)
        transducers = block(80)
        dims = block(8)
        pmin = [float(x) for x in block(8)]
        pmax = [float(x) for x in block(8)]
        dmin = [float(x) for x in block(8)]
        dmax = [float(x) for x in block(8)]
        prefilters = block(80)
        nsamp = [int(x) for x in block(8)]

    start: datetime | None = None
    try:
        dd, mm, yy = (int(x) for x in startdate.split("."))
        hh, mi, ss = (int(x) for x in starttime.split("."))
        year = 2000 + yy if yy < 85 else 1900 + yy
        start = datetime(year, mm, dd, hh, mi, ss, tzinfo=timezone.utc)
    except Exception:
        start = None

    sigs = [
        EdfSignalHeader(
            label=labels[i],
            transducer=transducers[i],
            physical_dim=dims[i],
            physical_min=pmin[i],
            physical_max=pmax[i],
            digital_min=dmin[i],
            digital_max=dmax[i],
            prefilter=prefilters[i],
            n_samples_per_record=nsamp[i],
        )
        for i in range(ns)
    ]
    return EdfHeader(
        version=version,
        patient_id=patient,
        recording_id=recording,
        start=start,
        n_header_bytes=n_header,
        reserved=reserved,
        n_records=n_records,
        record_duration=rec_dur,
        signals=sigs,
    )


def read_signals(
    path: str | Path, header: EdfHeader | None = None
) -> tuple[EdfHeader, dict[int, np.ndarray], list[tuple[float, float, str]]]:
    """Read an EDF file.

    Returns
    -------
    header
        The parsed header.
    signals
        ``{signal_index: float64 array}`` in the signal's own physical units
        and at its own native rate.  Annotation channels are excluded.
    annotations
        ``[(onset_s, duration_s, label), ...]`` decoded from EDF+ TALs; empty
        for plain EDF.
    """
    header = header or read_header(path)
    ns = len(header.signals)
    per_rec = [s.n_samples_per_record for s in header.signals]
    rec_len = int(sum(per_rec))
    data = np.fromfile(path, dtype="<i2", offset=header.n_header_bytes)
    n_records = header.n_records
    if n_records < 0 or n_records * rec_len > data.size:
        n_records = data.size // rec_len
    data = data[: n_records * rec_len].reshape(n_records, rec_len)

    starts = np.cumsum([0] + per_rec)
    out: dict[int, np.ndarray] = {}
    annots: list[tuple[float, float, str]] = []
    for i, sig in enumerate(header.signals):
        chunk = data[:, starts[i] : starts[i + 1]]
        if sig.is_annotation:
            annots.extend(_parse_tals(chunk))
            continue
        out[i] = chunk.reshape(-1).astype(np.float64) * sig.gain + sig.offset
    return header, out, annots


def _parse_tals(chunk: np.ndarray) -> list[tuple[float, float, str]]:
    """Decode EDF+ Time-stamped Annotation Lists."""
    raw = chunk.astype("<i2").tobytes().decode("latin-1")
    out: list[tuple[float, float, str]] = []
    for tal in raw.split("\x00"):
        if not tal.strip("\x14\x15 "):
            continue
        parts = tal.split("\x14")
        head = parts[0]
        if not head or head[0] not in "+-":
            continue
        if "\x15" in head:
            onset_s, dur_s = head.split("\x15", 1)
        else:
            onset_s, dur_s = head, "0"
        try:
            onset = float(onset_s)
            duration = float(dur_s) if dur_s.strip() else 0.0
        except ValueError:
            continue
        labels = [p for p in parts[1:] if p]
        for lab in labels:
            out.append((onset, duration, lab))
        if not labels:
            # a time-keeping TAL (first TAL of each record) - not an event
            continue
    return out


def read_annotations(path: str | Path) -> list[tuple[float, float, str]]:
    """Read an annotations-only EDF+ file (e.g. a Sleep-EDF hypnogram)."""
    _, _, ann = read_signals(path)
    return ann


def rate_groups(header: EdfHeader) -> dict[float, list[int]]:
    """``{native_rate_hz: [signal indices]}`` for non-annotation signals."""
    groups: dict[float, list[int]] = {}
    for i, sig in enumerate(header.signals):
        if sig.is_annotation:
            continue
        groups.setdefault(header.rate(i), []).append(i)
    return dict(sorted(groups.items()))


UNIT_ALIASES = {
    "uv": "uV",
    "µv": "uV",
    "\xb5v": "uV",
    "mv": "mV",
    "v": "V",
    "degc": "degC",
    "": "dimensionless",
    "-": "dimensionless",
}


def normalise_unit(dim: str) -> str:
    return UNIT_ALIASES.get(dim.strip().lower(), dim.strip())


__all__ = (
    "EdfHeader",
    "EdfSignalHeader",
    "normalise_unit",
    "rate_groups",
    "read_annotations",
    "read_header",
    "read_signals",
)
