"""Native-support containers returned by every loader.

Thesis §2.6: heterogeneous sources are *not* resampled onto a common raster.
A loader therefore returns the signal exactly as recorded, with

* ``data`` in the recording's own units,
* ``sfreq`` the native sampling rate of that channel group,
* ``channel_names`` / ``channel_types`` / ``channel_positions`` as stored,
* ``frame_id`` naming the coordinate frame the positions live in,
* ``clock_id`` naming the clock the samples are timed by,
* ``events`` on their own event clock.

A recording whose channels have *different* native rates (Sleep-EDF: 100 Hz
EEG next to 1 Hz respiration) is returned as several
:class:`NativeRecording` objects, one per rate, never merged.

:meth:`NativeRecording.resample` exists only to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..lineage import Lineage


class NativeSupportError(RuntimeError):
    """Raised when an operation would destroy native support."""


@dataclass
class Events:
    """Events on their own clock.

    ``onset`` is in seconds **on** ``clock_id`` (not necessarily the sample
    clock); ``sample`` is the index into the sample clock when the two are the
    same device, else ``None``.
    """

    onset: np.ndarray
    duration: np.ndarray
    label: np.ndarray
    clock_id: str
    sample: np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.onset))

    @property
    def unique_labels(self) -> tuple[str, ...]:
        return tuple(sorted({str(x) for x in self.label}))


@dataclass
class NativeRecording:
    """One continuous recording at one native sampling rate."""

    source_id: str
    data: np.ndarray  # (n_channels, n_times)
    units: tuple[str, ...]  # per channel, e.g. ("V", "V", ...)
    sfreq: float  # Hz, native
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    frame_id: str  # e.g. "captrak_head_RAS", "unknown"
    clock_id: str  # e.g. "eeg_amp"
    lineage: Lineage
    channel_positions: Mapping[str, tuple[float, float, float]] | None = None
    montage: str = "unknown"
    events: Events | None = None
    t0: float = 0.0  # start time on clock_id, seconds
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must be (n_channels, n_times)")
        n = self.data.shape[0]
        for name, seq in (
            ("units", self.units),
            ("channel_names", self.channel_names),
            ("channel_types", self.channel_types),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} has {len(seq)} entries for {n} channels")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")

    # -- geometry --------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_times / self.sfreq

    @property
    def dt(self) -> float:
        return 1.0 / self.sfreq

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.n_times, dtype=np.float64) / self.sfreq

    def pick(self, names: Sequence[str]) -> "NativeRecording":
        idx = [self.channel_names.index(n) for n in names]
        return NativeRecording(
            source_id=self.source_id,
            data=self.data[idx],
            units=tuple(self.units[i] for i in idx),
            sfreq=self.sfreq,
            channel_names=tuple(self.channel_names[i] for i in idx),
            channel_types=tuple(self.channel_types[i] for i in idx),
            frame_id=self.frame_id,
            clock_id=self.clock_id,
            lineage=self.lineage,
            channel_positions=self.channel_positions,
            montage=self.montage,
            events=self.events,
            t0=self.t0,
            meta=dict(self.meta),
        )

    # -- epoching (no resampling, integer sample offsets) ----------------
    def epochs(
        self,
        tmin: float,
        tmax: float,
        *,
        labels: Sequence[str] | None = None,
    ) -> "NativeEpochs":
        if self.events is None:
            raise NativeSupportError(
                f"{self.source_id}: no events attached; cannot epoch without an event clock"
            )
        if self.events.clock_id != self.clock_id and self.events.sample is None:
            raise NativeSupportError(
                f"{self.source_id}: events are on clock {self.events.clock_id!r} but samples on "
                f"{self.clock_id!r} and no sample index is stored; the clock relation is unknown "
                "(R01) - resolve it in the frame/clock graph before epoching"
            )
        keep = np.arange(len(self.events))
        if labels is not None:
            wanted = set(labels)
            keep = np.array([i for i in keep if str(self.events.label[i]) in wanted], dtype=int)
        if self.events.sample is not None:
            onsets = np.asarray(self.events.sample, dtype=np.int64)[keep]
        else:
            onsets = np.round(
                (np.asarray(self.events.onset, dtype=np.float64)[keep] - self.t0) * self.sfreq
            ).astype(np.int64)
        a = int(round(tmin * self.sfreq))
        b = int(round(tmax * self.sfreq))
        n = b - a
        good = [i for i, o in enumerate(onsets) if 0 <= o + a and o + b <= self.n_times]
        arr = np.empty((len(good), self.n_channels, n), dtype=self.data.dtype)
        for j, i in enumerate(good):
            o = int(onsets[i])
            arr[j] = self.data[:, o + a : o + b]
        return NativeEpochs(
            source_id=self.source_id,
            data=arr,
            units=self.units,
            sfreq=self.sfreq,
            channel_names=self.channel_names,
            channel_types=self.channel_types,
            frame_id=self.frame_id,
            clock_id=self.clock_id,
            lineage=self.lineage,
            tmin=a / self.sfreq,
            labels=tuple(str(self.events.label[keep[i]]) for i in good),
            onsets=tuple(float(self.events.onset[keep[i]]) for i in good),
            channel_positions=self.channel_positions,
            montage=self.montage,
            meta=dict(self.meta),
        )

    # -- refusals --------------------------------------------------------
    def resample(self, *args: Any, **kwargs: Any):
        raise NativeSupportError(
            f"{self.source_id}: refusing to resample. Native rate {self.sfreq} Hz is part of the "
            "source's declared temporal support (thesis §2.6); build a clock-graph edge with a "
            "declared group delay and jitter instead of rewriting the samples."
        )

    def to_common_raster(self, *args: Any, **kwargs: Any):
        raise NativeSupportError(
            f"{self.source_id}: refusing to project onto a common raster; see resample()."
        )

    def describe(self) -> str:
        return (
            f"{self.source_id}: {self.n_channels} ch x {self.n_times} samp @ {self.sfreq} Hz "
            f"({self.duration:.1f} s), units={sorted(set(self.units))}, "
            f"frame={self.frame_id}, clock={self.clock_id}, montage={self.montage}, "
            f"events={0 if self.events is None else len(self.events)}"
        )


@dataclass
class NativeEpochs:
    """Epoched data, still at the native rate."""

    source_id: str
    data: np.ndarray  # (n_epochs, n_channels, n_times)
    units: tuple[str, ...]
    sfreq: float
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    frame_id: str
    clock_id: str
    lineage: Lineage
    tmin: float
    labels: tuple[str, ...] = ()
    onsets: tuple[float, ...] = ()
    channel_positions: Mapping[str, tuple[float, float, float]] | None = None
    montage: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])

    def times(self) -> np.ndarray:
        return self.tmin + np.arange(self.n_times, dtype=np.float64) / self.sfreq

    def resample(self, *a: Any, **k: Any):
        raise NativeSupportError(f"{self.source_id}: refusing to resample epochs; see §2.6.")

    def describe(self) -> str:
        return (
            f"{self.source_id}: {self.n_epochs} epochs x {self.n_channels} ch x "
            f"{self.n_times} samp @ {self.sfreq} Hz, labels={sorted(set(self.labels))[:6]}"
        )


@dataclass
class MultiRateRecording:
    """Several :class:`NativeRecording` groups from one physical recording.

    A polysomnogram carries 100 Hz EEG/EOG beside 1 Hz respiration and
    temperature.  Merging them onto one raster would invent 99 samples per
    second of "measurement" that was never made, so they are kept apart and
    related only through ``clock_id`` (they share the recorder's clock).
    """

    source_id: str
    groups: Mapping[float, NativeRecording]
    lineage: Lineage
    clock_id: str
    events: Events | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rates(self) -> tuple[float, ...]:
        return tuple(sorted(self.groups))

    @property
    def primary(self) -> NativeRecording:
        """The fastest group (usually the EEG)."""
        return self.groups[max(self.groups)]

    def __getitem__(self, rate: float) -> NativeRecording:
        return self.groups[rate]

    def channel(self, name: str) -> NativeRecording:
        for rec in self.groups.values():
            if name in rec.channel_names:
                return rec.pick([name])
        raise KeyError(f"{name!r} not in {self.source_id}")

    def resample(self, *a: Any, **k: Any):
        raise NativeSupportError(
            f"{self.source_id}: refusing to merge rates {self.rates} onto one raster; "
            "the slow channels were not measured at the fast rate (thesis §2.6)."
        )

    def describe(self) -> str:
        head = f"{self.source_id}: {len(self.groups)} native rate group(s) {self.rates} Hz"
        body = "\n".join("  " + r.describe() for _, r in sorted(self.groups.items()))
        return head + "\n" + body


@dataclass
class VolumeSeries:
    """A 4-D image series (BOLD) kept in its native voxel grid and TR."""

    source_id: str
    data: np.ndarray  # (x, y, z, t) or (x, y, z)
    units: str
    tr: float | None  # seconds; None for anatomy
    affine: np.ndarray  # voxel -> frame_id, 4x4
    frame_id: str
    clock_id: str
    lineage: Lineage
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        m = np.asarray(self.affine)[:3, :3]
        return tuple(float(np.linalg.norm(m[:, i])) for i in range(3))  # type: ignore[return-value]

    def resample(self, *a: Any, **k: Any):
        raise NativeSupportError(f"{self.source_id}: refusing to resample the voxel grid.")

    def describe(self) -> str:
        return (
            f"{self.source_id}: {self.data.shape} voxels, {self.voxel_size} mm, "
            f"TR={self.tr}, units={self.units}, frame={self.frame_id}"
        )


__all__ = (
    "Events",
    "MultiRateRecording",
    "NativeEpochs",
    "NativeRecording",
    "NativeSupportError",
    "VolumeSeries",
)
