"""Measured human EEG windows for SC-WBD-001-beta, with participant-level splits.

Everything in this module reads **real recordings of real people**, downloaded
from public repositories and used under their published licences:

* **EEG Motor Movement/Imagery Database** (``eegmmidb``, PhysioNet, BCI2000):
  109 volunteers, 64 EEG channels of the extended 10-10 Sharbrough montage,
  160 Hz, 14 runs each.  Open Data Commons Attribution Licence; cite Schalk et
  al. 2004 and Goldberger et al. 2000.
* **Sleep-EDF Database Expanded**, sleep-cassette subset (PhysioNet): whole-night
  home polysomnograms, two bipolar EEG derivations (Fpz-Cz, Pz-Oz) at 100 Hz.
  Open Data Commons Attribution Licence; cite Kemp et al. 2000.

Why this module exists, stated plainly:

1. These are the **only** sources in the project that can support a claim about
   real brains.  Simulated corpora (:mod:`scwbd.foundation.simulate`) exercise
   physics, missingness and rare regimes; they can never be evidence that the
   model has learned anything about biology.  A claim of the form "SC-WBD
   predicts/reconstructs measured neural activity" must be evaluated here.
2. A **window-level score is not participant-level generalisation.**  Windows
   from one recording are massively autocorrelated and share an individual's
   anatomy, montage placement, impedance profile and mains environment.  A high
   score on windows drawn from a person the model was trained on measures
   memorisation of that person, not transfer.  Only the held-out-participant
   fold produced by :func:`participant_split` supports a generalisation claim,
   and even then the unit of statistical analysis is the *participant*, not the
   window: aggregate per participant before computing an interval.
3. The grouping unit is therefore the participant.  Every run and every night of
   a subject lands in exactly one fold (thesis refusal ``R10``).
   :func:`leakage_check` re-derives that property from the realised split and
   returns a machine-readable report; it is meant to be run and recorded, not
   trusted.

Preprocessing is deliberately conservative and fully recorded.  Each recording
is band-passed, optionally notched, resampled to a common rate, referenced, and
then divided by a **single robust scale factor** (the median across channels of
the per-channel median absolute deviation).  That factor is stored in volts in
the cache index, so every stored window can be mapped back to physical units:
``volts = stored_value * scale_volts``.  Nothing here is normalised per window,
because per-window normalisation destroys the amplitude information that
distinguishes, say, slow-wave sleep from wake.

The cache holds preprocessed windows as ``.npy`` shards (one per recording) that
are read back as memory maps, plus a JSON index.  The first epoch pays for
filtering and resampling; later epochs and later processes do not.  Because the
cache is per recording, a **partially downloaded dataset is fine**: recordings
that appear later are preprocessed on the next construction, and recordings that
cannot be opened are skipped and reported rather than crashing the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .heads import EEGMMIDB_CHANNELS

__all__ = [
    "RealEEGConfig",
    "WindowProvenance",
    "RealEEGDataset",
    "EEGMMIDBDataset",
    "SleepEDFDataset",
    "SLEEP_EDF_EEG_CHANNELS",
    "participant_split",
    "leakage_check",
    "make_loaders",
]

LOGGER = logging.getLogger(__name__)

#: The two scalp EEG derivations of the sleep-cassette montage, in file order.
#: They are **bipolar** (Fpz referenced to Cz, Pz referenced to Oz), not
#: monopolar scalp potentials, which is why they are never average-referenced
#: and never padded out to a 64-channel montage.
SLEEP_EDF_EEG_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")

#: Cache format version.  Bump when the on-disk layout or the preprocessing
#: semantics change, so stale shards are never silently reused.
CACHE_VERSION = 2

_SLEEP_PSG_RE = re.compile(r"^(?P<subject>SC4\d\d)(?P<night>\d)(?P<ver>[A-Z0-9]+)-PSG\.edf$")
_EEGMMIDB_RUN_RE = re.compile(r"^(?P<subject>S\d{3})(?P<run>R\d{2})\.edf$")


# ======================================================================
# configuration
# ======================================================================
@dataclass
class RealEEGConfig:
    """Everything that determines which measured windows exist and what is in them.

    The fields that change the *content* of a window (rates, filters, window
    length, artifact threshold) are hashed into the cache key; the fields that
    only change *how many* recordings are visited (``max_subjects``,
    ``max_runs_per_subject``) are not, so shrinking a run does not throw away a
    cache built by a larger one.
    """

    #: Root of the PhysioNet eegmmidb release (contains ``S001/`` ... ``S109/``).
    eegmmidb_root: Path = Path("/data/scwbd/eegmmidb/1.0.0")
    #: Root of the Sleep-EDF Expanded sleep-cassette subset (contains ``*-PSG.edf``).
    sleep_edfx_root: Path = Path("/data/scwbd/sleep-edfx/1.0.0/sleep-cassette")

    #: Common sampling rate every source is brought to.  125 Hz keeps the whole
    #: 0.5-45 Hz passband with margin while making sources commensurable.
    fs_target: float = 125.0
    window_s: float = 4.0
    #: Hop between window starts.  ``None`` means "no overlap" (hop == window),
    #: which is the honest default: overlapping windows are not independent
    #: samples and inflate any window-level score.
    window_stride_s: float | None = None

    l_freq: float | None = 0.5
    h_freq: float | None = 45.0
    #: Mains frequency to notch.  Skipped, with the reason recorded, when it sits
    #: above the recording's Nyquist or is already inside the low-pass stopband.
    notch: float | None = 60.0

    max_subjects: int | None = None
    max_runs_per_subject: int | None = None
    #: Cap on windows kept per recording.  Whole-night polysomnograms yield tens
    #: of thousands of windows and would otherwise drown the motor-imagery runs;
    #: the kept windows are spread evenly over the recording rather than taken
    #: from the start, because the first hour of a night is all wake.
    max_windows_per_recording: int | None = 512

    cache_dir: Path = Path("/data/scwbd/foundation_cache")
    seed: int = 0

    #: A window is dropped when any sample of any channel exceeds this many
    #: per-channel robust standard deviations (``|x - median| > z_max * 1.4826 *
    #: MAD``).  Twelve sigma is far outside physiology and catches blinks,
    #: electrode pops and movement; it typically rejects a third of the raw
    #: eegmmidb windows, which is normal for data with no ICA/EOG cleaning.
    z_max: float = 12.0

    def __post_init__(self) -> None:
        self.eegmmidb_root = Path(self.eegmmidb_root)
        self.sleep_edfx_root = Path(self.sleep_edfx_root)
        self.cache_dir = Path(self.cache_dir)
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.fs_target <= 0:
            raise ValueError("fs_target must be positive")

    # -- derived quantities ------------------------------------------------
    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.fs_target))

    @property
    def stride_samples(self) -> int:
        stride_s = self.window_s if self.window_stride_s is None else self.window_stride_s
        return max(1, int(round(stride_s * self.fs_target)))

    def as_json(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("eegmmidb_root", "sleep_edfx_root", "cache_dir"):
            d[key] = str(d[key])
        return d

    def cache_key(self, source: str, channels: Sequence[str], average_reference: bool) -> str:
        """Hash of every setting that changes the *content* of a cached window."""
        payload = {
            "cache_version": CACHE_VERSION,
            "source": source,
            "channels": list(channels),
            "average_reference": bool(average_reference),
            "fs_target": float(self.fs_target),
            "window_s": float(self.window_s),
            "stride_samples": int(self.stride_samples),
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch": self.notch,
            "z_max": float(self.z_max),
            "max_windows_per_recording": self.max_windows_per_recording,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


# ======================================================================
# provenance
# ======================================================================
@dataclass(frozen=True)
class WindowProvenance:
    """Where one window came from.  Sufficient to re-extract it from the EDF."""

    source: str
    subject: str
    session: str
    run: str
    recording_id: str
    path: str
    #: Start sample of the window *in the resampled recording* (fs = fs_target).
    sample_offset: int
    #: Multiply a stored sample by this to recover volts.
    scale_volts: float


# ======================================================================
# preprocessing helpers
# ======================================================================
def _normalise_channel_name(name: str) -> str:
    """Canonical form for matching EDF channel labels.

    The eegmmidb EDF labels are space/dot padded to a fixed width (``"C5.."``,
    ``"Fc5."``), and their capitalisation differs between the PhysioNet files
    and the montage tables.  Matching is therefore case-insensitive on the label
    with trailing dots and whitespace removed.
    """
    return name.strip().rstrip(".").strip().lower()


def _select_channels(
    file_channels: Sequence[str], wanted: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Map ``wanted`` (canonical order) onto the file's labels.

    Returns ``(file_labels_in_canonical_order, missing)``.  A fixed canonical
    order matters more than it looks: the model's channel axis is positional, so
    two recordings whose EDF headers happen to be ordered differently must still
    line up electrode for electrode.
    """
    lookup: dict[str, str] = {}
    for label in file_channels:
        lookup.setdefault(_normalise_channel_name(label), label)
    picked: list[str] = []
    missing: list[str] = []
    for want in wanted:
        label = lookup.get(_normalise_channel_name(want))
        if label is None:
            missing.append(want)
        else:
            picked.append(label)
    return picked, missing


def _robust_stats(data: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(scale, median_per_channel, mad_per_channel)`` for one recording.

    ``scale`` is the median across channels of the per-channel median absolute
    deviation, and it is the **only** number the stored windows are divided by:
    one scalar per recording, not per channel and not per window.  A per-channel
    scale would erase the topography (which is signal) and a per-window scale
    would erase amplitude differences between states (also signal).  The MAD is
    used rather than the standard deviation because a single electrode pop moves
    the SD by orders of magnitude.

    The per-channel median and MAD are returned as well because artifact
    detection needs a *different* statistic from amplitude normalisation: see
    :meth:`RealEEGDataset._preprocess_recording`.
    """
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    finite = mad[np.isfinite(mad) & (mad > 0)]
    scale = float(np.median(finite)) if finite.size else 0.0
    return scale, med, mad


def _window_starts(n_samples: int, win: int, hop: int, cap: int | None) -> np.ndarray:
    """Window start samples, thinned evenly (not truncated) when capped."""
    if n_samples < win:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n_samples - win + 1, hop, dtype=np.int64)
    if cap is not None and starts.size > cap > 0:
        idx = np.linspace(0, starts.size - 1, cap).round().astype(np.int64)
        starts = starts[np.unique(idx)]
    return starts


# ======================================================================
# base dataset
# ======================================================================
class RealEEGDataset(Dataset):
    """Windows of measured EEG, cached as memory-mapped ``.npy`` shards.

    Subclasses supply the discovery logic (:meth:`_discover`) and the montage;
    everything else -- preprocessing, caching, indexing, provenance -- is shared
    so that a split written against one source works unchanged against the other.

    A note on what an item is: one item is ``window_s`` seconds of one recording
    of one person.  It is *not* an independent sample. See the module docstring.
    """

    source: str = "unknown"
    average_reference: bool = True

    def __init__(self, cfg: RealEEGConfig, *, build: bool = True) -> None:
        self.cfg = cfg
        self.channel_names: tuple[str, ...] = tuple(self._montage())
        self._key = cfg.cache_key(self.source, self.channel_names, self.average_reference)
        self.cache_root = Path(cfg.cache_dir) / self.source / self._key
        self.recordings: list[dict[str, Any]] = []
        self.window_index: list[tuple[int, int]] = []
        self.n_dropped_windows: int = 0
        self.skipped_recordings: list[dict[str, str]] = []
        self.n_cache_hits: int = 0
        self.n_cache_misses: int = 0
        self._memmaps: dict[int, np.ndarray] = {}
        self._memmap_pid: int = os.getpid()
        if build:
            self.build()

    # -- to be provided by subclasses -----------------------------------
    def _montage(self) -> Sequence[str]:
        raise NotImplementedError

    def _discover(self) -> list[dict[str, str]]:
        """Return one dict per candidate recording with keys
        ``recording_id, subject, session, run, path``, already limited by
        ``max_subjects`` / ``max_runs_per_subject`` and sorted deterministically.
        """
        raise NotImplementedError

    # -- cache construction ---------------------------------------------
    def build(self) -> None:
        """Ensure every discovered recording has a shard, then index the windows."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        candidates = self._discover()
        recordings: list[dict[str, Any]] = []
        for cand in candidates:
            meta_path = self.cache_root / f"{cand['recording_id']}.json"
            meta = self._load_meta(meta_path)
            if meta is None:
                self.n_cache_misses += 1
                meta = self._preprocess_recording(cand)
                if meta is not None:
                    tmp = meta_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    tmp.replace(meta_path)
            else:
                self.n_cache_hits += 1
            if meta is None:
                continue
            if meta.get("status") != "ok":
                self.skipped_recordings.append(
                    {"recording_id": meta["recording_id"], "reason": meta.get("reason", "unknown")}
                )
                continue
            if not (self.cache_root / meta["shard"]).exists():
                # The sidecar survived but the shard did not (interrupted write,
                # manual cleanup).  Rebuild rather than index phantom windows.
                meta = self._preprocess_recording(cand)
                if meta is None or meta.get("status") != "ok":
                    continue
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            recordings.append(meta)

        self.recordings = recordings
        self.window_index = [
            (r_idx, w_idx)
            for r_idx, rec in enumerate(recordings)
            for w_idx in range(int(rec["n_windows"]))
        ]
        self.n_dropped_windows = sum(int(r.get("n_dropped", 0)) for r in recordings)
        self._write_index()

    def _load_meta(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning("unreadable cache sidecar %s (%s); rebuilding", path, exc)
            return None

    def _write_index(self) -> None:
        """Aggregate index: the auditable record of what the dataset contains."""
        index = {
            "cache_version": CACHE_VERSION,
            "source": self.source,
            "cache_key": self._key,
            "channels": list(self.channel_names),
            "average_reference": self.average_reference,
            "config": self.cfg.as_json(),
            "n_windows": len(self.window_index),
            "n_recordings": len(self.recordings),
            "n_subjects": len(self.subjects),
            "n_dropped_windows": self.n_dropped_windows,
            "skipped_recordings": self.skipped_recordings,
            "recordings": self.recordings,
        }
        path = self.cache_root / "index.json"
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # a read-only cache must not kill a training run
            LOGGER.warning("could not write cache index %s (%s)", path, exc)

    def _preprocess_recording(self, cand: dict[str, str]) -> dict[str, Any] | None:
        """Filter, resample, reference, robust-scale and window one EDF file.

        Returns the sidecar metadata (``status == "ok"`` when windows were
        written).  Failures are returned as metadata with a reason rather than
        raised: a half-downloaded archive should degrade the dataset, not abort
        the job.
        """
        import mne  # imported lazily: mne is heavy and only needed on a cache miss

        rec_id = cand["recording_id"]
        base = {
            "recording_id": rec_id,
            "subject": cand["subject"],
            "session": cand["session"],
            "run": cand["run"],
            "path": cand["path"],
            "source": self.source,
        }
        try:
            raw = mne.io.read_raw_edf(cand["path"], preload=False, verbose="error")
        except Exception as exc:  # noqa: BLE001 - mne raises many unrelated types
            LOGGER.warning("cannot open %s (%s); skipping", cand["path"], exc)
            return {**base, "status": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}

        picked, missing = _select_channels(raw.ch_names, self.channel_names)
        if missing:
            return {
                **base,
                "status": "montage_mismatch",
                "reason": f"missing channels {missing[:6]}",
            }

        fs_raw = float(raw.info["sfreq"])
        notch_applied = False
        notch_reason = "not requested"
        try:
            raw.pick(picked)
            raw.load_data(verbose="error")
            if self.cfg.l_freq is not None or self.cfg.h_freq is not None:
                raw.filter(
                    self.cfg.l_freq, self.cfg.h_freq, picks="all", verbose="error", n_jobs=1
                )
            if self.cfg.notch is not None:
                nyq = fs_raw / 2.0
                if self.cfg.notch >= 0.9 * nyq:
                    notch_reason = f"notch {self.cfg.notch} Hz at/above Nyquist {nyq} Hz"
                elif self.cfg.h_freq is not None and self.cfg.notch >= self.cfg.h_freq:
                    notch_reason = (
                        f"notch {self.cfg.notch} Hz already inside the "
                        f"{self.cfg.h_freq} Hz low-pass stopband"
                    )
                else:
                    raw.notch_filter(
                        [self.cfg.notch], picks="all", verbose="error", n_jobs=1
                    )
                    notch_applied = True
                    notch_reason = "applied"
            if abs(fs_raw - self.cfg.fs_target) > 1e-9:
                raw.resample(self.cfg.fs_target, verbose="error")
            # Reorder explicitly rather than trusting the pick order: the channel
            # axis is positional in the model, so a silent reordering would be a
            # silent relabelling of electrodes.
            order = [raw.ch_names.index(label) for label in picked]
            data = np.asarray(raw.get_data(), dtype=np.float64)[order]
        except Exception as exc:  # noqa: BLE001 - see above
            LOGGER.warning("preprocessing failed for %s (%s); skipping", cand["path"], exc)
            return {**base, "status": "preprocess_failed", "reason": f"{type(exc).__name__}: {exc}"}

        if self.average_reference:
            data = data - data.mean(axis=0, keepdims=True)

        scale, med, mad = _robust_stats(data)
        if not np.isfinite(scale) or scale <= 0:
            return {**base, "status": "flat", "reason": "robust scale is zero or non-finite"}
        data = data / scale
        # Artifact detection uses a *per-channel* robust z-score,
        # ``z = (x - median) / (1.4826 * MAD)``, which is not the same statistic
        # as the recording-wide amplitude scale above.  Using the recording-wide
        # scale here would let one persistently noisy electrode fail every window
        # of an otherwise usable recording; using the per-channel MAD makes the
        # threshold "unusual *for this electrode*", which is what an artifact is.
        # The 1.4826 factor puts the score in Gaussian-sigma units so that
        # ``z_max`` means what a reader expects it to mean.
        z_med = med / scale
        z_den = 1.4826 * np.maximum(mad / scale, 1e-12)

        win = self.cfg.window_samples
        starts = _window_starts(
            data.shape[1], win, self.cfg.stride_samples, self.cfg.max_windows_per_recording
        )
        kept: list[int] = []
        buf: list[np.ndarray] = []
        n_nonfinite = 0
        n_artifact = 0
        for s0 in starts.tolist():
            seg = data[:, s0 : s0 + win]
            if not np.isfinite(seg).all():
                n_nonfinite += 1
                continue
            if float(np.abs((seg - z_med) / z_den).max()) > self.cfg.z_max:
                n_artifact += 1
                continue
            kept.append(int(s0))
            buf.append(seg.T.astype(np.float32))  # (T, C): time-major for the model

        meta = {
            **base,
            "status": "ok" if kept else "no_clean_windows",
            "shard": f"{rec_id}.npy",
            "fs": float(self.cfg.fs_target),
            "fs_native": fs_raw,
            "upsampled": bool(fs_raw < self.cfg.fs_target - 1e-9),
            "channels": list(self.channel_names),
            "file_channels": list(picked),
            "average_reference": self.average_reference,
            "notch_applied": notch_applied,
            "notch_reason": notch_reason,
            "scale_volts": float(scale),
            "channel_mad_volts": [float(v) for v in mad.ravel()],
            "n_windows": len(kept),
            "n_candidate_windows": int(starts.size),
            "n_dropped": int(n_nonfinite + n_artifact),
            "n_dropped_nonfinite": int(n_nonfinite),
            "n_dropped_artifact": int(n_artifact),
            "window_starts": kept,
        }
        if not kept:
            meta["reason"] = "every candidate window was non-finite or exceeded z_max"
            return meta

        shard_path = self.cache_root / meta["shard"]
        tmp_path = shard_path.with_suffix(".npy.tmp")
        arr = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(len(buf), win, len(self.channel_names))
        )
        for i, seg in enumerate(buf):
            arr[i] = seg
        arr.flush()
        del arr
        tmp_path.replace(shard_path)
        return meta

    # -- memmap access ---------------------------------------------------
    def _shard(self, rec_idx: int) -> np.ndarray:
        # DataLoader workers are forked; a memmap opened in the parent must not
        # be shared across processes, so re-open after a fork.
        if os.getpid() != self._memmap_pid:
            self._memmaps = {}
            self._memmap_pid = os.getpid()
        mm = self._memmaps.get(rec_idx)
        if mm is None:
            path = self.cache_root / self.recordings[rec_idx]["shard"]
            mm = np.load(path, mmap_mode="r")
            self._memmaps[rec_idx] = mm
        return mm

    # -- Dataset protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec_idx, w_idx = self.window_index[int(idx)]
        rec = self.recordings[rec_idx]
        # Copy out of the memory map: the tensor must own writable memory, and
        # the copy is what makes a DataLoader worker's page cache reusable.
        window = np.array(self._shard(rec_idx)[w_idx], dtype=np.float32)
        return {
            "eeg": torch.from_numpy(window),  # (T, C), robust-scaled units
            "subject": rec["subject"],
            "run": rec["run"],
            "session": rec["session"],
            "source": self.source,
            "fs": float(rec["fs"]),
        }

    # -- provenance / auditing -------------------------------------------
    def provenance(self, idx: int) -> WindowProvenance:
        """Full provenance of one window, enough to re-extract it from the EDF."""
        rec_idx, w_idx = self.window_index[int(idx)]
        rec = self.recordings[rec_idx]
        return WindowProvenance(
            source=self.source,
            subject=rec["subject"],
            session=rec["session"],
            run=rec["run"],
            recording_id=rec["recording_id"],
            path=rec["path"],
            sample_offset=int(rec["window_starts"][w_idx]),
            scale_volts=float(rec["scale_volts"]),
        )

    @property
    def window_subjects(self) -> list[str]:
        """Subject id of every window, in dataset order.  The grouping unit."""
        return [self.recordings[r]["subject"] for r, _ in self.window_index]

    @property
    def subjects(self) -> list[str]:
        return sorted({rec["subject"] for rec in self.recordings})

    @property
    def windows_per_subject(self) -> dict[str, int]:
        counts: Counter[str] = Counter(self.window_subjects)
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        """Machine-readable description of what was actually loaded."""
        return {
            "source": self.source,
            "cache_dir": str(self.cache_root),
            "n_windows": len(self),
            "n_recordings": len(self.recordings),
            "n_subjects": len(self.subjects),
            "n_channels": len(self.channel_names),
            "n_samples_per_window": self.cfg.window_samples,
            "fs": float(self.cfg.fs_target),
            "channels": list(self.channel_names),
            "n_dropped_windows": self.n_dropped_windows,
            "n_cache_hits": self.n_cache_hits,
            "n_cache_misses": self.n_cache_misses,
            "skipped_recordings": list(self.skipped_recordings),
            "windows_per_subject": self.windows_per_subject,
        }

    def lineage_records(self) -> list[Any]:
        """One :class:`scwbd.sources.lineage.Record` per recording, for agent B's
        splitter.  Returns an empty list when ``scwbd.sources`` is unavailable.
        """
        try:
            from ..sources.lineage import Lineage, Record
        except ImportError:
            return []
        records = []
        for rec in self.recordings:
            records.append(
                Record(
                    id=rec["recording_id"],
                    source_id=self.source,
                    lineage=Lineage(
                        participant=rec["subject"],
                        site=self.source,
                        device=self.source,
                        session=rec["session"],
                        run=rec["run"],
                    ),
                    path=rec["path"],
                )
            )
        return records


# ======================================================================
# eegmmidb
# ======================================================================
class EEGMMIDBDataset(RealEEGDataset):
    """PhysioNet EEG Motor Movement/Imagery Database as fixed-length windows.

    64 monopolar scalp channels in the canonical :data:`EEGMMIDB_CHANNELS`
    order, average-referenced (the raw files are referenced to a single
    electrode, which makes the topography reference-dependent; the common
    average is the least arbitrary choice available without an individual head
    model).  One session per subject; ``run`` is the BCI2000 run label ``R01``
    ... ``R14``.
    """

    source = "eegmmidb"
    average_reference = True

    def _montage(self) -> Sequence[str]:
        return EEGMMIDB_CHANNELS

    def _discover(self) -> list[dict[str, str]]:
        root = self.cfg.eegmmidb_root
        if not root.is_dir():
            LOGGER.warning("eegmmidb root %s does not exist; dataset will be empty", root)
            return []
        by_subject: dict[str, list[Path]] = defaultdict(list)
        for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not re.fullmatch(r"S\d{3}", sub_dir.name):
                continue
            for edf in sorted(sub_dir.glob("*.edf")):
                m = _EEGMMIDB_RUN_RE.match(edf.name)
                if m is None or m.group("subject") != sub_dir.name:
                    continue
                # A file still being downloaded is far below the smallest real
                # run (~1.2 MB); skipping it here avoids a pointless mne failure.
                try:
                    if edf.stat().st_size < 100_000:
                        continue
                except OSError:
                    continue
                by_subject[sub_dir.name].append(edf)

        subjects = sorted(by_subject)
        if self.cfg.max_subjects is not None:
            subjects = subjects[: self.cfg.max_subjects]
        out: list[dict[str, str]] = []
        for subject in subjects:
            runs = sorted(by_subject[subject])
            if self.cfg.max_runs_per_subject is not None:
                runs = runs[: self.cfg.max_runs_per_subject]
            for edf in runs:
                run = _EEGMMIDB_RUN_RE.match(edf.name).group("run")
                out.append(
                    {
                        "recording_id": f"{subject}{run}",
                        "subject": subject,
                        "session": "ses-01",  # single visit per volunteer
                        "run": run,
                        "path": str(edf),
                    }
                )
        return out


# ======================================================================
# sleep-edf expanded
# ======================================================================
class SleepEDFDataset(RealEEGDataset):
    """Sleep-EDF Expanded (sleep-cassette) as fixed-length windows.

    **Two** EEG channels, not sixty-four, and they are bipolar derivations
    (Fpz-Cz, Pz-Oz).  This class deliberately does not zero-pad, interpolate or
    otherwise pretend to a 64-channel montage: a padded channel is fabricated
    data, and a model trained on it would learn that most of the scalp is
    silent.  Callers read :attr:`channel_names` and decide what to do -- train a
    separate observation head, mask the missing channels in the likelihood, or
    use this source for temporal structure only.

    For the same reason the montage is **not average-referenced**: the mean of
    two bipolar derivations is not a reference, and subtracting it would mix
    frontal and occipital signals into each other.

    ``subject`` is ``SC4<ss>`` and ``session`` is the night, so both nights of
    one volunteer share a participant and can never straddle a fold.
    """

    source = "sleep_edfx"
    average_reference = False

    def _montage(self) -> Sequence[str]:
        return SLEEP_EDF_EEG_CHANNELS

    def _discover(self) -> list[dict[str, str]]:
        root = self.cfg.sleep_edfx_root
        if not root.is_dir():
            LOGGER.warning("sleep-edfx root %s does not exist; dataset will be empty", root)
            return []
        by_subject: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for edf in sorted(root.glob("*-PSG.edf")):
            m = _SLEEP_PSG_RE.match(edf.name)
            if m is None:
                continue
            try:
                if edf.stat().st_size < 1_000_000:
                    continue
            except OSError:
                continue
            by_subject[m.group("subject")].append((m.group("night"), edf))

        subjects = sorted(by_subject)
        if self.cfg.max_subjects is not None:
            subjects = subjects[: self.cfg.max_subjects]
        out: list[dict[str, str]] = []
        for subject in subjects:
            nights = sorted(by_subject[subject])
            if self.cfg.max_runs_per_subject is not None:
                nights = nights[: self.cfg.max_runs_per_subject]
            for night, edf in nights:
                out.append(
                    {
                        "recording_id": edf.name[: -len("-PSG.edf")],
                        "subject": subject,
                        "session": f"night{night}",
                        "run": f"night{night}",  # one continuous run per night
                        "path": str(edf),
                    }
                )
        return out


# ======================================================================
# participant-level splitting
# ======================================================================
def _window_subjects(dataset: Any) -> list[str]:
    """Subject id per window, duck-typed so ``Subset`` and wrappers still work."""
    subjects = getattr(dataset, "window_subjects", None)
    if subjects is not None:
        return list(subjects)
    if isinstance(dataset, Subset):
        base = _window_subjects(dataset.dataset)
        return [base[i] for i in dataset.indices]
    return [str(dataset[i]["subject"]) for i in range(len(dataset))]


def _assign_groups(
    groups: Sequence[str], *, test_fraction: float, val_fraction: float, seed: int
) -> dict[str, str]:
    """Deterministically assign whole participants to train/val/test.

    Shuffling with an explicit seed and slicing by count (rather than hashing)
    keeps the realised fold sizes close to the requested fractions even for the
    small subject counts used in smoke tests.
    """
    ordered = sorted(set(groups))
    n = len(ordered)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    # With few participants, rounding can starve a fold entirely; give each
    # requested fold at least one participant while keeping train non-empty.
    if test_fraction > 0:
        n_test = max(1, n_test)
    if val_fraction > 0:
        n_val = max(1, n_val)
    while n_test + n_val >= n and (n_test + n_val) > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:  # pragma: no cover - unreachable while n >= 1
            break
    assignment: dict[str, str] = {}
    for i, g in enumerate(shuffled):
        if i < n_test:
            assignment[g] = "test"
        elif i < n_test + n_val:
            assignment[g] = "val"
        else:
            assignment[g] = "train"
    return assignment


def _hash_assign_groups(
    groups: Sequence[str], *, test_fraction: float, val_fraction: float, seed: int
) -> dict[str, str]:
    """Order-independent fallback: SHA-256 of ``seed:participant`` into [0, 1).

    Used when agent B's splitter cannot be imported or adapted.  It is stable
    under adding or removing participants (useful while a download completes),
    at the cost of noisier fold sizes.
    """
    assignment: dict[str, str] = {}
    for g in sorted(set(groups)):
        digest = hashlib.sha256(f"{seed}:{g}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if u < test_fraction:
            assignment[g] = "test"
        elif u < test_fraction + val_fraction:
            assignment[g] = "val"
        else:
            assignment[g] = "train"
    return assignment


def participant_split(
    dataset: Any,
    *,
    test_fraction: float = 0.25,
    val_fraction: float = 0.1,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Split *window indices* into train/val/test by **participant**.

    The grouping unit is the person.  Every run, night and session of a subject
    lands in exactly one fold, because the alternative -- windows of the same
    individual on both sides of the split -- measures memorisation of that
    individual and reports it as generalisation (thesis refusal ``R10``).

    The participant grouping is taken from :class:`scwbd.sources.splits.GroupedSplitter`
    when it can be imported and applied, so this module and agent B's evaluation
    code group identically (and so agent B's lineage checks, which refuse
    unresolved parentage, run here too).  If that import or adaptation fails,
    a deterministic hash-based participant split is used instead and
    ``dataset.participant_split_backend`` is set to ``"hash_fallback"``.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of integer indices
        into ``dataset``.
    """
    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("fractions must lie in [0, 1)")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must leave a training fold")
    if seed is None:
        seed = int(getattr(getattr(dataset, "cfg", None), "seed", 0))

    win_subjects = _window_subjects(dataset)
    if not win_subjects:
        setattr(dataset, "participant_split_backend", "empty")
        return {"train": [], "val": [], "test": []}

    backend = "hash_fallback"
    fallback_reason = "scwbd.sources.splits.GroupedSplitter not attempted"
    groups: list[str] = list(win_subjects)
    try:
        from ..sources.splits import GroupedSplitter

        records = dataset.lineage_records() if hasattr(dataset, "lineage_records") else []
        if not records:
            raise RuntimeError("dataset exposes no lineage records to group")
        splitter = GroupedSplitter(mode="participant", n_folds=2, seed=seed)
        # Use agent B's grouping (which enforces the lineage rules) but our own
        # fraction-based assignment: GroupedSplitter yields equal-sized k-folds,
        # while a foundation-model run wants an explicit train/val/test ratio.
        key_of_record = splitter.group_keys(records)
        group_of_subject = {
            rec["subject"]: key_of_record[rec["recording_id"]]
            for rec in dataset.recordings
            if rec["recording_id"] in key_of_record
        }
        if len(group_of_subject) < len({r["subject"] for r in dataset.recordings}):
            raise RuntimeError("GroupedSplitter did not key every recording")
        groups = [group_of_subject[s] for s in win_subjects]
        backend = "grouped_splitter"
        fallback_reason = ""
    except Exception as exc:  # noqa: BLE001 - any failure means: use the fallback
        LOGGER.info("GroupedSplitter unavailable (%s); using hash fallback", exc)
        fallback_reason = f"{type(exc).__name__}: {exc}"

    if backend == "grouped_splitter":
        assignment = _assign_groups(
            groups, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
        )
    else:
        assignment = _hash_assign_groups(
            groups, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
        )

    setattr(dataset, "participant_split_backend", backend)
    setattr(dataset, "participant_split_fallback_reason", fallback_reason)
    setattr(dataset, "participant_split_group_of_window", groups)

    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, group in enumerate(groups):
        split[assignment[group]].append(idx)
    return split


def leakage_check(split: dict[str, Iterable[int]], dataset: Any) -> dict[str, Any]:
    """Verify that no participant appears in more than one fold.

    This is an *audit*, not an assertion of good intent: it recomputes the
    participant of every window index from the dataset itself, so it catches
    hand-edited splits, concatenations and off-by-one index bugs, not just
    mistakes in :func:`participant_split`.

    Returns a machine-readable report; ``report["ok"]`` is the single bit that
    decides whether a downstream number may be called generalisation.
    """
    win_subjects = _window_subjects(dataset)
    n_total = len(win_subjects)

    folds = {name: [int(i) for i in idxs] for name, idxs in split.items()}
    subjects_per_fold: dict[str, list[str]] = {}
    for name, idxs in folds.items():
        subjects_per_fold[name] = sorted({win_subjects[i] for i in idxs if 0 <= i < n_total})

    violations: list[dict[str, Any]] = []

    out_of_range = {n: [i for i in idxs if not 0 <= i < n_total] for n, idxs in folds.items()}
    for name, bad in out_of_range.items():
        if bad:
            violations.append(
                {"kind": "index_out_of_range", "fold": name, "offending": bad[:10], "code": "R10"}
            )

    owner: dict[str, str] = {}
    for name in sorted(folds):
        for subj in subjects_per_fold[name]:
            prev = owner.setdefault(subj, name)
            if prev != name:
                violations.append(
                    {
                        "kind": "participant_across_folds",
                        "subject": subj,
                        "folds": sorted({prev, name}),
                        "code": "R10",
                    }
                )

    all_idx = [i for idxs in folds.values() for i in idxs]
    dupes = sorted({i for i, c in Counter(all_idx).items() if c > 1})
    if dupes:
        violations.append(
            {"kind": "duplicate_window_index", "offending": dupes[:10], "code": "R10"}
        )

    warnings: list[str] = []
    uncovered = n_total - len(set(all_idx))
    if uncovered:
        warnings.append(f"{uncovered} of {n_total} windows are in no fold")
    for name, subs in subjects_per_fold.items():
        if not subs and folds.get(name):
            warnings.append(f"fold {name!r} has windows but no resolvable subject")
        if not folds.get(name):
            warnings.append(f"fold {name!r} is empty")

    report: dict[str, Any] = {
        "ok": not violations,
        "code": "R10",
        "source": getattr(dataset, "source", "unknown"),
        "split_backend": getattr(dataset, "participant_split_backend", "unknown"),
        "split_fallback_reason": getattr(dataset, "participant_split_fallback_reason", ""),
        "n_windows_total": n_total,
        "n_windows_per_fold": {n: len(i) for n, i in folds.items()},
        "n_subjects_total": len(set(win_subjects)),
        "n_subjects_per_fold": {n: len(s) for n, s in subjects_per_fold.items()},
        "subjects_per_fold": subjects_per_fold,
        "violations": violations,
        "warnings": warnings,
        "note": (
            "Participant-disjoint folds make a generalisation claim possible; they do "
            "not make a window-level score a participant-level result. Aggregate per "
            "participant before reporting an interval."
        ),
    }

    # Cross-check against agent B's auditor when it is importable, so the two
    # implementations have to agree before a number is reported.
    try:
        from ..sources.splits import Fold, Split, leakage_audit

        records = dataset.lineage_records() if hasattr(dataset, "lineage_records") else []
        if records:
            rec_fold: dict[str, str] = {}
            for name, idxs in folds.items():
                for i in idxs:
                    if 0 <= i < n_total:
                        rec_fold[_recording_id_of_window(dataset, i)] = name
            names = sorted(folds)
            group_of = {r.id: r.lineage.group_key("participant") for r in records}
            b_folds = tuple(
                Fold(
                    index=k,
                    train_ids=tuple(sorted(i for i, f in rec_fold.items() if f != name)),
                    test_ids=tuple(sorted(i for i, f in rec_fold.items() if f == name)),
                    test_groups=tuple(
                        sorted({group_of[i] for i, f in rec_fold.items() if f == name})
                    ),
                )
                for k, name in enumerate(names)
            )
            b_split = Split(
                mode="participant",
                level="participant",
                seed=int(getattr(getattr(dataset, "cfg", None), "seed", 0)),
                folds=b_folds,
                group_of=group_of,
            )
            audit = leakage_audit(b_split, records)
            report["grouped_splitter_audit"] = {
                "ok": bool(audit.ok),
                "violations": [str(v) for v in audit.violations],
                "warnings": list(audit.warnings),
            }
            if not audit.ok:
                report["ok"] = False
    except Exception as exc:  # noqa: BLE001 - the audit is a bonus, not a dependency
        report["grouped_splitter_audit"] = {"ok": None, "unavailable": f"{type(exc).__name__}: {exc}"}

    return report


def _recording_id_of_window(dataset: Any, idx: int) -> str:
    """Recording id for a window index, for the record-level cross-audit."""
    if isinstance(dataset, Subset):
        return _recording_id_of_window(dataset.dataset, dataset.indices[idx])
    rec_idx, _ = dataset.window_index[idx]
    return str(dataset.recordings[rec_idx]["recording_id"])


# ======================================================================
# loaders
# ======================================================================
def make_loaders(
    cfg: RealEEGConfig,
    batch_size: int = 32,
    *,
    dataset: RealEEGDataset | None = None,
    source: str = "eegmmidb",
    test_fraction: float = 0.25,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    shuffle_train: bool = True,
    report: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    """Build participant-disjoint train/val/test loaders over measured EEG.

    ``report``, if a dict is passed in, is filled with the dataset summary and
    the leakage report so a training script can serialise the audit next to the
    checkpoint.  The split is refused (``RuntimeError``) if the audit fails --
    a leaky loader is worse than no loader, because it produces a number.

    Only the training loader is shuffled, and shuffling is a *window* shuffle
    within participant-disjoint folds; it does not and cannot repair a split.
    """
    if dataset is None:
        if source == "eegmmidb":
            dataset = EEGMMIDBDataset(cfg)
        elif source in ("sleep_edfx", "sleep-edfx"):
            dataset = SleepEDFDataset(cfg)
        else:
            raise ValueError(f"unknown source {source!r}; expected 'eegmmidb' or 'sleep_edfx'")

    split = participant_split(
        dataset, test_fraction=test_fraction, val_fraction=val_fraction, seed=cfg.seed
    )
    audit = leakage_check(split, dataset)
    if not audit["ok"]:
        raise RuntimeError(
            "participant-level leakage audit failed (R10): "
            + json.dumps(audit["violations"][:5])
        )
    if report is not None:
        report.clear()
        report.update({"summary": dataset.summary(), "leakage": audit, "config": cfg.as_json()})

    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, DataLoader] = {}
    for name, indices in split.items():
        subset = Subset(dataset, indices)
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=bool(shuffle_train and name == "train" and len(indices) > 0),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=bool(drop_last and name == "train"),
            generator=generator if name == "train" else None,
        )
    return loaders
