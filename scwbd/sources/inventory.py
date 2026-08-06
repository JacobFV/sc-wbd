"""Measure what is on disk, by opening the files.

Everything here is re-derived from the bytes.  Nothing is read off a source
card, a manifest, an earlier report or a dataset's own documentation:

* file counts and sizes come from walking the tree;
* participant counts come from the directories that exist;
* **sampling rates and durations come from the recording headers**, one file at
  a time — EDF and BrainVision and FIFF through MNE, NIfTI through nibabel,
  BIDS physio through its sidecar plus a row count of the decompressed file.

That distinction is the whole point.  ``scwbd/sources/report.py`` renders the
registry and the cards, which is a report about *what we said*.  This module is
a report about *what we have*, and when the two disagree the disagreement is
the finding.

Header-only reads throughout: no sample data is loaded, so a 336 MB
BrainVision run and a 450 MB BOLD series each cost a few kilobytes.  The one
exception is BIDS physio, whose duration cannot be known without counting rows;
those are streamed through gzip and counted, never held in memory.

::

    python -m scwbd.sources.inventory                 # every registered dataset
    python -m scwbd.sources.inventory ds002336        # one
    python -m scwbd.sources.inventory --json out.json
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "RecordingFact",
    "DatasetInventory",
    "measure_dataset",
    "measure_all",
    "modality_totals",
]


@dataclass
class RecordingFact:
    """One measured recording: what it is, how fast, how long."""

    relpath: str
    kind: str            # eeg | meg | bold | physio | anat | dwi
    sfreq_hz: float | None
    n_samples: int | None
    duration_s: float | None
    n_channels: int | None
    bytes: int
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetInventory:
    dataset_id: str
    version: str
    root: str
    exists: bool
    n_files: int = 0
    total_bytes: int = 0
    participants: list[str] = field(default_factory=list)
    recordings: list[RecordingFact] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def hours_by_kind(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.recordings:
            if r.duration_s:
                out[r.kind] += r.duration_s / 3600.0
        return dict(sorted(out.items()))

    def rates_by_kind(self) -> dict[str, list[float]]:
        out: dict[str, set[float]] = defaultdict(set)
        for r in self.recordings:
            if r.sfreq_hz:
                out[r.kind].add(round(float(r.sfreq_hz), 6))
        return {k: sorted(v) for k, v in sorted(out.items())}

    def counts_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.recordings:
            out[r.kind] += 1
        return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# per-format header readers
# ---------------------------------------------------------------------------
def _mne_raw_fact(path: Path, root: Path, kind: str) -> RecordingFact:
    import mne

    name = path.name.lower()
    if name.endswith(".edf"):
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    elif name.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    elif name.endswith(".fif"):
        raw = mne.io.read_raw_fif(path, preload=False, allow_maxshield=True, verbose="ERROR")
    elif name.endswith(".meg4"):
        raw = mne.io.read_raw_ctf(path.parent, preload=False, verbose="ERROR")
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"no header reader for {path.name}")
    sfreq = float(raw.info["sfreq"])
    n = int(raw.n_times)
    by_type = {t: len(i) for t, i in mne.channel_indices_by_type(raw.info).items() if i}
    return RecordingFact(
        relpath=str(path.relative_to(root)),
        kind=kind,
        sfreq_hz=sfreq,
        n_samples=n,
        duration_s=n / sfreq if sfreq else None,
        n_channels=len(raw.ch_names),
        bytes=path.stat().st_size,
        detail={"channel_types": by_type},
    )


def _bold_fact(path: Path, root: Path, source_id: str) -> RecordingFact:
    from .loaders.bids_bold import load_bold_run

    v = load_bold_run(path, source_id=source_id)
    return RecordingFact(
        relpath=str(path.relative_to(root)),
        kind="bold",
        sfreq_hz=float(v.meta["sfreq_hz"]),
        n_samples=int(v.meta["n_volumes"]),
        duration_s=float(v.meta["duration_s"]),
        n_channels=None,
        bytes=path.stat().st_size,
        detail={
            "shape": list(v.meta["shape"]),
            "tr_s": v.tr,
            "tr_source": v.meta["tr_source"],
            "voxel_mm": [round(x, 3) for x in v.voxel_size],
            "frame": v.frame_id,
            "task": v.meta.get("task"),
        },
    )


def _gz_rows(path: Path) -> int:
    """Row count of a gzipped TSV, streamed. Never held in memory."""
    n = 0
    with gzip.open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def _physio_fact(path: Path, root: Path) -> RecordingFact:
    from .loaders.bids_bold import bold_sidecar

    side = bold_sidecar(path)
    sfreq = side.get("SamplingFrequency")
    cols = side.get("Columns")
    if not sfreq or not cols:
        raise ValueError(
            "no BIDS physio sidecar states SamplingFrequency/Columns for this file"
        )
    n = _gz_rows(path)
    return RecordingFact(
        relpath=str(path.relative_to(root)),
        kind="physio",
        sfreq_hz=float(sfreq),
        n_samples=n,
        duration_s=n / float(sfreq),
        n_channels=len(cols),
        bytes=path.stat().st_size,
        detail={"columns": list(cols), "recording": _entity(path, "recording")},
    )


def _entity(path: Path, key: str) -> str | None:
    for token in path.name.split("_"):
        if token.startswith(f"{key}-"):
            return token.split("-", 1)[1]
    return None


def _volume_fact(path: Path, root: Path, kind: str) -> RecordingFact:
    """A 3-D structural/diffusion volume: shape and voxel size, no clock."""
    import nibabel as nib

    img = nib.load(path)
    zooms = [round(float(z), 3) for z in img.header.get_zooms()[:3]]
    return RecordingFact(
        relpath=str(path.relative_to(root)),
        kind=kind,
        sfreq_hz=None,
        n_samples=None,
        duration_s=None,
        n_channels=None,
        bytes=path.stat().st_size,
        detail={"shape": [int(x) for x in img.shape], "voxel_mm": zooms},
    )


# ---------------------------------------------------------------------------
# what counts as what
# ---------------------------------------------------------------------------
def _classify(path: Path) -> str | None:
    """Map a file to a measurable kind, or ``None`` to skip it.

    Deliberately conservative.  ``defacemask`` volumes and derivative trees are
    skipped because counting them as anatomy would inflate a number nobody
    would then be able to reproduce by looking at the tree.
    """
    n = path.name.lower()
    if "defacemask" in n:
        return None
    if n.endswith("_bold.nii.gz") or n.endswith("_bold.nii"):
        return "bold"
    if n.endswith("_physio.tsv.gz"):
        return "physio"
    if n.endswith(("_t1w.nii.gz", "_t2w.nii.gz", "_flash.nii.gz", "_veno.nii.gz",
                   "_angio.nii.gz")):
        return "anat"
    if n.endswith("_dwi.nii.gz"):
        return "dwi"
    if n.endswith(".edf"):
        # Sleep-EDF ships two EDFs per night: the PSG (real signal) and a
        # Hypnogram (EDF+ annotations only, no physiological channel). MNE
        # reports a duration for both, so counting them together double-counts
        # every recorded hour. They are different kinds and are counted apart.
        return "hypnogram" if "hypnogram" in n else "eeg"
    if n.endswith(".vhdr"):
        return "eeg"
    if n.endswith("_meg.fif") or n.endswith("_raw.fif"):
        return "meg"
    # CTF stores a run as a DIRECTORY; the .meg4 is the sample payload and the
    # sibling .res4 the header. Keyed on .meg4 so each run is counted once.
    if n.endswith(".meg4"):
        return "meg_ctf"
    return None


def _participants(root: Path) -> list[str]:
    """Participant directories that exist, whatever the tree's convention.

    BIDS puts ``sub-*`` at the root; MNE's BIDS-ish trees nest it one level
    down (``MNE-somato-data/sub-01``); PhysioNet uses ``S001..S109``.  A single
    hard-coded glob silently returns zero for two of the three, and "0
    participants" next to 3 GB of data reads as a broken tree rather than a
    naming convention.
    """
    subs = sorted({p.name for p in root.rglob("sub-*")
                   if p.is_dir() and "derivatives" not in p.relative_to(root).parts})
    if subs:
        return subs
    physionet = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and len(p.name) <= 6 and p.name[:1].isalpha()
        and p.name[1:].isdigit()
    )
    if physionet:
        return physionet
    # Sleep-EDF: no participant directory at all. The subject code is the
    # first five characters of the file name (SC4001E0-PSG.edf -> SC400), and
    # two nights of one sleeper share it. Returning [] here would report "0
    # participants" for 78 people and make every per-participant split look
    # impossible.
    edfs = sorted({p.name[:5] for p in root.rglob("*-PSG.edf")})
    if edfs:
        return edfs
    # A single-subject demonstration tree (mne-sample, mne-spm-face): the
    # subject is the FreeSurfer directory that is not a template.
    templates = {"fsaverage", "fsaverage_sym", "morph-maps"}
    named = sorted(
        p.name for p in root.rglob("subjects/*")
        if p.is_dir() and p.name not in templates
    )
    return named


def measure_dataset(dataset_id: str, *, limit: int | None = None) -> DatasetInventory:
    from .registry import get

    entry = get(dataset_id)
    root = entry.local_path
    inv = DatasetInventory(
        dataset_id=dataset_id,
        version=entry.version,
        root=str(root),
        exists=root.exists(),
    )
    if not root.exists():
        inv.notes.append("root is not on disk; nothing measured")
        return inv

    from .manifest import DEFAULT_EXCLUDES, iter_files

    files = iter_files(root, DEFAULT_EXCLUDES)
    inv.n_files = len(files)
    inv.total_bytes = sum(p.stat().st_size for p in files)
    inv.participants = _participants(root)

    # Leftovers from a resumable fetch. They are real bytes on disk and would
    # otherwise be counted as data, so they are reported, not silently dropped.
    stray = [p for p in files if p.name.endswith((".part", ".part.done"))]
    if stray:
        inv.notes.append(
            f"{len(stray)} partial-download artefact(s) present "
            f"(*.part / *.part.done); these are NOT data and must be removed "
            f"before the manifest is built"
        )

    n_seen = 0
    for p in files:
        if "derivatives" in p.relative_to(root).parts:
            continue
        kind = _classify(p)
        if kind is None:
            continue
        if limit is not None and n_seen >= limit:
            inv.notes.append(f"stopped after {limit} recordings (--limit)")
            break
        n_seen += 1
        try:
            if kind in ("eeg", "meg", "meg_ctf", "hypnogram"):
                inv.recordings.append(_mne_raw_fact(p, root, kind))
            elif kind == "bold":
                inv.recordings.append(_bold_fact(p, root, dataset_id))
            elif kind == "physio":
                inv.recordings.append(_physio_fact(p, root))
            else:
                inv.recordings.append(_volume_fact(p, root, kind))
        except Exception as exc:
            inv.unreadable.append((str(p.relative_to(root)), f"{type(exc).__name__}: {exc}"))
    return inv


def measure_all(dataset_ids: Sequence[str] | None = None, **kw: Any) -> dict[str, DatasetInventory]:
    from .registry import REGISTRY

    ids = list(dataset_ids) if dataset_ids else list(REGISTRY)
    out: dict[str, DatasetInventory] = {}
    for did in ids:
        print(f"  measuring {did} ...", file=sys.stderr, flush=True)
        out[did] = measure_dataset(did, **kw)
    return out


def modality_totals(invs: Iterable[DatasetInventory]) -> dict[str, dict[str, float]]:
    """Hours and file counts per kind, summed across datasets."""
    hours: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for inv in invs:
        for k, h in inv.hours_by_kind().items():
            hours[k] += h
        for k, c in inv.counts_by_kind().items():
            counts[k] += c
    return {
        k: {"hours": round(hours.get(k, 0.0), 3), "files": counts[k]}
        for k in sorted(counts)
    }


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.inventory")
    ap.add_argument("dataset_id", nargs="*")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    invs = measure_all(args.dataset_id or None, limit=args.limit)
    for did, inv in invs.items():
        if not inv.exists:
            print(f"{did:<22} NOT ON DISK ({inv.root})")
            continue
        print(
            f"{did:<22} {inv.n_files:>6} files {inv.total_bytes / 1e9:>8.3f} GB "
            f"{len(inv.participants):>4} participants  "
            f"{ {k: round(v, 2) for k, v in inv.hours_by_kind().items()} }"
        )
        for rel, why in inv.unreadable[:3]:
            print(f"      UNREADABLE {rel}: {why[:110]}")
        if len(inv.unreadable) > 3:
            print(f"      ... and {len(inv.unreadable) - 3} more unreadable")
        for n in inv.notes:
            print(f"      NOTE {n}")
    print("\nTOTALS", json.dumps(modality_totals(invs.values()), indent=1))
    if args.json:
        args.json.write_text(json.dumps({k: asdict(v) for k, v in invs.items()}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
