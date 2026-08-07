"""Windows of measured BOLD, already in parcel space, with their coverage.

The counterpart to :mod:`scwbd.foundation.realdata`, and the last piece between
this project and a second measured modality. Everything it composes already
existed and was unused: ``scwbd/anatomy/registration.py`` (the EPI ← T1w ←
template chain), ``scwbd/sources/parcellate_bold.py`` (the consumer that joins
it to the atlas), and ``FoundationTrainer.real_bold_losses`` (the parcel-space
likelihood).

**One item is ``window_frames`` frames of one run of one person**, exactly as in
``realdata``, and it is not an independent sample. Windows within a run share
the run's registration, its motion, and its physiological noise; runs within a
subject share an anatomy. Any interval computed over windows as though they were
independent will be too narrow, and the participant-level split is what makes
the grouping honest rather than the window count.

**Coverage travels with the data.** Every item carries ``bold_mask``, and
uncovered parcels are ``NaN`` in the timeseries rather than zero. A model that
receives a zero for a parcel that was never in the acquisition's field of view
cannot tell it from a real measurement of zero, and neither can its loss.

Registration is the expensive part — about 160 s per run — so parcellated runs
are cached to ``.npy`` beside a JSON sidecar carrying the coverage and the
provenance. The chain is estimated once per *subject* and reused across that
subject's runs, which is what makes 55 runs cost roughly the number of subjects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ParcelBOLDConfig", "ParcelBOLDDataset", "discover_bold_runs"]


@dataclass
class ParcelBOLDConfig:
    """Where the data is, how it is windowed, and where the cache lives."""

    root: str = "data/ds002336"
    cache_dir: str = "data/foundation_cache/parcel_bold"
    assets: str = "data/assets"
    atlas: str = "Schaefer400x7"
    source: str = "ds002336_real"
    #: Frames per window. At TR = 2 s, 32 frames is 64 s.
    window_frames: int = 32
    #: Frames advanced between consecutive windows; < window_frames overlaps.
    stride_frames: int = 16
    #: Skip the first N frames of every run (T1 saturation is not signal).
    drop_initial_frames: int = 5
    nonlinear: bool = False
    #: Refuse a run whose field of view covers fewer than this fraction of
    #: parcels. Not a quality threshold -- a sanity one: a run covering almost
    #: nothing is a registration failure, and averaging it in would be worse
    #: than dropping it loudly.
    min_coverage: float = 0.5
    tasks: tuple[str, ...] = ()


@dataclass
class _Run:
    subject: str
    task: str
    bold: Path
    t1w: Path
    cache_npy: Path
    cache_json: Path
    meta: dict[str, Any] = field(default_factory=dict)


_ENT = re.compile(r"sub-([A-Za-z0-9]+).*?task-([A-Za-z0-9]+)")


def discover_bold_runs(cfg: ParcelBOLDConfig) -> list[_Run]:
    """Every BOLD run under ``root`` that has a T1w for its subject."""
    root = Path(cfg.root)
    cache = Path(cfg.cache_dir)
    out: list[_Run] = []
    for bold in sorted(root.rglob("*_bold.nii.gz")):
        if "derivatives" in bold.parts:
            continue
        m = _ENT.search(bold.name)
        if not m:
            continue
        sub, task = m.group(1), m.group(2)
        if cfg.tasks and task not in cfg.tasks:
            continue
        anat = bold.parent.parent / "anat"
        t1w = next(iter(sorted(anat.glob(f"sub-{sub}_T1w.nii*"))), None)
        if t1w is None:
            # Recorded by omission rather than by imputing a template T1w: a
            # subject without their own anatomy cannot be registered to their
            # own brain, and substituting the template is the classic silent
            # several-mm error this whole path exists to avoid.
            continue
        stem = f"sub-{sub}_task-{task}"
        out.append(
            _Run(
                subject=sub,
                task=task,
                bold=bold,
                t1w=t1w,
                cache_npy=cache / f"{stem}__{cfg.atlas}.npy",
                cache_json=cache / f"{stem}__{cfg.atlas}.json",
            )
        )
    return out


class ParcelBOLDDataset(Dataset):
    """Parcel-space BOLD windows with coverage masks.

    Items are ``{"bold": (T, N), "bold_mask": (N,), "subject", "run",
    "source", "tr"}`` — the shape ``FoundationTrainer.real_bold_losses``
    expects.
    """

    def __init__(self, cfg: ParcelBOLDConfig, *, build: bool = True) -> None:
        self.cfg = cfg
        self.source = cfg.source
        self.runs = discover_bold_runs(cfg)
        if not self.runs:
            raise FileNotFoundError(
                f"no BOLD runs with a matching T1w under {cfg.root!r}. This dataset "
                "will not fall back to a template anatomy: registering a subject's "
                "EPI to somebody else's brain is a systematic error, not a "
                "degraded mode."
            )
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        if build:
            self.build_cache()
        self._index: list[tuple[int, int]] = []
        self._dropped: list[dict[str, Any]] = []
        self._reindex()

    # -- cache ---------------------------------------------------------
    def build_cache(self, *, force: bool = False) -> dict[str, Any]:
        """Parcellate every run that is not cached. ~160 s per uncached run."""
        from scwbd.sources.parcellate_bold import parcellate_run

        built, reused, chains = 0, 0, {}
        for r in self.runs:
            if r.cache_npy.exists() and r.cache_json.exists() and not force:
                reused += 1
                continue
            pb, chain = parcellate_run(
                r.bold,
                r.t1w,
                atlas=self.cfg.atlas,
                assets=self.cfg.assets,
                subject=r.subject,
                run=r.task,
                nonlinear=self.cfg.nonlinear,
                chain=chains.get(r.subject),
            )
            chains[r.subject] = chain
            np.save(r.cache_npy, pb.timeseries.astype(np.float32))
            r.cache_json.write_text(
                json.dumps(
                    {
                        **pb.describe(),
                        "covered": pb.covered.astype(bool).tolist(),
                        "n_voxels": pb.n_voxels.astype(int).tolist(),
                    },
                    indent=1,
                    default=str,
                )
            )
            built += 1
        return {"built": built, "reused": reused, "subjects": len(chains)}

    # -- indexing ------------------------------------------------------
    def _reindex(self) -> None:
        self._index.clear()
        self._dropped.clear()
        c = self.cfg
        for ri, r in enumerate(self.runs):
            if not (r.cache_npy.exists() and r.cache_json.exists()):
                continue
            meta = json.loads(r.cache_json.read_text())
            cov = np.asarray(meta["covered"], dtype=bool)
            frac = float(cov.mean())
            if frac < c.min_coverage:
                # Loud, not silent: a dropped run is a fact about the corpus.
                self._dropped.append(
                    {"subject": r.subject, "run": r.task, "coverage_fraction": round(frac, 4)}
                )
                continue
            r.meta = meta
            n_frames = int(meta["n_frames"])
            usable = n_frames - c.drop_initial_frames
            for start in range(0, max(0, usable - c.window_frames + 1), c.stride_frames):
                self._index.append((ri, start + c.drop_initial_frames))

    @property
    def dropped_runs(self) -> list[dict[str, Any]]:
        return list(self._dropped)

    def participants(self) -> list[str]:
        return sorted({r.subject for r in self.runs})

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ri, start = self._index[int(idx)]
        r = self.runs[ri]
        ts = np.load(r.cache_npy, mmap_mode="r")  # (N, T)
        win = np.array(ts[:, start : start + self.cfg.window_frames], dtype=np.float32).T
        cov = np.asarray(r.meta["covered"], dtype=bool)
        return {
            "bold": torch.from_numpy(win),  # (T, N)
            "bold_mask": torch.from_numpy(cov),  # (N,)
            "subject": r.subject,
            "run": r.task,
            "source": self.source,
            "tr": float(r.meta["tr_seconds"]),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "atlas": self.cfg.atlas,
            "runs_discovered": len(self.runs),
            "runs_cached": sum(1 for r in self.runs if r.cache_npy.exists()),
            "runs_dropped_low_coverage": len(self._dropped),
            "dropped": self._dropped,
            "participants": len(self.participants()),
            "windows": len(self),
            "window_frames": self.cfg.window_frames,
            "unobserved_are_nan": True,
        }
