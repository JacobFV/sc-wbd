"""Correct a checkpoint's ``model_id`` to the designation its config derives.

Why this exists rather than just fixing the trainer: ``checkpoint.py`` hardcoded
``model_id="SC-WBD-001-beta"`` for the whole of run 2, and the fix landed while
the run was still training. A live Python process does not re-read its modules,
so every checkpoint this run writes — including the final one — carries the
run-1 name. The repair has to be applied to the artifact after the fact.

It rewrites exactly one string. It does not touch weights, optimizer state, or
any recorded measurement, and it refuses rather than guessing whenever the
situation is not the one it was written for:

* refuses if the checkpoint's ``config`` does not derive a designation;
* refuses if ``model_id`` already matches (nothing to do is not a silent
  success — it means your assumption about the artifact was wrong);
* refuses to overwrite in place without ``--force``, writing beside the
  original instead, because a corrupted checkpoint at the end of a multi-hour
  run is not recoverable;
* records the previous value in ``extra.designation_restamp`` so the change is
  visible in the artifact itself rather than only in this file's git history.

Usage::

    python scripts/restamp_designation.py checkpoints/scwbd-002-pilot/last.pt
    python scripts/restamp_designation.py <ckpt> --force   # rewrite in place
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scwbd.foundation.config import FoundationConfig, designation  # noqa: E402


def restamp(path: Path, *, force: bool = False, dry_run: bool = False) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)

    raw = ck.get("config")
    if not raw:
        print(f"REFUSED: {path} records no config; the designation cannot be derived")
        return 2
    cfg = FoundationConfig.from_dict(raw) if isinstance(raw, dict) else raw

    want = designation(cfg)
    have = ck.get("model_id")
    if want == "SC-WBD-unnamed":
        print(f"REFUSED: {path}'s config derives no designation (got the unnamed fallback)")
        return 2
    if have == want:
        print(f"REFUSED: {path} already carries model_id={have!r}; nothing to restamp")
        return 3

    print(f"{path}\n  model_id: {have!r} -> {want!r}")
    if dry_run:
        print("  dry run, nothing written")
        return 0

    ck["model_id"] = want
    extra = ck.setdefault("extra", {})
    extra["designation_restamp"] = {
        "was": have,
        "now": want,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": (
            "checkpoint.py hardcoded the run-1 designation for the whole of run 2; "
            "the fix landed mid-run and a live process does not re-read its modules"
        ),
    }

    out = path if force else path.with_name(path.stem + ".restamped" + path.suffix)
    if force:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"  backup: {backup.name}")
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(out)
    print(f"  wrote: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--force", action="store_true", help="rewrite in place (keeps a .bak)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.checkpoint.exists():
        print(f"REFUSED: {a.checkpoint} does not exist")
        return 2
    return restamp(a.checkpoint, force=a.force, dry_run=a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
