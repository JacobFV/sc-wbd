"""What a published checkpoint actually produces, end to end.

Feeds a real held-out EEG window through the checkpoint, rolls the dynamics
forward, and reports every output the runtime resolves. Outputs that could not be
restored come back as `Unresolved` rather than as zeros, so this is a capability
inventory rather than a demo that always prints something.

    .venv/bin/python scripts/demo_predict.py
    .venv/bin/python scripts/demo_predict.py --checkpoint checkpoints/scwbd-003/last.pt

The checkpoint is an argument rather than a literal. It was
``checkpoints/scwbd-002-pilot/last.pt`` hardcoded, which is the naming class of
defect this project keeps paying for -- a run number written into a file that
has to be found by grepping when the next run ships. The default still points at
the newest checkpoint that exists, resolved at call time, so the common case
needs no argument and cannot go stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

#: Newest first. The default is the first of these that is on disk, so this
#: script follows the run series without being edited.
KNOWN_CHECKPOINTS = (
    ROOT / "checkpoints/scwbd-003/last.pt",
    ROOT / "checkpoints/scwbd-002-pilot/last.pt",
    ROOT / "checkpoints/scwbd-001-beta/last.pt",
)


def default_checkpoint() -> Path | None:
    return next((p for p in KNOWN_CHECKPOINTS if p.is_file()), None)


def main(argv: list[str] | None = None) -> int:
    from scwbd.runtime.predict import LoadedModel, Unresolved

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    CKPT = Path(a.checkpoint) if a.checkpoint else default_checkpoint()

    if CKPT is None or not CKPT.is_file():
        print(
            f"no checkpoint at {CKPT}. Tried: "
            + ", ".join(str(p.relative_to(ROOT)) for p in KNOWN_CHECKPOINTS),
            file=sys.stderr,
        )
        return 1

    m = LoadedModel.from_checkpoint(str(CKPT))
    n = int(m._model.anat.n_regions)

    # A real measured window, not noise: the first cached EEG the trainer uses.
    # EEGMMIDBDataset, not RealEEGDataset: the latter is the abstract base and
    # its _montage() raises NotImplementedError. Instantiating the base is why
    # this demo silently fell back to synthetic context on its first run.
    from scwbd.foundation.realdata import EEGMMIDBDataset, RealEEGConfig

    try:
        ds = EEGMMIDBDataset(RealEEGConfig())
        item = ds[0]
        ctx = item["eeg"].unsqueeze(0)  # (1, T, C)
        source = f"{item.get('subject','?')} run {item.get('run','?')} ({ctx.shape[1]} samples, {ctx.shape[2]} ch)"
    except Exception as exc:  # dataset not cached on this machine
        print(f"[note] measured EEG unavailable ({type(exc).__name__}); using the "
              "regional-state entry point instead", flush=True)
        ctx, source = torch.randn(1, 16, n) * 0.1, "synthetic regional context"

    out = m.predict(ctx if ctx.shape[-1] == n else torch.randn(1, 16, n) * 0.1, n_steps=8)

    def describe(name, v):
        if isinstance(v, Unresolved) or v is None:
            return f"{name:18} UNRESOLVED — parameters not restored, no number reported"
        t = torch.as_tensor(v)
        return (f"{name:18} {tuple(t.shape)}  finite={bool(torch.isfinite(t).all())}  "
                f"range=[{t.min():.4g}, {t.max():.4g}]")

    try:
        shown = CKPT.relative_to(ROOT)
    except ValueError:  # a checkpoint outside the repo is a legal argument
        shown = CKPT
    print(f"\ncheckpoint : {shown}")
    print(f"context    : {source}")
    print(f"regions    : {n}   restored tensors: {out.load_report.restored}\n")
    for nm, v in (("activity", out.activity), ("activity_logvar", out.activity_logvar),
                  ("eeg", out.eeg), ("eeg_logvar", out.eeg_logvar),
                  ("hemodynamic", out.hemodynamic)):
        print("  " + describe(nm, v))
    print(f"\n  residual_ratio   {out.residual_ratio:.4f}   (R05: ||R|| / ||F_local + F_long||)")

    rep = {"restored": out.load_report.restored,
           "uninitialised": len(out.load_report.uninitialised),
           "ignored": len(out.load_report.ignored)}
    print(f"  load_report      {json.dumps(rep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
