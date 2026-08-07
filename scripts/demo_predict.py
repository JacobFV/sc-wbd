"""What the published checkpoint actually produces, end to end.

Loads scwbd-002-pilot, feeds it a real held-out EEG window, rolls the dynamics
forward, and reports every output the runtime resolves. Outputs that could not be
restored come back as `Unresolved` rather than as zeros, so this is a capability
inventory rather than a demo that always prints something.

    .venv/bin/python scripts/demo_predict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/scwbd-002-pilot/last.pt"


def main() -> int:
    from scwbd.runtime.predict import LoadedModel, Unresolved

    if not CKPT.is_file():
        print(f"no checkpoint at {CKPT}", file=sys.stderr)
        return 1

    m = LoadedModel.from_checkpoint(str(CKPT))
    n = int(m._model.anat.n_regions)

    # A real measured window, not noise: the first cached EEG the trainer uses.
    from scwbd.foundation.realdata import RealEEGConfig, RealEEGDataset

    try:
        ds = RealEEGDataset(RealEEGConfig())
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

    print(f"\ncheckpoint : {CKPT.relative_to(ROOT)}")
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
