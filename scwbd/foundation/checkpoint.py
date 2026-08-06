"""Checkpointing: weights + optimizer + RNG + normalizer state + config + provenance.

A checkpoint that cannot be resumed bit-for-bit is not a checkpoint, and a
checkpoint that does not say which code produced it is not an artifact.  Every
save records the git SHA, the exact config, the state layout, the normalizer
state and the RNG state of every generator the run touches.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import FoundationConfig
from .manifest import ClaimManifest, hash_file
from .util import env_fingerprint, git_sha

__all__ = ["save_checkpoint", "load_checkpoint", "latest_checkpoint", "CheckpointError"]


class CheckpointError(RuntimeError):
    pass


def _rng_state() -> dict[str, Any]:
    import random

    import numpy as np

    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(d: Mapping[str, Any]) -> None:
    import random

    import numpy as np

    if "torch" in d:
        torch.set_rng_state(d["torch"].cpu() if torch.is_tensor(d["torch"]) else d["torch"])
    if d.get("torch_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s for s in d["torch_cuda"]])
        except Exception:  # noqa: BLE001 - different device count on resume
            pass
    if "numpy" in d:
        np.random.set_state(d["numpy"])
    if "python" in d:
        random.setstate(d["python"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: FoundationConfig,
    step: int,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    posterior: torch.nn.Module | None = None,
    individualizer: torch.nn.Module | None = None,
    normalizer_state: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    manifest: ClaimManifest | None = None,
    extra: Mapping[str, Any] | None = None,
    save_rng: bool = True,
) -> Path:
    """Write a resumable checkpoint directory-relative file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "scwbd-foundation-checkpoint/1",
        "model_id": "SC-WBD-001-beta",
        "step": int(step),
        "stage": stage,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "environment": env_fingerprint(),
        "config": config.as_dict(),
        "state_layout": model.layout.as_dict() if hasattr(model, "layout") else None,
        # Which arm of body.tex §11.4 this artifact is, as a machine-readable
        # property of the weights rather than a sentence in a report. Refusal
        # R12 reads it; run 1 had nowhere to put it and was described as the
        # treatment arm while being the control (reports/scope_gap.md).
        "regional_state": model.family_report() if hasattr(model, "family_report") else None,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "posterior": posterior.state_dict() if posterior is not None else None,
        "individualizer": individualizer.state_dict() if individualizer is not None else None,
        "normalizer": dict(normalizer_state or {}),
        "metrics": dict(metrics or {}),
        "rng": _rng_state() if save_rng else None,
        "extra": dict(extra or {}),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    # sidecars, always human-readable
    config.save(p.parent / "config.yaml")
    (p.parent / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": payload["git_sha"],
                "saved_utc": payload["saved_utc"],
                "step": step,
                "stage": stage,
                "environment": payload["environment"],
                "weights_sha256": hash_file(p),
                "checkpoint": p.name,
            },
            indent=2,
            default=str,
        )
    )
    if manifest is not None:
        if payload["regional_state"] is not None:
            manifest.declare_regional_state(payload["regional_state"])
        manifest.git_sha = payload["git_sha"]
        manifest.weights_hash = hash_file(p)
        manifest.environment = payload["environment"]
        # D8: the config is in scope here, and R12's prolongation half needs it.
        manifest.save(p.parent / "claim_manifest.json", config=config)
        (p.parent / "CLAIM_MANIFEST.md").write_text(manifest.to_markdown())
    return p


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    posterior: torch.nn.Module | None = None,
    individualizer: torch.nn.Module | None = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring every piece of run state."""
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"no checkpoint at {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    if payload.get("format") != "scwbd-foundation-checkpoint/1":
        raise CheckpointError(f"unrecognised checkpoint format {payload.get('format')!r}")
    if model is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if not strict and (missing or unexpected):
            payload.setdefault("load_report", {})["missing"] = list(missing)
            payload["load_report"]["unexpected"] = list(unexpected)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    # The load report must cover every module it appears to cover, and must
    # distinguish "loaded cleanly" from "was not there". A caller that gates on
    # load_report otherwise reads absence as success -- the silent-load failure,
    # one level up.
    for name, mod in (("posterior", posterior), ("individualizer", individualizer)):
        if mod is None:
            continue
        if not payload.get(name):
            payload.setdefault("load_report", {})[f"{name}_absent"] = True
            continue
        miss, unexp = mod.load_state_dict(payload[name], strict=strict)
        if not strict and (miss or unexp):
            rep = payload.setdefault("load_report", {})
            rep[f"{name}_missing"] = list(miss)
            rep[f"{name}_unexpected"] = list(unexp)
    if restore_rng and payload.get("rng"):
        _set_rng_state(payload["rng"])
    return payload


def latest_checkpoint(directory: str | Path, *, pattern: str = "*.pt") -> Path | None:
    d = Path(directory)
    if not d.exists():
        return None
    cands = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime)
    for c in reversed(cands):
        if c.name != "last.pt.tmp":
            return c
    return None
