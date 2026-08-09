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

from .config import FoundationConfig, designation
from .manifest import ClaimManifest, hash_file

from .util import env_fingerprint, git_sha

import logging

LOGGER = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "load_checkpoint", "latest_checkpoint", "CheckpointError", "drop_arm_dead_keys"]

#: Modules that belong to ONE arm and are simply absent on the other. A
#: checkpoint from before a module was gated carries its tensors; the current
#: model does not build them. Listed explicitly, so dropping a key is always a
#: named decision and never a side effect of relaxing `strict`.
#:
#: `msg_proj` -- the pooled arm's message projection. Both call sites prefer
#: `family_local.ports.message`, so on the family arm it is unreachable.
_ARM_DEAD_MODULES = frozenset({"msg_proj"})


def drop_arm_dead_keys(state: Mapping[str, Any], model: torch.nn.Module) -> tuple[dict, list[str]]:
    """Remove tensors for modules THIS arm does not build. Returns (state, dropped).

    Shared by `load_checkpoint` and by any caller loading a published checkpoint
    directly, so the allowance is defined once and every drop is a named module
    rather than a relaxed `strict` flag. A key absent from `_ARM_DEAD_MODULES` is
    left in place and still raises, which is the point: this must not become a
    general-purpose mismatch swallower.
    """
    own = set(model.state_dict())
    dropped = sorted(
        k for k in state if k not in own and k.split(".")[0] in _ARM_DEAD_MODULES
    )
    if not dropped:
        return dict(state), []
    keep = set(dropped)
    return {k: v for k, v in state.items() if k not in keep}, dropped



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
    tms_drive: torch.nn.Module | None = None,
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
        # Derived, never a literal.  This field was hardcoded to
        # "SC-WBD-001-beta", so every checkpoint the run-2 trainer wrote stamped
        # the run-1 name into its own payload -- the naming class again, in the
        # artifact itself rather than in a report about it.
        "model_id": designation(config),
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
        # The learned TMS drive. It is NOT part of `model`: it is built beside
        # the perturbation corpus and only when that corpus is on disk, so it
        # needs its own slot or its weights are trained and then dropped on
        # save. Its four tensors are the amplitude and the profile over motor
        # parcels -- the whole learned content of "what a pulse does".
        "tms_drive": tms_drive.state_dict() if tms_drive is not None else None,
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
    tms_drive: torch.nn.Module | None = None,
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
        sd = payload["model"]
        # ARM-DEAD MODULES. A checkpoint written before a module was gated on its
        # arm carries tensors the current model does not build, and `strict=True`
        # rejects the whole load over them.
        #
        # `msg_proj` is the case: it is the POOLED arm's message projection, and
        # a family-arm model no longer constructs it (72 unreachable parameters
        # that inflated every "fraction trained" denominator). Run 3's published
        # checkpoint has it; the current family-arm model does not.
        #
        # Dropped by NAME and REPORTED, never by relaxing `strict`. A blanket
        # `strict=False` would also swallow a genuine architecture mismatch,
        # which is the failure this whole file exists to make loud.
        sd, dead = drop_arm_dead_keys(sd, model)
        if dead:
            payload.setdefault("load_report", {})["arm_dead_dropped"] = dead
            LOGGER.info(
                "dropped %d arm-dead tensor(s) not built by this arm: %s", len(dead), dead
            )
        missing, unexpected = model.load_state_dict(sd, strict=strict)
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
    for name, mod in (
        ("posterior", posterior),
        ("individualizer", individualizer),
        ("tms_drive", tms_drive),
    ):
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
