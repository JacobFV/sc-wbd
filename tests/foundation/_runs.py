"""The runs the launch gates are pointed at, discovered rather than written down.

Four tests gate a training launch: stage permissions reach the model, regional
tensors move, the Balloon parameters are frozen where the config says they are,
and card patterns reach the model. All four hard-coded ``configs/run3`` and run-3
checkpoints. They were GREEN -- against a run that was finished and published,
while ``configs/run4`` was what was about to be launched and nothing gated it.

That is the same shape as the ``BindingDriftError`` that sat undetected for a
day: a guard that passes because it is aimed at the wrong object. A gate that
cannot fail on the thing you are about to do is decoration.

Discovery is by glob over ``configs/run*/`` so run 5 is covered the day someone
writes it, rather than the day someone remembers to add it to a list here.

Two rules this module exists to enforce:

* **An empty discovery is a failure, not a pass.** An unmatched glob is an empty
  permission set, not an error -- this repository lost 88.8% of run 2's
  parameters to exactly that, and a parametrize over an empty list is a silent
  green. :func:`training_runs` raises instead.
* **A missing checkpoint skips only the assertions that need weights.** Run 4
  has no checkpoints until it launches, and its config-only assertions -- stage
  permissions, card patterns, the freeze declaration -- are all readable from
  YAML. Those must RUN. Only the weight-reading assertions may skip, and they
  skip naming the file they wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

#: Configs that are not a training run and must not be gated as one: smoke
#: configs, corpus builders, licence manifests, and the ablation arms, which are
#: control models fitted by a different entry point.
_NOT_A_RUN = {
    "smoke.yaml",
    "corpus_rebuild.yaml",
    "licence.yaml",
    "pilot-families.yaml",
    "pilot-pooled-param-matched.yaml",
    "scwbd-001-families.yaml",
    "scwbd-001-pooled-param-matched.yaml",
}


@dataclass(frozen=True)
class TrainingRun:
    """One run config and the checkpoint directory it writes to."""

    run_id: str  # "run3", "run4" -- the configs/ subdirectory
    config: Path
    ckpt_dir: Path

    def __str__(self) -> str:  # pytest id
        return self.run_id

    @property
    def checkpoints(self) -> list[Path]:
        """Stage checkpoints, then ``last.pt``. Empty before a run launches."""
        if not self.ckpt_dir.is_dir():
            return []
        return sorted(self.ckpt_dir.glob("stage_*.pt")) + sorted(self.ckpt_dir.glob("last.pt"))

    def require_checkpoint(self) -> Path:
        """The newest checkpoint, or skip naming what was missing.

        Skips -- never passes vacuously. A gate that silently succeeds because
        the artifact it grades is absent is the failure mode this module is
        about.
        """
        cks = self.checkpoints
        if not cks:
            pytest.skip(
                f"{self.run_id}: no checkpoint under {self.ckpt_dir.relative_to(REPO)} yet "
                f"(expected stage_*.pt or last.pt). The config-only assertions still ran; "
                f"this one reads weights."
            )
        return cks[-1]

    @property
    def yaml(self) -> dict:
        return _load_yaml(self.config)


@lru_cache(maxsize=None)
def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


@lru_cache(maxsize=None)
def training_runs() -> tuple[TrainingRun, ...]:
    """Every ``configs/run*/`` training config, newest last.

    Raises rather than returning empty: a parametrize over nothing is a green
    test suite that measured nothing.
    """
    found: list[TrainingRun] = []
    for d in sorted((REPO / "configs").glob("run*")):
        if not d.is_dir():
            continue
        for cfg in sorted(d.glob("*.yaml")):
            if cfg.name in _NOT_A_RUN:
                continue
            doc = _load_yaml(cfg)
            train = (doc or {}).get("train") or {}
            out = train.get("out_dir")
            if out is None:
                # A config with no `train.out_dir` of its own inherits one via
                # `base:`; resolve it rather than dropping the run silently.
                out = _inherited_out_dir(cfg, doc)
            if out is None:
                continue
            found.append(TrainingRun(run_id=d.name, config=cfg, ckpt_dir=REPO / out))
    if not found:
        raise RuntimeError(
            f"no training-run configs discovered under {REPO / 'configs'}/run*/. "
            "The launch gates parametrize over this list, so an empty result would "
            "turn four gates into four silent passes. Either the layout moved or the "
            "_NOT_A_RUN filter is now excluding everything."
        )
    return tuple(found)


def _inherited_out_dir(cfg: Path, doc: dict, _depth: int = 0) -> str | None:
    if _depth > 4 or not doc:
        return None
    base = doc.get("base")
    if not base:
        return None
    parent = (cfg.parent / base).resolve()
    if not parent.is_file():
        return None
    pdoc = _load_yaml(parent)
    return ((pdoc or {}).get("train") or {}).get("out_dir") or _inherited_out_dir(parent, pdoc, _depth + 1)


def raw_stages(run: TrainingRun) -> list[dict]:
    """The stage dicts as written, base-config inherited."""
    doc = run.yaml
    raw = (doc.get("train") or {}).get("stages")
    if raw is None:
        base = doc.get("base")
        if base:
            parent = (run.config.parent / base).resolve()
            raw = ((_load_yaml(parent) or {}).get("train") or {}).get("stages") or []
        else:
            raw = []
    return list(raw)


def stages(run: TrainingRun) -> list[tuple[str, dict]]:
    """``(stage name, tier_permissions)`` for each stage, base-config inherited."""
    raw = raw_stages(run)
    out = []
    for s in raw:
        tp = ((s.get("extra") or {}).get("curriculum") or {}).get("tier_permissions") or {}
        out.append((s["name"], tp))
    return out


@lru_cache(maxsize=None)
def parameter_names(run: TrainingRun) -> tuple[str, ...]:
    """Every parameter name in the architecture THIS run's config describes.

    Built from the config, not read from a checkpoint, because the run being
    gated is the one that has not happened yet. Run 4's ``msg_proj`` gating is
    the case in point: it is absent from the model its config builds and present
    in every checkpoint on disk, so a glob check against saved weights would
    grade run 4 against run 3's architecture and pass a stage permission that
    names nothing.

    This is also the check that actually catches binding drift. Run 4's
    ``BindingDriftError`` was found by building a run-4 trainer and not by any
    test -- ``test_compiler_binding.py`` asserts only on the pooled arm.
    Construction is ~2 s and needs no data, no checkpoint and no GPU.
    """
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.config import load_config
    from scwbd.foundation.model import SCWBD

    cfg = load_config(str(run.config))
    model = SCWBD(cfg.model, load_anatomy())
    names = [n for n, _ in model.named_parameters()]

    # The posterior, individualizer and TMS drive are siblings of the model on
    # the trainer, not submodules of it, and a stage glob may legitimately name
    # them. Enumerate them by prefix from the config rather than constructing
    # each: a stage granting `posterior.*` is granting a namespace that exists.
    for sub in ("posterior", "individualizer", "tms_drive"):
        names.append(f"{sub}.__namespace__")
    return tuple(sorted(names))


def known_parameter_names(run: TrainingRun) -> tuple[str, ...]:
    """Config-built names, unioned with any this run's checkpoints carry.

    A glob is dead only if it names nothing in EITHER, which keeps a config that
    legitimately anticipates a module built later from reading as broken.
    """
    import torch

    names = list(parameter_names(run))
    for f in run.checkpoints:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        names += list((ck.get("model") or {}).keys())
        for sub in ("posterior", "individualizer", "tms_drive"):
            blob = ck.get(sub)
            if isinstance(blob, dict):
                names += [f"{sub}.{k}" for k in blob]
    return tuple(sorted(set(names)))


def parametrize_runs(fn):
    """``@parametrize_runs`` -> the test runs once per discovered run, id'd by run_id."""
    return pytest.mark.parametrize("run", training_runs(), ids=lambda r: r.run_id)(fn)
