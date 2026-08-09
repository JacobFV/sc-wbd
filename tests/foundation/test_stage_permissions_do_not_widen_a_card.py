"""The intersection of a card pattern and a stage glob must be inside BOTH.

`FoundationTrainer.stage_sources` says "restrict only" in its own docstring, and
for one case it did the opposite. When the two patterns were incomparable --
neither `fnmatch`es the other -- it took the STAGE's:

    card  eeg_montages.ds000117_real.*      one source's own operator
    stage eeg_montages.*.log_gain           every montage's gain
    old   eeg_montages.*.log_gain           BROADER THAN THE CARD

Every run-3 stage that calibrates the montage nuisance produced that, for three
of the four montages. Nothing leaked: `autograd.grad(..., allow_unused=True)`
returns `None` for a head the source's loss does not touch, so no gradient ever
arrived at another montage's gain. That is what makes it worth a test -- a
latent widening produces no symptom, and the permission audit in every
checkpoint reported the wide pattern as the effective one.

The property asserted is the general one, not the example: for every source and
every stage of every run config, no parameter name reachable under the effective
permission may be unreachable under the card's own.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import yaml

from scwbd.foundation.train import glob_intersection

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "configs/curriculum/source_cards"
CONFIGS = ("configs/run3/scwbd-003.yaml", "configs/run4/scwbd-004.yaml")

_ARCHITECTURES = (
    "checkpoints/scwbd-003/last.pt",
    "checkpoints/scwbd-003-smoke/last.pt",
    "checkpoints/scwbd-002-pilot/stage_T1_measured_founding.pt",
    # The only checkpoint on disk carrying an `individualizer` payload.
    "checkpoints/ci-smoke/last.pt",
)


def test_the_intersection_is_inside_both_patterns() -> None:
    """Unit cases, including the one that was wrong."""
    assert glob_intersection("eeg_montages.ds000117_real.*", "eeg_montages.*.log_gain") == (
        "eeg_montages.ds000117_real.log_gain"
    )
    assert glob_intersection("eeg_montages.sleepedf_real.*", "eeg_montages.*.nuisance*") == (
        "eeg_montages.sleepedf_real.nuisance*"
    )
    # disjoint segment -> the two patterns share no name at all
    assert glob_intersection("eeg_montages.a.*", "eeg_montages.b.log_gain") == ""
    assert glob_intersection("family_local.*", "coupling.*") == ""
    # comparable pairs still collapse to the narrower one
    assert glob_intersection("eeg.*", "eeg.log_gain") == "eeg.log_gain"
    assert glob_intersection("individualizer.*", "individualizer.*") == "individualizer.*"
    # unresolvable -> None, so the caller records rather than guesses
    assert glob_intersection("a.b", "a.b.c") is None


def _keys() -> list[str]:
    import torch

    out: list[str] = []
    for rel in _ARCHITECTURES:
        f = REPO / rel
        if not f.is_file():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        out += list((ck.get("model") or {}).keys())
        for c in ("posterior", "individualizer", "tms_drive"):
            sub = ck.get(c)
            if isinstance(sub, dict):
                out += [f"{c}.{k}" for k in sub]
    return sorted(set(out))


def _cards() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(CARDS.glob("*.yaml")):
        card = yaml.safe_load(f.read_text()) or {}
        if not card.get("enabled", True):
            continue
        out[f.stem] = [str(p).split("#")[0].strip() for p in (card.get("gradient_permission") or [])]
    return out


def _effective(card_pats: list[str], allow: list[str]) -> list[str]:
    """The narrowing `stage_sources` performs, in one place so it is testable."""
    if allow == ["*"]:
        return list(card_pats)
    out: list[str] = []
    for p in card_pats:
        if p == "*":
            continue
        for a in allow:
            if fnmatch.fnmatch(a, p):
                out.append(a)
            elif fnmatch.fnmatch(p, a):
                out.append(p)
            else:
                inter = glob_intersection(p, a)
                if inter:
                    out.append(inter)
    return list(dict.fromkeys(out))


@pytest.mark.parametrize("rel", CONFIGS)
def test_no_stage_widens_a_card(rel: str) -> None:
    path = REPO / rel
    if not path.is_file():
        pytest.skip(f"{rel} absent")
    keys = _keys()
    if not keys:
        pytest.skip("no architecture checkpoint on disk to match globs against")

    cards = _cards()
    cfg = yaml.safe_load(path.read_text())
    if "train" not in cfg:  # an inheriting config; resolve it properly
        from scwbd.foundation.config import load_config

        cfg = load_config(path).as_dict()

    widened: dict[str, list[str]] = {}
    for stage in cfg["train"]["stages"]:
        tp = ((stage.get("extra") or {}).get("curriculum") or {}).get("tier_permissions") or {}
        allow = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        if not allow:
            continue
        for sid, pats in cards.items():
            card_reach = {k for k in keys if any(fnmatch.fnmatch(k, p) for p in pats)}
            eff = _effective(pats, allow)
            eff_reach = {k for k in keys if any(fnmatch.fnmatch(k, p) for p in eff)}
            extra = sorted(eff_reach - card_reach)
            if extra:
                widened[f"{stage['name']}/{sid}"] = extra[:8]

    assert widened == {}, (
        f"the effective permission reaches parameters the card does not: {widened}. "
        "`stage_sources` restricts only -- an effective permission broader than "
        "the card's is a grant the card withheld, and the checkpoint's permission "
        "audit records the wide pattern as though the card had written it."
    )


@pytest.mark.parametrize("rel", CONFIGS)
def test_the_trainer_records_no_unresolvable_pair(rel: str) -> None:
    """`glob_intersection` returning None falls back to the stage's pattern.

    That fallback may widen, so it must not fire on any config this repo ships.
    Asserted on the real `stage_sources`, not the copy above.
    """
    path = REPO / rel
    if not path.is_file():
        pytest.skip(f"{rel} absent")
    from types import SimpleNamespace

    from scwbd.foundation.config import load_config
    from scwbd.foundation.curriculum_admission import stage_admission
    from scwbd.foundation.mixture import SourceSpec
    from scwbd.foundation.train import FoundationTrainer

    cfg = load_config(path)
    specs = {
        sid: SourceSpec(id=sid, gradient_permission=tuple(p))
        for sid, p in _cards().items()
    }
    tr = SimpleNamespace(sources=specs, cfg=cfg)
    for stage in cfg.train.stages:
        adm = stage_admission(stage, cards_dir=cfg.mixture_cards, strict=False)
        FoundationTrainer.stage_sources(tr, stage, adm)
        assert tr.unresolved_permission_pairs == [], (
            f"stage {stage.name}: {tr.unresolved_permission_pairs}. The "
            "intersection of these card/stage patterns could not be computed, so "
            "the effective permission fell back to the stage's -- which may grant "
            "what the card withheld. Rewrite one of the two patterns."
        )
