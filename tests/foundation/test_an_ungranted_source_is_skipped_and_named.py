"""A source admitted with an EMPTY permission set must not run.

`stage_sources` intersects each card's `A_k` with the stage allowlist, and the
intersection can be empty. An empty permission set is not an error to `fnmatch`
and not an error to `GradientGate`, which returns `{}` from `grads` without
calling autograd at all. The source then:

* costs a full forward pass every step,
* produces a loss VALUE that enters `mixture_total` and renormalises every other
  source's weight,
* produces no gradient, and
* is added to `_contributed`, so the attachment report says it contributed.

Measured on run 4's `T6_individual`, which admits tier 1 whole while granting
only `individualizer.*` and the observation nuisance: `ds002336_real` ran a
250-step Balloon rollout on every one of 12 steps, logged `real_bold_nll` every
step, and could not move one parameter. `ds000117_behaviour` the same.

It is SKIPPED AND NAMED rather than silently dropped. Dropping it changes the
mixture renormalisation, which is a change to the objective, so it belongs in
the log and in `_absent_admitted` -- the same "exercised versus contributed"
distinction the attachment report exists for, one stage earlier.
"""

from __future__ import annotations

import inspect

from scwbd.foundation.config import load_config
from scwbd.foundation.curriculum_admission import stage_admission
from scwbd.foundation.mixture import SourceSpec
from scwbd.foundation.train import FoundationTrainer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RUN4 = REPO / "configs/run4/scwbd-004.yaml"
CARDS = REPO / "configs/curriculum/source_cards"


def _specs() -> dict[str, SourceSpec]:
    out: dict[str, SourceSpec] = {}
    for f in sorted(CARDS.glob("*.yaml")):
        card = yaml.safe_load(f.read_text()) or {}
        if not card.get("enabled", True):
            continue
        out[f.stem] = SourceSpec(
            id=f.stem,
            gradient_permission=tuple(
                str(p).split("#")[0].strip() for p in (card.get("gradient_permission") or [])
            ),
        )
    return out


@pytest.mark.skipif(not RUN4.is_file(), reason="run-4 config absent")
def test_the_individualisation_stage_does_have_ungranted_sources() -> None:
    """The premise. If this stops being true the guard below is unexercised.

    T6 admits tier 1 as a whole -- admission is per TIER, there is no per-source
    switch -- and grants only what an individualisation stage should. Two of the
    seven tier-1 cards overlap that in nothing.
    """
    cfg = load_config(RUN4)
    specs = _specs()
    tr = SimpleNamespace(sources=specs, cfg=cfg)
    stage = next(s for s in cfg.train.stages if s.name == "T6_individual")
    adm = stage_admission(stage, cards_dir=cfg.mixture_cards, strict=False)
    out = FoundationTrainer.stage_sources(tr, stage, adm)
    empty = sorted(sid for sid, s in out.items() if not s.gradient_permission)
    assert empty, (
        "no admitted source has an empty permission set in T6, so the skip below "
        "is never taken and this file is testing nothing. If the stage's grants "
        "changed, re-derive which sources it can and cannot train."
    )
    assert "ds002336_real" in empty


@pytest.mark.skipif(not RUN4.is_file(), reason="run-4 config absent")
def test_every_loss_call_site_is_gated_on_the_granted_set() -> None:
    """Each measured term must consult `ungranted`.

    Asserted against the source: the alternative is a stage-length run on the
    GPU, and a call site added later would otherwise reintroduce the defect
    silently for its own source only.
    """
    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "ungranted = self.note_ungranted(" in src, (
        "run_stage never computes the ungranted set"
    )
    # EXACT counts, not a floor. Six measured attachments reach a loss and each
    # needs its own gate; a floor passes when one of two identical loops loses
    # its check, which is how the first version of this assertion was written
    # and why its mutation could not be anchored.
    expected = {
        "REAL_LOSS_KEY not in ungranted": 1,   # the founding montage
        "and sid not in ungranted": 1,         # the ds002336 singleton
        "if sid in ungranted:": 2,             # extra EEG corpora, extra BOLD corpora
        '"ds000117_behaviour" not in ungranted': 1,
        '"ds004024_perturb" not in ungranted': 1,
    }
    got = {m: src.count(m) for m in expected}
    assert got == expected, (
        f"loss call sites gated on `ungranted`: {got}, expected {expected}. A "
        "measured term that skips this check runs a source with an empty "
        "permission set: a full forward pass, a value in the mixture total, and "
        "no gradient."
    )


def test_the_skip_is_recorded_and_returned() -> None:
    """`_absent_admitted` is what the checkpoint and the report read.

    A skip that leaves no trace is worse than the term it removes: the run would
    show six sources where the config admits seven and nothing would say which
    one went or why. Exercised directly rather than grepped -- the grep version
    survived deleting the line that does the recording.
    """
    tr = SimpleNamespace(_absent_admitted={})
    specs = {
        "granted": SourceSpec(id="granted", gradient_permission=("eeg.log_gain",)),
        "empty_a": SourceSpec(id="empty_a", gradient_permission=()),
        "empty_b": SourceSpec(id="empty_b", gradient_permission=()),
    }
    out = FoundationTrainer.note_ungranted(tr, "T6_individual", specs)
    assert out == ["empty_a", "empty_b"]
    assert tr._absent_admitted == {"T6_individual": ["empty_a", "empty_b"]}

    # A stage that already recorded an absent source must keep it: the two are
    # different reasons for the same fact and both belong in the artifact.
    tr2 = SimpleNamespace(_absent_admitted={"T6_individual": ["no_loader_here"]})
    FoundationTrainer.note_ungranted(tr2, "T6_individual", specs)
    assert tr2._absent_admitted == {
        "T6_individual": ["empty_a", "empty_b", "no_loader_here"]
    }


def test_a_fully_granted_stage_records_nothing() -> None:
    """No entry, not an empty list: absence and emptiness read differently."""
    tr = SimpleNamespace(_absent_admitted={})
    specs = {"a": SourceSpec(id="a", gradient_permission=("family_local.*",))}
    assert FoundationTrainer.note_ungranted(tr, "T1", specs) == []
    assert tr._absent_admitted == {}
