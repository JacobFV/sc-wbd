"""The person effect must be BUILT, INDEXED, COUNTED and CHECKPOINTED.

`reports/training/evaluation_run3.json` records, after a completed 13,400-step
run::

    "individualization": {"applied": false,
                          "reason": "no individualizer on the trainer (population model)"}

The class existed, the checkpoint slot existed, `STAGE_PERMISSIONS` named
`individualizer.*`, and `real_losses` applied the module if it was there. Four
separate things were false at once and each one alone is enough to make the
individualisation claim unmeasurable:

* **nothing constructed it.** `individualize` is a property of a stage's
  curriculum block and no run-3 stage declared one.
* **the participant index covered one corpus.** It read `real_dataset` --
  eegmmidb -- so every sleep-edfx participant fell onto row 0 and 78 people
  shared one person effect. Sleep-EDFx is the only corpus with a second session
  of the same person, so it is the only one the claim can be measured on.
* **the session effect was never selected.** `Individualizer.forward` was called
  with `participant=` alone, leaving `zeta` reachable by nothing but
  `prior_penalty`, which shrinks it toward zero.
* **its parameters were in neither list `moved_since_init` reads.** A fitted
  person effect and an absent one produced the same report -- which is the
  measurement half of run 2's defect, on the module carrying run 4's claim.

Each test below is one of those, plus the round-trip `tms_drive` needed for the
same reason.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scwbd.foundation.checkpoint import load_checkpoint, save_checkpoint
from scwbd.foundation.config import FoundationConfig, ModelConfig
from scwbd.foundation.individual import Individualizer
from scwbd.foundation.train import FoundationTrainer


# ======================================================================
# fakes -- these tests are about indexing and bookkeeping, not about data
# ======================================================================
class _FakeEEG:
    """The two attributes `_participant_ids` and `_session_ids` duck-type on."""

    def __init__(self, subjects: list[str], sessions_per: list[str]) -> None:
        self.subjects = subjects
        self.window_sessions = [f"{s}/{v}" for s in subjects for v in sessions_per]


def _fake_trainer(**datasets: _FakeEEG) -> SimpleNamespace:
    return SimpleNamespace(
        eeg_datasets=dict(datasets),
        real_dataset=None,
        device=torch.device("cpu"),
        _subjects_of=FoundationTrainer._subjects_of,
    )


# ======================================================================
# 1. the index spans every corpus
# ======================================================================
def test_the_participant_index_spans_every_eeg_corpus() -> None:
    """Not `real_dataset` alone. That is the row-0 collision, exactly."""
    tr = _fake_trainer(
        eegmmidb_real=_FakeEEG(["S001", "S002"], ["ses-01"]),
        sleepedf_real=_FakeEEG(["SC4001", "SC4002", "SC4003"], ["night1", "night2"]),
    )
    ids = FoundationTrainer._participant_ids(tr)
    assert ids == ["S001", "S002", "SC4001", "SC4002", "SC4003"], (
        "the individualizer indexes people by this list. Covering only eegmmidb "
        "puts every sleep-edfx participant on row 0, where they share one person "
        "effect -- and the between-participant spread of the applied shift is "
        "then 0 for arithmetic reasons that are indistinguishable in the report "
        "from the split's reasons."
    )


def test_every_sleep_participant_gets_their_own_row() -> None:
    """The measurement half: distinct people must map to distinct rows."""
    tr = _fake_trainer(
        eegmmidb_real=_FakeEEG(["S001"], ["ses-01"]),
        sleepedf_real=_FakeEEG(["SC4001", "SC4002", "SC4003"], ["night1", "night2"]),
    )
    tr._participant_ids = lambda: FoundationTrainer._participant_ids(tr)
    idx = FoundationTrainer.participant_index(tr, ["SC4001", "SC4002", "SC4003"])
    assert len(set(int(v) for v in idx.tolist())) == 3
    assert 0 not in [int(v) for v in idx.tolist()][1:], "row 0 is the fallback for unknown ids"


def test_a_subject_id_in_two_corpora_raises_rather_than_merging_two_people() -> None:
    """Bare ids, so a collision must be loud.

    `session_individualisation` and every split fingerprint quote subject ids
    bare. Two corpora sharing one id would give two different people one person
    effect and report the merge as an individualisation result.
    """
    tr = _fake_trainer(
        a_real=_FakeEEG(["sub-01"], ["ses-01"]),
        b_real=_FakeEEG(["sub-01"], ["ses-01"]),
    )
    with pytest.raises(RuntimeError, match="more than one corpus"):
        FoundationTrainer._participant_ids(tr)


# ======================================================================
# 2. the session effect is selected
# ======================================================================
def test_the_session_index_distinguishes_two_nights_of_one_person() -> None:
    tr = _fake_trainer(sleepedf_real=_FakeEEG(["SC4001", "SC4002"], ["night1", "night2"]))
    ids = FoundationTrainer._session_ids(tr)
    assert ids == ["SC4001/night1", "SC4001/night2", "SC4002/night1", "SC4002/night2"]

    tr._session_ids = lambda: ids
    idx = FoundationTrainer.session_index(tr, ["SC4001", "SC4001"], ["night1", "night2"])
    assert idx.tolist()[0] != idx.tolist()[1], (
        "both nights of one person landed on one zeta row, so the session effect "
        "cannot separate them and `consolidate`'s gate -- do not turn a night "
        "into a trait -- would be guarding an empty channel."
    )


def test_an_unseen_session_takes_the_population_row_and_is_recorded() -> None:
    """The held-out night must NOT get a fitted session effect."""
    tr = _fake_trainer(sleepedf_real=_FakeEEG(["SC4001"], ["night1"]))
    tr._session_ids = lambda: FoundationTrainer._session_ids(tr)
    idx = FoundationTrainer.session_index(tr, ["SC4001"], ["night2"])
    assert idx.tolist() == [0]
    assert "SC4001/night2" in tr.unknown_sessions


def test_a_batch_missing_its_session_key_raises() -> None:
    """Silently mapping every window to row 0 is a fitted effect read as none."""
    tr = _fake_trainer(sleepedf_real=_FakeEEG(["SC4001"], ["night1"]))
    tr._session_ids = lambda: FoundationTrainer._session_ids(tr)
    with pytest.raises(ValueError, match="index the same windows"):
        FoundationTrainer.session_index(tr, ["SC4001", "SC4001"], [])


# ======================================================================
# 3. moved_since_init accounts for it
# ======================================================================
@pytest.fixture(scope="module")
def trainer():
    cfg = FoundationConfig()
    cfg.model = ModelConfig(
        n_regions=414, hidden=64, region_embed=16, encoder_channels=16, context_dim=16
    )
    cfg.train.device = "cpu"
    cfg.train.resume = False
    return FoundationTrainer(cfg, device="cpu", resume=False)


def test_the_individualizer_is_counted_by_moved_since_init(trainer) -> None:
    """A person effect that trains and is never counted reads as one never built."""
    trainer.individualizer = Individualizer(4, n_groups=2, n_participants=6, n_sessions=12)
    trainer._fingerprint_late_module("individualizer", trainer.individualizer)

    before = trainer.moved_since_init()
    assert "individualizer" in before["by_module"], (
        "`individualizer` is in neither list `moved_since_init` reads, so its "
        "parameters are invisible to the one guard that measures whether a "
        "module received a gradient."
    )
    assert before["by_module"]["individualizer"]["moved"] == 0
    n_before = before["parameters_moved"]

    with torch.no_grad():
        trainer.individualizer.z_person[3] += 0.125
    after = trainer.moved_since_init()

    assert after["by_module"]["individualizer"]["moved"] == 1
    assert after["parameters_moved"] == n_before + trainer.individualizer.z_person.numel()
    assert after["unfingerprinted"] == [], (
        "a parameter with no recorded initialisation is compared against None, "
        "which is unequal -- so a module that has never taken a step reports as "
        "moved. That is the false pass measured on `tms_drive`."
    )
    trainer.individualizer = None


def test_the_two_module_lists_cannot_disagree(trainer) -> None:
    """Guards the guard: hashes and sizes must come from ONE list.

    They were two copies, and the second could omit a module the first hashed --
    which reports a parameter with a size of zero rather than reporting it at
    all, so `parameters_moved` would be right and `fraction_parameters_moved`
    would be wrong.
    """
    trainer.individualizer = Individualizer(4, n_groups=2, n_participants=3, n_sessions=3)
    trainer._fingerprint_late_module("individualizer", trainer.individualizer)
    with torch.no_grad():
        trainer.individualizer.mu += 1.0
    rep = trainer.moved_since_init()
    hashed = set(trainer._fingerprint_parameters())
    sized = {
        (name if prefix == "model" else f"{prefix}.{name}")
        for prefix, mod in trainer._fingerprinted_modules()
        for name, _ in mod.named_parameters()
    }
    assert hashed == sized
    assert rep["parameters_total"] > 0
    trainer.individualizer = None


# ======================================================================
# 4. it survives a checkpoint
# ======================================================================
def test_a_fitted_person_effect_round_trips(trainer) -> None:
    src = Individualizer(4, n_groups=2, n_participants=5, n_sessions=10)
    with torch.no_grad():
        src.z_person[2] += 0.4
        src.z_session[7] += -0.9
    want_p = src.z_person.detach().clone()
    want_s = src.z_session.detach().clone()

    d = Path(tempfile.mkdtemp())
    save_checkpoint(d / "i.pt", model=trainer.model, config=trainer.cfg, step=3,
                    stage="individual", individualizer=src)

    dst = Individualizer(4, n_groups=2, n_participants=5, n_sessions=10)
    assert float(dst.z_person.abs().sum()) == 0.0
    load_checkpoint(d / "i.pt", individualizer=dst, strict=False)
    assert torch.equal(dst.z_person, want_p)
    assert torch.equal(dst.z_session, want_s)

    payload = torch.load(d / "i.pt", map_location="cpu", weights_only=False)
    assert payload["individualizer"] is not None
    assert "z_person" in payload["individualizer"]


def test_a_missing_person_effect_is_reported_not_skipped(trainer) -> None:
    d = Path(tempfile.mkdtemp())
    save_checkpoint(d / "n.pt", model=trainer.model, config=trainer.cfg, step=0,
                    stage="s", individualizer=None)
    dst = Individualizer(4, n_groups=2, n_participants=2, n_sessions=2)
    payload = load_checkpoint(d / "n.pt", individualizer=dst, strict=False)
    assert (payload.get("load_report") or {}).get("individualizer_absent") is True


# ======================================================================
# 5. R07's shrinkage is attached exactly once, by a source allowed to move it
# ======================================================================
def _spec(perm: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(gradient_permission=perm)


def test_the_shrinkage_penalty_goes_to_a_source_that_may_move_the_effect() -> None:
    tr = SimpleNamespace(
        sources={
            "sleepedf_real": _spec(("eeg_montages.sleepedf_real.*", "individualizer.*")),
            "ds000117_real": _spec(("eeg_montages.ds000117_real.*",)),
        },
        global_step=0,
    )
    assert FoundationTrainer._owns_individual_penalty(tr, "ds000117_real") is False, (
        "a source whose permission cannot reach `individualizer.*` would add a "
        "penalty whose gradient is masked away -- a shrinkage term that is "
        "reported and not applied."
    )
    assert FoundationTrainer._owns_individual_penalty(tr, "sleepedf_real") is True


def test_the_shrinkage_penalty_is_added_once_per_step() -> None:
    """Adding it per source scales R07's shrinkage by the corpus count."""
    tr = SimpleNamespace(
        sources={
            "sleepedf_real": _spec(("individualizer.*",)),
            "eegmmidb_real": _spec(("individualizer.*",)),
        },
        global_step=11,
    )
    first = FoundationTrainer._owns_individual_penalty(tr, "sleepedf_real")
    second = FoundationTrainer._owns_individual_penalty(tr, "eegmmidb_real")
    assert (first, second) == (True, False)
    # the owner is stable within the step, and re-derived on the next one
    assert FoundationTrainer._owns_individual_penalty(tr, "sleepedf_real") is True
    tr.global_step = 12
    assert FoundationTrainer._owns_individual_penalty(tr, "eegmmidb_real") is True
    assert FoundationTrainer._owns_individual_penalty(tr, "sleepedf_real") is False


# ======================================================================
# 6. the corpus that can measure the claim is allowed to fit the effect
# ======================================================================
def test_sleepedf_grants_the_individualizer() -> None:
    """The permission has to be on the corpus with two sessions per person.

    Through run 3 it was on eegmmidb, whose 109 participants have one session
    each, and frozen on sleep-edfx, whose 75 two-night participants are the only
    people the claim can be measured on.
    """
    import yaml

    card = yaml.safe_load(
        (Path(__file__).resolve().parents[2]
         / "configs/curriculum/source_cards/sleepedf_real.yaml").read_text()
    )
    grants = [str(g).split("#")[0].strip() for g in card["gradient_permission"]]
    assert "individualizer.*" in grants
    assert "individualizer.*" not in [str(f).split("#")[0].strip() for f in card["frozen"]], (
        "granted and frozen at once. `permits` reads only `gradient_permission`, "
        "so the freeze would be documentation contradicting the enforcement."
    )
