"""An empty split fold is a refusal, not a warning that scrolls past.

Found by the pre-launch smoke for run 4, which is the only check that builds the
real datasets. `--quick` gave `sleepedf_real` a two-participant roster;
`stable_hash_v2` hashed both outside train; `leakage_check` appended
``fold 'train' is empty`` to its warnings and left ``ok`` True, because an empty
fold is not a *leak*; `_audit_real_split` printed it and continued -- though its
own docstring says it "raises rather than warning" -- and forty lines later torch
raised

    num_samples should be a positive integer value, but got num_samples=0

from a `RandomSampler`, naming neither the source nor the fold. The same log had
already reported an empty TEST fold for `eegmmidb` one source earlier.

The crash is not the worst outcome; it is the best one. A source admitted with an
empty TRAIN fold contributes no gradient while every report still lists it as
admitted and ``leakage_audited`` -- run 2's defect wearing a different hat.

Two things are pinned here: the refusal fires on any empty fold, and the
``--quick`` roster is large enough that the split can populate three of them.
"""

from __future__ import annotations

import inspect

import pytest

from scwbd.foundation.train import FoundationTrainer


def test_the_quick_roster_can_populate_three_participant_disjoint_folds() -> None:
    """A roster below 3 cannot fill 3 folds; below ~10 it is luck.

    Pinned as a number with a reason rather than left to whichever value made a
    particular seed work: assignment is a hash of the participant id, so the
    margin has to survive a changed roster, a changed seed and a changed
    `test_fraction` without anyone re-deriving it.
    """
    n = FoundationTrainer.QUICK_MIN_SUBJECTS
    assert n >= 10, (
        f"QUICK_MIN_SUBJECTS is {n}. The split is participant-disjoint over three "
        "folds, so fewer than 3 CANNOT populate them and a single-digit roster only "
        "does so if the hash happens to cooperate. --quick gave sleep-edfx 2 and "
        "eegmmidb 6; both produced an empty fold."
    )


def test_every_participant_split_source_uses_the_quick_minimum() -> None:
    """No measured, participant-split corpus may carry its own smaller literal.

    `build_data` had three `max_subjects=... if not self.quick else N` sites with
    N of 6, 2 and 1. The 1 is the TMS epoch dataset, which is NOT
    participant-split -- it goes straight into one DataLoader with no
    `participant_split` and no `_audit_real_split` -- so it is legitimately
    exempt. The other two are split and must use the shared minimum.
    """
    src = inspect.getsource(FoundationTrainer.build_data)
    literals = [
        line.strip()
        for line in src.splitlines()
        if "max_subjects=None if not self.quick else" in line
    ]
    assert literals, "the quick roster limits have moved; this guard has stopped guarding"

    bad = [
        ln
        for ln in literals
        if "QUICK_MIN_SUBJECTS" not in ln and "else 1," not in ln
    ]
    assert not bad, (
        f"these quick rosters carry their own literal instead of QUICK_MIN_SUBJECTS: "
        f"{bad}. If a new source is genuinely not participant-split (like the TMS "
        "epochs, which bypass participant_split entirely), say so here; otherwise it "
        "will hit the empty-fold refusal the first time the hash does not cooperate."
    )


def test_the_audit_refuses_an_empty_fold_rather_than_warning() -> None:
    """`_audit_real_split` must RAISE on an empty fold, not print and continue.

    Asserted on the source because constructing a real `FoundationTrainer` builds
    a 26.3M-parameter model and reads several corpora off disk. The behaviour
    itself is exercised by the smoke in `scripts/launch_run4.sh` step 5, which is
    what found the defect.
    """
    src = inspect.getsource(FoundationTrainer._audit_real_split)
    assert "n_windows_per_fold" in src, (
        "the empty-fold check no longer reads `n_windows_per_fold`. `leakage_check` "
        "reports windows per fold under that key; `n_per_fold` does not exist and an "
        "attribute error here would be raised from inside a gate, which reads as the "
        "gate firing."
    )
    assert 'raise RuntimeError(' in src and "EMPTY fold" in src, (
        "an empty fold no longer raises. It was a warning once: leakage_check leaves "
        "`ok` True because an empty fold is not a leak, so nothing else stops it, and "
        "the run dies later in a RandomSampler that names neither source nor fold."
    )
    # It must cover all three folds, not just train. An empty val or test fold
    # makes every held-out number computed from it vacuous.
    assert '("train", "val", "test")' in src, (
        "the empty-fold refusal no longer covers all three folds. An empty train "
        "fold silently drops a source's gradient; an empty val/test fold makes any "
        "held-out number from it vacuous. Both are refusals."
    )


@pytest.mark.parametrize("fold", ["train", "val", "test"])
def test_the_refusal_message_names_what_to_fix(fold: str) -> None:
    """The message has to tell the next reader which lever to pull.

    The failure mode this replaced was a torch error with no source, no fold and
    no remedy in it. A refusal that is merely earlier is not an improvement.
    """
    src = inspect.getsource(FoundationTrainer._audit_real_split)
    assert "quick ROSTER is too small" in src, (
        "the refusal no longer points at the roster. Someone hitting this in --quick "
        "needs to know the roster is the lever, or the tempting fix is to relax the "
        "check -- which is how the empty fold became a warning in the first place."
    )
