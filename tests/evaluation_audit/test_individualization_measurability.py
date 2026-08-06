"""E16 -- B5 split: a forward-pass mismatch, and a structural impossibility.

Turing asked whether applying the individualizer at evaluation is a no-op on a
participant-disjoint holdout. Measured on the real Stage-V individualizer
(`stage_V_individual.pt`, step 9300) rather than reasoned about, the answer
splits in two, and only one half is a bug.

**B5a -- real defect, negligible magnitude.** ``train.real_losses`` computes
``th = individualizer(participant=pid, base=th)``, which is ``mu + th +
delta[pid]``. ``evaluate._scwbd_scores`` computes ``th`` alone. ``mu`` moved
during training (‖mu‖ = 0.003909), so the evaluation runs a **different forward
pass from the one that was trained**. That is a correctness defect regardless of
size, and its size here is ~0.

**B5b -- not a defect, an impossibility.** ``z_person`` is nonzero for exactly
the 71 training participants and **exactly zero for all 27 test participants**,
so ``delta[row] = 0`` for every held-out person and the between-participant
spread of the applied shift is **identically 0.000e+00**. R10 (participant-
disjoint folds) and G5 (individualization helps a person) are in direct
conflict: a held-out person has no fitted person-effect. No patch to
``evaluate.py`` can change this. G5 needs a **within-participant temporal
split** -- fit ``z_person`` on a test participant's earlier windows, score later
ones -- which is a different claim ("adaptation given some of this person's
data") and must be labelled as such.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

CKPT = "/home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta/stage_V_individual.pt"


@pytest.fixture(scope="module")
def stage_v():
    from pathlib import Path

    p = Path(CKPT)
    if not p.exists():
        pytest.skip(f"no Stage-V checkpoint at {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if not payload.get("individualizer"):
        pytest.skip("checkpoint carries no individualizer state")
    return payload


def _rows(cfg, real_eeg, real_split):
    pids = sorted(set(map(str, real_eeg.subjects)))
    pidx = {s: i for i, s in enumerate(pids)}
    subs = np.asarray(real_eeg.window_subjects)
    tr = sorted(set(subs[real_split["train"]]))
    te = sorted(set(subs[real_split["test"]]))
    return pids, pidx, tr, te


def test_held_out_participants_have_a_fitted_person_effect(cfg, real_eeg, real_split, stage_v):
    """B5b. Fails structurally, and no change to evaluate.py can make it pass."""
    zp = stage_v["individualizer"]["z_person"]
    pids, pidx, tr, te = _rows(cfg, real_eeg, real_split)
    te_rows = torch.tensor([pidx[s] for s in te])
    tr_rows = torch.tensor([pidx[s] for s in tr])
    n_te_fitted = int((zp[te_rows].abs().sum(1) > 0).sum())
    n_tr_fitted = int((zp[tr_rows].abs().sum(1) > 0).sum())
    assert n_te_fitted > 0, (
        f"z_person is nonzero for {n_tr_fitted} of {len(tr)} TRAINING participants "
        f"and {n_te_fitted} of {len(te)} TEST participants. Refusal R10 makes the "
        f"folds participant-disjoint, so no held-out person has a fitted person "
        f"effect and G5 cannot be measured on this holdout by any patch to the "
        f"evaluation. It needs a within-participant temporal split, reported as a "
        f"different claim."
    )


def test_individualizer_differentiates_held_out_participants(cfg, real_eeg, real_split, stage_v):
    """The sharpest form: every test participant gets the SAME 'personalisation'."""
    from scwbd.foundation.individual import Individualizer
    from scwbd.foundation.simulate import THETA_NAMES

    pids, pidx, tr, te = _rows(cfg, real_eeg, real_split)
    P = len(THETA_NAMES)
    iv = Individualizer(P, n_groups=2, n_participants=max(len(pids), 1),
                        n_sessions=max(len(pids) * 4, 1))
    miss, unexp = iv.load_state_dict(stage_v["individualizer"], strict=False)
    assert not miss and not unexp
    te_rows = torch.tensor([pidx[s] for s in te])
    base = torch.zeros(len(te_rows), P)
    with torch.no_grad():
        shift = iv(participant=te_rows, base=base) - base
    spread = float(shift.std(0).norm())
    assert spread > 1e-6, (
        f"the theta shift applied to every one of the {len(te)} test participants "
        f"is identical (between-participant spread {spread:.3e}); only the "
        f"population term mu survives. A model that applies the same "
        f"personalisation to everyone is not individualised on this fold."
    )


def test_evaluation_forward_pass_matches_training(cfg, real_eeg, real_split, stage_v):
    """B5a. The mu term is real and the evaluation omits it.

    Magnitude is small -- ‖mu‖ = 0.0039 against per-dimension posterior sds of
    0.5-1.5, so the induced NLL change is orders of magnitude below the 0.0053-nat
    gap that decides a rank. It is worth fixing because train and eval must run
    the same forward pass, not because it moves the number.
    """
    import inspect

    from scwbd.foundation import evaluate, train

    mu = stage_v["individualizer"]["mu"]
    assert float(mu.norm()) > 0, "mu never moved; re-derive before reading this test"
    train_src = inspect.getsource(train.FoundationTrainer.real_losses)
    eval_src = inspect.getsource(evaluate._scwbd_scores)
    assert "individualizer" in train_src, "training no longer applies it; re-derive"
    assert "individualizer" in eval_src, (
        f"train.real_losses applies individualizer(participant=pid, base=th) and "
        f"_scwbd_scores does not, so the evaluation scores a model that was never "
        f"trained: it omits the learned population term mu (‖mu‖={float(mu.norm()):.6f}). "
        f"Small, and still the wrong forward pass."
    )


def test_no_held_out_participant_inherits_another_persons_effect(
    cfg, real_eeg, real_split, stage_v
):
    """Verified clean, and recorded: the plausible worse failure does not occur.

    ``participant_index`` maps unknown ids to row 0, which would hand a held-out
    person somebody else's fitted effect. It does not happen here: ``_participant_ids``
    is built from the whole dataset, so all 109 participants have their own row.
    """
    pids, pidx, tr, te = _rows(cfg, real_eeg, real_split)
    collisions = [s for s in te if pidx[s] == 0 and s != pids[0]]
    assert not collisions, f"test participants sharing row 0: {collisions}"
    assert len({pidx[s] for s in te}) == len(te)
    assert len(pids) >= len(tr) + len(te)
