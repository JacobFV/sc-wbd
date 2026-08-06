"""Every code path must be exercised on the arm that actually uses it.

Five of the six defects that blocked run 2 had one shape: a path declared for
both arms and exercised on only one.  The control arm has no mechanistic
families and no family layout, so it silently skips the branches that the
treatment arm depends on -- and the suite ran on CPU, so it skipped the device
branches too.

    set_mechanistic_theta never called (trainer, 3 sites)  -- control has none
    FamilyStateLayout.zero_pad multiplied by a CPU mask    -- tests on CPU
    index/gather/scatter returned CPU index tensors        -- tests on CPU
    residual_penalty read self.residual                    -- control arm only
    predict() never bound mechanistic theta                -- built against run 1

**A test that never runs the shipping configuration on the shipping device is
not testing the shipping code.**  These tests build the *family* arm and
exercise the paths the control arm cannot reach.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD

PILOT = "configs/run2/pilot-families.yaml"


@pytest.fixture(scope="module")
def anat():
    return load_anatomy()


@pytest.fixture(scope="module")
def family_model(anat):
    cfg = load_config(PILOT)
    assert cfg.model.family_state, "this suite is meaningless on the control arm"
    return SCWBD(cfg.model, anat)


def test_the_family_arm_declares_mechanistic_families(family_model):
    """If this is empty the rest of the suite proves nothing."""
    mech = set(getattr(family_model.family_local, "mech", {}))
    assert mech, "no mechanistic families -- these tests would pass vacuously"


def test_residual_penalty_reads_the_arm_that_exists(family_model):
    """It read ``self.residual``, which is ``None`` under ``family_state``.

    The crash was the lucky outcome.  Had it resolved to something benign the
    regulariser that stops the learned residual replacing the mechanistic term
    would have been absent from exactly the arm whose mechanistic backends it
    exists to protect.
    """
    assert family_model.family_residual is not None
    assert family_model.residual is None
    pen = family_model.residual_penalty()
    assert torch.isfinite(pen) and float(pen) > 0.0


def test_rollout_without_binding_theta_refuses(family_model, anat):
    """The guard must fire, or a run completes with the conditioning dropped."""
    from scwbd.foundation.families import SpanViolation

    m = family_model
    m._family_packs = {}  # simulate "nobody called set_mechanistic_theta"
    with pytest.raises(SpanViolation, match="ParamPack"):
        m.rollout(
            y_context=torch.randn(2, 8, anat.n_regions),
            theta=torch.rand(2, 6),
            n_steps=2,
            enforce_r05=False,
        )


def test_binding_covers_every_mechanistic_family(family_model, anat):
    """One call must fan out; a partial bind is the silent-default failure."""
    th = torch.rand(2, 6)
    family_model.set_mechanistic_theta(th, anat)
    assert set(family_model._family_packs) == set(family_model.family_local.mech)


def test_rollout_and_backward_on_the_family_arm(family_model, anat):
    """Forward *and* backward: the crash that stopped run 2 was in a rollout,
    so a build-only smoke test would have missed it."""
    m = family_model
    th = torch.rand(2, 6)
    m.set_mechanistic_theta(th, anat)
    roll = m.rollout(
        y_context=torch.randn(2, 8, anat.n_regions), theta=th, n_steps=4, enforce_r05=False
    )
    mu, lv = m.eeg(roll.state)
    loss = (mu.float() ** 2).mean() + lv.float().mean()
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device-branch test")
def test_family_layout_indices_follow_the_tensor_device(anat):
    """``FamilyStateLayout`` is a plain object, so ``model.cuda()`` does not move
    its index tensors.  Every consumer that indexes a CUDA tensor must ask."""
    cfg = load_config(PILOT)
    m = SCWBD(cfg.model, anat).cuda()
    th = torch.rand(2, 6, device="cuda")
    m.set_mechanistic_theta(th, anat)
    roll = m.rollout(
        y_context=torch.randn(2, 8, anat.n_regions, device="cuda"),
        theta=th,
        n_steps=4,
        enforce_r05=False,
    )
    assert roll.state.is_cuda and torch.isfinite(roll.state).all()


def test_predict_binds_theta_for_a_family_checkpoint(tmp_path, anat):
    """``predict()`` was built and tested against run 1, whose control arm has
    no mechanistic families, so it never exercised the branch it supports."""
    from scwbd.runtime.predict import LoadedModel

    ckpt = "checkpoints/scwbd-002-pilot/last.pt"
    import os

    if not os.path.exists(ckpt):
        pytest.skip("no 002 checkpoint on disk yet")
    m = LoadedModel.from_checkpoint(ckpt)
    out = m.predict(torch.randn(2, 16, anat.n_regions), n_steps=4)
    assert torch.isfinite(out.activity).all()
    assert torch.isfinite(out.eeg).all()
