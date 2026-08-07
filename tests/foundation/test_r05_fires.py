"""R05 must be able to raise. Nothing in the suite proved it could.

``R05Violation`` is one of the trainer's live guarantees: the learned residual
``R_theta`` may not dominate the mechanistic terms it is supposed to correct. It
is raised from ``SCWBD.rollout`` when the EMA of
``rho = sqrt(E[residual] / E[mechanistic])`` exceeds ``residual_rho_max`` (0.35).

Before this file, ``grep -rn R05Violation tests/`` returned **zero** matches. A
guard with no test that makes it fire is the decorative-guard class this project
catalogues, sitting in the path that runs every rollout.

It also could not have fired for run 2 under any data, for a reason unrelated to
the guard: every residual net's output projection is zero-initialised so the
block starts as the identity, the module never received a gradient (see
``reports/RUN2.md`` §4), so the numerator was identically zero for 8,700 steps.
Run 3 grants ``family_residual.*``, so the quantity becomes live and this guard
starts doing work — which is why it needs a test now rather than later.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import ModelConfig
from scwbd.foundation.model import R05Violation, SCWBD


def _small(**kw) -> ModelConfig:
    base = dict(
        family_state=True, hidden=32, n_local_layers=1, region_embed=8,
        context_dim=16, encoder_channels=8, encoder_layers=1,
        family_allow_derived_partition=False,
    )
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


def _rollout(model, anat, *, steps=4, batch=2, enforce=True):
    ctx = torch.randn(batch, 8, anat.n_regions) * 0.1
    theta = torch.randn(batch, 6) * 0.2
    model.set_mechanistic_theta(theta, anat)
    return model.rollout(y_context=ctx, theta=theta, n_steps=steps, enforce_r05=enforce)


def test_r05_raises_when_the_residual_dominates(anat) -> None:
    """The discriminating test: inflate the residual until it dominates.

    The residual's output projections ship zero-initialised, so a fresh model has
    rho == 0 exactly. Scaling those layers up is the only way to construct the
    condition R05 exists to catch, and it is a legitimate construction: it is the
    state a run reaches if the residual learns to do the mechanistic term's job.
    """
    model = SCWBD(_small(), anat).eval()
    if model.family_residual is None:
        pytest.skip("this build has no learned residual")

    with torch.no_grad():
        touched = 0
        for name, p in model.family_residual.named_parameters():
            # `.4.` is the output layer of each per-family residual net -- the
            # zero-initialised one. Driving it large makes E[residual] dominate.
            if ".4." in name:
                p.copy_(torch.randn_like(p) * 50.0)
                touched += 1
        assert touched, "no output-projection parameters found to inflate"

    # `_rho_ema` moves 2% per rollout, so one pass cannot cross the threshold
    # from zero. Seed it at the value a sustained violation would produce; the
    # assertion is that the guard FIRES on that state, not that an EMA converges.
    model._rho_ema = 1.0
    with pytest.raises(R05Violation) as exc:
        _rollout(model, anat)
    assert exc.value.rho > exc.value.rho_max
    assert exc.value.code == "R05"


def test_r05_stays_silent_on_an_ordinary_model(anat) -> None:
    """The complement. A guard that always fires is as useless as one that cannot."""
    model = SCWBD(_small(), anat).eval()
    out = _rollout(model, anat)
    assert out is not None


def test_a_fresh_residual_contributes_exactly_zero(anat) -> None:
    """Documents WHY run 2's rho was 0.0000, and pins the zero-init convention.

    This is not a defect: zero-initialising a residual block's output so it
    starts as the identity is standard. It becomes one only when combined with a
    module that never receives a gradient, which is what happened to run 2 and is
    what ``tests/foundation/test_card_patterns_reach_the_model.py`` now prevents.
    """
    model = SCWBD(_small(), anat).eval()
    if model.family_residual is None:
        pytest.skip("this build has no learned residual")

    out_layers = [(n, p) for n, p in model.family_residual.named_parameters() if ".4." in n]
    assert out_layers, "no output projections found"
    assert all(bool((p == 0).all()) for _, p in out_layers), (
        "residual output projections are no longer zero-initialised. If that is "
        "deliberate the block no longer starts as the identity, and run 2's "
        "rho == 0.0000 in reports/RUN2.md §4 needs re-deriving."
    )

    _rollout(model, anat)
    assert model._rho_ema == 0.0, (
        f"a fresh residual produced rho_ema={model._rho_ema}; with zero output "
        "projections the residual energy must be exactly zero"
    )
