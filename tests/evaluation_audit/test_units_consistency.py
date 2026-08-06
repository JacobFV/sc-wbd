"""E1 -- both sides of the real-EEG comparison must be the same random variable.

``scwbd.foundation.evaluate._scwbd_scores`` scores SC-WBD on ``y = target / s``
where ``s`` is the target window's own standard deviation, with the matching
Jacobian folded into the log-variance.  ``scwbd.foundation.baselines.Baseline.
score`` scores every baseline on the raw target.  Those are different random
variables and their log-scores differ by an exact, model-independent constant.

The algebra, with ``s = std(target)`` per window:

    NLL_scaled = 0.5 [ log 2pi + (lv - 2 log s) + (t/s - mu/s)^2 exp(-(lv - 2 log s)) ]
               = 0.5 [ log 2pi + lv + (t - mu)^2 exp(-lv) ] - log s
               = NLL_raw - log s

The squared-error term is *invariant* -- the ``1/s^2`` from the residual and the
``s^2`` from ``exp(-lv)`` cancel exactly -- so the whole effect is the additive
``- log s`` carried by the log-variance term alone.  Consequently the rescale is
harmless in the training loss (``s`` does not depend on the parameters, so the
gradient is unchanged) and is a pure unearned advantage at evaluation time.

MSE is not invariant: ``MSE_scaled = MSE_raw / s^2``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _scwbd_expression(tgt: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor):
    """Verbatim transcription of evaluate.py:69-75 (SC-WBD's scoring)."""
    scale = tgt.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    y = (tgt / scale).float()
    m = (mu.float() / scale).float()
    v = (lv.float() - 2 * torch.log(scale)).clamp(-14, 14)
    nll = 0.5 * (math.log(2 * math.pi) + v + (y - m) ** 2 * torch.exp(-v))
    return nll.mean(dim=(1, 2)), ((y - m) ** 2).mean(dim=(1, 2))


def _baseline_expression(tgt: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor):
    """Verbatim transcription of baselines._gaussian_nll + Baseline.score."""
    from scwbd.foundation.baselines import _gaussian_nll

    nll = _gaussian_nll(mu, lv.clamp(-14, 14), tgt)
    return nll.mean(dim=(1, 2)), (mu - tgt).pow(2).mean(dim=(1, 2))


def _fixture(seed: int = 0, b: int = 64, t: int = 48, c: int = 64):
    g = torch.Generator().manual_seed(seed)
    # per-window amplitudes spanning the range measured on the real test fold
    amp = torch.exp(torch.randn(b, 1, 1, generator=g) * 0.35 + 0.59)
    tgt = torch.randn(b, t, c, generator=g) * amp
    mu = tgt + torch.randn(b, t, c, generator=g) * 0.4 * amp
    lv = torch.randn(b, t, c, generator=g) * 0.2 + 2 * torch.log(amp)
    return tgt, mu, lv


def test_scwbd_and_baseline_nll_differ_by_exactly_minus_log_s():
    """The algebra itself: this documents the mechanism and must always hold."""
    tgt, mu, lv = _fixture()
    scwbd_nll, _ = _scwbd_expression(tgt, mu, lv)
    base_nll, _ = _baseline_expression(tgt, mu, lv)
    log_s = torch.log(tgt.std(dim=(1, 2)).clamp_min(1e-8))
    assert torch.allclose(scwbd_nll, base_nll - log_s, atol=2e-4), (
        "the derivation NLL_scwbd = NLL_raw - log s no longer describes the code; "
        "re-derive before trusting any other test in this file"
    )


def test_scwbd_and_baselines_are_scored_on_the_same_random_variable():
    """E1 verdict test.  Fails while ``_scwbd_scores`` rescales and baselines do not.

    Tolerance is 0.01 nats: an order of magnitude below the ~0.035 nats that
    separates the non-trivial baselines from one another on the real test fold
    (ar16 2.0132, var4 2.0185, population_gaussian 2.0484), so a comparison
    perturbed by more than this cannot rank models.
    """
    tgt, mu, lv = _fixture()
    scwbd_nll, _ = _scwbd_expression(tgt, mu, lv)
    base_nll, _ = _baseline_expression(tgt, mu, lv)
    offset = float((base_nll - scwbd_nll).mean())
    assert abs(offset) < 0.01, (
        f"SC-WBD and the baselines are scored on different random variables: "
        f"SC-WBD's NLL is {offset:.4f} nats below the baselines' on identical "
        f"(target, mean, log-variance) inputs, entirely from the -log s term. "
        f"A comparison is only valid if both sides are the same quantity in the "
        f"same units."
    )


def test_scwbd_and_baselines_report_the_same_mse():
    """MSE is off by ``1/s^2``, and unlike the NLL term it does not cancel."""
    tgt, mu, lv = _fixture()
    _, scwbd_mse = _scwbd_expression(tgt, mu, lv)
    _, base_mse = _baseline_expression(tgt, mu, lv)
    ratio = float((base_mse / scwbd_mse).mean())
    assert abs(ratio - 1.0) < 0.02, (
        f"SC-WBD's reported MSE is the baselines' MSE divided by s^2: the ratio "
        f"is {ratio:.3f}x on this fixture. MSE is in squared data units for the "
        f"baselines and dimensionless for SC-WBD."
    )


def test_units_offset_measured_on_the_real_test_fold(cfg, real_eeg, real_split):
    """The magnitude, regenerated from the measured corpus rather than quoted.

    ``mean log s`` is the exact number of nats SC-WBD gains over every baseline
    before any modelling happens.
    """
    import torch.utils.data

    d = cfg.data
    c = d.context
    idx = np.asarray(real_split["test"])
    subs = np.asarray(real_eeg.window_subjects)[idx]
    # participant-representative sample: 40 evenly-spaced windows per participant
    keep: list[int] = []
    for s in sorted(set(subs)):
        loc = idx[subs == s]
        keep.extend(loc[np.unique(np.linspace(0, loc.size - 1, 40).round().astype(int))].tolist())
    eeg = torch.stack([torch.as_tensor(real_eeg[i]["eeg"]) for i in sorted(keep)])
    log_s = torch.log(eeg[:, c:].std(dim=(1, 2)).clamp_min(1e-8))
    mean_log_s = float(log_s.mean())
    assert abs(mean_log_s) < 0.01, (
        f"the units defect is worth {mean_log_s:.4f} nats per channel per sample on "
        f"{len(keep)} windows from {len(set(subs))} test participants. For scale, the "
        f"three non-trivial baselines span 0.035 nats on this same fold, so the "
        f"offset is ~17x the entire spread it would have to be compared against."
    )
