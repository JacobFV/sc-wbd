"""E1 -- both sides of the real-EEG comparison are the same random variable.

``scwbd.foundation.evaluate._scwbd_scores`` scores the **headline** NLL and MSE
on the raw target, and ``scwbd.foundation.baselines.Baseline.score`` scores every
baseline on the raw target.  The per-window rescale that once separated them is
retained, but only as the separately labelled ``nll_per_window_amplitude_
normalised`` column, which the return dict's ``units_note`` marks as *not*
comparable to the baselines.

**These tests used to assert the defect rather than the fix.**
``_scwbd_expression`` was a verbatim transcription of ``evaluate.py:69-75`` as it
stood: it divided the target by its own per-window standard deviation and folded
the Jacobian into the log-variance.  That WAS the evaluation's behaviour, and it
was repaired -- ``_scwbd_scores`` now carries the comment *"RAW data units,
matching baselines._gaussian_nll. Rescaling by the target's own per-window std
would compare densities of two DIFFERENT random variables."*  The transcription
was never updated, so the file kept reporting a fixed defect as live.

The algebra it established still governs the secondary column, with
``s = std(target)`` per window:

    NLL_scaled = 0.5 [ log 2pi + (lv - 2 log s) + (t/s - mu/s)^2 exp(-(lv - 2 log s)) ]
               = 0.5 [ log 2pi + lv + (t - mu)^2 exp(-lv) ] - log s
               = NLL_raw - log s

The squared-error term is *invariant* -- the ``1/s^2`` from the residual and the
``s^2`` from ``exp(-lv)`` cancel exactly -- so the whole effect is the additive
``- log s`` carried by the log-variance term alone.  MSE is not invariant:
``MSE_scaled = MSE_raw / s^2``.

A red in this file means the headline has gone back to scoring a rescaled target,
which is worth 0.5694 nats on the measured test fold -- 16x the entire 0.035-nat
spread that separates the non-trivial baselines from one another.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _scwbd_expression(tgt: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor):
    """Verbatim transcription of the HEADLINE score in ``_scwbd_scores``.

    ``m_k = mu.float()``; ``v_k = lv.float().clamp(-14, 14)``; the NLL is formed
    against ``y = tgt.float()`` with no rescale, and the MSE is
    ``((y - m_bar) ** 2).mean(dim=(1, 2))``.
    """
    y = tgt.float()
    m = mu.float()
    v = lv.float().clamp(-14, 14)
    nll = 0.5 * (math.log(2 * math.pi) + v + (y - m) ** 2 * torch.exp(-v))
    return nll.mean(dim=(1, 2)), ((y - m) ** 2).mean(dim=(1, 2))


def _scwbd_amplitude_normalised(tgt: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor):
    """The labelled SECONDARY column: ``nll_pw - log s``, as ``_scwbd_scores`` writes it."""
    nll, _ = _scwbd_expression(tgt, mu, lv)
    s = tgt.std(dim=(1, 2)).clamp_min(1e-8).float()
    return nll - torch.log(s)


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


def test_the_transcription_still_matches_the_code_it_transcribes():
    """The anti-drift guard, and the reason the rest of this file is not decorative.

    Every other test here compares ``_scwbd_expression`` against
    ``baselines._gaussian_nll``. ``_scwbd_expression`` is a *copy* of the
    headline score, so if ``_scwbd_scores`` changed and the copy did not, the
    comparison would keep passing on code the evaluation no longer runs. That is
    precisely how this file went stale the first time: it transcribed the
    rescaling version, the rescale was removed, and three tests went on
    reporting a repaired defect as live for want of anyone re-reading the source.

    So the copy is pinned to the original. A red here is not a defect in
    ``_scwbd_scores`` -- it means the headline score changed and every
    expectation in this file must be re-derived against it before being trusted.
    """
    import inspect

    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate._scwbd_scores)
    headline = "nll_el = 0.5 * (math.log(2 * math.pi) + v_k + (y - m_k) ** 2 * torch.exp(-v_k))"
    assert headline in src, (
        "the headline NLL in _scwbd_scores is no longer the line _scwbd_expression "
        "transcribes. Re-derive _scwbd_expression from the current source, then "
        "re-derive the tolerances in this file, before changing anything else."
    )
    assert "m_k = mu.float()" in src and "v_k = lv.float().clamp(-14, 14)" in src, (
        "the mean or log-variance fed to the headline NLL is no longer taken raw; "
        "_scwbd_expression's transcription is stale"
    )
    # The secondary column must stay a separate array, not be folded into the
    # headline -- that fold IS the units defect.
    assert "nll_pw - torch.log(s_)" in src, (
        "the amplitude-normalised column is no longer computed as `nll_pw - log s` "
        "beside the headline. If the -log s term has moved into nll_per_window, "
        "the units defect is back."
    )
    assert "RAW data units" in src, (
        "_scwbd_scores no longer declares its units. A number compared against a "
        "baseline must say which random variable it scores."
    )


def test_the_rescale_is_worth_minus_log_s():
    """The algebra itself: this documents the mechanism and must always hold.

    It is what makes the other tests in this file worth running -- the size of
    the effect a rescale would reintroduce is exactly ``mean log s``, and on the
    measured fold that is not small.
    """
    tgt, mu, lv = _fixture()
    raw, _ = _scwbd_expression(tgt, mu, lv)
    norm = _scwbd_amplitude_normalised(tgt, mu, lv)
    log_s = torch.log(tgt.std(dim=(1, 2)).clamp_min(1e-8))
    assert torch.allclose(norm, raw - log_s, atol=2e-4), (
        "the derivation NLL_scaled = NLL_raw - log s no longer describes the code; "
        "re-derive before trusting any other test in this file"
    )


def test_scwbd_and_baselines_are_scored_on_the_same_random_variable():
    """E1 verdict test: the headline and the baselines are one random variable.

    Tolerance is 0.01 nats: an order of magnitude below the ~0.035 nats that
    separates the non-trivial baselines from one another on the real test fold
    (ar16 2.0132, var4 2.0185, population_gaussian 2.0484), so a comparison
    perturbed by more than this cannot rank models.  A rescale of the target
    would put 0.5694 nats here.
    """
    tgt, mu, lv = _fixture()
    scwbd_nll, _ = _scwbd_expression(tgt, mu, lv)
    base_nll, _ = _baseline_expression(tgt, mu, lv)
    offset = float((base_nll - scwbd_nll).mean())
    assert abs(offset) < 0.01, (
        f"SC-WBD and the baselines are scored on different random variables: "
        f"SC-WBD's headline NLL is {offset:.4f} nats below the baselines' on "
        f"identical (target, mean, log-variance) inputs. The headline has gone "
        f"back to scoring a rescaled target; only "
        f"nll_per_window_amplitude_normalised may carry the -log s term, and it "
        f"is not comparable to a baseline."
    )


def test_scwbd_and_baselines_report_the_same_mse():
    """MSE is off by ``1/s^2`` under a rescale, and unlike the NLL it does not cancel."""
    tgt, mu, lv = _fixture()
    _, scwbd_mse = _scwbd_expression(tgt, mu, lv)
    _, base_mse = _baseline_expression(tgt, mu, lv)
    ratio = float((base_mse / scwbd_mse).mean())
    assert abs(ratio - 1.0) < 0.02, (
        f"SC-WBD's reported MSE is the baselines' MSE divided by s^2: the ratio "
        f"is {ratio:.3f}x on this fixture. MSE would be in squared data units for "
        f"the baselines and dimensionless for SC-WBD."
    )


def test_the_units_guard_is_not_vacuous_on_the_real_test_fold(cfg, real_eeg, real_split):
    """The fold's amplitude spread, regenerated from the measured corpus.

    The two tests above compare a raw score against a raw score, so they only
    have content while ``log s`` is materially non-zero on the data the
    evaluation actually reads.  If the corpus were ever amplitude-normalised
    upstream, ``s -> 1``, ``log s -> 0``, and both guards would pass no matter
    what ``_scwbd_scores`` did to the target.

    Measured: ``mean log s = 0.5694`` over 1,080 windows from 27 test
    participants -- 16x the 0.035-nat spread of the non-trivial baselines. That
    is the advantage the raw-units headline declines to take.
    """
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
    tgt = eeg[:, c:]
    mean_log_s = float(torch.log(tgt.std(dim=(1, 2)).clamp_min(1e-8)).mean())
    assert abs(mean_log_s) > 0.035, (
        f"mean log s is {mean_log_s:.4f} on {len(keep)} windows from "
        f"{len(set(subs))} test participants, which is below the 0.035-nat spread "
        f"of the non-trivial baselines. The measured corpus no longer has enough "
        f"amplitude spread for a rescale to change a ranking, so the two units "
        f"guards in this file would pass on a rescaled headline too. Re-derive "
        f"them against whatever now distinguishes the two random variables."
    )

    # And the distinction is live on this data, not only on the synthetic fixture:
    # a rescaled headline would sit exactly `mean log s` below every baseline.
    g = torch.Generator().manual_seed(0)
    mu = tgt + torch.randn(tgt.shape, generator=g) * 0.4 * tgt.std()
    lv = torch.zeros_like(tgt) + 2 * torch.log(tgt.std(dim=(1, 2), keepdim=True).clamp_min(1e-8))
    scwbd_nll, _ = _scwbd_expression(tgt, mu, lv)
    base_nll, _ = _baseline_expression(tgt, mu, lv)
    assert abs(float((base_nll - scwbd_nll).mean())) < 0.01, (
        "on measured windows the headline expression no longer agrees with "
        "baselines._gaussian_nll on identical inputs"
    )
