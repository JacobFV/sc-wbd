"""Amortized-posterior calibration smoke test + a CI-sized end-to-end training run.

The calibration test is deliberately weak as a *threshold* and strong as a
*contract*: it asserts the machinery produces well-formed SBC ranks and coverage
curves and that a deliberately over-confident posterior is caught.  Whether the
released model's posterior is calibrated is a claim gate, and a claim gate is
not graded by the module it grades.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scwbd.foundation.config import PosteriorConfig, load_config
from scwbd.foundation.posterior import (
    AmortizedPosterior,
    R09Violation,
    covariance_summary,
    expected_coverage,
    posterior_report,
    sbc_ranks,
    spectral_summary,
)
from scwbd.foundation.simulate import THETA_NAMES, ThetaPrior
from scwbd.foundation.util import set_determinism

REPO = Path(__file__).resolve().parents[2]


def _pcfg() -> PosteriorConfig:
    return PosteriorConfig(summary_channels=32, summary_layers=2, flow_layers=4, flow_hidden=64, n_pcs=6)


# ----------------------------------------------------------------------
def test_spectral_summary_finds_a_planted_oscillation():
    fs, T = 125.0, 256
    t = torch.arange(T) / fs
    y = torch.stack([torch.sin(2 * math.pi * 10.0 * t)] * 4, dim=-1).unsqueeze(0)
    s = spectral_summary(y, fs)
    # band order is delta, theta, alpha, ...; alpha (8-13 Hz) must dominate
    assert int(s[0, 0, :7].argmax()) == 2, s[0, 0, :7]


def test_covariance_summary_detects_global_coupling():
    T, C = 256, 8
    ind = torch.randn(1, T, C)
    common = torch.randn(1, T, 1).expand(1, T, C) + 0.1 * torch.randn(1, T, C)
    lo = covariance_summary(ind, 6)[0, 0]
    hi = covariance_summary(common, 6)[0, 0]
    assert float(hi) > float(lo), "a globally coupled signal must have a larger leading eigenvalue"


def test_flow_log_prob_is_a_normalised_density():
    """Integrating exp(log q) over a 1-D slice must give ~1: a flow that does not
    normalise is a pseudo-likelihood, and R09 forbids reporting one as a posterior."""
    set_determinism(0)
    from scwbd.foundation.posterior import ConditionalFlow

    f = ConditionalFlow(1, 3, n_layers=3, hidden=32)
    c = torch.randn(1, 3)
    xs = torch.linspace(-8, 8, 4001).reshape(-1, 1)
    lp = f.log_prob(xs, c.expand(xs.shape[0], -1))
    integral = torch.trapz(lp.exp(), xs.squeeze(-1))
    assert float(integral) == pytest.approx(1.0, abs=0.02), float(integral)


def test_posterior_learns_a_recoverable_parameter():
    """A parameter that is *in* the data must be recovered; this is the machinery
    check that a flat posterior would fail."""
    set_determinism(1)
    prior = ThetaPrior()
    n, T, C = 1024, 128, 6
    th = prior.sample(n, seed=2)
    amp = th[:, 2].reshape(n, 1, 1)  # ei_global drives the amplitude
    y = amp * torch.randn(n, T, C)
    post = AmortizedPosterior(_pcfg(), len(THETA_NAMES), prior=prior, fs=125.0)
    opt = torch.optim.Adam(post.parameters(), lr=3e-3)
    for i in range(220):
        idx = torch.randint(0, n, (128,))
        loss = post.loss(y[idx], th[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = post.sample(y[:256], 64)
    err_recoverable = float((s[:, :, 2].mean(1) - th[:256, 2]).abs().mean())
    spread = float(th[:, 2].std())
    assert err_recoverable < 0.8 * spread, (err_recoverable, spread)


def test_sbc_and_coverage_are_well_formed():
    set_determinism(2)
    prior = ThetaPrior()
    post = AmortizedPosterior(_pcfg(), len(THETA_NAMES), prior=prior, fs=125.0)
    y = torch.randn(48, 96, 5)
    th = prior.sample(48, seed=3)
    ranks = sbc_ranks(post, y, th, n_samples=32)
    assert ranks.shape == (48, len(THETA_NAMES))
    assert int(ranks.min()) >= 0 and int(ranks.max()) <= 32
    cov = expected_coverage(post, y, th, n_samples=32)
    assert len(cov["levels"]) == len(cov["coverage_mean"])
    assert all(0.0 <= c <= 1.0 for c in cov["coverage_mean"])
    assert 0.0 <= cov["coverage_mae"] <= 1.0
    rep = posterior_report(post, y, th, param_names=THETA_NAMES, n_samples=32)
    assert set(rep) >= {"sbc_ks_pvalue", "coverage_mae", "posterior_r2", "posterior_z_sd"}
    assert "simulator" in rep["note"].lower()


def test_coverage_catches_an_overconfident_posterior():
    """A posterior whose samples are far too tight must show coverage below nominal."""
    set_determinism(3)
    prior = ThetaPrior()

    class Overconfident(AmortizedPosterior):
        def sample(self, y, n=512):  # type: ignore[override]
            B = y.shape[0]
            mid = prior.denormalise(torch.zeros(1, len(THETA_NAMES)))
            return mid.expand(B, len(THETA_NAMES)).unsqueeze(1).repeat(1, n, 1) + 1e-4 * torch.randn(
                B, n, len(THETA_NAMES)
            )

    post = Overconfident(_pcfg(), len(THETA_NAMES), prior=prior, fs=125.0)
    y = torch.randn(64, 96, 5)
    th = prior.sample(64, seed=4)
    cov = expected_coverage(post, y, th, n_samples=32)
    assert cov["coverage_mean"][-1] < 0.3, "a degenerate posterior must not report nominal coverage"
    assert cov["coverage_mae"] > 0.3


def test_r09_blocks_marking_calibrated_without_evidence():
    post = AmortizedPosterior(_pcfg(), len(THETA_NAMES), prior=ThetaPrior(), fs=125.0)
    assert post.calibrated is False
    with pytest.raises(R09Violation):
        post.mark_calibrated({"train_loss": 0.1})
    post.mark_calibrated({"sbc_ks_pvalue": [0.4], "coverage_mae": 0.03})
    assert post.calibrated is True


# ----------------------------------------------------------------------
@pytest.mark.slow
def test_ci_sized_training_smoke_run(tmp_path):
    """The whole curriculum must actually run, checkpoint and reload."""
    corpus = Path("/data/scwbd/sim_corpus/index_fast.json")
    if not corpus.exists():
        pytest.skip("simulated corpus not generated on this machine")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the training smoke run")
    out = tmp_path / "ck"
    r = subprocess.run(
        [
            sys.executable, "-m", "scwbd.foundation.train",
            "--config", str(REPO / "configs" / "scwbd_ci_smoke.yaml"),
            "--quick", "--no-resume", "--out", str(out), "--max-wall", "600",
        ],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert (out / "last.pt").exists()
    assert (out / "config.yaml").exists()
    assert (out / "provenance.json").exists()
    payload = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert payload["format"] == "scwbd-foundation-checkpoint/1"
    assert payload["step"] > 0
    assert payload["metrics"]["completed_stages"]
