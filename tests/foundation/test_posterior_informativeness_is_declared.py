"""A posterior that explains no variance may be published; it may not be published quietly.

ISSUE-012.  Run 3's amortized posterior scored ``posterior_r2`` in
``[-0.016, 0.001]`` on all six parameters and passed every calibration test in
the same block (``sbc_ks_pvalue_min`` 0.098, ``coverage_mae`` 0.021,
``posterior_z_sd`` 0.96-1.00).  That is not a coincidence to be explained away:
a posterior that ignores its conditioning and returns the prior is calibrated
**by construction**, so the calibration block reads as a pass at exactly the
moment the posterior has stopped inferring anything.

These tests do not require a future run to be informative.  They require it to
say so when it is not, in the three places a reader could otherwise be misled:
the report the evaluator writes, the artifact on disk, and the release claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scwbd.foundation.posterior import (
    R2_INFORMATIVE_FLOOR,
    ConstantTargetDimension,
    AmortizedPosterior,
    informativeness,
    posterior_report,
)
from scwbd.foundation.simulate import THETA_NAMES, ThetaPrior

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS = sorted((ROOT / "reports" / "training").glob("evaluation*.json"))


class _PriorPosterior:
    """The failure mode itself: samples the prior, ignoring the data entirely.

    Deliberately not a mock of the flow.  What is under test is whether a
    posterior that has learned nothing is *reported* as having learned nothing,
    and the cheapest posterior that has learned nothing is the prior.
    """

    def __init__(self, prior: ThetaPrior, seed: int = 0) -> None:
        self.prior = prior
        self._seed = seed

    def sample(self, y: torch.Tensor, n: int = 1) -> torch.Tensor:
        B = y.shape[0]
        self._seed += 1
        s = self.prior.sample(B * n, seed=self._seed)
        return s.reshape(B, n, len(THETA_NAMES))


class _InformativePosterior(_PriorPosterior):
    """Knows the truth to within a tenth of the prior width."""

    def __init__(self, prior: ThetaPrior, theta: torch.Tensor, seed: int = 0) -> None:
        super().__init__(prior, seed)
        self._theta = theta

    def sample(self, y: torch.Tensor, n: int = 1) -> torch.Tensor:
        # y[:, 0, 0] carries the row index, so the double can find its own truth
        idx = y[:, 0, 0].long()
        b = self.prior.bounds()
        width = (b[:, 1] - b[:, 0]).reshape(1, 1, -1)
        g = torch.Generator().manual_seed(self._seed)
        self._seed += 1
        eps = torch.randn(len(idx), n, len(THETA_NAMES), generator=g) * 0.1 * width
        return (self._theta[idx].unsqueeze(1) + eps).clamp(
            b[:, 0].reshape(1, 1, -1), b[:, 1].reshape(1, 1, -1)
        )


def _fixture(n: int = 96):
    prior = ThetaPrior()
    theta = prior.sample(n, seed=7)
    y = torch.zeros(n, 8, 4)
    y[:, 0, 0] = torch.arange(n, dtype=torch.float32)
    return prior, theta, y


# ----------------------------------------------------------------------
# 1. the report says it
# ----------------------------------------------------------------------
def test_a_posterior_that_returns_the_prior_is_reported_as_uninformative():
    prior, theta, y = _fixture()
    rep = posterior_report(
        _PriorPosterior(prior), y, theta, param_names=THETA_NAMES, n_samples=64
    )
    assert rep["posterior_informative"] is False
    assert sorted(rep["uninformative_parameters"]) == sorted(THETA_NAMES)
    assert "ISSUE-012" in rep["informativeness_note"]
    # and the calibration numbers in the very same report look fine, which is
    # the whole reason the verdict has to be written down separately
    assert rep["coverage_mae"] < 0.15
    assert max(rep["posterior_r2"]) < R2_INFORMATIVE_FLOOR
    # the signature that separates "uninformative" from "noisy": prior width
    assert min(rep["posterior_sd_over_prior_sd"]) > 0.85


def test_a_posterior_that_recovers_theta_is_not_flagged():
    prior, theta, y = _fixture()
    rep = posterior_report(
        _InformativePosterior(prior, theta), y, theta, param_names=THETA_NAMES, n_samples=64
    )
    assert rep["posterior_informative"] is True
    assert rep["uninformative_parameters"] == []
    assert "informativeness_note" not in rep
    assert min(rep["posterior_r2"]) > R2_INFORMATIVE_FLOOR
    assert max(rep["posterior_sd_over_prior_sd"]) < 0.85


def test_one_recovered_parameter_is_enough_to_be_informative_and_the_rest_are_named():
    inf = informativeness(["a", "b", "c"], [0.6, 0.0, -0.2])
    assert inf["posterior_informative"] is True
    assert inf["uninformative_parameters"] == ["b", "c"]
    assert "b, c" in inf["informativeness_note"]


# ----------------------------------------------------------------------
# 2. the artifact on disk says it
# ----------------------------------------------------------------------
@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda p: p.name)
def test_every_published_evaluation_declares_whether_its_posterior_infers_anything(path: Path):
    cal = (json.loads(path.read_text()).get("posterior_calibration") or {})
    if not cal.get("available"):
        pytest.skip(f"{path.name} carries no posterior calibration block")
    r2 = cal.get("posterior_r2")
    assert r2, f"{path.name}: a calibration block without posterior_r2 cannot be read at all"
    assert "posterior_informative" in cal, (
        f"{path.name} publishes posterior_r2={[round(v, 4) for v in r2]} and no verdict on whether "
        "the posterior infers anything. Calibration alone cannot supply that verdict. "
        "Regenerate with scwbd.foundation.posterior.posterior_report, or add the declaration."
    )
    expected = any(float(v) >= R2_INFORMATIVE_FLOOR for v in r2)
    assert bool(cal["posterior_informative"]) is expected, (
        f"{path.name}: declares posterior_informative={cal['posterior_informative']} against "
        f"posterior_r2={[round(v, 4) for v in r2]} and a floor of {R2_INFORMATIVE_FLOOR}"
    )
    if not expected:
        assert cal.get("uninformative_parameters"), f"{path.name}: says uninformative, names nobody"
        assert "ISSUE" in (cal.get("informativeness_note") or ""), (
            f"{path.name}: an uninformative posterior must point at the open issue that tracks it"
        )


def test_the_run3_finding_itself_has_not_been_quietly_edited_away():
    """The measurement ISSUE-012 rests on. If a rerun replaces it, this must be read again."""
    cal = json.loads((ROOT / "reports/training/evaluation_run3.json").read_text())[
        "posterior_calibration"
    ]
    assert cal["n_datasets"] == 512
    assert max(cal["posterior_r2"]) < 0.01 and min(cal["posterior_r2"]) > -0.05
    assert cal["sbc_ks_pvalue_min"] > 0.01 and cal["coverage_mae"] < 0.05
    assert min(cal["posterior_z_sd"]) > 0.9
    assert cal["posterior_informative"] is False


# ----------------------------------------------------------------------
# 3. the release claim says it
# ----------------------------------------------------------------------
def _claim(cal: dict, tmp_path: Path):
    from scwbd.foundation.release import build_manifest

    ev = tmp_path / f"ev{abs(hash(json.dumps(cal, sort_keys=True)))}.json"
    ev.write_text(json.dumps({"posterior_calibration": cal, "anatomy": {"is_biological": True}}))
    sm = tmp_path / "summary.json"
    sm.write_text("{}")
    m = build_manifest(checkpoint=tmp_path / "no-such.pt", evaluation=ev, summary=sm)
    return m.as_dict()


def test_an_uninformative_posterior_cannot_be_claimed_as_self_consistent(tmp_path):
    base = {
        "available": True,
        "param_names": list(THETA_NAMES),
        "sbc_ks_pvalue": [0.5] * 6,
        "coverage_mae": 0.02,
        "posterior_r2": [-0.01, 0.0, -0.006, -0.015, -0.003, -0.005],
        "posterior_z_sd": [1.0] * 6,
        "coverage": {"n_datasets": 512},
    }
    blob = json.dumps(_claim(base, tmp_path))
    assert '"amortized_posterior_self_consistency"' in blob
    i = blob.index("amortized_posterior_self_consistency")
    assert '"unsupported"' in blob[i : i + 2000], (
        "SBC and coverage pass here by construction; the claim must not be graded on them alone"
    )
    assert "amortized_posterior_uninformative" in blob

    good = dict(base, posterior_r2=[0.6, 0.4, 0.5, 0.3, 0.5, 0.4])
    blob = json.dumps(_claim(good, tmp_path))
    j = blob.index("amortized_posterior_self_consistency")
    assert '"partial"' in blob[j : j + 2000]
    assert "amortized_posterior_uninformative" not in blob


# ----------------------------------------------------------------------
# 4. the defect that fed it: a density may not be scored on a constant
# ----------------------------------------------------------------------
def test_the_npe_objective_refuses_a_constant_target_column():
    cfg = type(
        "C", (), dict(summary_channels=8, summary_layers=2, flow_layers=2, flow_hidden=16, n_pcs=4)
    )()
    prior = ThetaPrior()
    post = AmortizedPosterior(cfg, len(THETA_NAMES), prior=prior, nuisance_dim=2)
    y = torch.randn(6, 32, 5)
    theta = prior.sample(6, seed=3)
    # exactly what train.py passed for every batch of run 3
    theta_full = torch.cat([theta, torch.zeros(6, 2)], -1)
    with pytest.raises(ConstantTargetDimension) as e:
        post.loss(y, theta_full)
    assert "[6, 7]" in str(e.value) and "nuisance_dim" in str(e.value)
    # a target that actually varies is scored, not refused
    post0 = AmortizedPosterior(cfg, len(THETA_NAMES), prior=prior, nuisance_dim=0)
    assert torch.isfinite(post0.loss(y, theta))


def test_the_default_config_does_not_ask_for_nuisance_dimensions_nobody_fills():
    from scwbd.foundation.config import PosteriorConfig

    assert PosteriorConfig().nuisance_dim == 0, (
        "run 2's pilot measured this and set its own file to 0; the default stayed at 2 and run 3 "
        "inherited it. Raise it only alongside code that estimates real nuisance values."
    )
