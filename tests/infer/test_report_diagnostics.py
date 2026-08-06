"""Unit tests for the report-side diagnostics (pure functions over result dicts)."""

from __future__ import annotations

import json

import numpy as np

from scwbd.infer.report import nuisance_identifiability, preregistration_delta


def _results(sd_post, sd_emp, fisher_diag, names=("a", "b", "c")):
    n = len(names)
    return {
        "regimes": {
            "reference": {
                "designs": {
                    "d0": {
                        "fisher_T4": {
                            "I_likelihood": np.diag(fisher_diag).tolist(),
                        },
                        "recovery": {
                            "parameter_names": list(names),
                            "posterior_sd_mean": list(sd_post),
                            "estimate_sd": list(sd_emp),
                        },
                    }
                }
            }
        }
    }


def test_ratio_recovers_the_information_that_generated_it():
    """sd_post/sd_emp = sqrt(1 + 1/I) must invert back to I."""
    truth = np.array([100.0, 1.0, 0.01])
    # posterior sd = 1/sqrt(I+1); empirical sd of the MAP = sqrt(I)/(I+1)
    sd_post = 1.0 / np.sqrt(truth + 1.0)
    sd_emp = np.sqrt(truth) / (truth + 1.0)
    out = nuisance_identifiability(_results(sd_post, sd_emp, truth))
    rows = out["reference"]["d0"]
    for nm, want in zip(("a", "b", "c"), truth):
        got = rows[nm]["implied_standardised_information"]
        assert np.isclose(got, want, rtol=1e-9), (nm, got, want)


def test_weakly_informed_parameter_is_flagged_but_absent_one_is_not():
    truth = np.array([50.0, 0.001, 0.0])
    sd_post = 1.0 / np.sqrt(truth + 1.0)
    sd_emp = np.sqrt(truth) / (truth + 1.0)
    sd_emp[-1] = 1e-12                      # no data => MAP pinned at prior mean
    rows = nuisance_identifiability(_results(sd_post, sd_emp, truth))["reference"]["d0"]
    assert not rows["a"]["prior_dominated"]
    # weakly but genuinely informed: flagged as a prior echo
    assert rows["b"]["prior_dominated"]
    assert not rows["b"]["structurally_absent_from_design"]
    assert rows["b"]["prior_fraction_of_posterior_precision"] > 0.99
    # no channel observes it at all: reported separately, not as a finding
    assert rows["c"]["structurally_absent_from_design"]
    assert not rows["c"]["prior_dominated"]


def test_overdispersion_is_flagged_as_the_opposite_failure():
    truth = np.array([10.0])
    sd_post = np.array([1.0 / np.sqrt(11.0)])
    sd_emp = sd_post * 2.0                  # estimates scatter wider than stated
    rows = nuisance_identifiability(
        _results(sd_post, sd_emp, truth, names=("a",))
    )["reference"]["d0"]
    assert rows["a"]["estimates_overdispersed"]


def test_preregistration_delta_reports_reductions_and_criteria_stability(tmp_path):
    rule = {"criteria": {"C1": "something"}}
    pre = {"status": "preregistered_before_run", "written_at": "T0",
           "decision_rule": rule, "n_recovery_replicates": 96,
           "n_monte_carlo_fisher_replicates": 256,
           "instrument": {"n_epochs": 32, "epoch_seconds": 6.0}}
    now = dict(pre, n_recovery_replicates=48, n_monte_carlo_fisher_replicates=64,
               instrument={"n_epochs": 16, "epoch_seconds": 6.0})
    (tmp_path / "manifest.preregistered.json").write_text(json.dumps(pre))
    (tmp_path / "manifest.json").write_text(json.dumps(now))
    md = preregistration_delta(tmp_path)
    assert "Decision criteria unchanged: yes" in md
    assert "| recovery replicates | 96 | 48 |" in md
    assert "| Monte-Carlo Fisher replicates | 256 | 64 |" in md
    assert "| epochs per record | 32 | 16 |" in md
    # epoch_seconds did not change, so it must not appear as a deviation
    assert "epoch length" not in md


def test_preregistration_delta_shouts_when_the_criteria_moved(tmp_path):
    pre = {"status": "preregistered_before_run", "written_at": "T0",
           "decision_rule": {"criteria": {"C1": "original"}}, "instrument": {}}
    now = dict(pre, decision_rule={"criteria": {"C1": "TUNED AFTER SEEING RESULTS"}})
    (tmp_path / "manifest.preregistered.json").write_text(json.dumps(pre))
    (tmp_path / "manifest.json").write_text(json.dumps(now))
    md = preregistration_delta(tmp_path)
    assert "NO — SEE BELOW" in md


def test_preregistration_delta_is_empty_without_a_preregistration(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "x"}))
    assert preregistration_delta(tmp_path) == ""
