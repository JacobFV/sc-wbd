"""The second artifact of ``thesis_contract.tex`` sec. 0.3.

Its five success criteria are tested individually, and the cheap structural ones
(schema refusal, leakage audit, misspecification diagnostic) are tested without
running the full end-to-end recovery so that they stay in the fast suite.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from scwbd.infer.synthetic_slice import (
    INVALID_SCHEMAS,
    SchemaRefusal,
    leakage_audit,
    run_synthetic_slice,
    validate_schema,
)

DEVICE = os.environ.get("SCWBD_TEST_DEVICE", "cpu")


VALID_CARD = {
    "name": "eeg_head",
    "units": "V",
    "frame": "subject_surface_RAS",
    "clock": "eeg_amp",
    "role": "likelihood",
    "bias_status": "externally_bounded",
    "external_bound_source": "phantom_2026",
    "posterior_class": "bayesian",
    "generative_factor": True,
    "grouping_keys": ["participant_id", "family_id"],
}


def test_valid_schema_is_accepted():
    validate_schema(VALID_CARD)      # must not raise


@pytest.mark.parametrize("name", sorted(INVALID_SCHEMAS))
def test_every_invalid_schema_is_refused(name):
    """A type system earns credibility by rejecting programs (thesis sec. 0.1)."""
    with pytest.raises(SchemaRefusal) as e:
        validate_schema(INVALID_SCHEMAS[name])
    assert e.value.code == name.split("_")[0]
    assert e.value.remedy


def test_refusal_codes_carry_a_remedy_and_the_offending_object():
    with pytest.raises(SchemaRefusal) as e:
        validate_schema(INVALID_SCHEMAS["R09_pseudo_as_likelihood"])
    assert e.value.code == "R09"
    assert "generalized" in e.value.remedy or "pseudo" in e.value.remedy
    assert e.value.offending_object


def test_prior_specified_sensitivity_is_allowed_without_an_estimator():
    """R08 forbids a *point estimate* without an estimator, not an honest
    declaration that the term is a swept sensitivity."""
    card = dict(VALID_CARD, bias_status="prior_specified_sensitivity")
    card.pop("external_bound_source")
    validate_schema(card)


def test_leakage_audit_detects_parent_and_derivative_crossings():
    parents = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    raw = np.array([0, 0, 2, 2, 4, 4, 6, 6])       # two derivatives per raw scan
    good = leakage_audit(parents, raw, np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]))
    assert good["leakage_free"] and good["refusal"] is None

    split_parent = leakage_audit(parents, raw, np.array([0, 2, 4, 6]),
                                 np.array([1, 3, 5, 7]))
    assert not split_parent["leakage_free"]
    assert split_parent["refusal"]["code"] == "R10"
    assert split_parent["shared_parents"] == [0, 1, 2, 3]

    # derivatives of one raw scan on both sides is also leakage
    p2 = np.array([0, 0, 1, 1])
    r2 = np.array([0, 0, 2, 2])
    d = leakage_audit(p2, r2, np.array([0, 2]), np.array([1, 3]))
    assert not d["leakage_free"] and d["shared_raw_records"] == [0, 2]


@pytest.mark.slow
def test_end_to_end_slice_reports_every_criterion():
    """The full artifact.  Criteria are *reported*, not assumed to pass.

    Two criteria are asserted because they are structural rather than
    statistical -- refusal of an invalid schema, and absence of leakage across
    simulated parent subjects.  The remaining three (nominal coverage, held-out
    log loss, misspecification detection) are scientific results whose values
    belong in the report.
    """
    from scwbd.infer.linear_gaussian import SystemConfig

    cfg = SystemConfig(
        device=DEVICE, dtype="float64", epoch_seconds=3.0, n_epochs=4,
        n_delay_taps=22, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )
    rep = run_synthetic_slice(cfg=cfg, seed=101, n_parents=6, n_sessions=2,
                              n_derivatives=2, n_newton=3, verbose=False)
    expected = {
        "refusal_of_invalid_schema",
        "no_leakage_across_parents",
        "recovery_intervals_nominal_coverage",
        "heldout_log_loss_beats_baselines",
        "misspecified_module_detected",
        "subgroup_calibration",
    }
    assert set(rep.criteria) == expected
    assert rep.criteria["refusal_of_invalid_schema"]["pass"]
    assert rep.criteria["no_leakage_across_parents"]["pass"]
    assert rep.criteria["no_leakage_across_parents"][
        "ungrouped_control_detected_as_leaking"
    ]
    d = rep.to_dict()
    assert "criteria" in d and "all_pass" in d
