"""The core contract: what was not identified is the prior, labelled, bit for bit.

The MRI-only patient is the case the project owner named.  Their coupling and
delay parameters must come back:

* labelled ``population_prior``;
* **exactly** equal to the population value -- ``==``, not ``allclose``, because
  a value that moved by 1e-16 is a value an optimiser touched, and the label
  would then be a claim about intent rather than about the number;
* with a recorded reason, not merely absent from the table.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.individualize import (
    INDIVIDUALIZED,
    POPULATION_PRIOR,
    ModalityAvailability,
    PopulationModel,
    individualize,
    patient_report,
    profile_identifiability,
    simulate_patient,
)
from scwbd.infer.linear_gaussian import PARAM_INDEX


def _mri_only(cfg):
    """MRI-only: the design is ``prior``, so neither Fisher nor a fit runs."""
    pop = PopulationModel.reference()
    av = ModalityAvailability.from_modalities(
        "P-MRI-ONLY", ["structural_mri", "dmri"]
    )
    return pop, av, individualize(pop, av, None, cfg=cfg)


@pytest.fixture
def mri_only(small_cfg):
    return _mri_only(small_cfg)


@pytest.fixture(scope="module")
def eeg_patient(small_cfg):
    pop = PopulationModel.reference()
    av = ModalityAvailability.from_modalities("P-EEG", ["structural_mri", "eeg"])
    prof = profile_identifiability(av, cfg=small_cfg)
    data = simulate_patient(av, cfg=small_cfg, seed=11)
    return pop, av, individualize(pop, av, data, profile=prof, cfg=small_cfg, n_newton=2)


# ------------------------------------------------------------------- MRI-only
def test_mri_only_coupling_is_labelled_population_prior(mri_only):
    _pop, _av, res = mri_only
    for g in ("coupling", "conduction_delay", "eeg_lead_field", "hemodynamic"):
        assert res.status_of(g) == POPULATION_PRIOR, g
        assert "NOT individualized" in res.outcomes[g].reason


def test_mri_only_coupling_is_bit_identical_to_the_prior(mri_only):
    pop, av, res = mri_only
    want = pop.population_value(av.group)
    for g in ("coupling", "conduction_delay", "eeg_lead_field", "hemodynamic"):
        for p in res.outcomes[g].parameters:
            i = PARAM_INDEX[p]
            assert res.theta_trait[i] == want[i], (p, res.theta_trait[i], want[i])
    # the whole vector, not just the groups we remembered to check
    assert np.array_equal(res.theta_trait, want)
    # and the runtime asserts it itself
    res.assert_population_prior_exact()


def test_mri_only_still_individualises_anatomy(mri_only):
    _pop, _av, res = mri_only
    assert res.status_of("head_geometry") == INDIVIDUALIZED
    assert res.status_of("structural_connectivity_prior") == INDIVIDUALIZED
    assert "coupling" in res.population_prior_groups


def test_no_optimiser_ran_for_an_mri_only_patient(mri_only):
    _pop, _av, res = mri_only
    assert not res.fit_mask.any()
    assert res.fit_diagnostics.get("per_session") == {}
    assert any("no optimiser was run" in n or "no patient records" in n
               for n in res.notes), res.notes


def test_absence_writes_something(mri_only):
    """Every declared group has a row; none is inferred from silence."""
    _pop, _av, res = mri_only
    res.assert_complete()
    d = res.to_dict()
    assert set(d["outcomes"]) == set(res.outcomes)
    for name, o in d["outcomes"].items():
        assert o["reason"].strip(), name
        assert o["uncertainty_ledger"], name


def test_ledger_says_the_variance_is_the_prior(mri_only):
    _pop, _av, res = mri_only
    led = res.outcomes["coupling"].ledger
    assert "no patient information" in led["variance_source"]
    assert all(v == 1.0 for v in led["prior_fraction"].values())


def test_assert_population_prior_exact_can_fail(mri_only):
    """The guard must be able to say no -- move a frozen value by 1e-12."""
    _pop, _av, res = mri_only
    res.assert_population_prior_exact()  # passes before
    res.decomposition.delta[PARAM_INDEX["a21"]] += 1e-12
    with pytest.raises(AssertionError) as e:
        res.assert_population_prior_exact()
    assert "wearing the wrong label" in str(e.value)


def test_a_nonzero_session_effect_on_a_frozen_group_is_caught(mri_only):
    _pop, _av, res = mri_only
    res.decomposition.zeta[0, PARAM_INDEX["a21"]] = 1e-12
    with pytest.raises(AssertionError, match="non-zero session effect"):
        res.assert_population_prior_exact()


def test_assert_complete_can_fail(mri_only):
    """Silence about a group must be caught, not tolerated."""
    _pop, _av, res = mri_only
    res.assert_complete()  # passes before
    res.outcomes = {k: v for k, v in res.outcomes.items() if k != "coupling"}
    with pytest.raises(AssertionError, match="silent about"):
        res.assert_complete()


# ------------------------------------------------------------------- EEG
def test_eeg_patient_individualises_the_theta_groups(eeg_patient):
    _pop, _av, res = eeg_patient
    assert "coupling" in res.individualized_groups
    assert "conduction_delay" in res.individualized_groups
    assert "eeg_lead_field" in res.individualized_groups


def test_eeg_patient_haemodynamics_stay_population(eeg_patient):
    """EEG carries exactly zero information about the BOLD head."""
    pop, av, res = eeg_patient
    assert res.status_of("hemodynamic") == POPULATION_PRIOR
    want = pop.population_value(av.group)
    for p in ("beta_hrf", "c_under", "gain_bold"):
        assert res.theta_trait[PARAM_INDEX[p]] == want[PARAM_INDEX[p]]


def test_the_fit_actually_moved_the_fitted_groups(eeg_patient):
    """Otherwise the bit-identity test above would pass for the wrong reason."""
    pop, av, res = eeg_patient
    want = pop.population_value(av.group)
    moved = [
        p for p in ("a21", "a32", "a13", "tau")
        if res.theta_trait[PARAM_INDEX[p]] != want[PARAM_INDEX[p]]
    ]
    assert len(moved) >= 3, moved


def test_single_session_split_is_reported_as_unidentified(eeg_patient):
    _pop, _av, res = eeg_patient
    assert res.decomposition.separable is False
    assert any("NOT identified" in n for n in res.notes)


def test_shrinkage_is_applied_once_in_the_fit_not_twice(eeg_patient):
    _pop, _av, res = eeg_patient
    assert res.decomposition.shrinkage_applied_in_fit is True
    d = res.fit_diagnostics["per_session"]["s0"]
    assert "individual" in d["prior_used"]


def test_profile_must_match_the_availability(small_cfg):
    pop = PopulationModel.reference()
    a = ModalityAvailability.from_modalities("P", ["eeg"])
    b = ModalityAvailability.from_modalities("P", ["fmri"])
    prof_a = profile_identifiability(a, cfg=small_cfg)
    with pytest.raises(ValueError, match="different"):
        individualize(pop, b, None, profile=prof_a, cfg=small_cfg)


# ------------------------------------------------------------------- report
@pytest.mark.parametrize("fixture", ["mri_only", "eeg_patient"])
def test_report_states_all_four_required_things(fixture, request):
    _pop, _av, res = request.getfixturevalue(fixture)
    md = patient_report(res)
    assert "## 1. Modalities present" in md
    assert "## 2. What was individualized" in md
    assert "## 3. What remained at the population value" in md
    assert "## 4. What CANNOT be individualized from this data, and why" in md
    assert "## 6. Uncertainty ledger" in md
    # every group appears somewhere
    for g in res.outcomes:
        assert f"`{g}`" in md, g


def test_mri_only_report_says_the_coupling_is_not_the_patients(mri_only):
    _pop, _av, res = mri_only
    md = patient_report(res)
    assert "POPULATION PRIOR -- NOT THIS PATIENT" in md
    assert "bit-identical" in md
    # the measured number is in the report, not an adjective
    assert "lambda_min" in md
