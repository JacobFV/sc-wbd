"""A query that needs an unidentifiable parameter must defer, and must not always.

The refusal is only worth having if it discriminates: the *same* query, the
*same* code path, must return a number for the EEG patient and a
:class:`Defer` for the MRI-only one.  Both directions are asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.individualize import (
    Defer,
    ModalityAvailability,
    PopulationModel,
    Query,
    QueryAnswer,
    answer,
    coupling_gain_query,
    individualize,
    profile_identifiability,
    simulate_patient,
)
from scwbd.infer.linear_gaussian import PARAM_INDEX


def _predicted_response(theta: np.ndarray) -> float:
    """A coupling- and delay-dependent number: the sort a clinician acts on."""
    a21 = theta[PARAM_INDEX["a21"]]
    a32 = theta[PARAM_INDEX["a32"]]
    tau = np.exp(theta[PARAM_INDEX["tau"]])
    return float(a21 * a32 * tau)


@pytest.fixture(scope="module")
def mri_result(small_cfg):
    pop = PopulationModel.reference()
    av = ModalityAvailability.from_modalities("P-MRI", ["structural_mri"])
    return individualize(
        pop, av, None, cfg=small_cfg, counterfactual_modalities=("eeg", "fmri")
    )


@pytest.fixture(scope="module")
def eeg_result(small_cfg):
    pop = PopulationModel.reference()
    av = ModalityAvailability.from_modalities("P-EEG", ["structural_mri", "eeg"])
    prof = profile_identifiability(av, cfg=small_cfg)
    data = simulate_patient(av, cfg=small_cfg, seed=5)
    return individualize(pop, av, data, profile=prof, cfg=small_cfg, n_newton=2)


def test_mri_only_patient_defers_on_a_coupling_query(mri_result):
    out = answer(mri_result, coupling_gain_query(), _predicted_response)
    assert isinstance(out, Defer)
    assert "coupling" in out.reason
    assert "POPULATION value" in out.reason
    assert out.detail["lambda_min[coupling]"] == 0.0


def test_the_same_query_answers_for_the_eeg_patient(eeg_result):
    """The discriminating half: the guard must not always fire."""
    out = answer(eeg_result, coupling_gain_query(), _predicted_response)
    assert isinstance(out, QueryAnswer)
    assert np.isfinite(out.value)
    assert set(out.group_status) == {"coupling", "conduction_delay"}


def test_the_defer_names_a_measured_remedy(mri_result):
    out = answer(mri_result, coupling_gain_query(), _predicted_response)
    assert isinstance(out, Defer)
    assert "Measured remedy" in out.reason
    assert "adding eeg" in out.reason
    assert out.suggested_action == "additional_calibration_measurement"


def test_fmri_is_not_offered_as_a_remedy_for_coupling(mri_result):
    out = answer(mri_result, coupling_gain_query(), _predicted_response)
    assert isinstance(out, Defer)
    assert "adding fmri would make coupling" not in out.reason


def test_the_evaluator_is_never_called_when_a_dependency_is_missing(mri_result):
    """The guard is upstream of the arithmetic, not a filter on its output."""
    calls = []

    def spy(theta):
        calls.append(theta)
        return 1.0

    out = answer(mri_result, coupling_gain_query(), spy)
    assert isinstance(out, Defer)
    assert calls == []


def test_haemodynamic_query_defers_even_for_the_eeg_patient(eeg_result):
    """EEG individualises coupling but carries zero information about the HRF."""
    q = Query(
        name="bold_amplitude_prediction",
        depends_on=("hemodynamic",),
        description="predicted BOLD amplitude for this patient",
    )
    out = answer(eeg_result, q, lambda t: 1.0)
    assert isinstance(out, Defer)
    assert "hemodynamic" in out.reason


def test_a_population_level_query_is_answered_but_labelled(mri_result):
    q = Query(
        name="population_typical_response",
        depends_on=("coupling",),
        scope="population_level",
    )
    out = answer(mri_result, q, _predicted_response)
    assert isinstance(out, QueryAnswer)
    assert "POPULATION-LEVEL" in out.notes[0]


def test_a_query_with_no_declared_dependencies_is_refused():
    with pytest.raises(ValueError, match="declares no parameter-group"):
        Query(name="mystery", depends_on=())


def test_a_query_naming_an_unknown_group_is_refused():
    with pytest.raises(KeyError):
        Query(name="q", depends_on=("not_a_group",))


def test_defer_is_falsy_enough_to_be_noticed(mri_result):
    """A caller that forgets to check the type still cannot get a float."""
    out = answer(mri_result, coupling_gain_query(), _predicted_response)
    with pytest.raises(TypeError):
        float(out)  # type: ignore[arg-type]
