"""Negative control: a patient whose "data" are noise must not look individualized.

This is the failure the identifiability profile alone cannot catch.  The profile
asks *"could data of this kind move this parameter?"* and answers from the design
-- so for a patient with an EEG file it says yes, whatever is actually in the
file.  A disconnected electrode, a corrupted export or a phantom scan filed
under a person all produce a file of the right shape, and the optimiser will
happily find the parameters that best explain it.

So the record is checked against the forward model separately
(:func:`scwbd.individualize.fit.data_consistency`), and when it fails, every
group reverts to the population prior with the reason recorded.  Both directions
are asserted: the control fires on noise and does **not** fire on a real
simulated record, otherwise it would be refusing everything and telling us
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.individualize import (
    POPULATION_PRIOR,
    ModalityAvailability,
    PopulationModel,
    individualize,
    profile_identifiability,
    simulate_patient,
)


@pytest.fixture(scope="module")
def setup(small_cfg):
    pop = PopulationModel.reference()
    av = ModalityAvailability.from_modalities("P-EEG", ["structural_mri", "eeg"])
    prof = profile_identifiability(av, cfg=small_cfg)
    return pop, av, prof, small_cfg


@pytest.fixture(scope="module")
def genuine(setup):
    pop, av, prof, cfg = setup
    data = simulate_patient(av, cfg=cfg, seed=21)
    return individualize(pop, av, data, profile=prof, cfg=cfg, n_newton=2)


@pytest.fixture(scope="module")
def noise(setup):
    pop, av, prof, cfg = setup
    data = simulate_patient(av, cfg=cfg, seed=21, pure_noise=True)
    return individualize(pop, av, data, profile=prof, cfg=cfg, n_newton=2)


def test_a_genuine_record_passes_the_control(genuine):
    """The discriminating half: the control must not reject everything."""
    assert genuine.consistency is not None
    assert genuine.consistency.passed, genuine.consistency.reason
    assert genuine.individualized_groups, "a real record must individualise something"


def test_a_pure_noise_record_is_rejected(noise):
    assert noise.consistency is not None
    assert not noise.consistency.passed, noise.consistency.reason
    assert abs(noise.consistency.statistic - 1.0) > noise.consistency.rel_tolerance


def test_a_rejected_patient_gets_no_individualized_dynamical_parameters(noise):
    for g in ("coupling", "conduction_delay", "eeg_lead_field", "hemodynamic"):
        assert noise.status_of(g) == POPULATION_PRIOR, g
        assert "negative control" in noise.outcomes[g].reason.lower()


def test_a_rejected_patient_gets_the_prior_bit_for_bit(noise):
    want = noise.population.population_value(noise.group)
    assert np.array_equal(noise.theta_trait, want)
    noise.assert_population_prior_exact()


def test_the_rejection_is_recorded_not_silent(noise):
    assert any("NEGATIVE CONTROL FIRED" in n for n in noise.notes), noise.notes
    d = noise.to_dict()
    assert d["data_consistency"]["passed"] is False
    assert d["data_consistency"]["whitened_innovation_mean_square"] > 0


def test_the_statistic_discriminates(genuine, noise):
    """Two inputs, two readings.  A statistic that could not differ is decoration."""
    a = genuine.consistency.statistic
    b = noise.consistency.statistic
    assert abs(a - 1.0) < abs(b - 1.0)
    assert abs(a - b) > 0.1, (a, b)


def test_the_report_tells_the_reader_the_record_was_rejected(noise):
    from scwbd.individualize import patient_report

    md = patient_report(noise)
    assert "NEGATIVE CONTROL FIRED" in md
    assert "POPULATION PRIOR -- NOT THIS PATIENT" in md
