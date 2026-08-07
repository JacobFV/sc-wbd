"""The profile must reproduce the measured ordering: EEG ~= joint >> fMRI-only.

Orderings and structural zeros are asserted at the small configuration (fast);
the *absolute* committed numbers are asserted in
``test_matches_fisher_benchmark.py`` at the benchmark configuration (slow).
Asserting the ordering at one configuration and the values at another is
deliberate: an ordering that only held at one record length would be a property
of that record length, not of the modalities.
"""

from __future__ import annotations

import pytest

from scwbd.individualize import (
    IdentifiabilityThresholds,
    ModalityAvailability,
    profile_identifiability,
)


@pytest.fixture(scope="module")
def profiles(small_cfg):
    out = {}
    for label, mods in (
        ("mri_only", ["structural_mri"]),
        ("eeg_only", ["structural_mri", "eeg"]),
        ("fmri_only", ["structural_mri", "fmri"]),
        ("joint", ["structural_mri", "eeg", "fmri"]),
        ("nothing", []),
    ):
        av = ModalityAvailability.from_modalities(f"P-{label}", mods)
        out[label] = profile_identifiability(av, cfg=small_cfg)
    return out


THETA_GROUPS = ("coupling", "conduction_delay")


@pytest.mark.parametrize("group", THETA_GROUPS)
def test_eeg_is_within_a_hair_of_joint(profiles, group):
    e = profiles["eeg_only"][group].lambda_min
    j = profiles["joint"][group].lambda_min
    assert e > 0 and j > 0
    assert abs(j - e) / j < 1e-3, f"{group}: eeg={e} joint={j}"
    assert j >= e * (1 - 1e-9), "joint must never carry less than EEG alone"


@pytest.mark.parametrize("group", THETA_GROUPS)
def test_fmri_only_is_orders_of_magnitude_behind(profiles, group):
    f = profiles["fmri_only"][group].lambda_min
    e = profiles["eeg_only"][group].lambda_min
    assert e / max(f, 1e-300) > 1e6, f"{group}: eeg={e} fmri={f}"


@pytest.mark.parametrize("group", THETA_GROUPS)
def test_statuses_follow_the_ordering(profiles, group):
    """At *this* record length the claim is the ordering, not the absolute status.

    A 1.5 s x 2-epoch record is short enough that coupling lands in
    ``weakly_identifiable`` rather than ``identifiable``; that is a fact about
    the record length, and asserting the stronger label here would be asserting
    the configuration.  The absolute statuses at the committed benchmark
    configuration are asserted in ``test_matches_fisher_benchmark.py``.
    """
    assert profiles["eeg_only"][group].may_be_individualized
    assert profiles["joint"][group].may_be_individualized
    assert profiles["fmri_only"][group].status == "not_identifiable"
    assert profiles["mri_only"][group].status == "not_identifiable"


def test_mri_only_information_is_structurally_zero_not_merely_small(profiles):
    p = profiles["mri_only"]
    assert p.design == "prior"
    assert p.provenance["computed"] == "structural_zero"
    for name in ("coupling", "conduction_delay", "eeg_lead_field", "hemodynamic"):
        gi = p[name]
        assert gi.lambda_min == 0.0
        assert gi.status == "not_identifiable"
        # and the posterior would be the prior, exactly
        assert gi.posterior_sd_ratio == 1.0


def test_mri_only_still_individualises_anatomy(profiles):
    """Not-identifiable for dynamics is not the same as nothing personalised."""
    p = profiles["mri_only"]
    assert p["head_geometry"].status == "identifiable"
    assert p["head_geometry"].evidence_kind == "modality_presence"
    # ... and it carries NO lambda_min, so it cannot be confused with a
    # measured one
    assert p["head_geometry"].lambda_min is None


def test_a_patient_with_nothing_has_nothing(profiles):
    p = profiles["nothing"]
    assert p.identifiable == ()
    assert set(p.not_identifiable) == set(p.groups)
    assert p.fittable_parameters() == ()
    assert not p.fit_mask().any()


def test_eeg_gives_lead_field_but_not_haemodynamics(profiles):
    p = profiles["eeg_only"]
    assert p["eeg_lead_field"].status == "identifiable"
    assert p["hemodynamic"].lambda_min == 0.0
    assert p["hemodynamic"].status == "not_identifiable"


def test_fmri_gives_no_lead_field_information_at_all(profiles):
    assert profiles["fmri_only"]["eeg_lead_field"].lambda_min == 0.0


def test_fit_mask_covers_exactly_the_admitted_groups(profiles):
    from scwbd.infer.linear_gaussian import PARAM_INDEX

    p = profiles["eeg_only"]
    mask = p.fit_mask()
    for name, gi in p.groups.items():
        if gi.kind != "likelihood":
            continue
        for param in gi.parameters:
            assert bool(mask[PARAM_INDEX[param]]) == gi.may_be_individualized, (
                f"{name}.{param}"
            )


def test_group_partition_agrees_with_fisher_adapters():
    """Our groups must refine agent Fisher's blocks exactly, not approximately.

    ``scwbd.infer.adapters`` owns the stable ``eta = (theta, ell, rho)``
    partition.  If a parameter were reassigned upstream and we kept our own
    copy, every lambda_min here would silently be about a different subspace
    than the committed benchmark's.
    """
    from scwbd.individualize.groups import LIKELIHOOD_GROUPS
    from scwbd.infer.adapters import PARAMETER_BLOCKS, nuisance_index, theta_index

    by_name = {g.name: set(g.index) for g in LIKELIHOOD_GROUPS}
    assert by_name["coupling"] | by_name["conduction_delay"] == set(theta_index())
    assert by_name["eeg_lead_field"] | by_name["hemodynamic"] == set(nuisance_index())
    assert by_name["eeg_lead_field"] == set(PARAMETER_BLOCKS["ell"])
    assert by_name["hemodynamic"] == set(PARAMETER_BLOCKS["rho"])


# -------------------------------------------------------------- discrimination
def test_thresholds_discriminate():
    """Three inputs, three different readings.  Otherwise it is decoration."""
    th = IdentifiabilityThresholds()
    assert th.classify(10.0) == "identifiable"
    assert th.classify(1e-2) == "weakly_identifiable"
    assert th.classify(1e-9) == "not_identifiable"
    assert th.classify(float("nan")) == "not_identifiable"


def test_profile_needs_no_patient_data(small_cfg):
    """The clinical claim: this can be shown to a clinician BEFORE fitting."""
    av = ModalityAvailability.from_modalities("P", ["eeg"])
    p = profile_identifiability(av, cfg=small_cfg)
    assert p["coupling"].lambda_min is not None
    # nothing patient-derived went in
    assert p.availability_digest == av.digest()


def test_counterfactual_remedy_is_measured_not_asserted(small_cfg):
    av = ModalityAvailability.from_modalities("P", ["structural_mri"])
    p = profile_identifiability(
        av, cfg=small_cfg, counterfactual_modalities=("eeg", "fmri")
    )
    rem = p.remedies("coupling")
    assert [r["add_modality"] for r in rem] == ["eeg"], rem
    assert rem[0]["lambda_min"] > 0
    # the status it would REACH is carried through, never upgraded
    assert rem[0]["status"] == p.counterfactuals["eeg"]["coupling"]["status"]
    # fMRI is measured NOT to be a remedy, which is the discriminating half
    assert p.counterfactuals["fmri"]["coupling"]["status"] == "not_identifiable"
