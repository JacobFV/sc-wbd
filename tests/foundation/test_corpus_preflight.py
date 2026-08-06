"""Guards on the corpus preflight, verified by MUTATION rather than by assertion.

The standard this fleet settled on (🧠 Cajal's): break the thing the guard
watches and every guard test must fail.  A guard that stays green when its
subject is broken is decorative, and `reports/decorative_guards.md` has 24
entries of exactly that.

📐 Fisher's corollary applies with force here, because this module's whole
reason to exist is a failure that *was* unrepresentable to the checks around
it.  ``simulate.ParameterMappingError`` cannot see an inert theta dimension: it
compares mapping keys against ``backend.defaults``, and ``ei_gradient`` wrote a
perfectly valid key whose value happened to be multiplied by zero.  Every check
was green and every check was correct.  So the mutations below do not stub the
validator -- they break the *anatomy*, which is the thing the real defect broke.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.corpus_preflight import (
    DegenerateAnatomyPrior,
    InertThetaDimension,
    check_theta_parameter_sensitivity,
    check_theta_sampling_diversity,
    preflight,
)
from scwbd.foundation.simulate import THETA_NAMES, CorpusSpec

CORPUS_BACKENDS = ("wilson_cowan", "jansen_rit", "wong_wang", "stuart_landau", "linear_gaussian")


@pytest.fixture(scope="module")
def anat():
    a = load_anatomy()
    if not a.is_biological():
        pytest.skip("real anatomy prior unavailable (assets/ not present)")
    return a


# ---------------------------------------------------------------------------
# The positive readings.  These are what the preflight asserts on the real prior.
# ---------------------------------------------------------------------------


def test_every_theta_dim_except_velocity_moves_a_backend_parameter(anat):
    rep = check_theta_parameter_sensitivity(anat, backends=CORPUS_BACKENDS)
    # log_velocity is inert AT THIS LEVEL by construction -- it never passes
    # through _regional_theta.  Naming it here is the point: if it ever starts
    # moving a parameter, that is a change worth noticing, not a silent pass.
    assert rep.inert == ["log_velocity"], (
        f"expected exactly log_velocity inert at parameter level, got {rep.inert}"
    )
    for name in THETA_NAMES:
        if name == "log_velocity":
            continue
        assert len(rep.movers[name]) == len(CORPUS_BACKENDS), (
            f"{name} moves a parameter on only {rep.movers[name]}"
        )


def test_ei_gradient_specifically_is_not_inert(anat):
    """The regression this module was written for.

    ``anat.gradient`` was all zeros on the real prior, so
    ``ei = ei_global * ei_prior * (1 + ei_gradient * gradient)`` cancelled
    ``ei_gradient`` algebraically on every backend.
    """
    assert float(anat.gradient.std().item()) > 0.0
    rep = check_theta_parameter_sensitivity(anat, backends=CORPUS_BACKENDS)
    assert rep.movers["ei_gradient"], "ei_gradient moves nothing -- the P4 defect is back"
    for bk in CORPUS_BACKENDS:
        assert rep.per_dim["ei_gradient"][bk] > 0.0


# ---------------------------------------------------------------------------
# MUTATION.  Break the anatomy; the guards above must go red.
# ---------------------------------------------------------------------------


def test_mutation_zero_gradient_makes_ei_gradient_inert(anat):
    """Restore the actual P4 defect and require the check to catch it.

    This is the mutation that matters: it is not a stubbed validator, it is the
    exact state `master` was in before `eb2d88d`.
    """
    broken = dataclasses.replace(anat, gradient=torch.zeros_like(anat.gradient))
    rep = check_theta_parameter_sensitivity(broken, backends=CORPUS_BACKENDS)
    assert "ei_gradient" in rep.inert, (
        "a zeros gradient did NOT register as inert -- this check is decorative"
    )
    assert rep.movers["ei_gradient"] == []
    # and the other dimensions must be unaffected by the mutation, or the check
    # is firing on something other than what it claims
    for name in ("log_G", "ei_global", "log_sigma", "drive"):
        assert rep.movers[name], f"{name} broke too; the mutation is not targeted"


def test_mutation_constant_ei_prior_is_refused_as_degenerate_not_inert(anat):
    """The 1.2776 ignorance prior: theta still works, the science does not.

    This test originally asserted ``InertThetaDimension`` and failed, which was
    correct of it -- a constant non-zero ei_prior leaves every theta dimension
    identifiable.  The mutation forced the two failures apart.
    """
    broken = dataclasses.replace(anat, ei_prior=torch.full_like(anat.ei_prior, 1.2776))
    spec = CorpusSpec(backends=CORPUS_BACKENDS)
    with pytest.raises(DegenerateAnatomyPrior, match="no regional structure"):
        preflight(spec=spec, anat=broken, trajectory_level=False)
    # and it is NOT inert: ei_global still moves every backend
    rep = check_theta_parameter_sensitivity(broken, backends=CORPUS_BACKENDS)
    assert rep.movers["ei_global"], "a constant ei_prior must not make ei_global inert"


def test_mutation_zero_gradient_is_refused_by_preflight(anat):
    broken = dataclasses.replace(anat, gradient=torch.zeros_like(anat.gradient))
    spec = CorpusSpec(backends=CORPUS_BACKENDS)
    with pytest.raises(InertThetaDimension, match="gradient is identically zero"):
        preflight(spec=spec, anat=broken, trajectory_level=False)


def test_mutation_synthetic_anatomy_is_refused_by_preflight(anat):
    """The corpus must not be generated on the synthetic fallback at all."""
    broken = dataclasses.replace(anat, provenance="synthetic_fallback")
    spec = CorpusSpec(backends=CORPUS_BACKENDS)
    with pytest.raises(InertThetaDimension, match="synthetic fallback"):
        preflight(spec=spec, anat=broken, trajectory_level=False)


def test_preflight_accepts_the_real_prior(anat):
    """The complement of the mutations: on the real prior it must pass.

    Without this, every test above would also pass if `preflight` raised
    unconditionally.
    """
    spec = CorpusSpec(backends=CORPUS_BACKENDS)
    out = preflight(spec=spec, anat=anat, trajectory_level=False)
    assert out["inert"] == []
    assert out["anatomy"]["n_regions"] == 414
    assert out["anatomy"]["is_biological"] is True


# ---------------------------------------------------------------------------
# The sampling-diversity accounting -- not a refusal, a recorded deficit.
# ---------------------------------------------------------------------------


def test_velocity_effective_sample_size_is_shards_not_trajectories():
    """`generate_corpus` shares one velocity per batch, and one batch is a shard.

    Run 1 shipped 37 distinct `log_velocity` values against 37,843 values of
    `log_G` over the same 37,888 trajectories.  This is not caught by any
    sensitivity check -- `log_velocity` genuinely moves the simulator -- so it
    is accounted for separately and written into the index.
    """
    spec = CorpusSpec(batch=256)
    d = check_theta_sampling_diversity(spec, n_shards=148)
    eff = d["effective_distinct_values"]
    assert eff["log_velocity"] == 148
    assert eff["log_G"] == 148 * 256
    assert d["velocity_deficit_factor"] == 256
    # and the deficit must scale with batch, which is the lever we actually have
    assert check_theta_sampling_diversity(CorpusSpec(batch=1024), n_shards=37)[
        "effective_distinct_values"
    ]["log_velocity"] == 37
