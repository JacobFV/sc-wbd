"""R07 -- centering, shrinkage, and a simulated recovery of the decomposition.

R07 is declared in three places and each declaration is watched firing:

* :meth:`PopulationModel.assert_centered` on a hand-broken model;
* the session-effect centering inside :func:`decompose_sessions`;
* the **compiler's** ``check_r07``, on a deliberately broken
  :func:`hierarchical_effect_declarations`.  That last one is the important
  one: it proves the declaration this package emits is load-bearing rather than
  decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.individualize.hierarchy import (
    PopulationModel,
    R07Violation,
    decompose_sessions,
    hierarchical_effect_declarations,
    recover_decomposition,
)


# --------------------------------------------------------------- centering
def test_group_effects_are_centered_by_construction():
    pop = PopulationModel(
        group_names=("a", "b", "c"),
        group_counts=np.array([10.0, 30.0, 60.0]),
        alpha_raw=np.random.default_rng(0).normal(size=(3, 9)),
    )
    w = (pop.group_counts / pop.group_counts.sum()).reshape(-1, 1)
    assert np.abs((w * pop.alpha).sum(0)).max() < 1e-12
    pop.assert_centered()


def test_assert_centered_fires_when_the_projection_is_bypassed():
    """Break it on purpose: the guard must be able to say no."""

    class Broken(PopulationModel):
        @property
        def alpha(self):  # bypasses the sum-to-zero projection
            return self.alpha_raw

    pop = Broken(
        group_names=("a", "b"),
        group_counts=np.array([1.0, 1.0]),
        alpha_raw=np.array([[1.0] * 9, [1.0] * 9]),
    )
    with pytest.raises(R07Violation) as e:
        pop.assert_centered()
    assert "R07" in str(e.value)


def test_shrinkage_scales_must_be_positive_and_finite():
    with pytest.raises(R07Violation):
        PopulationModel(person_sd=np.zeros(9), session_sd=np.zeros(9)).assert_centered()


def test_session_spread_may_not_exceed_person_spread():
    """Identity is the slower quantity (sec. 6.5)."""
    with pytest.raises(R07Violation):
        PopulationModel(
            person_sd=np.full(9, 0.1), session_sd=np.full(9, 0.9)
        )


# --------------------------------------------------------------- decomposition
def _pop():
    return PopulationModel.reference()


def test_single_session_reports_the_split_as_unidentified():
    pop = _pop()
    d = decompose_sessions(pop, "population", np.ones((1, 9)) * 0.3, ["s0"])
    assert d.separable is False
    assert "NOT identified" in d.separability_reason
    assert np.all(d.zeta == 0.0)


def test_multi_session_zeta_sums_to_zero_exactly():
    pop = _pop()
    rng = np.random.default_rng(1)
    offs = rng.normal(size=(5, 9)) * 0.2
    d = decompose_sessions(pop, "population", offs, [f"s{i}" for i in range(5)])
    assert d.separable is True
    assert np.abs(d.zeta.sum(0)).max() < 1e-12
    assert d.centering_residual < 1e-12


def test_delta_is_shrunk_strictly():
    """Shrinkage is not decoration: the factor must be < 1 and applied."""
    pop = _pop()
    offs = np.ones((3, 9)) * 0.5
    d = decompose_sessions(
        pop, "population", offs, ["a", "b", "c"], observation_sd=pop.prior_sd
    )
    assert np.all(d.shrinkage_factor < 1.0)
    assert np.all(np.abs(d.delta) < np.abs(offs.mean(0)))


def test_already_shrunk_offsets_are_not_shrunk_twice():
    """The fit already carried the individual prior; doing it again biases low."""
    pop = _pop()
    offs = np.ones((3, 9)) * 0.5
    once = decompose_sessions(
        pop, "population", offs, ["a", "b", "c"],
        observation_sd=pop.prior_sd, already_shrunk=True,
    )
    twice = decompose_sessions(
        pop, "population", offs, ["a", "b", "c"],
        observation_sd=pop.prior_sd, already_shrunk=False,
    )
    assert np.allclose(once.delta, offs.mean(0))
    assert np.all(np.abs(twice.delta) < np.abs(once.delta))
    # the factor is still reported, flagged as a diagnostic rather than applied
    assert once.shrinkage_applied_in_fit is True
    assert np.allclose(once.shrinkage_factor, twice.shrinkage_factor)


def test_more_sessions_shrink_less():
    """A shrinkage factor that never moves could not be doing anything."""
    pop = _pop()
    f = []
    for n in (1, 4, 16):
        d = decompose_sessions(
            pop,
            "population",
            np.ones((n, 9)) * 0.5,
            [f"s{i}" for i in range(n)],
            observation_sd=pop.prior_sd,
        )
        f.append(float(d.shrinkage_factor[0]))
    assert f[0] < f[1] < f[2] < 1.0


def test_masked_coordinates_get_exactly_zero_delta_and_zeta():
    pop = _pop()
    mask = np.zeros(9, dtype=bool)
    mask[:3] = True
    offs = np.ones((3, 9)) * 0.5
    d = decompose_sessions(
        pop, "population", offs, ["a", "b", "c"], mask=mask
    )
    assert np.all(d.delta[3:] == 0.0)
    assert np.all(d.zeta[:, 3:] == 0.0)
    # and the assembled trait equals the population value there, bit for bit
    assert np.array_equal(d.theta_trait[3:], pop.population_value("population")[3:])


# --------------------------------------------------------------- recovery
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_decomposition_is_recovered_in_simulation(seed):
    """delta_p recovery and centering, on synthetic subjects, across seeds.

    Seeds are parametrised because an earlier version of this check ran at 700
    Adam steps and produced ``delta_corr`` between 0.71 and 0.93 depending on
    the random stream, with the pass/fail bar inside that range.  That is a
    test whose verdict is a coin flip.  The fix was to converge the optimiser
    and to judge against what is *extractable* at this noise level, not against
    a constant.
    """
    rec = recover_decomposition(seed=seed)
    assert rec["converged"], rec["last_decile_gain"]
    assert rec["identified"], rec
    assert rec["group_centering_residual"] < 1e-4
    # judged against the correlation a perfect shrinkage estimator could reach
    assert rec["delta_corr_efficiency"] > 0.9, rec
    # and against the do-nothing baseline: predicting delta_p = 0 for everyone
    # has RMSE equal to the true between-person sd
    assert rec["delta_rmse"] < rec["delta_true_sd"], rec
    # the discriminating half: a high sum correlation alone would not do
    assert rec["sum_corr"] > 0.95


def test_recovery_test_can_fail():
    """A recovery check that always passes would tell us nothing.

    Swamp the person effect with observation noise: delta_p is then genuinely
    not recoverable, and the check must say so.  Note the *efficiency* bar
    alone would pass here -- a perfect estimator recovers nothing either -- so
    this is also the test that the absolute floor is load-bearing.
    """
    rec = recover_decomposition(
        seed=0,
        n_participants=12,
        n_sessions_per_participant=2,
        person_sd=0.02,
        session_sd=0.02,
        noise_sd=1.0,
        steps=300,
    )
    assert not rec["identified"], rec
    assert rec["delta_corr"] < 0.5, rec["delta_corr"]
    assert rec["delta_corr_efficiency"] > 0.9, (
        "the efficiency bar alone does not catch this; the absolute floor does"
    )


# --------------------------------------------------------------- the compiler
def test_compiler_r07_accepts_our_declaration():
    from scwbd.schema.sources import PopulationStructure

    effects = hierarchical_effect_declarations()
    ps = PopulationStructure(effects=effects)
    assert ps.unidentified_effects() == (), [
        (e.name, e.deficiency()) for e in ps.unidentified_effects()
    ]


@pytest.mark.parametrize(
    "kw",
    [
        {"parameterization_person": "unconstrained", "with_shrinkage": False},
        {"recovery_tested": False},
        {"parameterization_group": "unconstrained", "with_shrinkage": False},
    ],
)
def test_compiler_r07_fires_on_a_broken_declaration(kw):
    """Break the declaration on purpose and watch check_r07 refuse."""
    from scwbd.schema.sources import PopulationStructure

    ps = PopulationStructure(effects=hierarchical_effect_declarations(**kw))
    bad = ps.unidentified_effects()
    assert bad, f"check_r07 could not fire for {kw}"
    assert all(e.deficiency() for e in bad)
