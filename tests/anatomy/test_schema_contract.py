"""The contract this package owes agents A, E and I.

These tests pin the *interface*, not the numbers: if agent A's schema types
change, or if a prior stops being sampleable with an explicit seed, the failure
should land here rather than inside agent E's integrator.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from scwbd.anatomy import _compat
from scwbd.anatomy._compat import (
    EVIDENCE_ORDER,
    design_estimable_ledger,
    evidence_rank,
    externally_bounded_ledger,
    group_average_ledger,
    prior_quantile,
)


def test_we_use_agent_as_schema_types_not_local_copies():
    from scwbd.schema import Prior as SchemaPrior  # noqa: F401
    from scwbd.schema import UncertaintyLedger as SchemaLedger

    assert _compat.UncertaintyLedger is SchemaLedger


def test_evidence_classes_match_the_operator_spec_vocabulary():
    """ARCHITECTURE.md §2: OperatorSpec.evidence_class is hard/soft/proposed."""
    assert set(EVIDENCE_ORDER) == {"absent", "proposed", "soft", "hard"}
    assert evidence_rank("hard") > evidence_rank("soft") > evidence_rank("proposed")
    assert evidence_rank("proposed") > evidence_rank("absent")


# ---------------------------------------------------------------------------
# R08
# ---------------------------------------------------------------------------
def test_group_average_ledger_refuses_a_point_bias():
    """A point bias on a group average is exactly what R08 exists to catch."""
    with pytest.raises(ValueError, match="non-degenerate"):
        group_average_ledger(
            units="zscore",
            bias_interval=(0.0, 0.0),
            variance={"measurement": 1.0},
            forbidden_inference="x",
        )


def test_group_average_ledger_is_prior_specified_and_r08_clean():
    led = group_average_ledger(
        units="zscore",
        bias_interval=(-1.0, 1.0),
        variance={"measurement": 1.0},
        forbidden_inference="not a subject value",
        n_donors=8,
    )
    assert led.bias_status == "prior_specified_sensitivity"
    assert led.has_estimator()
    assert led.validity_domain["n_donors"] == 8
    assert led.validity_domain["forbidden_inference"]


def test_externally_bounded_ledger_requires_a_named_bound():
    with pytest.raises(ValueError, match="external_bound_source"):
        externally_bounded_ledger(
            units="mm", bias_interval=(-1.0, 1.0), external_bound_source="",
            variance={},
        )
    led = externally_bounded_ledger(
        units="mm", bias_interval=(-1.0, 1.0),
        external_bound_source="template voxel size", variance={},
    )
    assert led.has_estimator()


def test_design_estimable_ledger_requires_a_named_estimator():
    with pytest.raises(ValueError, match="bias_estimator"):
        design_estimable_ledger(
            units="mm", bias_interval=(0.0, 0.0), bias_estimator="", variance={},
        )


def test_variance_components_use_the_declared_vocabulary(brain_prior):
    from scwbd.schema.ledger import VARIANCE_COMPONENTS

    for name, led in brain_prior.ledger_summary().items():
        for k in led["variance"]:
            assert k in VARIANCE_COMPONENTS, f"{name}: undeclared variance component {k!r}"


def test_ledgers_never_collapse_bias_and_variance(brain_prior):
    for name, led in brain_prior.ledger_summary().items():
        assert "variance" in led and "bias_interval" in led
        assert isinstance(led["variance"], dict)


# ---------------------------------------------------------------------------
# priors as agent E consumes them
# ---------------------------------------------------------------------------
def test_every_prior_this_package_ships_carries_units_and_provenance():
    from scwbd.anatomy.connectome import (
        CONDUCTION_VELOCITY_PRIOR,
        EDR_LAMBDA_PRIOR,
        TORTUOSITY_PRIOR,
    )

    for p in (CONDUCTION_VELOCITY_PRIOR, TORTUOSITY_PRIOR, EDR_LAMBDA_PRIOR):
        assert p.units
        assert len(p.provenance) > 40, "an uncited prior is a hidden assumption"


def test_priors_sample_deterministically_from_an_explicit_seed(brain_prior):
    """ARCHITECTURE.md §3: determinism is a test, not an aspiration."""
    p = brain_prior.velocity_prior()
    a = np.asarray(p.sample(11, 100))
    b = np.asarray(p.sample(11, 100))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, np.asarray(p.sample(12, 100)))


def test_priors_reject_a_non_integer_seed(brain_prior):
    with pytest.raises(TypeError):
        brain_prior.velocity_prior().sample(0.5)  # type: ignore[arg-type]


def test_prior_quantile_helper_matches_sampling(brain_prior):
    p = brain_prior.velocity_prior()
    s = np.asarray(p.sample(0, 200_000))
    for q in (0.1, 0.5, 0.9):
        assert prior_quantile(p, q) == pytest.approx(np.quantile(s, q), rel=0.03)


def test_priors_serialise_and_round_trip(brain_prior):
    from scwbd.schema import as_prior

    p = brain_prior.velocity_prior()
    d = p.model_dump(mode="json")
    q = as_prior(d)
    assert q.mu == p.mu and q.units == p.units


def test_per_parcel_priors_are_json_serialisable(brain_prior):
    import json

    for getter in (brain_prior.ei_ratio_prior, brain_prior.timescale_prior):
        payload = [p.model_dump(mode="json") for p in getter()[:5]]
        json.loads(json.dumps(payload))


def test_delay_model_exposes_priors_rather_than_a_matrix_of_numbers(brain_prior):
    dm = brain_prior.delay_prior_ms()
    assert dm.velocity_prior.kind == "lognormal"
    assert dm.tortuosity_prior.kind == "lognormal"
    assert dm.length_source in ("euclidean", "geodesic")
    # the point estimate is available but must be asked for by name
    assert hasattr(dm, "median_delay_s")
    assert hasattr(dm, "sample_delay_s")
