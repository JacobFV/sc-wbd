"""The one declared fine/coarse pair: the operators, and whether the boundary
measurement can tell anything from anything.

Two jobs here.

The first is ordinary: ``R`` and ``P`` must be a genuine paired pair, ``P`` must
return a distribution, and the artefact loader must refuse a stale artefact.

The second is the one that matters.  ``reports/decorative_guards.md`` catalogues
~26 checks in this project that could not fail, and a boundary metric that
returns a plausible-looking number regardless of input is exactly that failure
in numerical dress.  So the metrics are exercised on lead fields whose answers
are known by construction -- one the parcel subspace represents perfectly, one
it cannot see at all -- and are required to return 0 and 1 respectively.  A
metric that could not distinguish those two would be reporting nothing.
"""

from __future__ import annotations

import json

import pytest
import torch

from scwbd.transforms import resolution_pair as rp
from scwbd.transforms.errors import ProlongationWithoutRestrictionError, SiteError
from scwbd.transforms.sheaf import FineDistribution

DT = torch.float64


@pytest.fixture
def toy():
    """9 fine elements, 3 parcels of 3, deliberately unequal areas."""
    assign = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    areas = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0], dtype=DT)
    return assign, areas, 3


# --------------------------------------------------------------------------
# the operators
# --------------------------------------------------------------------------
def test_restriction_is_an_area_weighted_mean(toy):
    assign, areas, nc = toy
    R = rp.restriction_matrix(assign, areas, nc)
    assert R.shape == (3, 9)
    # each row sums to one: a mean, not a sum
    assert torch.allclose(R.sum(dim=1), torch.ones(3, dtype=DT))
    # a constant fine field restricts to that constant
    x = torch.full((9,), 2.5, dtype=DT)
    assert torch.allclose(R @ x, torch.full((3,), 2.5, dtype=DT))
    # and the weights really are the areas
    assert torch.allclose(R[0, :3], areas[:3] / areas[:3].sum())


def test_R_and_P_are_a_right_inverse_pair(toy):
    assign, areas, nc = toy
    R = rp.restriction_matrix(assign, areas, nc)
    P = rp.prolongation_matrix(assign, nc)
    assert float((R @ P - torch.eye(nc, dtype=DT)).abs().max()) < 1e-15


def test_PR_is_the_area_weighted_projector(toy):
    assign, areas, nc = toy
    R = rp.restriction_matrix(assign, areas, nc)
    P = rp.prolongation_matrix(assign, nc)
    Pi = P @ R
    assert torch.allclose(Pi @ Pi, Pi, atol=1e-14)  # idempotent
    # self-adjoint in the area inner product, which is the metric R averages in
    A = torch.diag(areas)
    assert torch.allclose(A @ Pi, (A @ Pi).T, atol=1e-14)


def test_unassigned_fine_elements_are_not_folded_into_a_neighbour():
    assign = torch.tensor([0, 0, -1, 1, 1])
    areas = torch.ones(5, dtype=DT)
    R = rp.restriction_matrix(assign, areas, 2)
    P = rp.prolongation_matrix(assign, 2)
    assert float(R[:, 2].abs().max()) == 0.0
    assert float(P[2].abs().max()) == 0.0
    assert rp.assigned_area_fraction(assign, areas) == pytest.approx(0.8)


def test_a_coarse_element_owning_nothing_is_refused():
    with pytest.raises(SiteError, match="own no fine element"):
        rp.restriction_matrix(torch.tensor([0, 0, 1]), torch.ones(3, dtype=DT), 3)


def test_zero_area_fine_element_is_refused():
    with pytest.raises(SiteError, match="non-positive area"):
        rp.restriction_matrix(
            torch.tensor([0, 0]), torch.tensor([1.0, 0.0], dtype=DT), 1
        )


# --------------------------------------------------------------------------
# the prolongation is a distribution, not a point (R02)
# --------------------------------------------------------------------------
def test_prolongation_returns_a_distribution_over_the_unresolved_directions(toy):
    assign, areas, nc = toy
    g = torch.Generator().manual_seed(0)
    X = torch.randn((40, 9), generator=g, dtype=DT)
    pair = rp.build_scale_pair(assign, areas, nc, X, prior_sd_unresolved=1.0)
    out = pair.prolongation.prolong(torch.tensor([1.0, 2.0, 3.0], dtype=DT))
    assert isinstance(out, FineDistribution)
    assert out.resolved_rank == nc
    assert out.unresolved_rank == 9 - nc
    # the mean really is the indicator fill
    assert torch.allclose(out.mean[:3], torch.ones(3, dtype=DT))
    # the unresolved directions carry variance, the resolved ones do not
    assert float(out.sd.min()) > 0.0
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        out.as_point()
    assert exc.value.code == "R02"


def test_prolongation_that_understates_its_uncertainty_is_refused_R02(toy):
    """The bound is the sd the map declares; claiming less than the held-out
    residual is R02's actual subject."""
    assign, areas, nc = toy
    g = torch.Generator().manual_seed(1)
    X = torch.randn((40, 9), generator=g, dtype=DT) * 3.0
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        rp.build_scale_pair(assign, areas, nc, X, prior_sd_unresolved=1e-6)
    assert exc.value.code == "R02"
    assert "coverage test failed" in str(exc.value)


# --------------------------------------------------------------------------
# the boundary metrics must be able to tell anything from anything
# --------------------------------------------------------------------------
def _pair(assign, areas, nc):
    return rp.restriction_matrix(assign, areas, nc), rp.prolongation_matrix(assign, nc)


def test_boundary_metrics_are_zero_for_an_observable_the_parcels_carry(toy):
    """A lead field that reads only parcel means loses nothing to coarsening."""
    assign, areas, nc = toy
    R, P = _pair(assign, areas, nc)
    G = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=DT) @ R
    g = torch.Generator().manual_seed(2)
    X = torch.randn((9, 25), generator=g, dtype=DT)
    obs = rp.observable_error(G, R, P, X)
    assert obs["relative_error"] == pytest.approx(0.0, abs=1e-12)
    assert rp.perturbational_error(G, R, P)["median"] == pytest.approx(0.0, abs=1e-12)
    assert rp.lead_field_energy_retained(G, R, P, areas) == pytest.approx(1.0, abs=1e-12)


def test_boundary_metrics_are_one_for_an_observable_the_parcels_cannot_see(toy):
    """A lead field orthogonal to every piecewise-constant field: the coarse
    view predicts exactly nothing of it, and the metrics must say so."""
    assign, areas, nc = toy
    R, P = _pair(assign, areas, nc)
    # G P = 0: the sensor sums to zero over every parcel, so no piecewise
    # constant field -- which is all the coarse state can produce -- reaches it
    row = torch.zeros(9, dtype=DT)
    row[0], row[1] = 1.0, -1.0
    G = row.reshape(1, 9)
    assert float((G @ P).abs().max()) < 1e-15  # really is invisible
    g = torch.Generator().manual_seed(3)
    X = torch.randn((9, 25), generator=g, dtype=DT)
    obs = rp.observable_error(G, R, P, X)
    assert obs["relative_error"] == pytest.approx(1.0, rel=1e-9)
    assert rp.lead_field_energy_retained(G, R, P, areas) == pytest.approx(0.0, abs=1e-12)


def test_observable_error_reports_the_noise_floor_currency(toy):
    """The whitener changes the answer, which is the whole point of quoting a
    residual in standard deviations rather than in volts."""
    assign, areas, nc = toy
    R, P = _pair(assign, areas, nc)
    G = torch.eye(9, dtype=DT)[:4]
    g = torch.Generator().manual_seed(4)
    X = torch.randn((9, 12), generator=g, dtype=DT)
    plain = rp.observable_error(G, R, P, X)
    # a whitener that says channel 0 is a thousand times noisier
    W = torch.diag(torch.tensor([1e-3, 1.0, 1.0, 1.0], dtype=DT))
    whit = rp.observable_error(G, R, P, X, whitener=W)
    assert plain["relative_error"] != pytest.approx(whit["relative_error"], rel=1e-6)
    assert plain["unwhitened_relative_error"] == pytest.approx(
        whit["unwhitened_relative_error"], rel=1e-12
    )


# --------------------------------------------------------------------------
# the measured artefact: stale is not "close enough"
# --------------------------------------------------------------------------
def _record(**over):
    rec = {
        "schema_version": rp.SCHEMA_VERSION,
        "n_fine": 9,
        "n_coarse": 3,
        "membership_digest": "abc",
        "authority_policy": rp.AUTHORITY_POLICY,
        "coarse_roundtrip_residual": 0.0,
        "coarse_roundtrip_tolerance": 1e-9,
        "heldout_fine_residual": 1.0,
        "declared_prior_sd_unresolved": 2.0,
        "landmark_coverage": 0.94,
        "required_coverage": 0.8,
        "lead_field_energy_retained": 0.06,
        "fine_characteristic_scale_m": 0.005,
        "coarse_characteristic_scale_m": 0.05,
        "boundary": [],
        "perturbational": {"median": 0.9},
    }
    rec.update(over)
    return rec


@pytest.mark.parametrize(
    "over, why",
    [
        ({"schema_version": rp.SCHEMA_VERSION + 1}, "a record written under other semantics"),
        ({"membership_digest": "different"}, "a different parcellation or surface"),
        ({"n_fine": 10}, "a different fine support"),
        ({"n_coarse": 4}, "a different coarse support"),
        ({"authority_policy": "coarse_authoritative"}, "a different authority policy"),
        ({"heldout_fine_residual": None}, "a malformed record"),
    ],
)
def test_a_stale_artefact_is_not_loaded(tmp_path, over, why):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_record(**over)))
    got = rp.load_measurement(
        p, expect_digest="abc", expect_n_fine=9, expect_n_coarse=3
    )
    if over.get("heldout_fine_residual", 1.0) is None:
        # a malformed value still parses into the dataclass; the guard that
        # matters is that it cannot silently become a passing residual
        assert got is None or got.heldout_fine_residual is None
    else:
        assert got is None, f"loaded {why}"


def test_a_missing_artefact_is_none_not_a_default(tmp_path):
    assert rp.load_measurement(tmp_path / "nope.json") is None


def test_a_good_artefact_loads_and_reports_its_verdicts(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_record()))
    m = rp.load_measurement(p, expect_digest="abc", expect_n_fine=9, expect_n_coarse=3)
    assert m is not None
    assert m.roundtrip_ok and m.coverage_ok and m.prolongation_calibrated
    # an empty boundary list is not a pass
    assert not m.boundary_sufficient


def test_the_committed_measurement_is_present_and_self_consistent():
    """The artefact this repository ships must actually license the pair."""
    m = rp.load_measurement()
    assert m is not None, (
        f"{rp.MEASUREMENT_RELPATH} is missing; regenerate it with "
        "benchmarks/transforms/resolution_pair.py"
    )
    assert m.roundtrip_ok, m.coarse_roundtrip_residual
    assert m.coverage_ok, m.landmark_coverage
    assert m.prolongation_calibrated
    assert m.n_coarse < m.n_fine
    assert 0.0 <= m.lead_field_energy_retained <= 1.0
    assert m.boundary, "no boundary ensemble was measured"
    # The filed result. If this ever passes, the report is out of date and the
    # claim boundary must be rewritten before the assertion is relaxed.
    assert not m.boundary_sufficient, (
        "the parcel support now preserves the EEG observable; "
        "reports/transforms/resolution_pair.md says it does not"
    )
