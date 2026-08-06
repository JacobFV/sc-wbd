"""Equation (T5), its mandatory cross terms, and the nonlinear replacements.

The headline test is :func:`test_shared_session_calibration_understatement`:
a calibration error shared across a session's observations produces strictly
larger aggregate uncertainty than the independence assumption, and the test
prints how much independence understates it.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.transforms.errors import CovarianceError, LinearizationInvalidError
from scwbd.transforms.se3 import DTYPE, Pose, adjoint
from scwbd.transforms.uncertainty import (
    IntervalBox,
    PoseUncertainty,
    chain_jacobians,
    independence_understatement,
    interval_propagate,
    linearization_error,
    monte_carlo_propagate,
    propagate_chain,
    propagate_first_order,
    sample_chain,
)


def t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE)


# --------------------------------------------------------------------------
# T5 itself
# --------------------------------------------------------------------------


def test_first_order_matches_the_closed_form_for_an_affine_map() -> None:
    A = t([[2.0, -1.0], [0.5, 3.0]])
    B = t([[1.0, 0.0], [0.0, -2.0]])

    def f(x, c):
        return A @ x + B @ c

    Sx = t([[1.0, 0.2], [0.2, 0.5]])
    Sc = t([[0.3, 0.0], [0.0, 0.7]])
    Sxc = t([[0.1, -0.05], [0.02, 0.08]])
    r = propagate_first_order(f, t([1.0, 2.0]), t([0.5, -1.0]), Sx=Sx, Sc=Sc, Sxc=Sxc)
    expected = A @ Sx @ A.T + B @ Sc @ B.T + A @ Sxc @ B.T + B @ Sxc.T @ A.T
    assert torch.allclose(r.cov, expected, atol=1e-12)
    assert torch.allclose(r.terms["Jx"], A, atol=1e-12)
    assert torch.allclose(r.terms["Jc"], B, atol=1e-12)


def test_bias_propagates_separately_and_carries_model_discrepancy() -> None:
    A = t([[2.0, 0.0], [0.0, 1.0]])
    B = t([[1.0], [1.0]])

    def f(x, c):
        return A @ x + B @ c

    r = propagate_first_order(
        f,
        t([0.0, 0.0]),
        t([0.0]),
        Sx=torch.eye(2, dtype=DTYPE),
        Sc=torch.eye(1, dtype=DTYPE),
        bx=t([0.5, -0.25]),
        bc=t([0.1]),
        delta_f=t([0.01, 0.02]),
    )
    # b_z = Jx bx + Jc bc + delta_f, and never merged into cov
    assert torch.allclose(r.bias, A @ t([0.5, -0.25]) + B @ t([0.1]) + t([0.01, 0.02]))
    assert float(torch.trace(r.cov)) == pytest.approx(4.0 + 1.0 + 1.0 + 1.0)


def test_cross_terms_can_only_be_dropped_deliberately() -> None:
    def f(x, c):
        return x * c

    r_full = propagate_first_order(
        f, t([2.0]), t([3.0]), Sx=t([[1.0]]), Sc=t([[1.0]]), Sxc=t([[0.9]])
    )
    r_none = propagate_first_order(
        f, t([2.0]), t([3.0]), Sx=t([[1.0]]), Sc=t([[1.0]]), Sxc=t([[0.9]]),
        include_cross=False,
    )
    assert r_full.method == "first_order_T5"
    assert "NO_CROSS(invalid)" in r_none.method  # the wrong answer says so
    assert r_full.total_variance > r_none.total_variance


def test_inconsistent_cross_covariance_is_refused() -> None:
    """A declared correlation must be consistent with its marginals."""

    def f(x, c):
        return x + c

    with pytest.raises(CovarianceError) as exc:
        propagate_first_order(
            f, t([0.0]), t([0.0]), Sx=t([[1.0]]), Sc=t([[1.0]]), Sxc=t([[2.0]])
        )
    assert "not PSD" in str(exc.value)


# --------------------------------------------------------------------------
# THE shared-calibration test
# --------------------------------------------------------------------------


def test_shared_session_calibration_understatement() -> None:
    """A shared session gain understates uncertainty when assumed independent.

    Ten EEG channels in one session are scaled by one amplifier gain
    calibration ``c``.  The aggregate quantity is their mean amplitude.  Because
    ``c`` is common, the per-channel errors it induces are perfectly correlated
    and do *not* average away -- which is precisely what the T5 cross terms
    encode.  Assuming independence divides the calibration contribution by ``n``.
    """
    n = 10
    x0 = torch.full((n,), 5.0, dtype=DTYPE)  # microvolts, per-channel amplitude
    c0 = t([1.0])  # amplifier gain calibration

    def aggregate(x, c):
        return (x * c).mean().reshape(1)

    sigma_meas = 0.4  # independent per-channel measurement noise
    sigma_gain = 0.05  # 5% shared gain uncertainty
    Sx = torch.eye(n, dtype=DTYPE) * sigma_meas**2
    Sc = t([[sigma_gain**2]])
    # x and c are independent here; the sharing enters because one c multiplies
    # every channel. The T5 cross block for the *derived per-channel* quantities
    # is built below, and the aggregate is what makes the difference visible.
    rep = independence_understatement(
        aggregate, x0, c0, Sx=Sx, Sc=Sc, Sxc=torch.zeros((n, 1), dtype=DTYPE)
    )
    assert rep["trace_ratio"] == pytest.approx(1.0)  # no cross term declared yet

    # Now the realistic case: each channel's *reported* value already contains
    # the calibration, so Sigma_xc is nonzero -- the observations covary with the
    # calibration because they were produced by it.
    Sxc = torch.full((n, 1), x0[0].item() * sigma_gain**2, dtype=DTYPE)
    Sx_dep = Sx + (x0[:, None] * x0[None, :]) * sigma_gain**2
    rep = independence_understatement(
        aggregate, x0, c0, Sx=Sx_dep, Sc=Sc, Sxc=Sxc
    )
    print(
        "\nshared session calibration, aggregate over "
        f"{n} channels:\n"
        f"  total variance with T5 cross terms : {rep['trace_with_cross']:.6e}\n"
        f"  total variance assuming independence: {rep['trace_independent']:.6e}\n"
        f"  variance understated by a factor of : {rep['trace_ratio']:.3f}\n"
        f"  i.e. reported sd too small by       : "
        f"{100 * (1 - 1 / rep['sd_ratio']):.1f}%"
    )
    assert rep["understated"], "dropping the cross terms must not be free"
    assert rep["trace_ratio"] > 1.2
    assert rep["cross_term_trace"] > 0

    # and the correct answer agrees with Monte Carlo on the same joint law
    mc = monte_carlo_propagate(
        aggregate, x0, c0, Sx=Sx_dep, Sc=Sc, Sxc=Sxc, n=20000, seed=11
    )
    assert mc.total_variance == pytest.approx(rep["trace_with_cross"], rel=0.08)
    # while the independence assumption does not
    assert mc.total_variance > rep["trace_independent"] * 1.15


def test_shared_calibration_does_not_average_away_with_more_samples() -> None:
    """The understatement grows with n: that is the whole point of the term."""

    def make(n: int) -> float:
        x0 = torch.full((n,), 5.0, dtype=DTYPE)
        c0 = t([1.0])
        Sx = torch.eye(n, dtype=DTYPE) * 0.4**2 + (x0[:, None] * x0[None, :]) * 0.05**2
        Sxc = torch.full((n, 1), 5.0 * 0.05**2, dtype=DTYPE)
        rep = independence_understatement(
            lambda x, c: (x * c).mean().reshape(1),
            x0,
            c0,
            Sx=Sx,
            Sc=t([[0.05**2]]),
            Sxc=Sxc,
        )
        return rep["trace_ratio"]

    ratios = [make(n) for n in (2, 10, 50)]
    assert ratios[0] < ratios[1] < ratios[2]


# --------------------------------------------------------------------------
# nonlinear maps: Monte Carlo and intervals
# --------------------------------------------------------------------------


def test_linearization_error_flags_a_thresholded_map() -> None:
    """Thresholded tractography: T5 is meaningless through a hard threshold."""

    def thresholded(x, c):
        return torch.sigmoid(40.0 * (x * c - 1.0))

    rep = linearization_error(
        thresholded, t([1.0]), t([1.0]), Sx=t([[0.25]]), Sc=t([[0.04]]), seed=3, n=4000
    )
    assert not rep["linearization_valid"]
    with pytest.raises(LinearizationInvalidError) as exc:
        linearization_error(
            thresholded,
            t([1.0]),
            t([1.0]),
            Sx=t([[0.25]]),
            Sc=t([[0.04]]),
            seed=3,
            n=4000,
            raise_on_invalid=True,
        )
    assert "monte_carlo_propagate" in exc.value.remedy


def test_linearization_error_accepts_a_mild_map() -> None:
    def mild(x, c):
        return x * c + 0.01 * x**2

    rep = linearization_error(
        mild, t([1.0]), t([1.0]), Sx=t([[1e-4]]), Sc=t([[1e-4]]), seed=5, n=4000
    )
    assert rep["linearization_valid"]


def test_monte_carlo_honours_the_declared_cross_covariance() -> None:
    def f(x, c):
        return (x + c).reshape(1)

    kw = dict(Sx=t([[1.0]]), Sc=t([[1.0]]), n=40000, seed=7)
    pos = monte_carlo_propagate(f, t([0.0]), t([0.0]), Sxc=t([[0.8]]), **kw)
    neg = monte_carlo_propagate(f, t([0.0]), t([0.0]), Sxc=t([[-0.8]]), **kw)
    assert float(pos.cov[0, 0]) == pytest.approx(1 + 1 + 2 * 0.8, rel=0.05)
    assert float(neg.cov[0, 0]) == pytest.approx(1 + 1 - 2 * 0.8, rel=0.05)


def test_interval_propagation_is_exact_for_affine_and_labelled_otherwise() -> None:
    def affine(x, c):
        return (2.0 * x + 3.0 * c).reshape(1)

    box_x = IntervalBox.around(t([1.0]), t([0.1]))
    box_c = IntervalBox.around(t([2.0]), t([0.2]))
    r = interval_propagate(affine, box_x, box_c, method="corners")
    assert r.rigorous
    assert float(r.lo[0]) == pytest.approx(2 * 0.9 + 3 * 1.8)
    assert float(r.hi[0]) == pytest.approx(2 * 1.1 + 3 * 2.2)

    def focusing(x, c):
        # acoustic focusing: strongly nonlinear in the phase offset
        return torch.sin(6.0 * x + c).reshape(1)

    r2 = interval_propagate(focusing, box_x, box_c, method="lipschitz", n_probe=64, seed=1)
    assert not r2.rigorous  # honest: the Jacobian sup was sampled, not proven
    assert "not a certified bound" in r2.provenance["note"]
    mc = monte_carlo_propagate(
        focusing, t([1.0]), t([2.0]), Sx=t([[0.003]]), Sc=t([[0.012]]), n=4000, seed=2
    )
    assert float(r2.lo[0]) <= float(mc.value[0]) <= float(r2.hi[0])


# --------------------------------------------------------------------------
# SE(3) chains
# --------------------------------------------------------------------------


def test_chain_jacobians_are_the_expected_adjoints() -> None:
    ps = [Pose.from_twist(torch.randn(6, dtype=DTYPE) * 0.2, f"f{i}", f"f{i+1}", units="mm")
          for i in range(3)]
    Js = chain_jacobians(ps)
    assert torch.allclose(Js[-1], torch.eye(6, dtype=DTYPE))
    suffix = ps[1].matrix @ ps[2].matrix
    assert torch.allclose(Js[0], adjoint(torch.linalg.inv(suffix)), atol=1e-12)


def test_chain_covariance_matches_monte_carlo(chain_graph) -> None:
    """Adjoint propagation reproduces the sampled distribution of the chain."""
    path = chain_graph.path("atlas", "device", at=100.0).best
    poses = [e.pose for e in path.edges]
    uncs = [e.uncertainty for e in path.edges]
    analytic = propagate_chain(
        poses, uncs, shared_covariances=chain_graph.shared_covariances
    )
    draws = sample_chain(
        poses, uncs, shared_covariances=chain_graph.shared_covariances, n=20000, seed=13
    )
    emp = (draws - draws.mean(0)).T @ (draws - draws.mean(0)) / (draws.shape[0] - 1)
    assert float(torch.trace(emp)) == pytest.approx(
        float(torch.trace(analytic.cov)), rel=0.08
    )


def test_shared_tracker_calibration_inflates_the_pose_chain(chain_graph, tracker_cal) -> None:
    """The optical tracker calibration enters two edges; it must not cancel."""
    path = chain_graph.path("atlas", "device", at=100.0).best
    u = path.uncertainty
    rep = u.understatement()
    print(
        "\nequation (3) chain, shared optical-tracker calibration:\n"
        f"  trace(Sigma) with cross terms      : {rep['trace_with_shared_cross_terms']:.6e}\n"
        f"  trace(Sigma) assuming independence : {rep['trace_assuming_independence']:.6e}\n"
        f"  variance understated by factor     : {rep['trace_ratio']:.3f}\n"
        f"  translation sd (mm)                : {u.translation_sd.tolist()}"
    )
    assert tracker_cal in u.shared_sources
    assert rep["trace_ratio"] > 1.0, "shared calibration must not average away"


def test_systematic_offsets_propagate_as_a_separate_twist_bias() -> None:
    """§2.8: bias travels through the adjoints, but never joins the covariance."""
    p1 = Pose.from_twist(t([10.0, 0.0, 0.0, 0.0, 0.0, 0.3]), "a", "b", units="mm")
    p2 = Pose.from_twist(t([0.0, 5.0, 0.0, 0.1, 0.0, 0.0]), "b", "c", units="mm")
    u1 = PoseUncertainty(torch.eye(6, dtype=DTYPE) * 0.01, bias=t([1.0, 0, 0, 0, 0, 0]))
    u2 = PoseUncertainty(torch.eye(6, dtype=DTYPE) * 0.01)
    res = propagate_chain([p1, p2], [u1, u2])
    Js = chain_jacobians([p1, p2])
    assert torch.allclose(res.bias, Js[0] @ u1.bias)
    assert float(torch.linalg.norm(res.bias)) > 0
    # the bias does not appear in the covariance
    assert float(torch.trace(res.cov)) == pytest.approx(
        float(torch.trace(Js[0] @ u1.cov @ Js[0].T + Js[1] @ u2.cov @ Js[1].T))
    )


def test_chain_refuses_an_unquantified_shared_calibration() -> None:
    p = Pose.from_twist(t([1.0, 0, 0, 0, 0, 0]), "a", "b", units="mm")
    u = PoseUncertainty(
        torch.eye(6, dtype=DTYPE) * 0.01,
        calibration_source="undeclared_cal",
        sensitivity=torch.ones((6, 1), dtype=DTYPE),
    )
    with pytest.raises(Exception) as exc:
        propagate_chain([p], [u], shared_covariances={})
    assert "calibration source" in str(exc.value)


def test_pose_uncertainty_refuses_a_sensitivity_without_a_named_source() -> None:
    with pytest.raises(Exception):
        PoseUncertainty(torch.eye(6, dtype=DTYPE), sensitivity=torch.ones((6, 2), dtype=DTYPE))
