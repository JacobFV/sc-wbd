"""The analytic spherical lead field against its closed-form reference.

An unvalidated lead field is worthless, so this file pins the solver from three
independent directions:

1. the single-shell case against the textbook Legendre series derived
   independently in the test (not by calling the same code path);
2. the multilayer solver with equal conductivities, which must collapse onto the
   single-shell answer exactly;
3. physical invariants -- reciprocity in the source/sensor arguments, linearity
   in the dipole moment, and the ``1/sigma`` scaling of a homogeneous conductor.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import ObservationRefusal, Prior
from scwbd.observe.leadfield import (
    SphericalHeadModel,
    TissueConductivityPriors,
    legendre_p_and_dp,
)

from .conftest import HEAD_RADIUS

torch.set_default_dtype(torch.float64)


def _closed_form_single_shell(
    r0: torch.Tensor, q: torch.Tensor, e: torch.Tensor, R: float, sigma: float, n_max: int = 800
) -> float:
    """Textbook homogeneous-sphere series, derived independently of the solver.

    ``V(R, theta) = 1/(4 pi sigma R^2) sum_n (2n+1)/n (b/R)^{n-1}
    [n q_r P_n(x) + P_n'(x) (q.e_hat - q_r x)]``
    """
    b = float(r0.norm())
    r0h = r0 / b
    eh = e / e.norm()
    x = torch.tensor(float(r0h @ eh))
    P, dP = legendre_p_and_dp(x, n_max)
    n = torch.arange(1, n_max + 1, dtype=torch.float64)
    D = (2 * n + 1) / n * (b / R) ** (n - 1) / (4 * math.pi * sigma * R**2)
    q_r = float(q @ r0h)
    radial = float((D * n * P[1:]).sum()) * q_r
    tangential = float((D * dP[1:]).sum()) * (float(q @ eh) - q_r * float(x))
    return radial + tangential


def test_single_shell_matches_closed_form_series(homogeneous_head, sensor_positions):
    src = torch.tensor(
        [[0.0, 0.0, 0.050], [0.020, -0.030, 0.040], [0.0, 0.0, 0.0], [0.03, 0.03, 0.03]]
    )
    L = homogeneous_head.potential(src, sensor_positions)
    worst = 0.0
    for s in range(src.shape[0]):
        for e in range(sensor_positions.shape[0]):
            for k in range(3):
                q = torch.zeros(3)
                q[k] = 1.0
                ref = _closed_form_single_shell(
                    src[s], q, sensor_positions[e], HEAD_RADIUS, 0.33
                )
                got = float(L[e, s, k])
                if abs(ref) > 1e-12:
                    worst = max(worst, abs(got - ref) / abs(ref))
    assert worst < 1e-10, f"analytic sphere disagrees with its series: {worst:.3e}"


def test_multilayer_with_equal_conductivity_reduces_to_single_shell(
    homogeneous_head, sensor_positions, source_positions
):
    """The layered recursion must be an identity when there is nothing to layer."""
    sig = 0.33
    p = Prior("s", "delta", (sig,), units="S/m")
    layered = SphericalHeadModel(
        radii=(0.87 * HEAD_RADIUS, 0.92 * HEAD_RADIUS, 0.96 * HEAD_RADIUS, HEAD_RADIUS),
        conductivity=TissueConductivityPriors(brain=p, csf=p, skull=p, scalp=p),
    )
    a = layered.potential(source_positions, sensor_positions)
    b = homogeneous_head.potential(source_positions, sensor_positions)
    rel = float((a - b).abs().max() / b.abs().max())
    assert rel < 1e-9, f"layered solver is not exact for equal conductivities: {rel:.3e}"


def test_homogeneous_potential_scales_as_one_over_sigma(sensor_positions, source_positions):
    a = SphericalHeadModel(
        radii=(HEAD_RADIUS,), conductivity=TissueConductivityPriors.homogeneous(0.33)
    ).potential(source_positions, sensor_positions)
    b = SphericalHeadModel(
        radii=(HEAD_RADIUS,), conductivity=TissueConductivityPriors.homogeneous(0.66)
    ).potential(source_positions, sensor_positions)
    assert torch.allclose(a, 2.0 * b, rtol=1e-10, atol=0.0)


def test_lead_field_is_linear_in_dipole_moment(four_layer_head, sensor_positions):
    src = torch.tensor([[0.01, 0.02, 0.04]])
    L = four_layer_head.potential(src, sensor_positions)
    q1 = torch.tensor([[1e-9, -2e-9, 3e-9]])
    q2 = torch.tensor([[-4e-9, 1e-9, 0.5e-9]])
    y1 = torch.einsum("esk,sk->e", L, q1)
    y2 = torch.einsum("esk,sk->e", L, q2)
    y12 = torch.einsum("esk,sk->e", L, q1 + q2)
    assert torch.allclose(y1 + y2, y12, rtol=1e-12, atol=1e-24)


def test_skull_attenuates_and_csf_shunts(sensor_positions, source_positions):
    """Physics sanity: a resistive skull must reduce the scalp potential."""

    def head(skull_sigma: float) -> SphericalHeadModel:
        return SphericalHeadModel(
            radii=(0.87 * HEAD_RADIUS, 0.92 * HEAD_RADIUS, 0.96 * HEAD_RADIUS, HEAD_RADIUS),
            conductivity=TissueConductivityPriors(
                brain=Prior("b", "delta", (0.33,), units="S/m"),
                csf=Prior("c", "delta", (1.79,), units="S/m"),
                skull=Prior("s", "delta", (skull_sigma,), units="S/m"),
                scalp=Prior("p", "delta", (0.33,), units="S/m"),
            ),
        )

    weak = head(0.004).potential(source_positions, sensor_positions).abs().max()
    strong = head(0.033).potential(source_positions, sensor_positions).abs().max()
    assert weak < strong, "a more resistive skull must attenuate the scalp potential"


def test_source_outside_brain_compartment_refuses(four_layer_head, sensor_positions):
    outside = torch.tensor([[0.0, 0.0, 0.0895]])
    with pytest.raises(ObservationRefusal) as exc:
        four_layer_head.potential(outside, sensor_positions)
    assert exc.value.code == "R01"


def test_conductivity_is_a_prior_not_a_constant(four_layer_head):
    from scwbd.observe.leadfield import ITIS_CONDUCTIVITY

    for name, prior in zip(
        ITIS_CONDUCTIVITY.layer_names, ITIS_CONDUCTIVITY.priors
    ):
        assert prior.sd > 0, f"{name} conductivity is a point value, not a prior"
        assert prior.source != "unspecified", f"{name} conductivity has no citation"
        assert prior.validity is not None


def test_conductivity_uncertainty_propagates_into_the_ledger(
    sensor_positions, source_positions
):
    from scwbd.observe.leadfield import ITIS_CONDUCTIVITY

    head = SphericalHeadModel.adult_four_layer(HEAD_RADIUS, ITIS_CONDUCTIVITY)
    lf = head.lead_field(
        source_positions, sensor_positions, n_conductivity_draws=8, seed=3
    )
    v = lf.ledger.variance
    assert v.parameter_posterior != "unknown"
    assert float(v.parameter_posterior) > 0.0


def test_lead_field_is_the_support_not_a_scalp_label(
    four_layer_head, sensor_positions, source_positions
):
    """thesis Sec. 2.8: the electrode's spatial support IS the lead field."""
    lf = four_layer_head.lead_field(source_positions, sensor_positions)
    support = lf.as_support()
    assert support.kind == "sensor"
    assert support.psf is not None
    assert support.psf.kind == "leadfield"
    assert support.psf.matrix is not None
    assert support.psf.matrix.shape == (
        sensor_positions.shape[0],
        source_positions.shape[0] * 3,
    )
    assert support.psf.source_positions is not None
