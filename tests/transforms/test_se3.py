"""SE(3) algebra and the measurement type around it.

Covers, from the definition of done: unit mismatch caught, handedness /
reflection caught, determinant sign check, non-invertible transform refused.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.transforms.errors import (
    EpochMismatchError,
    FrameMismatchError,
    HandednessError,
    TransformError,
    UnitMismatchError,
    ValidityIntervalError,
)
from scwbd.transforms.se3 import (
    DTYPE,
    Pose,
    ValidityInterval,
    adjoint,
    check_rigid,
    compose_all,
    exp_se3,
    exp_so3,
    hat,
    inv_left_jacobian_so3,
    left_jacobian_se3,
    left_jacobian_so3,
    log_se3,
    log_so3,
    round_trip_residual,
    vee,
)
from scwbd.transforms.units import Handedness, convert, require_same_unit


def _twist(seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(6, dtype=DTYPE, generator=g)


# --------------------------------------------------------------------------
# Lie algebra
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_exp_log_round_trip(seed: int) -> None:
    xi = _twist(seed)
    assert torch.allclose(log_se3(exp_se3(xi)), xi, atol=1e-12)


def test_exp_log_small_and_near_pi() -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    for theta in [0.0, 1e-12, 1e-8, 1e-4, 1.0, math.pi - 1e-6, math.pi - 1e-9]:
        R = exp_so3(axis * theta)
        rec = log_so3(R)
        assert float(torch.linalg.norm(rec - axis * theta)) < 1e-8, theta


def test_hat_vee_inverse() -> None:
    v = torch.tensor([0.3, -1.2, 5.0], dtype=DTYPE)
    assert torch.allclose(vee(hat(v)), v)
    assert torch.allclose(hat(v), -hat(v).T)


def test_adjoint_is_a_homomorphism() -> None:
    T1, T2 = exp_se3(_twist(1)), exp_se3(_twist(2))
    assert torch.allclose(adjoint(T1 @ T2), adjoint(T1) @ adjoint(T2), atol=1e-10)


def test_adjoint_conjugation_identity() -> None:
    """``exp(Ad(T) xi) == T exp(xi) T^-1`` -- the identity chain propagation uses."""
    T, xi = exp_se3(_twist(3)), _twist(4, 0.2)
    lhs = exp_se3(adjoint(T) @ xi)
    rhs = T @ exp_se3(xi) @ torch.linalg.inv(T)
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_left_jacobian_matches_finite_difference() -> None:
    phi = torch.tensor([0.2, -0.4, 0.1], dtype=DTYPE)
    J = left_jacobian_so3(phi)
    eps = 1e-7
    for k in range(3):
        d = torch.zeros(3, dtype=DTYPE)
        d[k] = eps
        num = log_so3(exp_so3(phi + d) @ exp_so3(phi).T) / eps
        assert torch.allclose(num, J[:, k], atol=1e-5)


def test_inverse_left_jacobian() -> None:
    phi = torch.tensor([0.7, 0.1, -0.3], dtype=DTYPE)
    assert torch.allclose(
        left_jacobian_so3(phi) @ inv_left_jacobian_so3(phi),
        torch.eye(3, dtype=DTYPE),
        atol=1e-12,
    )


def test_se3_left_jacobian_first_order() -> None:
    xi, d = _twist(5, 0.3), _twist(6, 1e-7)
    J = left_jacobian_se3(xi)
    lhs = log_se3(exp_se3(xi + d) @ torch.linalg.inv(exp_se3(xi)))
    assert torch.allclose(lhs, J @ d, atol=1e-9)


# --------------------------------------------------------------------------
# the measurement type
# --------------------------------------------------------------------------


def test_pose_carries_frames_units_handedness_validity() -> None:
    p = Pose.from_twist(
        _twist(7, 0.1), "head", "tracker", units="mm", validity=ValidityInterval(0.0, 10.0)
    )
    assert p.label == "head<-tracker"
    assert p.units == "mm"
    assert p.handedness is Handedness.RIGHT
    assert p.validity.contains(5.0) and not p.validity.contains(11.0)


def test_unit_mismatch_is_refused_not_converted() -> None:
    """A mm pose and a m pose must not compose. Silent rescaling is the bug."""
    a = Pose.from_twist(_twist(8, 0.1), "a", "b", units="mm")
    b = Pose.from_twist(_twist(9, 0.1), "b", "c", units="m")
    with pytest.raises(UnitMismatchError) as exc:
        a.compose(b)
    assert exc.value.code == "R01"
    assert "mm" in str(exc.value) and "'m'" in str(exc.value)
    assert "convert" in exc.value.remedy


def test_explicit_conversion_is_available_but_must_be_asked_for() -> None:
    assert convert(1.0, "m", "mm") == pytest.approx(1000.0)
    with pytest.raises(UnitMismatchError):
        convert(1.0, "voxel", "mm")  # needs a frame-graph edge, not a scalar
    with pytest.raises(UnitMismatchError):
        require_same_unit("mm", "m", context="test")


def test_handedness_mismatch_is_refused() -> None:
    a = Pose.from_twist(_twist(10, 0.1), "a", "b", units="mm", handedness="right")
    b = Pose.from_twist(_twist(11, 0.1), "b", "c", units="mm", handedness="left")
    with pytest.raises(HandednessError) as exc:
        a.compose(b)
    assert exc.value.code == "R01"
    assert "mirror" in exc.value.remedy


def test_determinant_sign_check_rejects_a_reflection() -> None:
    """det(R) = -1 is a left/right flip wearing a rotation's clothes."""
    M = torch.eye(4, dtype=DTYPE)
    M[0, 0] = -1.0  # mirror the x axis: still orthonormal, det = -1
    assert check_rigid(M).is_reflection
    assert check_rigid(M).determinant == pytest.approx(-1.0)
    with pytest.raises(HandednessError) as exc:
        Pose(M, "a", "b", units="mm")
    assert "det(R)" in str(exc.value)


def test_hidden_scale_in_a_rigid_pose_is_rejected() -> None:
    M = torch.eye(4, dtype=DTYPE)
    M[:3, :3] *= 1.01
    with pytest.raises(TransformError) as exc:
        Pose(M, "a", "b", units="mm")
    assert "orthonormal" in str(exc.value) or "det" in str(exc.value)


def test_projective_bottom_row_is_rejected() -> None:
    M = torch.eye(4, dtype=DTYPE)
    M[3, 0] = 0.1
    with pytest.raises(TransformError):
        Pose(M, "a", "b", units="mm")


def test_non_orthonormal_rotation_is_rejected_and_can_be_projected() -> None:
    M = torch.eye(4, dtype=DTYPE)
    M[:3, :3] += 1e-3 * torch.ones((3, 3), dtype=DTYPE)
    with pytest.raises(TransformError):
        Pose(M, "a", "b", units="mm")
    # the fix is explicit and recorded
    ok = Pose(torch.eye(4, dtype=DTYPE), "a", "b", units="mm")
    fixed = Pose(M, "a", "b", units="mm", rigid_tol=1e-1).orthonormalized()
    assert fixed.provenance["orthonormalization_correction"] > 0
    assert check_rigid(fixed.matrix).orthonormality_residual < 1e-12
    assert ok.units == fixed.units


def test_frame_mismatch_is_refused() -> None:
    a = Pose.from_twist(_twist(12, 0.1), "a", "b", units="mm")
    c = Pose.from_twist(_twist(13, 0.1), "c", "d", units="mm")
    with pytest.raises(FrameMismatchError):
        a.compose(c)


def test_self_edge_is_refused() -> None:
    with pytest.raises(FrameMismatchError):
        Pose.identity("a", "a")


def test_epoch_mismatch_is_refused() -> None:
    """Tuesday's head<-tracker is not Wednesday's."""
    a = Pose.from_twist(_twist(14, 0.1), "a", "b", units="mm", epoch="ses-01")
    b = Pose.from_twist(_twist(15, 0.1), "b", "c", units="mm", epoch="ses-02")
    with pytest.raises(EpochMismatchError) as exc:
        a.compose(b)
    assert exc.value.code == "R01"


def test_non_overlapping_validity_is_refused() -> None:
    a = Pose.from_twist(
        _twist(16, 0.1), "a", "b", units="mm", validity=ValidityInterval(0.0, 10.0)
    )
    b = Pose.from_twist(
        _twist(17, 0.1), "b", "c", units="mm", validity=ValidityInterval(20.0, 30.0)
    )
    with pytest.raises(ValidityIntervalError):
        a.compose(b)


def test_query_outside_joint_validity_is_refused() -> None:
    a = Pose.from_twist(
        _twist(18, 0.1), "a", "b", units="mm", validity=ValidityInterval(0.0, 10.0)
    )
    b = Pose.from_twist(
        _twist(19, 0.1), "b", "c", units="mm", validity=ValidityInterval(5.0, 30.0)
    )
    assert a.compose(b, at=7.0).validity.contains(7.0)
    with pytest.raises(ValidityIntervalError):
        a.compose(b, at=2.0)


def test_validity_intervals_on_different_clocks_do_not_intersect() -> None:
    with pytest.raises(ValidityIntervalError):
        ValidityInterval(0, 1, "eeg_amp").intersect(ValidityInterval(0, 1, "scanner"))


# --------------------------------------------------------------------------
# inversion, interpolation, round trip
# --------------------------------------------------------------------------


def test_inverse_and_round_trip_residual_is_measured() -> None:
    p = Pose.from_twist(_twist(20, 0.5), "a", "b", units="mm")
    rec = round_trip_residual(p, p.inverse())
    assert rec["units"] == "mm"
    # measured, and reported even when it is tiny
    assert rec["translation"] < 1e-9
    assert set(rec) == {"translation", "rotation_rad", "rotation_deg", "twist_norm", "units"}


def test_round_trip_of_a_mismatched_reverse_transform_is_nonzero() -> None:
    """An independently measured reverse transform does not cancel exactly."""
    p = Pose.from_twist(_twist(21, 0.5), "a", "b", units="mm")
    perturbed = Pose(
        exp_se3(torch.tensor([0.3, -0.2, 0.1, 1e-3, 0.0, 0.0], dtype=DTYPE)) @ p.inverse().matrix,
        "b",
        "a",
        units="mm",
    )
    rec = round_trip_residual(p, perturbed)
    assert rec["translation"] > 0.3
    assert rec["rotation_deg"] > 0.0


def test_singular_pose_cannot_be_constructed_so_inversion_is_always_defined() -> None:
    M = torch.eye(4, dtype=DTYPE)
    M[:3, :3] = torch.zeros((3, 3), dtype=DTYPE)
    with pytest.raises(TransformError):
        Pose(M, "a", "b", units="mm")


def test_interpolation_is_geodesic_and_frame_checked() -> None:
    a = Pose.identity("head", "tracker", units="mm")
    b = Pose.from_twist(_twist(22, 0.4), "head", "tracker", units="mm")
    mid = a.interpolate(b, 0.5)
    # from the identity, applying the halfway pose twice returns the endpoint
    # (that is what "geodesic" means here, and it fails for naive matrix lerp)
    back = log_se3(torch.linalg.inv(mid.matrix @ mid.matrix) @ b.matrix)
    assert float(torch.linalg.norm(back)) < 1e-10
    assert torch.allclose(a.interpolate(b, 0.0).matrix, a.matrix)
    assert torch.allclose(a.interpolate(b, 1.0).matrix, b.matrix, atol=1e-12)


def test_interpolation_refuses_extrapolation_and_wrong_frames() -> None:
    a = Pose.identity("head", "tracker", units="mm")
    b = Pose.from_twist(_twist(23, 0.4), "head", "tracker", units="mm")
    with pytest.raises(TransformError):
        a.interpolate(b, 1.5)
    c = Pose.from_twist(_twist(24, 0.4), "head", "coil", units="mm")
    with pytest.raises(FrameMismatchError):
        a.interpolate(c, 0.5)


def test_compose_all_matches_pairwise_composition() -> None:
    ps = [
        Pose.from_twist(_twist(30 + i, 0.2), f"f{i}", f"f{i + 1}", units="mm")
        for i in range(4)
    ]
    manual = ps[0].compose(ps[1]).compose(ps[2]).compose(ps[3])
    assert torch.allclose(compose_all(ps).matrix, manual.matrix)
    assert compose_all(ps).label == "f0<-f4"


def test_apply_maps_child_points_into_parent() -> None:
    p = Pose.from_Rt(exp_so3(torch.tensor([0.0, 0.0, math.pi / 2], dtype=DTYPE)),
                     torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE), "a", "b", units="mm")
    out = p.apply(torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE))
    assert torch.allclose(out, torch.tensor([[1.0, 3.0, 3.0]], dtype=DTYPE), atol=1e-12)
