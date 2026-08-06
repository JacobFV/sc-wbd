"""SE(3) with a proper Lie algebra, plus the *measurement* type that surrounds it.

Thesis §2.8: "A location is not a three-number property of a brain.  It is a
point or pose in a named reference frame, measured through a calibration chain
at a particular session and time."  A bare 4x4 matrix is therefore never enough:
:class:`Pose` carries the frame pair ``(parent, child)``, its length unit, the
handedness of the frames it relates, a validity interval, an epoch, and
provenance.

Conventions (fixed here, used everywhere in ``scwbd.transforms``)
----------------------------------------------------------------
* A pose ``T`` with ``parent="a"``, ``child="b"`` is written ``T^{a<-b}``: it
  maps coordinates *expressed in b* to coordinates *expressed in a*.
  ``a.compose(b)`` therefore requires ``a.child == b.parent``, matching
  equation (3) of §2.8, which reads left-to-right as
  ``T^{atlas<-device} = T^{atlas<-image} T^{image<-head} T^{head<-tracker}
  T^{tracker<-device}``.
* Twists are ordered ``xi = (rho, phi)`` -- translation part first, rotation
  part second (Barfoot's convention), so
  ``Ad(T) = [[R, hat(t) R], [0, R]]``.
* Perturbations are **right** perturbations: ``T = T_bar exp(xi^)``.  Covariance
  and bias twists attached to a :class:`Pose` are in that convention, in the
  pose's own units for the translation block and radians for the rotation block.
* Internal dtype is ``float64``.  ARCHITECTURE.md §3 forbids reduced precision
  in covariance propagation, and Lie-algebra logs near theta ~ pi are the
  classic place where float32 quietly loses three digits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import torch

from .errors import (
    FrameMismatchError,
    HandednessError,
    NonInvertibleTransformError,
    TransformError,
    ValidityIntervalError,
)
from .units import Handedness, require_length_unit, require_same_handedness, require_same_unit

DTYPE = torch.float64
_EPS = 1e-10


# --------------------------------------------------------------------------
# validity intervals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityInterval:
    """Half-open validity interval ``[start, end)`` in seconds on a named clock.

    Appendix C layer 1 (frames) and layer 9 (calibration validity) both require
    one.  ``None`` bounds mean unbounded on that side; ``clock`` names the clock
    the bounds are expressed on so that "valid until 12:04" cannot silently mean
    a different device's 12:04.
    """

    start: float | None = None
    end: float | None = None
    clock: str = "wall"

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValidityIntervalError(
                f"validity interval end {self.end} precedes start {self.start}",
                remedy="Fix the calibration record's interval.",
                offending_object=self,
            )

    @staticmethod
    def unbounded(clock: str = "wall") -> "ValidityInterval":
        return ValidityInterval(None, None, clock)

    def contains(self, t: float) -> bool:
        if self.start is not None and t < self.start:
            return False
        if self.end is not None and t >= self.end:
            return False
        return True

    def extrapolation_distance(self, t: float) -> float:
        """Seconds by which ``t`` falls outside the interval (0.0 if inside)."""
        if self.start is not None and t < self.start:
            return float(self.start - t)
        if self.end is not None and t >= self.end:
            return float(t - self.end)
        return 0.0

    def intersect(self, other: "ValidityInterval") -> "ValidityInterval":
        if self.clock != other.clock:
            raise ValidityIntervalError(
                "cannot intersect validity intervals stated on different clocks "
                f"({self.clock!r} vs {other.clock!r})",
                remedy=(
                    "Align the intervals through the clock graph "
                    "(scwbd.transforms.clock_graph.ClockGraph.align) first."
                ),
                offending_object=(self, other),
            )
        start = _max_opt(self.start, other.start)
        end = _min_opt(self.end, other.end)
        if start is not None and end is not None and end < start:
            raise ValidityIntervalError(
                f"validity intervals do not overlap: {self} and {other}",
                remedy=(
                    "Recalibrate, or restrict the query to a time covered by "
                    "every edge on the path."
                ),
                offending_object=(self, other),
            )
        return ValidityInterval(start, end, self.clock)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lo = "-inf" if self.start is None else f"{self.start:g}"
        hi = "+inf" if self.end is None else f"{self.end:g}"
        return f"[{lo}, {hi}) on {self.clock}"


def _max_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# --------------------------------------------------------------------------
# so(3) / se(3) primitives
# --------------------------------------------------------------------------


def as_tensor(x: Any, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=DTYPE)
    if shape is not None and tuple(t.shape) != shape:
        raise TransformError(
            f"expected shape {shape}, got {tuple(t.shape)}",
            remedy="Check the argument's layout.",
            offending_object=tuple(t.shape),
        )
    return t


def hat(v: torch.Tensor) -> torch.Tensor:
    """so(3) hat: R^3 -> skew-symmetric 3x3."""
    v = as_tensor(v, shape=(3,))
    z = torch.zeros((), dtype=DTYPE)
    return torch.stack(
        [
            torch.stack([z, -v[2], v[1]]),
            torch.stack([v[2], z, -v[0]]),
            torch.stack([-v[1], v[0], z]),
        ]
    )


def vee(S: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`hat` (uses the antisymmetric part)."""
    S = as_tensor(S, shape=(3, 3))
    A = 0.5 * (S - S.T)
    return torch.stack([A[2, 1], A[0, 2], A[1, 0]])


def exp_so3(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, Taylor-stable at ``|phi| -> 0``."""
    phi = as_tensor(phi, shape=(3,))
    theta = torch.linalg.norm(phi)
    K = hat(phi)
    I = torch.eye(3, dtype=DTYPE)
    if float(theta) < 1e-8:
        # I + K + K^2/2 (second order); error O(theta^3)
        return I + K + 0.5 * (K @ K)
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / (theta * theta)
    return I + a * K + b * (K @ K)


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """so(3) log, robust at ``theta -> 0`` and ``theta -> pi``.

    ``theta`` comes from ``atan2(s, c)`` with ``s = ||vee(R - R^T)||/2`` measured
    directly rather than from ``acos((tr-1)/2)``.  Near ``theta = pi`` the
    ``acos`` route is ill-conditioned -- it costs four digits at
    ``theta = pi - 1e-6`` -- and pose logs near half-turns are exactly what a
    coil flipped end-for-end produces.
    """
    R = as_tensor(R, shape=(3, 3))
    w = vee(R - R.T)  # = 2 sin(theta) * axis
    s = float(torch.linalg.norm(w)) / 2.0
    c = float(R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = math.atan2(s, max(-1.0, min(1.0, c)))
    if s > 1e-8:
        return (theta / (2.0 * s)) * w
    if c > 0.0:
        # theta ~ 0: log(R) is the antisymmetric part to second order
        return 0.5 * w
    # theta ~ pi: R is symmetric to leading order; recover the axis from
    # R + I = 2 a a^T (up to sign, resolved by the antisymmetric remainder).
    M = R + torch.eye(3, dtype=DTYPE)
    idx = int(torch.argmax(torch.diagonal(M)))
    axis = M[:, idx]
    n = torch.linalg.norm(axis)
    if float(n) < _EPS:  # pragma: no cover - degenerate, R = -I impossible in SO(3)
        raise TransformError(
            "cannot recover rotation axis at theta = pi",
            remedy="Re-orthonormalize the rotation before taking a log.",
            offending_object=R,
        )
    axis = axis / n
    # sign disambiguation via the (small) antisymmetric part
    if float(torch.dot(w, axis)) < 0:
        axis = -axis
    return theta * axis


def left_jacobian_so3(phi: torch.Tensor) -> torch.Tensor:
    """``J_l(phi)`` with ``exp((phi+dphi)^) ~ exp(J_l dphi ^) exp(phi^)``."""
    phi = as_tensor(phi, shape=(3,))
    theta = float(torch.linalg.norm(phi))
    K = hat(phi)
    I = torch.eye(3, dtype=DTYPE)
    if theta < 1e-8:
        return I + 0.5 * K + (1.0 / 6.0) * (K @ K)
    a = (1.0 - math.cos(theta)) / (theta * theta)
    b = (theta - math.sin(theta)) / (theta**3)
    return I + a * K + b * (K @ K)


def inv_left_jacobian_so3(phi: torch.Tensor) -> torch.Tensor:
    phi = as_tensor(phi, shape=(3,))
    theta = float(torch.linalg.norm(phi))
    K = hat(phi)
    I = torch.eye(3, dtype=DTYPE)
    if theta < 1e-8:
        return I - 0.5 * K + (1.0 / 12.0) * (K @ K)
    # (1 + cos t) / (2 t sin t) == cos(t/2) / (2 t sin(t/2)), which stays
    # well conditioned as t -> pi (where the first form is 0/0 in floating point)
    half = 0.5 * theta
    c = (1.0 / (theta * theta)) - math.cos(half) / (2.0 * theta * math.sin(half))
    return I - 0.5 * K + c * (K @ K)


def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """``exp`` of a twist ``xi = (rho, phi)`` -> 4x4 homogeneous matrix."""
    xi = as_tensor(xi, shape=(6,))
    rho, phi = xi[:3], xi[3:]
    R = exp_so3(phi)
    t = left_jacobian_so3(phi) @ rho
    T = torch.eye(4, dtype=DTYPE)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_se3(T: torch.Tensor) -> torch.Tensor:
    """``log`` of a 4x4 rigid matrix -> twist ``(rho, phi)``."""
    T = as_tensor(T, shape=(4, 4))
    phi = log_so3(T[:3, :3])
    rho = inv_left_jacobian_so3(phi) @ T[:3, 3]
    return torch.cat([rho, phi])


def adjoint(T: torch.Tensor) -> torch.Tensor:
    """``Ad(T)`` for the ``(rho, phi)`` twist ordering.

    Satisfies ``exp((Ad(T) xi)^) = T exp(xi^) T^{-1}``.
    """
    T = as_tensor(T, shape=(4, 4))
    R = T[:3, :3]
    t = T[:3, 3]
    A = torch.zeros((6, 6), dtype=DTYPE)
    A[:3, :3] = R
    A[:3, 3:] = hat(t) @ R
    A[3:, 3:] = R
    return A


def left_jacobian_se3(xi: torch.Tensor) -> torch.Tensor:
    """Barfoot's SE(3) left Jacobian ``[[Jl, Q], [0, Jl]]``.

    Used when a covariance stated on ``xi`` must be moved onto the group (and
    back), e.g. when fusing a fitted registration's parameter covariance with a
    chain of poses.
    """
    xi = as_tensor(xi, shape=(6,))
    rho, phi = xi[:3], xi[3:]
    Jl = left_jacobian_so3(phi)
    Q = _Q_matrix(rho, phi)
    J = torch.zeros((6, 6), dtype=DTYPE)
    J[:3, :3] = Jl
    J[:3, 3:] = Q
    J[3:, 3:] = Jl
    return J


def _Q_matrix(rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    theta = float(torch.linalg.norm(phi))
    P = hat(rho)
    if theta < 1e-8:
        return 0.5 * P
    F = hat(phi)
    t2, t3, t4, t5 = theta**2, theta**3, theta**4, theta**5
    st, ct = math.sin(theta), math.cos(theta)
    c1 = (theta - st) / t3
    c2 = (1.0 - 0.5 * t2 - ct) / t4
    c3 = 0.5 * (c2 - 3.0 * (theta - st - t3 / 6.0) / t5)
    return (
        0.5 * P
        + c1 * (F @ P + P @ F + F @ P @ F)
        - c2 * (F @ F @ P + P @ F @ F - 3.0 * F @ P @ F)
        - c3 * (F @ P @ F @ F + F @ F @ P @ F)
    )


# --------------------------------------------------------------------------
# validation of a candidate rigid matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidCheck:
    """Report of the numerical rigidity checks run on a 4x4 matrix."""

    orthonormality_residual: float
    determinant: float
    bottom_row_residual: float
    is_reflection: bool


def check_rigid(T: torch.Tensor, *, tol: float = 1e-8) -> RigidCheck:
    """Measure (never assume) orthonormality, ``det`` sign, bottom row."""
    T = as_tensor(T, shape=(4, 4))
    R = T[:3, :3]
    I = torch.eye(3, dtype=DTYPE)
    ortho = float(torch.linalg.norm(R.T @ R - I))
    det = float(torch.linalg.det(R))
    bottom = float(torch.linalg.norm(T[3, :] - torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE)))
    return RigidCheck(ortho, det, bottom, det < 0.0)


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A rigid transform ``T^{parent<-child}`` with its measurement type.

    Attributes
    ----------
    matrix:
        4x4 homogeneous, ``float64``.
    parent, child:
        Frame ids.  ``matrix`` maps child coordinates into parent coordinates.
    units:
        Length unit of the translation block (``"m"``, ``"mm"``, ...).  Both
        frames must use it; a unit change is an explicit affine edge, not an
        attribute of a rigid pose.
    handedness:
        Chirality shared by both frames.
    validity:
        When this pose may be used.
    epoch:
        Session/acquisition identity, e.g. ``"sub-01_ses-02"``.  Two poses from
        different epochs do not compose: head-to-tracker from Tuesday is not
        head-to-tracker from Wednesday.
    provenance:
        Free-form record (fit method, residuals, operator, software version).
    """

    matrix: torch.Tensor
    parent: str
    child: str
    units: str = "m"
    handedness: Handedness = Handedness.RIGHT
    validity: ValidityInterval = field(default_factory=ValidityInterval.unbounded)
    epoch: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    rigid_tol: float = 1e-7

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", as_tensor(self.matrix, shape=(4, 4)))
        object.__setattr__(self, "handedness", Handedness.coerce(self.handedness))
        require_length_unit(self.units, context=f"pose {self.label}")
        if self.parent == self.child:
            raise FrameMismatchError(
                f"pose {self.label} relates a frame to itself",
                remedy="A self-edge carries no calibration; remove it.",
                offending_object=self.label,
            )
        chk = check_rigid(self.matrix)
        if chk.bottom_row_residual > 1e-9:
            raise TransformError(
                f"pose {self.label} has a non-affine bottom row "
                f"(residual {chk.bottom_row_residual:.3e})",
                remedy="A projective matrix is not a pose; use an affine/deformable edge.",
                offending_object=self.label,
            )
        if chk.orthonormality_residual > self.rigid_tol:
            raise TransformError(
                f"pose {self.label} rotation block is not orthonormal "
                f"(||R^T R - I|| = {chk.orthonormality_residual:.3e} > {self.rigid_tol:.1e})",
                remedy=(
                    "Re-orthonormalize explicitly (Pose.orthonormalized) and record "
                    "the correction, or declare the edge as affine."
                ),
                offending_object=self.label,
            )
        # determinant sign check: a rigid pose is proper. det = -1 is a
        # reflection, which flips left/right and must never be admitted here.
        if chk.is_reflection:
            raise HandednessError(
                f"pose {self.label} has det(R) = {chk.determinant:.6f} < 0: "
                "this is a reflection, not a rotation",
                remedy=(
                    "If a mirror is genuinely intended, declare an affine edge "
                    "with reflection_declared=True so it appears in provenance. "
                    "Otherwise fix the axis convention (a swapped L/R label is "
                    "the usual cause)."
                ),
                offending_object=self.label,
            )
        if abs(chk.determinant - 1.0) > max(self.rigid_tol, 1e-7):
            raise TransformError(
                f"pose {self.label} has det(R) = {chk.determinant:.9f} != 1 "
                "(scaling hidden in a rigid transform)",
                remedy="Declare a similarity/affine edge if a scale is intended.",
                offending_object=self.label,
            )

    @property
    def label(self) -> str:
        return f"{self.parent}<-{self.child}"

    @staticmethod
    def identity(parent: str, child: str, **kw: Any) -> "Pose":
        return Pose(torch.eye(4, dtype=DTYPE), parent, child, **kw)

    @staticmethod
    def from_Rt(R: Any, t: Any, parent: str, child: str, **kw: Any) -> "Pose":
        T = torch.eye(4, dtype=DTYPE)
        T[:3, :3] = as_tensor(R, shape=(3, 3))
        T[:3, 3] = as_tensor(t, shape=(3,))
        return Pose(T, parent, child, **kw)

    @staticmethod
    def from_twist(xi: Any, parent: str, child: str, **kw: Any) -> "Pose":
        return Pose(exp_se3(as_tensor(xi, shape=(6,))), parent, child, **kw)

    def orthonormalized(self) -> "Pose":
        """Project the rotation block back onto SO(3) and record the correction."""
        R = self.matrix[:3, :3]
        U, _, Vh = torch.linalg.svd(R)
        Rp = U @ Vh
        if float(torch.linalg.det(Rp)) < 0:
            U = U.clone()
            U[:, -1] = -U[:, -1]
            Rp = U @ Vh
        T = self.matrix.clone()
        T[:3, :3] = Rp
        prov = dict(self.provenance)
        prov["orthonormalization_correction"] = float(torch.linalg.norm(Rp - R))
        return replace(self, matrix=T, provenance=prov)

    # -- algebra -----------------------------------------------------------

    @property
    def R(self) -> torch.Tensor:
        return self.matrix[:3, :3]

    @property
    def t(self) -> torch.Tensor:
        return self.matrix[:3, 3]

    def log(self) -> torch.Tensor:
        return log_se3(self.matrix)

    def adjoint(self) -> torch.Tensor:
        return adjoint(self.matrix)

    def compose(self, other: "Pose", *, at: float | None = None) -> "Pose":
        """``self @ other`` = ``T^{a<-b} T^{b<-c}`` = ``T^{a<-c}``.

        Refuses on frame, unit, handedness or epoch mismatch, and intersects
        validity intervals (raising when the intersection is empty).
        """
        if self.child != other.parent:
            raise FrameMismatchError(
                f"cannot compose {self.label} with {other.label}: "
                f"{self.child!r} != {other.parent!r}",
                remedy=(
                    "Compose along a chain a<-b, b<-c; invert an edge explicitly "
                    "if you need the other direction."
                ),
                offending_object=(self.label, other.label),
            )
        units = require_same_unit(
            self.units, other.units, context=f"composition {self.label} @ {other.label}"
        )
        hand = require_same_handedness(
            self.handedness,
            other.handedness,
            context=f"composition {self.label} @ {other.label}",
        )
        epoch = _require_same_epoch(self, other)
        validity = self.validity.intersect(other.validity)
        if at is not None and not validity.contains(at):
            raise ValidityIntervalError(
                f"composition {self.label} @ {other.label} is not valid at t={at}: "
                f"joint validity {validity}",
                remedy="Recalibrate, or query inside the joint validity interval.",
                offending_object=(self.label, other.label, at),
            )
        return Pose(
            self.matrix @ other.matrix,
            self.parent,
            other.child,
            units=units,
            handedness=hand,
            validity=validity,
            epoch=epoch,
            provenance={
                "composed_from": [self.label, other.label],
                "components": [self.provenance, other.provenance],
            },
        )

    def __matmul__(self, other: "Pose") -> "Pose":
        return self.compose(other)

    def inverse(self) -> "Pose":
        """Exact rigid inverse, after re-checking invertibility numerically."""
        R = self.R
        det = float(torch.linalg.det(R))
        if abs(det) < 1e-9:
            raise NonInvertibleTransformError(
                f"pose {self.label} is singular (det = {det:.3e})",
                remedy="Do not invert a rank-deficient transform; declare the "
                "forward direction only.",
                offending_object=self.label,
            )
        T = torch.eye(4, dtype=DTYPE)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ self.t
        return Pose(
            T,
            self.child,
            self.parent,
            units=self.units,
            handedness=self.handedness,
            validity=self.validity,
            epoch=self.epoch,
            provenance={"inverse_of": self.label, "source": self.provenance},
        )

    def apply(self, points: Any) -> torch.Tensor:
        """Map ``(..., 3)`` points from the child frame into the parent frame."""
        p = torch.as_tensor(points, dtype=DTYPE)
        if p.shape[-1] != 3:
            raise TransformError(
                f"expected trailing dimension 3, got {tuple(p.shape)}",
                remedy="Pass Cartesian points in the child frame.",
                offending_object=tuple(p.shape),
            )
        return p @ self.R.T + self.t

    def interpolate(self, other: "Pose", alpha: float) -> "Pose":
        """Geodesic (screw) interpolation ``T1 exp(alpha log(T1^-1 T2))``.

        Both poses must relate the *same* frame pair -- interpolating between
        ``head<-tracker`` and ``atlas<-image`` is meaningless.
        """
        if (self.parent, self.child) != (other.parent, other.child):
            raise FrameMismatchError(
                f"cannot interpolate {self.label} with {other.label}: different frame pair",
                remedy="Interpolate two measurements of the same frame pair.",
                offending_object=(self.label, other.label),
            )
        require_same_unit(self.units, other.units, context="interpolation")
        require_same_handedness(self.handedness, other.handedness, context="interpolation")
        if not 0.0 <= alpha <= 1.0:
            raise TransformError(
                f"interpolation parameter alpha={alpha} outside [0, 1]",
                remedy=(
                    "Extrapolating a pose beyond its measured endpoints needs a "
                    "motion model and an inflated ledger; do it explicitly."
                ),
                offending_object=alpha,
            )
        rel = log_se3(torch.linalg.inv(self.matrix) @ other.matrix)
        return Pose(
            self.matrix @ exp_se3(alpha * rel),
            self.parent,
            self.child,
            units=self.units,
            handedness=self.handedness,
            validity=self.validity.intersect(other.validity),
            epoch=self.epoch if self.epoch == other.epoch else None,
            provenance={"interpolated": [self.label, alpha]},
        )

    # -- residuals ---------------------------------------------------------

    def residual_twist(self, other: "Pose") -> torch.Tensor:
        """``log(self^-1 other)`` -- the disagreement between two estimates."""
        if (self.parent, self.child) != (other.parent, other.child):
            raise FrameMismatchError(
                f"residual between {self.label} and {other.label} is not defined",
                remedy="Compare two estimates of the same frame pair.",
                offending_object=(self.label, other.label),
            )
        require_same_unit(self.units, other.units, context="residual")
        return log_se3(torch.linalg.inv(self.matrix) @ other.matrix)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        t = [f"{float(v):.4g}" for v in self.t]
        ang = float(torch.linalg.norm(log_so3(self.R)))
        return (
            f"Pose({self.label}, t=[{', '.join(t)}] {self.units}, "
            f"angle={math.degrees(ang):.3f}deg, {self.handedness.value}-handed, "
            f"epoch={self.epoch}, validity={self.validity})"
        )


def _require_same_epoch(a: Pose, b: Pose) -> str | None:
    from .errors import EpochMismatchError

    if a.epoch is None:
        return b.epoch
    if b.epoch is None:
        return a.epoch
    if a.epoch != b.epoch:
        raise EpochMismatchError(
            f"epoch mismatch composing {a.label} ({a.epoch}) with {b.label} ({b.epoch})",
            remedy=(
                "Session-specific calibrations do not transfer between sessions. "
                "Re-measure, or declare the edge as epoch-invariant (epoch=None) "
                "with the evidence that justifies it."
            ),
            offending_object=(a.epoch, b.epoch),
        )
    return a.epoch


def compose_all(poses: Sequence[Pose], *, at: float | None = None) -> Pose:
    """Left-to-right composition of a chain, e.g. equation (3) of §2.8."""
    if not poses:
        raise TransformError(
            "cannot compose an empty chain",
            remedy="Supply at least one pose.",
            offending_object=poses,
        )
    out = poses[0]
    for p in poses[1:]:
        out = out.compose(p, at=at)
    return out


def round_trip_residual(forward: Pose, backward: Pose) -> dict[str, float]:
    """Measure ``a->b->a``.  Never assumed zero (Appendix C layer 1).

    ``backward`` may be an independently measured reverse transform rather than
    ``forward.inverse()``; that is exactly the case worth measuring.

    The loop is deliberately *not* wrapped in a :class:`Pose`: a transform from
    a frame to itself is a residual, not a pose, and giving it the same type
    would invite it into a chain.
    """
    if forward.child != backward.parent or backward.child != forward.parent:
        raise FrameMismatchError(
            f"{forward.label} and {backward.label} do not form a round trip",
            remedy="Supply the a<-b and b<-a transforms.",
            offending_object=(forward.label, backward.label),
        )
    units = require_same_unit(forward.units, backward.units, context="round trip")
    require_same_handedness(forward.handedness, backward.handedness, context="round trip")
    xi = log_se3(forward.matrix @ backward.matrix)
    return {
        "translation": float(torch.linalg.norm(xi[:3])),
        "rotation_rad": float(torch.linalg.norm(xi[3:])),
        "rotation_deg": math.degrees(float(torch.linalg.norm(xi[3:]))),
        "twist_norm": float(torch.linalg.norm(xi)),
        "units": units,
    }


__all__ = [
    "DTYPE",
    "ValidityInterval",
    "Pose",
    "RigidCheck",
    "check_rigid",
    "hat",
    "vee",
    "exp_so3",
    "log_so3",
    "exp_se3",
    "log_se3",
    "adjoint",
    "left_jacobian_so3",
    "inv_left_jacobian_so3",
    "left_jacobian_se3",
    "compose_all",
    "round_trip_residual",
    "as_tensor",
]
