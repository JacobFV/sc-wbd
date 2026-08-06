"""Refusals for the transform / frame / clock runtime.

Every refusal in this package is a real exception with a real reason string,
a remedy, and the offending object.  The compiler refusal codes referenced
here are those of ``paper/thesis_contract.tex`` Table ``tab:compiler-refusals``
(mirrored in ``ARCHITECTURE.md`` §2):

======  =====================================================================
 code    rejected configuration
======  =====================================================================
 R01     unknown units / clock / support / frame / handedness / transform lineage
 R02     prolongation without declared restriction partner + tested coverage
 R03     global cross-scale state when overlap/cocycle residual exceeds tolerance
======  =====================================================================

R04..R11 are not this module's business.

Design rule (ARCHITECTURE.md §7): *numerical* compatibility is never allowed to
stand in for *measurement* compatibility.  A silent unit conversion, a silent
handedness flip, or a silently reused expired calibration is therefore a bug,
not a convenience.
"""

from __future__ import annotations

from typing import Any


class TransformError(Exception):
    """Base class for every refusal raised by ``scwbd.transforms``.

    Parameters
    ----------
    reason:
        Human-readable statement of *what* was wrong.  Never empty.
    remedy:
        What the caller must supply or change to make the request admissible.
    offending_object:
        The object (frame id, edge, path, clock pair, section, ...) that
        triggered the refusal.  Kept so a compiler can attach it to a
        ``CompilerRefusal``.
    code:
        Refusal code from ``tab:compiler-refusals`` when one applies.
    """

    code: str | None = None

    def __init__(
        self,
        reason: str,
        *,
        remedy: str = "",
        offending_object: Any = None,
        code: str | None = None,
    ) -> None:
        if not reason:
            raise ValueError("a refusal must carry a reason string")
        self.reason = reason
        self.remedy = remedy
        self.offending_object = offending_object
        if code is not None:
            self.code = code
        parts = [reason]
        if self.code:
            parts.insert(0, f"[{self.code}]")
        if remedy:
            parts.append(f"Remedy: {remedy}")
        if offending_object is not None:
            parts.append(f"Offending object: {offending_object!r}")
        super().__init__(" ".join(parts))

    def as_record(self) -> dict[str, Any]:
        """Machine-readable form, suitable for a claim report."""
        return {
            "code": self.code,
            "error": type(self).__name__,
            "reason": self.reason,
            "remedy": self.remedy,
            "offending_object": repr(self.offending_object),
        }


# --------------------------------------------------------------------------
# R01 family: unknown / mismatched units, frames, handedness, clocks, lineage
# --------------------------------------------------------------------------


class UnitError(TransformError):
    """Unknown or unregistered unit."""

    code = "R01"


class UnitMismatchError(UnitError):
    """Two quantities in different units met without an explicit conversion.

    The runtime never converts silently: a millimetre pose composed with a
    metre pose is a measurement error, and converting it quietly would make the
    error unobservable downstream.
    """

    code = "R01"


class HandednessError(TransformError):
    """Handedness mismatch, or a reflection hiding inside a "rigid" transform.

    A left-handed frame and a right-handed frame are different physical
    conventions.  Composing them without a declared, explicit mirror operator
    silently flips left and right hemispheres.
    """

    code = "R01"


class FrameMismatchError(TransformError):
    """Composition attempted across a frame pair that does not chain."""

    code = "R01"


class UnknownFrameError(TransformError):
    """A frame id was referenced that the graph does not declare."""

    code = "R01"


class NoPathError(TransformError):
    """No admissible transform path connects the requested frames."""

    code = "R01"


class ClockRelationUnknownError(TransformError):
    """Two clocks were compared with no declared or derivable relation.

    Appendix C layer 4 requires a physical synchronization event or an
    independent cross-correlation target.  "Both were recorded on the same
    afternoon" is not a clock relation.
    """

    code = "R01"


class EpochMismatchError(TransformError):
    """Transform epochs (session/acquisition identities) do not agree."""

    code = "R01"


# --------------------------------------------------------------------------
# Invertibility / validity
# --------------------------------------------------------------------------


class NonInvertibleTransformError(TransformError):
    """An inverse was requested where none exists or none was declared.

    Appendix C layer 2: "No inverse assumed where absent."  A deformable warp
    with a folded Jacobian has no well-defined inverse; a rank-deficient affine
    has none either.
    """

    code = "R01"


class CalibrationExpiredError(TransformError):
    """A calibration was used outside its declared validity interval.

    Appendix C layer 9: "Do not reuse past validity interval silently; inflate
    uncertainty or require recalibration when device, participant geometry or
    environment changes."  Under ``ExpiryPolicy.REFUSE`` this is raised; under
    ``ExpiryPolicy.INFLATE`` the path is returned with visibly inflated
    uncertainty and an ``inflated`` provenance flag -- never silently.
    """

    code = "R01"


class ValidityIntervalError(TransformError):
    """Validity intervals of composed edges do not overlap at the query time."""

    code = "R01"


# --------------------------------------------------------------------------
# Uncertainty propagation
# --------------------------------------------------------------------------


class LinearizationInvalidError(TransformError):
    """First-order (T5) propagation was requested for a map where it is invalid.

    Appendix C: "Monte Carlo or interval propagation replaces this
    approximation for nonlinear registration, thresholded tractography,
    acoustic focusing and other non-Gaussian maps."
    """


class CovarianceError(TransformError):
    """A covariance argument was not a valid covariance (shape/symmetry/PSD)."""


# --------------------------------------------------------------------------
# R02 / R03: the resolution poset
# --------------------------------------------------------------------------


class ProlongationWithoutRestrictionError(TransformError):
    """R02 -- prolongation lacking a declared restriction partner or coverage.

    A coarse observation would otherwise be converted into unsupported fine
    structure.  Provide paired maps, round-trip and held-out landmark tests,
    and an out-of-support uncertainty policy; otherwise return a distribution
    or refuse.
    """

    code = "R02"


class CocycleObstructionError(TransformError):
    """R03 -- overlap/cocycle residual exceeds tolerance; no global section.

    Carries the :class:`~scwbd.transforms.sheaf.ObstructionCertificate` that
    names the failed path.
    """

    code = "R03"

    def __init__(self, reason: str, *, certificate: Any = None, **kw: Any) -> None:
        self.certificate = certificate
        kw.setdefault("offending_object", certificate)
        kw.setdefault(
            "remedy",
            "Preserve the separate sections, branch or marginalize the "
            "posterior, and ship the obstruction certificate. Do not "
            "materialize a global raster.",
        )
        super().__init__(reason, **kw)


class SiteError(TransformError):
    """The declared site / cover is malformed (not a cover, missing rho, ...)."""

    code = "R01"


__all__ = [
    "TransformError",
    "UnitError",
    "UnitMismatchError",
    "HandednessError",
    "FrameMismatchError",
    "UnknownFrameError",
    "NoPathError",
    "ClockRelationUnknownError",
    "EpochMismatchError",
    "NonInvertibleTransformError",
    "CalibrationExpiredError",
    "ValidityIntervalError",
    "LinearizationInvalidError",
    "CovarianceError",
    "ProlongationWithoutRestrictionError",
    "CocycleObstructionError",
    "SiteError",
]
