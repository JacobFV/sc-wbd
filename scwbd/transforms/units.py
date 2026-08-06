"""Strict unit and handedness typing for the transform runtime.

Appendix C layer 1 requires every frame to declare *object, origin, axes,
handedness, units, validity interval*, with an "explicit distinction between
continuous mm, indices, normalized image coordinates and angular gaze".

The registry below is deliberately small and deliberately *strict*:

* Unknown unit  -> :class:`~scwbd.transforms.errors.UnitError` (R01).
* Mismatched units -> :class:`UnitMismatchError` (R01).  There is **no**
  automatic conversion anywhere in this package.  If a caller genuinely wants
  millimetres expressed in metres they call :func:`convert` and the conversion
  becomes a visible, provenance-carrying act.
* Dimensionally incompatible units (``mm`` vs ``voxel``) cannot be converted at
  all without a frame-specific scale, so :func:`convert` refuses them.

``voxel``, ``index`` and ``normalized`` are *not* lengths.  Voxel indices become
millimetres only through an affine edge in the frame graph that carries the
voxel size; that is the whole point of the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .errors import HandednessError, UnitError, UnitMismatchError


class Dimension(str, Enum):
    LENGTH = "length"
    TIME = "time"
    ANGLE = "angle"
    VOLTAGE = "voltage"
    MAGNETIC_FLUX_DENSITY = "magnetic_flux_density"
    PRESSURE = "pressure"
    CURRENT_DENSITY = "current_density"
    FREQUENCY = "frequency"
    INDEX = "index"  # dimensionless but *not* interconvertible with lengths
    NORMALIZED = "normalized"
    DIMENSIONLESS = "dimensionless"


@dataclass(frozen=True)
class UnitInfo:
    symbol: str
    dimension: Dimension
    to_si: float  # multiply a value in this unit by ``to_si`` to get SI


_REGISTRY: Final[dict[str, UnitInfo]] = {
    u.symbol: u
    for u in [
        UnitInfo("m", Dimension.LENGTH, 1.0),
        UnitInfo("cm", Dimension.LENGTH, 1e-2),
        UnitInfo("mm", Dimension.LENGTH, 1e-3),
        UnitInfo("um", Dimension.LENGTH, 1e-6),
        UnitInfo("s", Dimension.TIME, 1.0),
        UnitInfo("ms", Dimension.TIME, 1e-3),
        UnitInfo("us", Dimension.TIME, 1e-6),
        UnitInfo("Hz", Dimension.FREQUENCY, 1.0),
        UnitInfo("rad", Dimension.ANGLE, 1.0),
        UnitInfo("deg", Dimension.ANGLE, 3.141592653589793 / 180.0),
        UnitInfo("V", Dimension.VOLTAGE, 1.0),
        UnitInfo("uV", Dimension.VOLTAGE, 1e-6),
        UnitInfo("T", Dimension.MAGNETIC_FLUX_DENSITY, 1.0),
        UnitInfo("fT", Dimension.MAGNETIC_FLUX_DENSITY, 1e-15),
        UnitInfo("Pa", Dimension.PRESSURE, 1.0),
        UnitInfo("A/m^2", Dimension.CURRENT_DENSITY, 1.0),
        UnitInfo("voxel", Dimension.INDEX, 1.0),
        UnitInfo("index", Dimension.INDEX, 1.0),
        UnitInfo("normalized", Dimension.NORMALIZED, 1.0),
        UnitInfo("dimensionless", Dimension.DIMENSIONLESS, 1.0),
    ]
}

LENGTH_UNITS: Final[frozenset[str]] = frozenset(
    s for s, i in _REGISTRY.items() if i.dimension is Dimension.LENGTH
)


def unit_info(symbol: str) -> UnitInfo:
    """Look up a unit, refusing unknown symbols (R01)."""
    try:
        return _REGISTRY[symbol]
    except KeyError:
        raise UnitError(
            f"unit {symbol!r} is not in the unit registry",
            remedy=(
                "Declare the unit in scwbd.transforms.units._REGISTRY with its "
                "dimension and SI factor, or express the quantity in a "
                f"registered unit ({sorted(_REGISTRY)})."
            ),
            offending_object=symbol,
        ) from None


def dimension_of(symbol: str) -> Dimension:
    return unit_info(symbol).dimension


def require_same_unit(a: str, b: str, *, context: str) -> str:
    """Refuse unless ``a`` and ``b`` are the *same* unit. Never converts.

    Returns the common unit so call sites read as an assertion-with-value.
    """
    ia, ib = unit_info(a), unit_info(b)
    if ia.symbol != ib.symbol:
        if ia.dimension is ib.dimension:
            detail = (
                f"same dimension ({ia.dimension.value}) but different scale; "
                "an implicit conversion here would silently rescale geometry"
            )
        else:
            detail = (
                f"incompatible dimensions {ia.dimension.value} vs "
                f"{ib.dimension.value}"
            )
        raise UnitMismatchError(
            f"unit mismatch in {context}: {a!r} vs {b!r} -- {detail}",
            remedy=(
                "Call scwbd.transforms.units.convert() explicitly (and record "
                "it in provenance), or declare an affine edge in the frame "
                "graph that carries the scale."
            ),
            offending_object=(a, b),
        )
    return ia.symbol


def require_length_unit(symbol: str, *, context: str) -> str:
    info = unit_info(symbol)
    if info.dimension is not Dimension.LENGTH:
        raise UnitError(
            f"{context} requires a length unit, got {symbol!r} "
            f"(dimension {info.dimension.value})",
            remedy=(
                "Voxel indices and normalized image coordinates are not "
                "lengths. Insert the affine edge that carries voxel size / "
                "field of view before composing a metric pose."
            ),
            offending_object=symbol,
        )
    return info.symbol


def convert(value: float, frm: str, to: str) -> float:
    """Explicit, provenance-worthy unit conversion.

    Refuses across dimensions -- ``voxel`` -> ``mm`` needs a frame-graph edge,
    not a scalar.
    """
    ia, ib = unit_info(frm), unit_info(to)
    if ia.dimension is not ib.dimension:
        raise UnitMismatchError(
            f"cannot convert {frm!r} -> {to!r}: dimensions "
            f"{ia.dimension.value} vs {ib.dimension.value}",
            remedy=(
                "Index-like and normalized coordinates acquire physical units "
                "only through a declared transform edge carrying the sampling "
                "geometry."
            ),
            offending_object=(frm, to),
        )
    if ia.dimension in (Dimension.INDEX, Dimension.NORMALIZED):
        raise UnitMismatchError(
            f"refusing to rescale {ia.dimension.value} unit {frm!r} -> {to!r}",
            remedy="Use the frame-graph edge that declares the sampling grid.",
            offending_object=(frm, to),
        )
    return value * (ia.to_si / ib.to_si)


class Handedness(str, Enum):
    """Chirality of a coordinate frame's axis triple."""

    RIGHT = "right"
    LEFT = "left"

    @property
    def sign(self) -> float:
        """Expected sign of ``det`` for a proper transform *within* the frame."""
        return 1.0

    @classmethod
    def coerce(cls, value: "Handedness | str") -> "Handedness":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise HandednessError(
                f"unknown handedness {value!r}",
                remedy="Declare handedness as 'right' or 'left'.",
                offending_object=value,
            ) from None


def require_same_handedness(
    a: "Handedness | str", b: "Handedness | str", *, context: str
) -> Handedness:
    """Refuse a handedness mismatch. Never mirrors silently."""
    ha, hb = Handedness.coerce(a), Handedness.coerce(b)
    if ha is not hb:
        raise HandednessError(
            f"handedness mismatch in {context}: {ha.value} vs {hb.value}",
            remedy=(
                "Insert an explicit, declared mirror edge (kind='affine' with "
                "det<0 and reflection_declared=True) so the left/right flip is "
                "auditable, or fix the frame declaration. A silent flip "
                "swaps the hemispheres."
            ),
            offending_object=(ha.value, hb.value),
        )
    return ha


__all__ = [
    "Dimension",
    "UnitInfo",
    "Handedness",
    "LENGTH_UNITS",
    "unit_info",
    "dimension_of",
    "require_same_unit",
    "require_length_unit",
    "require_same_handedness",
    "convert",
]
