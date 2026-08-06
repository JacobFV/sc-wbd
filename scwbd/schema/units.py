"""Dimensional analysis for SC-WBD units.

ARCHITECTURE.md sec. 2 requires units *everywhere*, carried as ``Unit`` strings
"validated against a registry".  A string whitelist would accept ``"V*m"``
wherever ``"V/m"`` was meant, which is exactly the numerical-vs-measurement
compatibility confusion refusal R01 exists to prevent.  So this module
implements real dimensional analysis: unit expressions are parsed into a
7-tuple of rational exponents over the SI base dimensions plus a scale factor
to base SI, and comparison is by dimension, not by spelling.

    >>> Unit("V/m").dimension == Unit("V*m").dimension
    False
    >>> Unit("mV").to_base_factor
    0.001
    >>> (Unit("V") / Unit("m")).dimension == Unit("V/m").dimension
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

__all__ = [
    "BASE_DIMENSIONS",
    "Dimension",
    "Unit",
    "UnitError",
    "DIMENSIONLESS",
    "register_unit",
    "known_units",
]

BASE_DIMENSIONS: tuple[str, ...] = (
    "length",
    "mass",
    "time",
    "current",
    "temperature",
    "amount",
    "luminosity",
)

_ZERO = Fraction(0)


class UnitError(ValueError):
    """Raised when a unit expression cannot be parsed or is not registered."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """Rational exponents over :data:`BASE_DIMENSIONS`."""

    exponents: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise UnitError(
                f"dimension needs {len(BASE_DIMENSIONS)} exponents, "
                f"got {len(self.exponents)}"
            )

    @classmethod
    def base(cls, name: str, power: int | Fraction = 1) -> "Dimension":
        idx = BASE_DIMENSIONS.index(name)
        exps = [_ZERO] * len(BASE_DIMENSIONS)
        exps[idx] = Fraction(power)
        return cls(tuple(exps))

    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls(tuple(_ZERO for _ in BASE_DIMENSIONS))

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, n: int | Fraction) -> "Dimension":
        f = Fraction(n)
        return Dimension(tuple(a * f for a in self.exponents))

    @property
    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    def as_dict(self) -> dict[str, Fraction]:
        return {
            name: exp
            for name, exp in zip(BASE_DIMENSIONS, self.exponents)
            if exp != 0
        }

    def symbol(self) -> str:
        """Canonical base-dimension signature, e.g. ``L^1*M^1*T^-3*I^-1``."""
        letters = {
            "length": "L",
            "mass": "M",
            "time": "T",
            "current": "I",
            "temperature": "Theta",
            "amount": "N",
            "luminosity": "J",
        }
        parts = [
            f"{letters[name]}^{exp}"
            for name, exp in zip(BASE_DIMENSIONS, self.exponents)
            if exp != 0
        ]
        return "*".join(parts) if parts else "1"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.symbol()


DIMENSIONLESS = Dimension.dimensionless()

_L = Dimension.base("length")
_M = Dimension.base("mass")
_T = Dimension.base("time")
_I = Dimension.base("current")
_TH = Dimension.base("temperature")
_N = Dimension.base("amount")
_J = Dimension.base("luminosity")

# name -> (scale factor to base SI, dimension, allows SI prefixes)
_REGISTRY: dict[str, tuple[float, Dimension, bool]] = {}


def register_unit(
    name: str,
    scale: float,
    dimension: Dimension,
    *,
    prefixable: bool = True,
) -> None:
    """Add a named unit to the registry (used at import time and by agents)."""
    if not _IDENT_RE.fullmatch(name):
        raise UnitError(f"illegal unit name {name!r}")
    _REGISTRY[name] = (float(scale), dimension, prefixable)
    _PARSE_CACHE.clear()


def known_units() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


_PREFIXES: dict[str, float] = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,  # micro sign
    "μ": 1e-6,  # greek small mu
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
    "z": 1e-21,
    "y": 1e-24,
}

_IDENT_RE = re.compile(r"[A-Za-zµμ%][A-Za-z0-9µμ%_]*")
_NUM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_EXP_RE = re.compile(r"[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?")


def _seed_registry() -> None:
    # SI base units.
    for nm, dim in (
        ("m", _L),
        ("s", _T),
        ("A", _I),
        ("K", _TH),
        ("mol", _N),
        ("cd", _J),
    ):
        _REGISTRY[nm] = (1.0, dim, True)
    # kilogram is the odd one out: the prefix is part of the base unit name.
    _REGISTRY["kg"] = (1.0, _M, False)
    _REGISTRY["g"] = (1e-3, _M, True)

    derived: dict[str, tuple[float, Dimension]] = {
        "Hz": (1.0, _T ** -1),
        "N": (1.0, _M * _L * _T**-2),
        "Pa": (1.0, _M * _L**-1 * _T**-2),
        "J": (1.0, _M * _L**2 * _T**-2),
        "W": (1.0, _M * _L**2 * _T**-3),
        "C": (1.0, _I * _T),
        "V": (1.0, _M * _L**2 * _T**-3 * _I**-1),
        "F": (1.0, _M**-1 * _L**-2 * _T**4 * _I**2),
        "ohm": (1.0, _M * _L**2 * _T**-3 * _I**-2),
        "S": (1.0, _M**-1 * _L**-2 * _T**3 * _I**2),
        "Wb": (1.0, _M * _L**2 * _T**-2 * _I**-1),
        "T": (1.0, _M * _T**-2 * _I**-1),  # tesla
        "H": (1.0, _M * _L**2 * _T**-2 * _I**-2),
        "lm": (1.0, _J),
        "lx": (1.0, _J * _L**-2),
        "L": (1e-3, _L**3),
        "min": (60.0, _T),
        "hour": (3600.0, _T),
        "degC": (1.0, _TH),  # interval semantics only; offsets are not modeled
    }
    for nm, (scale, dim) in derived.items():
        _REGISTRY[nm] = (scale, dim, True)

    # Dimensionless / bookkeeping units.  ``au`` (arbitrary units) is admitted
    # but flagged: Appendix C lists PPG in arbitrary units, and a source card
    # that uses it must supply an amplitude calibration or be refused (R01).
    for nm in ("dimensionless", "unitless", "rad", "sr", "count", "index",
               "au", "adu", "percent", "prob", "zscore", "ratio"):
        _REGISTRY[nm] = (1.0, DIMENSIONLESS, False)
    _REGISTRY["%"] = (1e-2, DIMENSIONLESS, False)


_seed_registry()

#: Units whose numeric value has no physical scale and therefore cannot be
#: dimension-checked against an instrument. Compiler check R01 requires an
#: amplitude-calibration entry whenever one of these carries measured signal.
ARBITRARY_UNITS: frozenset[str] = frozenset({"au", "adu", "index", "zscore"})

_PARSE_CACHE: dict[str, tuple[float, Dimension]] = {}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _lookup(token: str) -> tuple[float, Dimension]:
    """Resolve a single identifier, trying exact match then SI prefixes."""
    if token in _REGISTRY:
        scale, dim, _ = _REGISTRY[token]
        return scale, dim
    for plen in (2, 1):
        if len(token) > plen:
            pfx, rest = token[:plen], token[plen:]
            if pfx in _PREFIXES and rest in _REGISTRY:
                scale, dim, prefixable = _REGISTRY[rest]
                if prefixable:
                    return scale * _PREFIXES[pfx], dim
    raise UnitError(
        f"unknown unit {token!r}; register it with scwbd.schema.units.register_unit "
        f"or use one of {', '.join(known_units()[:12])}, ..."
    )


class _Parser:
    """Recursive-descent parser for ``a*b/c^2`` style unit expressions."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, msg: str) -> UnitError:
        return UnitError(f"{msg} in unit expression {self.text!r} at {self.pos}")

    def peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def parse(self) -> tuple[float, Dimension]:
        if not self.text:
            raise self.error("empty unit")
        scale, dim = self.expr()
        if self.pos != len(self.text):
            raise self.error(f"unexpected {self.peek()!r}")
        return scale, dim

    def expr(self) -> tuple[float, Dimension]:
        scale, dim = self.term()
        while (c := self.peek()) in ("*", "/", "·"):
            self.pos += 1
            s2, d2 = self.term()
            if c == "/":
                scale, dim = scale / s2, dim / d2
            else:
                scale, dim = scale * s2, dim * d2
        return scale, dim

    def term(self) -> tuple[float, Dimension]:
        scale, dim = self.factor()
        if self.peek() == "^":
            self.pos += 1
            exp = self.exponent()
            scale, dim = scale ** float(exp), dim**exp
        return scale, dim

    def factor(self) -> tuple[float, Dimension]:
        c = self.peek()
        if c == "(":
            self.pos += 1
            scale, dim = self.expr()
            if self.peek() != ")":
                raise self.error("unbalanced parenthesis")
            self.pos += 1
            return scale, dim
        m = _NUM_RE.match(self.text, self.pos)
        if m and not _IDENT_RE.match(self.text, self.pos):
            self.pos = m.end()
            return float(m.group()), DIMENSIONLESS
        m = _IDENT_RE.match(self.text, self.pos)
        if not m:
            raise self.error(f"expected a unit name, found {c!r}")
        self.pos = m.end()
        return _lookup(m.group())

    def exponent(self) -> Fraction:
        m = _EXP_RE.match(self.text, self.pos)
        if not m:
            raise self.error("expected an exponent")
        self.pos = m.end()
        raw = m.group()
        if "/" in raw:
            num, den = raw.split("/")
            return Fraction(int(num), int(den))
        if "." in raw:
            return Fraction(raw).limit_denominator(10**6)
        return Fraction(int(raw))


def _parse(text: str) -> tuple[float, Dimension]:
    cached = _PARSE_CACHE.get(text)
    if cached is None:
        cached = _Parser(text).parse()
        _PARSE_CACHE[text] = cached
    return cached


def _normalize(raw: str) -> str:
    if not isinstance(raw, str):
        raise UnitError(f"unit must be a string, got {type(raw).__name__}")
    return "".join(raw.split())


# --------------------------------------------------------------------------
# Unit
# --------------------------------------------------------------------------


class Unit(str):
    """A dimensionally validated unit expression.

    ``Unit`` is a ``str`` subclass so that schemas serialize as plain strings,
    but it exposes real dimensional algebra.  Equality is *string* equality;
    physical compatibility is :meth:`same_dimension` / :meth:`compatible_with`.
    """

    __slots__ = ()

    def __new__(cls, value: Any) -> "Unit":
        text = _normalize(value)
        _parse(text)  # validate eagerly; raises UnitError
        return super().__new__(cls, text)

    # -- physical properties ------------------------------------------------
    @property
    def dimension(self) -> Dimension:
        return _parse(str(self))[1]

    @property
    def to_base_factor(self) -> float:
        """Multiplicative factor converting this unit to base SI."""
        return _parse(str(self))[0]

    @property
    def is_dimensionless(self) -> bool:
        return self.dimension.is_dimensionless

    @property
    def is_arbitrary(self) -> bool:
        """True when the expression mentions an uncalibrated arbitrary unit."""
        return any(tok in ARBITRARY_UNITS for tok in _IDENT_RE.findall(str(self)))

    def same_dimension(self, other: Any) -> bool:
        return self.dimension == Unit(other).dimension

    def compatible_with(self, other: Any) -> bool:
        """Alias of :meth:`same_dimension` (reads better at call sites)."""
        return self.same_dimension(other)

    def conversion_factor_to(self, other: Any) -> float:
        """Factor ``f`` with ``x_other = f * x_self``; raises on mismatch."""
        target = Unit(other)
        if not self.same_dimension(target):
            raise UnitError(
                f"cannot convert {str(self)!r} [{self.dimension.symbol()}] to "
                f"{str(target)!r} [{target.dimension.symbol()}]"
            )
        return self.to_base_factor / target.to_base_factor

    # -- algebra ------------------------------------------------------------
    @staticmethod
    def _wrap(text: str) -> str:
        return f"({text})" if any(c in text for c in "*/^") else text

    def __mul__(self, other: Any) -> "Unit":  # type: ignore[override]
        return Unit(f"{self._wrap(str(self))}*{self._wrap(str(Unit(other)))}")

    def __truediv__(self, other: Any) -> "Unit":
        return Unit(f"{self._wrap(str(self))}/{self._wrap(str(Unit(other)))}")

    def __pow__(self, n: int | Fraction) -> "Unit":
        f = Fraction(n)
        exp = f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        return Unit(f"{self._wrap(str(self))}^{exp}")

    def base_signature(self) -> str:
        """Dimension signature, stable across spellings of the same unit."""
        return self.dimension.symbol()

    def __repr__(self) -> str:
        return f"Unit({str(self)!r})"

    # -- pydantic -----------------------------------------------------------
    @classmethod
    def _validate(cls, value: Any) -> "Unit":
        return value if type(value) is cls else cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


def dimensions_of(units: Iterable[Any]) -> tuple[Dimension, ...]:
    return tuple(Unit(u).dimension for u in units)


def check_units(mapping: Mapping[str, Any], expected: Mapping[str, str]) -> list[str]:
    """Return a list of human-readable dimensional mismatches (empty = ok)."""
    problems: list[str] = []
    for key, want in expected.items():
        if key not in mapping:
            problems.append(f"missing unit declaration for {key!r}")
            continue
        got = Unit(mapping[key])
        if not got.same_dimension(want):
            problems.append(
                f"{key!r} declared {str(got)!r} [{got.dimension.symbol()}] "
                f"but must be dimensionally {want!r} [{Unit(want).dimension.symbol()}]"
            )
    return problems
