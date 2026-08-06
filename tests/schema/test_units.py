"""Unit dimensional analysis.

R01 exists so that "numerical compatibility" is never mistaken for
"measurement compatibility".  A string whitelist cannot tell ``V/m`` from
``V*m``; a dimension vector can.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from scwbd.schema import Unit, UnitError
from scwbd.schema.supports import TemporalSupport
from scwbd.schema.units import Dimension, known_units, register_unit


@pytest.mark.parametrize(
    "text,signature",
    [
        ("V", "L^2*M^1*T^-3*I^-1"),
        ("T", "M^1*T^-2*I^-1"),
        ("m", "L^1"),
        ("s", "T^1"),
        ("Hz", "T^-1"),
        ("A/m^2", "L^-2*I^1"),
        ("Pa", "L^-1*M^1*T^-2"),
        ("mol", "N^1"),
        ("K", "Theta^1"),
        ("dimensionless", "1"),
    ],
)
def test_registry_units_from_architecture(text: str, signature: str):
    """Every unit ARCHITECTURE.md sec. 2 names parses to the right dimension."""
    assert Unit(text).base_signature() == signature


def test_v_per_m_is_not_v_times_m():
    """The headline case: division and multiplication are not interchangeable."""
    per = Unit("V/m")
    times = Unit("V*m")
    assert per != times
    assert per.dimension != times.dimension
    assert not per.same_dimension(times)
    assert per.base_signature() == "L^1*M^1*T^-3*I^-1"
    assert times.base_signature() == "L^3*M^1*T^-3*I^-1"


def test_composites_agree_across_spellings():
    assert Unit("V/m").same_dimension("kg*m/(s^3*A)")
    assert Unit("Pa").same_dimension("N/m^2")
    assert Unit("Hz").same_dimension("1/s")
    assert Unit("Hz").same_dimension("s^-1")
    assert Unit("W").same_dimension("J/s")
    assert Unit("T").same_dimension("V*s/m^2")


def test_algebra_returns_units():
    assert (Unit("V") / Unit("m")).same_dimension("V/m")
    assert (Unit("A") / Unit("m") ** 2).same_dimension("A/m^2")
    assert (Unit("m") ** 3).same_dimension("m*m*m")
    assert (Unit("V") * Unit("s")).same_dimension("Wb")


def test_fractional_exponents():
    """Noise densities are real units: V/sqrt(Hz)."""
    u = Unit("V/Hz^1/2")
    assert u.dimension.as_dict()["time"] == Fraction(-5, 2)
    assert u.same_dimension("V*s^1/2")


def test_si_prefixes_scale_but_keep_dimension():
    assert Unit("mV").same_dimension("V")
    assert Unit("uV").to_base_factor == pytest.approx(1e-6)
    assert Unit("ms").to_base_factor == pytest.approx(1e-3)
    assert Unit("kg").to_base_factor == 1.0
    assert Unit("g").to_base_factor == pytest.approx(1e-3)
    # "T" is tesla, not the tera prefix on its own.
    assert Unit("T").same_dimension("Wb/m^2")
    assert Unit("mT").to_base_factor == pytest.approx(1e-3)


def test_conversion_factor():
    assert Unit("mV").conversion_factor_to("V") == pytest.approx(1e-3)
    assert Unit("V").conversion_factor_to("uV") == pytest.approx(1e6)
    with pytest.raises(UnitError, match="cannot convert"):
        Unit("V").conversion_factor_to("m")


@pytest.mark.parametrize("bad", ["bananas", "", "V/", "V**m", "V^", "(V", "3.5.2"])
def test_unparseable_units_are_refused(bad: str):
    with pytest.raises(UnitError):
        Unit(bad)


def test_unit_validates_inside_pydantic_models():
    with pytest.raises(ValidationError):
        TemporalSupport(clock="eeg_amp", dt=1e-3, units="bananas")  # type: ignore[call-arg]


def test_arbitrary_units_are_flagged():
    """PPG in arbitrary units is admissible but never silently physical."""
    assert Unit("au").is_arbitrary
    assert Unit("adu").is_arbitrary
    assert not Unit("V").is_arbitrary
    assert Unit("au").is_dimensionless


def test_dimension_algebra_is_a_group():
    d = Unit("V").dimension
    assert (d / d).is_dimensionless
    assert d * Dimension.dimensionless() == d
    assert (d**2) / d == d


def test_register_unit_extends_the_registry():
    register_unit("streamline", 1.0, Dimension.dimensionless(), prefixable=False)
    assert "streamline" in known_units()
    assert Unit("streamline/mm^3").same_dimension("1/m^3")


def test_units_serialize_as_plain_strings():
    support = TemporalSupport(clock="eeg_amp", dt=1e-3)
    assert isinstance(support.model_dump(mode="json")["clock"], str)
    assert str(Unit("uV")) == "uV"
    assert Unit(" u V ") == "uV"  # whitespace is normalized
