"""Namespaced identifier types.

``FrameId`` and ``ClockId`` are ``str`` subclasses rather than bare strings so
that a coordinate frame can never be passed where a clock is expected, and so
that pydantic rejects the empty/whitespace identifiers that make an
"unknown frame" refusal (R01) impossible to attribute to an object.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

__all__ = ["FrameId", "ClockId", "ScaleId", "IdError"]

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:\-]*$")


class IdError(ValueError):
    """Raised for malformed identifiers."""


class _NamedId(str):
    """Base for validated identifier strings."""

    __slots__ = ()
    _what = "identifier"

    def __new__(cls, value: Any) -> "_NamedId":
        if not isinstance(value, str):
            raise IdError(f"{cls._what} must be a string, got {type(value).__name__}")
        text = value.strip()
        if not _ID_RE.match(text):
            raise IdError(
                f"illegal {cls._what} {value!r}: must start with a letter and use "
                "only [A-Za-z0-9_.:-]"
            )
        return super().__new__(cls, text)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    @classmethod
    def _validate(cls, value: Any) -> "_NamedId":
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


class FrameId(_NamedId):
    """A physical reference frame, e.g. ``subject_surface_RAS``."""

    __slots__ = ()
    _what = "frame id"


class ClockId(_NamedId):
    """A measurement clock, e.g. ``eeg_amp`` or ``scanner_volume``."""

    __slots__ = ()
    _what = "clock id"


class ScaleId(_NamedId):
    """An element of the resolution poset, e.g. ``parcel`` or ``surface_vertex``."""

    __slots__ = ()
    _what = "scale id"
