"""Content-addressed frozen model base for every SC-WBD schema object.

Every object in ``scwbd.schema`` is an immutable ``pydantic`` v2 model that can
report a stable ``content_hash()``: a sha256 over canonical JSON with sorted
keys.  Hashes are the substrate for immutable lineage (ARCHITECTURE.md sec. 2)
and for the leakage barrier in refusal R10.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["SchemaModel", "canonical_json", "content_hash_of"]


def canonical_json(payload: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no whitespace, ASCII only."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash_of(payload: Any) -> str:
    """sha256 over the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class SchemaModel(BaseModel):
    """Frozen, extra-forbidding, content-addressed base model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    def canonical(self) -> dict[str, Any]:
        """JSON-mode dump used as the pre-image of :meth:`content_hash`."""
        return self.model_dump(mode="json")

    def content_hash(self) -> str:
        """Stable sha256 over canonical JSON with sorted keys."""
        return content_hash_of(self.canonical())

    @property
    def short_hash(self) -> str:
        return self.content_hash()[:12]
