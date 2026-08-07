"""Checkpoint taxonomy, provenance and licence propagation for SC-WBD-001-beta.

A checkpoint tag names which source families trained an artifact. This package
treats that name as a **claim** and supplies the machinery to check it:

* :mod:`~scwbd.release.tags` — parse/validate/format/order tags. Strict ISO 8601
  basic UTC timestamps; a closed variant set; the ``001-beta`` release alias.
* :mod:`~scwbd.release.families` — which family a source belongs to, derived
  from its card and gradient permission. Includes the mapping to Appendix D's
  D12 role buckets and the provenance-integrity tiers.
* :mod:`~scwbd.release.datasets` — links mixture sources to dataset cards for
  modality and licence.
* :mod:`~scwbd.release.licence` — the union of source governance terms, with
  non-commercial status split into inheritance and policy.
* :mod:`~scwbd.release.manifest` — :class:`SourceFamilyManifest` and the
  provenance block; :meth:`SourceFamilyManifest.validate_tag` is the check that
  makes a tag mean something.
* :mod:`~scwbd.release.collapse` — refuses to mint distinct tags for
  byte-identical artifacts.

Nothing here imports ``torch``: release and audit run while a training job may
hold the GPU.
"""

from __future__ import annotations

from .collapse import CollapseError, CollapseResult, TagAlias, collapse_identical
from .families import (
    ALL_FAMILIES,
    AUXILIARY_FAMILIES,
    FAMILY_TIER,
    REAL,
    SIMULATION,
    SYNTHETIC,
    TAG_FAMILIES,
    UNKNOWN,
    SourceRecord,
    VARIANT_FAMILIES,
    classify,
    load_source_records,
)
from .licence import LicenceTerm, LicenceUnion, union_of
from .manifest import (
    DOWNSTREAM_REACH_QUESTION,
    LICENCE_DECISION_HISTORY,
    OWNER_LICENCE_DECISION,
    OWNER_LICENCE_DECISION_2,
    OWNER_LICENCE_DECISION_V1,
    ProvenanceBlock,
    ProvenanceMismatch,
    SourceFamilyManifest,
    build_manifest,
    config_hash,
    git_sha,
    sha256_file,
)
from .tags import (
    ALIAS_VARIANT,
    BASE,
    VARIANT_ORDER,
    VARIANTS,
    CheckpointTag,
    TagFormatError,
    format_timestamp,
    parse_timestamp,
    sort_tags,
)

__all__ = [
    "ALIAS_VARIANT",
    "ALL_FAMILIES",
    "AUXILIARY_FAMILIES",
    "BASE",
    "CheckpointTag",
    "CollapseError",
    "CollapseResult",
    "DOWNSTREAM_REACH_QUESTION",
    "OWNER_LICENCE_DECISION",
    "OWNER_LICENCE_DECISION_V1",
    "OWNER_LICENCE_DECISION_2",
    "LICENCE_DECISION_HISTORY",
    "FAMILY_TIER",
    "LicenceTerm",
    "LicenceUnion",
    "ProvenanceBlock",
    "ProvenanceMismatch",
    "REAL",
    "SIMULATION",
    "SYNTHETIC",
    "SourceFamilyManifest",
    "SourceRecord",
    "TAG_FAMILIES",
    "TagAlias",
    "TagFormatError",
    "UNKNOWN",
    "VARIANTS",
    "VARIANT_FAMILIES",
    "VARIANT_ORDER",
    "build_manifest",
    "classify",
    "collapse_identical",
    "config_hash",
    "format_timestamp",
    "git_sha",
    "load_source_records",
    "parse_timestamp",
    "sha256_file",
    "sort_tags",
    "union_of",
]
