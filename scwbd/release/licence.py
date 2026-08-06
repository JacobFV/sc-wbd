"""Licence propagation: the union of what a checkpoint's sources actually require.

The rule implemented here is not invented for this module.  It is the rule
``reports/anatomy_prior.md`` §6 ("Licensing that must not be laundered")
already states and ``assets/MANIFEST.json`` already applies to every derived
artifact:

    a derived work **inherits the most restrictive licence of its inputs**.

A checkpoint is a derived work of its training sources, so the same rule
applies one level up.  This module computes that union rather than asserting
it, and reports it per tag variant.

Three properties are non-negotiable, and each exists because the opposite
failure is easy and quiet:

1. **Tri-state, not boolean.** Every term is ``True`` / ``False`` / ``None``,
   where ``None`` means *unknown*.  ``unknown`` never collapses to ``False``.
   Two of the datasets on disk (``mne-sample``, ``mne-spm-face``) genuinely
   ship no licence, and recording them as "commercial use permitted" would be
   a licence field asserting more than anyone established — the same class of
   defect as a provenance field claiming an audit that never ran.
2. **"Not non-commercial" is not "unrestricted".** ODC-By, CC0 and the
   PhysioNet terms still carry attribution and citation conditions. The union
   reports every term it found, and :meth:`LicenceUnion.summary` refuses to
   render an empty obligation list as "permissive".
3. **Inheritance is separated from policy.** ``by_inheritance`` is what a
   source *forces*; ``by_policy`` is what an owner *chose*. A reader must be
   able to tell which constraints are removable — a policy can be revoked by
   the person who set it, an inherited licence cannot. They are different
   fields and they are never merged.

Every :class:`LicenceTerm` carries ``provenance`` (the file the fact came from)
and ``verified`` (whether that file actually states it). A licence asserted in
a brief but absent from the repository is recorded ``verified=False`` and stays
visibly unverified rather than hardening into fact by repetition.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = [
    "LicenceTerm",
    "LicenceUnion",
    "UNKNOWN_TERM",
    "is_noncommercial_text",
    "is_share_alike_text",
    "term_from_licence_text",
    "term_from_dataset_card",
    "anatomy_nc_inputs",
    "union_of",
    "TriState",
]

TriState = bool | None

# Matches "NC" as a licence-token, "non-commercial"/"noncommercial", and the
# FSL-style prose "free for non-commercial research".  Deliberately *not* a
# bare substring search for "nc": that matches "Encoding", "Inc." and
# "Franchise", and a false NC is as wrong as a missed one.
_NC_PATTERNS = (
    re.compile(r"\bBY[-\s]?NC\b", re.I),
    re.compile(r"\bNC[-\s]?SA\b", re.I),
    re.compile(r"non[-\s]?commercial", re.I),
    re.compile(r"\bnoncommercial\b", re.I),
)
_SA_PATTERNS = (
    re.compile(r"\bSA\b"),
    re.compile(r"share[-\s]?alike", re.I),
)
_ATTRIB_PATTERNS = (
    re.compile(r"\bCC[-\s]?BY\b", re.I),
    re.compile(r"\bODC[-\s]?By\b", re.I),
    re.compile(r"attribution", re.I),
    re.compile(r"citation required", re.I),
    re.compile(r"acknowledge", re.I),
)
_UNKNOWN_PATTERNS = (re.compile(r"^\s*unknown", re.I),)


def is_noncommercial_text(text: str | None) -> TriState:
    """Does this licence string impose a non-commercial restriction?

    Returns ``None`` for an unknown/absent licence: an unlicensed dataset is
    not thereby commercially usable, and saying ``False`` here would be the
    laundering step.
    """
    if not text or _UNKNOWN_PATTERNS[0].match(str(text)):
        return None
    return any(p.search(str(text)) for p in _NC_PATTERNS)


def is_share_alike_text(text: str | None) -> TriState:
    """Does this licence carry a copyleft / share-alike obligation?

    Share-alike matters independently of NC and is routinely dropped in
    summaries. CC-BY-NC-SA-4.0 — which is what the Hansen receptor atlas
    carries, and therefore what every receptor-derived prior inherits — is
    *both*, and reporting only the NC half understates the obligation.
    """
    if not text or _UNKNOWN_PATTERNS[0].match(str(text)):
        return None
    return any(p.search(str(text)) for p in _SA_PATTERNS)


def _attribution_of(text: str | None) -> TriState:
    if not text or _UNKNOWN_PATTERNS[0].match(str(text)):
        return None
    if re.search(r"\bCC0\b|public domain|PDDL", str(text), re.I):
        return False
    return any(p.search(str(text)) for p in _ATTRIB_PATTERNS) or None


@dataclass(frozen=True)
class LicenceTerm:
    """One source's governance terms, with the file the facts came from."""

    source_id: str
    name: str
    noncommercial: TriState = None
    share_alike: TriState = None
    attribution: TriState = None
    #: ``full`` | ``none`` | ``unknown`` — may the artifact be redistributed?
    redistribution: str = "unknown"
    #: Named obligations that are not booleans (citation targets, DUA, etc).
    obligations: tuple[str, ...] = ()
    url: str | None = None
    #: Where this fact was read from. A path, or ``declared:<who>`` when the
    #: only authority is a human assertion.
    provenance: str = "unknown"
    #: True only when ``provenance`` is a file in this repository that states
    #: the licence. A brief is not a file.
    verified: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "noncommercial": self.noncommercial,
            "share_alike": self.share_alike,
            "attribution": self.attribution,
            "redistribution": self.redistribution,
            "obligations": list(self.obligations),
            "url": self.url,
            "provenance": self.provenance,
            "verified": self.verified,
            "notes": self.notes,
        }


#: The value used when a source's licence cannot be determined at all.
UNKNOWN_TERM = LicenceTerm(
    source_id="<unknown>",
    name="unknown",
    noncommercial=None,
    share_alike=None,
    attribution=None,
    redistribution="unknown",
    provenance="unknown",
    verified=False,
    notes=(
        "No licence could be read for this source. Unknown is not permissive: "
        "nothing here establishes that commercial use, redistribution or "
        "sublicensing is allowed."
    ),
)


def term_from_licence_text(
    source_id: str,
    text: str | None,
    *,
    provenance: str,
    verified: bool,
    url: str | None = None,
    obligations: Sequence[str] = (),
    redistribution: str = "unknown",
    notes: str = "",
) -> LicenceTerm:
    """Build a term by classifying a free-text licence statement."""
    return LicenceTerm(
        source_id=source_id,
        name=(text or "unknown").strip(),
        noncommercial=is_noncommercial_text(text),
        share_alike=is_share_alike_text(text),
        attribution=_attribution_of(text),
        redistribution=redistribution,
        obligations=tuple(obligations),
        url=url,
        provenance=provenance,
        verified=verified,
        notes=notes,
    )


def term_from_dataset_card(path: str | Path) -> LicenceTerm:
    """Read governance terms from a ``scwbd/sources/cards/*.yaml`` dataset card.

    These cards are the only place in the repository where dataset licences are
    written down in machine-readable form, so they are the authority. The
    training-mixture cards under ``configs/source_cards/`` carry **no** licence
    field at all — see ``reports/checkpoint_family.md`` §"What is missing".
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    gov = data.get("governance", {}) or {}
    ident = data.get("identity", {}) or {}
    text = gov.get("license")
    return term_from_licence_text(
        source_id=str(ident.get("id") or p.stem),
        text=text,
        provenance=str(p),
        verified=bool(text) and not str(text).lower().startswith("unknown"),
        url=gov.get("license_url"),
        obligations=tuple(gov.get("purpose_limits_list") or ()),
        redistribution=str(gov.get("redistribution_class") or "unknown"),
        notes=str(gov.get("unavailable_reason") or ""),
    )


def _resolve_assets_manifest(manifest_path: str | Path) -> Path | None:
    """Find ``MANIFEST.json``, following the ``assets/`` symlinks to the data root.

    ``assets/`` in the repository holds only ``.gitignore`` plus symlinks
    (``cache``, ``derived``, ``src``) into the data root; the manifest itself
    lives beside them at the *target* root. Resolving through the symlink means
    this works both in a checkout with data attached and in one without,
    instead of silently returning "no NC assets" — which would read as
    "commercial use is fine".
    """
    mp = Path(manifest_path)
    if mp.exists():
        return mp
    for sibling in ("derived", "src", "cache"):
        link = mp.parent / sibling
        if link.is_symlink() or link.exists():
            candidate = Path(os.path.realpath(link)).parent / mp.name
            if candidate.exists():
                return candidate
    return None


def anatomy_nc_inputs(
    manifest_path: str | Path = "assets/MANIFEST.json",
    src_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Which anatomy assets inherit a non-commercial term, and via which input.

    Resolves ``assets/MANIFEST.json``'s ``inputs`` (which are keys of
    ``scwbd.anatomy.sources.SRC``) against that registry's per-source licences.
    Returns ``{asset_path: [nc_input_key, ...]}``.

    This is the step that decides whether ``-raw`` and ``-with-simulation``
    are non-commercial, and the answer is not the one a reader expects: the
    Hansen receptor atlas is CC-BY-NC-SA-4.0, so every receptor-derived map and
    every Hansen-input connectome inherits **NC *and* share-alike** —
    independently of whether TRIBE is ever enabled. Whether the artifact
    actually touches those assets is a property of the run, not of this table,
    which is why the manifest resolves it per checkpoint.
    """
    if src_registry is None:
        from scwbd.anatomy.sources import SRC as src_registry  # local: avoids import cost
    mp = _resolve_assets_manifest(manifest_path)
    if mp is None:
        return {}
    payload = json.loads(mp.read_text())
    assets = payload.get("assets", payload) if isinstance(payload, dict) else {}
    nc_keys = {
        k for k, v in src_registry.items() if is_noncommercial_text(v.get("license")) is True
    }
    out: dict[str, list[str]] = {}
    for asset, meta in assets.items():
        if not isinstance(meta, dict):
            continue
        hits = sorted(set(meta.get("inputs") or ()) & nc_keys)
        if hits:
            out[asset] = hits
    return out


@dataclass(frozen=True)
class LicenceUnion:
    """The effective licence of a derived artifact, with its provenance split.

    ``by_inheritance`` and ``by_policy`` are separate on purpose and are never
    summed into a single "effective NC" boolean without both being visible.
    """

    terms: tuple[LicenceTerm, ...]
    #: Owner-chosen constraints, keyed by term name -> reason.
    policy: Mapping[str, str] = field(default_factory=dict)

    # -- computed terms ---------------------------------------------------
    @staticmethod
    def _reduce(values: Iterable[TriState]) -> TriState:
        """Most restrictive wins; unknown beats False but loses to True."""
        vals = list(values)
        if any(v is True for v in vals):
            return True
        if any(v is None for v in vals):
            return None
        return False

    @property
    def noncommercial_by_inheritance(self) -> TriState:
        return self._reduce(t.noncommercial for t in self.terms)

    @property
    def share_alike_by_inheritance(self) -> TriState:
        return self._reduce(t.share_alike for t in self.terms)

    @property
    def attribution_by_inheritance(self) -> TriState:
        return self._reduce(t.attribution for t in self.terms)

    @property
    def redistribution(self) -> str:
        classes = {t.redistribution for t in self.terms}
        if "none" in classes:
            return "none"
        if "unknown" in classes or not classes:
            return "unknown"
        if "partial" in classes:
            return "partial"
        return "full"

    @property
    def noncommercial_by_policy(self) -> bool:
        return "noncommercial" in self.policy

    @property
    def noncommercial_effective(self) -> TriState:
        if self.noncommercial_by_policy:
            return True
        return self.noncommercial_by_inheritance

    @property
    def noncommercial_is_removable(self) -> bool:
        """True when NC is a *choice* that could be revoked.

        Only meaningful when NC is in force. Policy NC is removable by whoever
        set the policy; inherited NC is not removable by anyone in this
        project. A reader who cannot tell the two apart cannot tell which
        constraints survive a change of mind.
        """
        return self.noncommercial_by_policy and self.noncommercial_by_inheritance is not True

    # -- reporting --------------------------------------------------------
    @property
    def inheritance_sources(self) -> tuple[str, ...]:
        """Sources that actually force NC."""
        return tuple(t.source_id for t in self.terms if t.noncommercial is True)

    @property
    def unknown_sources(self) -> tuple[str, ...]:
        """Sources whose licence could not be established. Never hidden."""
        return tuple(t.source_id for t in self.terms if t.noncommercial is None)

    @property
    def unverified_sources(self) -> tuple[str, ...]:
        """Sources whose licence is asserted but not backed by a file here."""
        return tuple(t.source_id for t in self.terms if not t.verified)

    @property
    def obligations(self) -> tuple[str, ...]:
        out: set[str] = set()
        for t in self.terms:
            out.update(t.obligations)
        if self.attribution_by_inheritance is True:
            out.add("attribution_required")
        if self.share_alike_by_inheritance is True:
            out.add("share_alike_required")
        return tuple(sorted(out))

    def summary(self) -> str:
        """One-line human summary. Never renders as a bare 'permissive'."""
        def tri(v: TriState, yes: str, no: str) -> str:
            return yes if v is True else (no if v is False else "UNKNOWN")

        bits = [
            f"non-commercial: {tri(self.noncommercial_effective, 'yes', 'no')}",
            f"share-alike: {tri(self.share_alike_by_inheritance, 'yes', 'no')}",
            f"attribution: {tri(self.attribution_by_inheritance, 'required', 'not required')}",
            f"redistribution: {self.redistribution}",
        ]
        if self.unknown_sources:
            bits.append(
                f"{len(self.unknown_sources)} source(s) with UNKNOWN licence "
                f"({', '.join(self.unknown_sources)}) — unknown is not permissive"
            )
        return "; ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective": {
                "noncommercial": self.noncommercial_effective,
                "share_alike": self.share_alike_by_inheritance,
                "attribution": self.attribution_by_inheritance,
                "redistribution": self.redistribution,
                "obligations": list(self.obligations),
                "summary": self.summary(),
            },
            "by_inheritance": {
                "noncommercial": self.noncommercial_by_inheritance,
                "share_alike": self.share_alike_by_inheritance,
                "forced_by": list(self.inheritance_sources),
            },
            "by_policy": {
                "noncommercial": self.noncommercial_by_policy,
                "terms": dict(self.policy),
            },
            "noncommercial_is_removable": self.noncommercial_is_removable,
            "unknown_licence_sources": list(self.unknown_sources),
            "unverified_licence_sources": list(self.unverified_sources),
            "terms": [t.as_dict() for t in self.terms],
        }


def union_of(
    terms: Iterable[LicenceTerm], *, policy: Mapping[str, str] | None = None
) -> LicenceUnion:
    """Union a set of source terms, with optional owner policy overlay."""
    return LicenceUnion(terms=tuple(terms), policy=dict(policy or {}))
