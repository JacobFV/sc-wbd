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
    "is_vacuous_licence_text",
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

#: Strings that are *present* but state no terms. Added 2026-08-06 (🍃 Mendel).
#:
#: The guard below already refused to launder an **absent** licence into
#: ``False``. It laundered a **vacuous** one, which is epistemically identical:
#: ``"As distributed via neuromaps"`` names no terms, and neuromaps' own entry
#: says ``"per-annotation source terms"``, so the pair resolves in a circle
#: while every individual field is non-empty and the classifier reports "no
#: restriction". Six of the twelve sources on the anatomy default path were
#: being read this way (``reports/licence_audit.md`` §1).
#:
#: This is the register's own pattern — an unknown recorded as a zero — sitting
#: inside the machinery built to prevent it. The docstrings stated the right
#: principle and the code implemented it for one of the two cases.
#:
#: **Reading these as ``None`` is expected to move several arms from "clear" to
#: "unknown". That is the correct direction: unknown is what we have.**
_VACUOUS_PATTERNS = (
    # "As distributed via neuromaps", "As released with the cited papers"
    re.compile(r"^\s*as (distributed|released|provided|published)\b", re.I),
    # "See repository LICENSE (open, academic use)", "See LICENSE file"
    re.compile(r"^\s*see\s+(the\s+)?(repository|license|licence|LICENSE)\b", re.I),
    # a bare regime name with no terms: "FreeSurfer license", "EBRAINS terms",
    # "HCP open-access data-use terms", "ADNI Data Use Agreement"
    re.compile(
        r"^[\w\-/ ]{0,60}?\b(licen[cs]e|terms|agreement|dua)\b[\s\.]*$", re.I
    ),
)

#: Phrases that *deny* a constraint. ``\bnon[-\s]?commercial\b`` matches "no
#: non-commercial term" exactly as it matches "non-commercial use only", and a
#: licence field written to be helpful ("Attribution required; no
#: non-commercial or share-alike term") was classified NC by this module on
#: 2026-08-06. Found by writing exactly that field and watching a test fail.
#:
#: This is a *narrow* fix and it is not a sentence parser: it suppresses a match
#: only when the constraint word is immediately preceded by an explicit denial.
#: The durable remedy is upstream — a ``license`` field states the licence's own
#: terms and nothing else — which is why ``sources.SRC["tian2020"]`` now carries
#: its commentary in ``bias``.
_NEGATION = re.compile(
    r"\b(no|not|without|free\s+of|neither)\s+(\w+\s+){0,2}$", re.I
)


def _matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Any pattern matches, ignoring occurrences that are explicitly negated."""
    for p in patterns:
        for m in p.finditer(text):
            if not _NEGATION.search(text[: m.start()]):
                return True
    return False


#: Clauses that **defer** to terms this repository has not read, appearing
#: anywhere in a field rather than at its start.
#:
#: This catches the *compound* case, which the anchored patterns above miss and
#: which is where the remaining laundering lived:
#: ``"BSD-3-Clause code; HCP open-access data-use terms for the underlying
#: scans"`` reads ``False`` under an anchored test because its first clause is a
#: real licence — while the clause that governs the **data** states nothing.
#: ``"MIT (CBIG); underlying GSP data under its own terms"`` is the same shape.
#:
#: Deliberately conservative: a field with any deferring clause is undetermined
#: for *every* constraint, because the deferred clause could carry any of them.
#: This is expected to move sources from "no restriction" to "unknown", which is
#: the direction the evidence supports.
_DEFERRAL_MARKERS = (
    re.compile(r"under (its|their) own terms", re.I),
    re.compile(r"per[-\s]annotation source terms", re.I),
    re.compile(r"\b(data[-\s]use|open[-\s]access|EBRAINS)\s+terms\b", re.I),
    re.compile(r"\b(research|academic)\s+use\b", re.I),
    re.compile(r"\b(data use agreement|material transfer agreement|\bDUA\b)", re.I),
    re.compile(r"as (released|distributed) with", re.I),
)


def is_vacuous_licence_text(text: str | None) -> bool:
    """Is this a licence field that *points at* terms instead of stating them?

    ``True`` means the field establishes nothing about at least one of the works
    it covers, so every constraint must be reported as unknown. It is not a
    judgement about the upstream licence — only about whether this repository
    has read it.
    """
    if not text:
        return False  # absent, not vacuous; the callers handle absent first
    s = str(text).strip()
    return any(p.match(s) for p in _VACUOUS_PATTERNS) or any(
        p.search(s) for p in _DEFERRAL_MARKERS
    )


def _undetermined(text: str | None) -> bool:
    return (
        not text
        or bool(_UNKNOWN_PATTERNS[0].match(str(text)))
        or is_vacuous_licence_text(text)
    )


def is_noncommercial_text(text: str | None) -> TriState:
    """Does this licence string impose a non-commercial restriction?

    Returns ``None`` for an absent, unknown **or vacuous** licence: an
    unlicensed dataset is not thereby commercially usable, and saying ``False``
    here would be the laundering step. A field that points at a licence without
    stating it is in exactly the same position as one that is missing.
    """
    if _undetermined(text):
        return None
    return _matches(str(text), _NC_PATTERNS)


def is_share_alike_text(text: str | None) -> TriState:
    """Does this licence carry a copyleft / share-alike obligation?

    Share-alike matters independently of NC and is routinely dropped in
    summaries. CC-BY-NC-SA-4.0 — which is what the Hansen receptor atlas
    carries — is *both*, and reporting only the NC half understates the
    obligation.
    """
    if _undetermined(text):
        return None
    return _matches(str(text), _SA_PATTERNS)


def _attribution_of(text: str | None) -> TriState:
    if _undetermined(text):
        return None
    if re.search(r"\bCC0\b|public domain|PDDL", str(text), re.I):
        return False
    return _matches(str(text), _ATTRIB_PATTERNS) or None


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

    # -- generic per-constraint accessors ---------------------------------
    #: Every constraint the union tracks. Each is computed the same way and
    #: reported with the same structure, so share-alike cannot quietly become a
    #: footnote to non-commercial. SA is the **more viral** term — a derivative
    #: work must be released under the same licence — and it reaches further
    #: than NC does, so it gets equal machinery, not a mention.
    CONSTRAINTS: tuple[str, ...] = ("noncommercial", "share_alike", "attribution")

    def by_inheritance(self, constraint: str) -> TriState:
        """Is ``constraint`` forced by some source? Tri-state."""
        self._check(constraint)
        return self._reduce(getattr(t, constraint) for t in self.terms)

    def sources_forcing(self, constraint: str) -> tuple[str, ...]:
        """Which sources force ``constraint``. This is the removability evidence.

        A reader must be able to see that removing non-commercial means
        dropping *these datasets*, not asking the owner to change their mind.
        """
        self._check(constraint)
        return tuple(t.source_id for t in self.terms if getattr(t, constraint) is True)

    def by_policy(self, constraint: str) -> bool:
        """Is ``constraint`` imposed by owner choice rather than by a source?"""
        self._check(constraint)
        return constraint in self.policy

    def effective(self, constraint: str) -> TriState:
        self._check(constraint)
        if self.by_policy(constraint):
            return True
        return self.by_inheritance(constraint)

    def is_removable(self, constraint: str) -> bool:
        """True when ``constraint`` is in force *only* because someone chose it.

        False when a source forces it — including when policy also asks for it,
        because a policy cannot lift an inherited obligation.
        """
        self._check(constraint)
        return self.by_policy(constraint) and self.by_inheritance(constraint) is not True

    def removal_requires(self, constraint: str) -> str:
        """Plain statement of what it would take to lift ``constraint``."""
        self._check(constraint)
        if self.effective(constraint) is not True:
            return "not in force"
        forced = self.sources_forcing(constraint)
        if forced:
            return (
                "dropping the source(s) that carry it: " + ", ".join(forced)
                + " -- an owner decision cannot lift this"
            )
        if self.by_policy(constraint):
            return f"revoking the owner policy: {self.policy.get(constraint, '')}"
        return "unknown"  # pragma: no cover - defensive

    def term_status(self, constraint: str) -> dict[str, Any]:
        """The full, symmetric record for one constraint."""
        self._check(constraint)
        return {
            "effective": self.effective(constraint),
            "by_inheritance": self.by_inheritance(constraint),
            "forced_by": list(self.sources_forcing(constraint)),
            "by_policy": self.by_policy(constraint),
            "policy_reason": self.policy.get(constraint),
            "removable": self.is_removable(constraint),
            "removal_requires": self.removal_requires(constraint),
        }

    @staticmethod
    def _check(constraint: str) -> None:
        if constraint not in LicenceUnion.CONSTRAINTS:
            raise KeyError(
                f"unknown constraint {constraint!r}; tracked constraints are "
                f"{list(LicenceUnion.CONSTRAINTS)}"
            )

    # -- named aliases (readability at call sites) ------------------------
    @property
    def noncommercial_by_inheritance(self) -> TriState:
        return self.by_inheritance("noncommercial")

    @property
    def share_alike_by_inheritance(self) -> TriState:
        return self.by_inheritance("share_alike")

    @property
    def attribution_by_inheritance(self) -> TriState:
        return self.by_inheritance("attribution")

    @property
    def share_alike_effective(self) -> TriState:
        return self.effective("share_alike")

    @property
    def share_alike_sources(self) -> tuple[str, ...]:
        return self.sources_forcing("share_alike")

    @property
    def share_alike_is_removable(self) -> bool:
        return self.is_removable("share_alike")

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
        return self.by_policy("noncommercial")

    @property
    def noncommercial_effective(self) -> TriState:
        return self.effective("noncommercial")

    @property
    def noncommercial_is_removable(self) -> bool:
        """True when NC is a *choice* that could be revoked.

        Only meaningful when NC is in force. Policy NC is removable by whoever
        set the policy; inherited NC is not removable by anyone in this
        project. A reader who cannot tell the two apart cannot tell which
        constraints survive a change of mind.
        """
        return self.is_removable("noncommercial")

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
            f"share-alike: {tri(self.share_alike_effective, 'yes', 'no')}",
            f"attribution: {tri(self.attribution_by_inheritance, 'required', 'not required')}",
            f"redistribution: {self.redistribution}",
        ]
        if self.share_alike_effective is True:
            bits.append(
                "SHARE-ALIKE IN FORCE: derivative works must be released under "
                "the same licence"
            )
        if self.unknown_sources:
            bits.append(
                f"{len(self.unknown_sources)} source(s) with UNKNOWN licence "
                f"({', '.join(self.unknown_sources)}) — unknown is not permissive"
            )
        return "; ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            # Every tracked constraint, reported with identical structure.
            # Share-alike is not a footnote to non-commercial here.
            "constraints": {c: self.term_status(c) for c in self.CONSTRAINTS},
            "effective": {
                "noncommercial": self.noncommercial_effective,
                "share_alike": self.share_alike_effective,
                "attribution": self.attribution_by_inheritance,
                "redistribution": self.redistribution,
                "obligations": list(self.obligations),
                "summary": self.summary(),
            },
            "by_inheritance": {
                "noncommercial": self.noncommercial_by_inheritance,
                "share_alike": self.share_alike_by_inheritance,
                "forced_by": list(self.inheritance_sources),
                "share_alike_forced_by": list(self.share_alike_sources),
            },
            "by_policy": {
                "noncommercial": self.noncommercial_by_policy,
                "share_alike": self.by_policy("share_alike"),
                "terms": dict(self.policy),
            },
            "noncommercial_is_removable": self.noncommercial_is_removable,
            "share_alike_is_removable": self.share_alike_is_removable,
            "removal_requires": {
                c: self.removal_requires(c) for c in self.CONSTRAINTS
            },
            "unknown_licence_sources": list(self.unknown_sources),
            "unverified_licence_sources": list(self.unverified_sources),
            "terms": [t.as_dict() for t in self.terms],
        }


def union_of(
    terms: Iterable[LicenceTerm], *, policy: Mapping[str, str] | None = None
) -> LicenceUnion:
    """Union a set of source terms, with optional owner policy overlay."""
    return LicenceUnion(terms=tuple(terms), policy=dict(policy or {}))
