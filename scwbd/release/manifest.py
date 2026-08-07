"""The source-family manifest: what actually trained a checkpoint, and its licence.

This is the load-bearing module of the package. A checkpoint tag is a *claim*
about provenance written into a filename; a manifest is the *evidence* for that
claim, derived from the run's own source cards and gradient permissions. The
two are compared by :meth:`SourceFamilyManifest.validate_tag`, and a tag that
disagrees with its manifest is a defect that fails validation.

The asymmetry is deliberate. Nothing here reads a filename to decide what
trained a model, and nothing trusts a tag because it parses. A checkpoint
tagged ``-raw`` whose manifest shows a simulated source is refused, loudly,
with the offending sources named.

Licence is computed here too, not declared. Each contributing source supplies a
:class:`~scwbd.release.licence.LicenceTerm`; the union is the effective licence
(most restrictive input wins — ``reports/anatomy_prior.md`` §6). Non-commercial
status is split into **inheritance** (a source forces it) and **policy** (an
owner chose it) so a reader can tell which constraints are removable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .datasets import DatasetInfo, link_sources_to_datasets, load_dataset_cards
from .families import (
    FAMILY_TIER,
    FAMILY_TO_D12_BUCKET,
    TAG_FAMILIES,
    TIER_GAP_REASON,
    UNKNOWN,
    SourceRecord,
    VARIANT_FAMILIES,
    load_source_records,
)
from .licence import (
    LicenceTerm,
    LicenceUnion,
    UNKNOWN_TERM,
    anatomy_nc_inputs,
    union_of,
)
from .tags import CheckpointTag

__all__ = [
    "OWNER_LICENCE_DECISION",
    "OWNER_LICENCE_DECISION_V1",
    "OWNER_LICENCE_DECISION_2",
    "LICENCE_DECISION_HISTORY",
    "DOWNSTREAM_REACH_QUESTION",
    "ProvenanceMismatch",
    "SourceFamilyManifest",
    "ProvenanceBlock",
    "NON_DATASET_TERMS",
    "sha256_file",
    "git_sha",
    "config_hash",
    "build_manifest",
]

#: Where the claim boundary for this release line is written (agent J / Popper).
CLAIM_BOUNDARY_PATH = "reports/CLAIM_BOUNDARY.md"

#: The owner's licence decision, recorded as a decision rather than applied as
#: a silent default.
#:
#: 2026-08-06: **accept CC-BY-NC-SA-4.0.** The Hansen receptor maps stay in the
#: pipeline, and the release accepts the non-commercial *and* share-alike terms
#: they carry.
#:
#: The important consequence for this module is what it does **not** do: it adds
#: **no policy overlay**. Accepting an inherited constraint is not the same act
#: as imposing one. If NC were recorded as policy, ``noncommercial_is_removable``
#: would read ``True`` and a future reader would conclude the owner could lift it
#: by changing their mind. They cannot: it is forced by ``hansen_receptors`` via
#: ``BrainPrior``, and removing it means dropping that dataset. The policy
#: mapping therefore stays empty and the constraint is reported as inheritance.
#: **SUPERSEDED 2026-08-06 by** :data:`OWNER_LICENCE_DECISION_2`. Kept verbatim.
#:
#: The decision below was correct on the facts available when it was made, and
#: one of those facts has since changed: the receptor-derived E/I prior is no
#: longer on the default path, so the share-alike term it forced is no longer
#: inherited. See ``reports/ei_ordering_substitution.md``.
#:
#: It is **superseded, not edited**, and the register says why
#: (``reports/decorative_guards.md``): *you may not amend a fired trigger; you
#: may supersede it going forward, with both versions visible.* Editing this
#: dict in place would leave a record that reads as though the owner had always
#: decided the current thing, erasing that a decision was taken on facts that
#: later moved — which is the whole content of the event.
#:
#: Note in particular the sentence that has been falsified, and by what:
#: *"removing it means dropping that dataset"* — true, and the dataset was
#: dropped from the default path. The decision was never wrong; it was
#: **conditional on an arrangement that no longer holds.**
OWNER_LICENCE_DECISION_V1: Mapping[str, Any] = {
    "date": "2026-08-06",
    "decision": "accept CC-BY-NC-SA-4.0 inherited from hansen_receptors",
    "terms_accepted": ["noncommercial", "share_alike", "attribution"],
    "policy_overlay": {},  # deliberately empty -- see above
    "rationale": (
        "The receptor-derived E/I prior is retained for its scientific value. "
        "Accepting an inherited licence is not the same act as imposing a policy: "
        "recording it as policy would misrepresent it as removable by the owner."
    ),
    "applies_to": "every arm built with the real (biological) anatomical prior",
    "does_not_apply_to": (
        "artifacts built with the synthetic fallback anatomy, which carry no "
        "Hansen input and are genuinely not NC-SA -- see reports/checkpoint_family.md §4.3"
    ),
    "superseded_by": "OWNER_LICENCE_DECISION_2",
    "superseded_on": "2026-08-06",
    "superseded_because": (
        "The default anatomical prior no longer reads hansen_receptors, so the "
        "share-alike term this decision accepted is no longer inherited. The "
        "decision is not withdrawn and was not wrong: the arrangement it was "
        "about was changed. Both versions stay visible."
    ),
}

#: The current decision. **It does not restate v1's terms; it records what
#: changed and what is still open**, because a replacement that merely reads
#: cleaner than its predecessor is how a record loses the fact that anything
#: happened.
#:
#: What changed: 🍃 Mendel substituted the receptor-derived E/I ordering
#: (2026-08-06, cf37755) and the Harvard-Oxford subcortical geometry, both for
#: permissively-licensed inputs, keeping each original available as an explicit
#: opt-in that records itself.
#:
#: **What this does NOT establish**, and the reason it is written as a decision
#: about *removal* rather than a claim of clearance: with
#: ``is_vacuous_licence_text`` now in force, 18 of 27 anatomy sources state no
#: terms at all and read ``unknown``. Removing the two established restrictions
#: does not make the family commercially clear — it makes it **unresolved**, and
#: that is a different and weaker claim. See ``reports/licence_audit.md`` §8.
OWNER_LICENCE_DECISION_2: Mapping[str, Any] = {
    "date": "2026-08-06",
    "supersedes": "OWNER_LICENCE_DECISION_V1",
    "decision": (
        "substitute the two established restricted inputs out of the default "
        "path and retain each as a self-recording opt-in; make no claim that "
        "the result is commercially clear"
    ),
    "terms_accepted": ["attribution"],
    "terms_no_longer_inherited_on_the_default_path": {
        "share_alike": (
            "forced by hansen_receptors via BrainPrior.ei_ratio_prior(); the "
            "default ordering is now 'hcp_hierarchy' (hcps1200_maps). Opt back "
            "in with ei_ratio_prior('hansen_receptors')."
        ),
        "noncommercial": (
            "forced by hansen_receptors, and separately by harvardoxford via the "
            "Aseg14 subcortical geometry. The default subcortical atlas is now "
            "Aseg14T (tian2020, attribution-only). Opt back in with "
            "BrainPrior.load(subcortical_atlas='Aseg14')."
        ),
    },
    "policy_overlay": {},  # still deliberately empty -- nothing here is owner policy
    "rationale": (
        "Both substitutions were made on a criterion committed before the "
        "candidates were measured (97086e7, 36d5ba6), with the measured cost "
        "reported rather than the licence outcome alone. Neither original was "
        "deleted: receptor identity has no substitute for the neuromodulator "
        "control fields of thesis S5, and no measurement in this repository "
        "establishes that the Melbourne subcortex atlas segments better than "
        "Harvard-Oxford."
    ),
    "still_unresolved": (
        "18 of 27 anatomy sources have licence fields that state no terms and "
        "now correctly read unknown -- including hcps1200_maps, which the new "
        "default E/I ordering depends on. 'No established restriction remains' "
        "is the strongest supportable claim; 'commercially clear' is not."
    ),
    "applies_to": "every arm built with the real (biological) anatomical prior",
    "does_not_apply_to": (
        "artifacts already built. Every checkpoint produced before this date "
        "used the synthetic fallback anatomy (load_anatomy() returns "
        "provenance='synthetic_fallback'), so none of them inherited either "
        "term in the first place -- see reports/licence_audit.md headline 5."
    ),
}

#: The decision in force. Aliased so existing readers keep working; the history
#: is in :data:`LICENCE_DECISION_HISTORY` and neither entry is ever rewritten.
OWNER_LICENCE_DECISION: Mapping[str, Any] = OWNER_LICENCE_DECISION_2

#: Oldest first. A record with one entry and no history cannot show that a
#: decision was ever revisited.
LICENCE_DECISION_HISTORY: tuple[Mapping[str, Any], ...] = (
    OWNER_LICENCE_DECISION_V1,
    OWNER_LICENCE_DECISION_2,
)

#: An unsettled legal question, recorded so it is visible and *not* answered.
#:
#: ``~/Documents/robotics`` loads SC-WBD checkpoints (``ServedModel.load``).
#: Whether a model trained on CC-BY-NC-SA data is itself a derivative work of
#: that data, and whether a downstream *consumer* of the model inherits
#: share-alike, is legally unsettled. Nobody on this project is competent to
#: settle it, so the manifest carries the question and the conservative reading
#: rather than a verdict. A licence field that asserted an answer here would be
#: the same defect as one that asserted an unverified restriction.
DOWNSTREAM_REACH_QUESTION: Mapping[str, Any] = {
    "status": "unsettled -- no answer asserted",
    "question": (
        "Is a model trained on CC-BY-NC-SA-4.0 data a derivative work of that "
        "data, and does a downstream consumer of the model inherit share-alike?"
    ),
    "conservative_reading": (
        "Assume yes: treat the checkpoint as a derivative work carrying NC-SA, "
        "and assume a consumer that redistributes it or its outputs inherits the "
        "same obligation. This is the reading that fails safe; it is not a legal "
        "conclusion."
    ),
    "known_consumer": "~/Documents/robotics (packages/tms-lab, tms-core)",
    "consumer_state_measured_2026_08_06": (
        "not yet triggered: reports/robotics_integration.md records that no "
        "trained SC-WBD-001-beta checkpoint exists in that tree; ServedModel.load "
        "finds none and sets weights_status='analytic_backend'. The question "
        "becomes live the moment a trained checkpoint is served there."
    ),
    "resolve_with": "qualified legal advice, not this repository",
}


class ProvenanceMismatch(ValueError):
    """A checkpoint's tag claims a source-family set its manifest contradicts.

    Always an error. A tag is the only provenance most consumers will ever
    read, so a tag that overstates what trained the artifact is a
    misattribution, not a cosmetic problem.
    """


# ----------------------------------------------------------------------
# licence terms for sources that are not datasets
# ----------------------------------------------------------------------
#: Sources with no entry in ``scwbd/sources/cards/``. Each needs an explicit
#: term, because falling through to "no term" would read as "no constraints".
#:
#: ``tribe_v2_teacher`` is the one to read carefully. My brief states TRIBE v2
#: is CC BY-NC 4.0. **That licence is not recorded anywhere in this
#: repository**: ``configs/source_cards/tribe_v2_teacher.yaml`` has no
#: governance section and no licence field, and no other file states it. It is
#: therefore recorded with ``verified=False`` and provenance ``declared:brief``
#: and shows up in every manifest as unverified. It is *not* silently promoted
#: to fact by being repeated here.
NON_DATASET_TERMS: Mapping[str, LicenceTerm] = {
    "sim_wholebrain": LicenceTerm(
        source_id="sim_wholebrain",
        name="generated by scwbd.foundation.simulate (this repository)",
        noncommercial=False,
        share_alike=False,
        attribution=False,
        redistribution="full",
        provenance="scwbd/foundation/simulate.py",
        verified=True,
        notes=(
            "Our own simulator output carries no third-party term of its own. It is "
            "NOT unconditionally free: when the simulator is conditioned on the "
            "anatomical prior, the prior's terms flow through and are added "
            "separately as the 'anatomical_prior' term."
        ),
    ),
    "montage_calibration": LicenceTerm(
        source_id="montage_calibration",
        name="derived from the electrode montage of the linked EEG corpus",
        noncommercial=None,
        share_alike=None,
        attribution=None,
        redistribution="unknown",
        provenance="configs/source_cards/montage_calibration.yaml",
        verified=False,
        notes=(
            "The card declares no governance. Calibration constants are derived "
            "from a measured montage, so they inherit that corpus's terms, but the "
            "card does not say which corpus. Recorded unknown rather than assumed "
            "open."
        ),
    ),
    "negative_control_shuffled": LicenceTerm(
        source_id="negative_control_shuffled",
        name="phase-shuffled surrogate of the linked EEG corpus",
        noncommercial=None,
        share_alike=None,
        attribution=None,
        redistribution="unknown",
        provenance="configs/source_cards/negative_control_shuffled.yaml",
        verified=False,
        notes=(
            "Contributes an audit, never a gradient (gradient_permission is empty), "
            "so it does not enter the union of a trained artifact. Recorded so its "
            "absence from the union is visibly a consequence rather than an "
            "omission."
        ),
    ),
    "tribe_v2_teacher": LicenceTerm(
        source_id="tribe_v2_teacher",
        name="CC BY-NC 4.0 (ASSERTED, NOT VERIFIED IN THIS REPOSITORY)",
        noncommercial=True,
        share_alike=False,
        attribution=True,
        redistribution="unknown",
        url=None,
        provenance="declared:brief",
        verified=False,
        notes=(
            "configs/source_cards/tribe_v2_teacher.yaml carries no licence field and "
            "no governance section; no file in this repository states TRIBE v2's "
            "licence. The non-commercial term is carried here because acting as if "
            "an unverified restriction did not exist is the more dangerous error, "
            "but it is flagged unverified and must be confirmed against the "
            "upstream release before any commercial decision rests on it."
        ),
    ),
}


# ----------------------------------------------------------------------
# fingerprints
# ----------------------------------------------------------------------
def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file. Used for the weight hash that decides collapse."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_sha(repo: str | Path = ".") -> str:
    """Current commit, with ``-dirty`` when the tree has uncommitted changes.

    Returns ``"unknown"`` rather than raising: a checkpoint built outside a git
    tree still needs a provenance block, and ``"unknown"`` is a recordable
    answer where a crash is not.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def config_hash(config: Mapping[str, Any] | str | Path) -> str:
    """sha256 of the *canonicalised* config, so key order cannot change it."""
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text()) or {}
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFamilyManifest:
    """Which source families contributed to a checkpoint, and under what terms.

    Built from source cards, never from a filename. :meth:`validate_tag` is the
    check that makes a tag mean something.
    """

    records: tuple[SourceRecord, ...]
    dataset_links: Mapping[str, DatasetInfo | None] = field(default_factory=dict)
    #: Owner-chosen licence constraints: ``{term: reason}``. Empty means the
    #: licence is whatever the sources force and nothing more.
    policy: Mapping[str, str] = field(default_factory=dict)
    #: Anatomy assets that carry a non-commercial input, when the run used the
    #: real (non-fallback) anatomical prior. Empty tuple means "not used";
    #: ``None`` means "could not determine", which is not the same thing.
    anatomy_nc: Mapping[str, Sequence[str]] | None = None
    #: Was the anatomical prior biological, or the labelled synthetic fallback?
    #: ``None`` when the run did not record it.
    anatomy_is_biological: bool | None = None

    # -- families ---------------------------------------------------------
    @property
    def contributing(self) -> tuple[SourceRecord, ...]:
        """Sources that could actually have moved a weight in this run."""
        return tuple(r for r in self.records if r.contributes_gradient)

    @property
    def contributing_families(self) -> frozenset[str]:
        """Families with at least one gradient-bearing source.

        This is the set a tag is checked against. It is computed from
        ``enabled``, role and ``A_k`` — never from the tag.
        """
        return frozenset(r.family for r in self.contributing)

    @property
    def declared_families(self) -> frozenset[str]:
        """Every family present in the cards, contributing or not."""
        return frozenset(r.family for r in self.records)

    @property
    def tag_families(self) -> frozenset[str]:
        """Contributing families that a tag actually names (real/sim/synthetic)."""
        return self.contributing_families & frozenset(TAG_FAMILIES)

    @property
    def unknown_sources(self) -> tuple[str, ...]:
        """Sources whose family could not be determined. Recorded, never dropped."""
        return tuple(r.id for r in self.records if r.family == UNKNOWN)

    @property
    def excluded(self) -> tuple[tuple[str, str], ...]:
        """``(source_id, reason)`` for every source that contributed nothing.

        An empty reason is impossible by construction: each exclusion names the
        specific gate that stopped it. A source that vanished from a manifest
        without a reason is indistinguishable from one that was never declared.
        """
        out: list[tuple[str, str]] = []
        for r in self.records:
            if r.contributes_gradient:
                continue
            if not r.enabled:
                out.append((r.id, "card sets enabled: false"))
            elif r.role in ("negative_control", "evaluation_only"):
                out.append((r.id, f"role {r.role!r} licenses no loss family; contributes an audit, not a gradient"))
            elif not r.gradient_permission:
                out.append((r.id, "gradient permission A_k is empty; no parameter it may update"))
            else:  # pragma: no cover - defensive
                out.append((r.id, "excluded for an unrecorded reason; this is a bug in the manifest"))
        return tuple(out)

    # -- modality ---------------------------------------------------------
    @property
    def measured_modalities(self) -> tuple[str, ...]:
        """Union of measured modalities over contributing sources with a card.

        Two checkpoints may both be legitimately tagged ``-raw`` and differ
        substantially in what they saw; this is the field that shows it. An
        EEG-only ``-raw`` and one that also saw haemodynamics are different
        artifacts and the manifest must not render them identically.
        """
        out: set[str] = set()
        for r in self.contributing:
            info = self.dataset_links.get(r.id)
            if info is not None:
                out.update(info.measured_modalities)
        return tuple(sorted(out))

    @property
    def sources_without_dataset_card(self) -> tuple[str, ...]:
        return tuple(
            r.id for r in self.contributing if self.dataset_links.get(r.id) is None
        )

    # -- licence ----------------------------------------------------------
    def licence_terms(self) -> tuple[LicenceTerm, ...]:
        """One term per contributing source. Never silently short.

        A contributing source with neither a dataset card nor an entry in
        :data:`NON_DATASET_TERMS` gets :data:`UNKNOWN_TERM` rebound to its id,
        so it appears in the union as an unknown rather than not at all.
        """
        terms: list[LicenceTerm] = []
        for r in self.contributing:
            if r.id == "anatomical_prior":
                continue  # runtime-determined; added once by _anatomy_term below
            info = self.dataset_links.get(r.id)
            if info is not None:
                terms.append(info.licence)
            elif r.id in NON_DATASET_TERMS:
                terms.append(NON_DATASET_TERMS[r.id])
            else:
                terms.append(
                    LicenceTerm(
                        **{
                            **UNKNOWN_TERM.as_dict(),
                            "source_id": r.id,
                            "obligations": (),
                            "provenance": r.source_path or "unknown",
                        }
                    )
                )
        # The anatomical prior's terms are runtime-determined: the same card
        # means CC-BY-NC-SA-4.0 when agent C's atlas-derived connectome is on
        # disk and no third-party term at all when the labelled synthetic
        # fallback was used. Resolve it from the run, not from the card.
        terms.append(self._anatomy_term())
        return tuple(terms)

    def _anatomy_term(self) -> LicenceTerm:
        has_anatomy = any(r.id == "anatomical_prior" for r in self.contributing)
        if not has_anatomy:
            return LicenceTerm(
                source_id="anatomical_prior",
                name="not a contributing source in this run",
                noncommercial=False, share_alike=False, attribution=False,
                redistribution="full", provenance="configs/source_cards/",
                verified=True,
                notes="anatomical_prior did not contribute a gradient in this run",
            )
        if self.anatomy_is_biological is False:
            return LicenceTerm(
                source_id="anatomical_prior",
                name="labelled synthetic fallback connectome (no third-party inputs)",
                noncommercial=False, share_alike=False, attribution=False,
                redistribution="full",
                provenance="scwbd/foundation/anatomy.py (provenance='synthetic_fallback')",
                verified=True,
                notes=(
                    "The run used the synthetic fallback, so no Hansen/atlas terms "
                    "flow through. This also means the artifact supports no "
                    "anatomical claim at all (see scwbd/foundation/release.py)."
                ),
            )
        if self.anatomy_nc:
            via = sorted({k for v in self.anatomy_nc.values() for k in v})
            return LicenceTerm(
                source_id="anatomical_prior",
                name="CC-BY-NC-SA-4.0 (inherited from atlas inputs)",
                noncommercial=True, share_alike=True, attribution=True,
                redistribution="unknown",
                provenance="assets/MANIFEST.json + scwbd/anatomy/sources.py",
                verified=True,
                notes=(
                    "Inherited via " + ", ".join(via) + ". reports/anatomy_prior.md §6: "
                    "'Every derived map artifact records hansen_receptors in its inputs "
                    "and inherits the most restrictive input license.' This term is "
                    "independent of TRIBE: it makes the artifact non-commercial AND "
                    "share-alike even with no teacher source enabled."
                ),
            )
        return LicenceTerm(
            source_id="anatomical_prior",
            name="unknown (anatomy provenance not recorded for this run)",
            noncommercial=None, share_alike=None, attribution=None,
            redistribution="unknown",
            provenance="configs/source_cards/anatomical_prior.yaml",
            verified=False,
            notes=(
                "The card says 'agent C, or the labelled synthetic fallback' and the "
                "run did not record which. The distinction decides whether this "
                "artifact is non-commercial, so it is recorded as unknown and must be "
                "resolved before release."
            ),
        )

    def licence(self) -> LicenceUnion:
        """Effective licence: the union of contributing sources' terms."""
        return union_of(self.licence_terms(), policy=self.policy)

    # -- validation -------------------------------------------------------
    def validate_tag(self, tag: CheckpointTag | str) -> None:
        """Check that ``tag``'s claim matches this manifest. Raises on mismatch.

        Both directions are checked, because both are wrong:

        * **Overclaim** — the tag names a family that contributed nothing.
        * **Underclaim** — a family contributed and the tag does not name it.
          This is the ``-raw``-with-simulated-sources defect, and it is the
          more dangerous of the two: it understates what the artifact saw.
        """
        t = tag if isinstance(tag, CheckpointTag) else CheckpointTag.parse(tag)
        allowed = VARIANT_FAMILIES[t.variant]
        # Compare on the tag axis only. Auxiliary families (calibration,
        # anatomical prior, controls) are present in every arm and are reported
        # separately; gating the tag on them would make every run 'combined'.
        actual = self.contributing_families & frozenset(TAG_FAMILIES)

        forbidden = sorted(actual - allowed)
        if forbidden:
            offenders = {
                f: sorted(r.id for r in self.contributing if r.family == f)
                for f in forbidden
            }
            raise ProvenanceMismatch(
                f"checkpoint tagged {t.format()!r} claims variant {t.variant!r}, which "
                f"admits families {sorted(allowed)}, but its manifest shows "
                f"gradient-bearing sources from {forbidden}: {offenders}. "
                "The tag understates what trained this artifact. Retag it or disable "
                "the sources; the name is a provenance claim and must be true."
            )

        # A tag-axis family the variant requires but that contributed nothing.
        required = allowed & frozenset(TAG_FAMILIES)
        missing = sorted(required - actual)
        if missing:
            raise ProvenanceMismatch(
                f"checkpoint tagged {t.format()!r} claims variant {t.variant!r}, which "
                f"requires families {sorted(required)}, but nothing from {missing} "
                "contributed a gradient in this run. The tag overstates what trained "
                "this artifact. Use the variant that matches the manifest."
            )

    def validates(self, tag: CheckpointTag | str) -> bool:
        """Non-raising form, for reporting over a set of candidate tags."""
        try:
            self.validate_tag(tag)
            return True
        except ProvenanceMismatch:
            return False

    def best_variant(self) -> str:
        """The narrowest variant whose claim this manifest actually supports."""
        from .tags import VARIANT_ORDER

        actual = self.contributing_families & frozenset(TAG_FAMILIES)
        for v in VARIANT_ORDER:
            if VARIANT_FAMILIES[v] == actual:
                return v
        raise ProvenanceMismatch(
            f"no variant describes the contributing tag-axis families {sorted(actual)}. "
            "The taxonomy has no name for this mixture, which means either a source "
            "was enabled that the release family does not anticipate, or a family "
            "that every variant requires contributed nothing."
        )

    # -- D12 interface ----------------------------------------------------
    def d12_families(self) -> dict[str, list[str]]:
        """``{role bucket: [source ids]}`` for
        :func:`scwbd.bench.leakage.audit_dataset_family_breadth`.

        Keyed by D12's five role buckets rather than by this package's family
        names, because that is the vocabulary Appendix D's control is written
        in. Sources whose bucket is unknown are grouped under ``"unknown"`` and
        left for D12 to report as a gap — not quietly folded into a neighbour.
        """
        out: dict[str, list[str]] = {}
        for r in self.contributing:
            bucket = FAMILY_TO_D12_BUCKET.get(r.family, "unknown")
            out.setdefault(bucket, []).append(r.id)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def d12_roles(self) -> dict[str, str]:
        """``{family: role}`` companion argument for the same function."""
        return {k: k for k in self.d12_families()}

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        lic = self.licence()
        return {
            "contributing_families": sorted(self.contributing_families),
            "declared_families": sorted(self.declared_families),
            "tag_families": sorted(self.tag_families),
            "measured_modalities": list(self.measured_modalities),
            "integrity_tiers": {
                f: FAMILY_TIER.get(f) for f in sorted(self.contributing_families)
            },
            "integrity_tier_gaps": {
                f: TIER_GAP_REASON.get(f, "no reason recorded")
                for f in sorted(self.contributing_families)
                if FAMILY_TIER.get(f) is None
            },
            "sources": [
                {
                    "id": r.id,
                    "role": r.role,
                    "family": r.family,
                    "tier": FAMILY_TIER.get(r.family),
                    "enabled": r.enabled,
                    "contributes_gradient": r.contributes_gradient,
                    "gradient_permission": list(r.gradient_permission),
                    "dataset_card": (
                        self.dataset_links.get(r.id).card_path
                        if self.dataset_links.get(r.id) is not None else None
                    ),
                    "modalities": (
                        list(self.dataset_links[r.id].modalities)
                        if self.dataset_links.get(r.id) is not None else []
                    ),
                    "card": r.source_path,
                }
                for r in self.records
            ],
            "excluded": [{"id": i, "reason": why} for i, why in self.excluded],
            "unknown_family_sources": list(self.unknown_sources),
            "sources_without_dataset_card": list(self.sources_without_dataset_card),
            "anatomy": {
                "is_biological": self.anatomy_is_biological,
                "noncommercial_assets": (
                    {k: list(v) for k, v in self.anatomy_nc.items()}
                    if self.anatomy_nc else self.anatomy_nc
                ),
            },
            "licence": lic.as_dict(),
        }


# ----------------------------------------------------------------------
# provenance block
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProvenanceBlock:
    """The record written into every released checkpoint.

    Everything here is derived: the tag from the manifest, the licence from the
    sources, the hashes from the bytes. The only free text is
    ``claim_boundary``, which points at the document that says what the
    artifact may be used to assert rather than restating it — a summary of a
    claim boundary is how a claim boundary gets widened.
    """

    tag: CheckpointTag
    manifest: SourceFamilyManifest
    weights_sha256: str
    git_sha: str
    config_hash: str
    created_utc: str
    claim_boundary: str = CLAIM_BOUNDARY_PATH
    #: Set when this artifact is byte-identical to another variant.
    alias_of: str | None = None
    alias_reason: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        lic = self.manifest.licence()
        return {
            "schema": "scwbd-checkpoint-provenance/1.0.0",
            "tag": self.tag.format(),
            "tag_canonical": self.tag.format(as_alias=False),
            "variant": self.tag.variant,
            "is_alias": self.tag.is_alias,
            "timestamp_utc": self.tag.timestamp_text,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "weights_sha256": self.weights_sha256,
            "source_family_manifest": self.manifest.as_dict(),
            "effective_licence": lic.as_dict(),
            "licence_summary": lic.summary(),
            "claim_boundary": self.claim_boundary,
            "owner_licence_decision": dict(OWNER_LICENCE_DECISION),
            # both versions travel; a superseded decision that vanishes from the
            # artifact is indistinguishable from one that was never made
            "owner_licence_decision_history": [dict(d) for d in LICENCE_DECISION_HISTORY],
            "downstream_reach_question": dict(DOWNSTREAM_REACH_QUESTION),
            "alias_of": self.alias_of,
            "alias_reason": self.alias_reason,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False) + "\n")


def build_manifest(
    *,
    card_dir: str | Path = "configs/source_cards",
    config: Mapping[str, Any] | str | Path | None = None,
    policy: Mapping[str, str] | None = None,
    anatomy_is_biological: bool | None = None,
    assets_manifest: str | Path = "assets/MANIFEST.json",
    dataset_card_dir: str | Path | None = None,
) -> SourceFamilyManifest:
    """Build a manifest from a run's cards and config.

    ``anatomy_is_biological`` should come from the run's own evaluation record
    (``evaluation.json`` -> ``anatomy.is_biological``). Leaving it ``None``
    yields an explicitly unknown anatomy licence rather than a guess.
    """
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text()) or {}
    cfg: Mapping[str, Any] = config or {}
    records = tuple(load_source_records(card_dir))
    cards = load_dataset_cards(dataset_card_dir) if dataset_card_dir else load_dataset_cards()
    links = link_sources_to_datasets([r.id for r in records], config=cfg, cards=cards)
    nc = anatomy_nc_inputs(assets_manifest) if anatomy_is_biological is not False else {}
    return SourceFamilyManifest(
        records=records,
        dataset_links=links,
        policy=dict(policy or {}),
        anatomy_nc=nc if anatomy_is_biological is not None else None,
        anatomy_is_biological=anatomy_is_biological,
    )
