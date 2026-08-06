"""Source families: what actually trained a checkpoint, derived from its cards.

The checkpoint tag says which families trained an artifact.  This module works
out which families *did*, from the run's own source cards and gradient
permissions, so the two can be compared.  Nothing here reads a filename.

Two vocabularies meet here and they are not the same size, which is worth
stating plainly rather than smoothing over:

* :mod:`scwbd.foundation.mixture` has **seven** source roles (``prior``,
  ``likelihood``, ``boundary_target``, ``distillation``, ``calibration``,
  ``negative_control``, ``evaluation_only``).
* Appendix D's D12 control (``paper/appendix.tex`` l.1523, executable as
  ``scwbd.bench.leakage.audit_dataset_family_breadth``) names **five** role
  buckets: empirical, boundary-only, calibration, synthetic, evaluation-only.
* The owner's release taxonomy uses **three** tag-bearing families: real,
  simulation, synthetic.

The mapping between them is written down (:data:`ROLE_TO_D12_BUCKET`,
:func:`classify`) and tested, because an undocumented mapping between three
vocabularies is where provenance quietly goes wrong.  Where the mapping is
genuinely undetermined the answer is :data:`UNKNOWN` — recorded, counted and
reported.  It is never dropped, and it never defaults to a convenient bucket.

``torch`` is deliberately not imported here.  This module runs during release
and audit on machines that may have a training job holding the GPU, so it reads
source cards as YAML and duck-types anything richer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "REAL",
    "SIMULATION",
    "SYNTHETIC",
    "CALIBRATION",
    "BOUNDARY",
    "EVALUATION_ONLY",
    "NEGATIVE_CONTROL",
    "UNKNOWN",
    "TAG_FAMILIES",
    "AUXILIARY_FAMILIES",
    "ALL_FAMILIES",
    "ROLES",
    "ROLE_TO_D12_BUCKET",
    "FAMILY_TO_D12_BUCKET",
    "VARIANT_FAMILIES",
    "STRUCTURALLY_IDENTICAL_VARIANTS",
    "FAMILY_TIER",
    "TIER_GAP_REASON",
    "TIER_MEASUREMENT",
    "TIER_POPULATION_PRIOR",
    "TIER_SIMULATION",
    "TIER_TEACHER_PREDICTION",
    "SourceRecord",
    "classify",
    "load_source_records",
]

# -- family names -------------------------------------------------------
#: Measured data from real subjects.
REAL = "real"
#: Output of our own simulator (``scwbd.foundation.simulate``).
SIMULATION = "simulation"
#: Teacher-derived / distilled corpora (TRIBE v2 and anything like it).
SYNTHETIC = "synthetic"
#: Instrument calibration sources: gains, offsets, montages.
CALIBRATION = "calibration"
#: Boundary-condition / structural targets.
BOUNDARY = "boundary"
#: Sources held out of training entirely and used only to evaluate.
EVALUATION_ONLY = "evaluation_only"
#: Shuffled / scrambled controls that must *fail* to be informative.
NEGATIVE_CONTROL = "negative_control"
#: A source whose family could not be determined from its card.  This is a
#: value, not a gap: house rule is that absence writes something.
UNKNOWN = "unknown"

#: The three families the release tag actually names.
TAG_FAMILIES: tuple[str, ...] = (REAL, SIMULATION, SYNTHETIC)

#: Families that exist in the mixture but are not named by any tag.  They are
#: what makes ``-combined`` potentially distinct from
#: ``-with-simulation-and-synthetic``; see ``reports/checkpoint_family.md``.
AUXILIARY_FAMILIES: tuple[str, ...] = (
    CALIBRATION,
    BOUNDARY,
    EVALUATION_ONLY,
    NEGATIVE_CONTROL,
    UNKNOWN,
)

ALL_FAMILIES: tuple[str, ...] = TAG_FAMILIES + AUXILIARY_FAMILIES

#: Mirror of ``scwbd.foundation.mixture.ROLES``.  Duplicated rather than
#: imported so that this module stays torch-free; kept honest by
#: ``tests/release/test_families.py::test_role_vocabulary_matches_foundation``,
#: which imports the real one and asserts equality.  A copy with a test is a
#: fact; a copy without one is a future contradiction.
ROLES: tuple[str, ...] = (
    "prior",
    "likelihood",
    "boundary_target",
    "distillation",
    "calibration",
    "negative_control",
    "evaluation_only",
)

#: Role -> D12 role bucket, for ``audit_dataset_family_breadth(roles=...)``.
#: ``prior`` is absent on purpose: a prior may be a simulated corpus
#: (``synthetic`` bucket) or a structural prior (undetermined), and only the
#: card's ``is_simulated`` flag can tell them apart.  :func:`classify` resolves
#: it; a static table cannot, so this one does not pretend to.
ROLE_TO_D12_BUCKET: Mapping[str, str] = {
    "likelihood": "empirical",
    "boundary_target": "boundary-only",
    "calibration": "calibration",
    "distillation": "synthetic",
    "negative_control": "evaluation-only",
    "evaluation_only": "evaluation-only",
}

#: Family -> D12 bucket, for the families this module emits.
FAMILY_TO_D12_BUCKET: Mapping[str, str] = {
    REAL: "empirical",
    SIMULATION: "synthetic",
    SYNTHETIC: "synthetic",
    CALIBRATION: "calibration",
    BOUNDARY: "boundary-only",
    EVALUATION_ONLY: "evaluation-only",
    NEGATIVE_CONTROL: "evaluation-only",
    UNKNOWN: "unknown",
}

#: Which **tag-axis** families each variant claims contributed gradients.
#:
#: Only the three tag families (real, simulation, synthetic) are on this axis.
#: The auxiliary families are *infrastructure* — a calibration factor and an
#: anatomical prior are present in every arm of the ablation, so gating a
#: variant on them would make every real run ``combined`` and the taxonomy
#: would carry no information. :data:`AUXILIARY_FAMILIES` are therefore
#: admitted by every variant and reported separately in the manifest rather
#: than folded into the tag.
#:
#: Note what this makes visible: under the owner's own definition
#: (``combined`` = "all families"), ``combined`` and
#: ``with-simulation-and-synthetic`` name the **same set** on this axis. They
#: are not two arms; they are two names. :mod:`scwbd.release.collapse` will
#: refuse to mint both whenever they are trained, and
#: :data:`STRUCTURALLY_IDENTICAL_VARIANTS` records the fact up front rather
#: than leaving it to be discovered by a hash comparison at release time.
VARIANT_FAMILIES: Mapping[str, frozenset[str]] = {
    "raw": frozenset({REAL}),
    "with-simulation": frozenset({REAL, SIMULATION}),
    "with-simulation-and-synthetic": frozenset({REAL, SIMULATION, SYNTHETIC}),
    "combined": frozenset({REAL, SIMULATION, SYNTHETIC}),
}

#: Variant pairs whose tag-axis family sets are identical, i.e. that can never
#: describe different training mixtures. Reported by the release tooling as a
#: taxonomy fact, not discovered per-run.
_VARIANT_SEQUENCE: tuple[str, ...] = (
    "raw",
    "with-simulation",
    "with-simulation-and-synthetic",
    "combined",
)

STRUCTURALLY_IDENTICAL_VARIANTS: tuple[tuple[str, str], ...] = tuple(
    (a, b)
    for i, a in enumerate(_VARIANT_SEQUENCE)
    for b in _VARIANT_SEQUENCE[i + 1 :]
    if VARIANT_FAMILIES[a] == VARIANT_FAMILIES[b]
)


# -- integrity tiers ----------------------------------------------------
#: Provenance tiers, shared with 📐 Bernoulli's integrity-ordered curriculum.
#:
#: The tier boundary is **provenance, not modality**: fMRI is not lower
#: integrity than EEG, because both are measurements of a real subject. What
#: separates tiers is whether a number came from an instrument pointed at a
#: person, from a population summary, from a simulator, or from another model's
#: prediction.
#:
#: Tiers 1, 3, 4 and 5 were specified to me. **Tier 2 was not**, and it is left
#: unassigned rather than filled with a plausible guess — calibration is the
#: obvious candidate and that is exactly why inventing it would be hard to
#: notice later. :data:`FAMILY_TIER` maps what is known and returns ``None``
#: for what is not; ``None`` propagates into the manifest as an explicit
#: ``tier: null`` with a reason, never as a default.
TIER_MEASUREMENT = 1
TIER_UNASSIGNED_2: None = None  # reserved; no authority in this repo defines it
TIER_POPULATION_PRIOR = 3
TIER_SIMULATION = 4
TIER_TEACHER_PREDICTION = 5

#: Family -> integrity tier. ``None`` means "no tier has been assigned by any
#: source I could verify", which is a fact about our vocabulary, not about the
#: data.
FAMILY_TIER: Mapping[str, int | None] = {
    REAL: TIER_MEASUREMENT,
    SIMULATION: TIER_SIMULATION,
    SYNTHETIC: TIER_TEACHER_PREDICTION,
    CALIBRATION: None,       # candidate tier 2; unconfirmed, so unassigned
    BOUNDARY: None,
    EVALUATION_ONLY: None,   # held out of training: no curriculum position
    NEGATIVE_CONTROL: None,
    UNKNOWN: None,
}

#: Why a family has no tier. Present for every ``None`` above so the manifest
#: can say *why* rather than merely omit.
TIER_GAP_REASON: Mapping[str, str] = {
    CALIBRATION: (
        "calibration estimates instrument gains and offsets from measured "
        "hardware; it is plausibly tier 2 but no specification in this "
        "repository defines tier 2, so it is left unassigned rather than "
        "guessed"
    ),
    BOUNDARY: "no tier specified for boundary-condition targets",
    EVALUATION_ONLY: "held out of training entirely; has no curriculum position",
    NEGATIVE_CONTROL: "contributes an audit, never a gradient; not on the curriculum",
    UNKNOWN: "family itself is undetermined, so no tier can follow from it",
}


@dataclass(frozen=True)
class SourceRecord:
    """The provenance-relevant projection of one source card.

    Deliberately a small structure: it carries what decides *family* and
    *whether this source could move a weight*, and nothing about losses,
    reliability or optimisation.  Built from a YAML card
    (:meth:`from_mapping`) or duck-typed from a live
    ``scwbd.foundation.mixture.SourceSpec`` (:meth:`from_spec`).
    """

    id: str
    role: str
    enabled: bool = True
    is_simulated: bool = False
    is_teacher: bool = False
    gradient_permission: tuple[str, ...] = ()
    frozen: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    source_path: str | None = None

    # -- the two questions this record exists to answer ------------------
    @property
    def family(self) -> str:
        return classify(self)

    @property
    def contributes_gradient(self) -> bool:
        """Could this source have moved a weight in this run?

        Three ways the answer is no, and all three are checked because each has
        occurred in a real mixture:

        1. the card is ``enabled: false`` (TRIBE v2 today);
        2. the role licenses no loss family at all (``negative_control`` and
           ``evaluation_only`` contribute an audit, never a gradient);
        3. ``A_k`` is empty, so there is no parameter it is permitted to touch.

        This is what makes the manifest independent of the filename: a family
        counts as having trained the artifact only if some source in it could
        actually have changed a number.
        """
        if not self.enabled:
            return False
        if self.role in ("negative_control", "evaluation_only"):
            return False
        return bool(self.gradient_permission)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source_path: str | None = None) -> "SourceRecord":
        sid = data.get("id")
        if not sid:
            raise ValueError(
                f"source card {source_path or '<mapping>'} has no 'id'; an anonymous "
                "source cannot be attributed to a family, and an unattributable source "
                "may not enter a release manifest."
            )
        role = str(data.get("role", "likelihood"))
        return cls(
            id=str(sid),
            role=role,
            enabled=bool(data.get("enabled", True)),
            is_simulated=bool(data.get("is_simulated", False)),
            is_teacher=bool(data.get("is_teacher", False)),
            gradient_permission=tuple(data.get("gradient_permission", ()) or ()),
            frozen=tuple(data.get("frozen", ()) or ()),
            losses=tuple(data.get("losses", ()) or ()),
            source_path=source_path,
        )

    @classmethod
    def from_spec(cls, spec: Any, *, source_path: str | None = None) -> "SourceRecord":
        """Duck-type a live ``SourceSpec`` without importing torch."""
        g = lambda name, default: getattr(spec, name, default)  # noqa: E731
        return cls(
            id=str(g("id", "")),
            role=str(g("role", "likelihood")),
            enabled=bool(g("enabled", True)),
            is_simulated=bool(g("is_simulated", False)),
            is_teacher=bool(g("is_teacher", False)),
            gradient_permission=tuple(g("gradient_permission", ()) or ()),
            frozen=tuple(g("frozen", ()) or ()),
            losses=tuple(g("losses", ()) or ()),
            source_path=source_path,
        )


def classify(record: SourceRecord) -> str:
    """Family of one source, from its card fields only.

    Order matters.  ``is_teacher`` is checked before ``is_simulated`` and both
    before ``role``, because a teacher corpus is synthetic regardless of how it
    was produced, and the role vocabulary cannot express the
    simulated/structural split inside ``prior``.

    A non-simulated ``prior`` (``anatomical_prior`` in this repo) returns
    :data:`UNKNOWN`.  That is a decision, not an oversight: a structural prior
    is neither measured subject data nor simulator output nor teacher-derived,
    and the tag vocabulary has no name for it.  Guessing ``boundary`` would put
    an unreviewed claim into every manifest, so it is recorded as unknown and
    surfaces in ``SourceFamilyManifest.unknown_sources``.  See
    ``reports/checkpoint_family.md`` §"Open questions".
    """
    if record.is_teacher or record.role == "distillation":
        return SYNTHETIC
    if record.is_simulated:
        return SIMULATION
    if record.role == "likelihood":
        return REAL
    if record.role == "calibration":
        return CALIBRATION
    if record.role == "boundary_target":
        return BOUNDARY
    if record.role == "negative_control":
        return NEGATIVE_CONTROL
    if record.role == "evaluation_only":
        return EVALUATION_ONLY
    return UNKNOWN


def load_source_records(card_dir: str | Path) -> list[SourceRecord]:
    """Load every ``*.yaml`` source card in ``card_dir``.

    Sorted by id so a manifest built twice from the same directory is
    byte-identical.  An unreadable card raises: skipping it would silently drop
    a family from the provenance record, which is the exact failure this
    package exists to prevent.
    """
    d = Path(card_dir)
    if not d.is_dir():
        raise FileNotFoundError(
            f"source card directory {d} does not exist. A checkpoint whose cards "
            "cannot be found has no verifiable provenance and may not be tagged."
        )
    out: list[SourceRecord] = []
    for p in sorted(d.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception as exc:  # pragma: no cover - malformed YAML on disk
            raise ValueError(f"source card {p} does not parse: {exc}") from exc
        out.append(SourceRecord.from_mapping(data, source_path=str(p)))
    if not out:
        raise FileNotFoundError(
            f"no source cards in {d}. Training without source cards is what "
            "ARCHITECTURE.md §7 rule 2 forbids; releasing without them is worse."
        )
    return sorted(out, key=lambda r: r.id)
