"""Typed annotations over one region identity -- sec. 2b, O-3, O-4 and O-7.

Three sec. 2b items, one omission
---------------------------------
O-3 asks for region properties to become annotations with their own provenance,
licence, coverage and admissibility.  O-4 asks for epistemic status to be
*derived*.  O-7 asks for the two region vocabularies to become one.  They are
the same defect that O-1 and O-2 are:

    **a value that does not carry the support it is indexed by.**

``RegionFamily`` is a record per *family*; a per-parcel label vector is a record
per *parcel*.  Reading one as the other is not a naming problem, it is a support
mismatch -- 9 elements read as 414 -- and it happened three times in one day
because nothing in the type system knew which support a value was indexed by.
:class:`Annotation` carries a :class:`~scwbd.schema.supports.Support`, and
:meth:`AnnotationSet.values_on` refuses to hand a value to a consumer expecting
a different one.  That is the whole fix; everything else here follows from it.

What is here
------------
* :class:`Annotation` -- one named quantity over one support, with provenance,
  licence, coverage and ``admissible_for``.
* :class:`AnnotationSet` -- annotations keyed by ``(name, support)``, with the
  support check on read.
* :func:`derived_training_status` / :func:`derived_evidence_tier` -- O-4: the
  status is computed from whether admissible annotations with data exist, and
  there is nothing to declare and therefore nothing to be inconsistent with.

What is deliberately *not* here
-------------------------------
Arrays.  An annotation names where its values live (``values_ref``) and how many
of them there are; it does not carry them, because a frozen content-hashable
schema object that carries a 414-vector is a schema object whose hash changes
when a map is rebuilt.  :meth:`AnnotationSet.values_on` is the checked accessor
that a value provider plugs into.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .supports import ElementSpec, OntologyError, Support
from .units import Unit

__all__ = [
    "AnnotationPurpose",
    "EvidenceStatus",
    "Licence",
    "Coverage",
    "AnnotationProvenance",
    "Annotation",
    "AnnotationSet",
    "SupportMismatch",
    "InadmissibleEvidence",
    "derived_training_status",
    "derived_evidence_tier",
]


class SupportMismatch(OntologyError):
    """A value indexed by one support was requested as if indexed by another.

    The three-times-in-one-day bug, given a name and a type that raises.
    """


class InadmissibleEvidence(OntologyError):
    """An annotation was cited for a purpose it declares itself barred from."""


#: What an annotation may be *used for*.  Barring is a property of the datum,
#: not a rule someone remembers: cytoarchitecture is carried, is real, and fails
#: globally on every measured block, so it declares ``admissible_for=()`` and
#: any attempt to cite it as the reason a family exists raises.
AnnotationPurpose = Literal[
    "family_separation",
    "operator_assignment",
    "state_prior",
    "observation_operator",
    "registration",
    "description",
    "evaluation",
]

#: What an annotation's values mean.  ``measured_not_separating`` is retained
#: from ``scwbd.anatomy.families.FIELD_STATUS`` deliberately: it is a real
#: distinction that the anatomy prior earned and that this type must not lose.
EvidenceStatus = Literal[
    "measured",
    "measured_not_separating",
    "prior_only",
    "derived",
    "not_established",
]


class Licence(SchemaModel):
    """The **one** licence surface.

    ``membership_licence`` and ``provenance[].licence`` are currently both
    authoritative for different fields of the same object, so answering "may
    this reach a checkpoint?" requires knowing which field you came from.  An
    annotation carries its own licence and there is nothing else to consult.
    """

    spdx: str
    #: True when the licence forbids commercial use.  Carried as a flag rather
    #: than re-derived from ``spdx`` at each call site, because re-deriving it
    #: is where a "NC" gets missed.
    non_commercial: bool = False
    attribution: str = ""
    may_reach_checkpoint: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Licence":
        if not self.spdx.strip():
            raise OntologyError("a licence must name an identifier")
        if self.non_commercial and self.may_reach_checkpoint:
            raise OntologyError(
                f"licence {self.spdx!r} is non-commercial but declares "
                "may_reach_checkpoint=True",
                remedy=(
                    "The two cannot both be true under the shipped checkpoint "
                    "policy. If an exemption exists it is a claim-manifest "
                    "override, not a field on the licence."
                ),
                offending_object=self.spdx,
            )
        return self


class Coverage(SchemaModel):
    """How much of the support this annotation actually has values for.

    Two numbers, kept separate because they answer different questions and the
    anatomy prior already measures both: ``n_with_value / n_elements`` is what
    fraction of elements carry a value, and ``weighted_fraction`` is what
    fraction of the *support's measure* (cortical area, volume) they cover.  A
    parcellation can be 95 % covered by count and 60 % by area.
    """

    n_elements: int = Field(ge=0)
    n_with_value: int = Field(ge=0)
    weighted_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_name: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Coverage":
        if self.n_with_value > self.n_elements:
            raise OntologyError(
                f"coverage claims {self.n_with_value} values over "
                f"{self.n_elements} elements"
            )
        if self.weighted_fraction is not None and not self.weight_name:
            raise OntologyError(
                "a weighted coverage fraction must name the weight it is "
                "weighted by (area? volume? number of vertices?)",
                remedy="An unnamed weight is a number nobody can reproduce.",
            )
        return self

    @property
    def fraction(self) -> float:
        return self.n_with_value / self.n_elements if self.n_elements else 0.0

    @property
    def is_complete(self) -> bool:
        return self.n_elements > 0 and self.n_with_value == self.n_elements


class AnnotationProvenance(SchemaModel):
    """Where an annotation's values came from and exactly how."""

    source_key: str
    citation: str
    method: str
    licence: Licence
    asset_hash: str | None = None
    #: Free-form record of the null/test that established (or failed to
    #: establish) this annotation, when one exists.
    evidence: dict[str, Any] = Field(default_factory=dict)


class Annotation(SchemaModel):
    """One named quantity over one support, with everything needed to use it.

    The two fields that do the work:

    ``support``
        What the values are indexed by.  A ``FamilyPartition`` is an annotation
        on a support of 9 elements; a per-parcel label vector is an annotation
        on a support of 414.  A consumer asks for values *on a named support*
        and gets a :class:`SupportMismatch` otherwise.
    ``admissible_for``
        What this datum may be cited for.  Empty means it may be carried and
        described and cited for nothing.  Cytoarchitecture is the worked case.
    """

    id: str
    name: str
    #: What the values are indexed by.  Its ``element`` says whether each entry
    #: is a scalar, a 3-vector, or a label.
    support: Support
    units: Unit
    status: EvidenceStatus
    provenance: AnnotationProvenance
    coverage: Coverage
    #: Purposes this annotation may be cited for.  ``()`` is a bar, not an
    #: oversight, and :meth:`assert_admissible` says which.
    admissible_for: tuple[AnnotationPurpose, ...] = ()
    #: Why it is barred, required whenever ``admissible_for`` is empty.
    bar_reason: str = ""
    #: Where the values live.  ``None`` means the annotation is declared and has
    #: no data behind it, which :func:`derived_training_status` reads.
    values_ref: str | None = None
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Annotation":
        if self.support.element is None:
            raise OntologyError(
                f"annotation {self.id!r} declares a support with no ElementSpec",
                remedy=(
                    "An annotation whose support has no arity cannot say whether "
                    "each entry is one number or three, which is exactly the "
                    "confusion O-2 exists to remove."
                ),
                offending_object=self.id,
            )
        if not self.units.same_dimension(self.support.element.units):
            raise OntologyError(
                f"annotation {self.id!r} declares units {self.units!r} over a "
                f"support whose elements are {self.support.element.units!r}"
            )
        if self.support.n_elements is not None and (
            self.coverage.n_elements != self.support.n_elements
        ):
            raise OntologyError(
                f"annotation {self.id!r}: coverage is over "
                f"{self.coverage.n_elements} elements but its support has "
                f"{self.support.n_elements}",
                remedy="Coverage is a statement about this support, not another.",
                offending_object=self.id,
            )
        if not self.admissible_for and not self.bar_reason:
            raise OntologyError(
                f"annotation {self.id!r} is admissible for nothing and gives no "
                "reason",
                remedy=(
                    "An empty admissible_for is a bar and must say why. "
                    "Otherwise it reads as an oversight and someone will 'fix' it."
                ),
                offending_object=self.id,
            )
        if self.status == "not_established" and self.values_ref is not None:
            raise OntologyError(
                f"annotation {self.id!r} is 'not_established' but names values at "
                f"{self.values_ref!r}",
                remedy="A value that exists is established; say what it is.",
            )
        if self.coverage.n_with_value > 0 and self.values_ref is None:
            raise OntologyError(
                f"annotation {self.id!r} claims {self.coverage.n_with_value} "
                "values but names no values_ref",
                remedy="Coverage counts values; if there are none, coverage is 0.",
            )
        return self

    # -- admissibility ------------------------------------------------------
    @property
    def has_data(self) -> bool:
        """True when the annotation has measured values behind it."""
        return (
            self.values_ref is not None
            and self.coverage.n_with_value > 0
            and self.status in ("measured", "measured_not_separating")
        )

    @property
    def is_barred(self) -> bool:
        return not self.admissible_for

    def is_admissible_for(self, purpose: str) -> bool:
        return purpose in self.admissible_for

    def assert_admissible(self, purpose: str) -> "Annotation":
        """Raise unless this annotation may be cited for ``purpose``."""
        if self.is_admissible_for(purpose):
            return self
        raise InadmissibleEvidence(
            f"annotation {self.id!r} ({self.name!r}) is not admissible for "
            f"{purpose!r}; admissible_for={list(self.admissible_for)}"
            + (f"; barred because: {self.bar_reason}" if self.is_barred else ""),
            remedy=(
                "Admissibility is a property of the datum. If this evidence "
                "should count, the thing to change is the measurement that "
                "barred it, not the call site."
            ),
            offending_object=self.id,
        )

    # -- support checking ---------------------------------------------------
    def assert_on(self, support: Support | str) -> "Annotation":
        """Raise unless this annotation is indexed by ``support``.

        The check is on the *shape of the index*, not on object identity:
        ``n_elements`` and per-element arity must agree, and the resolution/label
        must agree when both declare one.  That is enough to catch a 9-element
        family record being read as a 414-element parcel vector, which is the
        bug this type exists for.
        """
        mine = self.support
        if isinstance(support, str):
            if (mine.label or mine.resolution or "") != support:
                raise SupportMismatch(
                    f"annotation {self.id!r} is indexed by "
                    f"{mine.label or mine.resolution!r}, requested on {support!r}",
                    offending_object=self.id,
                )
            return self
        if mine.n_elements != support.n_elements:
            raise SupportMismatch(
                f"annotation {self.id!r} has {mine.n_elements} elements "
                f"({mine.label or mine.kind}); requested on a support of "
                f"{support.n_elements} ({support.label or support.kind})",
                remedy=(
                    "These are different supports. A per-family record is not a "
                    "per-parcel vector; broadcasting one onto the other is the "
                    "defect O-7 names."
                ),
                offending_object=self.id,
            )
        a_arity = mine.element.arity if mine.element else None
        b_arity = support.element.arity if support.element else None
        if a_arity is not None and b_arity is not None and a_arity != b_arity:
            raise SupportMismatch(
                f"annotation {self.id!r} has arity {a_arity}; requested on a "
                f"support of arity {b_arity}",
                offending_object=self.id,
            )
        if (
            mine.label
            and support.label
            and mine.label != support.label
            and mine.n_elements == support.n_elements
        ):
            raise SupportMismatch(
                f"annotation {self.id!r} is indexed by {mine.label!r} and was "
                f"requested on {support.label!r}; the two happen to have the same "
                "size, which is exactly when this goes unnoticed",
                offending_object=self.id,
            )
        return self


class AnnotationSet(SchemaModel):
    """Annotations over one region identity, addressable and checked.

    O-7's "one region ontology" in one object: the regions are named once, and
    everything else is an annotation keyed to a support over them.
    """

    #: The identity support -- the region index everything else is keyed to.
    regions: Support
    #: Ordered region ids.  This is the *only* region vocabulary.
    region_ids: tuple[str, ...]
    annotations: tuple[Annotation, ...] = ()
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "AnnotationSet":
        if self.regions.n_elements is not None and len(self.region_ids) != (
            self.regions.n_elements
        ):
            raise OntologyError(
                f"{len(self.region_ids)} region ids over a support of "
                f"{self.regions.n_elements} elements"
            )
        dupes = {r for r in self.region_ids if self.region_ids.count(r) > 1}
        if dupes:
            raise OntologyError(f"duplicate region ids: {sorted(dupes)[:5]}")
        ids = [a.id for a in self.annotations]
        adupes = {i for i in ids if ids.count(i) > 1}
        if adupes:
            raise OntologyError(f"duplicate annotation ids: {sorted(adupes)}")
        return self

    def get(self, name: str, *, on: Support | str | None = None) -> Annotation:
        """The annotation called ``name``, checked against ``on`` if given."""
        matches = [a for a in self.annotations if a.name == name]
        if not matches:
            raise KeyError(
                f"no annotation named {name!r}; have "
                f"{sorted({a.name for a in self.annotations})}"
            )
        if on is None:
            if len(matches) > 1:
                raise SupportMismatch(
                    f"{len(matches)} annotations named {name!r} on different "
                    f"supports ({[m.support.label or m.support.kind for m in matches]}); "
                    "name the support you want",
                    remedy="Ambiguity between supports is the bug; resolving it "
                    "silently by taking the first is how it stayed hidden.",
                )
            return matches[0]
        errors = []
        for m in matches:
            try:
                return m.assert_on(on)
            except SupportMismatch as e:  # keep looking; report all on failure
                errors.append(str(e))
        raise SupportMismatch(
            f"no annotation named {name!r} on the requested support; tried "
            f"{len(matches)}: " + " | ".join(errors)
        )

    def for_purpose(self, purpose: str) -> tuple[Annotation, ...]:
        """Every annotation admissible for ``purpose`` -- barred ones excluded."""
        return tuple(a for a in self.annotations if a.is_admissible_for(purpose))

    def barred(self) -> tuple[Annotation, ...]:
        return tuple(a for a in self.annotations if a.is_barred)

    def non_commercial(self) -> tuple[Annotation, ...]:
        """Annotations whose licence forbids reaching a checkpoint."""
        return tuple(
            a for a in self.annotations if not a.provenance.licence.may_reach_checkpoint
        )


# ==========================================================================
# O-4: status is derived, never declared
# ==========================================================================
def derived_training_status(
    annotations: tuple[Annotation, ...], *, purpose: str = "state_prior"
) -> Literal["has_regional_data", "prior_only_untrained"]:
    """O-4: derive training status instead of declaring it.

    ``has_regional_data`` exactly when at least one annotation is **admissible
    for the purpose** and **has data behind it**.  Both clauses matter and both
    are already recorded on the annotation, which is the point: two declared
    fields that cannot disagree are one field and a chance to be inconsistent.
    """
    for a in annotations:
        if a.is_admissible_for(purpose) and a.has_data:
            return "has_regional_data"
    return "prior_only_untrained"


def derived_evidence_tier(
    annotations: tuple[Annotation, ...],
) -> Literal["measured_separation", "atlas_separation", "synthetic"]:
    """O-4: derive the evidence tier from the admissible annotations.

    ``measured_separation`` requires an annotation admissible for
    ``family_separation`` that has measured data; ``synthetic`` is reserved for
    the case where the only annotations present are declared without data *and*
    without a real atlas behind them.  Nothing here can be set by declaration.
    """
    separating = [
        a
        for a in annotations
        if a.is_admissible_for("family_separation") and a.has_data
    ]
    if separating:
        return "measured_separation"
    real = [
        a
        for a in annotations
        if a.provenance.source_key and a.status != "not_established"
    ]
    return "atlas_separation" if real else "synthetic"
