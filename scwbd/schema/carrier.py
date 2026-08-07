"""The carrier and its views -- ARCHITECTURE.md sec. 2b, O-1.

    "There is no object for the thing sources are observations *of*.  The latent
     is implicitly 'the dense (B,T,N,D) state tensor', so an observation
     operator maps from a *slice of an array* rather than from a declared
     field."

This module names that object.

A :class:`Carrier` is what the model owns: an ordered, region-indexed,
**heterogeneous** field.  A :class:`View` is what one source sees of it --
``(operator, support, uncertainty)`` -- and belongs to a ``SourceCard``, never
to the model.  Two electrode montages with different channel counts and
different lead fields are then two views of one carrier, and neither needs
anything built for it that the other does not.

The invariant this module exists to make explicit
-------------------------------------------------
``Support.n_elements`` is **not** the dimension of the space a support carries.
The dimension is

    n_dof = sum over elements of arity(element)

and every shape in the system -- a lead field's column count, a restriction's
input width, the width of a state plane -- is a statement about ``n_dof``, not
about ``n_elements``.  ``LeadField`` being ``(n_channels, n_regions)`` is the
special case ``arity == 1`` with the orientation silently spent; the free
orientation case is ``(n_channels, n_regions, 3)`` and the two differ by a
factor of 9 in retained lead-field energy (``reports/transforms/
resolution_pair.md`` sec. 3.5: 5.6 % against 51.7 %).  A type that cannot tell
those apart cannot refuse the wrong one, which is why :class:`ElementSpec`
exists and why ``Support`` now carries one.

Heterogeneity
-------------
``body.tex`` sec. 2.1 writes ``X_i in 𝒳_i``: the state *space* carries the
index.  :attr:`Carrier.overrides` is where that lives -- a per-element
``ElementSpec`` that need not agree with any other element's.  :meth:`Carrier.
offsets` therefore returns a **ragged** layout (prefix sums of per-element
arity), which is the interface O-6 needs; this module declares it and does not
implement the storage behind it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import FrameId
from .ledger import UncertaintyLedger
from .supports import (
    PSF,
    ArityError,
    ElementKind,
    ElementSpec,
    OntologyError,
    Support,
    TemporalSupport,
)
from .units import Unit

__all__ = [
    "ElementKind",
    "ElementSpec",
    "CarrierElement",
    "Carrier",
    "ViewOperator",
    "View",
    "OntologyError",
    "ArityError",
    "ViewMismatch",
]


class ViewMismatch(OntologyError):
    """A view's operator does not compose with the carrier it names."""


class CarrierElement(SchemaModel):
    """One element of a carrier: an identity and what it carries."""

    id: str
    spec: ElementSpec


class Carrier(SchemaModel):
    """The field sources are observations *of*.  The model owns one.

    ``support`` gives the geometry and the *default* element spec; ``overrides``
    give the per-element heterogeneity of ``body.tex`` sec. 2.1.  Nothing here
    stores values -- a ``Carrier`` is a declaration of a space, and its
    :meth:`offsets` are the layout any storage must honour.
    """

    id: str
    support: Support
    frame: FrameId
    #: Ordered element ids.  Empty means the elements are anonymous and indexed
    #: by position, which is admissible but blocks per-element overrides.
    element_ids: tuple[str, ...] = ()
    #: Per-element spec overrides, keyed by element id.  This is where
    #: "the components need not have equal shape" actually lives.
    overrides: tuple[CarrierElement, ...] = ()
    #: Ids of annotations/priors that constrain this carrier -- a connectome is
    #: one of these, which is the only place it can constrain every view at once.
    priors: tuple[str, ...] = ()
    ledger: UncertaintyLedger | None = None
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Carrier":
        if self.support.element is None:
            raise OntologyError(
                f"carrier {self.id!r} declares a support with no ElementSpec, so "
                "its dimension is not defined",
                remedy=(
                    "Declare Support.element. n_elements is not the dimension of "
                    "the space; sum-of-arity is."
                ),
                offending_object=self.support,
            )
        n = self.support.n_elements
        if self.element_ids:
            if n is not None and len(self.element_ids) != n:
                raise OntologyError(
                    f"carrier {self.id!r} names {len(self.element_ids)} elements "
                    f"but its support declares n_elements={n}",
                    remedy="One id per element, in support order.",
                )
            dupes = {e for e in self.element_ids if self.element_ids.count(e) > 1}
            if dupes:
                raise OntologyError(
                    f"carrier {self.id!r} has duplicate element ids: {sorted(dupes)}"
                )
        known = set(self.element_ids)
        for ov in self.overrides:
            if not self.element_ids:
                raise OntologyError(
                    f"carrier {self.id!r} declares per-element overrides but no "
                    "element_ids to key them by",
                    remedy="Name the elements before overriding them.",
                    offending_object=ov.id,
                )
            if ov.id not in known:
                raise OntologyError(
                    f"carrier {self.id!r} overrides unknown element {ov.id!r}",
                    remedy="An override keyed by a name the carrier does not have "
                    "is silently dropped otherwise.",
                    offending_object=ov.id,
                )
        return self

    # -- the dimension, which is the whole point ----------------------------
    @property
    def default_spec(self) -> ElementSpec:
        assert self.support.element is not None  # enforced in _check
        return self.support.element

    def spec_for(self, element: str | int) -> ElementSpec:
        """The spec of one element, honouring overrides."""
        if isinstance(element, int):
            if not self.element_ids:
                return self.default_spec
            element = self.element_ids[element]
        for ov in self.overrides:
            if ov.id == element:
                return ov.spec
        return self.default_spec

    @property
    def n_elements(self) -> int:
        if self.element_ids:
            return len(self.element_ids)
        if self.support.n_elements is None:
            raise OntologyError(
                f"carrier {self.id!r} declares neither element_ids nor "
                "support.n_elements, so it has no size"
            )
        return int(self.support.n_elements)

    @property
    def is_homogeneous(self) -> bool:
        return not self.overrides

    def arities(self) -> tuple[int, ...]:
        return tuple(self.spec_for(i).arity for i in range(self.n_elements))

    @property
    def n_dof(self) -> int:
        """``sum_i arity_i`` -- the dimension of the space, ragged or not."""
        return int(sum(self.arities()))

    def offsets(self) -> tuple[int, ...]:
        """Prefix sums of arity: the ragged layout, length ``n_elements + 1``.

        A padded layout is recoverable from this (pad each span to
        ``max(arities)``); the reverse is not, which is why the ragged form is
        the declaration and the padded one is the storage narrowing.
        """
        out = [0]
        for a in self.arities():
            out.append(out[-1] + a)
        return tuple(out)

    def span(self, element: str | int) -> tuple[int, int]:
        """Half-open ``[start, stop)`` of one element in the flat dof vector."""
        idx = element
        if isinstance(element, str):
            if not self.element_ids:
                raise OntologyError(
                    f"carrier {self.id!r} has anonymous elements; index by position"
                )
            idx = self.element_ids.index(element)
        off = self.offsets()
        return off[idx], off[idx + 1]

    @property
    def carries_orientation(self) -> bool:
        """True when *any* element carries more than one number."""
        return any(a > 1 for a in self.arities())


class ViewOperator(SchemaModel):
    """The declared linear map ``carrier -> view support``.

    ``shape`` is ``(n_view_dof, n_carrier_dof)`` and is checked against both
    ends.  A lead field is one of these; so is a parcel restriction, so is a
    haemodynamic convolution.  Declaring the shape in dof rather than in
    elements is the fix: ``(n_channels, n_regions)`` is only correct when every
    region carries exactly one number, and nothing today says whether it does.
    """

    kind: Literal[
        "lead_field",
        "projection",
        "restriction",
        "aggregation",
        "identity",
        "kernel",
        "learned",
    ]
    #: ``(n_view_dof, n_carrier_dof)``.
    shape: tuple[int, int]
    #: Units of the map itself, so that ``operator.units * carrier.units`` has
    #: the dimension of the view's units.  For an EEG lead field on a dipole
    #: carrier this is ``V/(A*m)``.
    units: Unit
    #: Where the realised operator lives (asset id / path).  ``None`` means the
    #: map is declared but not materialised, which callers must be able to see.
    operator_ref: str | None = None
    linear: bool = True
    psf: PSF | None = None
    ledger: UncertaintyLedger | None = None
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ViewOperator":
        if any(s <= 0 for s in self.shape):
            raise OntologyError(f"view operator shape must be positive, got {self.shape}")
        return self

    @property
    def is_materialised(self) -> bool:
        return self.operator_ref is not None


class View(SchemaModel):
    """``(operator, support, uncertainty)`` -- what one source sees.

    Owned by a ``SourceCard``, never by the model.  ``carrier`` names the object
    the view is a view *of*, which is the target
    ``ObservationModel.target_ports`` was reaching for.
    """

    id: str
    #: Id of the :class:`Carrier` this view observes.
    carrier: str
    operator: ViewOperator
    #: The view's own support -- its channels, voxels, or bands.
    support: Support
    temporal: TemporalSupport
    ledger: UncertaintyLedger
    #: Free text naming the instrument/montage, for reports.
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "View":
        if self.support.element is None:
            raise OntologyError(
                f"view {self.id!r} declares a support with no ElementSpec",
                remedy="A view's support has a dimension too; declare it.",
                offending_object=self.support,
            )
        n = self.support.n_elements
        if n is not None:
            expected = n * self.support.element.arity
            if self.operator.shape[0] != expected:
                raise ViewMismatch(
                    f"view {self.id!r}: operator produces {self.operator.shape[0]} "
                    f"dof but its support has {n} elements of arity "
                    f"{self.support.element.arity} = {expected} dof",
                    remedy="n_elements is not the dimension; sum-of-arity is.",
                    offending_object=self.id,
                )
        return self

    def check_against(self, carrier: Carrier) -> None:
        """Raise unless this view actually composes with ``carrier``.

        Three separate claims, each able to fail on its own:

        1. the view names this carrier;
        2. the operator's input width equals the carrier's ``n_dof``;
        3. ``operator.units * carrier element units`` has the dimension of the
           view's own units.

        (2) is where the scalar/vector confusion surfaces: a lead field built
        for a free-orientation source space has three times the input width of
        one built for a fixed-orientation space, and today both are called
        ``(n_channels, n_regions)``.
        """
        if self.carrier != carrier.id:
            raise ViewMismatch(
                f"view {self.id!r} names carrier {self.carrier!r}, checked against "
                f"{carrier.id!r}",
                offending_object=self.carrier,
            )
        if self.operator.shape[1] != carrier.n_dof:
            hint = ""
            arities = set(carrier.arities())
            if arities == {1} and self.operator.shape[1] == 3 * carrier.n_dof:
                hint = (
                    " -- the operator is free-orientation (3 dof per element) and "
                    "the carrier is scalar"
                )
            elif arities == {3} and 3 * self.operator.shape[1] == carrier.n_dof:
                hint = (
                    " -- the operator is fixed-orientation (1 dof per element) and "
                    "the carrier is vector-valued"
                )
            raise ArityError(
                f"view {self.id!r}: operator consumes {self.operator.shape[1]} dof "
                f"but carrier {carrier.id!r} has {carrier.n_dof} "
                f"({carrier.n_elements} elements, arities "
                f"{sorted(set(carrier.arities()))}){hint}",
                remedy=(
                    "Declare the carrier's ElementSpec to match the operator, or "
                    "project the operator onto a declared direction field and say "
                    "so with ElementSpec.projected_along."
                ),
                offending_object=self.id,
            )
        got = self.operator.units * carrier.default_spec.units
        want = self.support.element.units if self.support.element else None
        if want is not None and not got.same_dimension(want):
            raise ViewMismatch(
                f"view {self.id!r}: operator units {self.operator.units!r} times "
                f"carrier units {carrier.default_spec.units!r} give {got!r}, which "
                f"is not dimensionally {want!r}",
                remedy="R01: numerical shape compatibility is not measurement "
                "compatibility.",
                offending_object=self.id,
            )
