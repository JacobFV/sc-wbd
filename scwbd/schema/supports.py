"""Physical and temporal support (thesis sec. 2.6, Appendix C).

A datum is never a bare coordinate.  It has a *support*: the physical set it
integrates over, in a named frame, with declared units and a point-spread /
lead-field / integration kernel.  It also has a *temporal support*: a named
clock, a native sampling period, an exposure window, a filter group delay and
a jitter.  R01 exists because numerical shape compatibility is not measurement
compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pydantic import Field

from .base import SchemaModel
from .ids import ClockId, FrameId, ScaleId
from .ledger import UncertaintyLedger
from .units import Unit

__all__ = [
    "PSF",
    "Support",
    "TemporalSupport",
    "SupportKind",
    "ElementKind",
    "ElementSpec",
    "OntologyError",
    "ArityError",
]


class OntologyError(ValueError):
    """A declared object does not describe what it claims to describe."""

    def __init__(
        self, message: str, *, remedy: str = "", offending_object: object = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.offending_object = offending_object


class ArityError(OntologyError):
    """A scalar support was used where a vector-valued one was required.

    Its own class because this is the failure the measurement found: a
    per-parcel scalar and a per-parcel 3-vector are different objects, and the
    system's only defence against confusing them is that this raises.
    """


#: What one element of a support carries.  There is no ``"vector"`` of arity 1
#: -- that is a scalar that has forgotten which way it points, which is exactly
#: the confusion this type exists to prevent.
ElementKind = Literal["scalar", "vector", "covector", "tensor2"]

_ARITY_OF_KIND: dict[str, int | None] = {
    "scalar": 1,
    "vector": 3,
    "covector": 3,
    "tensor2": 9,
}


class ElementSpec(SchemaModel):
    """What **one** element of a support carries (ARCHITECTURE.md sec. 2b O-2).

    ``arity`` is the number of degrees of freedom per element.  It is the field
    ``Support`` was missing: without it a per-parcel scalar amplitude and a
    per-parcel net dipole moment are the same declared object, and every map
    between them typechecks.
    """

    kind: ElementKind = "scalar"
    #: Degrees of freedom per element.  Redundant with ``kind`` for the kinds
    #: whose arity is fixed, and checked against it -- a disagreement is a
    #: declaration error, not a value to reconcile.
    arity: int = Field(default=1, ge=1)
    units: Unit
    #: Frame the components are expressed in.  Required for ``arity > 1``: the
    #: components of a directed quantity are undefined without one.
    component_frame: FrameId | None = None
    #: Component names, e.g. ``("x", "y", "z")``.  Optional; when supplied it
    #: must match ``arity``.
    components: tuple[str, ...] = ()
    #: For a scalar that is the projection of a vector quantity onto a declared
    #: direction field -- a fixed-orientation cortical dipole is exactly this.
    #: Naming the direction is what makes the projection auditable; a scalar
    #: with no ``projected_along`` is a scalar that never had an orientation.
    projected_along: str | None = None
    label: str = ""

    @model_validator(mode="after")
    def _check_element(self) -> "ElementSpec":
        expected = _ARITY_OF_KIND.get(self.kind)
        if expected is not None and self.arity != expected:
            raise OntologyError(
                f"element kind {self.kind!r} has arity {expected}, not {self.arity}",
                remedy=(
                    "A 'vector' of arity 1 is a scalar that has forgotten which "
                    "way it points. Declare kind='scalar' with projected_along=, "
                    "or give the vector its 3 components."
                ),
                offending_object=self.kind,
            )
        if self.arity > 1 and self.component_frame is None:
            raise OntologyError(
                f"element of arity {self.arity} must declare a component_frame",
                remedy="The components of a directed quantity are undefined "
                "without the frame they are expressed in (R01).",
            )
        if self.components and len(self.components) != self.arity:
            raise OntologyError(
                f"{len(self.components)} component names for arity {self.arity}",
                remedy="Name every component or name none.",
            )
        if self.kind != "scalar" and self.projected_along is not None:
            raise OntologyError(
                f"element kind {self.kind!r} declares projected_along="
                f"{self.projected_along!r}; only a scalar is a projection",
                remedy="A vector is not the projection of anything; drop the field.",
            )
        return self

    # -- constructors -------------------------------------------------------
    @classmethod
    def scalar(
        cls, units: str, *, label: str = "", projected_along: str | None = None
    ) -> "ElementSpec":
        return cls(
            kind="scalar",
            arity=1,
            units=Unit(units),
            label=label,
            projected_along=projected_along,
        )

    @classmethod
    def vector3(cls, units: str, frame: str, *, label: str = "") -> "ElementSpec":
        return cls(
            kind="vector",
            arity=3,
            units=Unit(units),
            component_frame=FrameId(frame),
            components=("x", "y", "z"),
            label=label,
        )

    @property
    def is_oriented(self) -> bool:
        """True when this element carries a direction a map could destroy.

        A scalar declaring ``projected_along`` is *derived* from an oriented
        quantity but does not itself carry the orientation -- it is the 32.1 %
        case, against 83.4 % for the 3-vector.  It answers ``False`` on purpose.
        """
        return self.arity > 1

    @property
    def orientation_source(self) -> str | None:
        """Where a scalar's lost orientation went, if it was declared at all."""
        return self.projected_along

SupportKind = Literal[
    "voxel",
    "surface_vertex",
    "parcel",
    "sensor",
    "mesh",
    "field",
    "event",
    "band",
]


class PSF(SchemaModel):
    """Point-spread / lead-field / integration kernel of a support.

    A nominal coordinate substituted for a physical support is a rejection
    condition in Appendix B ("Spatial support and point-spread"), so a
    ``Support`` that claims fine resolution must say *how* it smears.
    """

    kind: Literal[
        "point",
        "gaussian",
        "lead_field",
        "integration_kernel",
        "hemodynamic",
        "acoustic",
        "tract_endpoint",
        "empirical",
    ]
    #: Full width at half maximum per spatial axis, in ``extent_units``.
    fwhm: tuple[float, ...] | None = None
    #: Units of the kernel's *output* (e.g. "V" for a lead field mapping A*m).
    units: Unit = Unit("dimensionless")
    #: Units in which ``fwhm`` is expressed.
    extent_units: Unit = Unit("m")
    #: Reference to the stored kernel/lead-field asset (agent C/F own the data).
    kernel_ref: str | None = None
    #: True when the kernel is only a nominal/placeholder description.
    nominal: bool = False
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "PSF":
        if self.kind != "point" and self.fwhm is None and self.kernel_ref is None:
            raise ValueError(
                f"psf kind={self.kind!r} must declare either fwhm or kernel_ref; "
                "a named kernel with no description is an unknown support"
            )
        if self.fwhm is not None and any(v < 0.0 for v in self.fwhm):
            raise ValueError("psf fwhm entries must be non-negative")
        return self


class Support(SchemaModel):
    """Physical support of a datum. Never a bare coordinate.

    ``n_elements`` counts the *elements*; it is **not** the dimension of the
    space.  The dimension is ``sum_i arity_i``, and ``element`` is what declares
    the arity.  See :mod:`scwbd.schema.carrier` for why that distinction is
    load-bearing rather than pedantic: on Schaefer400x7 a per-parcel scalar
    carries 32.1 % of the whitened EEG lead field and a per-parcel 3-vector
    carries 83.4 %, and until ``element`` existed those were the same declared
    object.
    """

    kind: SupportKind
    frame: FrameId
    units: Unit
    psf: PSF | None = None
    extent: tuple[float, ...] | None = None
    n_elements: int | None = None

    # -- additions used by the compiler (not part of the sec. 2 core tuple) --
    #: Element of the resolution poset this support realizes.
    resolution: ScaleId | None = None
    #: Units in which ``extent`` is expressed.
    extent_units: Unit = Unit("m")
    label: str = ""

    # -- sec. 2b O-1/O-2: what one element carries ---------------------------
    #: What a single element of this support carries (arity, orientation).
    #: ``None`` means undeclared, which is the pre-2b state: such a support has
    #: no defined dimension and :mod:`scwbd.schema.support_algebra` refuses it
    #: rather than assuming ``arity == 1``.  Assuming 1 is the defect.
    element: ElementSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> "Support":
        if self.n_elements is not None and self.n_elements <= 0:
            raise ValueError(f"n_elements must be positive, got {self.n_elements}")
        if self.extent is not None and any(v < 0.0 for v in self.extent):
            raise ValueError("extent entries must be non-negative")
        if self.element is not None and not self.units.same_dimension(self.element.units):
            raise ValueError(
                f"support units {self.units!r} and element units "
                f"{self.element.units!r} are not the same dimension"
            )
        return self

    @property
    def is_nominal(self) -> bool:
        """True when no physical support was supplied beyond a label."""
        return self.psf is None and self.extent is None

    @property
    def declares_dof(self) -> bool:
        """True when this support says how many numbers each element carries."""
        return self.element is not None and self.n_elements is not None

    @property
    def n_dof(self) -> int:
        """``n_elements * arity``.  Raises when the arity was never declared.

        Deliberately not defaulted to ``n_elements``: a support that never said
        whether its elements are scalars or vectors does not have a dimension,
        and inventing one is how a scalar lead field came to be multiplied by a
        support that was never scalar.
        """
        if self.element is None:
            raise ValueError(
                f"support {self.label or self.kind!r} declares no ElementSpec, so "
                "it has no dimension; n_elements is a count, not a dimension"
            )
        if self.n_elements is None:
            raise ValueError(
                f"support {self.label or self.kind!r} declares no n_elements"
            )
        return int(self.n_elements) * int(self.element.arity)

    @property
    def carries_orientation(self) -> bool:
        return self.element is not None and self.element.arity > 1


class TemporalSupport(SchemaModel):
    """Sampling clock, exposure window, filter delay and jitter of a datum."""

    clock: ClockId
    dt: float = Field(gt=0.0)  # seconds, native
    integration_window: float = Field(default=0.0, ge=0.0)  # 0 for instantaneous
    group_delay: float = Field(default=0.0)  # seconds, filter delay
    jitter_sd: float = Field(default=0.0, ge=0.0)

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    @property
    def effective_latency(self) -> float:
        """Delay from physical event to reported sample."""
        return self.group_delay + 0.5 * self.integration_window

    def nyquist_hz(self) -> float:
        return 0.5 / self.dt
