"""O-2: an algebra over supports, so relating two of them is a computation.

``Support`` (see ``supports.py``) is a passive descriptor: ``kind, frame, units,
psf, extent, n_elements, resolution`` and **no operators**. Two supports cannot
be related without hand-writing a map, which is why the restriction/prolongation
pair used for cross-scale work had to be declared as a one-off rather than
derived. This module supplies the missing algebra.

The design commitment that matters is **what it refuses**. An algebra that
always returns a map is worse than none, because a fabricated correspondence
between two parcellations is indistinguishable at the type level from a real
one. Everything here derives a map only from what the supports actually
declare, and otherwise raises ``SupportIncompatible`` naming the missing
declaration. That is the same fail-closed posture as the compiler's R-series.

Three things it carries that ``Support`` cannot express today:

**Element type.** A support says how many elements it has, never what one *is*.
That distinction is load-bearing and was measured, not assumed: a per-parcel
scalar carries 5.6% of the whitened lead field where three numbers per parcel
carry 51.7%. A scalar support and a 3-vector support over the same parcels are
different objects, and relating them requires an orientation field rather than a
reshape. ``ElementType`` makes the difference visible and the algebra refuses to
paper over it.

**PSF composition.** Mapping through a support smears the datum. Composing two
Gaussian kernels adds their FWHM in quadrature; composing anything with a
kernel that only has a ``kernel_ref`` is not derivable from declarations, and is
refused rather than guessed.

**The uncertainty a map introduces.** Restriction (fine -> coarse) is a
projection and loses information it cannot recover. Prolongation (coarse ->
fine) *invents* structure and must say so. A map that reports no uncertainty in
the prolongation direction is claiming that upsampling is free.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .supports import PSF, Support, TemporalSupport
from .units import Unit

__all__ = [
    "ElementType",
    "MapKind",
    "SupportIncompatible",
    "SupportMap",
    "compose_psf",
    "common_temporal_refinement",
    "relate",
]


class SupportIncompatible(ValueError):
    """Raised when no map between two supports is derivable from declarations.

    Always names the missing declaration.  A refusal nobody can act on is a
    wall, not a gate.
    """


class ElementType(SchemaModel):
    """What a single element of a support *is*, as opposed to how many there are.

    ``rank=0`` is a scalar per element; ``rank=1`` with ``dim=3`` is a free
    3-vector per element (an unconstrained dipole moment, say).  This is the
    distinction ``Support`` cannot currently make, and it is the one the lead
    field measurement turned on.
    """

    rank: Literal[0, 1] = 0
    dim: int = Field(default=1, ge=1)
    #: Frame the vector components are expressed in.  Meaningless for rank 0,
    #: and required for rank 1: three numbers with no frame are three numbers.
    component_frame: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ElementType":
        if self.rank == 0 and self.dim != 1:
            raise ValueError(f"rank-0 elements are scalars; dim must be 1, got {self.dim}")
        if self.rank == 1:
            if self.dim < 2:
                raise ValueError(f"rank-1 elements need dim >= 2, got {self.dim}")
            if not self.component_frame:
                raise ValueError(
                    "rank-1 elements must declare component_frame; vector components "
                    "without a frame are not vectors"
                )
        return self

    @property
    def width(self) -> int:
        """Numbers per element."""
        return 1 if self.rank == 0 else self.dim


MapKind = Literal["identity", "restriction", "prolongation", "orientation"]


class SupportMap(SchemaModel):
    """A derived map between two supports, with what it costs.

    ``lossy`` and ``invents`` are deliberately separate. Restriction is lossy
    and honest; prolongation invents structure that was never measured. A
    pipeline may tolerate the first and must declare the second.
    """

    kind: MapKind
    #: Numbers in / numbers out, after element width is accounted for.
    n_in: int
    n_out: int
    psf: PSF | None = None
    #: Information discarded (fine -> coarse). Coarse output is a summary.
    lossy: bool = False
    #: Structure asserted that no measurement supports (coarse -> fine).
    invents: bool = False
    #: One line on where the uncertainty comes from; empty only for identity.
    uncertainty_note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "SupportMap":
        if self.kind != "identity" and not self.uncertainty_note:
            raise ValueError(
                f"a {self.kind} map must state its uncertainty; only identity is free"
            )
        if self.kind == "prolongation" and not self.invents:
            raise ValueError(
                "prolongation asserts structure no measurement supports; invents=False "
                "would claim upsampling is free"
            )
        if self.kind == "restriction" and not self.lossy:
            raise ValueError("restriction discards information; lossy=False is a false claim")
        return self


def compose_psf(a: PSF | None, b: PSF | None) -> PSF | None:
    """Compose two kernels applied in sequence.

    Gaussians compose in closed form -- FWHM adds in quadrature. Everything
    else does not, from declarations alone: a ``kernel_ref`` is an opaque asset
    and composing two of them requires the assets, not their descriptions.
    Refuse rather than approximate, because an approximated PSF is precisely the
    "nominal coordinate substituted for a physical support" that Appendix B
    rejects.
    """
    if a is None:
        return b
    if b is None:
        return a
    if a.kind == "point":
        return b
    if b.kind == "point":
        return a
    if a.kind == "gaussian" and b.kind == "gaussian":
        if a.fwhm is None or b.fwhm is None:
            raise SupportIncompatible(
                "gaussian psf without fwhm cannot be composed; declare fwhm on both"
            )
        if len(a.fwhm) != len(b.fwhm):
            raise SupportIncompatible(
                f"psf axis count differs: {len(a.fwhm)} vs {len(b.fwhm)}"
            )
        if a.extent_units != b.extent_units:
            raise SupportIncompatible(
                f"psf extent units differ: {a.extent_units} vs {b.extent_units}; "
                "convert before composing"
            )
        return PSF(
            kind="gaussian",
            fwhm=tuple(math.hypot(x, y) for x, y in zip(a.fwhm, b.fwhm)),
            units=b.units,
            extent_units=a.extent_units,
            nominal=a.nominal or b.nominal,
        )
    raise SupportIncompatible(
        f"cannot compose psf kinds {a.kind!r} and {b.kind!r} from declarations alone; "
        "composing opaque kernels needs the kernel assets, not their descriptions"
    )


def common_temporal_refinement(
    a: TemporalSupport, b: TemporalSupport
) -> tuple[TemporalSupport, SupportMap, SupportMap]:
    """Put two clocks on a common footing, and say what each map costs.

    This is the "5000 Hz EEG against 0.5 Hz BOLD on the same subject" case. The
    common refinement is the *finer* clock: the fast signal is already there,
    and the slow one has to be held or interpolated across the gap -- which is
    invention, and is reported as such.

    Refuses across clocks, because two named clocks with no declared
    relationship cannot be aligned; that is what a synchronisation record is
    for, and its absence is exactly the defect worth failing on.
    """
    if a.clock != b.clock:
        raise SupportIncompatible(
            f"clocks {a.clock!r} and {b.clock!r} have no declared relationship; "
            "align them with a synchronisation record before relating their supports"
        )
    fine, coarse = (a, b) if a.dt <= b.dt else (b, a)
    ratio = coarse.dt / fine.dt

    def _to_fine(src: TemporalSupport) -> SupportMap:
        if src.dt == fine.dt:
            return SupportMap(kind="identity", n_in=1, n_out=1)
        return SupportMap(
            kind="prolongation",
            n_in=1,
            n_out=max(1, int(round(ratio))),
            invents=True,
            uncertainty_note=(
                f"one sample every {src.dt:g}s asserted across {ratio:.4g} slots of a "
                f"{fine.dt:g}s clock; the intervening values were never measured, and "
                f"the source's {src.integration_window:g}s integration window means even "
                "the sample itself is not instantaneous"
            ),
        )

    return fine, _to_fine(a), _to_fine(b)


def relate(
    src: Support,
    dst: Support,
    *,
    src_elements: ElementType | None = None,
    dst_elements: ElementType | None = None,
    orientation_ref: str | None = None,
) -> SupportMap:
    """Derive the map ``src -> dst``, or refuse and name what is missing.

    What is derivable from declarations:

    - same frame, same kind, and a declared ``n_elements`` on both -> a
      restriction or prolongation, by which side is finer;
    - identical supports -> identity.

    What is not, and is therefore refused:

    - different frames, with no registration declared;
    - different units;
    - a rank change (scalar <-> vector), unless an ``orientation_ref`` is
      supplied. This refusal is the measured one: collapsing three numbers per
      parcel to one is not a reshape, it is a projection onto an orientation,
      and which orientation is a physical fact about the cortex rather than a
      layout choice.
    """
    if src.frame != dst.frame:
        raise SupportIncompatible(
            f"frames differ ({src.frame!r} -> {dst.frame!r}) and no registration is "
            "declared; supply a registration rather than assuming the frames coincide"
        )
    if src.units != dst.units:
        raise SupportIncompatible(
            f"units differ ({src.units} -> {dst.units}); convert before relating supports"
        )

    se = src_elements or ElementType()
    de = dst_elements or ElementType()

    if se.rank != de.rank:
        if orientation_ref is None:
            raise SupportIncompatible(
                f"element rank changes ({se.rank} -> {de.rank}) and no orientation_ref is "
                "given; projecting a vector onto a scalar requires an orientation field. "
                "Measured, not hypothetical: a per-parcel scalar carries 5.6% of the "
                "whitened lead field where three numbers per parcel carry 51.7%"
            )
        collapsing = se.rank > de.rank
        return SupportMap(
            kind="orientation",
            n_in=(src.n_elements or 1) * se.width,
            n_out=(dst.n_elements or 1) * de.width,
            psf=compose_psf(src.psf, dst.psf),
            lossy=collapsing,
            invents=not collapsing,
            uncertainty_note=(
                f"projection onto the orientation field {orientation_ref!r}; the component "
                "normal to it is discarded and cannot be recovered"
                if collapsing
                else f"components assigned from the orientation field {orientation_ref!r}; "
                "the tangential degrees of freedom were never measured"
            ),
        )

    if se.rank == 1 and se.component_frame != de.component_frame:
        raise SupportIncompatible(
            f"vector components are expressed in different frames "
            f"({se.component_frame!r} -> {de.component_frame!r}); rotate before relating"
        )

    n_src, n_dst = src.n_elements, dst.n_elements
    if n_src is None or n_dst is None:
        raise SupportIncompatible(
            "both supports must declare n_elements to be related; a support of unknown "
            "size cannot be mapped into one of known size"
        )

    if n_src == n_dst and src.kind == dst.kind:
        return SupportMap(kind="identity", n_in=n_src * se.width, n_out=n_dst * de.width)

    psf = compose_psf(src.psf, dst.psf)
    if n_src > n_dst:
        return SupportMap(
            kind="restriction",
            n_in=n_src * se.width,
            n_out=n_dst * de.width,
            psf=psf,
            lossy=True,
            uncertainty_note=(
                f"{n_src} elements summarised into {n_dst}; within-element variance is "
                "discarded and the result is a summary, not a measurement of the coarse "
                "support"
            ),
        )
    return SupportMap(
        kind="prolongation",
        n_in=n_src * se.width,
        n_out=n_dst * de.width,
        psf=psf,
        invents=True,
        uncertainty_note=(
            f"{n_src} elements spread over {n_dst}; the added detail is asserted by the "
            "map, not measured, and any structure finer than the source support is an "
            "artifact of this step"
        ),
    )
