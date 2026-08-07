"""An algebra for :class:`~scwbd.schema.supports.Support` -- sec. 2b, O-2.

    "``Support`` today is a passive descriptor ... and **no operators**.  Two
     supports cannot be related without hand-writing a map ... It needs an
     algebra: given supports ``a`` and ``b``, produce their common refinement
     and the two operators into it, with the composed PSF and the uncertainty
     each map introduces."

What is here
------------
* :func:`element_join` -- the least element spec in which two supports' values
  can both be expressed.  **This is where orientation is enforced.**  Joining a
  bare scalar with a 3-vector *raises*: there is no direction along which to
  embed the scalar, so the join does not exist.  Joining a scalar that declares
  ``projected_along`` with a vector succeeds and the resulting map is flagged as
  manufacturing degrees of freedom.
* :func:`common_refinement` -- given two supports over a shared atom set and
  each one's membership, the cell-wise common refinement ``c``, the two
  restrictions ``c -> a`` and ``c -> b``, the two prolongations ``a -> c`` and
  ``b -> c``, the composed PSF, and the rank each map does not resolve.
* :func:`project_along` -- the arity-lowering map (a 3-vector support onto a
  declared direction field).  It is the map that turns the 51.7 % carrier into
  the 5.6 % one, and it cannot be built without naming the directions.
* :func:`temporal_common_refinement` -- the same question on clocks, which is
  what "5000 Hz EEG against 0.5 Hz BOLD on the same subject" actually needs.

What is deliberately *not* here
-------------------------------
A default uncertainty for a prolongation.  Every map reports the rank it does
not resolve; none of them invents a prior standard deviation for those
directions.  ``reports/transforms/resolution_pair.md`` sec. 5 records what
happens when a point estimate is used to bound a quantity, and the answer is
that the refusal becomes a coin flip.  :attr:`SupportMap.prior_sd_unresolved`
is therefore ``None`` until a measurement sets it, and
:meth:`SupportMap.as_prolongation_inputs` refuses to hand the map to R02
machinery while it is.

Direction convention, and a disagreement with sec. 2b
-----------------------------------------------------
sec. 2b asks for "the two operators **into** it".  The operators into a common
refinement are prolongations, and a prolongation is precisely the object R02
governs -- it manufactures structure no observation supports.  The operators
that are free are the ones **out of** the refinement.  :class:`Refinement`
therefore returns both directions and labels them honestly: ``to_a``/``to_b``
are restrictions and are safe; ``from_a``/``from_b`` are prolongations, carry
``manufactures_dof=True``, and are inadmissible until measured.  See
``reports/schema/ontology.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .ids import FrameId
from .supports import (
    PSF,
    ArityError,
    ElementSpec,
    OntologyError,
    Support,
    TemporalSupport,
)
from .units import Unit

__all__ = [
    "MapDirection",
    "SupportMap",
    "Refinement",
    "TemporalRefinement",
    "element_join",
    "common_refinement",
    "project_along",
    "embed_along",
    "restriction_between",
    "dense",
    "temporal_common_refinement",
]

MapDirection = Literal["restriction", "prolongation", "projection", "isomorphism"]


# ==========================================================================
# element algebra -- where orientation is enforced
# ==========================================================================
def element_join(a: ElementSpec, b: ElementSpec) -> ElementSpec:
    """The least element spec in which both ``a`` and ``b`` can be expressed.

    Rules, in the order they are checked:

    1. **Units must share a dimension.**  Otherwise there is no common space
       and the join does not exist (R01).
    2. **Equal arity** -> the join is the richer of the two (a vector beats a
       scalar of the same arity only if arities differ, so this is a no-op).
    3. **Unequal arity, and the lower one is a bare scalar** -> raise
       :class:`ArityError`.  A scalar that never declared an orientation cannot
       be embedded in a vector space; choosing a direction for it *is* the
       manufacture of structure.  This is the check the 5.6 %/51.7 % result
       asks for.
    4. **Unequal arity, and the lower one declares ``projected_along``** ->
       the join is the higher-arity spec, and the embedding is a prolongation
       (flagged by :func:`common_refinement`).
    5. **Vector components must agree on their frame.**  Two 3-vectors in
       different frames are not the same 3-vector.
    """
    if not a.units.same_dimension(b.units):
        raise OntologyError(
            f"cannot join elements with units {a.units!r} and {b.units!r}: "
            "different dimensions",
            remedy="R01 -- convert to a common dimension or accept that these "
            "two supports are not comparable.",
            offending_object=(str(a.units), str(b.units)),
        )
    lo, hi = (a, b) if a.arity <= b.arity else (b, a)
    if lo.arity == hi.arity:
        if lo.arity > 1 and lo.component_frame != hi.component_frame:
            raise OntologyError(
                f"cannot join arity-{lo.arity} elements expressed in frames "
                f"{lo.component_frame!r} and {hi.component_frame!r}",
                remedy="Two 3-vectors in different frames are not the same "
                "3-vector; transform one into the other's frame first.",
                offending_object=(lo.component_frame, hi.component_frame),
            )
        return hi
    if lo.arity > 1:
        raise OntologyError(
            f"cannot join arity {lo.arity} with arity {hi.arity}: no declared "
            "embedding between two multi-component elements",
            remedy="Declare the embedding explicitly; the algebra will not "
            "guess which components correspond.",
        )
    if lo.projected_along is None:
        raise ArityError(
            f"cannot join a bare scalar with an arity-{hi.arity} element: the "
            "scalar declares no direction to be embedded along",
            remedy=(
                "A per-parcel scalar carries 5.6 % of the whitened EEG lead "
                "field against 51.7 % for three numbers per parcel "
                "(reports/transforms/resolution_pair.md sec. 3.5). Either declare "
                "ElementSpec.projected_along so the projection is auditable, or "
                "accept that the two supports are not comparable."
            ),
            offending_object=(lo.model_dump(mode="json"), hi.arity),
        )
    return hi


# ==========================================================================
# the maps
# ==========================================================================
@dataclass(frozen=True)
class SupportMap:
    """One declared linear map between two supports, with what it costs.

    ``matrix`` has shape ``(dst.n_dof, src.n_dof)``.  Blocks over element
    components are explicit -- an arity-3 restriction is *not* the arity-1
    restriction applied three times by broadcasting, it is the Kronecker
    product with ``I_3``, and writing it out is what makes the shape check
    meaningful.
    """

    src: Support
    dst: Support
    matrix: np.ndarray
    direction: MapDirection
    method: str
    #: True when ``dst`` has more degrees of freedom than the map can resolve
    #: from ``src`` -- i.e. the map invents structure.  R02's subject.
    manufactures_dof: bool
    #: True when the map reduces per-element arity.  Always accompanied by
    #: ``projected_along``, because the constructor refuses otherwise.
    lowers_arity: bool
    projected_along: str | None = None
    #: ``src.n_dof - rank(matrix)``: the directions of the source this map
    #: cannot see.  Computed, not declared.
    unresolved_rank: int = 0
    #: The prior sd this map must declare for its unresolved directions.
    #: ``None`` until a measurement sets it -- never defaulted.
    prior_sd_unresolved: float | None = None
    psf: PSF | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        M = self.matrix if _is_sparse(self.matrix) else np.asarray(self.matrix, float)
        object.__setattr__(self, "matrix", M)
        want = (self.dst.n_dof, self.src.n_dof)
        if M.shape != want:
            raise OntologyError(
                f"support map {self.label} must be {want} "
                f"(dst.n_dof x src.n_dof), got {M.shape}",
                remedy="Shapes are in degrees of freedom, not elements.",
                offending_object=M.shape,
            )
        if self.lowers_arity and self.projected_along is None:
            raise ArityError(
                f"support map {self.label} lowers arity from "
                f"{self.src.element.arity if self.src.element else '?'} to "
                f"{self.dst.element.arity if self.dst.element else '?'} without "
                "naming the direction field it projected onto",
                remedy=(
                    "Set ElementSpec.projected_along on the destination. An "
                    "unnamed projection is unrecoverable and unauditable."
                ),
                offending_object=self.label,
            )

    @property
    def label(self) -> str:
        return (
            f"{self.direction}[{self.src.label or self.src.kind}"
            f"->{self.dst.label or self.dst.kind}]"
        )

    @property
    def is_exact(self) -> bool:
        """True when the map resolves every direction of its source."""
        return self.unresolved_rank == 0

    @property
    def needs_uncertainty_policy(self) -> bool:
        return self.unresolved_rank > 0 and self.prior_sd_unresolved is None

    def apply(self, x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        flat = v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(1, -1)
        if flat.shape[-1] != self.matrix.shape[1]:
            raise OntologyError(
                f"{self.label} expects {self.matrix.shape[1]} dof, got "
                f"{flat.shape[-1]}",
                remedy="Apply a map to a vector on the support it is declared for.",
            )
        out = np.asarray(
            (self.matrix @ flat.T).T if _is_sparse(self.matrix) else flat @ self.matrix.T
        )
        return out[0] if v.ndim == 1 else out

    def with_uncertainty(self, prior_sd: float) -> "SupportMap":
        """Attach a *measured* prior sd for the unresolved directions."""
        if not (prior_sd > 0.0 and math.isfinite(prior_sd)):
            raise OntologyError(
                f"prior sd for unresolved directions must be positive and finite, "
                f"got {prior_sd!r}",
                remedy="Zero variance on an unmeasured subspace is a claim of "
                "infinite precision.",
            )
        return SupportMap(
            src=self.src,
            dst=self.dst,
            matrix=self.matrix,
            direction=self.direction,
            method=self.method,
            manufactures_dof=self.manufactures_dof,
            lowers_arity=self.lowers_arity,
            projected_along=self.projected_along,
            unresolved_rank=self.unresolved_rank,
            prior_sd_unresolved=float(prior_sd),
            psf=self.psf,
            notes=self.notes,
        )

    def as_prolongation_inputs(self) -> dict[str, Any]:
        """Hand this map to R02 machinery, or refuse to.

        A prolongation with unresolved directions and no declared prior sd is
        exactly what R02 exists to reject, so this raises rather than emitting a
        record that will be rejected downstream with less context.
        """
        if self.direction != "prolongation":
            raise OntologyError(
                f"{self.label} is a {self.direction}, not a prolongation"
            )
        if self.needs_uncertainty_policy:
            raise OntologyError(
                f"{self.label} leaves {self.unresolved_rank} of "
                f"{self.matrix.shape[0]} destination directions unresolved and "
                "declares no prior sd for them",
                remedy=(
                    "Measure the residual on held-out data and attach it with "
                    ".with_uncertainty(). R02: a coarse observation must not be "
                    "converted into unsupported fine structure."
                ),
                offending_object=self.label,
            )
        return {
            "matrix": self.matrix,
            "prior_sd_unresolved": self.prior_sd_unresolved,
            "method": self.method,
            "unresolved_rank": self.unresolved_rank,
        }

    # -- the measurement that decides whether a projection was affordable ---
    def retained_energy(
        self, forward: np.ndarray, *, prior_weights: np.ndarray | None = None
    ) -> float:
        """Fraction of a forward operator's energy that survives this map.

        ``eta = tr(F Pi F^T) / tr(F F^T)`` where ``Pi`` is the orthogonal
        projector onto ``row(matrix)`` in the metric of ``prior_weights``.  With
        ``forward`` a whitened EEG lead field and ``prior_weights`` the per-dof
        prior variance, this is exactly the quantity
        ``reports/transforms/resolution_pair.md`` sec. 3.5 reports as
        ``lead_field_energy_retained``.  It is computed here, from the operator
        the caller supplies, and nothing is quoted.
        """
        F = np.asarray(forward, dtype=float)
        if F.shape[1] != self.matrix.shape[1]:
            raise OntologyError(
                f"forward operator has {F.shape[1]} columns; {self.label} "
                f"consumes {self.matrix.shape[1]} dof"
            )
        if prior_weights is None:
            w = np.ones(F.shape[1])
        else:
            w = np.asarray(prior_weights, dtype=float).reshape(-1)
            if w.shape[0] != F.shape[1]:
                raise OntologyError("prior_weights must have one entry per source dof")
        s = np.sqrt(w)
        Fs = F * s[None, :]
        Ms = dense(self.matrix) * s[None, :]
        # orthonormal basis of row(Ms)
        _, sv, Vh = np.linalg.svd(Ms, full_matrices=False)
        tol = max(Ms.shape) * (sv.max() if sv.size else 0.0) * np.finfo(float).eps
        B = Vh[sv > tol]
        num = float(np.sum((Fs @ B.T) ** 2))
        den = float(np.sum(Fs**2))
        return num / den if den > 0 else 0.0


@dataclass(frozen=True)
class Refinement:
    """The common refinement of two supports and the four maps around it.

    ``to_a`` / ``to_b`` are restrictions **out of** the refinement and are
    always admissible.  ``from_a`` / ``from_b`` are prolongations **into** it
    and are R02's subject: they carry ``manufactures_dof`` and refuse to be
    used until a measured uncertainty is attached.
    """

    support: Support
    to_a: SupportMap
    to_b: SupportMap
    from_a: SupportMap
    from_b: SupportMap
    #: One row per refinement cell: ``(a_index, b_index, weight)``.
    cells: tuple[tuple[int, int, float], ...]
    #: Atoms that neither support claims -- never averaged away.
    n_unassigned_atoms: int
    composed_psf: PSF | None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_dof": self.support.n_dof,
            "arity": self.support.element.arity if self.support.element else None,
            "n_unassigned_atoms": self.n_unassigned_atoms,
            "to_a": {
                "shape": list(self.to_a.matrix.shape),
                "unresolved_rank": self.to_a.unresolved_rank,
                "manufactures_dof": self.to_a.manufactures_dof,
            },
            "to_b": {
                "shape": list(self.to_b.matrix.shape),
                "unresolved_rank": self.to_b.unresolved_rank,
                "manufactures_dof": self.to_b.manufactures_dof,
            },
            "from_a": {
                "shape": list(self.from_a.matrix.shape),
                "unresolved_rank": self.from_a.unresolved_rank,
                "manufactures_dof": self.from_a.manufactures_dof,
                "needs_uncertainty_policy": self.from_a.needs_uncertainty_policy,
            },
            "from_b": {
                "shape": list(self.from_b.matrix.shape),
                "unresolved_rank": self.from_b.unresolved_rank,
                "manufactures_dof": self.from_b.manufactures_dof,
                "needs_uncertainty_policy": self.from_b.needs_uncertainty_policy,
            },
            "composed_psf": (
                self.composed_psf.model_dump(mode="json") if self.composed_psf else None
            ),
            "provenance": self.provenance,
        }


# ==========================================================================
# construction
# ==========================================================================
def _is_sparse(m: Any) -> bool:
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover - scipy is a hard dependency here
        return False
    return bool(sp.issparse(m))


def dense(m: Any) -> np.ndarray:
    """A dense copy of a map's matrix, for callers that need one."""
    return np.asarray(m.toarray() if _is_sparse(m) else m, dtype=float)


#: Above this many entries a map is built sparse.  An indicator prolongation
#: from 68 parcels onto 7498 free-orientation dipoles is 22494x204 -- small --
#: but the *identity-shaped* refinement map in the same construction is
#: 22494x22494, which is 4 GB dense.  A support algebra that only works on toy
#: supports would not be an algebra.
_SPARSE_ABOVE = 4_000_000


def _block(matrix: Any, arity: int) -> Any:
    """Lift an element-level map to a dof-level one: ``kron(M, I_arity)``.

    Written out rather than broadcast on purpose.  Broadcasting a scalar map
    across components is what makes a per-parcel scalar and a per-parcel vector
    look interchangeable.
    """
    if arity == 1:
        return matrix
    if _is_sparse(matrix):
        import scipy.sparse as sp

        return sp.kron(matrix, sp.eye(arity, format="csr"), format="csr")
    return np.kron(matrix, np.eye(arity))


def _maybe_sparse(rows: int, cols: int, data: Any) -> Any:
    if rows * cols > _SPARSE_ABOVE:
        import scipy.sparse as sp

        return sp.csr_matrix(data)
    return data


def _rank(M: Any, *, known: int | None = None) -> int:
    """Rank of a map.

    ``known`` is the combinatorial rank the constructor already knows -- the
    number of non-empty destination cells for a restriction, the number of
    distinct source elements referenced for an indicator prolongation.  It is
    used in preference to an SVD because both maps are structurally rank-known
    and an SVD of a 22494-square matrix is not a thing to do casually.  When it
    is absent the rank is measured.
    """
    if known is not None:
        return int(known)
    return int(np.linalg.matrix_rank(dense(M)))


def restriction_between(
    src: Support,
    dst: Support,
    membership: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    *,
    method: str = "weighted_mean",
) -> SupportMap:
    """Weighted-mean restriction ``src -> dst`` given per-src-element membership.

    ``membership[i]`` is the ``dst`` element that ``src`` element ``i`` belongs
    to, or ``-1`` for "belongs to none" -- which is carried as a zero row, never
    filled by averaging neighbours.
    """
    if src.element is None or dst.element is None:
        raise OntologyError(
            "restriction_between requires both supports to declare an ElementSpec",
            remedy="Support.element is what gives a support a dimension.",
        )
    join = element_join(src.element, dst.element)
    if join.arity != src.element.arity:
        raise ArityError(
            f"restriction from arity {src.element.arity} to arity "
            f"{dst.element.arity} would raise arity; that is a prolongation",
            remedy="Build the prolongation explicitly and pass it through R02.",
        )
    lowers = dst.element.arity < src.element.arity
    if lowers:
        raise ArityError(
            f"restriction {src.label!r} -> {dst.label!r} lowers arity "
            f"{src.element.arity} -> {dst.element.arity}",
            remedy="Use project_along(): an arity-lowering map must name the "
            "direction field it projects onto.",
        )
    m = np.asarray(membership, dtype=np.int64).reshape(-1)
    n_src = int(src.n_elements or m.shape[0])
    n_dst = int(dst.n_elements or (int(m.max()) + 1))
    if m.shape[0] != n_src:
        raise OntologyError(
            f"membership has {m.shape[0]} entries for a source support of "
            f"{n_src} elements"
        )
    w = (
        np.ones(n_src)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if w.shape[0] != n_src:
        raise OntologyError("weights must have one entry per source element")
    if np.any(w < 0):
        raise OntologyError("restriction weights must be non-negative")

    ok = m >= 0
    tot = np.zeros(n_dst)
    np.add.at(tot, m[ok], w[ok])
    live = ok & (tot[np.clip(m, 0, n_dst - 1)] > 0)
    rows = m[live]
    colsi = np.nonzero(live)[0]
    vals = w[live] / tot[rows]
    import scipy.sparse as sp

    R = sp.csr_matrix((vals, (rows, colsi)), shape=(n_dst, n_src))
    n_nonempty = int(np.count_nonzero(tot > 0))
    arity = src.element.arity
    M = _block(R, arity)
    if M.shape[0] * M.shape[1] <= _SPARSE_ABOVE:
        M = dense(M)
    return SupportMap(
        src=src,
        dst=dst,
        matrix=M,
        direction="restriction",
        method=method,
        manufactures_dof=False,
        lowers_arity=False,
        unresolved_rank=int(M.shape[1] - _rank(M, known=n_nonempty * arity)),
        psf=dst.psf,
        notes=f"{int((m < 0).sum())} source element(s) belong to no destination "
        "element and are carried as zero rows, not averaged into a neighbour",
    )


def project_along(
    src: Support,
    directions: np.ndarray,
    *,
    name: str,
    units: str | None = None,
) -> tuple[Support, SupportMap]:
    """The arity-lowering map: project a vector support onto a direction field.

    ``directions`` is ``(n_elements, arity)`` and need not be normalised -- the
    norm is the *coherence* of the element and is preserved, because a parcel
    whose face normals partly cancel contributes less than its area suggests.
    Normalising here would throw that away, which is the same error in a
    different place.

    Returns the derived scalar support (which declares ``projected_along=name``,
    so the projection is recoverable) and the map onto it.
    """
    if src.element is None:
        raise OntologyError("project_along requires src.element")
    if src.element.arity == 1:
        raise ArityError(
            f"support {src.label!r} is already scalar; there is nothing to project"
        )
    D = np.asarray(directions, dtype=float)
    n = int(src.n_elements or D.shape[0])
    if D.shape != (n, src.element.arity):
        raise OntologyError(
            f"directions must be {(n, src.element.arity)}, got {D.shape}"
        )
    dst_element = ElementSpec.scalar(
        units or str(src.element.units),
        label=f"{src.element.label or src.element.kind} projected onto {name}",
        projected_along=name,
    )
    dst = Support(
        kind=src.kind,
        frame=src.frame,
        units=Unit(units or str(src.units)),
        psf=src.psf,
        extent=src.extent,
        n_elements=n,
        resolution=src.resolution,
        extent_units=src.extent_units,
        label=f"{src.label}|{name}" if src.label else name,
        element=dst_element,
    )
    import scipy.sparse as sp

    ar = src.element.arity
    rows = np.repeat(np.arange(n), ar)
    colsi = np.arange(n * ar)
    M: Any = sp.csr_matrix((D.reshape(-1), (rows, colsi)), shape=(n, n * ar))
    if n * n * ar <= _SPARSE_ABOVE:
        M = dense(M)
    return dst, SupportMap(
        src=src,
        dst=dst,
        matrix=M,
        direction="projection",
        method=f"inner product with {name}",
        manufactures_dof=False,
        lowers_arity=True,
        projected_along=name,
        unresolved_rank=int(
            M.shape[1] - _rank(M, known=int(np.count_nonzero(np.abs(D).sum(1) > 0)))
        ),
        psf=src.psf,
        notes=(
            "direction norms are preserved, not normalised: |sum a_v n_v| / sum a_v "
            "is the element's coherence and a parcel at coherence 0.28 contributes "
            "a quarter of what its area suggests"
        ),
    )


def embed_along(
    src: Support,
    directions: np.ndarray,
    *,
    name: str,
    component_frame: str | None = None,
) -> tuple[Support, SupportMap]:
    """The arity-**raising** map: a declared scalar back into the vector space.

    This is :func:`project_along` run backwards, and it is admissible for
    exactly one reason: the scalar declared ``projected_along``, so the
    direction it was projected onto is known and the embedding is *determined*
    rather than chosen.  A scalar with ``projected_along=None`` has no embedding
    and :func:`element_join` already refuses it.

    Why this matters concretely: the "parcel net dipole moment" restriction that
    retains 51.7 % of the whitened lead field against the declared pair's 5.6 %
    is ``area-weighted sum . embed_along(surface normals)``.  It is available
    only because the normals are a declared annotation.  Orientation is not
    recovered by the algebra; it is *carried* by it, from the one place it was
    ever measured.

    ``manufactures_dof`` is ``False`` -- the map invents no values -- but
    ``unresolved_rank`` counts the destination directions the source cannot
    reach, and a *state* on the destination must carry prior variance there.
    """
    if src.element is None or src.element.arity != 1:
        raise ArityError(
            "embed_along takes a scalar support",
            offending_object=src.element,
        )
    if src.element.projected_along is None:
        raise ArityError(
            f"scalar support {src.label!r} declares no projected_along, so there "
            "is no direction to embed it along",
            remedy=(
                "This is the same refusal as element_join's. An orientation that "
                "was never declared cannot be recovered, only invented."
            ),
            offending_object=src.label,
        )
    if src.element.projected_along != name:
        raise ArityError(
            f"scalar support {src.label!r} was projected along "
            f"{src.element.projected_along!r}; embedding it along {name!r} would "
            "silently change what the numbers mean",
            offending_object=src.element.projected_along,
        )
    D = np.asarray(directions, dtype=float)
    n = int(src.n_elements or D.shape[0])
    if D.ndim != 2 or D.shape[0] != n:
        raise OntologyError(f"directions must be ({n}, arity), got {D.shape}")
    arity = int(D.shape[1])
    frame = component_frame or str(src.frame)
    dst_element = ElementSpec.vector3(str(src.element.units), frame) if arity == 3 else (
        ElementSpec(
            kind="tensor2" if arity == 9 else "vector",
            arity=arity,
            units=src.element.units,
            component_frame=FrameId(frame),
        )
    )
    dst = src.model_copy(
        update={
            "element": dst_element,
            "label": f"{src.label}^{name}" if src.label else name,
        }
    )
    import scipy.sparse as sp

    rows = np.arange(n * arity)
    colsi = np.repeat(np.arange(n), arity)
    M: Any = sp.csr_matrix((D.reshape(-1), (rows, colsi)), shape=(n * arity, n))
    if n * n * arity <= _SPARSE_ABOVE:
        M = dense(M)
    rank = int(np.count_nonzero(np.abs(D).sum(1) > 0))
    return dst, SupportMap(
        src=src,
        dst=dst,
        matrix=M,
        direction="prolongation",
        method=f"embedding along the declared direction field {name!r}",
        manufactures_dof=False,
        lowers_arity=False,
        projected_along=name,
        unresolved_rank=int(M.shape[0] - rank),
        psf=src.psf,
        notes=(
            "determined, not chosen: the direction field was declared by the "
            "scalar's projected_along. The destination directions outside the "
            "image are set to zero, which is a claim, so a state on the "
            "destination must carry prior variance there."
        ),
    )


def common_refinement(
    a: Support,
    b: Support,
    assign_a: Sequence[int] | np.ndarray,
    assign_b: Sequence[int] | np.ndarray,
    *,
    atom_weights: Sequence[float] | np.ndarray | None = None,
    atom_frame: str | None = None,
    label: str = "",
) -> Refinement:
    """The common refinement of ``a`` and ``b`` and the four maps around it.

    ``assign_a[i]`` / ``assign_b[i]`` give, for each atom of a shared finest
    index set, the element of ``a`` / ``b`` it belongs to (``-1`` for none).
    The refinement's elements are the non-empty ``(a_j, b_k)`` cells.

    Both supports must declare an :class:`ElementSpec`; the refinement's element
    is :func:`element_join` of the two, so joining a scalar support with a
    vector one raises unless the scalar named the direction it was projected
    along.
    """
    if a.element is None or b.element is None:
        raise OntologyError(
            "common_refinement requires both supports to declare an ElementSpec; "
            f"a.element={a.element!r}, b.element={b.element!r}",
            remedy=(
                "A support with no ElementSpec has no dimension. Declaring one is "
                "the whole content of O-2's orientation constraint."
            ),
        )
    if a.frame != b.frame:
        raise OntologyError(
            f"supports live in different frames ({a.frame!r} vs {b.frame!r}); the "
            "refinement of two supports in unrelated frames is not defined",
            remedy="Transform one into the other's frame through the FrameGraph "
            "first (R01).",
        )
    join = element_join(a.element, b.element)

    ia = np.asarray(assign_a, dtype=np.int64).reshape(-1)
    ib = np.asarray(assign_b, dtype=np.int64).reshape(-1)
    if ia.shape != ib.shape:
        raise OntologyError(
            f"the two memberships describe different atom sets: {ia.shape} vs "
            f"{ib.shape}"
        )
    n_atoms = ia.shape[0]
    w = (
        np.ones(n_atoms)
        if atom_weights is None
        else np.asarray(atom_weights, dtype=float).reshape(-1)
    )
    if w.shape[0] != n_atoms:
        raise OntologyError("atom_weights must have one entry per atom")

    ok = (ia >= 0) & (ib >= 0)
    pairs = np.stack([ia[ok], ib[ok]], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_cells = int(uniq.shape[0])
    if n_cells == 0:
        raise OntologyError(
            "the two supports share no atom; their common refinement is empty",
            remedy="These supports are not comparable; say so rather than "
            "producing a zero-dimensional map.",
        )
    cell_w = np.zeros(n_cells)
    np.add.at(cell_w, inv, w[ok])

    n_a = int(a.n_elements or (int(ia.max()) + 1))
    n_b = int(b.n_elements or (int(ib.max()) + 1))

    c = Support(
        kind=a.kind if a.kind == b.kind else "mesh",
        frame=a.frame if atom_frame is None else FrameId(atom_frame),
        units=join.units,
        psf=None,
        extent=None,
        n_elements=n_cells,
        resolution=None,
        extent_units=a.extent_units,
        label=label or f"{a.label or a.kind}^{b.label or b.kind}",
        element=join,
    )

    # --- restrictions out of the refinement (always admissible) -----------
    to_a = restriction_between(
        c, _retyped(a, join), uniq[:, 0], cell_w, method="weight-averaged cell mean"
    )
    to_b = restriction_between(
        c, _retyped(b, join), uniq[:, 1], cell_w, method="weight-averaged cell mean"
    )

    # --- prolongations into it (R02's subject) ----------------------------
    from_a = _indicator_prolongation(_retyped(a, join), c, uniq[:, 0], n_a, join.arity)
    from_b = _indicator_prolongation(_retyped(b, join), c, uniq[:, 1], n_b, join.arity)

    # --- composed PSF ------------------------------------------------------
    psf = _compose_psf(a, b, c, cell_w)

    c = c.model_copy(update={"psf": psf})
    to_a = _replace_src(to_a, c)
    to_b = _replace_src(to_b, c)
    from_a = _replace_dst(from_a, c)
    from_b = _replace_dst(from_b, c)

    return Refinement(
        support=c,
        to_a=to_a,
        to_b=to_b,
        from_a=from_a,
        from_b=from_b,
        cells=tuple(
            (int(uniq[k, 0]), int(uniq[k, 1]), float(cell_w[k])) for k in range(n_cells)
        ),
        n_unassigned_atoms=int((~ok).sum()),
        composed_psf=psf,
        provenance={
            "a": a.label or a.kind,
            "b": b.label or b.kind,
            "n_atoms": n_atoms,
            "n_a_elements": n_a,
            "n_b_elements": n_b,
            "arity_a": a.element.arity,
            "arity_b": b.element.arity,
            "arity_join": join.arity,
            "orientation_manufactured_from_a": join.arity > a.element.arity,
            "orientation_manufactured_from_b": join.arity > b.element.arity,
            "method": "cell-wise intersection over a shared atom set",
        },
    )


def _retyped(s: Support, element: ElementSpec) -> Support:
    """``s`` with the joined element spec, for shape bookkeeping only."""
    if s.element is not None and s.element.arity == element.arity:
        return s
    return s.model_copy(update={"element": element, "units": element.units})


def _replace_src(m: SupportMap, src: Support) -> SupportMap:
    return SupportMap(
        src=src,
        dst=m.dst,
        matrix=m.matrix,
        direction=m.direction,
        method=m.method,
        manufactures_dof=m.manufactures_dof,
        lowers_arity=m.lowers_arity,
        projected_along=m.projected_along,
        unresolved_rank=m.unresolved_rank,
        prior_sd_unresolved=m.prior_sd_unresolved,
        psf=m.psf,
        notes=m.notes,
    )


def _replace_dst(m: SupportMap, dst: Support) -> SupportMap:
    return SupportMap(
        src=m.src,
        dst=dst,
        matrix=m.matrix,
        direction=m.direction,
        method=m.method,
        manufactures_dof=m.manufactures_dof,
        lowers_arity=m.lowers_arity,
        projected_along=m.projected_along,
        unresolved_rank=m.unresolved_rank,
        prior_sd_unresolved=m.prior_sd_unresolved,
        psf=m.psf,
        notes=m.notes,
    )


def _indicator_prolongation(
    src: Support, dst: Support, membership: np.ndarray, n_src: int, arity: int
) -> SupportMap:
    """``(P x)_cell = x_{element containing cell}`` -- the map the model implies.

    Not ``pinv``: "every cell in this element takes the element's value" is what
    an element-level state *means*, and it is the map the artifact already
    applies whenever it treats a region variable as describing that region.
    """
    import scipy.sparse as sp

    n_cells = int(membership.shape[0])
    P = sp.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), membership)), shape=(n_cells, n_src)
    )
    known = int(np.unique(membership).size) * arity
    M = _block(P, arity)
    if M.shape[0] * M.shape[1] <= _SPARSE_ABOVE:
        M = dense(M)
    rank = _rank(M, known=known)
    return SupportMap(
        src=src,
        dst=dst,
        matrix=M,
        direction="prolongation",
        method="indicator fill",
        manufactures_dof=M.shape[0] > rank,
        lowers_arity=False,
        unresolved_rank=int(M.shape[0] - rank),
        psf=None,
        notes=(
            "the unresolved destination directions carry prior variance, never "
            "reconstructed structure (R02); prior_sd_unresolved is None until "
            "measured"
        ),
    )


def _compose_psf(a: Support, b: Support, c: Support, cell_w: np.ndarray) -> PSF | None:
    """The refinement's point-spread, composed from both parents'.

    Two contributions, kept separate rather than summed into one number:

    * the **geometric** spread of a refinement cell, ``sqrt(mean cell weight)``
      when the weights are areas -- the finest scale the refinement can express;
    * the **inherited** kernels of ``a`` and ``b``, which do not disappear
      because the support got finer.  A refinement of two blurred supports is
      not sharp; it is blurred by both.

    The composed FWHM is the quadrature sum of the parents', floored at the
    geometric scale.  Quadrature is exact for Gaussians and an approximation
    otherwise, and ``nominal=True`` says so whenever either parent's kernel was
    itself nominal or absent.
    """
    scale = float(np.sqrt(np.mean(cell_w))) if cell_w.size else 0.0
    parents = [p for p in (a.psf, b.psf) if p is not None]
    fwhms: list[float] = []
    nominal = len(parents) < 2
    for p in parents:
        if p.fwhm is None:
            nominal = True
            continue
        if p.nominal:
            nominal = True
        fwhms.append(float(max(p.fwhm)))
    composed = math.sqrt(sum(f * f for f in fwhms)) if fwhms else 0.0
    composed = max(composed, scale)
    if composed <= 0.0:
        return None
    return PSF(
        kind="integration_kernel",
        fwhm=(composed,),
        units=c.units,
        extent_units=a.extent_units,
        kernel_ref=None,
        nominal=nominal,
        ledger=None,
    )


# ==========================================================================
# clocks
# ==========================================================================
@dataclass(frozen=True)
class TemporalRefinement:
    """The common refinement of two clocks, and what each map costs in time.

    This is the "5000 Hz EEG against 0.5 Hz BOLD" case.  The refinement is the
    finer clock; the map onto the coarser one is an explicit boxcar over its
    ``integration_window`` (or over one sample period when the window is zero,
    because a sample is not an instant).  ``group_delay`` and ``jitter_sd``
    compose rather than being dropped.
    """

    support: TemporalSupport
    to_a_kernel: np.ndarray
    to_b_kernel: np.ndarray
    to_a_stride: int
    to_b_stride: int
    #: Extra delay the coarse map introduces, seconds, per side.
    delay_a: float
    delay_b: float
    #: Jitter in units of *refinement* samples, per side -- an integer stride is
    #: only exact when the two clocks are commensurate, and this says by how
    #: much they are not.
    incommensurability_a: float
    incommensurability_b: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "dt": self.support.dt,
            "clock": str(self.support.clock),
            "to_a_stride": self.to_a_stride,
            "to_b_stride": self.to_b_stride,
            "to_a_kernel_len": int(self.to_a_kernel.shape[0]),
            "to_b_kernel_len": int(self.to_b_kernel.shape[0]),
            "delay_a_s": self.delay_a,
            "delay_b_s": self.delay_b,
            "incommensurability_a_samples": self.incommensurability_a,
            "incommensurability_b_samples": self.incommensurability_b,
            "provenance": self.provenance,
        }


def _boxcar(dt_fine: float, window: float) -> np.ndarray:
    n = max(1, int(round(window / dt_fine)))
    return np.full(n, 1.0 / n)


def temporal_common_refinement(
    a: TemporalSupport, b: TemporalSupport, *, clock: str | None = None
) -> TemporalRefinement:
    """Common refinement of two temporal supports: the finer clock, with maps.

    The maps are honest about two things the usual resample is not:

    * a sample of the coarse clock integrates over a **window**, so the map is a
      boxcar, not a pick;
    * the two clocks are generally **incommensurate**, so the decimation stride
      is an integer that does not exactly divide the ratio.  The residual is
      reported in refinement samples rather than rounded away.
    """
    fine, coarse = (a, b) if a.dt <= b.dt else (b, a)
    dt = fine.dt
    support = TemporalSupport(
        clock=clock or str(fine.clock),
        dt=dt,
        integration_window=fine.integration_window,
        group_delay=fine.group_delay,
        jitter_sd=fine.jitter_sd,
    )

    def side(t: TemporalSupport) -> tuple[np.ndarray, int, float, float]:
        ratio = t.dt / dt
        stride = max(1, int(round(ratio)))
        window = t.integration_window if t.integration_window > 0 else t.dt
        k = _boxcar(dt, window)
        delay = t.group_delay + 0.5 * window
        return k, stride, delay, abs(ratio - stride)

    ka, sa, da, ia = side(a)
    kb, sb, db, ib = side(b)
    return TemporalRefinement(
        support=support,
        to_a_kernel=ka,
        to_b_kernel=kb,
        to_a_stride=sa,
        to_b_stride=sb,
        delay_a=da,
        delay_b=db,
        incommensurability_a=ia,
        incommensurability_b=ib,
        provenance={
            "a_dt": a.dt,
            "b_dt": b.dt,
            "a_clock": str(a.clock),
            "b_clock": str(b.clock),
            "finer": "a" if a.dt <= b.dt else "b",
            "note": (
                "a coarse sample integrates over its window; the map is a boxcar, "
                "not a pick. Clocks are not assumed commensurate."
            ),
        },
    )
