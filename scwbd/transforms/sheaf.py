"""The resolution poset, made operational -- thesis §2.6 and §4.2.

    "The design can be made operational rather than merely 'sheaf-like.' Let the
     site C contain declared domain--support objects and admissible covers; let
     F(U) be the set or distribution of states admissible on U; and let every
     inclusion V -> U own a versioned restriction rho^U_V."

    "Omega_abc(x) = || rho^{Ua}_{Uc} x - rho^{Ub}_{Uc} rho^{Ua}_{Ub} x ||_{Wc}
     <= tau_abc"

    "If they fail, the nonzero residual is an *obstruction certificate*: the
     runtime preserves separate views, branches or marginalizes the posterior,
     or refuses the global materialization."

Two refusals live here:

* **R02** -- a :class:`Prolongation` cannot exist without a declared
  :class:`Restriction` partner *and* tested coverage.  Constructing one raises.
  Prolongation returns a :class:`FineDistribution`, never a point: the
  unresolved directions (the null space of the restriction) get prior variance
  instead of invented structure.
* **R03** -- :func:`glue` returns an :class:`ObstructionCertificate` naming the
  failed path instead of a global section, and
  :attr:`GluingResult.global_section` raises if you ask for the raster anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .errors import (
    CocycleObstructionError,
    ProlongationWithoutRestrictionError,
    SiteError,
)
from .se3 import DTYPE, as_tensor

# --------------------------------------------------------------------------
# objects of the site
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportObject:
    """A domain--support object ``U`` of the site ``C``.

    Elements need not be grids: §2.6 admits "surface meshes, parcels, laminar
    depths, voxel fields, tract endpoint distributions, spectral bands, event
    windows, behavioral episodes, or sparse local tiles".  Here an object is
    identified by an ordered tuple of *element keys* (whatever indexes that
    support) plus its resolution level and the weight matrix ``W`` in which
    residuals on it are measured.
    """

    id: str
    elements: tuple[Any, ...]
    resolution: str = "native"
    kind: str = "field"
    units: str = "dimensionless"
    weight: torch.Tensor | None = None
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if len(set(self.elements)) != len(self.elements):
            raise SiteError(
                f"support object {self.id!r} has duplicate elements",
                remedy="Element keys index the support and must be unique.",
                offending_object=self.id,
            )
        if self.weight is not None:
            W = as_tensor(self.weight)
            n = len(self.elements)
            if W.shape != (n, n):
                raise SiteError(
                    f"weight matrix for {self.id!r} must be {n}x{n}, got {tuple(W.shape)}",
                    remedy="W_c weights residuals on this support.",
                    offending_object=self.id,
                )
            object.__setattr__(self, "weight", 0.5 * (W + W.T))

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def W(self) -> torch.Tensor:
        n = len(self.elements)
        return torch.eye(n, dtype=DTYPE) if self.weight is None else self.weight

    def norm(self, r: torch.Tensor) -> float:
        """``||r||_W`` -- the weighted norm the tolerance is stated in."""
        r = torch.as_tensor(r, dtype=DTYPE).reshape(-1)
        return float(torch.sqrt(torch.clamp(r @ self.W @ r, min=0.0)))


@dataclass(frozen=True)
class Restriction:
    """A versioned restriction ``rho^U_V`` for an inclusion ``V -> U``.

    §2.6: "every inclusion V -> U owns a versioned restriction".  The version is
    part of the identity: replacing a parcellation's averaging kernel changes
    the restriction, and a cocycle residual computed under mixed versions is
    meaningless.
    """

    source: str  # U (the larger support)
    target: str  # V (the smaller support)
    matrix: torch.Tensor
    version: str = "1.0.0"
    method: str = "undeclared"
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", as_tensor(self.matrix))
        if self.matrix.ndim != 2:
            raise SiteError(
                f"restriction {self.label} must be a matrix, got shape "
                f"{tuple(self.matrix.shape)}",
                remedy="Supply rho^U_V as a linear operator.",
                offending_object=self.label,
            )

    @property
    def label(self) -> str:
        return f"rho^{self.source}_{self.target}@{self.version}"

    def apply(self, x: Any) -> torch.Tensor:
        v = torch.as_tensor(x, dtype=DTYPE).reshape(-1)
        if v.numel() != self.matrix.shape[1]:
            raise SiteError(
                f"{self.label} expects {self.matrix.shape[1]} inputs, got {v.numel()}",
                remedy="Restrict a section defined on the source support.",
                offending_object=self.label,
            )
        return self.matrix @ v


@dataclass(frozen=True)
class Section:
    """A local section ``x_a in F(U_a)`` with its ledger."""

    support: str
    values: torch.Tensor
    cov: torch.Tensor | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", torch.as_tensor(self.values, dtype=DTYPE).reshape(-1))
        if self.cov is not None:
            C = as_tensor(self.cov)
            n = self.values.numel()
            if C.shape != (n, n):
                raise SiteError(
                    f"section on {self.support!r} has covariance of shape "
                    f"{tuple(C.shape)}, expected {(n, n)}",
                    remedy="Match the covariance to the section.",
                    offending_object=self.support,
                )
            object.__setattr__(self, "cov", 0.5 * (C + C.T))


@dataclass(frozen=True)
class Cover:
    """An admissible cover ``{U_a} -> U`` of the site."""

    target: str
    parts: tuple[str, ...]
    id: str = "cover"

    def __post_init__(self) -> None:
        if not self.parts:
            raise SiteError(
                f"cover {self.id!r} of {self.target!r} has no parts",
                remedy="A cover needs at least one member.",
                offending_object=self.id,
            )


# --------------------------------------------------------------------------
# the site
# --------------------------------------------------------------------------


class Site:
    """Declared supports, restrictions and covers, with the sheaf checks."""

    def __init__(self) -> None:
        self._objects: dict[str, SupportObject] = {}
        self._rho: dict[tuple[str, str], Restriction] = {}
        self._covers: dict[str, Cover] = {}

    # -- declaration -------------------------------------------------------

    def add_object(self, obj: SupportObject) -> SupportObject:
        if obj.id in self._objects:
            raise SiteError(
                f"support object {obj.id!r} is already declared",
                remedy="Support ids are unique.",
                offending_object=obj.id,
            )
        self._objects[obj.id] = obj
        return obj

    def object(self, oid: str) -> SupportObject:
        try:
            return self._objects[oid]
        except KeyError:
            raise SiteError(
                f"support object {oid!r} is not declared (known: {sorted(self._objects)})",
                remedy="Declare the support before restricting to it.",
                offending_object=oid,
            ) from None

    def add_restriction(self, rho: Restriction) -> Restriction:
        U, V = self.object(rho.source), self.object(rho.target)
        if rho.matrix.shape != (len(V), len(U)):
            raise SiteError(
                f"{rho.label} must be {len(V)}x{len(U)} to map F({U.id}) -> F({V.id}), "
                f"got {tuple(rho.matrix.shape)}",
                remedy="Restriction maps the larger support onto the smaller.",
                offending_object=rho.label,
            )
        key = (rho.source, rho.target)
        if key in self._rho and self._rho[key].version != rho.version:
            raise SiteError(
                f"restriction {rho.source}->{rho.target} is already declared at "
                f"version {self._rho[key].version}; re-declaring it at "
                f"{rho.version} would silently change every past residual",
                remedy="Version the supports, or start a new site.",
                offending_object=rho.label,
            )
        self._rho[key] = rho
        return rho

    def restriction(self, source: str, target: str) -> Restriction:
        try:
            return self._rho[(source, target)]
        except KeyError:
            raise SiteError(
                f"no declared restriction rho^{source}_{target}",
                remedy=(
                    "§2.6: 'Some pairs have no defensible map.' Declare the "
                    "restriction with its version and method, or accept that "
                    "these two supports are not comparable."
                ),
                offending_object=(source, target),
            ) from None

    def has_restriction(self, source: str, target: str) -> bool:
        return (source, target) in self._rho

    def add_cover(self, cover: Cover) -> Cover:
        self.object(cover.target)
        for p in cover.parts:
            self.object(p)
        target_elems = set(self.object(cover.target).elements)
        covered: set[Any] = set()
        for p in cover.parts:
            covered |= set(self.object(p).elements)
        missing = target_elems - covered
        if missing:
            raise SiteError(
                f"cover {cover.id!r} does not cover {len(missing)} element(s) of "
                f"{cover.target!r} (e.g. {sorted(map(str, missing))[:5]})",
                remedy=(
                    "An admissible cover must span the target support. Missing "
                    "regions are never filled by averaging the neighbours."
                ),
                offending_object=cover.id,
            )
        self._covers[cover.id] = cover
        return cover

    def cover(self, cid: str) -> Cover:
        try:
            return self._covers[cid]
        except KeyError:
            raise SiteError(
                f"cover {cid!r} is not declared",
                remedy="Declare the cover.",
                offending_object=cid,
            ) from None

    @property
    def objects(self) -> dict[str, SupportObject]:
        return dict(self._objects)

    @property
    def restrictions(self) -> dict[tuple[str, str], Restriction]:
        return dict(self._rho)

    # -- overlap and cocycle ----------------------------------------------

    def overlap_residual(
        self, sec_a: Section, sec_b: Section, overlap: str
    ) -> "OverlapResidual":
        """``|| rho^{Ua}_{Uc} x_a - rho^{Ub}_{Uc} x_b ||_{Wc}``."""
        Uc = self.object(overlap)
        ra = self.restriction(sec_a.support, overlap)
        rb = self.restriction(sec_b.support, overlap)
        d = ra.apply(sec_a.values) - rb.apply(sec_b.values)
        return OverlapResidual(
            a=sec_a.support,
            b=sec_b.support,
            overlap=overlap,
            residual=d,
            norm=Uc.norm(d),
            tolerance=Uc.tolerance,
            path_a=ra.label,
            path_b=rb.label,
        )

    def cocycle_residual(
        self, x: Any, a: str, b: str, c: str, *, tolerance: float | None = None
    ) -> "CocycleResidual":
        """``Omega_abc(x)`` exactly as written in §2.6.

        Requires the nesting ``U_c ⊆ U_b ⊆ U_a`` to own all three restrictions:
        the direct ``rho^{Ua}_{Uc}`` and the composite
        ``rho^{Ub}_{Uc} rho^{Ua}_{Ub}``.  The two are different objects with
        different versions, and the whole point of the check is that they need
        not agree.
        """
        Uc = self.object(c)
        r_ac = self.restriction(a, c)
        r_ab = self.restriction(a, b)
        r_bc = self.restriction(b, c)
        v = torch.as_tensor(x, dtype=DTYPE).reshape(-1)
        direct = r_ac.apply(v)
        composed = r_bc.apply(r_ab.apply(v))
        d = direct - composed
        tau = Uc.tolerance if tolerance is None else tolerance
        return CocycleResidual(
            a=a,
            b=b,
            c=c,
            residual=d,
            norm=Uc.norm(d),
            tolerance=tau,
            direct_path=r_ac.label,
            composed_path=f"{r_bc.label} o {r_ab.label}",
        )

    def nested_triples(self) -> list[tuple[str, str, str]]:
        """All ``(a, b, c)`` for which both the direct and composite paths exist."""
        out = []
        for (a, b) in self._rho:
            for (b2, c) in self._rho:
                if b2 != b or c == a:
                    continue
                if (a, c) in self._rho:
                    out.append((a, b, c))
        return out


# --------------------------------------------------------------------------
# residual / certificate records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapResidual:
    a: str
    b: str
    overlap: str
    residual: torch.Tensor
    norm: float
    tolerance: float
    path_a: str
    path_b: str

    @property
    def within_tolerance(self) -> bool:
        return self.norm <= self.tolerance

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "overlap",
            "sections": [self.a, self.b],
            "overlap": self.overlap,
            "norm_W": self.norm,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "path_a": self.path_a,
            "path_b": self.path_b,
        }


@dataclass(frozen=True)
class CocycleResidual:
    a: str
    b: str
    c: str
    residual: torch.Tensor
    norm: float
    tolerance: float
    direct_path: str
    composed_path: str

    @property
    def within_tolerance(self) -> bool:
        return self.norm <= self.tolerance

    @property
    def failed_path(self) -> str:
        """Human-readable identification of the path that did not commute."""
        return (
            f"Omega_{{{self.a},{self.b},{self.c}}}: direct [{self.direct_path}] "
            f"vs composed [{self.composed_path}]"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "cocycle",
            "triple": [self.a, self.b, self.c],
            "norm_W": self.norm,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "direct_path": self.direct_path,
            "composed_path": self.composed_path,
            "failed_path": self.failed_path,
            "worst_element_index": int(torch.argmax(self.residual.abs())),
        }


@dataclass(frozen=True)
class ObstructionCertificate:
    """Machine-readable "no global section, and here is why" (refusal R03).

    Carries the failing overlap and cocycle residuals -- each naming the exact
    restriction path that did not commute -- and the sections that must be kept
    apart.  A consumer is expected to branch or marginalize the posterior over
    :attr:`branches`; it must not average them into a raster.
    """

    target: str
    cover: str
    failed_overlaps: tuple[OverlapResidual, ...]
    failed_cocycles: tuple[CocycleResidual, ...]
    branches: tuple[Section, ...]
    reason: str

    @property
    def failed_paths(self) -> tuple[str, ...]:
        return tuple(
            [f"overlap {o.overlap}: [{o.path_a}] vs [{o.path_b}]" for o in self.failed_overlaps]
            + [c.failed_path for c in self.failed_cocycles]
        )

    @property
    def max_residual(self) -> float:
        vals = [o.norm for o in self.failed_overlaps] + [c.norm for c in self.failed_cocycles]
        return max(vals) if vals else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "refusal": "R03",
            "target": self.target,
            "cover": self.cover,
            "reason": self.reason,
            "failed_paths": list(self.failed_paths),
            "max_residual_W": self.max_residual,
            "overlaps": [o.as_record() for o in self.failed_overlaps],
            "cocycles": [c.as_record() for c in self.failed_cocycles],
            "branches": [s.support for s in self.branches],
            "action": (
                "separate views preserved; posterior branched or marginalized; "
                "no global raster materialized"
            ),
        }

    def raise_(self) -> None:
        raise CocycleObstructionError(
            f"cannot glue a global section on {self.target!r}: {self.reason} "
            f"(worst residual {self.max_residual:.6g}; failed paths: "
            f"{'; '.join(self.failed_paths)})",
            certificate=self,
        )


@dataclass(frozen=True)
class GlobalSection:
    """A glued global section, with the gluing residual that earned it."""

    support: str
    values: torch.Tensor
    cov: torch.Tensor | None
    gluing_residual: float
    per_part_residual: dict[str, float]
    overlaps: tuple[OverlapResidual, ...]
    cocycles: tuple[CocycleResidual, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "gluing_residual": self.gluing_residual,
            "per_part_residual": self.per_part_residual,
            "overlaps": [o.as_record() for o in self.overlaps],
            "cocycles": [c.as_record() for c in self.cocycles],
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class GluingResult:
    """Either a global section or an obstruction certificate -- never both."""

    section: GlobalSection | None
    certificate: ObstructionCertificate | None

    @property
    def ok(self) -> bool:
        return self.section is not None

    @property
    def global_section(self) -> GlobalSection:
        """Refuses to materialize a raster when an obstruction exists (R03)."""
        if self.certificate is not None:
            self.certificate.raise_()
        assert self.section is not None
        return self.section

    def as_record(self) -> dict[str, Any]:
        return (
            self.section.as_record() if self.ok else self.certificate.as_record()
        )


# --------------------------------------------------------------------------
# gluing
# --------------------------------------------------------------------------


def glue(
    site: Site,
    sections: Sequence[Section],
    cover: str | Cover,
    *,
    check_cocycles: bool = True,
    ridge: float = 1e-10,
) -> GluingResult:
    """Attempt a global section over ``cover``; otherwise return a certificate.

    Overlap compatibility is tested on every declared pairwise overlap support,
    the cocycle condition on every nested triple reachable from the cover's
    parts.  If both pass, the global section is the ``W``-weighted least-squares
    solution of ``rho^{U}_{U_a} g = x_a``, and the *gluing residual is recorded*
    rather than assumed zero.
    """
    cov_obj = site.cover(cover) if isinstance(cover, str) else cover
    target = site.object(cov_obj.target)
    by_support = {s.support: s for s in sections}
    missing = [p for p in cov_obj.parts if p not in by_support]
    if missing:
        raise SiteError(
            f"cover {cov_obj.id!r} needs sections on {missing}, which were not supplied",
            remedy=(
                "Missing data is never imputed as zero or an average brain "
                "(ARCHITECTURE.md §7.1). Supply the section or shrink the cover."
            ),
            offending_object=missing,
        )

    # --- overlaps -------------------------------------------------------
    overlaps: list[OverlapResidual] = []
    for i, a in enumerate(cov_obj.parts):
        for b in cov_obj.parts[i + 1 :]:
            for oid, obj in site.objects.items():
                if oid in (a, b):
                    continue
                if site.has_restriction(a, oid) and site.has_restriction(b, oid):
                    overlaps.append(site.overlap_residual(by_support[a], by_support[b], oid))

    # --- cocycles -------------------------------------------------------
    cocycles: list[CocycleResidual] = []
    if check_cocycles:
        for (a, b, c) in site.nested_triples():
            if a not in by_support:
                continue
            cocycles.append(site.cocycle_residual(by_support[a].values, a, b, c))

    failed_o = tuple(o for o in overlaps if not o.within_tolerance)
    failed_c = tuple(c for c in cocycles if not c.within_tolerance)
    if failed_o or failed_c:
        reason_bits = []
        if failed_o:
            reason_bits.append(
                f"{len(failed_o)} overlap residual(s) exceed tolerance "
                f"(worst {max(o.norm for o in failed_o):.6g} > "
                f"{min(o.tolerance for o in failed_o):.6g})"
            )
        if failed_c:
            reason_bits.append(
                f"{len(failed_c)} cocycle residual(s) exceed tolerance "
                f"(worst {max(c.norm for c in failed_c):.6g} > "
                f"{min(c.tolerance for c in failed_c):.6g})"
            )
        cert = ObstructionCertificate(
            target=cov_obj.target,
            cover=cov_obj.id,
            failed_overlaps=failed_o,
            failed_cocycles=failed_c,
            branches=tuple(sections),
            reason="; ".join(reason_bits),
        )
        return GluingResult(None, cert)

    # --- least-squares glue ---------------------------------------------
    n = len(target)
    A = torch.zeros((n, n), dtype=DTYPE)
    rhs = torch.zeros(n, dtype=DTYPE)
    for p in cov_obj.parts:
        rho = site.restriction(cov_obj.target, p)
        W = site.object(p).W
        R = rho.matrix
        A = A + R.T @ W @ R
        rhs = rhs + R.T @ W @ by_support[p].values
    rank = int(torch.linalg.matrix_rank(A))
    A = A + ridge * torch.eye(n, dtype=DTYPE)
    if rank < n:
        raise SiteError(
            f"cover {cov_obj.id!r} does not determine a global section on "
            f"{cov_obj.target!r} (normal-matrix rank {rank} < {n})",
            remedy=(
                "Some degrees of freedom on the target are unobserved by every "
                "part. Refuse the global state or restrict the target support "
                "to what the cover actually determines."
            ),
            offending_object=cov_obj.id,
        )
    g = torch.linalg.solve(A, rhs)

    per_part: dict[str, float] = {}
    total = 0.0
    for p in cov_obj.parts:
        rho = site.restriction(cov_obj.target, p)
        r = rho.apply(g) - by_support[p].values
        per_part[p] = site.object(p).norm(r)
        total += per_part[p] ** 2

    return GluingResult(
        GlobalSection(
            support=cov_obj.target,
            values=g,
            cov=None,
            gluing_residual=math.sqrt(total),
            per_part_residual=per_part,
            overlaps=tuple(overlaps),
            cocycles=tuple(cocycles),
            provenance={
                "cover": cov_obj.id,
                "parts": list(cov_obj.parts),
                "restriction_versions": {
                    p: site.restriction(cov_obj.target, p).version for p in cov_obj.parts
                },
                "method": "W-weighted least squares",
                "note": "gluing residual measured, not assumed zero",
            },
        ),
        None,
    )


def glue_or_raise(site: Site, sections: Sequence[Section], cover: str | Cover, **kw: Any):
    """:func:`glue`, but an obstruction raises :class:`CocycleObstructionError`."""
    res = glue(site, sections, cover, **kw)
    return res.global_section


# --------------------------------------------------------------------------
# restriction / prolongation pairs (§4.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """Tested coverage of a prolongation (the second half of refusal R02).

    ``landmarks`` are held-out fine states with known coarse observations;
    ``heldout_error`` is the error of the prolongation on them.  ``tested`` is
    only ``True`` when a real test produced these numbers -- there is no way to
    set it from a declaration alone.
    """

    n_landmarks: int
    heldout_error: float
    coarse_roundtrip_error: float
    fraction_of_fine_variance_explained: float
    domain: dict[str, Any] = field(default_factory=dict)

    @property
    def tested(self) -> bool:
        return self.n_landmarks > 0

    def as_record(self) -> dict[str, Any]:
        return {
            "n_landmarks": self.n_landmarks,
            "heldout_error": self.heldout_error,
            "coarse_roundtrip_error": self.coarse_roundtrip_error,
            "fraction_of_fine_variance_explained": self.fraction_of_fine_variance_explained,
            "tested": self.tested,
            "domain": self.domain,
        }


@dataclass(frozen=True)
class FineDistribution:
    """What a prolongation returns: a *distribution* over fine states.

    §2.6: "a prolongation P_{b->a} may produce a distribution rather than a
    unique fine state."  §4.2: "P_i expands incoming coarse messages into a
    conditional distribution over fine states."  The mean is the minimum-norm
    reconstruction; the covariance carries prior variance on the null space of
    the restriction -- i.e. exactly the structure the coarse observation cannot
    see.  Collapsing this to its mean is how coarse data becomes fictitious fine
    precision.
    """

    mean: torch.Tensor
    cov: torch.Tensor
    resolved_rank: int
    unresolved_rank: int
    unresolved_basis: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)

    def sample(self, n: int = 1, *, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(seed))
        evals, evecs = torch.linalg.eigh(self.cov)
        L = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
        z = torch.randn((n, self.mean.numel()), dtype=DTYPE, generator=g)
        return self.mean + z @ L.T

    @property
    def sd(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(torch.diagonal(self.cov), min=0.0))

    def as_point(self) -> torch.Tensor:
        raise ProlongationWithoutRestrictionError(
            "a prolongation result is a distribution over fine states, not a point; "
            f"{self.unresolved_rank} of {self.mean.numel()} fine directions are "
            "unresolved by the coarse observation",
            remedy=(
                "Use .mean explicitly if you accept a point estimate, and carry "
                ".cov with it. Never hand the mean onward as if it were measured."
            ),
            offending_object="FineDistribution.as_point",
        )


@dataclass(frozen=True)
class Prolongation:
    """``P_i`` -- admissible only with a restriction partner and tested coverage.

    Refusal R02: "Prolongation without a declared restriction partner and tested
    coverage ... A coarse observation would be converted into unsupported fine
    structure."  Both conditions are enforced in ``__post_init__``, so an
    inadmissible prolongation cannot be constructed, let alone called.
    """

    restriction: Restriction | None
    matrix: torch.Tensor | None
    coverage: CoverageReport | None
    prior_sd_unresolved: float
    prior_sd_resolved: float = 0.0
    version: str = "1.0.0"
    method: str = "pseudoinverse"
    max_heldout_error: float = math.inf

    def __post_init__(self) -> None:
        if self.restriction is None:
            raise ProlongationWithoutRestrictionError(
                "prolongation declared without a restriction partner",
                remedy=(
                    "Provide paired maps R_i and P_i, round-trip and held-out "
                    "landmark tests, and an out-of-support uncertainty policy; "
                    "otherwise return a distribution or refuse."
                ),
                offending_object=self.method,
            )
        if self.coverage is None or not self.coverage.tested:
            raise ProlongationWithoutRestrictionError(
                "prolongation declared without tested coverage "
                f"(coverage={self.coverage})",
                remedy=(
                    "Run held-out landmark tests (measure_coverage) and attach the "
                    "CoverageReport. An untested prolongation manufactures fine "
                    "structure that no observation supports."
                ),
                offending_object=self.restriction.label,
            )
        if self.coverage.heldout_error > self.max_heldout_error:
            raise ProlongationWithoutRestrictionError(
                f"prolongation coverage test failed: held-out landmark error "
                f"{self.coverage.heldout_error:.6g} exceeds the declared maximum "
                f"{self.max_heldout_error:.6g}",
                remedy="Improve the map, narrow its validity domain, or refuse it.",
                offending_object=self.restriction.label,
            )
        if self.prior_sd_unresolved <= 0:
            raise ProlongationWithoutRestrictionError(
                "prolongation must declare a positive prior sd for the fine "
                "directions the coarse observation cannot resolve",
                remedy=(
                    "The null space of the restriction is unmeasured. Zero "
                    "variance there is a claim of infinite precision."
                ),
                offending_object=self.prior_sd_unresolved,
            )
        R = self.restriction.matrix
        if self.matrix is None:
            # minimum-norm right inverse
            object.__setattr__(self, "matrix", torch.linalg.pinv(R))
        M = as_tensor(self.matrix)
        if M.shape != (R.shape[1], R.shape[0]):
            raise ProlongationWithoutRestrictionError(
                f"prolongation must be {R.shape[1]}x{R.shape[0]} to partner "
                f"{self.restriction.label}, got {tuple(M.shape)}",
                remedy="P_i maps coarse -> fine; R_i maps fine -> coarse.",
                offending_object=self.restriction.label,
            )
        object.__setattr__(self, "matrix", M)

    @property
    def label(self) -> str:
        return f"P^{self.restriction.target}_{self.restriction.source}@{self.version}"

    def __call__(self, coarse: Any, *, coarse_cov: Any | None = None) -> FineDistribution:
        return self.prolong(coarse, coarse_cov=coarse_cov)

    def prolong(self, coarse: Any, *, coarse_cov: Any | None = None) -> FineDistribution:
        """Expand a coarse state into a distribution over fine states."""
        c = torch.as_tensor(coarse, dtype=DTYPE).reshape(-1)
        R = self.restriction.matrix
        if c.numel() != R.shape[0]:
            raise SiteError(
                f"{self.label} expects a coarse state of size {R.shape[0]}, got {c.numel()}",
                remedy="Prolong a state defined on the coarse support.",
                offending_object=self.label,
            )
        mean = self.matrix @ c
        n_fine = R.shape[1]

        # split fine space into what R resolves (row space) and what it does not
        U, S, Vh = torch.linalg.svd(R, full_matrices=True)
        tol = max(R.shape) * float(S.max() if S.numel() else 0.0) * 1e-12
        rank = int((S > tol).sum())
        V = Vh.T
        row_space = V[:, :rank]
        null_space = V[:, rank:]

        cov = (self.prior_sd_unresolved**2) * (null_space @ null_space.T)
        if self.prior_sd_resolved > 0:
            cov = cov + (self.prior_sd_resolved**2) * (row_space @ row_space.T)
        if coarse_cov is not None:
            C = as_tensor(coarse_cov)
            cov = cov + self.matrix @ C @ self.matrix.T
        cov = 0.5 * (cov + cov.T)

        return FineDistribution(
            mean=mean,
            cov=cov,
            resolved_rank=rank,
            unresolved_rank=n_fine - rank,
            unresolved_basis=null_space,
            provenance={
                "prolongation": self.label,
                "restriction_partner": self.restriction.label,
                "coverage": self.coverage.as_record(),
                "prior_sd_unresolved": self.prior_sd_unresolved,
                "note": (
                    "the unresolved subspace carries prior variance, not "
                    "reconstructed structure"
                ),
            },
        )


def measure_coverage(
    restriction: Restriction,
    prolongation_matrix: Any,
    fine_landmarks: Any,
    *,
    domain: Mapping[str, Any] | None = None,
) -> CoverageReport:
    """Held-out landmark + round-trip test that licenses a prolongation.

    * ``coarse_roundtrip_error`` -- ``R P x_coarse`` vs ``x_coarse``.  This one
      *can* be ~0 for a proper pair, and if it is not the pair is broken.
    * ``heldout_error`` -- ``P R x_fine`` vs ``x_fine`` on held-out fine
      landmarks.  This one is generally **not** zero, and pretending otherwise
      is the error R02 exists to catch.
    """
    R = restriction.matrix
    P = as_tensor(prolongation_matrix)
    X = torch.as_tensor(fine_landmarks, dtype=DTYPE)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != R.shape[1]:
        raise SiteError(
            f"landmarks have {X.shape[1]} fine elements, restriction expects {R.shape[1]}",
            remedy="Test on the fine support the restriction is declared for.",
            offending_object=tuple(X.shape),
        )
    coarse = X @ R.T
    back_coarse = coarse @ P.T @ R.T
    rt = float(torch.sqrt(((back_coarse - coarse) ** 2).mean()))
    recon = coarse @ P.T
    err = float(torch.sqrt(((recon - X) ** 2).mean()))
    var = float(X.var(unbiased=False)) + 1e-300
    explained = max(0.0, 1.0 - (err**2) / var)
    return CoverageReport(
        n_landmarks=int(X.shape[0]),
        heldout_error=err,
        coarse_roundtrip_error=rt,
        fraction_of_fine_variance_explained=explained,
        domain=dict(domain or {}),
    )


@dataclass(frozen=True)
class ScalePair:
    """A registered ``(R_i, P_i)`` pair -- §4.2's paired maps, kept together."""

    restriction: Restriction
    prolongation: Prolongation

    def round_trip_report(self, fine_states: Any) -> dict[str, float]:
        """Both directions, reported separately.  Neither is assumed zero."""
        X = torch.as_tensor(fine_states, dtype=DTYPE)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        R, P = self.restriction.matrix, self.prolongation.matrix
        coarse = X @ R.T
        fine_back = coarse @ P.T
        coarse_back = fine_back @ R.T
        return {
            "coarse_fine_coarse_rms": float(torch.sqrt(((coarse_back - coarse) ** 2).mean())),
            "fine_coarse_fine_rms": float(torch.sqrt(((fine_back - X) ** 2).mean())),
            "unresolved_rank": self.prolongation.prolong(coarse[0]).unresolved_rank,
        }


__all__ = [
    "SupportObject",
    "Restriction",
    "Section",
    "Cover",
    "Site",
    "OverlapResidual",
    "CocycleResidual",
    "ObstructionCertificate",
    "GlobalSection",
    "GluingResult",
    "glue",
    "glue_or_raise",
    "CoverageReport",
    "FineDistribution",
    "Prolongation",
    "ScalePair",
    "measure_coverage",
]
