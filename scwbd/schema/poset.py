"""The resolution poset (thesis sec. 2.6).

``R = (calR, <=)`` is a *partial* order, not a ladder.  Elements may be surface
meshes, parcels, laminar depths, voxel fields, tract endpoint distributions,
spectral bands, event windows, behavioral episodes or sparse local tiles, and
**some pairs are simply incomparable** - that is the point of using a poset
rather than a resolution index.  ``leq(a, b)`` here follows the thesis
convention ``r_a <= r_b``: ``a`` is the finer/more-informative support, a
*restriction* maps ``a -> b``, and a *prolongation* maps ``b -> a`` and may
produce a distribution rather than a unique fine state.

Two refusals read off this object:

* **R02** - a prolongation without a declared restriction partner and tested
  coverage.
* **R03** - a request for global cross-scale state when the overlap/cocycle
  residual exceeds tolerance; the failing path is returned as an obstruction
  certificate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import ScaleId
from .ledger import UncertaintyLedger
from .units import Unit

__all__ = [
    "ScaleNode",
    "MapSpec",
    "ScaleMapPair",
    "CocycleCheck",
    "GluingPolicy",
    "ResolutionPoset",
    "ObstructionCertificate",
]


class ScaleNode(SchemaModel):
    """One element of the resolution poset."""

    id: ScaleId
    label: str = ""
    axis: Literal[
        "spatial", "temporal", "anatomical", "representational", "spectral", "behavioral"
    ] = "spatial"
    #: Nominal characteristic size (metres for spatial, seconds for temporal).
    characteristic_scale: float | None = Field(default=None, gt=0.0)
    characteristic_units: Unit = Unit("m")
    n_elements: int | None = Field(default=None, gt=0)
    description: str = ""


class MapSpec(SchemaModel):
    """A calibrated map between two scales. Never automatic resampling."""

    name: str
    kind: Literal[
        "restriction", "prolongation", "projection", "aggregation", "interpolation"
    ]
    #: A prolongation that returns a point estimate where a distribution is
    #: required is exactly the "unsupported fine structure" failure of R02.
    returns_distribution: bool = False
    #: Round-trip residual ``||P(R(x)) - x||`` on the declared validity set.
    roundtrip_residual: float | None = Field(default=None, ge=0.0)
    roundtrip_tolerance: float | None = Field(default=None, gt=0.0)
    units: Unit = Unit("dimensionless")
    ledger: UncertaintyLedger | None = None
    asset_ref: str | None = None

    @property
    def roundtrip_ok(self) -> bool:
        if self.roundtrip_residual is None or self.roundtrip_tolerance is None:
            return False
        return self.roundtrip_residual <= self.roundtrip_tolerance


class ScaleMapPair(SchemaModel):
    """A registered restriction/prolongation pair between ``fine <= coarse``."""

    fine: ScaleId
    coarse: ScaleId
    restriction: MapSpec | None = None
    prolongation: MapSpec | None = None
    #: Held-out landmark/correspondence coverage of the prolongation domain.
    landmark_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    required_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    #: Whether round-trip and held-out landmark tests were actually run.
    roundtrip_tested: bool = False
    landmark_tested: bool = False
    #: What happens to a query outside the tested support.
    out_of_support_policy: (
        Literal["refuse", "return_distribution", "inflate_uncertainty"] | None
    ) = None

    @model_validator(mode="after")
    def _check(self) -> "ScaleMapPair":
        if self.fine == self.coarse:
            raise ValueError(f"scale map pair is degenerate at {self.fine}")
        if self.restriction is not None and self.restriction.kind not in (
            "restriction",
            "aggregation",
            "projection",
        ):
            raise ValueError(
                f"restriction slot of {self.fine}->{self.coarse} holds a "
                f"{self.restriction.kind!r} map"
            )
        if self.prolongation is not None and self.prolongation.kind not in (
            "prolongation",
            "interpolation",
        ):
            raise ValueError(
                f"prolongation slot of {self.coarse}->{self.fine} holds a "
                f"{self.prolongation.kind!r} map"
            )
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.fine), str(self.coarse))

    def coverage_tested(self) -> bool:
        return (
            self.roundtrip_tested
            and self.landmark_tested
            and self.landmark_coverage is not None
            and self.landmark_coverage >= self.required_coverage
        )


class CocycleCheck(SchemaModel):
    """Measured commutation residual along an alternative path.

    ``Omega_abc(x) = || rho^{Ua}_{Uc} x - rho^{Ub}_{Uc} rho^{Ua}_{Ub} x ||_Wc
    <= tau_abc`` (thesis sec. 2.6).
    """

    path: tuple[ScaleId, ScaleId, ScaleId]
    residual: float = Field(ge=0.0)
    tolerance: float = Field(gt=0.0)
    units: Unit = Unit("dimensionless")
    n_samples: int = Field(default=0, ge=0)

    @property
    def passes(self) -> bool:
        return self.residual <= self.tolerance


class ObstructionCertificate(SchemaModel):
    """Emitted when gluing fails: which path failed, by how much."""

    failed_paths: tuple[tuple[str, str, str], ...]
    residuals: tuple[float, ...]
    tolerances: tuple[float, ...]
    remedy: str = (
        "Preserve separate sections, branch or marginalize the posterior, and "
        "emit an obstruction certificate identifying the failed path"
    )

    @property
    def is_obstructed(self) -> bool:
        return bool(self.failed_paths)


class GluingPolicy(SchemaModel):
    """Whether and how a global cross-scale section may be materialized."""

    #: Request a single global section across scales.  When True the compiler
    #: enforces R03 against ``cocycle_checks``.
    materialize_global: bool = False
    cocycle_checks: tuple[CocycleCheck, ...] = ()
    #: Behaviour when the cocycle test fails.
    on_failure: Literal["preserve_sections", "branch_posterior", "marginalize", "refuse"] = (
        "preserve_sections"
    )

    def failures(self) -> tuple[CocycleCheck, ...]:
        return tuple(c for c in self.cocycle_checks if not c.passes)

    def certificate(self) -> ObstructionCertificate:
        bad = self.failures()
        return ObstructionCertificate(
            failed_paths=tuple(tuple(str(s) for s in c.path) for c in bad),  # type: ignore[arg-type]
            residuals=tuple(c.residual for c in bad),
            tolerances=tuple(c.tolerance for c in bad),
        )


class ResolutionPoset(SchemaModel):
    """A partially ordered set of supports with registered maps between them."""

    nodes: tuple[ScaleNode, ...] = ()
    #: Covering relations ``(fine, coarse)`` meaning ``fine <= coarse``.
    relations: tuple[tuple[ScaleId, ScaleId], ...] = ()
    maps: tuple[ScaleMapPair, ...] = ()
    gluing: GluingPolicy = Field(default_factory=GluingPolicy)

    @model_validator(mode="after")
    def _check(self) -> "ResolutionPoset":
        ids = [str(n.id) for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate scale nodes: {sorted(dupes)}")
        known = set(ids)
        for a, b in self.relations:
            if str(a) not in known or str(b) not in known:
                raise ValueError(f"relation ({a}, {b}) references an undeclared scale")
            if str(a) == str(b):
                raise ValueError(f"relation ({a}, {b}) is reflexive; that is implicit")
        for pair in self.maps:
            if str(pair.fine) not in known or str(pair.coarse) not in known:
                raise ValueError(
                    f"map pair {pair.key} references an undeclared scale"
                )
        # Antisymmetry: the covering relation must be acyclic, else <= is not a
        # partial order at all.
        closure = _transitive_closure(known, self.relations)
        for a in known:
            for b in closure[a]:
                if a != b and a in closure.get(b, set()):
                    raise ValueError(
                        f"resolution relation is cyclic: {a} <= {b} <= {a}; "
                        "a poset requires antisymmetry"
                    )
        return self

    # -- order ------------------------------------------------------------
    def _closure(self) -> dict[str, set[str]]:
        return _transitive_closure({str(n.id) for n in self.nodes}, self.relations)

    def ids(self) -> tuple[str, ...]:
        return tuple(str(n.id) for n in self.nodes)

    def has(self, scale: str) -> bool:
        return str(scale) in {str(n.id) for n in self.nodes}

    def node(self, scale: str) -> ScaleNode:
        for n in self.nodes:
            if str(n.id) == str(scale):
                return n
        raise KeyError(f"unknown scale {scale!r}")

    def leq(self, a: str, b: str) -> bool:
        """``a <= b``: a is the finer support, restriction maps a -> b."""
        a, b = str(a), str(b)
        if not self.has(a) or not self.has(b):
            raise KeyError(f"unknown scale in leq({a!r}, {b!r})")
        if a == b:
            return True  # reflexivity
        return b in self._closure()[a]

    def lt(self, a: str, b: str) -> bool:
        return a != b and self.leq(a, b)

    def comparable(self, a: str, b: str) -> bool:
        return self.leq(a, b) or self.leq(b, a)

    def incomparable(self, a: str, b: str) -> bool:
        """True when no defensible order - and therefore no map - exists."""
        return not self.comparable(a, b)

    def incomparable_pairs(self) -> tuple[tuple[str, str], ...]:
        out: list[tuple[str, str]] = []
        ids = self.ids()
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if self.incomparable(a, b):
                    out.append((a, b))
        return tuple(out)

    def upper_bounds(self, a: str) -> tuple[str, ...]:
        return tuple(sorted(self._closure()[str(a)]))

    def lower_bounds(self, a: str) -> tuple[str, ...]:
        a = str(a)
        closure = self._closure()
        return tuple(sorted(x for x, ups in closure.items() if a in ups))

    # -- maps -------------------------------------------------------------
    def map_pair(self, fine: str, coarse: str) -> ScaleMapPair | None:
        for pair in self.maps:
            if pair.key == (str(fine), str(coarse)):
                return pair
        return None

    def prolongations(self) -> tuple[ScaleMapPair, ...]:
        return tuple(p for p in self.maps if p.prolongation is not None)

    def orphan_prolongations(self) -> tuple[ScaleMapPair, ...]:
        """Prolongations lacking a restriction partner or tested coverage (R02)."""
        return tuple(
            p
            for p in self.prolongations()
            if p.restriction is None
            or not p.coverage_tested()
            or p.out_of_support_policy is None
        )

    def unordered_maps(self) -> tuple[ScaleMapPair, ...]:
        """Maps declared between scales that are not actually comparable."""
        return tuple(p for p in self.maps if not self.leq(str(p.fine), str(p.coarse)))


def _transitive_closure(
    nodes: set[str], relations: tuple[tuple[str, str], ...] | tuple
) -> dict[str, set[str]]:
    """Reachability over the covering relation (Floyd-Warshall style)."""
    reach: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in relations:
        reach.setdefault(str(a), set()).add(str(b))
    changed = True
    while changed:
        changed = False
        for a in list(reach):
            for b in list(reach[a]):
                for c in reach.get(b, ()):  # a <= b <= c
                    if c not in reach[a]:
                        reach[a].add(c)
                        changed = True
    return reach
