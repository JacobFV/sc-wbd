"""The compiled uncertainty ledger.

An aggregate view over every object's ledger.  It deliberately does **not**
produce a single score: "bias and variance NEVER collapse into one score"
(thesis sec. 2.7).  What it does produce is a per-object index, per-component
variance totals, and the census of bias statuses that a claim report must show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.ledger import VARIANCE_COMPONENTS, UncertaintyLedger
from ..schema.schema import BrainSchema

__all__ = ["CompiledLedger", "build_ledger"]


@dataclass(frozen=True)
class CompiledLedger:
    """Per-object uncertainty ledgers plus separate aggregate views."""

    by_object: dict[str, UncertaintyLedger]
    #: variance component -> sum over objects (never combined with bias).
    variance_totals: dict[str, float]
    #: bias status -> count of objects.
    bias_status_counts: dict[str, int]
    #: Objects whose bias is a point estimate lacking an estimator or bound.
    unbacked_bias: tuple[str, ...] = ()
    #: Objects declaring a model discrepancy term.
    model_discrepancy: dict[str, float] = field(default_factory=dict)
    #: Variance keys outside the declared taxonomy.
    unknown_components: tuple[str, ...] = ()

    def __getitem__(self, path: str) -> UncertaintyLedger:
        return self.by_object[path]

    def __len__(self) -> int:
        return len(self.by_object)

    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_object))

    def widest_bias(self) -> tuple[str, float] | None:
        if not self.by_object:
            return None
        path = max(self.by_object, key=lambda p: self.by_object[p].bias_width)
        return (path, self.by_object[path].bias_width)

    def sensitivity_terms(self) -> tuple[str, ...]:
        """Objects whose bias must be swept, not advertised as estimated."""
        return tuple(
            sorted(
                p
                for p, l in self.by_object.items()
                if l.bias_status == "prior_specified_sensitivity"
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "n_objects": len(self.by_object),
            "variance_totals": dict(sorted(self.variance_totals.items())),
            "bias_status_counts": dict(sorted(self.bias_status_counts.items())),
            "unbacked_bias": list(self.unbacked_bias),
            "model_discrepancy": dict(sorted(self.model_discrepancy.items())),
            "unknown_components": list(self.unknown_components),
        }

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.bias_status_counts.items()))
        return f"CompiledLedger({len(self.by_object)} objects; {counts})"


def build_ledger(schema: BrainSchema) -> CompiledLedger:
    by_object: dict[str, UncertaintyLedger] = {}
    for path, obj in schema.all_ledgers():
        assert isinstance(obj, UncertaintyLedger)
        by_object[path] = obj

    variance_totals: dict[str, float] = {k: 0.0 for k in VARIANCE_COMPONENTS}
    status_counts: dict[str, int] = {}
    unbacked: list[str] = []
    discrepancy: dict[str, float] = {}
    unknown: set[str] = set()

    for path, l in by_object.items():
        for key, value in l.variance.items():
            variance_totals[key] = variance_totals.get(key, 0.0) + float(value)
        unknown.update(l.unknown_variance_components())
        status_counts[l.bias_status] = status_counts.get(l.bias_status, 0) + 1
        if not l.has_estimator():
            unbacked.append(path)
        if l.model_discrepancy is not None:
            discrepancy[path] = float(l.model_discrepancy)

    return CompiledLedger(
        by_object=by_object,
        variance_totals=variance_totals,
        bias_status_counts=status_counts,
        unbacked_bias=tuple(sorted(unbacked)),
        model_discrepancy=discrepancy,
        unknown_components=tuple(sorted(unknown)),
    )
