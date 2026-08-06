"""Block-sparse adjacency, typed by evidence class (thesis sec. 2.2).

Pathways compile into three classes - ``hard``, ``soft``, ``proposed`` - and the
masks are kept **separate**, never summed into one weight matrix.  A zero entry
creates exact independence across that direct edge inside the compiled model;
it does not establish biological independence.

Proposed edges carry a complexity/distance/provenance penalty so that model
comparison can charge for them explicitly rather than letting them in free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..schema.operators import OperatorSpec
from ..schema.schema import BrainSchema

__all__ = ["Adjacency", "build_adjacency", "EVIDENCE_CLASSES"]

EVIDENCE_CLASSES: tuple[str, ...] = ("hard", "soft", "proposed")


@dataclass(frozen=True)
class Adjacency:
    """Per-evidence-class region adjacency plus per-edge block indices."""

    region_ids: tuple[str, ...]
    #: evidence class -> [2, n_edges] int64 index tensor (src_idx, dst_idx).
    indices: dict[str, torch.Tensor]
    #: evidence class -> sparse COO bool mask of shape [N, N].
    masks: dict[str, torch.Tensor]
    #: evidence class -> operator keys, aligned with ``indices`` columns.
    edge_keys: dict[str, tuple[str, ...]]
    #: operator key -> penalty charged in model comparison (proposed edges).
    penalties: dict[str, float] = field(default_factory=dict)

    # -- accessors ---------------------------------------------------------
    def index_of(self, region: str) -> int:
        return self.region_ids.index(region)

    def n_regions(self) -> int:
        return len(self.region_ids)

    def dense(self, evidence_class: str) -> torch.Tensor:
        """Dense bool mask [N, N] for one evidence class."""
        return self.masks[evidence_class].to_dense()

    def combined_dense(self, classes: tuple[str, ...] = EVIDENCE_CLASSES) -> torch.Tensor:
        """Union of the requested classes.

        Provided for convenience only.  Anything that *learns* from adjacency
        must use the per-class masks: merging hard anatomical support with
        proposed edges is precisely the confusion sec. 2.2 warns about.
        """
        n = self.n_regions()
        out = torch.zeros((n, n), dtype=torch.bool)
        for c in classes:
            out |= self.dense(c)
        return out

    def n_edges(self, evidence_class: str) -> int:
        return int(self.indices[evidence_class].shape[1])

    def total_edges(self) -> int:
        return sum(self.n_edges(c) for c in EVIDENCE_CLASSES)

    def density(self) -> dict[str, float]:
        n = max(self.n_regions(), 1)
        return {c: self.n_edges(c) / (n * n) for c in EVIDENCE_CLASSES}

    def total_penalty(self) -> float:
        return float(sum(self.penalties.values()))

    def summary(self) -> str:
        counts = ", ".join(f"{c}={self.n_edges(c)}" for c in EVIDENCE_CLASSES)
        return f"Adjacency({self.n_regions()} regions, {counts})"


def _penalty(op: OperatorSpec) -> float:
    """Complexity + distance + provenance penalty for a proposed edge.

    The penalty is a declared bookkeeping quantity, not a tuned hyperparameter:
    one unit per free parameter (complexity), one per 10 cm of separation
    (distance), and one when identification rests on passive correlation only
    (provenance).  Agent H's model comparison decides what to do with it.
    """
    complexity = float(len(op.params) + 1)
    distance = 0.0 if op.distance_m is None else float(op.distance_m) / 0.10
    provenance = 1.0 if op.identification.is_passive_only else 0.0
    return complexity + distance + provenance


def build_adjacency(schema: BrainSchema) -> Adjacency:
    region_ids = schema.region_ids()
    index = {rid: i for i, rid in enumerate(region_ids)}
    n = len(region_ids)

    rows: dict[str, list[tuple[int, int]]] = {c: [] for c in EVIDENCE_CLASSES}
    keys: dict[str, list[str]] = {c: [] for c in EVIDENCE_CLASSES}
    penalties: dict[str, float] = {}

    for op in schema.operators:
        cls = op.evidence_class
        rows[cls].append((index[op.src], index[op.dst]))
        keys[cls].append(op.key)
        if cls == "proposed":
            penalties[op.key] = _penalty(op)

    indices: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for cls in EVIDENCE_CLASSES:
        pairs = rows[cls]
        if pairs:
            idx = torch.tensor(
                [[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.int64
            )
        else:
            idx = torch.zeros((2, 0), dtype=torch.int64)
        indices[cls] = idx
        # Indices are constructed here, not user-supplied, so the global
        # invariant check is redundant work on every compile.
        with torch.sparse.check_sparse_tensor_invariants(False):
            masks[cls] = torch.sparse_coo_tensor(
                idx,
                torch.ones(idx.shape[1], dtype=torch.bool),
                size=(n, n),
            ).coalesce()

    return Adjacency(
        region_ids=region_ids,
        indices=indices,
        masks=masks,
        edge_keys={c: tuple(keys[c]) for c in EVIDENCE_CLASSES},
        penalties=penalties,
    )
