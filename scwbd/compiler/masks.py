"""Gradient masks: ``A_k`` compiled into an executable permission (Appendix B).

Rule 2 of ARCHITECTURE.md sec. 7: "a source updates only the modules its
``GradientPermission`` names."  The compiler enumerates every named parameter
group in the schema and, per source card, emits a boolean mask over them plus
an audit record.

Permission entries are matched with shell globs so a schema can write
``"operator:*"`` or ``"region:V1:*"`` instead of enumerating hundreds of names;
an entry that matches nothing is reported rather than silently ignored, because
a typo in a permission is a silent grant of nothing (or, worse, evidence that
the author believed something was being trained that is not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import torch

from ..schema.schema import BrainSchema
from ..schema.sources import SourceCard

__all__ = ["GradientMask", "GradientMasks", "build_gradient_masks", "parameter_groups_of"]


def parameter_groups_of(schema: BrainSchema) -> tuple[str, ...]:
    """Every named, trainable parameter group in the compiled model.

    Names are structured ``kind:object[:slot]`` so glob permissions are useful.
    """
    names: list[str] = []
    for region in schema.regions:
        for comp_name in region.state.names:
            names.append(f"region:{region.id}:state:{comp_name}")
        for port in region.ports:
            names.append(f"port:{region.id}.{port.name}")
    for op in schema.operators:
        names.append(f"operator:{op.key}:params")
        names.append(f"operator:{op.key}:delay")
        if op.residual is not None:
            names.append(f"operator:{op.key}:residual")
    for pair in schema.resolution_poset.maps:
        for slot in ("restriction", "prolongation"):
            if getattr(pair, slot) is not None:
                names.append(f"scale_map:{pair.fine}->{pair.coarse}:{slot}")
    for edge in schema.frames.edges:
        if edge.parameters:
            names.append(f"frame_edge:{edge.src}->{edge.dst}:calibration")
    for source in schema.sources:
        if source.observation is not None:
            names.append(f"observation:{source.id}:nuisance")
        if source.intervention is not None:
            names.append(f"intervention:{source.id}:nuisance")
    return tuple(sorted(set(names)))


@dataclass(frozen=True)
class GradientMask:
    """Boolean permission over named parameter groups for one source card."""

    source_id: str
    role: str
    group_names: tuple[str, ...]
    allowed: tuple[bool, ...]
    #: Permission entries that matched no parameter group.
    unmatched_patterns: tuple[str, ...] = ()
    #: Groups explicitly frozen by this source.
    frozen: tuple[str, ...] = ()
    evaluation_only: bool = False
    _index: dict[str, int] = field(repr=False, default_factory=dict)

    def allows(self, group: str) -> bool:
        idx = self._index.get(group)
        if idx is None:
            raise KeyError(f"unknown parameter group {group!r}")
        return self.allowed[idx]

    def allowed_groups(self) -> tuple[str, ...]:
        return tuple(n for n, a in zip(self.group_names, self.allowed) if a)

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.allowed, dtype=torch.bool)

    def n_allowed(self) -> int:
        return sum(self.allowed)

    def summary(self) -> str:
        return (
            f"GradientMask({self.source_id}, role={self.role}, "
            f"{self.n_allowed()}/{len(self.group_names)} groups)"
        )


@dataclass(frozen=True)
class GradientMasks:
    """All per-source masks plus the group ordering they share."""

    group_names: tuple[str, ...]
    masks: dict[str, GradientMask]

    def __getitem__(self, source_id: str) -> GradientMask:
        return self.masks[source_id]

    def __contains__(self, source_id: object) -> bool:
        return source_id in self.masks

    def __len__(self) -> int:
        return len(self.masks)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.masks))

    def matrix(self) -> torch.Tensor:
        """[n_sources, n_groups] bool matrix in sorted source order."""
        if not self.masks:
            return torch.zeros((0, len(self.group_names)), dtype=torch.bool)
        return torch.stack([self.masks[k].as_tensor() for k in self.keys()])

    def sources_updating(self, group: str) -> tuple[str, ...]:
        return tuple(k for k in self.keys() if self.masks[k].allows(group))

    def unreachable_groups(self) -> tuple[str, ...]:
        """Groups no source may update - they will never train."""
        return tuple(g for g in self.group_names if not self.sources_updating(g))

    def summary(self) -> str:
        return (
            f"GradientMasks({len(self.masks)} sources, "
            f"{len(self.group_names)} parameter groups)"
        )


def _matches(patterns: tuple[str, ...], group: str) -> bool:
    return any(fnmatchcase(group, p) or group == p for p in patterns)


def build_mask(source: SourceCard, group_names: tuple[str, ...]) -> GradientMask:
    perm = source.gradient_permission
    # Every declared target space is a grant surface; scales are matched
    # against the ``scale_map:`` and ``region:...:state:`` namespaces.
    grants = tuple(perm.modules) + tuple(perm.parameter_groups)
    port_grants = tuple(f"port:{p}" if not p.startswith("port:") else p for p in perm.ports)
    scale_grants = tuple(f"*{s}*" for s in perm.scales)
    frozen_patterns = tuple(perm.frozen)

    allowed: list[bool] = []
    matched: set[str] = set()
    for group in group_names:
        granted = False
        for pattern in grants + port_grants + scale_grants:
            if fnmatchcase(group, pattern) or group == pattern:
                granted = True
                matched.add(pattern)
        if _matches(frozen_patterns, group):
            granted = False
        if perm.evaluation_only or source.role in ("evaluation_only", "negative_control"):
            granted = False
        allowed.append(granted)

    unmatched = tuple(
        sorted(p for p in grants + port_grants + scale_grants if p not in matched)
    )
    frozen_groups = tuple(g for g in group_names if _matches(frozen_patterns, g))
    return GradientMask(
        source_id=source.id,
        role=source.role,
        group_names=group_names,
        allowed=tuple(allowed),
        unmatched_patterns=unmatched,
        frozen=frozen_groups,
        evaluation_only=perm.evaluation_only or source.role == "evaluation_only",
        _index={n: i for i, n in enumerate(group_names)},
    )


def build_gradient_masks(schema: BrainSchema) -> GradientMasks:
    groups = parameter_groups_of(schema)
    return GradientMasks(
        group_names=groups,
        masks={s.id: build_mask(s, groups) for s in schema.sources},
    )
