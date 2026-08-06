"""Per-family operators, ports and residual — ``F_local`` at family granularity.

``scwbd/foundation/model.py`` resolves **one** ``local_core`` string and applies
that operator to all regions (``reports/scope_gap.md`` G-1).  This module is the
per-family replacement: every family gets the operator its own
:class:`~scwbd.foundation.families.RegionFamily` declares, over its own
components, and messages cross family boundaries through **declared ports**
rather than raw state slices.

Three things are deliberate:

* **Capacity is not inflated.**  Learned families that share a state dimension
  share one :class:`~scwbd.foundation.model.RegionalOperator`; per-parcel
  identity already enters through FiLM.  A family arm with 11 copies of the
  trunk would not be comparable to the equal-capacity generic control that
  body.tex §11.4 requires, and the comparison is the point.
* **Nothing writes outside a span.**  Each family emits exactly ``d_f`` channels
  and they are assembled by concatenation in family order, so the pad is
  *structurally* zero.  ``FamilyStateLayout.assert_clean`` then has something
  real to check, and it fires the moment a caller substitutes a full-``D``
  operator — see ``tests/foundation/test_family_state.py``.
* **The engineered backends are actually run.**  ``thalamic_relay``,
  ``basal_ganglia_gate``, ``hippocampal_code`` and ``cerebellar_forward_model``
  supply the drift for their families; they are not decoration.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .families import FamilyStateLayout, RegionFamily, SpanViolation
from .uncertainty import UNCERTAINTY_COMPONENT, UncertaintyPropagator

__all__ = ["FamilyPorts", "FamilyLocalOperator", "FamilyResidual", "FamilyReadout", "MechanisticFamilyCore"]


def _pad_to(x: Tensor, dim: int) -> Tensor:
    """Zero-pad the last axis to ``dim``.  The pad is exactly zero, by construction."""
    extra = dim - x.shape[-1]
    if extra < 0:
        raise SpanViolation(f"value is {x.shape[-1]} channels wide but the padded layout is {dim}")
    if extra == 0:
        return x
    return torch.cat([x, x.new_zeros(*x.shape[:-1], extra)], dim=-1)


# ======================================================================
# ports
# ======================================================================
class FamilyPorts(nn.Module):
    """Message out through declared out-ports; message in through in-ports.

    ``body.tex`` §2.1: a cross-region operator "maps a declared source boundary
    into a compatible destination port".  The projection is per family because
    the boundary is: a hippocampal family exports ``(v, rho)``, a basal-ganglia
    family exports a disinhibition ``gate``, and a cortical family exports its
    spectral quadrature — these are different objects and a single shared
    ``msg_proj`` over a common slice cannot express that.
    """

    def __init__(self, flayout: FamilyStateLayout, *, message_dim: int, in_dim: int) -> None:
        super().__init__()
        self.flayout = flayout
        self.message_dim = int(message_dim)
        self.out_proj = nn.ModuleDict()
        self.in_proj = nn.ModuleDict()
        self._out_ports: dict[str, tuple[str, ...]] = {}
        for f in flayout:
            names = tuple(p.name for p in f.out_ports())
            width = sum(f.port_dim(n) for n in names)
            self._out_ports[f.name] = names
            self.out_proj[f.name] = nn.Linear(width, message_dim)
            self.in_proj[f.name] = nn.Sequential(nn.Linear(message_dim, in_dim), nn.GELU())

    def out_width(self, family: str) -> int:
        f = self.flayout.family(family)
        return sum(f.port_dim(n) for n in self._out_ports[family])

    def message(self, x: Tensor) -> Tensor:
        """``(..., N, D) -> (..., N, message_dim)`` built only from out-ports."""
        chunks = []
        for f in self.flayout:
            src = torch.cat([self.flayout.port(x, f.name, p) for p in self._out_ports[f.name]], dim=-1)
            chunks.append(self.out_proj[f.name](src))
        return self.flayout.assemble(chunks)

    def readin(self, family: str, coupling: Tensor) -> Tensor:
        """Project the arriving message into the family's in-port feature space."""
        return self.in_proj[family](coupling)


# ======================================================================
# mechanistic family core
# ======================================================================
class MechanisticFamilyCore(nn.Module):
    """One family's engineered backend, evaluated in fp32 over its own components.

    The backend's native state lives in the family's declared
    ``backend_components``; the shared ``rate_e``/``rate_i`` interface is pulled
    toward the backend's own observables so the observation heads see one
    quantity regardless of which mechanism produced it.
    """

    def __init__(self, family: RegionFamily, *, message_dim: int, relax_hz: float = 20.0) -> None:
        super().__init__()
        from .backends import resolve_backend

        self.family_name = family.name
        self.backend_name = family.backend
        self.backend = resolve_backend(family.backend)
        family.check_backend(self.backend)
        self.family = family
        self.relax_hz = float(relax_hz)
        self._slices = family.backend_slices()
        self._dims = [family.layout.spec(c).dim for c in family.backend_components]
        # in-port projection: the shared message space -> this backend's coupling
        # channels.  A backend that consumes two channels (hippocampal cue and
        # context; cerebellar mossy and climbing fibre) says so in
        # ``n_coupling_channels`` and gets two.
        self.coupling_in = nn.Linear(message_dim, int(self.backend.n_coupling_channels))
        nn.init.zeros_(self.coupling_in.bias)

    def forward(self, xf: Tensor, coupling: Tensor, pack) -> Tensor:
        """``xf (B, n_f, d_f)``, ``coupling (B, n_f, message_dim)`` -> ``(B, n_f, d_f)``."""
        m = torch.cat([xf[..., s] for s in self._slices], dim=-1).float()
        c = self.coupling_in(coupling).float()
        d = self.backend.drift(m, c, pack)
        out = torch.zeros_like(xf, dtype=torch.float32)
        off = 0
        for s, w in zip(self._slices, self._dims):
            out[..., s] = d[..., off : off + w]
            off += w
        obs = self.backend.observables(m)
        lay = self.family.layout
        for name in ("rate_e", "rate_i"):
            if name in lay and name in obs:
                sl = lay.slice(name)
                out[..., sl] = (obs[name].unsqueeze(-1) - xf[..., sl].float()) * self.relax_hz
        return out.to(xf.dtype)


# ======================================================================
# the local operator
# ======================================================================
class FamilyLocalOperator(nn.Module):
    """``F_local`` dispatched per family; assembled without touching any pad."""

    def __init__(
        self,
        flayout: FamilyStateLayout,
        cfg: ModelConfig,
        *,
        in_extra: int,
        message_dim: int,
    ) -> None:
        super().__init__()
        from .model import RegionalOperator

        self.flayout = flayout
        self.cfg = cfg
        self.dim = flayout.dim
        self.ports = FamilyPorts(flayout, message_dim=message_dim, in_dim=in_extra)

        # -- learned families, grouped by state dimension ------------------
        self._learned_group: dict[str, str] = {}
        groups: dict[int, list[RegionFamily]] = {}
        self.mech = nn.ModuleDict()
        for f in flayout:
            if f.backend == "learned":
                groups.setdefault(f.dim, []).append(f)
            else:
                self.mech[f.name] = MechanisticFamilyCore(f, message_dim=message_dim)
        self.learned = nn.ModuleDict()
        self._group_members: dict[str, tuple[str, ...]] = {}
        for d, fams in sorted(groups.items()):
            key = f"d{d}"
            n = sum(f.n_regions for f in fams)
            self.learned[key] = RegionalOperator(fams[0].layout, n, cfg, in_extra=in_extra)
            self._group_members[key] = tuple(f.name for f in fams)
            for f in fams:
                self._learned_group[f.name] = key
        # per-region log-dt within each learned group (the old ``log_dt_scale``)
        self.log_dt = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(sum(flayout.family(n).n_regions for n in v)))
             for k, v in self._group_members.items()}
        )

        # -- X_i^uncertainty, per family (body.tex §2.1) --------------------
        # One propagator per family, because how uncertainty accumulates in a
        # thalamic relay is not how it accumulates in a cortical column. It
        # OVERWRITES whatever the family's core wrote into the uncertainty
        # channel, so that channel has one law rather than being an incidental
        # output of a generic operator (learned families) or untouched for the
        # whole rollout (mechanistic families, whose backends emit nothing there).
        self.uncertainty = nn.ModuleDict()
        self._unc_slice: dict[str, slice] = {}
        if cfg.state_dependent_variance:
            for f in flayout:
                if UNCERTAINTY_COMPONENT not in f.layout:
                    raise SpanViolation(
                        f"family {f.name!r} declares no {UNCERTAINTY_COMPONENT!r} component; "
                        "body.tex §2.1 names X_i^uncertainty and the predictive variance is "
                        "sourced from it"
                    )
                self._unc_slice[f.name] = f.layout.slice(UNCERTAINTY_COMPONENT)
                self.uncertainty[f.name] = UncertaintyPropagator(
                    f.dim,
                    f.layout.spec(UNCERTAINTY_COMPONENT).dim,
                    in_extra=in_extra,
                    hidden=max(cfg.hidden // 4, 32),
                    dt=cfg.dt_model,
                )

    # -- context -----------------------------------------------------------
    def prepare(self, ctx: Tensor, dtype) -> dict[str, list]:
        return {k: op.prepare(ctx, dtype) for k, op in self.learned.items()}

    def group_regions(self, key: str) -> Tensor:
        return torch.cat([self.flayout.index(n) for n in self._group_members[key]])

    # -- one step ----------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        coupling: Tensor,
        films: Mapping[str, list],
        packs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """``x (B, N, D)``, ``coupling (B, N, message_dim)`` -> ``dx (B, N, D)``.

        The result is assembled by concatenating per-family blocks in family
        order, so out-of-span channels are zero *because nothing produced them*,
        not because they were masked afterwards.  That distinction is what makes
        :meth:`FamilyStateLayout.assert_clean` a live guard rather than a
        tautology.
        """
        packs = packs or {}
        out: dict[str, Tensor] = {}
        feat_by_family: dict[str, Tensor] = {}
        # learned groups: one operator call per (dim) group, not per family
        for key, op in self.learned.items():
            idx = self.group_regions(key).to(x.device)
            d = int(key[1:])
            xg = x.index_select(-2, idx)[..., :d]
            cg = coupling.index_select(-2, idx)
            # the in-port projection is per family; a group may hold several
            feats = []
            offset = 0
            for name in self._group_members[key]:
                n = self.flayout.family(name).n_regions
                fe = self.ports.readin(name, cg[..., offset : offset + n, :])
                feat_by_family[name] = fe
                feats.append(fe)
                offset += n
            extra = torch.cat(feats, dim=-2)
            scale = torch.sigmoid(self.log_dt[key]).to(x.dtype).reshape(1, -1, 1) * 2.0
            dg = op(xg, extra, films[key]) * scale
            offset = 0
            for name in self._group_members[key]:
                n = self.flayout.family(name).n_regions
                out[name] = dg[..., offset : offset + n, :]
                offset += n
        # mechanistic families: their own backend, fp32, scaled to dt in the caller
        for name, core in self.mech.items():
            idx = self.flayout.index(name).to(x.device)
            f = self.flayout.family(name)
            xf = x.index_select(-2, idx)[..., : f.dim]
            cf = coupling.index_select(-2, idx)
            pack = packs.get(name)
            if pack is None:
                raise SpanViolation(
                    f"family {name!r} has mechanistic backend {core.backend_name!r} but no ParamPack "
                    "was bound; call SCWBD.set_mechanistic_theta before rolling out. Running it on "
                    "backend defaults would silently drop the anatomical conditioning."
                )
            out[name] = core(xf, cf, pack) * self.cfg.dt_model
            feat_by_family[name] = self.ports.readin(name, cf)
        # X_i^uncertainty: one law, overwriting whatever the core wrote there.
        for name, prop in self.uncertainty.items():
            idx = self.flayout.index(name).to(x.device)
            f = self.flayout.family(name)
            xf = x.index_select(-2, idx)[..., : f.dim]
            sl = self._unc_slice[name]
            du = prop(xf, xf[..., sl], feat_by_family[name])
            d = out[name]
            out[name] = torch.cat([d[..., : sl.start], du.to(d.dtype), d[..., sl.stop :]], dim=-1)
        return self.flayout.assemble([_pad_to(out[f.name], self.dim) for f in self.flayout])

    def describe(self) -> dict[str, Any]:
        return {
            "learned_groups": {k: list(v) for k, v in self._group_members.items()},
            "mechanistic_families": {n: c.backend_name for n, c in self.mech.items()},
            "message_dim": self.ports.message_dim,
        }


# ======================================================================
# residual and readout
# ======================================================================
class FamilyResidual(nn.Module):
    """``R_theta`` at family width.  Emits ``d_f`` channels, never ``D``."""

    def __init__(self, flayout: FamilyStateLayout, cfg: ModelConfig, *, in_extra: int) -> None:
        super().__init__()
        self.flayout = flayout
        self.dim = flayout.dim
        H = max(cfg.hidden // 2, 64)
        self.nets = nn.ModuleDict()
        self.embed = nn.ParameterDict()
        self._dims: dict[str, int] = {}
        for f in flayout:
            key = f"d{f.dim}"
            self._dims[f.name] = f.dim
            if key in self.nets:
                continue
            self.nets[key] = nn.Sequential(
                nn.Linear(f.dim + cfg.region_embed + in_extra, H), nn.GELU(),
                nn.Linear(H, H), nn.GELU(), nn.Linear(H, f.dim),
            )
            nn.init.zeros_(self.nets[key][-1].weight)
            nn.init.zeros_(self.nets[key][-1].bias)
        self.region_embed = nn.Parameter(torch.randn(flayout.n_regions, cfg.region_embed) * 0.02)
        import math

        self.log_scale = nn.Parameter(torch.tensor(math.log(max(cfg.residual_init_scale, 1e-6))))

    def forward(self, x: Tensor, extra: Tensor) -> Tensor:
        chunks = []
        s = self.log_scale.exp().to(x.dtype)
        for f in self.flayout:
            idx = self.flayout.index(f.name).to(x.device)
            xf = x.index_select(-2, idx)[..., : f.dim]
            ef = self.region_embed.index_select(0, idx).to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
            exf = extra.index_select(-2, idx)
            r = self.nets[f"d{f.dim}"](torch.cat([xf, ef, exf], dim=-1)) * s
            chunks.append(_pad_to(r, self.dim))
        return self.flayout.assemble(chunks)


class FamilyReadout(nn.Module):
    """Predicted regional activity (mean, log-variance) from each family's out-ports."""

    def __init__(self, flayout: FamilyStateLayout, cfg: ModelConfig) -> None:
        super().__init__()
        self.flayout = flayout
        self.heads = nn.ModuleDict()
        self._ports: dict[str, tuple[str, ...]] = {}
        for f in flayout:
            names = tuple(p.name for p in f.out_ports())
            width = sum(f.port_dim(n) for n in names)
            self._ports[f.name] = names
            self.heads[f.name] = nn.Sequential(
                nn.Linear(width, cfg.hidden // 2), nn.GELU(), nn.Linear(cfg.hidden // 2, 2)
            )

    def forward(self, x: Tensor) -> Tensor:
        chunks = []
        for f in self.flayout:
            src = torch.cat([self.flayout.port(x, f.name, p) for p in self._ports[f.name]], dim=-1)
            chunks.append(self.heads[f.name](src))
        return self.flayout.assemble(chunks)
