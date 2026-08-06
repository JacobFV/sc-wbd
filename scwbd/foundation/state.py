"""Structured per-parcel state layout for SC-WBD-001-beta.

ARCHITECTURE.md §5: "per-parcel structured state over ``N_regions`` ... each with
E/I rates, adaptation, spectral modes, hemodynamic compartments, uncertainty
channel."  A single scalar per region is the *named failure mode* of the thesis
(§2.1) and is retained here only as an explicit ablation baseline
(:func:`scalar_layout`), never as the default.

The layout is a compiled memory map (thesis §2.3): a packed offset table over
named components, each with its own dimension, units and clock.  Nothing in the
network is allowed to index the state by a bare integer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor

__all__ = ["ComponentSpec", "StateLayout", "default_layout", "scalar_layout"]


@dataclass(frozen=True)
class ComponentSpec:
    """One named block of a region's private state ``X_i``."""

    name: str
    dim: int
    units: str
    clock: str  # "fast" (neural) | "slow" (hemodynamic) | "meta"
    #: whether long-range messages may read this component (typed ports, §6.3)
    exported: bool = True
    #: whether the stochastic differential noise acts on this component
    stochastic: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"component {self.name!r} must have dim > 0")
        if self.clock not in ("fast", "slow", "meta"):
            raise ValueError(f"component {self.name!r} has unknown clock {self.clock!r}")


@dataclass(frozen=True)
class StateLayout:
    """Packed offsets per component.  Immutable and hashable."""

    components: tuple[ComponentSpec, ...]
    _offsets: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        off = 0
        table: dict[str, tuple[int, int]] = {}
        seen: set[str] = set()
        for c in self.components:
            if c.name in seen:
                raise ValueError(f"duplicate component {c.name!r}")
            seen.add(c.name)
            table[c.name] = (off, off + c.dim)
            off += c.dim
        object.__setattr__(self, "_offsets", table)

    # -- geometry ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return sum(c.dim for c in self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[ComponentSpec]:
        return iter(self.components)

    def __contains__(self, name: str) -> bool:
        return name in self._offsets

    def span(self, name: str) -> tuple[int, int]:
        if name not in self._offsets:
            raise KeyError(f"no state component {name!r}; have {sorted(self._offsets)}")
        return self._offsets[name]

    def slice(self, name: str) -> slice:
        a, b = self.span(name)
        return slice(a, b)

    def get(self, x: Tensor, name: str) -> Tensor:
        """``x[..., component]`` — the only sanctioned way to read state."""
        return x[..., self.slice(name)]

    def set(self, x: Tensor, name: str, value: Tensor) -> Tensor:
        a, b = self.span(name)
        out = x.clone()
        out[..., a:b] = value
        return out

    def spec(self, name: str) -> ComponentSpec:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    # -- masks -------------------------------------------------------------
    def mask(self, names: Sequence[str], *, device=None, dtype=torch.float32) -> Tensor:
        """0/1 vector of length :attr:`dim` selecting the named components."""
        m = torch.zeros(self.dim, device=device, dtype=dtype)
        for n in names:
            a, b = self.span(n)
            m[a:b] = 1.0
        return m

    def clock_names(self, clock: str) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.clock == clock)

    def exported_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.exported)

    def stochastic_mask(self, *, device=None, dtype=torch.float32) -> Tensor:
        return self.mask([c.name for c in self.components if c.stochastic], device=device, dtype=dtype)

    def clock_mask(self, clock: str, *, device=None, dtype=torch.float32) -> Tensor:
        return self.mask(self.clock_names(clock), device=device, dtype=dtype)

    def export_mask(self, *, device=None, dtype=torch.float32) -> Tensor:
        return self.mask(self.exported_names(), device=device, dtype=dtype)

    # -- provenance --------------------------------------------------------
    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": c.name,
                "dim": c.dim,
                "offset": self._offsets[c.name][0],
                "units": c.units,
                "clock": c.clock,
                "exported": c.exported,
                "stochastic": c.stochastic,
                "description": c.description,
            }
            for c in self.components
        ]

    def as_dict(self) -> dict[str, object]:
        return {"dim": self.dim, "components": self.describe()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateLayout":
        comps = []
        for c in payload["components"]:  # type: ignore[index]
            comps.append(
                ComponentSpec(
                    name=str(c["name"]),
                    dim=int(c["dim"]),
                    units=str(c["units"]),
                    clock=str(c["clock"]),
                    exported=bool(c.get("exported", True)),
                    stochastic=bool(c.get("stochastic", True)),
                    description=str(c.get("description", "")),
                )
            )
        return cls(tuple(comps))


def default_layout(*, n_spectral_modes: int = 8, n_adaptation: int = 2, n_uncertainty: int = 4) -> StateLayout:
    """The SC-WBD-001-beta structured regional state.

    ``rate_e``/``rate_i``   mean-field excitatory / inhibitory activity
    ``adaptation``          slow spike-frequency-adaptation / synaptic-depression channels
    ``spectral``            2*K quadrature amplitudes of learned regional oscillatory modes
    ``hemo``                Balloon-Windkessel compartments (s, f, v, q) on the slow clock
    ``uncertainty``         per-region log-variance channel (the uncertainty ledger's
                            state-level carrier; §2.7 — variance never collapses with bias)
    """
    return StateLayout(
        (
            ComponentSpec("rate_e", 1, "Hz", "fast", True, True, "excitatory mean-field rate"),
            ComponentSpec("rate_i", 1, "Hz", "fast", True, True, "inhibitory mean-field rate"),
            ComponentSpec("adaptation", n_adaptation, "dimensionless", "fast", False, False, "adaptation currents"),
            ComponentSpec(
                "spectral", 2 * n_spectral_modes, "dimensionless", "fast", True, True, "quadrature spectral modes"
            ),
            ComponentSpec("hemo", 4, "dimensionless", "slow", False, False, "Balloon-Windkessel s,f,v,q"),
            ComponentSpec(
                "uncertainty", n_uncertainty, "log_var", "meta", False, False, "per-region predictive log-variance"
            ),
        )
    )


def scalar_layout() -> StateLayout:
    """The ablation baseline: **one scalar per region**.

    This is the failure mode ARCHITECTURE.md §5 names explicitly.  It exists so
    the structured-state claim can be tested against it at matched compute, not
    because it is a defensible model of a cortical region.
    """
    return StateLayout((ComponentSpec("activity", 1, "dimensionless", "fast", True, True, "single scalar per region"),))
