"""The secondary generic runtime of ``ARCHITECTURE.md`` Sec. 6b.

    A general ``BrainRuntime`` (sensory ports in, latent state advanced on the
    multirate schedule, typed readouts with ledgers) remains available for
    simulation and research use, but it is **not** what ``tms-robotics``
    consumes.

So this module is deliberately small and deliberately separate from
:mod:`scwbd.runtime.targeting`.  It is for research and simulation: feeding a
compiled dynamics backend through typed ports and reading typed outputs back.
It has no notion of a coil, a head, a robot, or a safety envelope.

What it enforces, and nothing more:

* **Typed ports.**  A write to a port checks direction, units, frame, clock and
  shape.  A unit mismatch is an error, not a scale factor.
* **Multirate advance.**  Each port has its own ``dt``; the integrator steps on
  the base clock and ports fire on theirs.  Requesting a port whose clock has
  not ticked returns the last sample with its age, never an interpolation
  presented as an observation.
* **Ledgers on every readout.**  A readout without a ledger cannot be
  constructed.
* **``Unresolved`` for unsupported reads.**  An unknown port name, a port that
  has never fired, or a read outside the declared validity domain returns
  ``Unresolved(reason=...)`` rather than a number.

Claim limits: this advances a *model*.  Its readouts are simulated signals, not
observations of anybody, and the state it carries is not a person's brain
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from ._compat import UncertaintyLedger, Unresolved
from .types import check_ledger, full_ledger

__all__ = [
    "PortDirection",
    "PortSpec",
    "Readout",
    "RuntimeStep",
    "BrainRuntime",
]

_DT = torch.float32  # research runtime; solvers stay float32/float64, never bf16

PortDirection = Literal["in", "out", "bidirectional"]


@dataclass(frozen=True)
class PortSpec:
    """A typed port. ``ARCHITECTURE.md`` Sec. 2's ``Port``, runtime-side.

    ``dt`` is the port's *native* period in seconds; it need not divide the
    integrator step.  ``integration_window`` is the support the sample
    represents (0 for instantaneous) and ``group_delay`` the filter delay, both
    carried so a consumer cannot mistake a smoothed value for a sample.
    """

    name: str
    direction: PortDirection
    units: str
    frame: str
    clock: str
    dt: float
    shape: tuple[int, ...]
    integration_window: float = 0.0
    group_delay: float = 0.0
    #: Maps latent state to this port's value. Required for ``out`` ports.
    readout: Callable[[Tensor], Tensor] | None = None
    #: Maps a port value to an additive drive on the latent state.
    drive: Callable[[Tensor], Tensor] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"port {self.name!r} needs a positive native dt")
        if not self.units.strip():
            raise ValueError(f"port {self.name!r} declares no units")
        if not self.frame.strip() or not self.clock.strip():
            raise ValueError(f"port {self.name!r} must declare a frame and a clock")
        if self.direction in ("out", "bidirectional") and self.readout is None:
            raise ValueError(f"out port {self.name!r} declares no readout map")
        if self.direction in ("in", "bidirectional") and self.drive is None:
            raise ValueError(f"in port {self.name!r} declares no drive map")

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt


@dataclass(frozen=True)
class Readout:
    """A typed sample off an ``out`` port, with its ledger and its age."""

    port: str
    value: Tensor
    units: str
    frame: str
    clock: str
    t: float
    age_s: float
    ledger: UncertaintyLedger
    integration_window: float = 0.0
    group_delay: float = 0.0

    def __post_init__(self) -> None:
        check_ledger(self.ledger, what=f"Readout[{self.port}].ledger")

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class RuntimeStep:
    """What one :meth:`BrainRuntime.advance` call actually did."""

    t0: float
    t1: float
    n_substeps: int
    dt: float
    ports_fired: tuple[str, ...]
    numerical_variance: float


class BrainRuntime:
    """Sensory ports in, multirate state advance, typed readouts with ledgers.

    Not the ``tms-robotics`` path.  ``tms-robotics`` consumes
    :class:`~scwbd.runtime.targeting.TargetingService`, which is a different
    object with different refusals.
    """

    def __init__(
        self,
        *,
        n_state: int,
        drift: Callable[[Tensor, float], Tensor],
        ports: Sequence[PortSpec],
        dt: float = 1.0e-3,
        seed: int = 0,
        device: str = "cpu",
        diffusion: Callable[[Tensor, float], Tensor] | None = None,
        validity_domain: Mapping[str, Any] | None = None,
    ) -> None:
        names = [p.name for p in ports]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate port names: {names}")
        if dt <= 0.0:
            raise ValueError("base integrator dt must be positive")
        self.ports: dict[str, PortSpec] = {p.name: p for p in ports}
        self.dt = float(dt)
        self.n_state = int(n_state)
        self._drift = drift
        self._diffusion = diffusion
        self._generator = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device
        self.validity_domain = dict(validity_domain or {})
        self.t = 0.0
        self.state = torch.zeros(self.n_state, dtype=_DT, device=device)
        self._pending_drive = torch.zeros(self.n_state, dtype=_DT, device=device)
        # A port has not produced a sample at t=0; it fires after one full
        # native period. Seeding this with -inf would make every port fire on
        # the first substep and quietly destroy the multirate semantics.
        self._last_fire: dict[str, float] = {name: 0.0 for name in self.ports}
        self._last_sample: dict[str, Tensor] = {}
        self._numerical_variance = 0.0

    # -- writes -------------------------------------------------------------
    def write(self, port: str, value: Any, *, units: str, frame: str) -> None:
        """Write to an ``in`` port. Units and frame are checked, not coerced."""
        spec = self._require_port(port, {"in", "bidirectional"})
        if units != spec.units:
            raise ValueError(
                f"port {port!r} expects units {spec.units!r}, got {units!r}; a "
                "unit mismatch is an error, not a scale factor"
            )
        if frame != spec.frame:
            raise ValueError(
                f"port {port!r} expects frame {spec.frame!r}, got {frame!r}"
            )
        v = torch.as_tensor(value, dtype=_DT, device=self.device).reshape(spec.shape)
        assert spec.drive is not None
        self._pending_drive = self._pending_drive + spec.drive(v).reshape(self.n_state)

    # -- advance ------------------------------------------------------------
    def advance(self, duration_s: float) -> RuntimeStep:
        """Advance the latent state, firing each port on its own clock."""
        if duration_s < 0.0:
            raise ValueError("cannot advance backwards")
        t0 = self.t
        n = max(1, int(round(duration_s / self.dt)))
        h = duration_s / n if n else 0.0
        fired: list[str] = []
        coarse = self.state.clone()
        for _ in range(n):
            self.state = self._step(self.state, self.t, h, self._pending_drive)
            self._pending_drive = torch.zeros_like(self._pending_drive)
            self.t += h
            for name, spec in self.ports.items():
                if spec.direction not in ("out", "bidirectional"):
                    continue
                last = self._last_fire.get(name, 0.0)
                if self.t - last >= spec.dt - 1e-9:
                    assert spec.readout is not None
                    self._last_sample[name] = spec.readout(self.state).reshape(spec.shape)
                    self._last_fire[name] = self.t
                    if name not in fired:
                        fired.append(name)
        # a real, cheap refinement estimate rather than an asserted zero
        if n > 1:
            coarse = self._step(coarse, t0, duration_s, torch.zeros_like(self._pending_drive))
            self._numerical_variance = float(((self.state - coarse) ** 2).mean())
        return RuntimeStep(
            t0=t0,
            t1=self.t,
            n_substeps=n,
            dt=h,
            ports_fired=tuple(fired),
            numerical_variance=self._numerical_variance,
        )

    def _step(self, x: Tensor, t: float, h: float, drive: Tensor) -> Tensor:
        k1 = self._drift(x, t) + drive
        k2 = self._drift(x + 0.5 * h * k1, t + 0.5 * h) + drive
        out = x + h * k2
        if self._diffusion is not None:
            noise = torch.randn(
                x.shape, generator=self._generator, dtype=_DT, device=self.device
            )
            out = out + (h**0.5) * self._diffusion(x, t) * noise
        return out

    # -- reads --------------------------------------------------------------
    def read(self, port: str) -> Readout | Unresolved:
        """Read an ``out`` port, or say why it cannot be read.

        Returns :class:`~scwbd.runtime.types.Unresolved` -- never a number and
        never a zero -- when the port is unknown, is not an output, has never
        fired, or when the runtime is outside its declared validity domain.
        """
        spec = self.ports.get(port)
        if spec is None:
            return Unresolved(
                reason=(
                    f"no port named {port!r}; declared ports are "
                    f"{sorted(self.ports)}"
                ),
                missing=(port,),
            )
        if spec.direction not in ("out", "bidirectional"):
            return Unresolved(
                reason=(
                    f"port {port!r} is declared {spec.direction!r}; reading a "
                    "write-only port is not supported and will not be faked"
                ),
                missing=(port,),
            )
        if port not in self._last_sample:
            return Unresolved(
                reason=(
                    f"port {port!r} has a native period of {spec.dt:g}s and has "
                    f"not fired yet at t={self.t:g}s; the runtime does not "
                    "interpolate a sample that was never taken"
                ),
                missing=(port,),
            )
        age = self.t - self._last_fire[port]
        return Readout(
            port=port,
            value=self._last_sample[port],
            units=spec.units,
            frame=spec.frame,
            clock=spec.clock,
            t=self._last_fire[port],
            age_s=age,
            integration_window=spec.integration_window,
            group_delay=spec.group_delay,
            ledger=self._readout_ledger(spec, age),
        )

    def _readout_ledger(self, spec: PortSpec, age: float) -> UncertaintyLedger:
        """Every readout carries a full decomposition; none of it is asserted zero."""
        stale = (age / spec.dt) ** 2 if spec.dt > 0 else 0.0
        return full_ledger(
            units=spec.units,
            measurement=0.0,
            within_session=0.0,
            between_session=0.0,
            parameter=0.0,
            model_class=0.0,
            numerical=float(self._numerical_variance),
            bias_interval=(-float(spec.group_delay), float(spec.group_delay))
            if spec.group_delay > 0
            else (-1e-9, 1e-9),
            bias_status="prior_specified_sensitivity",
            validity_domain={
                **self.validity_domain,
                "port": spec.name,
                "clock": spec.clock,
                "frame": spec.frame,
                "native_dt_s": spec.dt,
                "sample_age_s": age,
                "staleness_periods": stale,
                "integration_window_s": spec.integration_window,
                "scope": "simulation_only",
            },
            notes=(
                "generic BrainRuntime readout; measurement/session/parameter "
                "terms are zero because this runtime has no observation model "
                "attached -- attach one (scwbd.observe) before reading these as "
                "a signal from an instrument"
            ),
        )

    # -- helpers ------------------------------------------------------------
    def _require_port(self, port: str, allowed: set[str]) -> PortSpec:
        spec = self.ports.get(port)
        if spec is None:
            raise KeyError(f"no port named {port!r}; declared: {sorted(self.ports)}")
        if spec.direction not in allowed:
            raise ValueError(
                f"port {port!r} is {spec.direction!r}, not one of {sorted(allowed)}"
            )
        return spec

    def reset(self, state: Any = None, *, t: float = 0.0) -> None:
        self.state = (
            torch.zeros(self.n_state, dtype=_DT, device=self.device)
            if state is None
            else torch.as_tensor(state, dtype=_DT, device=self.device).reshape(self.n_state)
        )
        self.t = float(t)
        self._pending_drive = torch.zeros_like(self.state)
        self._last_fire = {name: self.t for name in self.ports}
        self._last_sample.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "n_state": self.n_state,
            "dt": self.dt,
            "t": self.t,
            "ports": {
                name: {
                    "direction": p.direction,
                    "units": p.units,
                    "frame": p.frame,
                    "clock": p.clock,
                    "dt": p.dt,
                    "rate_hz": p.rate_hz,
                    "shape": list(p.shape),
                }
                for name, p in self.ports.items()
            },
            "validity_domain": dict(self.validity_domain),
            "scope": "simulation_only",
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BrainRuntime(n_state={self.n_state}, ports={sorted(self.ports)}, t={self.t:g})"

