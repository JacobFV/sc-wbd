"""Physical and temporal support (thesis sec. 2.6, Appendix C).

A datum is never a bare coordinate.  It has a *support*: the physical set it
integrates over, in a named frame, with declared units and a point-spread /
lead-field / integration kernel.  It also has a *temporal support*: a named
clock, a native sampling period, an exposure window, a filter group delay and
a jitter.  R01 exists because numerical shape compatibility is not measurement
compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import ClockId, FrameId, ScaleId
from .ledger import UncertaintyLedger
from .units import Unit

__all__ = ["PSF", "Support", "TemporalSupport", "SupportKind"]

SupportKind = Literal[
    "voxel",
    "surface_vertex",
    "parcel",
    "sensor",
    "mesh",
    "field",
    "event",
    "band",
]


class PSF(SchemaModel):
    """Point-spread / lead-field / integration kernel of a support.

    A nominal coordinate substituted for a physical support is a rejection
    condition in Appendix B ("Spatial support and point-spread"), so a
    ``Support`` that claims fine resolution must say *how* it smears.
    """

    kind: Literal[
        "point",
        "gaussian",
        "lead_field",
        "integration_kernel",
        "hemodynamic",
        "acoustic",
        "tract_endpoint",
        "empirical",
    ]
    #: Full width at half maximum per spatial axis, in ``extent_units``.
    fwhm: tuple[float, ...] | None = None
    #: Units of the kernel's *output* (e.g. "V" for a lead field mapping A*m).
    units: Unit = Unit("dimensionless")
    #: Units in which ``fwhm`` is expressed.
    extent_units: Unit = Unit("m")
    #: Reference to the stored kernel/lead-field asset (agent C/F own the data).
    kernel_ref: str | None = None
    #: True when the kernel is only a nominal/placeholder description.
    nominal: bool = False
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "PSF":
        if self.kind != "point" and self.fwhm is None and self.kernel_ref is None:
            raise ValueError(
                f"psf kind={self.kind!r} must declare either fwhm or kernel_ref; "
                "a named kernel with no description is an unknown support"
            )
        if self.fwhm is not None and any(v < 0.0 for v in self.fwhm):
            raise ValueError("psf fwhm entries must be non-negative")
        return self


class Support(SchemaModel):
    """Physical support of a datum. Never a bare coordinate."""

    kind: SupportKind
    frame: FrameId
    units: Unit
    psf: PSF | None = None
    extent: tuple[float, ...] | None = None
    n_elements: int | None = None

    # -- additions used by the compiler (not part of the sec. 2 core tuple) --
    #: Element of the resolution poset this support realizes.
    resolution: ScaleId | None = None
    #: Units in which ``extent`` is expressed.
    extent_units: Unit = Unit("m")
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Support":
        if self.n_elements is not None and self.n_elements <= 0:
            raise ValueError(f"n_elements must be positive, got {self.n_elements}")
        if self.extent is not None and any(v < 0.0 for v in self.extent):
            raise ValueError("extent entries must be non-negative")
        return self

    @property
    def is_nominal(self) -> bool:
        """True when no physical support was supplied beyond a label."""
        return self.psf is None and self.extent is None


class TemporalSupport(SchemaModel):
    """Sampling clock, exposure window, filter delay and jitter of a datum."""

    clock: ClockId
    dt: float = Field(gt=0.0)  # seconds, native
    integration_window: float = Field(default=0.0, ge=0.0)  # 0 for instantaneous
    group_delay: float = Field(default=0.0)  # seconds, filter delay
    jitter_sd: float = Field(default=0.0, ge=0.0)

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    @property
    def effective_latency(self) -> float:
        """Delay from physical event to reported sample."""
        return self.group_delay + 0.5 * self.integration_window

    def nyquist_hz(self) -> float:
        return 0.5 / self.dt
