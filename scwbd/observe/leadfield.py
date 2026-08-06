"""EEG/MEG forward operators: the lead field *is* the sensor's spatial support.

Thesis anchors
--------------
* body.tex Sec. 2.8 --- "An EEG electrode is first a contact in a cap or
  digitizer frame.  It acquires a subject-MRI location only after digitization
  and coregistration, and it does not become a cortical MNI source coordinate
  without an explicit head model, inverse operator, and atlas warp."  The
  consequence enforced here: :class:`LeadField` is emitted as a
  :class:`~scwbd.observe.base.PSF` of ``kind="leadfield"`` attached to a
  ``Support(kind="sensor")``.  A scalp label is never a support.
* body.tex Table ``tab:modalities`` --- dominant EEG/MEG bias-variance terms are
  contact/helmet location, impedance, conductivity, reference, artifacts,
  coregistration, source non-identifiability.  Each has a named handle below.
* Tissue conductivities enter as :class:`~scwbd.observe.base.Prior` objects, not
  constants (IT'IS database v4.1 low-frequency values plus the McCann et al.
  2019 skull meta-analysis).

Contents
--------
``SphericalHeadModel``
    Exact analytic 3/4-layer concentric-sphere EEG forward solution obtained by
    solving the ``n``-th Legendre boundary-value problem in a numerically stable
    log-scaled downward recursion.  Reduces analytically to the closed-form
    homogeneous sphere (tested), and is validated against MNE's Berg-approximated
    multilayer sphere model (tested).
``sarvas_meg``
    Sarvas (1987) closed-form magnetic field of a current dipole in a spherically
    symmetric conductor.  Conductivity-independent by construction.
``BEMLeadField``
    Thin wrapper over ``mne`` BEM/forward machinery for realistic subject heads.
    We wrap, we do not reimplement; the wrapper is validated against MNE's own
    forward solution to numerical tolerance.
``ReferenceOperator``
    average / linked-mastoid / REST, applied as an explicit matrix on the lead
    field and the data alike.
``ElectrodePositionUncertainty``
    Propagates an electrode-position covariance (including cross-electrode
    blocks such as a whole-cap rotation) into lead-field and sensor-space
    variance, retaining cross terms per ARCHITECTURE.md Sec. 3.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch

from .base import (
    AMPERE_METER,
    DIMENSIONLESS,
    TESLA,
    VOLT,
    BiasTerm,
    ObservationRefusal,
    PSF,
    Prior,
    Provenance,
    Support,
    UNKNOWN,
    UncertaintyLedger,
    VarianceDecomposition,
)

__all__ = [
    "MU0",
    "TissueConductivityPriors",
    "ITIS_CONDUCTIVITY",
    "STANDARD_EEG_CONDUCTIVITY",
    "SphericalHeadModel",
    "LeadField",
    "sarvas_meg",
    "BEMLeadField",
    "ReferenceOperator",
    "ElectrodeImpedance",
    "ElectrodePositionUncertainty",
    "legendre_p_and_dp",
]

MU0 = 4.0e-7 * math.pi
"""Vacuum permeability, H/m."""

BOLTZMANN = 1.380649e-23


# ==========================================================================
# tissue properties -- priors, never constants
# ==========================================================================


@dataclass(frozen=True)
class TissueConductivityPriors:
    """Conductivity of each concentric layer as a prior with provenance.

    Layers are ordered **inner to outer**: ``brain, csf, skull, scalp`` for the
    four-layer model; ``brain, skull, scalp`` for the three-layer model (CSF
    omitted, which is itself a modelled bias term -- see
    :meth:`csf_omission_bias`).
    """

    brain: Prior
    skull: Prior | None = None
    scalp: Prior | None = None
    csf: Prior | None = None
    reference: str = "unspecified"

    @property
    def layer_names(self) -> tuple[str, ...]:
        if self.skull is None and self.scalp is None:
            return ("brain",)
        if self.csf is None:
            return ("brain", "skull", "scalp")
        return ("brain", "csf", "skull", "scalp")

    @property
    def priors(self) -> tuple[Prior, ...]:
        return tuple(getattr(self, n) for n in self.layer_names)

    @classmethod
    def homogeneous(cls, sigma: float = 0.33, source: str = "asserted") -> "TissueConductivityPriors":
        """Single-compartment conductor -- the analytic regression target."""
        return cls(
            brain=Prior("sigma_brain", "delta", (sigma,), units="S/m", source=source),
            reference="homogeneous single-shell reference model",
        )

    def means(self) -> tuple[float, ...]:
        return tuple(p.mean for p in self.priors)

    def sample(self, *, seed: int) -> tuple[float, ...]:
        return tuple(
            float(p.sample((), seed=seed + i)) for i, p in enumerate(self.priors)
        )

    def csf_omission_bias(self, units: str = VOLT) -> BiasTerm:
        """Bias induced by collapsing CSF into brain in a three-layer model.

        Externally bounded by the published three- versus four-layer comparisons
        (Vorwerk et al. 2014; Ramon et al. 2006), which report relative
        differences of roughly 10--20 % in scalp topographies.  Expressed here as
        a *relative* interval so the caller multiplies by the local signal scale.
        """
        return BiasTerm(
            name="conductivity_model_csf_omission",
            interval=(-0.20, 0.20),
            status="externally_bounded",
            units=DIMENSIONLESS,
            external_bound="Vorwerk et al. 2014 / Ramon et al. 2006 three- vs "
            "four-compartment head-model comparison (relative topography error)",
            note="relative multiplicative bias on scalp potential; multiply by "
            "the local field magnitude before adding to a volt-valued ledger",
        )


ITIS_CONDUCTIVITY = TissueConductivityPriors(
    brain=Prior.lognormal_from_mean_cv(
        "sigma_brain",
        0.33,
        0.25,
        units="S/m",
        source="IT'IS v4.1 low-frequency brain (grey/white mixture); "
        "Geddes & Baker 1967 range 0.12-0.48 S/m",
        validity=(0.10, 0.60),
    ),
    csf=Prior.lognormal_from_mean_cv(
        "sigma_csf",
        1.79,
        0.06,
        units="S/m",
        source="Baumann et al. 1997 in-vitro CSF at body temperature; "
        "IT'IS v4.1 cerebrospinal fluid",
        validity=(1.45, 2.10),
    ),
    skull=Prior.lognormal_from_mean_cv(
        "sigma_skull",
        0.0160,
        0.45,
        units="S/m",
        source="McCann, Pisano & Beltrachini 2019 meta-analysis of human skull "
        "conductivity (reported spread 0.0041-0.0333 S/m); IT'IS v4.1 cortical "
        "and cancellous bone bracket the same range",
        validity=(0.0041, 0.0333),
    ),
    scalp=Prior.lognormal_from_mean_cv(
        "sigma_scalp",
        0.4137,
        0.20,
        units="S/m",
        source="IT'IS v4.1 wet skin at low frequency (0.4137 S/m); "
        "Geddes & Baker 1967 scalp 0.33 S/m lies inside the prior",
        validity=(0.20, 0.60),
    ),
    reference="IT'IS Foundation tissue property database v4.1 + McCann 2019",
)
"""Four-layer prior set anchored on the IT'IS database and the skull meta-analysis."""


STANDARD_EEG_CONDUCTIVITY = TissueConductivityPriors(
    brain=Prior("sigma_brain", "delta", (0.33,), units="S/m", source="FieldTrip/MNE convention"),
    csf=Prior("sigma_csf", "delta", (1.79,), units="S/m", source="Baumann 1997"),
    skull=Prior("sigma_skull", "delta", (0.0042,), units="S/m", source="FieldTrip convention (1:80)"),
    scalp=Prior("sigma_scalp", "delta", (0.33,), units="S/m", source="FieldTrip/MNE convention"),
    reference="conventional EEG head-model triple; delta priors record that these "
    "values are asserted, not estimated",
)
"""Conventional asserted values.  Delta priors make the assertion visible."""


# ==========================================================================
# Legendre machinery
# ==========================================================================


def legendre_p_and_dp(x: torch.Tensor, n_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``P_n(x)`` and ``P_n'(x)`` for ``n = 0..n_max``.

    Shapes: ``(n_max+1, *x.shape)``.  The derivative uses the singularity-free
    recurrence ``P_n' = x P_{n-1}' + n P_{n-1}`` so that ``x = +-1`` is exact
    rather than a removable 0/0.
    """
    x = x.to(torch.float64)
    shape = (n_max + 1,) + tuple(x.shape)
    P = torch.empty(shape, dtype=torch.float64, device=x.device)
    dP = torch.empty(shape, dtype=torch.float64, device=x.device)
    P[0] = 1.0
    dP[0] = 0.0
    if n_max >= 1:
        P[1] = x
        dP[1] = 1.0
    for n in range(2, n_max + 1):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
        dP[n] = x * dP[n - 1] + n * P[n - 1]
    return P, dP


# ==========================================================================
# analytic multilayer spherical head model
# ==========================================================================


@dataclass(frozen=True)
class LeadField:
    """A forward operator plus everything needed to interpret it as a support.

    ``matrix`` has shape ``(n_sensors, n_sources)`` for fixed-orientation
    sources, or ``(n_sensors, n_sources, 3)`` for free orientation.  Units are
    V/(A*m) for EEG and T/(A*m) for MEG: the lead field is a *transfer
    function*, and the sensor unit only appears after multiplying by a dipole
    moment.
    """

    matrix: torch.Tensor
    sensor_units: str
    source_positions: torch.Tensor
    sensor_positions: torch.Tensor
    frame: str
    modality: Literal["eeg", "meg"]
    sensor_names: tuple[str, ...] = ()
    reference: str = "infinity"
    orientation: Literal["fixed", "free"] = "free"
    ledger: UncertaintyLedger | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_sources(self) -> int:
        return int(self.matrix.shape[1])

    def as_matrix(self) -> torch.Tensor:
        """Flatten free orientation to ``(n_sensors, n_sources * 3)``."""
        if self.orientation == "fixed":
            return self.matrix
        return self.matrix.reshape(self.n_sensors, -1)

    def project(self, orientations: torch.Tensor) -> "LeadField":
        """Constrain to fixed orientations (e.g. cortical surface normals)."""
        if self.orientation == "fixed":
            raise ValueError("lead field is already fixed-orientation")
        if orientations.shape != (self.n_sources, 3):
            raise ValueError("orientations must have shape (n_sources, 3)")
        o = orientations.to(self.matrix.dtype)
        m = torch.einsum("esk,sk->es", self.matrix, o)
        return LeadField(
            matrix=m,
            sensor_units=self.sensor_units,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            frame=self.frame,
            modality=self.modality,
            sensor_names=self.sensor_names,
            reference=self.reference,
            orientation="fixed",
            ledger=self.ledger,
            meta={**self.meta, "orientation_constraint": "supplied"},
        )

    # -- the thesis Sec. 2.8 requirement -----------------------------------
    def as_psf(self) -> PSF:
        """The lead field *is* the point-spread function of these sensors."""
        return PSF(
            kind="leadfield",
            frame=self.frame,
            units=f"{self.sensor_units}/({AMPERE_METER})",
            matrix=self.as_matrix(),
            source_positions=self.source_positions,
            meta={
                "modality": self.modality,
                "reference": self.reference,
                "orientation": self.orientation,
                **dict(self.meta),
            },
        )

    def as_support(self) -> Support:
        """``Support(kind="sensor")`` whose ``psf`` is the forward operator."""
        return Support(
            kind="sensor",
            frame=self.frame,
            units=self.sensor_units,
            psf=self.as_psf(),
            n_elements=self.n_sensors,
            labels=self.sensor_names or None,
        )


@dataclass(frozen=True)
class SphericalHeadModel:
    """Exact concentric-sphere EEG forward model with layered conductivity.

    Parameters
    ----------
    radii
        Outer radius of each layer, **inner to outer**, metres.  Example
        four-layer adult head: ``(0.079, 0.082, 0.086, 0.090)`` for
        brain / CSF / skull / scalp.
    conductivity
        A :class:`TissueConductivityPriors`.  The forward solve consumes a
        *realisation*; :meth:`sample_conductivity` draws one and
        :meth:`conductivity_sensitivity` sweeps the prior.
    center
        Sphere centre in the working frame, metres.

    Notes
    -----
    For a dipole at radius ``b`` inside the innermost layer with moment ``q``,
    the surface potential at a unit vector ``e`` is

    .. math::
        V = \\sum_{n\\ge1} D_n\\bigl[\\,n\\,q_r P_n(x)
            + P_n'(x)\\,(q\\cdot e - q_r x)\\bigr],\\qquad
        x = e\\cdot \\hat r_0,

    where ``D_n`` is the layered radial factor obtained by solving the two-point
    boundary-value problem for ``A_j r^n + B_j r^{-(n+1)}`` with continuity of
    ``V`` and ``sigma dV/dr`` at every interface and ``sigma dV/dr = 0`` at the
    scalp.  For a single homogeneous layer ``D_n`` collapses to
    ``(2n+1) b^{n-1} / (4 pi sigma n R^{n+1})``, reproducing the textbook
    closed form (asserted in ``tests/observe/test_leadfield_sphere.py``).
    """

    radii: tuple[float, ...]
    conductivity: TissueConductivityPriors = ITIS_CONDUCTIVITY
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: str = "head_sphere"
    n_terms: int = 0  # 0 -> chosen adaptively

    def __post_init__(self) -> None:
        r = self.radii
        if len(r) < 1:
            raise ValueError("at least one layer required")
        if any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
            raise ValueError("radii must be strictly increasing inner->outer")
        if len(r) != len(self.conductivity.layer_names):
            raise ValueError(
                f"{len(r)} radii but conductivity declares "
                f"{len(self.conductivity.layer_names)} layers "
                f"{self.conductivity.layer_names}"
            )

    # -- convenience constructors ------------------------------------------
    @classmethod
    def adult_four_layer(
        cls,
        head_radius: float = 0.090,
        conductivity: TissueConductivityPriors = ITIS_CONDUCTIVITY,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "SphericalHeadModel":
        """Relative radii 0.87/0.92/0.96/1.0 (brain/CSF/skull/scalp)."""
        rel = (0.87, 0.92, 0.96, 1.0)
        return cls(
            radii=tuple(head_radius * x for x in rel),
            conductivity=conductivity,
            center=center,
        )

    @classmethod
    def adult_three_layer(
        cls,
        head_radius: float = 0.090,
        conductivity: TissueConductivityPriors | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "SphericalHeadModel":
        c = conductivity
        if c is None:
            c = TissueConductivityPriors(
                brain=ITIS_CONDUCTIVITY.brain,
                skull=ITIS_CONDUCTIVITY.skull,
                scalp=ITIS_CONDUCTIVITY.scalp,
                csf=None,
                reference=ITIS_CONDUCTIVITY.reference + " (CSF collapsed into brain)",
            )
        rel = (0.87, 0.92, 1.0)
        return cls(radii=tuple(head_radius * x for x in rel), conductivity=c, center=center)

    @property
    def R(self) -> float:
        return self.radii[-1]

    @property
    def n_layers(self) -> int:
        return len(self.radii)

    def sample_conductivity(self, *, seed: int) -> tuple[float, ...]:
        return self.conductivity.sample(seed=seed)

    # -- the radial factor D_n ---------------------------------------------
    def _radial_factor(
        self,
        b: torch.Tensor,
        sigmas: Sequence[float],
        n_max: int,
    ) -> torch.Tensor:
        """``D_n`` for every ``n`` and every source radius ``b``.

        Returns shape ``(n_max, *b.shape)`` indexed by ``n = 1..n_max``.

        The recursion carries the projective pair ``(a, b)`` of the outward and
        inward radial solutions **normalised to unit max-abs** together with a
        separate accumulated log-scale, so that the ``(R/r_1)^(2n+1)`` growth
        never overflows and no catastrophic cancellation occurs.
        """
        radii = self.radii
        N = len(radii)
        r1 = radii[0]
        sig = [float(s) for s in sigmas]
        n = torch.arange(1, n_max + 1, dtype=torch.float64)

        # state at the scalp: sigma dV/dr = 0  ->  n*alpha - (n+1)*beta = 0
        va = torch.ones_like(n)
        vb = n / (n + 1.0)
        logscale = torch.log(n + 1.0)  # physical (alpha, beta) = e^L * (va, vb)

        def _renorm(a: torch.Tensor, bb: torch.Tensor, L: torch.Tensor):
            m = torch.maximum(a.abs(), bb.abs())
            m = torch.where(m > 0, m, torch.ones_like(m))
            return a / m, bb / m, L + torch.log(m)

        # walk inward: layer N down to layer 1
        for j in range(N - 1, 0, -1):
            r_out, r_in = radii[j], radii[j - 1]
            # propagate inside layer j+1 (index j) from r_out to r_in
            ta = torch.log(va.abs().clamp_min(1e-300)) + n * math.log(r_in / r_out)
            tb = torch.log(vb.abs().clamp_min(1e-300)) + (n + 1.0) * math.log(r_out / r_in)
            M = torch.maximum(ta, tb)
            va = torch.sign(va) * torch.exp(ta - M)
            vb = torch.sign(vb) * torch.exp(tb - M)
            logscale = logscale + M
            # cross the interface at r_in: sigma_out (layer j) -> sigma_in (layer j-1)
            sr = sig[j] / sig[j - 1]
            denom = 2.0 * n + 1.0
            a_new = (va * (n * sr + n + 1.0) + vb * (n + 1.0) * (1.0 - sr)) / denom
            b_new = (va * n * (1.0 - sr) + vb * (n + n * sr + sr)) / denom
            va, vb, logscale = _renorm(a_new, b_new, logscale)

        # physical inward-solution amplitude at r_1, for the normalisation
        # V(R) = 2n+1.  Source term beta_1 = S_n r_1^{-(n+1)} fixes the scale.
        # D_n = (2n+1) b^(n-1) r_1^{-(n+1)} / (4 pi sigma_1 * e^L * vb)
        b = b.to(torch.float64)
        nb = n.reshape((-1,) + (1,) * b.dim())
        log_b = torch.log(b.clamp_min(1e-30)).unsqueeze(0)
        log_num = (nb - 1.0) * log_b  # b^(n-1), shape (n_max, *b.shape)

        log_den = (
            logscale
            + torch.log(vb.abs().clamp_min(1e-300))
            + (n + 1.0) * math.log(r1)
            + math.log(4.0 * math.pi * sig[0])
        )
        sgn = torch.sign(vb)
        D = sgn.reshape((-1,) + (1,) * b.dim()) * torch.exp(
            torch.log(2.0 * nb + 1.0) + log_num - log_den.reshape((-1,) + (1,) * b.dim())
        )
        # a dipole exactly at the centre only radiates n = 1
        D = torch.where(torch.isfinite(D), D, torch.zeros_like(D))
        return D

    def _auto_n_terms(self, b_max: float) -> int:
        if self.n_terms > 0:
            return self.n_terms
        ratio = min(max(b_max / self.radii[0], 1e-6), 0.999)
        # truncate where (b/r1)^n < 1e-14
        n = int(math.ceil(math.log(1e-14) / math.log(ratio)))
        return int(min(max(n, 40), 4000))

    # -- the forward solve -------------------------------------------------
    def potential(
        self,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        sigmas: Sequence[float] | None = None,
        *,
        chunk: int = 4096,
    ) -> torch.Tensor:
        """Free-orientation EEG lead field, shape ``(n_sensors, n_sources, 3)``.

        Units V/(A*m).  ``source_pos`` and ``sensor_pos`` are in metres in
        :attr:`frame`; sensors are projected radially onto the scalp sphere
        (their radial deviation is reported as a bias term by
        :meth:`lead_field`).
        """
        sig = list(sigmas) if sigmas is not None else list(self.conductivity.means())
        c = torch.tensor(self.center, dtype=torch.float64, device=source_pos.device)
        rq = source_pos.to(torch.float64) - c
        re = sensor_pos.to(torch.float64) - c

        b = rq.norm(dim=-1)
        if bool((b >= self.radii[0]).any()):
            raise ObservationRefusal(
                code="R01",
                message=(
                    "dipole outside the innermost (brain) layer of the spherical "
                    f"head model: max |r0| = {float(b.max()):.4f} m >= "
                    f"r_brain = {self.radii[0]:.4f} m"
                ),
                remedy="move the source inside the brain compartment or use a "
                "BEM/FEM head model whose geometry contains it",
            )
        e_hat = re / re.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        r0_hat = rq / b.clamp_min(1e-12).unsqueeze(-1)

        n_max = self._auto_n_terms(float(b.max()))
        n_idx = torch.arange(1, n_max + 1, dtype=torch.float64, device=source_pos.device)

        out = torch.zeros(
            (re.shape[0], rq.shape[0], 3), dtype=torch.float64, device=source_pos.device
        )
        for s0 in range(0, rq.shape[0], chunk):
            s1 = min(s0 + chunk, rq.shape[0])
            bb = b[s0:s1]
            D = self._radial_factor(bb, sig, n_max)  # (n_max, n_src)
            x = e_hat @ r0_hat[s0:s1].T  # (n_sens, n_src)
            P, dP = legendre_p_and_dp(x, n_max)  # (n_max+1, n_sens, n_src)
            P = P[1:]
            dP = dP[1:]
            Dx = D.unsqueeze(1)  # (n_max, 1, n_src)
            nn = n_idx.reshape(-1, 1, 1)
            # radial coefficient:  sum_n D_n * n * P_n(x)
            cr = (Dx * nn * P).sum(0)  # (n_sens, n_src)
            # tangential coefficient: sum_n D_n * P_n'(x)
            ct = (Dx * dP).sum(0)  # (n_sens, n_src)
            # V = cr * q_r + ct * (q.e - q_r x)   with q_r = q . r0_hat
            #   = q . [ (cr - ct*x) r0_hat + ct * e_hat ]
            r0h = r0_hat[s0:s1]  # (n_src, 3)
            out[:, s0:s1, :] = (cr - ct * x).unsqueeze(-1) * r0h.unsqueeze(0) + ct.unsqueeze(
                -1
            ) * e_hat.unsqueeze(1)
        return out

    def lead_field(
        self,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        *,
        sensor_names: Sequence[str] | None = None,
        sigmas: Sequence[float] | None = None,
        n_conductivity_draws: int = 0,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> LeadField:
        """Full EEG lead field with an uncertainty ledger.

        ``n_conductivity_draws > 0`` runs a Monte-Carlo sweep of the conductivity
        prior and reports its spread as the ``parameter_posterior`` variance
        component, in ``(V/(A*m))**2``.
        """
        L = self.potential(source_pos, sensor_pos, sigmas)

        param_var: float | str = UNKNOWN
        if n_conductivity_draws > 0:
            acc = []
            for k in range(n_conductivity_draws):
                s = self.conductivity.sample(seed=seed + 1000 * k)
                acc.append(self.potential(source_pos, sensor_pos, s))
            stack = torch.stack(acc)
            param_var = float(stack.var(dim=0, unbiased=True).mean())

        # radial deviation of the supplied sensors from the modelled scalp
        c = torch.tensor(self.center, dtype=torch.float64, device=sensor_pos.device)
        dev = (sensor_pos.to(torch.float64) - c).norm(dim=-1) - self.R
        dev_max = float(dev.abs().max()) if dev.numel() else 0.0

        bias = [
            BiasTerm(
                name="spherical_geometry_discrepancy",
                interval=(-0.30, 0.30),
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound="published sphere-vs-BEM/FEM comparisons report "
                "relative topography errors of order 10-30 % for realistic adult "
                "heads (Vatta et al. 2010; Vorwerk et al. 2014)",
                note="relative multiplicative bias; a sphere is a geometry model, "
                "not the subject's head",
            ),
            BiasTerm(
                name="sensor_radial_projection",
                interval=(-dev_max, dev_max),
                status="design_estimable",
                units="m",
                estimator="difference between the digitised electrode radius and "
                "the fitted scalp sphere radius, measured per electrode",
            ),
        ]
        if self.conductivity.csf is None:
            bias.append(self.conductivity.csf_omission_bias())

        ledger = UncertaintyLedger(
            variance=VarianceDecomposition(
                measurement=0.0,
                within_session=0.0,
                between_session=UNKNOWN,
                parameter_posterior=param_var,
                model_class=UNKNOWN,
                numerical=float(self._truncation_variance(source_pos)),
                units="V^2/(A*m)^2",
            ),
            bias=tuple(bias),
            model_discrepancy=UNKNOWN,
            model_discrepancy_flag=True,
            validity_domain={
                "geometry": "concentric spheres",
                "layers": self.conductivity.layer_names,
                "radii_m": self.radii,
                "source_radius_max_m": self.radii[0],
                "quasi_static": True,
                "frequency_range_hz": (0.0, 1000.0),
            },
            provenance=Provenance(
                operator="SphericalHeadModel",
                frames=(self.frame,),
                references=(
                    "de Munck & Peters 1993 (multilayer sphere)",
                    self.conductivity.reference,
                ),
                seed=seed,
                extras={
                    "n_terms": self._auto_n_terms(
                        float((source_pos.to(torch.float64) - c).norm(dim=-1).max())
                    ),
                    "sigmas": list(sigmas) if sigmas is not None else list(self.conductivity.means()),
                },
            ),
            notes=(
                "model_discrepancy_flag is True by construction: a concentric "
                "sphere is never the subject's head geometry.",
            ),
        )
        return LeadField(
            matrix=L.to(dtype),
            sensor_units=VOLT,
            source_positions=source_pos,
            sensor_positions=sensor_pos,
            frame=self.frame,
            modality="eeg",
            sensor_names=tuple(sensor_names or ()),
            reference="infinity",
            orientation="free",
            ledger=ledger,
            meta={"head_model": "analytic_multilayer_sphere"},
        )

    def _truncation_variance(self, source_pos: torch.Tensor) -> float:
        """Series-truncation error as a numerical variance component."""
        c = torch.tensor(self.center, dtype=torch.float64, device=source_pos.device)
        b = (source_pos.to(torch.float64) - c).norm(dim=-1)
        ratio = float((b / self.radii[0]).max().clamp(max=0.999))
        n = self._auto_n_terms(float(b.max()))
        return float(ratio ** (2 * n))


# ==========================================================================
# MEG: Sarvas
# ==========================================================================


def sarvas_meg(
    source_pos: torch.Tensor,
    sensor_pos: torch.Tensor,
    sensor_normal: torch.Tensor,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Sarvas (1987) magnetometer lead field, shape ``(n_sensors, n_sources, 3)``.

    Units T/(A*m).  The result is independent of the conductivity profile, which
    is the defining property of a spherically symmetric conductor: MEG therefore
    carries *no* skull-conductivity uncertainty but is blind to radial sources.
    """
    c = torch.as_tensor(center, dtype=torch.float64, device=source_pos.device)
    r0 = source_pos.to(torch.float64) - c  # (S, 3)
    r = sensor_pos.to(torch.float64) - c  # (E, 3)
    nrm = sensor_normal.to(torch.float64)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    a_vec = r.unsqueeze(1) - r0.unsqueeze(0)  # (E, S, 3)
    a = a_vec.norm(dim=-1)  # (E, S)
    rn = r.norm(dim=-1).unsqueeze(1)  # (E, 1)
    r_dot_r0 = (r.unsqueeze(1) * r0.unsqueeze(0)).sum(-1)  # (E, S)
    a_dot_r = (a_vec * r.unsqueeze(1)).sum(-1)  # (E, S)

    F = a * (rn * a + rn * rn - r_dot_r0)  # (E, S)
    gradF = (
        (a * a / rn + a_dot_r / a + 2.0 * a + 2.0 * rn).unsqueeze(-1) * r.unsqueeze(1)
        - (a + 2.0 * rn + a_dot_r / a).unsqueeze(-1) * r0.unsqueeze(0)
    )  # (E, S, 3)

    # B = mu0/(4 pi F^2) [ F (q x r0) - ((q x r0).r) gradF ]
    # Write as a linear map on q:  B . n = sum_k G_k q_k
    # q x r0 = -r0 x q = -[r0]_x q ; with [r0]_x the cross-product matrix.
    S = r0.shape[0]
    E = r.shape[0]
    eye = torch.eye(3, dtype=torch.float64, device=source_pos.device)
    out = torch.zeros((E, S, 3), dtype=torch.float64, device=source_pos.device)
    for k in range(3):
        ek = eye[k].expand(S, 3)
        qxr0 = torch.cross(ek, r0, dim=-1)  # (S, 3)
        term = F.unsqueeze(-1) * qxr0.unsqueeze(0) - (
            (qxr0.unsqueeze(0) * r.unsqueeze(1)).sum(-1, keepdim=True) * gradF
        )
        B = (MU0 / (4.0 * math.pi)) * term / (F * F).unsqueeze(-1)
        out[:, :, k] = (B * nrm.unsqueeze(1)).sum(-1)
    return out


def meg_lead_field(
    source_pos: torch.Tensor,
    sensor_pos: torch.Tensor,
    sensor_normal: torch.Tensor,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    frame: str = "head_sphere",
    sensor_names: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float32,
) -> LeadField:
    """MEG lead field with the ledger that MEG actually earns."""
    L = sarvas_meg(source_pos, sensor_pos, sensor_normal, center)
    ledger = UncertaintyLedger(
        variance=VarianceDecomposition(
            measurement=0.0,
            within_session=0.0,
            between_session=UNKNOWN,
            parameter_posterior=0.0,
            model_class=UNKNOWN,
            numerical=0.0,
            units="T^2/(A*m)^2",
        ),
        bias=(
            BiasTerm(
                name="radial_source_blindness",
                interval=(-1.0, 0.0),
                status="design_estimable",
                units=DIMENSIONLESS,
                estimator="the radial component of the lead field is identically "
                "zero for a spherically symmetric conductor; the null direction "
                "is computable per source position",
                note="not a nuisance to be reduced: a structural null space",
            ),
            BiasTerm(
                name="sphere_vs_realistic_conductor",
                interval=(-0.05, 0.05),
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound="MEG single-sphere vs BEM comparisons report a few "
                "percent topography difference (Tarkiainen et al. 2003)",
            ),
        ),
        model_discrepancy=UNKNOWN,
        model_discrepancy_flag=True,
        validity_domain={
            "geometry": "spherically symmetric conductor",
            "conductivity_independent": True,
            "sensor_type": "magnetometer (projection onto supplied normal)",
        },
        provenance=Provenance(
            operator="sarvas_meg",
            frames=(frame,),
            references=("Sarvas 1987, Phys. Med. Biol. 32:11-22",),
        ),
    )
    return LeadField(
        matrix=L.to(dtype),
        sensor_units=TESLA,
        source_positions=source_pos,
        sensor_positions=sensor_pos,
        frame=frame,
        modality="meg",
        sensor_names=tuple(sensor_names or ()),
        reference="none",
        orientation="free",
        ledger=ledger,
        meta={"head_model": "sarvas_sphere"},
    )


# ==========================================================================
# realistic heads: wrap MNE, do not reimplement
# ==========================================================================


class BEMLeadField:
    """Thin wrapper around ``mne`` BEM / forward machinery.

    We wrap rather than reimplement, and then *validate*: the accompanying test
    ``tests/observe/test_leadfield_mne.py::test_bem_wrapper_reproduces_mne``
    asserts that :meth:`from_mne_forward` reproduces MNE's own gain matrix to
    machine precision, and the sphere test asserts that the analytic solver in
    this module agrees with MNE's Berg-approximated multilayer sphere.
    """

    def __init__(self) -> None:  # pragma: no cover - namespace class
        raise TypeError("BEMLeadField is a namespace of constructors")

    # -- constructors -------------------------------------------------------
    @staticmethod
    def from_mne_forward(fwd: Any, *, frame: str = "subject_head_RAS") -> LeadField:
        """Convert an ``mne.Forward`` into a :class:`LeadField` with a ledger.

        No numerical work is done here beyond a units-preserving reshape, which
        is exactly why the reproduction test can assert bit-level agreement.
        """
        import numpy as np

        gain = np.asarray(fwd["sol"]["data"], dtype=np.float64)
        n_sens = gain.shape[0]
        fixed = bool(fwd["source_ori"] == 1)  # FIFF.FIFFV_MNE_FIXED_ORI
        src_pos = np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]])
        chs = fwd["info"]["chs"]
        sensor_pos = np.array([c["loc"][:3] for c in chs], dtype=np.float64)
        names = tuple(fwd["sol"]["row_names"])

        kinds = {c["kind"] for c in chs}
        # 2 = FIFFV_EEG_CH, 1 = FIFFV_MEG_CH
        modality: Literal["eeg", "meg"] = "eeg" if kinds == {2} else "meg"
        units = VOLT if modality == "eeg" else TESLA
        if len(kinds) > 1:
            raise ObservationRefusal(
                code="R01",
                message="mixed EEG/MEG forward passed to BEMLeadField; the two "
                "have different units and must not share one matrix",
                remedy="use fwd.pick_types(meg=..., eeg=...) and build one "
                "LeadField per modality",
                offending_object=sorted(kinds),
            )

        m = torch.from_numpy(gain)
        if not fixed:
            m = m.reshape(n_sens, -1, 3)

        ledger = UncertaintyLedger(
            variance=VarianceDecomposition(
                measurement=0.0,
                within_session=0.0,
                between_session=UNKNOWN,
                parameter_posterior=UNKNOWN,
                model_class=UNKNOWN,
                numerical=UNKNOWN,
                units="V^2/(A*m)^2",
            ),
            bias=(
                BiasTerm(
                    name="bem_conductivity",
                    interval=(-0.25, 0.25),
                    status="prior_specified_sensitivity",
                    units=DIMENSIONLESS,
                    sensitivity_grid=(-0.25, -0.1, 0.0, 0.1, 0.25),
                    note="skull conductivity is not measured per subject; sweep "
                    "the McCann 2019 range and propagate the resulting spread",
                ),
                BiasTerm(
                    name="coregistration",
                    interval=(-0.005, 0.005),
                    status="design_estimable",
                    units="m",
                    estimator="fiducial + head-shape ICP residual reported by the "
                    "coregistration (mne.coreg), per session",
                ),
                BiasTerm(
                    name="segmentation_surface_error",
                    interval=(-0.002, 0.002),
                    status="externally_bounded",
                    units="m",
                    external_bound="FreeSurfer/watershed BEM surface accuracy "
                    "against manual delineation in published validation studies",
                ),
            ),
            model_discrepancy=UNKNOWN,
            model_discrepancy_flag=True,
            validity_domain={
                "geometry": "subject BEM surfaces",
                "quasi_static": True,
                "source_space": "supplied with the forward",
            },
            provenance=Provenance(
                operator="BEMLeadField.from_mne_forward",
                frames=(frame,),
                references=("mne-python forward solution (linear collocation BEM)",),
                extras={"mne_source_ori": int(fwd["source_ori"])},
            ),
        )
        return LeadField(
            matrix=m,
            sensor_units=units,
            source_positions=torch.from_numpy(src_pos),
            sensor_positions=torch.from_numpy(sensor_pos),
            frame=frame,
            modality=modality,
            sensor_names=names,
            reference="average" if modality == "eeg" else "none",
            orientation="fixed" if fixed else "free",
            ledger=ledger,
            meta={"head_model": "mne_bem"},
        )

    @staticmethod
    def from_subject(
        subject: str,
        subjects_dir: str,
        info: Any,
        *,
        trans: Any = "fsaverage",
        src: Any = None,
        conductivity: Sequence[float] = (0.3, 0.006, 0.3),
        ico: int = 4,
        spacing: str = "oct5",
        eeg: bool = True,
        meg: bool = False,
        frame: str = "subject_head_RAS",
    ) -> LeadField:
        """Build a BEM forward for a real subject head with ``mne``."""
        import mne

        if src is None:
            src = mne.setup_source_space(
                subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False
            )
        model = mne.make_bem_model(
            subject=subject,
            ico=ico,
            conductivity=tuple(conductivity),
            subjects_dir=subjects_dir,
        )
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem, eeg=eeg, meg=meg, verbose="error"
        )
        return BEMLeadField.from_mne_forward(fwd, frame=frame)

    @staticmethod
    def mne_sphere_reference(
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        *,
        radii: Sequence[float],
        sigmas: Sequence[float],
        center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> torch.Tensor:
        """MNE's own multilayer-sphere EEG forward, for validating ours.

        Returns ``(n_sensors, n_sources, 3)`` in V/(A*m).  MNE uses the Berg
        three-term approximation to the multilayer series, so agreement with the
        exact solver in this module is expected at the Berg-approximation level
        (order 1 %), not at machine precision -- the test states that explicitly.
        """
        import numpy as np
        import mne

        R = float(radii[-1])
        sphere = mne.make_sphere_model(
            r0=tuple(float(x) for x in center),
            head_radius=R,
            relative_radii=tuple(float(r) / R for r in radii),
            sigmas=tuple(float(s) for s in sigmas),
            verbose="error",
        )
        sp = np.asarray(sensor_pos.detach().cpu(), dtype=np.float64)
        names = [f"E{i + 1:03d}" for i in range(sp.shape[0])]
        info = mne.create_info(names, sfreq=1000.0, ch_types="eeg")
        montage = mne.channels.make_dig_montage(
            ch_pos={n: p for n, p in zip(names, sp)}, coord_frame="head"
        )
        info.set_montage(montage)

        qp = np.asarray(source_pos.detach().cpu(), dtype=np.float64)
        n_src = qp.shape[0]
        pos = dict(
            rr=qp,
            nn=np.tile(np.array([[0.0, 0.0, 1.0]]), (n_src, 1)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fwd = mne.make_forward_solution(
                info,
                trans=None,
                src=mne.setup_volume_source_space(pos=pos, sphere_units="m", verbose="error"),
                bem=sphere,
                eeg=True,
                meg=False,
                verbose="error",
            )
        g = np.asarray(fwd["sol"]["data"], dtype=np.float64)
        return torch.from_numpy(g.reshape(sp.shape[0], n_src, 3))


# ==========================================================================
# reference schemes
# ==========================================================================


@dataclass(frozen=True)
class ReferenceOperator:
    """EEG reference as an explicit linear operator on sensors.

    The reference is a *modelling choice with a bias*, not a preprocessing
    detail: an average reference on a non-closed sensor surface leaves a
    spatially structured residual, and a linked-mastoid reference injects the
    mastoid's own source activity into every channel.
    """

    kind: Literal["infinity", "average", "linked_mastoid", "single", "rest"]
    indices: tuple[int, ...] = ()
    matrix: torch.Tensor | None = None

    @staticmethod
    def average(n: int, dtype: torch.dtype = torch.float32) -> "ReferenceOperator":
        M = torch.eye(n, dtype=dtype) - torch.full((n, n), 1.0 / n, dtype=dtype)
        return ReferenceOperator(kind="average", matrix=M)

    @staticmethod
    def linked_mastoid(
        n: int, mastoids: Sequence[int], dtype: torch.dtype = torch.float32
    ) -> "ReferenceOperator":
        M = torch.eye(n, dtype=dtype)
        w = 1.0 / len(mastoids)
        for m in mastoids:
            M[:, m] -= w
        return ReferenceOperator(kind="linked_mastoid", indices=tuple(mastoids), matrix=M)

    @staticmethod
    def single(n: int, index: int, dtype: torch.dtype = torch.float32) -> "ReferenceOperator":
        M = torch.eye(n, dtype=dtype)
        M[:, index] -= 1.0
        return ReferenceOperator(kind="single", indices=(index,), matrix=M)

    @staticmethod
    def rest(
        lead_field: LeadField, *, rcond: float = 1e-6, dtype: torch.dtype = torch.float32
    ) -> "ReferenceOperator":
        """Reference Electrode Standardization Technique (Yao 2001).

        ``V_inf = L pinv(C L) V_avg``.  REST is *not* reference-free: it is only
        as good as the lead field, so the operator carries its lead field's
        provenance rather than pretending the reference problem was solved.
        """
        L = lead_field.as_matrix().to(torch.float64)
        n = L.shape[0]
        C = torch.eye(n, dtype=torch.float64) - torch.full((n, n), 1.0 / n, dtype=torch.float64)
        Lar = C @ L
        M = L @ torch.linalg.pinv(Lar, rcond=rcond)
        return ReferenceOperator(kind="rest", matrix=M.to(dtype))

    @staticmethod
    def infinity(n: int, dtype: torch.dtype = torch.float32) -> "ReferenceOperator":
        return ReferenceOperator(kind="infinity", matrix=torch.eye(n, dtype=dtype))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to sensor-space data ``(n_sensors, ...)`` or to a lead field."""
        if self.matrix is None:
            return x
        M = self.matrix.to(x.dtype)
        return torch.tensordot(M, x, dims=([1], [0]))

    def bias_term(self) -> BiasTerm:
        if self.kind == "average":
            return BiasTerm(
                name="reference_average_incomplete_surface",
                interval=(-1.0, 1.0),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
                note="an average reference over an open, non-uniformly sampled "
                "scalp surface does not equal the potential at infinity; the "
                "residual is a spatially structured, montage-dependent offset "
                "expressed here as a relative sensitivity range",
            )
        if self.kind == "linked_mastoid":
            return BiasTerm(
                name="reference_mastoid_activity",
                interval=(-1.0, 1.0),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
                note="mastoid electrodes carry brain signal; the reference is a "
                "source mixture, not a silent point",
            )
        if self.kind == "single":
            return BiasTerm(
                name="reference_single_electrode_activity",
                interval=(-1.0, 1.0),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-1.0, 0.0, 1.0),
            )
        if self.kind == "rest":
            return BiasTerm(
                name="reference_rest_leadfield_error",
                interval=(-0.30, 0.30),
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound="REST accuracy is bounded by head-model error; "
                "simulation studies report 10-30 % relative error for realistic "
                "head-model mismatch (Yao 2001; Qin et al. 2010)",
            )
        return BiasTerm(
            name="reference_infinity_assumed",
            interval=(0.0, 0.0),
            status="design_estimable",
            units=DIMENSIONLESS,
            estimator="no reference transformation applied; the forward model "
            "already returns potentials relative to infinity",
        )


# ==========================================================================
# impedance
# ==========================================================================


@dataclass(frozen=True)
class ElectrodeImpedance:
    """Electrode-tissue impedance: a gain *and* a noise source.

    Voltage-divider attenuation with the amplifier input impedance gives
    ``g_e = Z_in / (Z_in + Z_e)``; the electrode's real part contributes Johnson
    noise ``4 k_B T R df``.  Both are ordinary physics that a "preprocessed"
    pipeline silently drops.
    """

    z_electrode: torch.Tensor  # ohm, per channel
    z_input: float = 1e9  # ohm, modern high-impedance amplifier
    temperature_k: float = 305.0
    bandwidth_hz: float = 100.0

    @property
    def gain(self) -> torch.Tensor:
        return self.z_input / (self.z_input + self.z_electrode)

    @property
    def thermal_noise_variance(self) -> torch.Tensor:
        """Johnson-Nyquist variance in V**2 per channel."""
        return 4.0 * BOLTZMANN * self.temperature_k * self.z_electrode * self.bandwidth_hz

    def imbalance_bias(self) -> BiasTerm:
        """Inter-channel impedance imbalance -> common-mode rejection failure."""
        z = self.z_electrode
        spread = float((z.max() - z.min()) / z.mean().clamp_min(1e-12)) if z.numel() else 0.0
        return BiasTerm(
            name="impedance_imbalance_cmrr",
            interval=(-spread, spread),
            status="design_estimable",
            units=DIMENSIONLESS,
            estimator="per-channel impedance check recorded at session start and "
            "end; the imbalance is measured, not assumed",
        )


# ==========================================================================
# electrode position uncertainty -> lead-field uncertainty
# ==========================================================================


def _uncertainty_backend() -> Any | None:
    """Use agent D's covariance propagation when it lands; else propagate here."""
    try:  # pragma: no cover - depends on a parallel agent
        from scwbd.transforms import uncertainty as _u  # type: ignore

        return _u
    except Exception:
        return None


@dataclass(frozen=True)
class ElectrodePositionUncertainty:
    """Propagate electrode-position covariance into lead-field variance.

    ``position_cov`` is the full ``(3 n_e, 3 n_e)`` covariance in the sensor
    frame.  Cross-electrode blocks are *retained*: a cap rotation or a
    digitiser-to-head coregistration error moves every electrode together, and
    dropping ``J_x Sigma_xc J_c^T`` is a bug (ARCHITECTURE.md Sec. 3, T5).
    """

    position_cov: torch.Tensor
    step: float = 5e-4  # metres, central-difference step

    @staticmethod
    def isotropic(n_sensors: int, sd_m: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return (sd_m**2) * torch.eye(3 * n_sensors, dtype=dtype)

    @staticmethod
    def with_common_mode(
        sensor_pos: torch.Tensor,
        *,
        independent_sd_m: float,
        rotation_sd_rad: float,
        translation_sd_m: float = 0.0,
        center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> torch.Tensor:
        """Covariance with independent jitter **plus** a shared cap pose error.

        The shared term is the physically important one: it is fully correlated
        across electrodes, so it does not average away with more channels.
        """
        p = sensor_pos.to(torch.float64)
        n = p.shape[0]
        c = torch.as_tensor(center, dtype=torch.float64)
        rel = p - c
        # d(pos)/d(small rotation w) = -[rel]_x  (per electrode, 3x3)
        J = torch.zeros((3 * n, 6), dtype=torch.float64)
        for i in range(n):
            x, y, z = rel[i]
            skew = torch.tensor(
                [[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]], dtype=torch.float64
            )
            J[3 * i : 3 * i + 3, 0:3] = skew
            J[3 * i : 3 * i + 3, 3:6] = torch.eye(3, dtype=torch.float64)
        pose_cov = torch.diag(
            torch.tensor(
                [rotation_sd_rad**2] * 3 + [translation_sd_m**2] * 3, dtype=torch.float64
            )
        )
        return (independent_sd_m**2) * torch.eye(3 * n, dtype=torch.float64) + J @ pose_cov @ J.T

    def jacobian(
        self,
        head: SphericalHeadModel,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        source_moment: torch.Tensor,
    ) -> torch.Tensor:
        """``d y_e / d p_e`` for ``y = L(p) q``, shape ``(n_sensors, 3 n_sensors)``.

        Central differences on the analytic solver.  Only the diagonal blocks are
        non-zero (electrode ``e``'s position affects only channel ``e``), but the
        full matrix is returned so that a *correlated* position covariance still
        produces correlated sensor variance.
        """
        p = sensor_pos.to(torch.float64).clone()
        n = p.shape[0]
        q = source_moment.to(torch.float64)
        J = torch.zeros((n, 3 * n), dtype=torch.float64)
        for k in range(3):
            for sgn in (+1, -1):
                pp = p.clone()
                pp[:, k] += sgn * self.step
                L = head.potential(source_pos, pp)  # (E, S, 3)
                y = torch.einsum("esk,sk->e", L, q)
                for e in range(n):
                    J[e, 3 * e + k] += sgn * y[e] / (2.0 * self.step)
        return J

    def conductivity_jacobian(
        self,
        head: SphericalHeadModel,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        source_moment: torch.Tensor,
        *,
        rel_step: float = 1e-3,
    ) -> torch.Tensor:
        """``d y / d sigma_layer``, shape ``(n_sensors, n_layers)``.

        Central differences on the layer conductivities.  Conductivity error is
        *shared across every sample in a session*, which is exactly the case
        thesis T5 warns about: it does not average away and its cross term with
        the electrode positions must be retained.
        """
        sig0 = list(head.conductivity.means())
        q = source_moment.to(torch.float64)
        n_e = sensor_pos.shape[0]
        J = torch.zeros((n_e, len(sig0)), dtype=torch.float64)
        for k, s0 in enumerate(sig0):
            h = max(abs(s0) * rel_step, 1e-12)
            ys = []
            for sgn in (+1, -1):
                sig = list(sig0)
                sig[k] = s0 + sgn * h
                L = head.potential(source_pos, sensor_pos, sig)
                ys.append(torch.einsum("esk,sk->e", L, q))
            J[:, k] = (ys[0] - ys[1]) / (2.0 * h)
        return J

    def propagate(
        self,
        head: SphericalHeadModel,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        source_moment: torch.Tensor,
        *,
        conductivity_cov: torch.Tensor | None = None,
        cross_cov: torch.Tensor | None = None,
        include_cross: bool = True,
    ) -> Any:
        """Joint T5 propagation of electrode-position **and** conductivity error.

        The physics Jacobians are computed here; the propagation algebra is
        agent D's (``scwbd.transforms.uncertainty.propagate_first_order``), so
        the mandatory cross term ``J_x Sigma_xc J_c^T`` is applied by the module
        that owns it rather than re-derived.  The lead field is handed over as an
        exactly linear surrogate built from the measured Jacobians, so agent D
        recovers them without needing this solver to be autograd-differentiable.

        Returns their ``Propagated`` dataclass (value, cov, bias, terms).
        """
        backend = _uncertainty_backend()
        if backend is None or not hasattr(backend, "propagate_first_order"):
            raise ObservationRefusal(
                code="R01",
                message="scwbd.transforms.uncertainty is unavailable, so the T5 "
                "cross-covariance propagation cannot be performed",
                remedy="install/repair the transforms package; dropping the cross "
                "term is a bug, not a fallback (ARCHITECTURE.md Sec. 3)",
            )
        Jp = self.jacobian(head, source_pos, sensor_pos, source_moment)
        Jc = self.conductivity_jacobian(head, source_pos, sensor_pos, source_moment)
        L0 = head.potential(source_pos, sensor_pos)
        y0 = torch.einsum("esk,sk->e", L0, source_moment.to(torch.float64))

        p0 = sensor_pos.to(torch.float64).reshape(-1)
        c0 = torch.tensor(head.conductivity.means(), dtype=torch.float64)

        def surrogate(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return y0 + Jp @ (p - p0) + Jc @ (c - c0)

        n_c = c0.numel()
        Sc = (
            conductivity_cov
            if conductivity_cov is not None
            else torch.diag(
                torch.tensor(
                    [p.sd**2 for p in head.conductivity.priors], dtype=torch.float64
                )
            )
        )
        Sxc = cross_cov if cross_cov is not None else torch.zeros(
            (p0.numel(), n_c), dtype=torch.float64
        )
        return backend.propagate_first_order(
            surrogate,
            p0,
            c0,
            Sx=self.position_cov.to(torch.float64),
            Sc=Sc,
            Sxc=Sxc,
            include_cross=include_cross,
        )

    def sensor_covariance(
        self,
        head: SphericalHeadModel,
        source_pos: torch.Tensor,
        sensor_pos: torch.Tensor,
        source_moment: torch.Tensor,
    ) -> torch.Tensor:
        """``Sigma_y = J Sigma_p J^T`` in V**2 from the electrode positions alone.

        Cross terms *between electrodes* are retained (a shared cap-pose error
        does not average away).  For the joint position + conductivity ledger,
        including the ``J_x Sigma_xc J_c^T`` term of T5, use :meth:`propagate`.
        """
        J = self.jacobian(head, source_pos, sensor_pos, source_moment)
        return J @ self.position_cov.to(torch.float64) @ J.T

    def bias_term(self) -> BiasTerm:
        sd = float(torch.diagonal(self.position_cov).mean().sqrt())
        return BiasTerm(
            name="electrode_position",
            interval=(-3.0 * sd, 3.0 * sd),
            status="design_estimable",
            units="m",
            estimator="repeated electrode digitisation within and across "
            "sessions; the 3-sigma envelope of the measured digitiser "
            "reproducibility",
        )
