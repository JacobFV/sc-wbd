"""Source estimation, presented as the non-unique problem it is.

body.tex Table ``tab:modalities`` lists "source non-identifiability" as a
dominant EEG/MEG term.  body.tex Sec. 2.8 states that an electrode "does not
become a cortical MNI source coordinate without an explicit head model, inverse
operator, and atlas warp."  Sec. 2.6 requires that a prolongation from a coarse
to a fine support "may produce a distribution rather than a unique fine state",
and refusal **R02** rejects prolongation without a declared restriction partner
and tested coverage.

This module therefore never returns "the source".  It returns an
:class:`InverseSolutionSet`:

* the minimum-norm particular solution for a stated regularisation,
* an explicit orthonormal basis for the **null space** of the lead field, so
  that ``x = x_hat + N alpha`` enumerates the data-equivalent solutions,
* the **resolution matrix** ``R = K L`` with its point-spread (columns) and
  cross-talk (rows) functions, plus localisation error and spatial dispersion,
* a **depth-bias** analysis, and
* a **regularisation-sensitivity** family over lambda.

:meth:`InverseSolutionSet.point_estimate` refuses unless the caller passes
``acknowledge_non_uniqueness=True``, which is a small piece of friction that
makes the epistemic claim explicit at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch

from .base import (
    DIMENSIONLESS,
    UNKNOWN,
    BiasTerm,
    ObservationRefusal,
    Provenance,
    UncertaintyLedger,
    VarianceDecomposition,
)
from .leadfield import LeadField

__all__ = [
    "InverseSolutionSet",
    "ResolutionAnalysis",
    "minimum_norm_operator",
    "dspm_operator",
    "sloreta_operator",
    "lcmv_beamformer",
    "solve_inverse",
    "regularization_sweep",
]


# ==========================================================================
# kernels
# ==========================================================================


def _whitener(noise_cov: torch.Tensor, rcond: float = 1e-12) -> torch.Tensor:
    e, U = torch.linalg.eigh(noise_cov.to(torch.float64))
    e = e.clamp_min(e.max() * rcond)
    return U @ torch.diag(e.rsqrt()) @ U.T


def minimum_norm_operator(
    L: torch.Tensor,
    noise_cov: torch.Tensor,
    *,
    lambda2: float = 1.0 / 9.0,
    source_cov: torch.Tensor | None = None,
    depth_weighting: float = 0.0,
) -> torch.Tensor:
    """``K = R L^T (L R L^T + lambda2 tr(LRL^T)/tr(C) C)^{-1}``.

    ``lambda2`` follows the MNE convention: ``1/SNR**2`` with a default SNR of 3.
    ``depth_weighting`` is the exponent of the classic depth compensation
    ``R = diag(||L_i||^{-2 gamma})``; it *trades* depth bias for spatial
    resolution, which is why it is a parameter with a reported consequence
    rather than a hidden default.
    """
    L = L.to(torch.float64)
    C = noise_cov.to(torch.float64)
    n_src = L.shape[1]
    if source_cov is None:
        if depth_weighting > 0:
            w = L.pow(2).sum(0).clamp_min(1e-30) ** (-depth_weighting)
            R = torch.diag(w / w.mean())
        else:
            R = torch.eye(n_src, dtype=torch.float64)
    else:
        R = source_cov.to(torch.float64)

    LRLt = L @ R @ L.T
    scale = float(torch.trace(LRLt) / torch.trace(C).clamp_min(1e-30))
    G = LRLt + lambda2 * scale * C
    return R @ L.T @ torch.linalg.pinv(G)


def dspm_operator(
    L: torch.Tensor, noise_cov: torch.Tensor, *, lambda2: float = 1.0 / 9.0, **kw: Any
) -> torch.Tensor:
    """dSPM: MNE kernel row-normalised by projected noise standard deviation."""
    K = minimum_norm_operator(L, noise_cov, lambda2=lambda2, **kw)
    C = noise_cov.to(torch.float64)
    noise_sd = torch.sqrt(torch.einsum("ij,jk,ik->i", K, C, K).clamp_min(1e-300))
    return K / noise_sd.unsqueeze(-1)


def sloreta_operator(
    L: torch.Tensor, noise_cov: torch.Tensor, *, lambda2: float = 1.0 / 9.0, **kw: Any
) -> torch.Tensor:
    """sLORETA: MNE kernel normalised by the resolution matrix diagonal.

    Zero localisation error for a single point source *in the noiseless limit*
    -- a property that says nothing about resolution when two sources are
    present, which is why this module always ships the resolution analysis
    alongside.
    """
    K = minimum_norm_operator(L, noise_cov, lambda2=lambda2, **kw)
    Rm = K @ L.to(torch.float64)
    d = torch.diagonal(Rm).clamp_min(1e-300).sqrt()
    return K / d.unsqueeze(-1)


def lcmv_beamformer(
    L: torch.Tensor,
    data_cov: torch.Tensor,
    noise_cov: torch.Tensor,
    *,
    reg: float = 0.05,
    unit_noise_gain: bool = True,
) -> torch.Tensor:
    """Scalar LCMV beamformer, one row per source.

    Beamformers fail exactly where correlated sources exist -- they are not a
    cure for non-uniqueness, they are a different set of assumptions.  The
    accompanying test demonstrates the failure numerically.
    """
    L = L.to(torch.float64)
    C = data_cov.to(torch.float64)
    C = C + reg * float(torch.trace(C) / C.shape[0]) * torch.eye(
        C.shape[0], dtype=torch.float64
    )
    Cinv = torch.linalg.pinv(C)
    num = Cinv @ L  # (n_sens, n_src)
    den = (L * num).sum(0).clamp_min(1e-300)  # (n_src,)
    W = (num / den).T  # (n_src, n_sens)
    if unit_noise_gain:
        ng = torch.sqrt(torch.einsum("ij,jk,ik->i", W, noise_cov.to(torch.float64), W).clamp_min(1e-300))
        W = W / ng.unsqueeze(-1)
    return W


# ==========================================================================
# resolution analysis
# ==========================================================================


@dataclass(frozen=True)
class ResolutionAnalysis:
    """``R = K L``: what the inverse operator actually reports about each source.

    ``psf[:, j]`` is the point-spread function of source ``j`` (how a unit
    source at ``j`` is smeared across the estimate); ``ctf[i, :]`` is the
    cross-talk function of estimate ``i`` (which true sources leak into it).
    """

    resolution: torch.Tensor
    source_positions: torch.Tensor

    @property
    def psf(self) -> torch.Tensor:
        return self.resolution

    @property
    def ctf(self) -> torch.Tensor:
        return self.resolution.T

    def peak_localization_error_m(self) -> torch.Tensor:
        """Distance between each source and the peak of its point spread."""
        p = self.source_positions.to(torch.float64)
        peak = self.resolution.abs().argmax(dim=0)
        return (p[peak] - p).norm(dim=-1)

    def spatial_dispersion_m(self) -> torch.Tensor:
        """RMS distance of the point spread from the true source location."""
        p = self.source_positions.to(torch.float64)
        w = self.resolution.abs() ** 2
        w = w / w.sum(0, keepdim=True).clamp_min(1e-300)
        d2 = torch.cdist(p, p) ** 2
        return (w * d2).sum(0).sqrt()

    def crosstalk_ratio(self) -> torch.Tensor:
        """Off-diagonal energy fraction of each estimate: 0 = perfect isolation."""
        R = self.resolution.abs() ** 2
        diag = torch.diagonal(R)
        total = R.sum(dim=1).clamp_min(1e-300)
        return 1.0 - diag / total

    def effective_rank(self, tol: float = 1e-6) -> int:
        s = torch.linalg.svdvals(self.resolution)
        return int((s > tol * s.max()).sum())


# ==========================================================================
# the solution set
# ==========================================================================


@dataclass(frozen=True)
class InverseSolutionSet:
    """A *set*, not a solution.  The type name is the API documentation."""

    particular: torch.Tensor
    """Minimum-norm particular solution, ``(n_sources, n_time)`` in A*m."""
    null_basis: torch.Tensor
    """``(n_sources, n_null)`` orthonormal basis of ``ker(L)``."""
    posterior_cov_diag: torch.Tensor
    """Diagonal of the Gaussian posterior covariance, per source."""
    resolution: ResolutionAnalysis
    lead_field: LeadField
    method: str
    lambda2: float
    depth_weighting: float
    noise_cov: torch.Tensor
    ledger: UncertaintyLedger
    lambda_family: Mapping[float, torch.Tensor] = field(default_factory=dict)

    # -- the non-uniqueness API --------------------------------------------
    @property
    def n_null(self) -> int:
        return int(self.null_basis.shape[1])

    def null_space_fraction(self) -> float:
        """Fraction of the source space the data say nothing about."""
        n_src = int(self.particular.shape[0])
        return self.n_null / max(n_src, 1)

    def admissible(
        self, alpha: torch.Tensor | None = None, *, seed: int | None = None, scale: float = 1.0
    ) -> torch.Tensor:
        """Another solution that fits the data **exactly as well**.

        ``x = particular + N alpha`` for any ``alpha``.  Used by the
        non-uniqueness test to construct two distinct sources with identical
        sensor data.
        """
        if self.n_null == 0:
            raise ObservationRefusal(
                code="R02",
                message="the lead field has trivial null space for this source "
                "space; admissible-set enumeration is not meaningful here",
                remedy="use a source space finer than the sensor count, or read "
                "the regularisation family instead",
            )
        if alpha is None:
            if seed is None:
                raise ValueError("supply alpha or a seed")
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            alpha = scale * torch.randn(self.n_null, generator=g, dtype=torch.float64)
        return self.particular + (
            self.null_basis.to(torch.float64) @ alpha.to(torch.float64)
        ).unsqueeze(-1)

    def sample_posterior(self, n: int, *, seed: int) -> torch.Tensor:
        """Draw ``n`` source configurations from the Gaussian posterior."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        sd = self.posterior_cov_diag.clamp_min(0).sqrt().unsqueeze(-1)
        eps = torch.randn(
            (n,) + tuple(self.particular.shape), generator=g, dtype=torch.float64
        )
        return self.particular.unsqueeze(0) + sd.unsqueeze(0) * eps

    def point_estimate(self, *, acknowledge_non_uniqueness: bool = False) -> torch.Tensor:
        """Refuses unless the caller states that the answer is not unique."""
        if not acknowledge_non_uniqueness:
            raise ObservationRefusal(
                code="R02",
                message=(
                    f"a single {self.method} solution was requested as 'the "
                    f"source'. The lead field has a {self.n_null}-dimensional "
                    f"null space ({100 * self.null_space_fraction():.1f} % of the "
                    "source space); infinitely many source configurations "
                    "reproduce this data exactly"
                ),
                remedy="use .admissible(), .sample_posterior(), or .resolution "
                "to report the set; if a point estimate is genuinely wanted for "
                "display, pass acknowledge_non_uniqueness=True and carry the "
                "resolution analysis alongside it",
                offending_object=self.method,
            )
        return self.particular

    def depth_bias(self) -> dict[str, torch.Tensor]:
        """The classic MNE depth bias: deep sources are localised too superficially.

        ``radial_shift_m`` is the signed radial displacement of each source's
        point-spread peak.  A systematically positive mean is the outward
        (pial-ward) bias; depth weighting is expected to shrink
        ``|radial_shift_m|`` while costing spatial resolution, and both numbers
        are returned so the trade is visible rather than asserted.
        """
        p = self.lead_field.source_positions.to(torch.float64)
        centre = p.mean(0)
        radius = (p - centre).norm(dim=-1)
        peak = self.resolution.resolution.abs().argmax(dim=0)
        return {
            "radius_m": radius,
            "depth_m": radius,  # distance from centre: larger = more superficial
            "radial_shift_m": radius[peak] - radius,
            "mean_abs_radial_shift_m": (radius[peak] - radius).abs().mean(),
            "resolution_diagonal": torch.diagonal(self.resolution.resolution),
            "leadfield_norm": self.lead_field.as_matrix().to(torch.float64).norm(dim=0),
            "peak_localization_error_m": self.resolution.peak_localization_error_m(),
            "spatial_dispersion_m": self.resolution.spatial_dispersion_m(),
        }


# ==========================================================================
# solve
# ==========================================================================


_OPERATORS = {
    "MNE": minimum_norm_operator,
    "dSPM": dspm_operator,
    "sLORETA": sloreta_operator,
}


def solve_inverse(
    data: torch.Tensor,
    lead_field: LeadField,
    noise_cov: torch.Tensor,
    *,
    method: Literal["MNE", "dSPM", "sLORETA", "LCMV"] = "MNE",
    lambda2: float = 1.0 / 9.0,
    depth_weighting: float = 0.0,
    data_cov: torch.Tensor | None = None,
    null_tol: float = 1e-10,
    lambda_sweep: Sequence[float] = (1.0 / 81.0, 1.0 / 9.0, 1.0, 9.0),
) -> InverseSolutionSet:
    """Solve the inverse problem and return the admissible **set**.

    ``data`` is ``(n_sensors, n_time)`` in the lead field's sensor units.
    """
    L = lead_field.as_matrix().to(torch.float64)
    y = data.to(torch.float64)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if y.shape[0] != L.shape[0]:
        raise ObservationRefusal(
            code="R01",
            message=f"data has {y.shape[0]} channels, lead field has {L.shape[0]}",
            remedy="align the montage before inversion; channel order is part of "
            "the calibration chain, not a detail",
        )

    if method == "LCMV":
        dc = data_cov if data_cov is not None else (y @ y.T) / max(y.shape[1], 1)
        K = lcmv_beamformer(L, dc, noise_cov)
    else:
        K = _OPERATORS[method](
            L, noise_cov, lambda2=lambda2, depth_weighting=depth_weighting
        )

    x_hat = K @ y
    Rm = K @ L
    res = ResolutionAnalysis(resolution=Rm, source_positions=lead_field.source_positions)

    # explicit null space of L
    U, S, Vh = torch.linalg.svd(L, full_matrices=True)
    tol = null_tol * float(S.max()) if S.numel() else 0.0
    rank = int((S > tol).sum())
    null_basis = Vh[rank:].T.contiguous()

    post_diag = torch.diagonal(K @ noise_cov.to(torch.float64) @ K.T).clamp_min(0.0)

    family: dict[float, torch.Tensor] = {}
    if method != "LCMV":
        for lam in lambda_sweep:
            Kl = _OPERATORS[method](
                L, noise_cov, lambda2=float(lam), depth_weighting=depth_weighting
            )
            family[float(lam)] = Kl @ y

    ledger = _inverse_ledger(
        method=method,
        lambda2=lambda2,
        depth_weighting=depth_weighting,
        n_null=int(null_basis.shape[1]),
        n_src=int(L.shape[1]),
        rank=rank,
        res=res,
        family=family,
        lead_field=lead_field,
    )
    return InverseSolutionSet(
        particular=x_hat,
        null_basis=null_basis,
        posterior_cov_diag=post_diag,
        resolution=res,
        lead_field=lead_field,
        method=method,
        lambda2=lambda2,
        depth_weighting=depth_weighting,
        noise_cov=noise_cov.to(torch.float64),
        ledger=ledger,
        lambda_family=family,
    )


def regularization_sweep(sol: InverseSolutionSet) -> dict[str, float]:
    """How much of the answer is the regularisation rather than the data."""
    if not sol.lambda_family:
        return {"spread": float("nan")}
    keys = sorted(sol.lambda_family)
    stack = torch.stack([sol.lambda_family[k] for k in keys])
    ref = stack.mean(0)
    rel = (stack - ref).norm(dim=(1, 2)) / ref.norm().clamp_min(1e-300)
    peaks = [int(sol.lambda_family[k].abs().sum(-1).argmax()) for k in keys]
    return {
        "lambda_min": keys[0],
        "lambda_max": keys[-1],
        "max_relative_deviation": float(rel.max()),
        "peak_source_changes": float(len(set(peaks)) > 1),
        "n_distinct_peaks": float(len(set(peaks))),
    }


def _inverse_ledger(
    *,
    method: str,
    lambda2: float,
    depth_weighting: float,
    n_null: int,
    n_src: int,
    rank: int,
    res: ResolutionAnalysis,
    family: Mapping[float, torch.Tensor],
    lead_field: LeadField,
) -> UncertaintyLedger:
    ple = res.peak_localization_error_m()
    disp = res.spatial_dispersion_m()
    ct = res.crosstalk_ratio()

    var_model: float | str = UNKNOWN
    if family:
        stack = torch.stack([family[k] for k in sorted(family)])
        var_model = float(stack.var(dim=0, unbiased=True).mean())

    bias = (
        BiasTerm(
            name="source_non_identifiability",
            interval=(-1.0, 1.0),
            status="prior_specified_sensitivity",
            units=DIMENSIONLESS,
            sensitivity_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
            note=f"the lead field has a {n_null}-dimensional null space out of "
            f"{n_src} source components (rank {rank}); any component in that "
            "subspace is invisible to the data and is set by the prior alone",
        ),
        BiasTerm(
            name="depth_bias",
            interval=(-float(ple.max()), float(ple.max())),
            status="design_estimable",
            units="m",
            estimator="peak localisation error of the resolution matrix, "
            "computed per source position from the same head model "
            f"(depth_weighting={depth_weighting})",
        ),
        BiasTerm(
            name="spatial_leakage",
            interval=(-float(ct.max()), float(ct.max())),
            status="design_estimable",
            units=DIMENSIONLESS,
            estimator="cross-talk function of the resolution matrix; a measured "
            "property of this operator on this source space",
        ),
        BiasTerm(
            name="regularization_choice",
            interval=(-0.5, 0.5),
            status="prior_specified_sensitivity",
            units=DIMENSIONLESS,
            sensitivity_grid=tuple(sorted(family)) or (lambda2,),
            note="lambda encodes an assumed SNR; the sweep shows how much of the "
            "estimate it determines",
        ),
    )
    return UncertaintyLedger(
        variance=VarianceDecomposition(
            measurement=UNKNOWN,
            within_session=UNKNOWN,
            between_session=UNKNOWN,
            parameter_posterior=UNKNOWN,
            model_class=var_model,
            numerical=0.0,
            units="dimensionless",
        ),
        bias=bias,
        model_discrepancy=UNKNOWN,
        model_discrepancy_flag=True,
        validity_domain={
            "method": method,
            "lambda2": lambda2,
            "depth_weighting": depth_weighting,
            "leadfield_rank": rank,
            "null_dimension": n_null,
            "n_source_components": n_src,
            "mean_peak_localization_error_m": float(ple.mean()),
            "mean_spatial_dispersion_m": float(disp.mean()),
            "mean_crosstalk_ratio": float(ct.mean()),
            "head_model": lead_field.meta.get("head_model", "unknown"),
            "claim_boundary": "an admissible set of source configurations under "
            "the stated head model, noise model and prior. NOT the source. "
            "A single map from this object is a visualisation, not a measurement "
            "(body.tex Sec. 2.6, Sec. 2.8).",
        },
        provenance=Provenance(
            operator=f"solve_inverse[{method}]",
            frames=(lead_field.frame,),
            references=(
                "Hamalainen & Ilmoniemi 1994 (MNE)",
                "Dale et al. 2000 (dSPM)",
                "Pascual-Marqui 2002 (sLORETA)",
                "Van Veen et al. 1997 (LCMV)",
                "Hauk, Stenroos & Treder 2019 (resolution metrics)",
            ),
        ),
        notes=(
            "point_estimate() refuses by default; use admissible() or "
            "sample_posterior() to report the set.",
        ),
    )
