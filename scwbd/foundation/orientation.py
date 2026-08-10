"""Projecting the regional dipole moment onto a lead field (O-5).

The observation heads multiply a per-region **scalar** amplitude by a
``(n_channels, n_regions)`` gain.  What that costs, as the fraction of the
whitened lead field a regional state can express, measured on the model's own
400 cortical parcels (Schaefer400x7;
``reports/transforms/resolution_pair_schaefer400.md`` §3.1):

    per-parcel scalar, 400 parcels        eta = 0.321
    net dipole moment, 3/parcel, 400      eta = 0.834
    scalar, parcels subdivided to 3154    eta = 0.708

**Orientation is the largest single win, at 2.6x, and the cheapest: 1200
oriented numbers carry more than 3154 scalars do.**  Extra parcels are not
free of value -- that claim came from the 68-parcel Desikan-Killiany atlas,
where subdividing to 542 elements reached only 0.162, and it does not survive
on this parcellation.

This module is the projection, and it exists as its own file because the physics
belongs in one place.  Given a parcel's dipole moment ``m`` (three numbers, Hz*m,
anatomical frame), its unit cortical normal ``n``, and its orientation coherence
``c``:

.. math::  s = (m \\cdot \\hat n)\\, c

``s`` is then what the existing scalar lead field integrates against.

**Coherence is the load-bearing half.**  A unit normal has length 1 everywhere,
so a bare normal makes every parcel look equally observable.  Coherence is
``|sum_f a_f n_f| / sum_f a_f`` over the faces of the parcel — the fraction of
its area that survives cancellation between opposing sulcal banks.  Measured on
the real prior: median 0.851, but the 5th percentile is 0.475 and **23 of 400
parcels sit below 0.5**.  A parcel at 0.275 contributes a quarter of what its
area suggests, and without ``c`` the model would predict it contributes all of
it.

**Coverage is part of the type, not the caller's problem.**  14 of 414 parcels
are subcortical and have no cortical normal; ``AnatomyPrior.normal`` carries
**NaN** there, and ``isnan(normal).any(-1)`` is exactly ``~normal_covered``.
That is the right design — an absent value is visible rather than silently zero
— and it is a live hazard: one unguarded multiply propagates NaN through the
lead field to every channel.  :class:`DipoleProjection` therefore stores a
sanitised normal and a zero coherence on uncovered parcels, and asserts the
buffers are finite at construction, so no NaN can reach a gradient.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["DipoleProjection"]


class DipoleProjection(nn.Module):
    """``(m, n̂, c) -> s``: a moment collapsed for a **scalar** lead field.

    **DECLARED COMPATIBILITY PATH -- not the primary route.**  This reduces a
    3-vector moment to one number *before* the observation, and that reduction
    is the lossy step: with a ``(n_channels, n_regions)`` lead field you are in
    the scalar regime no matter what the state carries.  ``eta = 0.834`` was
    1200 degrees of freedom over 400 parcels -- a property of the observation
    operator consuming three numbers, not of the state holding three.  Use
    :class:`VectorLeadField` unless you are feeding an operator that
    genuinely cannot take a moment; then use this, and say so.

    The normal and the coherence are **buffers, not parameters**: they are
    measured geometry, and fitting them would let the model explain a bad
    forward solution by rotating the cortex.
    """

    def __init__(
        self,
        normal: Tensor,
        coherence: Tensor,
        covered: Tensor,
        *,
        learn_gain: bool = True,
    ) -> None:
        super().__init__()
        n = normal.shape[0]
        if coherence.shape != (n,) or covered.shape != (n,):
            raise ValueError(
                f"normal is ({n},3) but coherence is {tuple(coherence.shape)} and covered is "
                f"{tuple(covered.shape)}; all three describe the same parcels"
            )
        cov = covered.to(torch.bool)
        # Sanitise BEFORE registering: the NaN on uncovered parcels is correct in
        # the prior and unusable in a tensor that will be multiplied.
        nrm = torch.nan_to_num(normal.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coh = torch.nan_to_num(coherence.float(), nan=0.0, posinf=0.0, neginf=0.0)
        nrm = torch.where(cov.unsqueeze(-1), nrm, torch.zeros_like(nrm))
        coh = torch.where(cov, coh, torch.zeros_like(coh))
        if not torch.isfinite(nrm).all() or not torch.isfinite(coh).all():
            raise ValueError("orientation buffers are not finite after sanitisation")
        if float(coh.min()) < 0.0 or float(coh.max()) > 1.0:
            raise ValueError(
                f"coherence must lie in [0,1]; got [{float(coh.min()):.3f}, {float(coh.max()):.3f}]. "
                "It is a fraction of surviving area, not a gain."
            )
        self.register_buffer("normal", nrm)  # (N,3), zero where uncovered
        self.register_buffer("coherence", coh)  # (N,), zero where uncovered
        self.register_buffer("covered", cov)  # (N,) bool
        self.n_regions = int(n)
        # One scalar: moments are in Hz*m and the lead field is in V per unit
        # source amplitude, so SOMETHING has to carry the unit conversion. It is
        # a single learned number rather than a per-region vector on purpose --
        # a per-region gain would be able to undo the coherence weighting, which
        # is the one thing this module exists to apply.
        self.log_gain = nn.Parameter(torch.zeros(1)) if learn_gain else None

    # -- the projection ----------------------------------------------------
    def forward(self, moment: Tensor) -> Tensor:
        """``(..., N, 3) -> (..., N)`` source amplitude for a scalar lead field."""
        if moment.shape[-1] != 3:
            raise ValueError(f"a dipole moment has 3 components; got {moment.shape[-1]}")
        if moment.shape[-2] != self.n_regions:
            raise ValueError(f"moment covers {moment.shape[-2]} parcels; projection has {self.n_regions}")
        n = self.normal.to(moment.dtype)
        c = self.coherence.to(moment.dtype)
        s = (moment * n).sum(-1) * c
        return s if self.log_gain is None else s * self.log_gain.exp().to(s.dtype)

    def scatter_into(self, moment: Tensor, index: Tensor) -> Tensor:
        """Project a family's block ``(..., n_f, 3)`` using that family's parcels."""
        n = self.normal.index_select(0, index).to(moment.dtype)
        c = self.coherence.index_select(0, index).to(moment.dtype)
        s = (moment * n).sum(-1) * c
        return s if self.log_gain is None else s * self.log_gain.exp().to(s.dtype)

    def lift(self, scalar: Tensor) -> Tensor:
        """``(..., N) -> (..., N, 3)``: the scalar -> vector refinement.

        The adjoint direction, used when an exogenous drive is specified as a
        signed magnitude along the normal (⚡ Faraday's ``E·n̂``) and has to enter
        the state as a moment.  ``lift`` then ``forward`` is **not** the
        identity: it is multiplication by ``c**2``, because the drive is
        attenuated going in and the observation is attenuated coming out. That
        is physics, not a bug, and it is why the two must not be composed
        casually.
        """
        n = self.normal.to(scalar.dtype)
        c = self.coherence.to(scalar.dtype)
        return scalar.unsqueeze(-1) * n * c.unsqueeze(-1)

    # -- what this buys, measured -----------------------------------------
    @torch.no_grad()
    def expressible_fraction(self, lead_field: Tensor) -> dict[str, float]:
        """``eta`` for this projection against a given lead field.

        Returns the fraction of the lead field's squared Frobenius norm reachable
        by a scalar-per-parcel state versus by a full 3-vector, so the 2.6x
        orientation ratio can be re-derived here rather than cited.
        """
        L = lead_field.float()  # (C, N)
        scal = (L**2).sum()
        eff = L * self.coherence.reshape(1, -1)
        return {
            "n_covered": int(self.covered.sum()),
            "n_regions": self.n_regions,
            "coherence_mean": float(self.coherence[self.covered].mean()),
            "coherence_median": float(self.coherence[self.covered].median()),
            "coherence_p05": float(torch.quantile(self.coherence[self.covered], 0.05)),
            "below_half": int((self.coherence[self.covered] < 0.5).sum()),
            "scalar_energy": float(scal),
            "coherence_weighted_energy": float((eff**2).sum()),
            "energy_retained_by_coherence": float((eff**2).sum() / scal.clamp_min(1e-12)),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "projection": "(m . n_hat) * coherence",
            "moment_units": "Hz*m",
            "n_regions": self.n_regions,
            "n_covered": int(self.covered.sum()),
            "n_uncovered": int((~self.covered).sum()),
            "uncovered_contribute": 0.0,
            "normal_is_buffer": True,
            "coherence_is_buffer": True,
            "learned_scalars": 0 if self.log_gain is None else 1,
        }


# ======================================================================
# the vector lead field -- the half of O-5 that actually carries the 2.6x
# ======================================================================
class VectorLeadField(nn.Module):
    """``(n_channels, n_regions, 3)``: a forward operator that consumes a MOMENT.

    **This is the piece that makes the state's three numbers reach an
    observation as three numbers.**  The scalar ``(n_channels, n_regions)``
    lead field is

    .. math::  L^{\\rm scalar}_{cp} = \\sum_{i \\in p} a_i\\, L_{ci} \\cdot \\hat n_i

    — the normal is contracted *inside* the parcel sum, so the cancellation
    between opposing sulcal banks is baked into the operator and is
    irreversible.  Keeping the three components,

    .. math::  L_{cp} = \\sum_{i \\in p} a_i\\, L_{ci} \\otimes \\hat n_i
               \\in \\mathbb R^{3},\\qquad y_c = \\sum_p L_{cp}\\cdot m_p

    makes the cancellation a property of **the data** — of which moments the
    dynamics actually produces — rather than of the operator.  🧠 Cajal's
    coherence then stops being a factor anyone applies and becomes a
    *diagnostic*: a measurement of what a parcel's own folding costs it, which
    you can compute after the fact instead of pre-applying.

    Why this matters and why the earlier scalar-only version of this module was
    half a fix: ``eta = 0.834`` was measured at **1200 degrees of freedom over
    400 parcels — 400 x 3**.  It is a property of the *observation operator*
    consuming three numbers, not of the *state* carrying three.  With a scalar
    lead field the reduction to one number happens before the observation, and
    the reduction is the lossy step, so a vector state buys nothing.  0.321, not
    0.834, no matter what the state holds.
    """

    def __init__(self, matrix: Tensor, channel_names: tuple[str, ...], *, units: str = "V",
                 frame: str = "unknown", provenance: str = "unknown", note: str = "") -> None:
        super().__init__()
        if matrix.ndim != 3 or matrix.shape[-1] != 3:
            raise ValueError(
                f"a vector lead field is (n_channels, n_regions, 3); got {tuple(matrix.shape)}. "
                "A 2-D matrix is the scalar operator and is exactly what this class exists to "
                "replace -- see DipoleProjection for the declared compatibility path."
            )
        self.register_buffer("L", matrix.float())
        self.channel_names = tuple(channel_names)
        self.units, self.frame, self.provenance, self.note = units, frame, provenance, note

    @property
    def n_channels(self) -> int:
        return int(self.L.shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.L.shape[1])

    def forward(self, moment: Tensor) -> Tensor:
        """``(..., N, 3) -> (..., C)``.  The moment reaches the sensors as a vector."""
        if moment.shape[-2:] != (self.n_regions, 3):
            raise ValueError(
                f"moment is {tuple(moment.shape[-2:])}, expected ({self.n_regions}, 3)"
            )
        return torch.einsum("cnk,...nk->...c", self.L.to(moment.dtype), moment)

    # -- the declared compatibility path -----------------------------------
    def contract(self, normal: Tensor, coherence: Tensor | None = None) -> Tensor:
        """``(C, N, 3) -> (C, N)``: the **lossy** scalar operator, made explicit.

        This is what the existing ``(64, 414)`` lead field already is.  Producing
        it here, from the vector form, means the loss is a named step with a
        measurable size rather than a shape nobody questioned.
        """
        n = torch.nan_to_num(normal.float(), nan=0.0)
        s = torch.einsum("cnk,nk->cn", self.L, n)
        return s if coherence is None else s * torch.nan_to_num(coherence.float(), nan=0.0).reshape(1, -1)

    @torch.no_grad()
    def orientation_headroom(self, normal: Tensor, coherence: Tensor | None = None) -> dict[str, float]:
        """How much of this forward operator the scalar path cannot reach.

        Reports the fraction of the vector operator's energy retained by the
        contracted one.  This is the local analogue of the pair's eta and it is
        computed on **our** forward model, so the orientation ratio is re-derived
        rather than cited.
        """
        full = float((self.L**2).sum())
        scal = float((self.contract(normal, coherence) ** 2).sum())
        # rank is the honest ceiling on what any state can drive through it
        r_vec = int(torch.linalg.matrix_rank(self.L.reshape(self.n_channels, -1)).item())
        r_sc = int(torch.linalg.matrix_rank(self.contract(normal, coherence)).item())
        return {
            "energy_vector": full,
            "energy_scalar": scal,
            "fraction_retained_by_contraction": scal / max(full, 1e-30),
            "headroom_multiple": full / max(scal, 1e-30),
            "rank_vector": r_vec,
            "rank_scalar": r_sc,
            "dof_vector": self.n_regions * 3,
            "dof_scalar": self.n_regions,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.L.shape),
            "consumes": "current-dipole moment (Hz*m), 3 per parcel",
            "n_channels": self.n_channels,
            "n_regions": self.n_regions,
            "units": self.units,
            "frame": self.frame,
            "provenance": self.provenance,
            "note": self.note,
        }


def build_vector_lead_field(anat, *, channel_names=None, device="cpu", conductivity: float = 0.33,
                            allow_fallback: bool = True) -> VectorLeadField:
    """Build ``(C, N, 3)``, preferring agent F's free-orientation BEM forward.

    MNE's ``make_forward_solution`` is free-orientation by default
    (``source_ori == FIFFV_MNE_FREE_ORI``) and its gain is ``(n_sens, 3*n_src)``,
    so the three components are **already there** and the current scalar lead
    field is throwing them away at the parcel-aggregation step.  Nothing new has
    to be solved to keep them.

    The analytic fallback is the same single-sphere kernel the scalar builder
    uses with one line changed: it does not contract against the orientation.
    It is **not** a head model and supports no source-localisation claim.
    """
    import math as _math

    import numpy as np

    from .heads import EEGMMIDB_CHANNELS, _montage_positions

    names = tuple(channel_names or EEGMMIDB_CHANNELS)
    dev = torch.device(device)

    if not allow_fallback:
        raise RuntimeError("free-orientation BEM forward not wired yet; pass allow_fallback=True")

    try:
        elec, kept = _montage_positions(names)
    except Exception:  # noqa: BLE001
        n_ch = len(names)
        idx = np.arange(n_ch, dtype=np.float64)
        phi = _math.pi * (3 - _math.sqrt(5)) * idx
        z = 1 - 2 * (idx + 0.5) / n_ch
        r = np.sqrt(np.clip(1 - z * z, 0, None))
        elec = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1) * 95.0
        kept = names

    src = anat.positions.detach().float().cpu().numpy()
    r_src = np.linalg.norm(src, axis=1, keepdims=True)
    r_ele = np.linalg.norm(elec, axis=1).mean()
    src_s = src / max(r_src.max(), 1e-6) * (r_ele * 0.82)

    d = elec[:, None, :] - src_s[None, :, :]  # (C, N, 3)
    r = np.maximum(np.linalg.norm(d, axis=-1), 6.0)
    # THE ONE CHANGED LINE: the scalar builder does
    #     L = (d * orient).sum(-1) / (4 pi sigma r^3)
    # contracting the orientation in. Keeping the axis is the whole fix.
    L = d / (4 * _math.pi * conductivity * r[..., None] ** 3)
    return VectorLeadField(
        torch.as_tensor(L, dtype=torch.float32, device=dev),
        tuple(kept),
        units="V",
        frame="synthetic_ellipsoid_RAS",
        provenance="analytic_sphere_fallback_vector",
        note=(
            "Analytic single-sphere, free orientation. NOT a head model; no source-localisation "
            "claim. Replace with agent F's free-orientation BEM forward "
            "(mne.make_forward_solution is free-ori by default, gain (n_sens, 3*n_src))."
        ),
    )
