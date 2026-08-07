"""Voxel BOLD → parcel-space observations. The consumer `registration.py` lacked.

`scwbd/anatomy/registration.py` has been on master with a full EPI ← T1w ←
template chain, `labels_to_epi_grid`, and `ParcelCoverage` — and was imported by
nothing but its own test file. Every source card that needs it says the
registration "has not been run", which is true and reads as "none is available".
This module is the consumer, so that sentence stops being true.

**Zero voxels means unobserved, never zero signal.** Parcels outside the
acquisition get `NaN`, not `0.0`, and the coverage mask travels with the
timeseries so a downstream likelihood can drop them rather than fit them. That
is `ARCHITECTURE.md` §7 rule 1 and it is the difference between a measurement
and an imputation — a BOLD value of 0.0 for a parcel that was never in the field
of view is a fabricated observation, and it is indistinguishable from a real one
once it is in an array.

The registration is *per subject*, not per run: EPI ← T1w is estimated from one
run's reference volume and reused, because the subject does not move between
runs by more than the within-run motion already unmodelled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ParcelBold", "parcellate_run", "DEFAULT_ATLAS"]

DEFAULT_ATLAS = "Schaefer400x7"


@dataclass
class ParcelBold:
    """Parcel-space BOLD for one run, with its coverage and its provenance."""

    subject: str
    run: str
    #: ``(n_parcels, n_frames)``; `NaN` wherever the parcel has no voxels.
    timeseries: np.ndarray
    #: ``(n_parcels,)`` bool — True where at least one EPI voxel landed.
    covered: np.ndarray
    #: ``(n_parcels,)`` int — how many EPI voxels contributed.
    n_voxels: np.ndarray
    labels: np.ndarray
    tr_seconds: float
    atlas: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_parcels(self) -> int:
        return int(self.timeseries.shape[0])

    @property
    def coverage_fraction(self) -> float:
        return float(self.covered.mean())

    def describe(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "run": self.run,
            "atlas": self.atlas,
            "n_parcels": self.n_parcels,
            "n_frames": int(self.timeseries.shape[1]),
            "tr_seconds": self.tr_seconds,
            "parcels_covered": int(self.covered.sum()),
            "coverage_fraction": round(self.coverage_fraction, 4),
            "unobserved_are_nan": True,
            "provenance": self.provenance,
        }


def _reference_volume(data: np.ndarray) -> np.ndarray:
    """A motion-robust reference: the median over time, not the first frame.

    The first frame is the worst available choice — it carries the largest
    T1 saturation difference from the rest of the run.
    """
    return np.nanmedian(data, axis=3)


def parcellate_run(
    bold_path: str | Path,
    t1w_path: str | Path,
    *,
    atlas: str = DEFAULT_ATLAS,
    assets: str | Path = "data/assets",
    template_path: str | Path | None = None,
    subject: str = "",
    run: str = "",
    nonlinear: bool = True,
    chain: Any | None = None,
) -> tuple[ParcelBold, Any]:
    """Register one BOLD run to the atlas and average within parcels.

    Returns ``(ParcelBold, TransformChain)``; pass the chain back in via
    ``chain=`` to reuse a subject's registration across their runs, which is
    where nearly all the wall-clock goes.
    """
    import nibabel as nib

    from scwbd.anatomy.registration import (
        labels_to_epi_grid,
        load_atlas_volume,
        parcel_coverage,
        register_epi_to_template,
    )

    bold_img = nib.load(str(bold_path))
    if bold_img.ndim != 4:
        raise ValueError(f"{bold_path} is {bold_img.ndim}D; a BOLD run must be 4D")
    bold = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    epi_affine = np.asarray(bold_img.affine, dtype=float)
    tr = float(bold_img.header.get_zooms()[3])

    voxel_labels, atlas_affine, labels, hemi, meta = load_atlas_volume(atlas, assets=assets)
    n_parcels = int(len(labels))

    if chain is None:
        if template_path is None:
            template_path = (
                Path(assets) / "cache/neuromaps/atlases/MNI152/tpl-MNI152NLin6Asym_res-1mm_T1w.nii.gz"
            )
        if not Path(template_path).exists():
            raise FileNotFoundError(
                f"no MNI152NLin6Asym template at {template_path}. The atlas is defined on "
                "that template and registering to a different MNI variant is a systematic "
                "several-mm error, so this refuses rather than substituting one."
            )
        t1w_img = nib.load(str(t1w_path))
        tpl_img = nib.load(str(template_path))
        chain = register_epi_to_template(
            _reference_volume(bold),
            epi_affine,
            np.asanyarray(t1w_img.dataobj, dtype=np.float32),
            np.asarray(t1w_img.affine, dtype=float),
            np.asanyarray(tpl_img.dataobj, dtype=np.float32),
            np.asarray(tpl_img.affine, dtype=float),
            subject=subject,
            nonlinear=nonlinear,
        )

    labels_epi = labels_to_epi_grid(
        chain, bold.shape[:3], epi_affine, voxel_labels, atlas_affine, hemi
    )
    cov = parcel_coverage(
        labels_epi, n_parcels, np.asarray(labels), epi_affine, subject=subject, run=run
    )

    # Average within each parcel. NaN — not zero — where nothing was observed.
    n_frames = bold.shape[3]
    flat = bold.reshape(-1, n_frames)
    lab_flat = labels_epi.ravel()
    ts = np.full((n_parcels, n_frames), np.nan, dtype=np.float32)
    for p in range(n_parcels):
        sel = lab_flat == p
        if sel.any():
            ts[p] = flat[sel].mean(axis=0)

    pb = ParcelBold(
        subject=subject,
        run=run,
        timeseries=ts,
        covered=np.asarray(cov.covered),
        n_voxels=np.asarray(cov.n_voxels),
        labels=np.asarray(labels),
        tr_seconds=tr,
        atlas=atlas,
        provenance={
            "bold": str(bold_path),
            "t1w": str(t1w_path),
            "atlas_template": meta.get("provenance", {}).get("template_id"),
            "registration_engine": getattr(chain, "engine", "?"),
            "registration_lineage": getattr(chain, "lineage", "?"),
            "epi_voxel_mm3": cov.epi_voxel_mm3,
            "coverage_basis": cov.basis,
            "coverage_note": cov.notes,
        },
    )
    return pb, chain
