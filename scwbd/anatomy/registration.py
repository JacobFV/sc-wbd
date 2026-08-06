"""BOLD -> parcel registration: bring the atlas to the subject, never the reverse.

Why this direction
------------------
The obvious pipeline resamples BOLD into MNI and averages parcels there.  This
module does the opposite: it composes the transforms and pulls the **atlas
labels into each subject's native EPI grid** with nearest-neighbour, then
averages inside the EPI voxels.  Three reasons, and the first is binding:

1. **The support stays declarable.** Interpolating BOLD before averaging makes
   the effective support ``EPI voxel ⊛ interpolation kernel ⊛ local warp
   Jacobian``, which cannot be written into ``Support.psf`` -- and R01 refuses
   an unstated support.  In native space the PSF *is* the EPI voxel.
2. **The data is never interpolated.**  Only the label field is, and labels are
   resampled nearest-neighbour, which is exact for a categorical field.
3. **Coverage falls out for free.**  A parcel with zero EPI voxels is visibly
   uncovered rather than silently averaged from nothing.

The frame trap this module exists to avoid
------------------------------------------
``Schaefer400x7__MNI152-1mm`` is stored on the **MNI152NLin6Asym** grid with
``affine[0,0] = -1`` (voxel x increases as world x *decreases*).  The
templateflow ``MNI152NLin6Asym`` T1w image covers the identical physical space
with ``affine[0,0] = +1``.  Indexing one with the other's voxel indices is a
**181 mm left-right flip**, and because the atlas is bilateral the result looks
entirely reasonable -- it would corrupt exactly the lateralisation check the
motor-localizer validation relies on.

Every coordinate here therefore travels as a **world coordinate** and every grid
change goes through that grid's own affine.  :func:`assert_hemisphere_convention`
is the executable form of that rule and it runs inside
:func:`labels_to_epi_grid`.

Registration engine
-------------------
dipy (``AffineRegistration`` + ``SymmetricDiffeomorphicRegistration``).  antspyx
publishes no Linux ``aarch64`` wheel, so on this box it would require a source
build of ITK+ANTs.  The nonlinear stage is SyN (Avants 2008) **as implemented in
dipy**; the Klein 2009 evaluation that makes SyN the best-validated method
covers the *ANTs* implementation, not this one, and that distinction is recorded
rather than glossed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ParcelCoverage",
    "TransformChain",
    "assert_hemisphere_convention",
    "load_atlas_volume",
    "register_epi_to_template",
    "labels_to_epi_grid",
    "parcel_coverage",
    "parcel_fov_fraction",
]

#: The template the shipped Schaefer volume is defined on.  Not a free choice:
#: registering to ``MNI152NLin2009cAsym`` and then applying labels defined on
#: ``MNI152NLin6Asym`` is a systematic several-mm error, largest in ventral and
#: temporal cortex.
ATLAS_TEMPLATE = "MNI152NLin6Asym"


@dataclass(frozen=True)
class ParcelCoverage:
    """How many EPI voxels each parcel actually got, for one subject/run.

    ``covered`` is the (subject x parcel) mask the architect ruled must be
    declared and never imputed.  A parcel at zero is *unobserved in this
    acquisition*, which is not the same as a parcel whose BOLD is zero.
    """

    subject: str
    run: str
    labels: np.ndarray  # (P,) parcel names
    n_voxels: np.ndarray  # (P,) EPI voxels assigned to each parcel
    covered: np.ndarray  # (P,) bool
    epi_voxel_mm3: float
    #: ``"brain_mask"`` when counted inside a brain mask, ``"field_of_view"``
    #: when counted over the whole EPI array.
    #:
    #: These are NOT interchangeable and the difference is not small. The EPI
    #: array is a rectangular box; without a mask, every parcel that falls
    #: anywhere inside that box counts as "covered" whether or not there is
    #: brain signal there. On sub-xp107 the unmasked count is 400/400 covered
    #: -- which says only that the atlas fits inside the box, and would be
    #: read as "every parcel is observed". The field is mandatory so an
    #: FOV count can never be quoted as a signal count.
    basis: str
    #: ``(P,)`` observed volume / expected volume, or ``nan`` when no expectation
    #: was supplied.
    #:
    #: ``covered`` alone is the wrong statistic and this field exists because it
    #: misled me: a parcel 60% outside the slab still has voxels, so binary
    #: coverage reported 400/400 for a subject with 59% vertex tissue and 400/400
    #: for one with 100%. The clipping was entirely real and entirely invisible
    #: until the per-parcel volumes were compared against what the atlas says the
    #: parcel should be.
    fraction_observed: np.ndarray | None = None
    notes: str = ""

    @property
    def n_parcels(self) -> int:
        return int(self.n_voxels.shape[0])

    def summary(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "run": self.run,
            "n_parcels": self.n_parcels,
            "n_covered": int(self.covered.sum()),
            "n_uncovered": int((~self.covered).sum()),
            "median_voxels_per_covered_parcel": (
                float(np.median(self.n_voxels[self.covered])) if self.covered.any() else 0.0
            ),
            "epi_voxel_mm3": round(self.epi_voxel_mm3, 3),
            "basis": self.basis,
            **(
                {}
                if self.fraction_observed is None
                else {
                    "fraction_observed_median": round(
                        float(np.nanmedian(self.fraction_observed)), 4
                    ),
                    "n_parcels_under_50pct": int(
                        np.nansum(self.fraction_observed < 0.5)
                    ),
                    "n_parcels_under_10pct": int(
                        np.nansum(self.fraction_observed < 0.1)
                    ),
                }
            ),
        }


@dataclass(frozen=True)
class TransformChain:
    """EPI <- T1w <- template, as world-to-world affines plus optional warp.

    ``epi_from_template`` maps a **template world point** to an **EPI world
    point**.  That is the direction the label pull needs: for each EPI voxel we
    invert it to ask which template voxel the EPI voxel sits on.
    """

    subject: str
    epi_from_t1w: np.ndarray  # (4,4) world->world
    t1w_from_template: np.ndarray  # (4,4) world->world
    engine: str
    lineage: str
    warp: Any | None = None  # dipy DiffeomorphicMap, template<->t1w
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def epi_from_template(self) -> np.ndarray:
        return self.epi_from_t1w @ self.t1w_from_template

    def perturbed(self, translation_mm: tuple[float, float, float]) -> "TransformChain":
        """Return this chain with a deliberate translation injected.

        The 10 mm perturbation control: a validation that still passes under
        this is not measuring registration quality at parcel scale.  Applied at
        the EPI<-T1w stage because that is where a real misregistration of this
        size would sit.
        """
        d = np.eye(4)
        d[:3, 3] = np.asarray(translation_mm, dtype=float)
        return TransformChain(
            subject=self.subject,
            epi_from_t1w=d @ self.epi_from_t1w,
            t1w_from_template=self.t1w_from_template,
            engine=self.engine + f"+perturbed{tuple(translation_mm)}",
            lineage=self.lineage + f" | PERTURBED by {translation_mm} mm (control, not a real transform)",
            warp=self.warp,
            diagnostics=dict(self.diagnostics),
        )


def load_atlas_volume(
    atlas: str = "Schaefer400x7", *, assets: str | Path = "assets"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """``(voxel_labels, affine, labels, hemi, meta)`` for a volumetric atlas."""
    path = Path(assets) / "derived" / "parcellations" / f"{atlas}__MNI152-1mm.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no volumetric atlas at {path}. Run `python -m scwbd.anatomy.build` "
            "to materialise it; this module will not fall back to a surface "
            "parcellation, which lives on a different support entirely."
        )
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["_meta"]))
    tpl = str(meta.get("provenance", {}).get("template_id", ""))
    if tpl and tpl != ATLAS_TEMPLATE:
        raise ValueError(
            f"{atlas} is defined on {tpl!r} but this module registers to "
            f"{ATLAS_TEMPLATE!r}. Mixing MNI variants is a systematic several-mm "
            "error, largest in ventral and temporal cortex. Fix the template "
            "rather than proceeding."
        )
    return z["voxel_labels"], z["affine"], z["labels"], z["hemi"], meta


def assert_hemisphere_convention(
    voxel_labels: np.ndarray, affine: np.ndarray, hemi: np.ndarray
) -> dict[str, float]:
    """Raise unless left-hemisphere parcels sit at negative world x.

    This is the executable form of "never index a grid with another grid's
    voxel indices". If the affine is dropped or replaced by a same-shaped one
    with the opposite x direction, the atlas mirrors, every bilateral summary
    stays plausible, and only a lateralisation claim reveals it -- by then it is
    a result, not a bug report.
    """
    import nibabel as nib

    idx = np.argwhere(voxel_labels >= 0)
    ids = voxel_labels[voxel_labels >= 0]
    world = nib.affines.apply_affine(np.asarray(affine, dtype=float), idx)
    out: dict[str, float] = {}
    for h in ("L", "R"):
        sel = np.isin(ids, np.where(np.asarray(hemi) == h)[0])
        if not sel.any():
            continue
        out[f"mean_world_x_{h}"] = float(world[sel, 0].mean())
    if "mean_world_x_L" in out and "mean_world_x_R" in out:
        if not (out["mean_world_x_L"] < 0 < out["mean_world_x_R"]):
            raise ValueError(
                "hemisphere convention violated: left-hemisphere parcels do not "
                f"sit at negative world x ({out}). The atlas is mirrored -- almost "
                "certainly a grid indexed with another grid's voxel indices. A "
                "bilateral atlas hides this in every summary except lateralisation."
            )
    return out


def register_epi_to_template(
    epi_ref: np.ndarray,
    epi_affine: np.ndarray,
    t1w: np.ndarray,
    t1w_affine: np.ndarray,
    template: np.ndarray,
    template_affine: np.ndarray,
    *,
    subject: str = "",
    nonlinear: bool = True,
    level_iters: tuple[int, ...] = (100, 50, 25),
    sampling: float | None = 0.25,
) -> TransformChain:
    """Two-stage registration: EPI<-T1w (affine), T1w<-template (affine [+SyN]).

    Both stages are driven by image similarity in the pair that shares tissue
    contrast.  EPI is never compared directly to the template.
    """
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import AffineTransform3D, RigidTransform3D

    # sampling_proportion=None means "use every voxel", which on 1 mm volumes
    # costs tens of minutes per subject for no measurable accuracy gain. A 25%
    # random sample is the standard compromise and is what makes 10 subjects x
    # 2 tasks tractable on a shared box.
    metric = MutualInformationMetric(nbins=32, sampling_proportion=sampling)
    areg = AffineRegistration(
        metric=metric,
        level_iters=list(level_iters),
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
        verbosity=0,
    )

    def _fit(static, static_aff, moving, moving_aff, *, allow_scale: bool):
        """Return world->world affine mapping STATIC world to MOVING world.

        ``allow_scale`` is NOT a tuning knob. Within a subject and session the
        EPI and the T1w are the same head, so the only free parameters are
        rotation and translation -- 6 DOF. Fitting 12 DOF there lets the
        optimiser *scale the anatomy to fill the acquisition*, and with a
        partial-FOV slab that is precisely what it does: every parcel lands
        inside the box and clipping becomes invisible. Between subject and
        template the scale difference is real, so 12 DOF is correct there.
        """
        c = transform_centers_of_mass(static, static_aff, moving, moving_aff)
        rig = areg.optimize(
            static, moving, RigidTransform3D(), None, static_aff, moving_aff,
            starting_affine=c.affine,
        )
        if not allow_scale:
            return np.asarray(rig.affine, dtype=float)
        aff = areg.optimize(
            static, moving, AffineTransform3D(), None, static_aff, moving_aff,
            starting_affine=rig.affine,
        )
        return np.asarray(aff.affine, dtype=float)

    # EPI <- T1w: RIGID. See _fit.__doc__ -- a 12 DOF fit here hides FOV clipping.
    epi_from_t1w = _fit(t1w, t1w_affine, epi_ref, epi_affine, allow_scale=False)
    t1w_from_template = _fit(template, template_affine, t1w, t1w_affine, allow_scale=True)

    # Record the scale actually applied at each stage. A rigid stage whose
    # singular values drift from 1 means the fit is not rigid and the guard
    # above has been bypassed.
    def _sv(m):
        return tuple(round(float(x), 4) for x in np.linalg.svd(m[:3, :3], compute_uv=False))

    diagnostics = {
        "epi_from_t1w_singular_values": _sv(epi_from_t1w),
        "t1w_from_template_singular_values": _sv(t1w_from_template),
        "epi_stage_dof": 6,
        "template_stage_dof": 12,
    }
    sv = np.asarray(diagnostics["epi_from_t1w_singular_values"])
    if np.abs(sv - 1.0).max() > 1e-3:
        raise ValueError(
            f"EPI<-T1w stage is not rigid (singular values {tuple(sv)}). A scaling "
            "EPI<-T1w transform resizes the subject's anatomy to fit the "
            "acquisition, which makes field-of-view clipping undetectable: every "
            "parcel lands inside the slab and coverage reads full."
        )

    warp = None
    engine = "dipy.AffineRegistration(rigid->affine, MI)"
    if nonlinear:
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import CCMetric

        sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=[25, 15, 8])
        warp = sdr.optimize(
            template, t1w, template_affine, t1w_affine, t1w_from_template
        )
        engine += " + dipy.SyN(CC)"

    return TransformChain(
        subject=subject,
        epi_from_t1w=epi_from_t1w,
        t1w_from_template=t1w_from_template,
        engine=engine,
        warp=warp,
        diagnostics=diagnostics,
        lineage=(
            "EPI<-T1w RIGID 6-DOF (MI, within-session, shares scanner world frame; "
            "rigid because a scaling fit would resize anatomy to fill a partial FOV); "
            f"T1w<-{ATLAS_TEMPLATE} affine (MI)"
            + (" then SyN (Avants 2008) as implemented in dipy -- the Klein 2009 "
               "evaluation covers the ANTs implementation, not this one" if nonlinear else "")
        ),
    )


def labels_to_epi_grid(
    chain: TransformChain,
    epi_shape: tuple[int, int, int],
    epi_affine: np.ndarray,
    voxel_labels: np.ndarray,
    atlas_affine: np.ndarray,
    hemi: np.ndarray | None = None,
) -> np.ndarray:
    """Pull atlas labels onto the EPI grid, nearest-neighbour, via world coords.

    Never indexes one grid with another's voxel indices: every hop is
    ``apply_affine`` with that grid's own matrix.
    """
    import nibabel as nib
    from scipy import ndimage

    if hemi is not None:
        assert_hemisphere_convention(voxel_labels, atlas_affine, hemi)

    gi, gj, gk = np.meshgrid(
        np.arange(epi_shape[0]), np.arange(epi_shape[1]), np.arange(epi_shape[2]),
        indexing="ij",
    )
    epi_vox = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1).astype(float)
    epi_world = nib.affines.apply_affine(np.asarray(epi_affine, float), epi_vox)

    # EPI world -> template world is the inverse of the composed chain.
    tpl_world = nib.affines.apply_affine(np.linalg.inv(chain.epi_from_template), epi_world)

    if chain.warp is not None:
        # SyN is defined template<->T1w, so undo it in template space.
        tpl_world = _apply_warp_to_points(chain.warp, tpl_world)

    atlas_vox = nib.affines.apply_affine(np.linalg.inv(np.asarray(atlas_affine, float)), tpl_world)
    out = ndimage.map_coordinates(
        voxel_labels.astype(np.int32), atlas_vox.T, order=0, mode="constant", cval=-1
    )
    return out.reshape(epi_shape).astype(np.int32)


def _apply_warp_to_points(warp: Any, pts: np.ndarray) -> np.ndarray:
    """Displace template-space world points by the diffeomorphic map.

    dipy's ``DiffeomorphicMap`` resamples images rather than points, so the
    displacement is sampled at each point's position in the warp's own grid.
    """
    import nibabel as nib
    from scipy import ndimage

    disp = np.asarray(warp.get_backward_field())
    grid2world = np.asarray(warp.domain_grid2world, dtype=float)
    vox = nib.affines.apply_affine(np.linalg.inv(grid2world), pts).T
    d = np.stack(
        [ndimage.map_coordinates(disp[..., k], vox, order=1, mode="nearest") for k in range(3)],
        axis=1,
    )
    return pts + d


def parcel_coverage(
    labels_in_epi: np.ndarray,
    n_parcels: int,
    parcel_names: np.ndarray,
    epi_affine: np.ndarray,
    *,
    subject: str = "",
    run: str = "",
    brain_mask: np.ndarray | None = None,
    expected_mm3: np.ndarray | None = None,
    volume_scale: float = 1.0,
) -> ParcelCoverage:
    """Count EPI voxels per parcel; zero is *unobserved*, not zero-signal."""
    if brain_mask is None:
        lab = labels_in_epi
        basis = "field_of_view"
    else:
        lab = np.where(brain_mask, labels_in_epi, -1)
        basis = "brain_mask"
    counts = np.bincount(lab[lab >= 0].ravel(), minlength=n_parcels)[:n_parcels]
    vox_mm3 = float(abs(np.linalg.det(np.asarray(epi_affine, float)[:3, :3])))
    frac = None
    if expected_mm3 is not None:
        exp = np.asarray(expected_mm3, dtype=float) * float(volume_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (counts * vox_mm3) / np.where(exp > 0, exp, np.nan)

    return ParcelCoverage(
        subject=subject,
        run=run,
        labels=np.asarray(parcel_names),
        n_voxels=counts,
        covered=counts > 0,
        epi_voxel_mm3=vox_mm3,
        basis=basis,
        fraction_observed=frac,
        notes=(
            "Zero voxels means the parcel is outside this acquisition's field of "
            "view, not that its BOLD is zero. Never impute (ARCHITECTURE.md §7 "
            "rule 1)."
            + (
                " COUNTED OVER THE WHOLE EPI BOX, NOT A BRAIN MASK: 'covered' here "
                "means the parcel falls inside the acquisition's rectangular array, "
                "which is a far weaker statement than 'this parcel has signal'."
                if basis == "field_of_view"
                else ""
            )
        ),
    )


def parcel_fov_fraction(
    chain: "TransformChain",
    epi_shape: tuple[int, int, int],
    epi_affine: np.ndarray,
    voxel_labels: np.ndarray,
    atlas_affine: np.ndarray,
    n_parcels: int,
) -> np.ndarray:
    """``(P,)`` fraction of each parcel's volume lying inside the EPI array.

    **Pure geometry.** No brain mask, no signal, no partial-volume sampling --
    just "is this bit of the atlas inside the acquisition's box".

    This exists because I conflated three different quantities and drew a wrong
    conclusion twice. They are independent and only the first is FOV clipping:

    1. **FOV clipping** -- is the parcel inside the acquisition box at all. This
       function. Geometric, exact, unaffected by signal quality.
    2. **Brain-mask coverage** -- does the EPI have brain signal there.
       Confounded by dropout in orbitofrontal and temporal cortex.
    3. **Partial-volume sampling** -- can a 2x2x3.8 mm grid resolve a ~3 mm
       cortical ribbon. It cannot, fully: a ribbon voxel whose *centre* falls in
       white matter or CSF gets no label, which costs roughly a third of every
       parcel's nominal volume uniformly, clipped or not.

    ``ParcelCoverage.fraction_observed`` mixes 2 and 3, so it cannot be read as
    a clipping measure -- which is exactly the error that made a *more* clipped
    subject look better covered than a less clipped one.
    """
    import nibabel as nib

    sel = voxel_labels >= 0
    idx = np.argwhere(sel).astype(float)
    ids = voxel_labels[sel]
    tpl_world = nib.affines.apply_affine(np.asarray(atlas_affine, float), idx)
    if chain.warp is not None:
        tpl_world = _apply_warp_to_points(chain.warp, tpl_world)
    epi_world = nib.affines.apply_affine(chain.epi_from_template, tpl_world)
    vox = nib.affines.apply_affine(np.linalg.inv(np.asarray(epi_affine, float)), epi_world)
    dim = np.asarray(epi_shape, dtype=float)
    inside = np.all((vox >= -0.5) & (vox <= dim - 0.5), axis=1)

    tot = np.bincount(ids, minlength=n_parcels)[:n_parcels].astype(float)
    ins = np.bincount(ids[inside], minlength=n_parcels)[:n_parcels].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, ins / np.where(tot > 0, tot, np.nan), np.nan)
