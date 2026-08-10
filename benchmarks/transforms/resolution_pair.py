"""Measure the declared fine/coarse resolution pair -- thesis §4.2, N-12.

Writes ``reports/transforms/resolution_pair.json``.  Every number in
``reports/transforms/resolution_pair.md`` comes from one run of this script and
from nothing else; no figure in that report is quoted from an earlier one.

What is measured, and on what
-----------------------------
Subject ``sample`` of the MNE sample dataset -- the only subject on disk that
ships a real MRI, a real three-layer BEM, a digitised electrode montage, a
head<->MRI transform, a real EEG recording and an empty-room noise covariance
from the same session.  Everything below is that one subject.

* ``G``   -- the precomputed BEM EEG forward solution, converted to fixed
  (surface-normal) orientation with cortical patch statistics, 59 good
  electrodes after dropping the recording's own bad channel, average-referenced
  explicitly on both the lead field and the data.
* fine support -- the 7498 oct-6 source-space dipoles ``G`` is defined on.
  Weight ``a_v`` is the white-surface patch area of source ``v``, summed from
  the full-resolution triangulation over the vertices in ``pinfo``.
* coarse support -- a cortical parcellation of that same surface.  Two are
  measurable here and both are measured, into separate artefacts:

  ``aparc``          the subject's own Desikan-Killiany annot, 68 parcels,
                     excluding the ``unknown`` label.
  ``Schaefer400x7``  the 400 parcels **the model actually runs on**.  Its 414
                     regions are Schaefer400x7 plus the 14 Aseg14T/Melbourne
                     subcortical volumes; a cortical-surface source space has no
                     fine support under the 14, so 400 is what this pair can
                     carry and 414 is what the model has.  See ``--parc``.

  Sources inside no parcel keep weight zero and are counted as a coverage
  deficit rather than folded into a neighbour.
* fine states -- five ensembles, deliberately not one:
  ``evoked_*``   MNE source estimates of the four real audiovisual evoked
                 responses (0-300 ms post-stimulus);
  ``patch_*``    same-signed focal geodesic patches, the physiological unit of
                 EEG generation, at four radii;
  ``grf_*``      geodesically smooth Gaussian random fields at four length
                 scales.
  The first is real but minimum-norm, so its fine structure is an estimator's
  as much as the brain's; the last two are synthetic but their smoothness is
  declared, so between them they bracket the answer.  If the verdict changed
  across ensembles it would not be reportable.  It does not.
* whitening -- the session's own noise covariance, so a residual is quoted in
  standard deviations of the noise this instrument actually has.

The pre-registered criterion
----------------------------
Fixed before any number below was computed: the coarse view *preserves* the
observable when the fine-vs-coarse difference is not detectable in this
recording, i.e. when the whitened residual is at most one noise standard
deviation per channel.  Anything above that is a difference the instrument
could see.

Run::

    PYTHONPATH=. python benchmarks/transforms/resolution_pair.py
    PYTHONPATH=. python benchmarks/transforms/resolution_pair.py --parc Schaefer400x7

Each ``--parc`` writes its own artefact -- ``resolution_pair.json`` and
``resolution_pair_schaefer400.json`` -- so neither run can land on the other's
file.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scwbd.transforms import resolution_pair as rp  # noqa: E402
from scwbd.transforms.sheaf import ScalePair  # noqa: E402

#: Pre-registered.  One noise sd per channel: the point below which the coarse
#: and fine observable predictions are not distinguishable in this recording.
TOLERANCE_SD_PER_CHANNEL = 1.0
TOLERANCE_BASIS = (
    "whitened by the session's own noise covariance (sample_audvis-cov.fif); "
    "a residual of 1.0 is one standard deviation of that noise per channel, "
    "i.e. the largest coarsening error this instrument could not detect"
)

#: The split.  Left-lateralised conditions set the prolongation's declared
#: uncertainty; right-lateralised conditions test it.  The two engage different
#: cortex, so this is a real held-out test and not a re-run.
TRAIN_CONDITIONS = ("Left Auditory", "Left visual")
TEST_CONDITIONS = ("Right Auditory", "Right visual")


# --------------------------------------------------------------------------
# the subject
# --------------------------------------------------------------------------
def _sample_root() -> Path:
    from scwbd.sources.loaders.mne_datasets import anatomy_paths

    paths = anatomy_paths("mne-sample")
    root = Path(paths["root"])
    if not (root / "MEG" / "sample").is_dir():
        raise SystemExit(
            f"MNE sample dataset not found under {root}. This measurement is "
            "not runnable without it; it is not approximated with a template."
        )
    return root


def _patch_areas(s: dict[str, Any]) -> np.ndarray:
    """White-surface area each decimated source stands for, in m^2."""
    p, t = s["rr"], s["tris"]
    tri_a = np.linalg.norm(
        np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), axis=1
    ) / 2.0
    vert_a = np.zeros(s["np"])
    for k in range(3):
        np.add.at(vert_a, t[:, k], tri_a / 3.0)
    return np.array([vert_a[np.asarray(s["pinfo"][i])].sum() for i in s["patch_inds"]])


def _membership(src, subjects_dir: Path, n_fine: int, parc: str):
    import mne

    labels = [
        l
        for l in mne.read_labels_from_annot(
            "sample", parc=parc, subjects_dir=str(subjects_dir), verbose="error"
        )
        if "unknown" not in l.name and "?" not in l.name
    ]
    assign = np.full(n_fine, -1, np.int64)
    off = 0
    for h, s in enumerate(src):
        pos = np.full(s["np"], -1, np.int64)
        pos[s["vertno"]] = np.arange(len(s["vertno"]))
        for li, l in enumerate(labels):
            if (l.hemi == "lh") != (h == 0):
                continue
            k = pos[l.vertices]
            assign[off + k[k >= 0]] = li
        off += len(s["vertno"])
    return [l.name for l in labels], assign


# --------------------------------------------------------------------------
# Schaefer400x7: the parcellation the model runs on
# --------------------------------------------------------------------------
#: FreeSurfer ``.annot`` files for Schaefer2018 in **fsaverage5** space,
#: redistributed by the ENIGMA Toolbox and vendored in this repository.  Licence
#: is established here and not by pointer: ``assets/src/enigma/LICENSE`` is
#: BSD-3-Clause (`enigmatoolbox` in ``scwbd/anatomy/sources``), and the
#: parcellation itself is `schaefer2018`, MIT (CBIG).  Both satisfy
#: ``reports/licence_audit.md``'s rule that a licence be identifiable *in this
#: repository*.
SCHAEFER_ANNOT_DIR = "assets/src/enigma/enigmatoolbox/permutation_testing/annot"

#: fsaverage5 is ico-5, and FreeSurfer's icosahedra are nested: the first 10242
#: vertices of ``fsaverage``'s 163842-vertex sphere *are* fsaverage5's vertices,
#: in order.  That identity is what lets an fsaverage5 annot be read as a label
#: on fsaverage without resampling, and it is not assumed -- it is checked in
#: ``tests/transforms/test_resolution_pair_schaefer.py`` two ways: the subset's
#: nearest-neighbour spacing is icosahedrally uniform (max/min 1.20, against 7.2
#: for a random subset of the same size), and the Schaefer annot read onto it is
#: spatially contiguous (0.82 of 6-neighbour pairs share a label, against 0.012
#: under permutation).  A wrong slice would make the parcellation noise, and
#: noise still produces a plausible-looking energy fraction.
FSAVERAGE5_N_VERTICES = 10242


def fsaverage5_vertices(fsaverage_sphere: np.ndarray) -> np.ndarray:
    """The fsaverage5 vertices of an ``fsaverage`` surface, in fsaverage5 order.

    One line, and it is a function so that the guard in
    ``tests/transforms/test_resolution_pair_schaefer.py`` tests *this* slice
    rather than a copy of it.  A test that restates the expression it is
    checking checks nothing.
    """
    return fsaverage_sphere[:FSAVERAGE5_N_VERTICES]


def _membership_schaefer(
    src, subjects_dir: Path, n_fine: int, n_rois: int = 400, networks: int = 7
):
    """Assign each fine source to a Schaefer parcel, via spherical registration.

    The chain is one nearest-neighbour lookup, composed from two exact steps:
    ``sample``'s ``?h.sphere.reg`` puts its white-surface vertices in
    ``fsaverage``'s spherical frame, and the annot's fsaverage5 vertices are
    ``fsaverage``'s first :data:`FSAVERAGE5_N_VERTICES`.  So for each source
    dipole the nearest fsaverage5 vertex on the sphere carries its label -- the
    same transfer ``mri_surf2surf`` performs, done here directly so the step is
    visible rather than buried in a resampling.

    fsaverage5's 3.49 mm mean spacing is finer than oct-6's 4.9 mm, so parcel
    boundaries are resolved *below* the fine support's own scale; the transfer
    is not the limiting approximation. Label 0 is the FreeSurfer medial wall and
    becomes ``-1``: unassigned, counted as a coverage deficit.
    """
    import nibabel as nib
    from scipy.spatial import cKDTree

    root = Path(__file__).resolve().parents[2] / SCHAEFER_ANNOT_DIR
    names: list[str] = []
    assign = np.full(n_fine, -1, np.int64)
    off = 0
    base = 0
    max_nn_mm = 0.0
    for h, hemi in enumerate(("lh", "rh")):
        annot = root / f"fsa5_{hemi}_schaefer_{n_rois}.annot"
        if not annot.is_file():
            raise SystemExit(
                f"{annot} not found. The Schaefer pair is measured on the "
                "vendored ENIGMA annot and is not approximated from another "
                "atlas; without it there is no measurement to report."
            )
        lab, _, raw = nib.freesurfer.read_annot(annot)
        raw = [n.decode() if isinstance(n, bytes) else n for n in raw]
        if lab.size != FSAVERAGE5_N_VERTICES:
            raise SystemExit(
                f"{annot.name} has {lab.size} vertices, not fsaverage5's "
                f"{FSAVERAGE5_N_VERTICES}; the nesting identity this transfer "
                "rests on does not hold for it."
            )
        # raw[0] is 'Background+FreeSurfer_Defined_Medial_Wall'
        hemi_names = raw[1:]
        fs_sphere, _ = nib.freesurfer.read_geometry(
            subjects_dir / "fsaverage" / "surf" / f"{hemi}.sphere"
        )
        sub_sphere, _ = nib.freesurfer.read_geometry(
            subjects_dir / "sample" / "surf" / f"{hemi}.sphere.reg"
        )
        v = src[h]["vertno"]
        d, nn = cKDTree(fsaverage5_vertices(fs_sphere)).query(sub_sphere[v])
        max_nn_mm = max(max_nn_mm, float(d.max()))
        l = lab[nn]
        m = l > 0
        assign[off : off + len(v)][m] = base + (l[m] - 1)
        names.extend(hemi_names)
        off += len(v)
        base += len(hemi_names)
    if base != n_rois:
        raise ValueError(f"Schaefer{n_rois}x{networks}: annots name {base} parcels")
    empty = int(np.bincount(assign[assign >= 0], minlength=base).min() == 0)
    if empty:
        raise SystemExit(
            "a Schaefer parcel received no source dipole; R has a zero row and "
            "R P = I cannot hold. The pair is not buildable at this fineness."
        )
    return names, assign, max_nn_mm


#: ``name -> (n_coarse, provenance sentence)``.  Everything parcellation-shaped
#: that varies between the two artefacts lives here.
PARCELLATIONS = {
    "aparc": (
        68,
        "aparc (Desikan-Killiany), subject's own annot, 'unknown' excluded",
    ),
    "Schaefer400x7": (
        400,
        "Schaefer400x7 (ENIGMA Toolbox fsaverage5 annot, BSD-3-Clause; "
        "Schaefer 2018, MIT/CBIG), transferred to the subject by spherical "
        "registration; medial wall excluded. This is 400 of the model's 414 "
        "regions -- the other 14 are Aseg14T subcortical volumes, which a "
        "cortical-surface source space has no fine support under",
    ),
}


def _geodesics(src) -> list[np.ndarray]:
    """Geodesic distance among used sources, from the source space's own graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    out = []
    for s in src:
        d = s["dist"]
        if d is None:
            raise SystemExit(
                "source space carries no distance graph; the patch and GRF "
                "ensembles need geodesics and will not fall back to Euclidean, "
                "which crosses sulcal banks."
            )
        sub = d.tocsr()[s["vertno"]][:, s["vertno"]]
        out.append(dijkstra(csr_matrix(sub), directed=False))
    return out


# --------------------------------------------------------------------------
def load_subject(verbose: bool = True, parc: str = "aparc") -> dict[str, Any]:
    import mne

    mne.set_log_level("ERROR")
    root = _sample_root()
    meg = root / "MEG" / "sample"
    subjects_dir = root / "subjects"

    fwd = mne.read_forward_solution(meg / "sample_audvis-eeg-oct-6-fwd.fif")
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True
    )
    G_all = np.asarray(fwd["sol"]["data"], float)
    src = fwd["src"]
    ch_fwd = list(fwd["sol"]["row_names"])

    evokeds = mne.read_evokeds(meg / "sample_audvis-ave.fif")
    info = evokeds[0].copy().pick(ch_fwd).info
    good = [c for c in ch_fwd if c not in info["bads"]]
    G = G_all[[ch_fwd.index(c) for c in good]]
    n = len(good)
    # explicit average reference, applied to the lead field and the data alike
    G = (np.eye(n) - np.ones((n, n)) / n) @ G

    cov = mne.pick_channels_cov(mne.read_cov(meg / "sample_audvis-cov.fif"), good, ordered=True)
    W, _ = mne.cov.compute_whitener(
        cov, evokeds[0].copy().pick(good).info, picks="eeg", pca=False, verbose="error"
    )

    areas = np.concatenate([_patch_areas(s) for s in src])
    expect_n, parc_note = PARCELLATIONS[parc]
    max_nn_mm = None
    if parc == "aparc":
        names, assign = _membership(src, subjects_dir, G.shape[1], "aparc")
    else:
        n_rois, networks = (int(x) for x in parc.removeprefix("Schaefer").split("x"))
        names, assign, max_nn_mm = _membership_schaefer(
            src, subjects_dir, G.shape[1], n_rois=n_rois, networks=networks
        )
    if len(names) != expect_n:
        raise ValueError(f"{parc}: built {len(names)} parcels, expected {expect_n}")

    inv = mne.minimum_norm.read_inverse_operator(
        meg / "sample_audvis-eeg-oct-6-eeg-inv.fif"
    )
    states: dict[str, np.ndarray] = {}
    for e in evokeds:
        stc = mne.minimum_norm.apply_inverse(
            e.copy().pick(good), inv, lambda2=1.0 / 9.0, method="MNE",
            pick_ori="normal", verbose="error",
        )
        sel = (stc.times >= 0.0) & (stc.times <= 0.300)
        states[e.comment] = stc.data[:, sel]

    if verbose:
        print(
            f"subject sample: {G.shape[1]} fine dipoles, {len(names)} parcels, "
            f"{n} good electrodes (dropped {info['bads']}), "
            f"cortex {areas.sum()*1e4:.0f} cm^2",
            flush=True,
        )
    return {
        "G": G, "W": W, "src": src, "areas": areas, "assign": assign,
        "parcel_names": names, "channels": good, "bads": list(info["bads"]),
        "states": states, "subjects_dir": subjects_dir, "root": root,
        "n_fine": int(G.shape[1]), "mne_version": mne.__version__,
        "parc": parc, "parc_note": parc_note, "label_transfer_max_mm": max_nn_mm,
    }


# --------------------------------------------------------------------------
# synthetic fine-state ensembles
# --------------------------------------------------------------------------
def patch_states(src, D, n_fine: int, radius_m: float, n_per_hemi: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = []
    off = 0
    for h, (s, d) in enumerate(zip(src, D)):
        for c in rng.choice(len(d), n_per_hemi, replace=False):
            x = np.zeros(n_fine)
            x[off + np.where(d[c] <= radius_m)[0]] = 1.0
            cols.append(x)
        off += len(d)
    return np.stack(cols, axis=1)


def scale_to_reference(G, W, X: np.ndarray, ref: float) -> np.ndarray:
    """Give a synthetic ensemble a physical amplitude.

    A patch or a random field has no natural units, so a residual quoted in
    noise standard deviations would be whatever amplitude the generator
    happened to pick.  Each state is scaled so the whitened sensor field it
    produces has the same norm as the *largest* field in the real held-out
    evoked responses: "a source configuration that drives this subject's
    electrodes as hard as their measured evoked response does".  The relative
    error is invariant to this; only the sd-per-channel column depends on it.
    """
    F = (W @ G) @ X
    nrm = np.linalg.norm(F, axis=0)
    keep = nrm > 0
    return X[:, keep] * (ref / nrm[keep])


def grf_states(src, D, n_fine: int, ell_m: float, n_per_hemi: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = []
    off = 0
    for h, (s, d) in enumerate(zip(src, D)):
        K = np.exp(-0.5 * (np.nan_to_num(d, posinf=1e3) / ell_m) ** 2)
        for _ in range(n_per_hemi):
            x = np.zeros(n_fine)
            x[off : off + len(d)] = K @ rng.standard_normal(len(d))
            cols.append(x)
        off += len(d)
        del K
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
def measure(subject: dict[str, Any], *, verbose: bool = True) -> rp.PairMeasurement:
    G, W, areas, assign = subject["G"], subject["W"], subject["areas"], subject["assign"]
    n_fine, n_coarse = subject["n_fine"], len(subject["parcel_names"])
    R = rp.restriction_matrix(assign, areas, n_coarse)
    P = rp.prolongation_matrix(assign, n_coarse)

    # -- is the pair paired at all -------------------------------------
    rt = float(torch.abs(R @ P - torch.eye(n_coarse, dtype=R.dtype)).max())

    # -- the prolongation's declared uncertainty, on a real split -------
    # The declared sd is an upper quantile of the training-split residual, not
    # its mean.  A prolongation's prior sd is used as a *bound* -- R02 refuses
    # the map when the held-out residual exceeds it -- and a point estimate of a
    # quantity is the wrong kind of object to bound it with: it is beaten by the
    # next sample about half the time, which makes the refusal a coin flip
    # rather than evidence.  The point estimate is measured anyway and recorded,
    # because it is what was tried first and R02 did refuse it.
    train = np.concatenate([subject["states"][c] for c in TRAIN_CONDITIONS], axis=1)
    test = np.concatenate([subject["states"][c] for c in TEST_CONDITIONS], axis=1)
    Xtr = torch.as_tensor(train.T, dtype=R.dtype)
    resid_tr = Xtr @ R.T @ P.T - Xtr
    point_sd = float(torch.sqrt((resid_tr**2).mean()))
    per_sample = torch.sqrt((resid_tr**2).mean(dim=1))
    prior_sd = float(torch.quantile(per_sample, 0.95))
    pair: ScalePair = rp.build_scale_pair(
        assign, areas, n_coarse, test.T, prior_sd_unresolved=prior_sd
    )
    heldout = float(pair.prolongation.coverage.heldout_error)

    # -- prior-free: can this coarse support carry the observable at all?
    eta = rp.lead_field_energy_retained(G, R, P, areas, whitener=W)

    # physical reference amplitude: the strongest whitened sensor field in the
    # held-out evoked responses.  Focal perturbations and synthetic ensembles
    # are scaled to it so their residuals are quoted in the same noise
    # standard deviations as the real data.
    ref = float(np.linalg.norm((W @ G) @ test, axis=0).max())
    n_ch = G.shape[0]
    pert = rp.perturbational_error(G, R, P, whitener=W)
    pert["reference_whitened_field_norm"] = ref
    pert["median_residual_sd_per_channel_at_evoked_amplitude"] = (
        pert["median"] * ref / np.sqrt(n_ch)
    )
    pert["p10_residual_sd_per_channel_at_evoked_amplitude"] = (
        pert["p10"] * ref / np.sqrt(n_ch)
    )

    # -- §4.2 boundary, twelve ensembles --------------------------------
    D = _geodesics(subject["src"])
    ensembles: list[tuple[str, str, np.ndarray]] = [
        (
            f"evoked:{c}",
            "MNE source estimate of the real evoked response, 0-300 ms"
            + (" (held out)" if c in TEST_CONDITIONS else " (split used to set prior sd)"),
            subject["states"][c],
        )
        for c in subject["states"]
    ]
    for r_mm in (5, 10, 20, 40):
        ensembles.append((
            f"patch:{r_mm}mm",
            f"120 same-signed focal geodesic patches of radius {r_mm} mm, "
            "scaled to the peak held-out evoked field",
            scale_to_reference(
                G, W, patch_states(subject["src"], D, n_fine, r_mm / 1000.0, 60, seed=r_mm), ref
            ),
        ))
    for ell_mm in (5, 10, 20, 40):
        ensembles.append((
            f"grf:{ell_mm}mm",
            f"20 geodesically smooth Gaussian random fields, length scale {ell_mm} mm, "
            "scaled to the peak held-out evoked field",
            scale_to_reference(
                G, W, grf_states(subject["src"], D, n_fine, ell_mm / 1000.0, 10, seed=100 + ell_mm), ref
            ),
        ))

    boundary = []
    for name, desc, X in ensembles:
        obs = rp.observable_error(G, R, P, X, whitener=W)
        bc = rp.BoundaryConsistency(
            ensemble=name,
            description=desc,
            observable=obs,
            tolerance_sd_per_channel=TOLERANCE_SD_PER_CHANNEL,
            tolerance_basis=TOLERANCE_BASIS,
        )
        boundary.append(bc.as_record())
        if verbose:
            print(
                f"  {name:20s} rel {obs['relative_error']:.3f}  "
                f"resid {obs['residual_sd_per_channel_mean']:5.2f} sd/ch  "
                f"signal {obs['signal_sd_per_channel_mean']:5.2f} sd/ch  "
                f"{'PASS' if bc.passes else 'FAIL'}",
                flush=True,
            )

    return rp.PairMeasurement(
        schema_version=rp.SCHEMA_VERSION,
        n_fine=n_fine,
        n_coarse=n_coarse,
        membership_digest=rp.membership_digest(assign, areas, n_coarse),
        authority_policy=rp.AUTHORITY_POLICY,
        coarse_roundtrip_residual=rt,
        coarse_roundtrip_tolerance=1e-9,
        heldout_fine_residual=heldout,
        declared_prior_sd_unresolved=prior_sd,
        landmark_coverage=rp.assigned_area_fraction(assign, areas),
        required_coverage=0.8,
        lead_field_energy_retained=eta,
        fine_characteristic_scale_m=float(np.sqrt(areas.mean())),
        coarse_characteristic_scale_m=float(
            np.sqrt(np.mean([areas[assign == p].sum() for p in range(n_coarse)]))
        ),
        boundary=tuple(boundary),
        perturbational=pert,
        rejected_point_estimate_prior_sd=point_sd,
        provenance={
            "generator": "benchmarks/transforms/resolution_pair.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "subject": "mne-sample::sample",
            "forward": "sample_audvis-eeg-oct-6-fwd.fif (BEM 5120-5120-5120)",
            "source_space": "oct-6, fixed normal orientation, use_cps=True",
            "parcellation": subject["parc_note"],
            **(
                {"label_transfer_max_sphere_mm": subject["label_transfer_max_mm"]}
                if subject["label_transfer_max_mm"] is not None
                else {}
            ),
            "inverse": "MNE, lambda2=1/9, pick_ori='normal'",
            "noise_cov": "sample_audvis-cov.fif",
            "reference": "explicit average reference on lead field and data",
            "channels": len(subject["channels"]),
            "bads_dropped": subject["bads"],
            "train_conditions": list(TRAIN_CONDITIONS),
            "test_conditions": list(TEST_CONDITIONS),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mne": subject["mne_version"],
        },
    )


# --------------------------------------------------------------------------
# specificity: does the boundary metric respond to anything?
# --------------------------------------------------------------------------
def specificity(subject: dict[str, Any]) -> dict[str, Any]:
    """A measurement that cannot distinguish good from bad is decoration.

    Comparisons on the same head, the same lead field and the same held-out
    states: the declared restriction, a finer one of the same construction, and
    the declared parcellation with its spatial structure destroyed.  If the
    metric did not separate these it would not be evidence about anything.

    When the declared pair is not ``aparc``, the 68-parcel Desikan-Killiany
    restriction is measured here too, so the two artefacts can be read against
    each other without comparing numbers from different runs.
    """
    G, W, areas = subject["G"], subject["W"], subject["areas"]
    n_fine = subject["n_fine"]
    X = np.concatenate([subject["states"][c] for c in TEST_CONDITIONS], axis=1)
    out = {}

    def row(tag: str, assign: np.ndarray, nc: int, note: str) -> None:
        R = rp.restriction_matrix(assign, areas, nc)
        P = rp.prolongation_matrix(assign, nc)
        out[tag] = {
            "n_coarse": nc,
            "note": note,
            "observable_relative_error": rp.observable_error(
                G, R, P, X, whitener=W
            )["relative_error"],
            "lead_field_energy_retained": rp.lead_field_energy_retained(
                G, R, P, areas, whitener=W
            ),
        }

    assign = subject["assign"]
    parc = subject["parc"]
    row(f"declared_{parc}", assign, len(subject["parcel_names"]), "the declared pair")
    if parc != "aparc":
        n_dk, a_dk = _membership(
            subject["src"], subject["subjects_dir"], n_fine, "aparc"
        )
        row("aparc", a_dk, len(n_dk),
            "Desikan-Killiany on the same head: the other declared artefact's "
            "coarse support, measured here so the two are comparable")
    n2, a2 = _membership(subject["src"], subject["subjects_dir"], n_fine, "aparc.a2009s")
    row("aparc_a2009s", a2, len(n2), "2.2x more parcels, same construction")
    rng = np.random.default_rng(0)
    sc = assign.copy()
    m = sc >= 0
    sc[m] = rng.permutation(sc[m])
    row("scrambled", sc, len(subject["parcel_names"]),
        "declared parcellation with membership permuted: same sizes, no contiguity")

    # what a restriction that keeps orientation would retain
    nn = np.vstack([s["nn"][s["vertno"]] for s in subject["src"]])
    nc = len(subject["parcel_names"])
    R3 = np.zeros((3 * nc, n_fine))
    for p in range(nc):
        idx = np.where(assign == p)[0]
        for k in range(3):
            R3[3 * p + k, idx] = areas[idx] * nn[idx, k]
    out["net_dipole_moment"] = {
        "n_coarse": 3 * nc,
        "note": "3 numbers per parcel: the parcel's net dipole moment vector",
        "observable_relative_error": None,
        "lead_field_energy_retained": rp.lead_field_energy_retained(
            G, R3, np.linalg.pinv(R3), areas, whitener=W
        ),
    }
    # Does the answer simply need more parcels? A dose-response in parcel count,
    # holding construction fixed: each declared parcel split into k clusters by
    # position, measured rather than extrapolated.
    #
    # An earlier version of this comment said the production model ran 454
    # regions and that no annot at that count existed for this subject, so
    # subdivision was the only way to reach it. Both halves were wrong. The model
    # runs 414 (Schaefer400x7 + 14 Aseg14T subcortical volumes; 454 is the inert
    # `ModelConfig.n_regions` default that run 3's config overrides), and the
    # Schaefer annot does exist in this tree -- `--parc Schaefer400x7` measures
    # the real thing. These rows stay because a dose-response is still worth
    # having, not because the count cannot be reached directly.
    from scipy.cluster.vq import kmeans2

    pos = np.vstack([s["rr"][s["vertno"]] for s in subject["src"]])
    for k in (2, 4, 8):
        sub = np.full(n_fine, -1, np.int64)
        nxt = 0
        for p in range(len(subject["parcel_names"])):
            idx = np.where(assign == p)[0]
            kk = min(k, len(idx))
            _, lab = kmeans2(pos[idx], kk, seed=0, minit="++", missing="raise")
            sub[idx] = nxt + lab
            nxt += kk
        row(f"subdivided_x{k}", sub, nxt,
            f"each declared parcel split into {k} geodesic-position clusters")

    out["best_possible"] = {
        "n_coarse": nc,
        "note": f"any {nc}-dim restriction containing row(G); rank(WG)="
                f"{int(np.linalg.matrix_rank(W @ G))}",
        "observable_relative_error": 0.0,
        "lead_field_energy_retained": 1.0,
    }
    return out


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output json (default: the declared path)")
    ap.add_argument("--no-specificity", action="store_true")
    ap.add_argument(
        "--parc", default="aparc", choices=sorted(PARCELLATIONS),
        help="coarse support. 'aparc' is 68 Desikan-Killiany parcels; "
             "'Schaefer400x7' is 400 of the 414 regions the model runs on.",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    subject = load_subject(parc=args.parc)
    m = measure(subject)
    rec = m.as_record()
    if not args.no_specificity:
        rec["specificity"] = specificity(subject)
    rec["provenance"]["wall_seconds"] = round(time.time() - t0, 1)

    # Each parcellation has its own artefact path, so a run of one cannot land
    # on the other's file. That mattered here: the two answers differ by 6x, and
    # a scratch run writing over the declared artefact is exactly ISSUE-010's
    # shape.
    out = Path(args.out) if args.out else rp.measurement_path(parc=args.parc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")

    print(f"\nwrote {out}")
    print(f"  parcellation {args.parc}, n_coarse {m.n_coarse}")
    print(f"  R P = I to {m.coarse_roundtrip_residual:.3g}          -> paired: {m.roundtrip_ok}")
    print(f"  coverage {m.landmark_coverage:.4f} of cortical area   -> ok: {m.coverage_ok}")
    print(f"  held-out fine residual {m.heldout_fine_residual:.4g} vs declared prior sd "
          f"{m.declared_prior_sd_unresolved:.4g} -> calibrated: {m.prolongation_calibrated}")
    print(f"  lead-field energy retained by the coarse support: {m.lead_field_energy_retained:.4f}")
    print(f"  perturbational median relative error: {m.perturbational['median']:.3f}")
    print(f"  BOUNDARY SUFFICIENT (§4.2): {m.boundary_sufficient}")
    if not args.no_specificity:
        print("\nspecificity (lead-field energy retained):")
        for k, v in rec["specificity"].items():
            print(f"  {k:22s} n={v['n_coarse']:4d}  eta={v['lead_field_energy_retained']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
