"""Two electrode montages as two views of one carrier -- ARCHITECTURE.md 2b O-1.

The test sec. 2b says defines success:

    "two different electrode montages, with different channel counts and
     different lead fields, both declared as views of one carrier, both
     contributing to inference over it."

Everything below runs on the real ``mne-sample::sample`` subject: a real
three-layer BEM forward solution on that subject's MRI, the session's own noise
covariance, and the subject's own ``aparc`` parcellation.  Nothing is quoted
from another report; every number this writes is computed here.

    PYTHONPATH=. python benchmarks/schema/two_montage_views.py

writes ``reports/schema/two_montage_views.json``.

What is measured
----------------
1. **The carrier/view split is expressible.**  One :class:`Carrier` (parcel-level
   net dipole moment, arity 3), two :class:`View`s with different channel
   counts and different lead fields, both checked against the carrier.
2. **Both views contribute.**  Closed-form Gaussian inference over the carrier
   from montage A alone, montage B alone, and both.  The montages are
   hemispherically biased, so each alone is nearly blind to one hemisphere.
3. **The mutation.**  Re-run (2) with montage B replaced by a second copy of
   montage A -- the state the system is in today, where there is no carrier for
   two different views to be views *of*.  If the joint result does not degrade,
   the measurement in (2) was not measuring anything.
4. **Orientation, regenerated from source.**  ``lead_field_energy_retained`` for
   the scalar parcel carrier and the vector parcel carrier, computed through
   :meth:`SupportMap.retained_energy` rather than through
   ``scwbd.transforms.resolution_pair``, so the two code paths can disagree.
5. **The refusal.**  A free-orientation lead field declared as a view of a
   scalar carrier raises :class:`ArityError`.  Today that construction
   typechecks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scwbd.schema.carrier import Carrier, View, ViewOperator  # noqa: E402
from scwbd.schema.ledger import UncertaintyLedger  # noqa: E402
from scwbd.schema.supports import (  # noqa: E402
    PSF,
    ArityError,
    ElementSpec,
    Support,
    TemporalSupport,
)
from scwbd.schema.support_algebra import (  # noqa: E402
    SupportMap,
    common_refinement,
    project_along,
)

FRAME = "subject_surface_RAS"
OUT = Path(__file__).resolve().parents[2] / "reports/schema/two_montage_views.json"


# --------------------------------------------------------------------------
# the subject -- free orientation, because the carrier is a vector field
# --------------------------------------------------------------------------
def load_subject() -> dict[str, Any]:
    import mne

    from scwbd.sources.loaders.mne_datasets import anatomy_paths

    mne.set_log_level("ERROR")
    root = Path(anatomy_paths("mne-sample")["root"])
    meg = root / "MEG" / "sample"
    subjects_dir = root / "subjects"
    if not meg.is_dir():
        raise SystemExit(
            f"MNE sample dataset not found under {root}; this measurement is not "
            "runnable without it and is not approximated with a template."
        )

    raw_fwd = mne.read_forward_solution(meg / "sample_audvis-eeg-oct-6-fwd.fif")
    # surf_ori=True, force_fixed=False -> free orientation with the third
    # component along the surface normal.  The carrier is a vector field; the
    # whole point is not to spend the orientation before the schema sees it.
    fwd = mne.convert_forward_solution(raw_fwd, surf_ori=True, force_fixed=False)
    G3_all = np.asarray(fwd["sol"]["data"], float)  # (n_ch, 3 * n_src)
    # the fixed-orientation forward, built exactly as the transforms benchmark
    # builds it (use_cps=True), so the orientation comparison below asks the
    # same question rather than a similar-sounding one
    fwd_fix = mne.convert_forward_solution(
        raw_fwd, surf_ori=True, force_fixed=True, use_cps=True
    )
    G_fixed_all = np.asarray(fwd_fix["sol"]["data"], float)  # (n_ch, n_src)
    src = fwd["src"]
    ch_fwd = list(fwd["sol"]["row_names"])

    evokeds = mne.read_evokeds(meg / "sample_audvis-ave.fif")
    info = evokeds[0].copy().pick(ch_fwd).info
    good = [c for c in ch_fwd if c not in info["bads"]]
    rows = [ch_fwd.index(c) for c in good]
    G3 = G3_all[rows]
    G_fixed = G_fixed_all[rows]

    cov = mne.pick_channels_cov(
        mne.read_cov(meg / "sample_audvis-cov.fif"), good, ordered=True
    )
    C = np.asarray(cov["data"], float)

    pos = np.array([info["chs"][ch_fwd.index(c)]["loc"][:3] for c in good], float)

    # patch areas and parcel membership, same construction as the transforms
    # benchmark so the two are comparable
    from benchmarks.transforms.resolution_pair import _membership, _patch_areas

    areas = np.concatenate([_patch_areas(s) for s in src])
    n_fine = G3.shape[1] // 3
    names, assign = _membership(src, subjects_dir, n_fine, "aparc")
    nn = np.vstack([s["nn"][s["vertno"]] for s in src])

    return {
        "G3": G3,
        "G_fixed": G_fixed,
        "cov": C,
        "channels": good,
        "sensor_pos": pos,
        "areas": areas,
        "assign": assign,
        "normals": nn,
        "parcel_names": names,
        "n_fine": int(n_fine),
        "mne_version": mne.__version__,
        "bads": list(info["bads"]),
    }


# --------------------------------------------------------------------------
# the declared objects
# --------------------------------------------------------------------------
def _ledger(units: str) -> UncertaintyLedger:
    return UncertaintyLedger(
        variance={"measurement": 0.0},
        bias_interval=(0.0, 0.0),
        bias_status="design_estimable",
        validity_domain={"subject": "mne-sample::sample", "units": units},
    )


def build_carrier(subject: dict[str, Any]) -> tuple[Carrier, np.ndarray, Support]:
    """The parcel-level vector carrier, and the prolongation onto the fine mesh.

    The carrier is *not* the fine mesh: the model's state lives at parcels.  The
    lead field is defined on the mesh, so the view operator is the fine lead
    field composed with the indicator prolongation parcels -> mesh, which is the
    map the model already applies implicitly whenever it treats a region
    variable as describing that region.  Here it is built by the algebra.
    """
    n_fine = subject["n_fine"]
    assign = subject["assign"]
    n_parcel = len(subject["parcel_names"])

    vec = ElementSpec.vector3("A*m", FRAME, label="net dipole moment")
    fine = Support(
        kind="surface_vertex",
        frame=FRAME,
        units="A*m",
        psf=PSF(kind="point", fwhm=(0.0,), units="A*m"),
        n_elements=n_fine,
        element=vec,
        label="cortical_source_dipole",
        resolution="cortical_source_dipole",
    )
    parcel = Support(
        kind="parcel",
        frame=FRAME,
        units="A*m",
        psf=PSF(
            kind="integration_kernel",
            fwhm=(float(np.sqrt(subject["areas"].sum() / n_parcel)),),
            units="A*m",
        ),
        n_elements=n_parcel,
        element=vec,
        label="parcel",
        resolution="parcel",
    )

    # the atom set is the fine mesh; parcels are one support over it, the mesh
    # itself is the other. common_refinement of a support with the atom set
    # returns the atom set, and its `from_a` is exactly the indicator fill.
    ref = common_refinement(
        parcel,
        fine,
        assign,
        np.arange(n_fine),
        atom_weights=subject["areas"],
        label="parcel^dipole",
    )
    P = ref.from_a  # parcels -> refinement cells (== assigned fine dipoles)

    carrier = Carrier(
        id="cortical_moment_field",
        support=parcel,
        frame=FRAME,
        element_ids=tuple(subject["parcel_names"]),
        priors=("connectome:desikan", "anatomy:surface_normals"),
        ledger=_ledger("A*m"),
        label="parcel net dipole moment; the field every montage is a view of",
    )
    return carrier, P.matrix, ref.support


def make_view(
    view_id: str,
    carrier: Carrier,
    channels: list[str],
    gain: np.ndarray,
    label: str,
) -> View:
    volt = ElementSpec.scalar("V", label="scalp potential")
    support = Support(
        kind="sensor",
        frame="subject_head_RAS",
        units="V",
        psf=PSF(kind="lead_field", fwhm=None, units="V", kernel_ref=f"{view_id}:G"),
        n_elements=len(channels),
        element=volt,
        label=view_id,
    )
    op = ViewOperator(
        kind="lead_field",
        shape=(gain.shape[0], gain.shape[1]),
        units="V/(A*m)",
        operator_ref=f"{view_id}:G",
        psf=support.psf,
        label=label,
    )
    v = View(
        id=view_id,
        carrier=carrier.id,
        operator=op,
        support=support,
        temporal=TemporalSupport(clock="eeg_amp", dt=1.0 / 600.0),
        ledger=_ledger("V"),
        label=label,
    )
    v.check_against(carrier)
    return v


# --------------------------------------------------------------------------
# inference over the carrier
# --------------------------------------------------------------------------
def posterior(
    gains: list[np.ndarray], whiteners: list[np.ndarray], prior_sd: float, n_dof: int
) -> np.ndarray:
    """Posterior precision of ``m`` under ``y_k = G_k m + e_k``, ``e_k ~ N(0, C_k)``.

    Whitened, so each view enters in units of its own instrument's noise -- a
    view with a noisier amplifier contributes less, which is the only defensible
    way to combine two montages.
    """
    A = np.eye(n_dof) / (prior_sd**2)
    for G, W in zip(gains, whiteners):
        WG = W @ G
        A = A + WG.T @ WG
    return A


def infer(
    gains: list[np.ndarray],
    whiteners: list[np.ndarray],
    ys: list[np.ndarray],
    prior_sd: float,
    n_dof: int,
) -> dict[str, Any]:
    A = posterior(gains, whiteners, prior_sd, n_dof)
    rhs = np.zeros((n_dof, ys[0].shape[1]))
    for G, W, y in zip(gains, whiteners, ys):
        rhs = rhs + (W @ G).T @ (W @ y)
    mean = np.linalg.solve(A, rhs)
    # effective number of resolved directions: tr(I - prior_prec A^-1)
    Ainv = np.linalg.inv(A)
    dof_resolved = float(n_dof - np.trace(Ainv) / (prior_sd**2))
    sign, logdet = np.linalg.slogdet(A)
    return {
        "mean": mean,
        "resolved_dof": dof_resolved,
        "log_det_precision": float(logdet) if sign > 0 else float("-inf"),
        "posterior_sd_mean": float(np.sqrt(np.clip(np.diag(Ainv), 0, None)).mean()),
    }


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-draws", type=int, default=64)
    args = ap.parse_args(argv)

    t0 = time.time()
    s = load_subject()
    G3 = s["G3"]
    n_fine = s["n_fine"]
    n_parcel = len(s["parcel_names"])
    print(
        f"subject sample: {G3.shape[0]} good electrodes (dropped {s['bads']}), "
        f"{n_fine} fine dipoles x 3, {n_parcel} parcels, "
        f"cortex {s['areas'].sum() * 1e4:.0f} cm^2",
        flush=True,
    )

    carrier, P, refined = build_carrier(s)
    print(
        f"carrier {carrier.id!r}: {carrier.n_elements} elements, arity "
        f"{sorted(set(carrier.arities()))}, n_dof {carrier.n_dof}; "
        f"prolongation onto {P.shape[0] // 3} assigned dipoles",
        flush=True,
    )

    # --- the parcel-level lead field, by composition ----------------------
    # G3 columns are (src, component) in that order; P rows are (cell, component).
    # The refinement dropped unassigned dipoles, so select their columns.
    assign = s["assign"]
    keep = np.where(assign >= 0)[0]
    cols = (keep[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
    G_parcel = G3[:, cols] @ P  # (n_ch, 3 * n_parcel)
    print(
        f"composed parcel lead field {G_parcel.shape}; "
        f"{n_fine - keep.size} unassigned dipoles excluded, never averaged in",
        flush=True,
    )

    # --- two montages ------------------------------------------------------
    pos = s["sensor_pos"]
    chans = s["channels"]
    order = np.argsort(pos[:, 0])  # left-right axis
    idx_a = np.sort(order[: int(0.55 * len(order))])  # left-biased, 32 ch
    idx_b = np.sort(order[int(0.55 * len(order)) :])  # right-biased, 27 ch

    def montage(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        n = idx.size
        H = np.eye(n) - np.ones((n, n)) / n  # each montage's own average ref
        G = H @ G_parcel[idx]
        C = H @ s["cov"][np.ix_(idx, idx)] @ H.T
        # whiten with this montage's own noise covariance, on its own rank
        ev, U = np.linalg.eigh(C)
        k = ev > ev.max() * 1e-10
        W = (U[:, k] / np.sqrt(ev[k])).T
        return G, W, [chans[i] for i in idx]

    GA, WA, chA = montage(idx_a)
    GB, WB, chB = montage(idx_b)
    print(
        f"montage A: {len(chA)} channels (x from {pos[idx_a,0].min()*1e3:.0f} to "
        f"{pos[idx_a,0].max()*1e3:.0f} mm), lead field {GA.shape}\n"
        f"montage B: {len(chB)} channels (x from {pos[idx_b,0].min()*1e3:.0f} to "
        f"{pos[idx_b,0].max()*1e3:.0f} mm), lead field {GB.shape}",
        flush=True,
    )

    view_a = make_view("montage_A", carrier, chA, GA, "left-biased 10-20 subset")
    view_b = make_view("montage_B", carrier, chB, GB, "right-biased subset")
    print(
        f"declared views: {view_a.id} ({view_a.operator.shape}) and "
        f"{view_b.id} ({view_b.operator.shape}); both check_against"
        f"({carrier.id!r}) -> ok",
        flush=True,
    )

    # --- the refusal that today's LeadField cannot raise -------------------
    scalar_carrier = Carrier(
        id="cortical_scalar_field",
        support=Support(
            kind="parcel",
            frame=FRAME,
            units="A*m",
            n_elements=n_parcel,
            element=ElementSpec.scalar("A*m", projected_along="surface_normal"),
            label="parcel_scalar",
        ),
        frame=FRAME,
        element_ids=tuple(s["parcel_names"]),
    )
    # the same free-orientation lead field, declared as a view of the *scalar*
    # carrier.  This is the construction today's `LeadField(n_channels,
    # n_regions)` cannot tell apart from the correct one.
    mis = view_a.model_copy(update={"id": "montage_A_mis", "carrier": scalar_carrier.id})
    refusal: dict[str, Any]
    try:
        mis.check_against(scalar_carrier)
    except ArityError as e:
        refusal = {"raised": "ArityError", "message": str(e), "remedy": e.remedy}
        print(f"REFUSED as designed: {str(e)[:150]}", flush=True)
    else:  # pragma: no cover - the guard would be decorative
        refusal = {"raised": None}
        print("!! the arity guard did NOT fire", flush=True)

    # --- inference ---------------------------------------------------------
    n_dof = carrier.n_dof
    rng = np.random.default_rng(0)
    # Fix the prior scale by the SNR the instrument actually has: draw carrier
    # states whose whitened sensor field is 1.9 noise sd per channel, which is
    # the amplitude of the real evoked response on this subject
    # (reports/transforms/resolution_pair.md sec. 3.2). A prior chosen for
    # numerical convenience would make every comparison below a statement about
    # the prior instead of about the montages.
    TARGET_SD_PER_CHANNEL = 1.9
    WGA0, WGB0 = WA @ GA, WB @ GB
    both = np.vstack([WGA0, WGB0])
    prior_sd = float(
        TARGET_SD_PER_CHANNEL * np.sqrt(both.shape[0] / np.sum(both**2))
    )
    print(f"prior sd fixed by SNR: {prior_sd:.4g} A*m per carrier dof", flush=True)
    M = rng.normal(0.0, prior_sd, size=(n_dof, args.n_draws))
    yA = WA @ (GA @ M) + rng.normal(0, 1, size=(WA.shape[0], args.n_draws))
    yB = WB @ (GB @ M) + rng.normal(0, 1, size=(WB.shape[0], args.n_draws))
    # already whitened; pass identity whiteners with the whitened gains
    IA = np.eye(WA.shape[0])
    IB = np.eye(WB.shape[0])
    WGA, WGB = WA @ GA, WB @ GB

    def rmse(mean: np.ndarray) -> float:
        return float(np.sqrt(np.mean((mean - M) ** 2)))

    # which carrier dof belong to which hemisphere -- each montage is blind to
    # the far side, and that is the physical content of "both views contribute"
    hemi = np.array(
        [1 if nm.endswith("-rh") else 0 for nm in s["parcel_names"]], dtype=int
    )
    hemi_dof = np.repeat(hemi, 3)

    def hemi_resolved(gs, ws) -> tuple[float, float]:
        A = posterior(gs, ws, prior_sd, n_dof)
        d = np.diag(np.linalg.inv(A))
        r = 1.0 - d / (prior_sd**2)
        return float(r[hemi_dof == 0].sum()), float(r[hemi_dof == 1].sum())

    res: dict[str, Any] = {}
    for tag, gs, ws, obs in (
        ("A_only", [WGA], [IA], [yA]),
        ("B_only", [WGB], [IB], [yB]),
        ("A_and_B", [WGA, WGB], [IA, IB], [yA, yB]),
    ):
        r = infer(gs, ws, obs, prior_sd, n_dof)
        lh, rh = hemi_resolved(gs, ws)
        res[tag] = {
            "rmse_in_prior_sd": rmse(r["mean"]) / prior_sd,
            "resolved_dof": r["resolved_dof"],
            "resolved_dof_lh": lh,
            "resolved_dof_rh": rh,
            "log_det_precision": r["log_det_precision"],
            "posterior_sd_over_prior_sd": r["posterior_sd_mean"] / prior_sd,
            "n_channels": int(sum(g.shape[0] for g in gs)),
        }
        print(
            f"  {tag:9s} n_ch={res[tag]['n_channels']:3d} "
            f"resolved_dof={r['resolved_dof']:7.3f} "
            f"(lh {lh:6.3f} / rh {rh:6.3f})  "
            f"err={res[tag]['rmse_in_prior_sd']:.4f} prior-sd  "
            f"post_sd/prior_sd={res[tag]['posterior_sd_over_prior_sd']:.4f}",
            flush=True,
        )

    # --- the mutation: two views of the same thing ------------------------
    yA2 = WA @ (GA @ M) + rng.normal(0, 1, size=(WA.shape[0], args.n_draws))
    r = infer([WGA, WGA], [IA, IA], [yA, yA2], prior_sd, n_dof)
    lh, rh = hemi_resolved([WGA, WGA], [IA, IA])
    res["A_and_A_mutation"] = {
        "rmse_in_prior_sd": rmse(r["mean"]) / prior_sd,
        "resolved_dof": r["resolved_dof"],
        "resolved_dof_lh": lh,
        "resolved_dof_rh": rh,
        "log_det_precision": r["log_det_precision"],
        "posterior_sd_over_prior_sd": r["posterior_sd_mean"] / prior_sd,
        "n_channels": 2 * int(WGA.shape[0]),
        "note": "montage B replaced by a second independent recording of montage "
        "A -- MORE channels than A_and_B, on one support instead of two",
    }
    print(
        f"  mutation  n_ch={res['A_and_A_mutation']['n_channels']:3d} "
        f"resolved_dof={r['resolved_dof']:7.3f} "
        f"(lh {lh:6.3f} / rh {rh:6.3f})  "
        f"err={res['A_and_A_mutation']['rmse_in_prior_sd']:.4f} prior-sd",
        flush=True,
    )

    # --- orientation, regenerated through the algebra ---------------------
    # Deliberately the *same question* reports/transforms/resolution_pair.md
    # sec. 3.5 asks, on the same fixed-orientation lead field and the same
    # prior, so that the two code paths can disagree. A different question with
    # a similar name would be a re-derivation that cannot fail.
    areas = s["areas"][keep]
    nn = s["normals"][keep]
    n_cells = int(keep.size)
    Gfix = s["G_fixed"][:, keep]  # (n_ch, n_cells), normal-oriented
    n = len(chans)
    H = np.eye(n) - np.ones((n, n)) / n
    Cf = H @ s["cov"] @ H.T
    ev, U = np.linalg.eigh(Cf)
    kk = ev > ev.max() * 1e-10
    Wf = (U[:, kk] / np.sqrt(ev[kk])).T
    Gw_fix = Wf @ (H @ Gfix)
    w_sca = 1.0 / np.maximum(areas, 1e-30)

    fine_scalar = Support(
        kind="surface_vertex",
        frame=FRAME,
        units="A*m",
        n_elements=n_cells,
        element=ElementSpec.scalar("A*m", projected_along="surface_normal"),
        label="cortical_source_dipole",
    )
    parcel_sca = Support(
        kind="parcel",
        frame=FRAME,
        units="A*m",
        n_elements=n_parcel,
        element=ElementSpec.scalar("A*m", projected_along="surface_normal"),
        label="parcel_scalar",
    )
    from scwbd.schema.support_algebra import embed_along, restriction_between

    # (a) the declared pair: area-weighted parcel mean of the normal amplitude
    R_sca = restriction_between(fine_scalar, parcel_sca, assign[keep], areas)
    eta_sca = R_sca.retained_energy(Gw_fix, prior_weights=w_sca)

    # (b) the parcel net dipole moment: area-weighted SUM of the embedded
    #     vector field. Built as `sum . embed_along(normals)` -- the embedding is
    #     admissible only because the scalar declared projected_along.
    fine_vec, emb = embed_along(fine_scalar, nn, name="surface_normal")
    parcel_vec = Support(
        kind="parcel",
        frame=FRAME,
        units="A*m",
        n_elements=n_parcel,
        element=ElementSpec.vector3("A*m", FRAME),
        label="parcel_moment",
    )
    import scipy.sparse as sp

    rows = np.repeat(assign[keep] * 3, 3) + np.tile(np.arange(3), n_cells)
    colsv = np.arange(3 * n_cells)
    S_sum = sp.csr_matrix(
        (np.repeat(areas, 3), (rows, colsv)), shape=(3 * n_parcel, 3 * n_cells)
    )
    R_mom = SupportMap(
        src=fine_scalar,
        dst=parcel_vec,
        matrix=(S_sum @ emb.matrix),
        direction="restriction",
        method="area-weighted sum of the embedded normal moment (3 per parcel)",
        manufactures_dof=False,
        lowers_arity=False,
        unresolved_rank=0,
    )
    eta_mom = R_mom.retained_energy(Gw_fix, prior_weights=w_sca)
    print(
        f"  regenerated on the fixed-orientation lead field, prior white per "
        f"unit area:\n"
        f"    scalar parcel mean   ({R_sca.matrix.shape[0]:3d} dof): eta = "
        f"{eta_sca:.4f}   (filed 0.0561)\n"
        f"    parcel dipole moment ({R_mom.matrix.shape[0]:3d} dof): eta = "
        f"{eta_mom:.4f}   (filed 0.5171)\n"
        f"    ratio {eta_mom / eta_sca:.2f}x",
        flush=True,
    )

    # (c) the same two questions asked of the *free-orientation* carrier the
    #     views above actually observe. A different question, reported as one.
    Gw_free = Wf @ (H @ G3[:, cols])
    w_vec = np.repeat(w_sca, 3)
    fine_vec_free = Support(
        kind="surface_vertex",
        frame=FRAME,
        units="A*m",
        n_elements=n_cells,
        element=ElementSpec.vector3("A*m", FRAME),
        label="cortical_source_dipole_free",
    )
    R_vec_free = restriction_between(
        fine_vec_free,
        Support(
            kind="parcel",
            frame=FRAME,
            units="A*m",
            n_elements=n_parcel,
            element=ElementSpec.vector3("A*m", FRAME),
            label="parcel_moment",
        ),
        assign[keep],
        areas,
    )
    _, proj = project_along(fine_vec_free, nn, name="surface_normal")
    R_sca_free = SupportMap(
        src=fine_vec_free,
        dst=parcel_sca,
        matrix=R_sca.matrix @ proj.matrix,
        direction="projection",
        method="parcel mean of the normal component of a free-orientation field",
        manufactures_dof=False,
        lowers_arity=True,
        projected_along="surface_normal",
        unresolved_rank=0,
    )
    eta_free_vec = R_vec_free.retained_energy(Gw_free, prior_weights=w_vec)
    eta_free_sca = R_sca_free.retained_energy(Gw_free, prior_weights=w_vec)
    print(
        f"  free-orientation carrier (isotropic 3-component prior):\n"
        f"    scalar parcel mean   ( 68 dof): eta = {eta_free_sca:.4f}\n"
        f"    vector parcel mean   (204 dof): eta = {eta_free_vec:.4f}",
        flush=True,
    )

    record = {
        "generated_by": "benchmarks/schema/two_montage_views.py",
        "subject": "mne-sample::sample",
        "mne_version": s["mne_version"],
        "n_fine_dipoles": n_fine,
        "n_assigned_dipoles": int(keep.size),
        "n_parcels": n_parcel,
        "carrier": {
            "id": carrier.id,
            "n_elements": carrier.n_elements,
            "arities": sorted(set(carrier.arities())),
            "n_dof": carrier.n_dof,
            "content_hash": carrier.content_hash(),
        },
        "views": {
            "montage_A": {
                "n_channels": len(chA),
                "channels": chA,
                "operator_shape": list(view_a.operator.shape),
                "content_hash": view_a.content_hash(),
            },
            "montage_B": {
                "n_channels": len(chB),
                "channels": chB,
                "operator_shape": list(view_b.operator.shape),
                "content_hash": view_b.content_hash(),
            },
        },
        "lead_fields_differ": {
            "frobenius_A": float(np.linalg.norm(GA)),
            "frobenius_B": float(np.linalg.norm(GB)),
            "shapes": [list(GA.shape), list(GB.shape)],
        },
        "inference": res,
        "arity_refusal": refusal,
        "orientation": {
            "whitener": "session noise covariance sample_audvis-cov.fif",
            "fixed_orientation_carrier": {
                "question": "identical to reports/transforms/resolution_pair.md "
                "sec. 3.5: fixed-orientation fine lead field, prior white per "
                "unit cortical area on the scalar normal amplitude",
                "eta_parcel_scalar": eta_sca,
                "eta_parcel_scalar_filed": 0.0561,
                "eta_parcel_dipole_moment": eta_mom,
                "eta_parcel_dipole_moment_filed": 0.5171,
                "ratio": eta_mom / eta_sca if eta_sca > 0 else None,
                "n_dof_scalar": int(R_sca.matrix.shape[0]),
                "n_dof_moment": int(R_mom.matrix.shape[0]),
                "moment_restriction_built_as": R_mom.method,
            },
            "free_orientation_carrier": {
                "question": "the carrier the two views above actually observe: "
                "free-orientation dipoles, isotropic 3-component prior. A "
                "different question; reported so it is not confused with the one "
                "above",
                "eta_parcel_scalar": eta_free_sca,
                "eta_parcel_vector": eta_free_vec,
            },
        },
        "wall_time_s": round(time.time() - t0, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out} in {record['wall_time_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
