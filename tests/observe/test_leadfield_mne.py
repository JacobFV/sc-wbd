"""Validation against MNE-Python: the external reference implementation.

Three levels, with honest tolerances:

* **MEG / Sarvas** -- MNE uses the same closed form, so agreement is asserted at
  machine precision (``1e-12`` relative).
* **EEG multilayer sphere** -- MNE uses the Berg three-term approximation to the
  same Legendre series our solver evaluates exactly, so agreement is asserted at
  the Berg level (2 % relative topography, 1 % amplitude).  A tighter tolerance
  would be asserting that an approximation is exact.
* **BEM forward** -- the wrapper must reproduce MNE's own gain matrix exactly,
  because it does no numerical work.  The realistic-head test additionally runs
  against agent B's MNE sample subject (real MRI, BEM surfaces, and a
  precomputed forward solution) when it is on disk.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from scwbd.observe.leadfield import BEMLeadField, sarvas_meg

from .conftest import (
    HEAD_RADIUS,
    MNE_SPHERE_RELATIVE_RADII,
    MNE_SPHERE_SIGMAS,
    fsaverage_dir,
    mne_sample_path,
    requires_fsaverage,
    requires_mne_sample,
)

mne = pytest.importorskip("mne")
torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# EEG: exact multilayer series vs MNE's Berg approximation
# --------------------------------------------------------------------------


def test_spherical_eeg_matches_mne_within_berg_approximation(
    four_layer_head, sensor_positions, source_positions, sphere_radii
):
    mine = four_layer_head.potential(source_positions, sensor_positions)
    ref = BEMLeadField.mne_sphere_reference(
        source_positions,
        sensor_positions,
        radii=sphere_radii,
        sigmas=MNE_SPHERE_SIGMAS,
    )
    # MNE's EEG sphere forward is average referenced; ours is referenced to
    # infinity.  Comparing topographies requires the same reference.
    a = mine - mine.mean(0, keepdim=True)
    b = ref - ref.mean(0, keepdim=True)

    rdm = float((a - b).norm() / b.norm())
    assert rdm < 0.02, f"topography differs from MNE by {rdm:.4f} (> Berg level)"

    scale = float((a * b).sum() / (b * b).sum())
    assert abs(scale - 1.0) < 0.01, f"amplitude ratio to MNE is {scale:.5f}"

    # every individual source must agree, not just the aggregate
    for s in range(source_positions.shape[0]):
        r = float((a[:, s] - b[:, s]).norm() / b[:, s].norm())
        assert r < 0.03, f"source {s} topography differs from MNE by {r:.4f}"


# --------------------------------------------------------------------------
# MEG: Sarvas, machine precision
# --------------------------------------------------------------------------


def _magnetometer_info(pos: torch.Tensor, normal: torch.Tensor):
    names = [f"MEG{i + 1:03d}" for i in range(pos.shape[0])]
    info = mne.create_info(names, 1000.0, ch_types="mag")
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[:3] = pos[i].numpy()
        ez = normal[i].numpy()
        tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ex = np.cross(tmp, ez)
        ex /= np.linalg.norm(ex)
        loc[3:6] = ex
        loc[6:9] = np.cross(ez, ex)
        loc[9:12] = ez
        ch["loc"] = loc
        ch["coil_type"] = 2000  # FIFFV_COIL_POINT_MAGNETOMETER
    return info


def test_sarvas_meg_matches_mne_to_machine_precision(source_positions):
    g = torch.Generator().manual_seed(11)
    pos = torch.randn(16, 3, generator=g, dtype=torch.float64)
    pos = pos / pos.norm(dim=1, keepdim=True) * 0.115
    normal = pos / pos.norm(dim=1, keepdim=True)

    mine = sarvas_meg(source_positions, pos, normal)

    info = _magnetometer_info(pos, normal)
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=None, verbose="error")
    src = mne.setup_volume_source_space(
        pos=dict(
            rr=source_positions.numpy(),
            nn=np.tile([[0.0, 0.0, 1.0]], (source_positions.shape[0], 1)),
        ),
        sphere_units="m",
        verbose="error",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, meg=True, eeg=False, verbose="error"
        )
    ref = torch.from_numpy(
        fwd["sol"]["data"].reshape(pos.shape[0], source_positions.shape[0], 3)
    )
    rel = float((mine - ref).norm() / ref.norm())
    assert rel < 1e-12, f"Sarvas MEG differs from MNE by {rel:.3e}"


def test_meg_is_blind_to_radial_sources(source_positions):
    """The structural null space that the ledger declares as a bias term."""
    g = torch.Generator().manual_seed(5)
    pos = torch.randn(24, 3, generator=g, dtype=torch.float64)
    pos = pos / pos.norm(dim=1, keepdim=True) * 0.115
    L = sarvas_meg(source_positions, pos, pos / pos.norm(dim=1, keepdim=True))
    radial = source_positions / source_positions.norm(dim=-1, keepdim=True)
    y = torch.einsum("esk,sk->es", L, radial)
    assert float(y.abs().max()) < 1e-18, "radial sources are not in the MEG null space"


# --------------------------------------------------------------------------
# BEM wrapper: exact reproduction of MNE's gain matrix
# --------------------------------------------------------------------------


def test_bem_wrapper_reproduces_mne_forward_exactly(sensor_positions, source_positions):
    """The wrapper does no numerical work, so it must be bit-for-bit faithful."""
    sp = sensor_positions.numpy()
    names = [f"E{i + 1:03d}" for i in range(sp.shape[0])]
    info = mne.create_info(names, 1000.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={n: p for n, p in zip(names, sp)}, coord_frame="head"
        )
    )
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=HEAD_RADIUS,
        relative_radii=MNE_SPHERE_RELATIVE_RADII,
        sigmas=MNE_SPHERE_SIGMAS,
        verbose="error",
    )
    src = mne.setup_volume_source_space(
        pos=dict(
            rr=source_positions.numpy(),
            nn=np.tile([[0.0, 0.0, 1.0]], (source_positions.shape[0], 1)),
        ),
        sphere_units="m",
        verbose="error",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose="error"
        )

    lf = BEMLeadField.from_mne_forward(fwd, frame="test_head")
    assert lf.modality == "eeg"
    assert lf.sensor_units == "V"
    assert lf.sensor_names == tuple(fwd["sol"]["row_names"])
    np.testing.assert_array_equal(
        lf.as_matrix().numpy(), np.asarray(fwd["sol"]["data"], dtype=np.float64)
    )
    assert lf.ledger is not None and lf.ledger.provenance is not None
    assert lf.as_support().psf.kind == "leadfield"


def test_bem_wrapper_refuses_mixed_modality_forward(sensor_positions, source_positions):
    """EEG volts and MEG tesla must never share one matrix (refusal R01)."""
    from scwbd.observe.base import ObservationRefusal

    sp = sensor_positions.numpy()
    eeg_names = [f"E{i + 1:03d}" for i in range(sp.shape[0])]
    meg_names = [f"MEG{i + 1:03d}" for i in range(sp.shape[0])]
    info = mne.create_info(eeg_names + meg_names, 1000.0, ch_types=["eeg"] * len(eeg_names) + ["mag"] * len(meg_names))
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    info.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={n: p for n, p in zip(eeg_names, sp)}, coord_frame="head"
        ),
        on_missing="ignore",
    )
    for i, ch in enumerate(info["chs"][len(eeg_names):]):
        loc = np.zeros(12)
        loc[:3] = (sensor_positions[i] * 1.3).numpy()
        ez = (sensor_positions[i] / sensor_positions[i].norm()).numpy()
        tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ex = np.cross(tmp, ez)
        ex /= np.linalg.norm(ex)
        loc[3:6] = ex
        loc[6:9] = np.cross(ez, ex)
        loc[9:12] = ez
        ch["loc"] = loc
        ch["coil_type"] = 2000
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=HEAD_RADIUS,
        relative_radii=MNE_SPHERE_RELATIVE_RADII,
        sigmas=MNE_SPHERE_SIGMAS,
        verbose="error",
    )
    src = mne.setup_volume_source_space(
        pos=dict(
            rr=source_positions.numpy(),
            nn=np.tile([[0.0, 0.0, 1.0]], (source_positions.shape[0], 1)),
        ),
        sphere_units="m",
        verbose="error",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=True, verbose="error"
        )
    with pytest.raises(ObservationRefusal) as exc:
        BEMLeadField.from_mne_forward(fwd)
    assert exc.value.code == "R01"


# --------------------------------------------------------------------------
# realistic subject head (agent B's sample dataset)
# --------------------------------------------------------------------------


@requires_mne_sample
def test_precomputed_sample_forward_round_trips():
    """Load MNE's own precomputed subject forward and wrap it losslessly."""
    root = mne_sample_path()
    fwd_path = root / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not fwd_path.exists():
        pytest.skip(f"{fwd_path} not present")
    fwd = mne.read_forward_solution(str(fwd_path), verbose="error")
    fwd_eeg = mne.pick_types_forward(fwd, meg=False, eeg=True)
    lf = BEMLeadField.from_mne_forward(fwd_eeg, frame="sample_subject_head")
    np.testing.assert_array_equal(
        lf.as_matrix().numpy(), np.asarray(fwd_eeg["sol"]["data"], dtype=np.float64)
    )
    assert lf.n_sensors == fwd_eeg["nchan"]
    assert lf.source_positions.shape[0] * (1 if lf.orientation == "fixed" else 1) > 0
    assert lf.ledger is not None
    assert lf.ledger.bias_by_name("coregistration") is not None


def _matched_vs_null(G: torch.Tensor, S: torch.Tensor) -> dict[str, float]:
    """Matched vs permuted-null topography agreement, amplitude-normalised."""
    a = G / G.norm(dim=0, keepdim=True)
    b = S / S.norm(dim=0, keepdim=True)
    C = (a.T @ b).abs()
    matched = torch.diagonal(C)
    off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
    sgn = torch.sign((a * b).sum(0))
    rdm = (a - b * sgn).norm(dim=0)
    return {
        "matched_median_r": float(matched.median()),
        "null_median_r": float(off.median()),
        "null_p95_r": float(off.quantile(0.95)),
        "frac_above_null_p95": float((matched > off.quantile(0.95)).to(torch.float64).mean()),
        "median_rdm": float(rdm.median()),
    }


def _sphere_vs_bem(fwd, n_sources: int = 300) -> dict[str, float]:
    from scwbd.observe.leadfield import SphericalHeadModel

    fixed = mne.convert_forward_solution(fwd, force_fixed=True, verbose="error")
    lf = BEMLeadField.from_mne_forward(fixed)
    G = lf.as_matrix().to(torch.float64)
    G = G - G.mean(0, keepdim=True)
    sensor_pos = lf.sensor_positions.to(torch.float64)
    src_pos = lf.source_positions.to(torch.float64)

    head = SphericalHeadModel.fitted_to(sensor_pos)
    centre = torch.tensor(head.center, dtype=torch.float64)
    depth = (src_pos - centre).norm(dim=-1)
    keep = (depth < head.radii[0] * 0.97).nonzero().flatten()
    if keep.numel() < 50:
        pytest.skip("too few sources inside the fitted brain sphere")
    keep = keep[torch.linspace(0, keep.numel() - 1, min(n_sources, keep.numel())).long()]

    normals = src_pos[keep] - centre
    normals = normals / normals.norm(dim=-1, keepdim=True)
    S = torch.einsum("esk,sk->es", head.potential(src_pos[keep], sensor_pos), normals)
    S = S - S.mean(0, keepdim=True)
    stats = _matched_vs_null(G[:, keep], S)
    stats["fitted_radius_m"] = head.R
    stats["n_sources"] = float(keep.numel())
    return stats


@requires_mne_sample
def test_sphere_does_not_substitute_for_a_subject_bem(capsys):
    """NEGATIVE RESULT, recorded rather than tuned away.

    On the MNE sample subject's own BEM forward, a least-squares fitted
    four-layer sphere reproduces single-source scalp topographies only weakly:
    matched |r| is above a permutation null, so the sphere is not noise, but the
    median RDM is of order 1 and only a small minority of sources beat the
    null's 95th percentile.  The published 10-30 % sphere-vs-BEM figures do not
    bound this use.

    The assertions therefore encode what was measured, not a hoped-for level:
    the sphere carries *some* geometric information, it is decisively *not* a
    substitute head model, and the discrepancy it produces must fall inside the
    interval the sphere's own ledger declares.
    """
    root = mne_sample_path()
    fwd_path = root / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not fwd_path.exists():
        pytest.skip(f"{fwd_path} not present")
    fwd = mne.read_forward_solution(str(fwd_path), verbose="error")
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
    stats = _sphere_vs_bem(fwd)
    with capsys.disabled():
        print(f"\n  sphere vs sample-subject BEM: {stats}")

    # 1. the sphere is not noise: matched agreement beats the permutation null
    assert stats["matched_median_r"] > stats["null_median_r"], (
        f"a fitted sphere carries no source-specific information at all: {stats}"
    )
    # 2. it is nonetheless not a substitute: most sources do not beat the null
    assert stats["frac_above_null_p95"] < 0.5, (
        "the sphere reproduced the subject BEM well enough that the "
        "spherical_geometry_discrepancy bias term would be overstated -- "
        f"re-derive the bound rather than keeping a stale one: {stats}"
    )
    assert stats["median_rdm"] > 0.3, f"unexpectedly small discrepancy: {stats}"

    # 3. the ledger must already declare an interval that covers what we measured
    from scwbd.observe.leadfield import SphericalHeadModel

    lf = BEMLeadField.from_mne_forward(fwd)
    head = SphericalHeadModel.fitted_to(lf.sensor_positions.to(torch.float64))
    sphere_lf = head.lead_field(
        torch.tensor([[0.0, 0.0, 0.04]], dtype=torch.float64) + torch.tensor(head.center),
        lf.sensor_positions.to(torch.float64),
    )
    term = sphere_lf.ledger.bias_by_name("spherical_geometry_discrepancy")
    assert term is not None
    assert term.half_width >= stats["median_rdm"] / 2.0, (
        f"the declared relative bias interval {term.interval} does not cover the "
        f"measured median RDM {stats['median_rdm']:.3f}; the ledger is optimistic"
    )


@requires_mne_sample
def test_subject_bem_forward_amplitudes_are_physiological():
    root = mne_sample_path()
    fwd_path = root / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not fwd_path.exists():
        pytest.skip(f"{fwd_path} not present")
    fwd = mne.read_forward_solution(str(fwd_path), verbose="error")
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="error")
    G = torch.from_numpy(np.asarray(fwd["sol"]["data"], dtype=np.float64))
    G = G - G.mean(0, keepdim=True)
    peak = float((G * 10e-9).abs().max(0).values.median())
    assert 1e-7 < peak < 5e-5, (
        f"median peak scalp potential from a 10 nA*m dipole on the sample "
        f"subject is {peak * 1e6:.2f} uV, outside the physiological band"
    )


def test_least_squares_sphere_fit_recovers_a_known_sphere():
    from scwbd.observe.leadfield import SphericalHeadModel

    g = torch.Generator().manual_seed(31)
    c_true = torch.tensor([0.004, -0.011, 0.031], dtype=torch.float64)
    R_true = 0.0917
    u = torch.randn((200, 3), generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    pts = c_true + R_true * u
    c, R = SphericalHeadModel.fit_sphere(pts)
    assert R == pytest.approx(R_true, abs=1e-9)
    assert torch.allclose(torch.tensor(c, dtype=torch.float64), c_true, atol=1e-9)


# --------------------------------------------------------------------------
# realistic head: fsaverage BEM surfaces
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fsaverage_forward():
    """A real three-layer BEM forward: 64 electrodes on anatomy-derived surfaces."""
    root = fsaverage_dir()
    if root is None:
        pytest.skip("fsaverage not downloaded")
    subj = root / "fsaverage"
    bem_sol = subj / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    src_file = subj / "bem" / "fsaverage-ico-5-src.fif"
    trans = subj / "bem" / "fsaverage-trans.fif"
    for f in (bem_sol, src_file, trans):
        if not f.exists():
            pytest.skip(f"{f} missing")

    montage = mne.channels.make_standard_montage("standard_1020")
    names = montage.ch_names[:64]
    info = mne.create_info(names, 1000.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    src = mne.read_source_spaces(str(src_file), verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = mne.make_forward_solution(
            info,
            trans=str(trans),
            src=src,
            bem=str(bem_sol),
            eeg=True,
            meg=False,
            mindist=5.0,
            verbose="error",
        )
    return fwd


@requires_fsaverage
def test_bem_wrapper_is_lossless_on_a_realistic_head(fsaverage_forward):
    """The realistic-head path, end to end, with no numerical drift in the wrap."""
    fwd = fsaverage_forward
    lf = BEMLeadField.from_mne_forward(fwd, frame="fsaverage_head_RAS")
    assert lf.modality == "eeg"
    assert lf.sensor_units == "V"
    assert lf.orientation == "free"
    assert lf.n_sensors == fwd["nchan"] == 64
    assert lf.n_sources == fwd["nsource"]
    np.testing.assert_array_equal(
        lf.as_matrix().numpy(), np.asarray(fwd["sol"]["data"], dtype=np.float64)
    )
    # ... and the support it emits is the lead field, not an electrode label
    support = lf.as_support()
    assert support.psf is not None and support.psf.kind == "leadfield"
    assert support.psf.matrix.shape == (64, 3 * fwd["nsource"])
    assert support.labels == tuple(fwd["sol"]["row_names"])
    assert lf.ledger is not None
    assert lf.ledger.bias_by_name("segmentation_surface_error") is not None
    for units, led in lf.ledger.to_schema_all().items():
        assert led.has_estimator(), f"realistic-head ledger[{units}] fails R08"


@requires_fsaverage
def test_realistic_head_amplitudes_are_physiological(fsaverage_forward):
    """A 10 nA*m cortical dipole on a real head gives microvolt scalp potentials."""
    fixed = mne.convert_forward_solution(
        fsaverage_forward, force_fixed=True, verbose="error"
    )
    G = torch.from_numpy(np.asarray(fixed["sol"]["data"], dtype=np.float64))
    G = G - G.mean(0, keepdim=True)
    peak = float((G * 10e-9).abs().max(0).values.median())
    assert 1e-7 < peak < 5e-5, (
        f"median peak scalp potential from a 10 nA*m dipole is {peak * 1e6:.2f} uV, "
        "outside the physiological 0.1-50 uV band"
    )


@requires_fsaverage
def test_sphere_vs_fsaverage_bem_is_measured_not_asserted(fsaverage_forward, capsys):
    """The same measurement on a template head, for comparison with the subject.

    fsaverage is more sphere-like than the sample subject, so this is the
    *favourable* case; recording both numbers is what makes the sphere ledger's
    bias bound defensible rather than a citation.
    """
    stats = _sphere_vs_bem(fsaverage_forward)
    with capsys.disabled():
        print(f"\n  sphere vs fsaverage BEM: {stats}")
    assert stats["matched_median_r"] > stats["null_median_r"], (
        f"even on a template head the sphere carries no source-specific "
        f"information: {stats}"
    )
    assert stats["median_rdm"] > 0.2, (
        "sphere and BEM agree essentially perfectly, which would mean the "
        f"spherical_geometry_discrepancy bias term is fictitious: {stats}"
    )
