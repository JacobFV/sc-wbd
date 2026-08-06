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

import math
import warnings

import numpy as np
import pytest
import torch

from scwbd.observe.leadfield import BEMLeadField, sarvas_meg

from .conftest import (
    HEAD_RADIUS,
    MNE_SPHERE_RELATIVE_RADII,
    MNE_SPHERE_SIGMAS,
    mne_sample_path,
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


@requires_mne_sample
def test_spherical_model_and_subject_bem_agree_in_gross_topography():
    """A sphere is a geometry model, not the subject's head -- quantify the gap.

    This test *records* the discrepancy that the ledger's
    ``spherical_geometry_discrepancy`` bias term claims to bound, rather than
    asserting the two agree.
    """
    root = mne_sample_path()
    fwd_path = root / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not fwd_path.exists():
        pytest.skip(f"{fwd_path} not present")
    fwd = mne.read_forward_solution(str(fwd_path), verbose="error")
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="error")

    bem_lf = BEMLeadField.from_mne_forward(fwd, frame="sample_subject_head")
    G = bem_lf.as_matrix().to(torch.float64)
    G = G - G.mean(0, keepdim=True)

    sensor_pos = bem_lf.sensor_positions.to(torch.float64)
    src_pos = bem_lf.source_positions.to(torch.float64)
    centre = sensor_pos.mean(0)
    R = float((sensor_pos - centre).norm(dim=-1).mean())

    from scwbd.observe.leadfield import ITIS_CONDUCTIVITY, SphericalHeadModel

    head = SphericalHeadModel.adult_four_layer(
        R, ITIS_CONDUCTIVITY, center=tuple(float(c) for c in centre)
    )
    keep = ((src_pos - centre).norm(dim=-1) < head.radii[0] * 0.97).nonzero().flatten()[:200]
    if keep.numel() == 0:
        pytest.skip("no sources inside the fitted brain sphere")
    normals = (src_pos[keep] - centre)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    S = head.potential(src_pos[keep], sensor_pos)
    S = torch.einsum("esk,sk->es", S, normals)
    S = S - S.mean(0, keepdim=True)

    Gk = G[:, keep]
    corr = torch.stack(
        [
            torch.corrcoef(torch.stack([Gk[:, i], S[:, i]]))[0, 1].abs()
            for i in range(keep.numel())
        ]
    )
    assert float(corr.median()) > 0.5, (
        "sphere and subject BEM topographies are unrelated; the sphere model is "
        f"not usable even as an approximation (median |r| = {float(corr.median()):.3f})"
    )
    # and the residual is large enough that the ledger's discrepancy flag is honest
    assert float(corr.median()) < 0.999
