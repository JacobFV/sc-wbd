"""The haemodynamic loader, exercised on real BOLD bytes.

``VolumeSeries`` existed and was never constructed; this file is the evidence
that it now is, and that its refusals fire.  Every assertion runs against
files on disk — no synthetic NIfTI stands in for a real one except in the
negative controls, where a *deliberately broken* file is the point.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from scwbd.sources import registry
from scwbd.sources.loaders.base import NativeSupportError, VolumeSeries
from scwbd.sources.loaders.bids_bold import (
    BoldReadError,
    bold_sidecar,
    frame_from_nifti,
    iter_bold,
    load_bold_run,
    load_physio,
)

nib = pytest.importorskip("nibabel")


def _bold_paths(dataset_id: str, n: int = 3):
    entry = registry.get(dataset_id)
    if not entry.local_path.exists():
        pytest.skip(f"{dataset_id} not on disk at {entry.local_path}")
    paths = list(iter_bold(entry.local_path))[:n]
    if not paths:
        pytest.skip(f"{dataset_id}: no *_bold.nii.gz under {entry.local_path}")
    return paths


# ---------------------------------------------------------------------------
# real bytes
# ---------------------------------------------------------------------------
def test_real_bold_runs_load_at_their_native_tr():
    for p in _bold_paths("ds000117"):
        v = load_bold_run(p, source_id="ds000117")
        assert isinstance(v, VolumeSeries)
        assert v.tr and v.tr > 0
        assert v.meta["n_volumes"] > 1
        assert v.meta["sfreq_hz"] == pytest.approx(1.0 / v.tr)
        assert v.frame_id != "unknown", f"{p}: no coordinate frame in the NIfTI header"
        assert v.affine.shape == (4, 4)


def test_bold_is_far_slower_than_any_electrophysiology_source():
    """§2.6 non-nesting, measured rather than asserted.

    The whole point of a haemodynamic source is that it is on a clock two to
    three decades slower than the electrophysiology.  If this ever came out
    equal, something resampled.
    """
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    eeg_rate = 160.0  # eegmmidb, the run-1 likelihood corpus
    assert v.meta["sfreq_hz"] < eeg_rate / 100


def test_bold_clock_is_never_an_amplifier_clock():
    """A silent concat of BOLD and EEG must be impossible to spell."""
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    assert v.clock_id.startswith("mri_scanner:")
    assert "amp" not in v.clock_id


def test_header_only_is_the_default_and_costs_no_voxels():
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    assert v.data.size == 0
    assert v.meta["n_volumes_loaded"] == 0
    assert len(v.meta["shape"]) == 4


def test_requesting_volumes_returns_the_real_voxels():
    p = _bold_paths("ds000117")[0]
    v = load_bold_run(p, source_id="ds000117", volumes=slice(0, 4))
    x, y, z, _ = v.meta["shape"]
    assert v.data.shape == (x, y, z, 4)
    assert np.isfinite(v.data).all()
    assert v.data.max() > 0, "an all-zero BOLD volume is not data"


def test_sidecar_inheritance_finds_a_root_level_tr():
    """ds000117 states RepetitionTime once, at the dataset root, for 18 runs."""
    p = _bold_paths("ds000117")[0]
    side = bold_sidecar(p)
    assert "RepetitionTime" in side
    assert side["_sidecars"], "no sidecar was located at all"


def test_tr_is_corroborated_by_two_independent_statements():
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    assert v.meta["tr_sidecar"] is not None
    assert v.meta["tr_header"] is not None
    assert abs(v.meta["tr_sidecar"] - v.meta["tr_header"]) < 1e-3


def test_lineage_carries_the_participant_for_grouping():
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    assert v.lineage.participant and v.lineage.participant != "unknown"
    assert v.lineage.family == f"singleton:{v.lineage.participant}"


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_a_missing_tr_is_refused_not_defaulted(tmp_path):
    """The failure this refusal exists for: a plausible TR nobody measured."""
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 5), dtype=np.float32), np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, 0.0))
    img.header.set_xyzt_units("mm", "unknown")
    out = tmp_path / "sub-99_task-x_bold.nii.gz"
    nib.save(img, out)
    with pytest.raises(BoldReadError, match="Refusing to assume a TR"):
        load_bold_run(out, source_id="test")


def test_disagreeing_tr_statements_are_refused(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 5), dtype=np.float32), np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    img.header.set_xyzt_units("mm", "sec")
    out = tmp_path / "sub-99_task-x_bold.nii.gz"
    nib.save(img, out)
    (tmp_path / "sub-99_task-x_bold.json").write_text(json.dumps({"RepetitionTime": 3.0}))
    with pytest.raises(BoldReadError, match="disagrees"):
        load_bold_run(out, source_id="test")


def test_a_3d_volume_is_not_a_time_series(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    out = tmp_path / "sub-99_T1w.nii.gz"
    nib.save(img, out)
    with pytest.raises(BoldReadError, match="4-D"):
        load_bold_run(out, source_id="test")


def test_the_voxel_grid_refuses_to_be_resampled():
    v = load_bold_run(_bold_paths("ds000117")[0], source_id="ds000117")
    with pytest.raises(NativeSupportError):
        v.resample(2.0)


def test_an_unstated_coordinate_frame_reads_as_unknown(tmp_path):
    """Frame code 0 must not acquire a plausible-sounding default."""
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4))
    img.header["sform_code"] = 0
    img.header["qform_code"] = 0
    assert frame_from_nifti(img.header) == ("unknown", "none")


# ---------------------------------------------------------------------------
# physiological channels (the interoceptive series §6.1 asks for)
# ---------------------------------------------------------------------------
def _physio_paths(dataset_id: str, n: int = 2):
    entry = registry.get(dataset_id)
    if not entry.local_path.exists():
        pytest.skip(f"{dataset_id} not on disk")
    # Only the files that carry a `recording-` entity: those are the ones a
    # BIDS sidecar can be resolved for. ds000113 also ships physio files with
    # no such entity and no sidecar anywhere, and load_physio refuses them by
    # design -- that refusal has its own test below.
    paths = [p for p in sorted(entry.local_path.rglob("*recording-*_physio.tsv.gz"))
             if "derivatives" not in p.parts][:n]
    if not paths:
        pytest.skip(f"{dataset_id}: no *_physio.tsv.gz on disk")
    return paths


def test_physio_loads_on_its_own_clock_at_its_own_rate():
    for p in _physio_paths("ds000113"):
        r = load_physio(p, source_id="ds000113")
        assert r.sfreq > 0
        assert r.n_channels == len(r.channel_names)
        assert r.clock_id.startswith("physio_amp:")
        assert r.duration > 0


def test_the_sidecarless_physio_in_ds000113_is_actually_refused():
    """The refusal, exercised on the real files that trigger it.

    ds000113's ses-auditoryperception physio files carry no `recording-`
    entity and no sidecar matches them at any level of the tree. A loader that
    guessed they matched the cardresp sidecar would produce a plausible rate
    and plausible column names for data whose rate and columns nobody states.
    """
    entry = registry.get("ds000113")
    if not entry.local_path.exists():
        pytest.skip("ds000113 not on disk")
    orphans = [p for p in entry.local_path.rglob("*_physio.tsv.gz")
               if "recording-" not in p.name]
    if not orphans:
        pytest.skip("no sidecar-less physio on disk")
    with pytest.raises(BoldReadError, match="Columns"):
        load_physio(orphans[0], source_id="ds000113")


def test_physio_without_a_sidecar_is_refused(tmp_path):
    out = tmp_path / "sub-99_task-x_physio.tsv.gz"
    with gzip.open(out, "wt") as fh:
        fh.write("0.1\t0.2\n0.3\t0.4\n")
    with pytest.raises(BoldReadError, match="Columns"):
        load_physio(out, source_id="test")


def test_physio_column_count_must_match_the_sidecar(tmp_path):
    out = tmp_path / "sub-99_task-x_physio.tsv.gz"
    with gzip.open(out, "wt") as fh:
        fh.write("0.1\t0.2\n0.3\t0.4\n")
    (tmp_path / "sub-99_task-x_physio.json").write_text(
        json.dumps({"SamplingFrequency": 100.0, "StartTime": 0.0,
                    "Columns": ["cardiac", "respiratory", "trigger"]})
    )
    with pytest.raises(BoldReadError, match="names 3 columns"):
        load_physio(out, source_id="test")
