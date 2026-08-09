"""Fixtures for the independent audit of the evaluation path (⚖️ Neyman).

Design rule for this directory, taken from ``reports/decorative_guards.md``:

    A skip that reads as green is a decorative guard.

So the fixtures below distinguish *"the artifact does not exist on this
machine"* (an honest skip, which the audit report records as unexercised) from
*"the artifact exists and could not be used"* (a failure).  A silent skip is
never allowed to stand in for a pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "scwbd_001_beta.yaml"

#: The checkpoints that were produced by a CUDA run with ``model.compile: true``.
#: These are the only artifacts on which the ``_orig_mod.`` prefix can appear,
#: and therefore the only ones on which the load-integrity test has content.
COMPILED_CKPTS = (
    Path("/home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta/stage_III_sliced.pt"),
    REPO / "checkpoints" / "scwbd-001-beta" / "last.pt",
)


@pytest.fixture(scope="session")
def cfg():
    from scwbd.foundation.config import load_config

    if not CONFIG.exists():
        pytest.skip(f"released config absent: {CONFIG}")
    return load_config(str(CONFIG))


@pytest.fixture(scope="session")
def real_eeg(cfg):
    """The measured-EEG dataset the evaluation actually consumes."""
    from scwbd.foundation.realdata import EEGMMIDBDataset, RealEEGConfig

    d = cfg.data
    root = Path(d.real_eeg_root)
    if not root.exists():
        pytest.skip(f"measured EEG corpus absent: {root}")
    win = d.window + d.context
    rc = RealEEGConfig(
        window_s=win / d.fs_hz,
        fs_target=d.fs_hz,
        max_subjects=None,
        max_runs_per_subject=None,
        seed=d.seed,
    )
    ds = EEGMMIDBDataset(rc)
    assert len(ds) > 0, (
        f"{root} exists but yielded zero windows; that is a failure, not a skip -- "
        "an audit that cannot see the data it audits must say so loudly"
    )
    return ds


@pytest.fixture(scope="session")
def window_subjects(real_eeg):
    """``real_eeg.window_subjects`` materialised ONCE, as an array.

    ``EEGMMIDBDataset.window_subjects`` is an uncached ``@property`` that rebuilds
    ``[recordings[r]["subject"] for r, _ in window_index]`` -- one entry per
    window, 235k on the released eegmmidb -- on **every access**.  Indexing it
    inside a loop is therefore quadratic.

    That is not hypothetical: ``{real_eeg.window_subjects[i] for i in fold}`` in
    ``test_sampling_representativeness`` rebuilt the full list once per index of a
    13k-window fold and ran for over ten minutes without finishing, which was
    read as "the fixture builds the whole corpus".  The fixture builds in about
    two seconds; the loop was the cost.  Take this array, never the property.

    ``scwbd.foundation.evaluate._window_subject(ds, i)`` is the O(1) single-window
    accessor and is what the evaluation itself uses.
    """
    import numpy as np

    return np.asarray(real_eeg.window_subjects)


@pytest.fixture(scope="session")
def real_split(cfg, real_eeg):
    from scwbd.foundation.realdata import participant_split

    d = cfg.data
    return participant_split(
        real_eeg, test_fraction=d.real_test_fraction, val_fraction=0.1, seed=d.seed
    )


@pytest.fixture(scope="session")
def sim_val(cfg):
    from scwbd.foundation.simulate import SimCorpus

    d = cfg.data
    idx = Path(d.sim_index_fast)
    if not idx.exists():
        pytest.skip(f"simulated corpus index absent: {idx}")
    return SimCorpus(
        str(idx),
        window=d.window + d.context,
        trajectory_subset="val",
        val_fraction=d.val_fraction,
        seed=d.seed,
    )


@pytest.fixture(scope="session")
def compiled_checkpoint():
    """A checkpoint written by a run with ``torch.compile`` active."""
    import torch

    for p in COMPILED_CKPTS:
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if any("_orig_mod." in k for k in payload["model"]):
            return p, payload
    pytest.skip(
        "no checkpoint carrying torch.compile's '_orig_mod.' prefix is present on "
        "this machine; the load-integrity test has no artifact to exercise"
    )
