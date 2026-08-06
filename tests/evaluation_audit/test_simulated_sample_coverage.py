"""E5 -- the simulated-validation head slice, and what it can and cannot report.

``posterior_calibration``, ``backend_comparison`` and ``_sim_val_nll`` all build
``DataLoader(trainer.sim_val, shuffle=False)`` and stop after a fixed number of
items.  ``SimCorpus.items`` is built shard-by-shard, so a head slice is a head
slice of the *shard order*, not a sample of the fold.

``backend_comparison`` is the sharpest case: it exists solely to report a
per-backend breakdown, and its slice contains zero windows from two of the five
backends.  It does write ``None`` for them (which is the right behaviour -- the
null case writes something), but the row it cannot compute is exactly the row a
reader would use to decide whether the operator generalises across mechanistic
families.
"""

from __future__ import annotations

import numpy as np
import pytest

# (label, number of leading windows the evaluation consumes)
SLICES = [
    ("posterior_calibration(n_datasets=512)", 512),
    ("posterior_calibration(quick, n_datasets=128)", 128),
    ("backend_comparison(max_batches=6, bs=64)", 6 * 64),
    ("backend_comparison(quick, max_batches=2)", 2 * 64),
    ("_sim_val_nll(max_batches=8, bs=64)", 8 * 64),
    ("_sim_val_nll(quick, max_batches=2)", 2 * 64),
]


def _backend_of_item(sim_val) -> np.ndarray:
    return np.asarray([b for _, _, b in sim_val.items])


@pytest.mark.parametrize("label,n_take", SLICES)
def test_every_backend_appears_in_the_evaluated_slice(sim_val, label, n_take):
    """E5 verdict test.  Fails while the head slice omits whole backends."""
    names = list(sim_val.backend_names)
    bi = _backend_of_item(sim_val)
    head = bi[:n_take]
    counts = {names[j]: int((head == j).sum()) for j in range(len(names))}
    absent = sorted(k for k, v in counts.items() if v == 0)
    full = {names[j]: int((bi == j).sum()) for j in range(len(names))}
    assert not absent, (
        f"{label} reads the first {min(n_take, bi.size)} of {bi.size} validation "
        f"windows with shuffle=False and sees {counts}; backends {absent} are "
        f"entirely absent although they are {sum(full[k] for k in absent)} of "
        f"{bi.size} windows ({100 * sum(full[k] for k in absent) / bi.size:.1f}%) "
        f"in the fold."
    )


def test_head_slice_backend_mix_matches_the_fold(sim_val):
    """Even where a backend is present, the slice must not re-weight the fold.

    ``sim_val_nll`` is quoted as the model's held-out simulated forecast score,
    and ``source_ablation`` adjudicates negative transfer with it.  A slice whose
    backend mix differs from the fold's answers a different question.
    """
    names = list(sim_val.backend_names)
    bi = _backend_of_item(sim_val)
    head = bi[: 8 * 64]
    p_head = np.array([(head == j).mean() for j in range(len(names))])
    p_full = np.array([(bi == j).mean() for j in range(len(names))])
    tv = 0.5 * np.abs(p_head - p_full).sum()
    assert tv < 0.05, (
        f"the _sim_val_nll slice has total-variation distance {tv:.3f} from the "
        f"fold's backend composition: "
        f"{dict(zip(names, p_head.round(3)))} vs {dict(zip(names, p_full.round(3)))}"
    )


def test_simcorpus_returns_the_same_window_for_the_same_index(sim_val):
    """``source_ablation``'s arms must be scored on identical data.

    ``SimCorpus.__getitem__`` draws its time offset from the *global* numpy RNG
    (``np.random.randint``), so the same index yields a different window on every
    call.  ``source_ablation`` calls ``_sim_val_nll`` once per arm, after a
    training stage that has consumed an arm-dependent amount of randomness, so
    the arms are compared on different windows and the deltas that decide
    ``negative_transfer`` carry an unmeasured resampling term.
    """
    import numpy as np

    np.random.seed(0)
    a = np.asarray(sim_val[0]["activity"])
    np.random.seed(1)
    b = np.asarray(sim_val[0]["activity"])
    assert np.array_equal(a, b), (
        "SimCorpus[i] is not a pure function of i: the window offset comes from "
        "the global numpy RNG. Two evaluation arms therefore see different data, "
        "and source_ablation reports the sign of their difference with no "
        "interval and no repeated seeds."
    )
