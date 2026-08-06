"""Regenerate agent Fisher's committed numbers from source; do not audit the table.

``reports/identifiability/results.json`` was produced on CUDA at
``epoch_seconds=3.0, n_epochs=30, seed=20260805`` (read off
``reports/identifiability/manifest.json``, ``extra.command``).  This test
recomputes ``theta_profile_min_eigenvalue_nonprior`` on **CPU** through this
package's own machinery -- which is also a device-parity check, since the
committed run used a GPU this one may not touch.

Two tolerances, because there are two regimes of conditioning and pretending
otherwise would make the test fail for a reason that is not a defect.  The
well-determined designs reproduce to ~1e-13 *relative*.  The fMRI-only value is
~3e-06 against an information matrix with entries up to ~25 -- the residue of a
Schur complement that cancelled seven orders of magnitude -- so its relative
precision is ~1e-7 while its absolute precision is ~1e-13.  Measured: two CPU
runs of the same computation, differing only in BLAS thread count, gave
``2.9294013073562117e-06`` and ``2.9294015374228234e-06``: 7.9e-08 apart
relatively, 2.3e-13 apart absolutely.  The pass criterion is therefore
``rel < 1e-9 OR abs < 1e-11``, and which one carried each row is recorded.
None of this touches any conclusion: the value is six orders of magnitude below
the EEG one either way.

Marked ``slow``: ~20-60 s per (regime, design).

This reads the **working tree** copy of ``results.json``, which other agents
also write.  That is deliberate: if the benchmark's headline numbers move, the
package that builds on them should fail rather than keep quoting the old ones.
At the time of writing the nine values here are identical in the working tree
and at ``HEAD``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scwbd.individualize.profile import (
    _fisher_for_design,
    benchmark_config,
    profiled_information,
)
from scwbd.infer.identifiability import REGIMES
from scwbd.infer.linear_gaussian import PARAM_INDEX, THETA_NAMES

RESULTS = Path(__file__).resolve().parents[2] / "reports/identifiability/results.json"

CASES = [
    (r.name, d)
    for r in REGIMES
    for d in ("eeg_only", "fmri_only", "joint_native")
]


@pytest.fixture(scope="module")
def committed():
    if not RESULTS.exists():  # pragma: no cover - the file is committed
        pytest.skip(f"{RESULTS} not present")
    return json.loads(RESULTS.read_text())["decision"]["per_regime"]


#: pass if EITHER holds; see the module docstring for why one is not enough.
RTOL = 1e-9
ATOL = 1e-11


@pytest.mark.slow
@pytest.mark.parametrize("regime_name,design", CASES)
def test_theta_profile_reproduces_the_committed_value(regime_name, design, committed):
    regime = next(r for r in REGIMES if r.name == regime_name)
    I, _prov = _fisher_for_design(
        design, cfg=benchmark_config(), regime=regime, eta=None, seed=20260805
    )
    idx = [PARAM_INDEX[t] for t in THETA_NAMES]
    S, _ = profiled_information(I, idx, nuisance_prior=0.0)
    lam = float(np.linalg.eigvalsh(0.5 * (S + S.T)).min())
    want = committed[regime_name]["theta_profile_min_eigenvalue_nonprior"][design]
    rel = abs(lam - want) / max(abs(want), 1e-300)
    absd = abs(lam - want)
    assert rel < RTOL or absd < ATOL, (
        f"{regime_name}/{design}: recomputed {lam!r} vs committed {want!r} "
        f"(rel {rel:.3g}, abs {absd:.3g})"
    )


@pytest.mark.slow
def test_the_tolerance_can_fail():
    """A tolerance that accepts anything is not a tolerance."""
    for lam, want in ((16.0, 16.1), (2.9e-06, 5.0e-06)):
        rel = abs(lam - want) / abs(want)
        absd = abs(lam - want)
        assert not (rel < RTOL or absd < ATOL), (lam, want)


@pytest.mark.slow
def test_the_ordering_the_whole_design_rests_on(committed):
    """EEG ~= joint >> fMRI-only, in EVERY committed regime."""
    for r in REGIMES:
        m = committed[r.name]["theta_profile_min_eigenvalue_nonprior"]
        e, j, f = m["eeg_only"], m["joint_native"], m["fmri_only"]
        assert abs(j - e) / j < 1e-5, (r.name, e, j)
        assert e / f > 1e5, (r.name, e, f)
