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


#: Per-group theta/nuisance profiles at the benchmark configuration, reference
#: regime.  These are the numbers printed in ``reports/individualize/README.md``
#: sec. 2; pinning them here is what stops the report and the code drifting
#: apart.  Regenerate with ``python -m scwbd.individualize.cli table``.
REFERENCE_GROUPS = {
    "prior": {
        "coupling": 0.0,
        "conduction_delay": 0.0,
        "eeg_lead_field": 0.0,
        "hemodynamic": 0.0,
    },
    "eeg_only": {
        "coupling": 16.009321214072735,
        "conduction_delay": 59.14982189298154,
        "eeg_lead_field": 329.5198580471902,
        "hemodynamic": 0.0,
    },
    "fmri_only": {
        "coupling": 2.9392138322281394e-06,
        "conduction_delay": 1.2057609304989076e-05,
        "eeg_lead_field": 0.0,
        "hemodynamic": 8.86669245179386e-09,
    },
    "joint_native": {
        "coupling": 16.009344448904702,
        "conduction_delay": 59.150607172235574,
        "eeg_lead_field": 329.5661676163677,
        "hemodynamic": 2.598263060032391e-08,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("design", sorted(REFERENCE_GROUPS))
def test_per_group_table_matches_the_report(design):
    from scwbd.individualize.groups import LIKELIHOOD_GROUPS

    regime = REGIMES[0]
    assert regime.name == "reference"
    I, _ = _fisher_for_design(
        design, cfg=benchmark_config(), regime=regime, eta=None, seed=20260805
    )
    for g in LIKELIHOOD_GROUPS:
        S, _ = profiled_information(I, g.index, nuisance_prior=0.0)
        lam = float(np.linalg.eigvalsh(0.5 * (S + S.T)).min())
        want = REFERENCE_GROUPS[design][g.name]
        rel = abs(lam - want) / max(abs(want), 1e-300)
        assert rel < RTOL or abs(lam - want) < ATOL, (
            f"{design}/{g.name}: {lam!r} vs reported {want!r}"
        )


@pytest.mark.slow
def test_the_two_statistics_disagree_only_where_the_report_says_they_do():
    """The report claims 45/48 agreement; the reference regime's share is 15/16."""
    from scwbd.individualize.groups import LIKELIHOOD_GROUPS
    from scwbd.individualize.profile import IdentifiabilityThresholds

    th = IdentifiabilityThresholds()
    disagree = []
    for design in sorted(REFERENCE_GROUPS):
        I, _ = _fisher_for_design(
            design, cfg=benchmark_config(), regime=REGIMES[0], eta=None,
            seed=20260805,
        )
        for g in LIKELIHOOD_GROUPS:
            a, _ = profiled_information(I, g.index, nuisance_prior=0.0)
            b, _ = profiled_information(I, g.index, nuisance_prior=1.0)
            sa = th.classify(float(np.linalg.eigvalsh(0.5 * (a + a.T)).min()))
            sb = th.classify(float(np.linalg.eigvalsh(0.5 * (b + b.T)).min()))
            if sa != sb:
                disagree.append((design, g.name, sa, sb))
    assert disagree == [
        ("fmri_only", "coupling", "not_identifiable", "weakly_identifiable")
    ], disagree


#: Measured, per regime: ``(joint - eeg)/joint`` and ``eeg/fmri``.  Bounds are
#: set from these rather than picked: the EEG-vs-joint gap is largest in
#: ``low_snr_short_delay`` at 8.58e-05, and the EEG-vs-fMRI ratio is smallest in
#: ``weak_coupling_long_delay`` at 3.93e+06.  A first version of this test used
#: 1e-05 for the gap and failed on ``low_snr_short_delay`` -- the bound was
#: chosen before the number was looked at, which is the wrong order.
MAX_EEG_JOINT_GAP = 1e-4
MIN_EEG_FMRI_RATIO = 1e6


@pytest.mark.slow
def test_the_ordering_the_whole_design_rests_on(committed):
    """EEG ~= joint >> fMRI-only, in EVERY committed regime."""
    gaps, ratios = {}, {}
    for r in REGIMES:
        m = committed[r.name]["theta_profile_min_eigenvalue_nonprior"]
        e, j, f = m["eeg_only"], m["joint_native"], m["fmri_only"]
        gaps[r.name] = abs(j - e) / j
        ratios[r.name] = e / f
        assert gaps[r.name] < MAX_EEG_JOINT_GAP, (r.name, e, j)
        assert ratios[r.name] > MIN_EEG_FMRI_RATIO, (r.name, e, f)
    # the bounds must not be so loose they could not fail: the worst measured
    # values have to sit within an order of magnitude of them
    assert max(gaps.values()) > 0.1 * MAX_EEG_JOINT_GAP, gaps
    assert min(ratios.values()) < 10 * MIN_EEG_FMRI_RATIO, ratios
