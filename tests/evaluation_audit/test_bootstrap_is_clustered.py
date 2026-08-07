"""E6 -- verifying the component that was expected to be broken and is not.

``baselines.bootstrap_ci`` is load-bearing for every interval in the final
report.  A window-level bootstrap mislabelled as participant-clustered would
understate every interval by the design effect ``sqrt(1 + (m-1) rho)`` and would
be invisible in the output.

A "clean" verdict asserted is worth nothing, so the failure is constructed: the
same values are passed with and without ``groups`` and the widths are required
to differ by the predicted factor.  ``test_the_check_can_fail`` then confirms
this test could have come out the other way -- without it, this file would be
the very thing ``reports/decorative_guards.md`` catalogues.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.foundation.baselines import bootstrap_ci


def _clustered_values(n_sub=27, m=40, rho=0.15, seed=0):
    """Per-window values with a known intra-participant correlation ``rho``."""
    rng = np.random.default_rng(seed)
    sd_b = np.sqrt(rho)
    sd_w = np.sqrt(1.0 - rho)
    subj = rng.normal(0.0, sd_b, size=n_sub)
    vals = np.concatenate([s + rng.normal(0.0, sd_w, size=m) for s in subj])
    groups = np.repeat([f"P{i:03d}" for i in range(n_sub)], m)
    return vals, groups, rho, m


def test_point_estimate_matches_the_reported_headline():
    """The interval and ``nll_per_sample`` must centre on the same statistic."""
    v, g, _, _ = _clustered_values()
    point, lo, hi = bootstrap_ci(v, g, n_boot=1000, seed=0)
    assert point == pytest.approx(float(v.mean()))
    assert lo < point < hi


def test_clustering_widens_the_interval_by_the_design_effect():
    """The constructed failure: if this were a window bootstrap, the ratio is 1."""
    v, g, rho, m = _clustered_values()
    _, clo, chi = bootstrap_ci(v, g, n_boot=4000, seed=0)
    _, wlo, whi = bootstrap_ci(v, None, n_boot=4000, seed=0)
    ratio = (chi - clo) / (whi - wlo)
    predicted = np.sqrt(1.0 + (m - 1) * rho)
    assert ratio == pytest.approx(predicted, rel=0.25), (
        f"observed width ratio {ratio:.2f}x, design effect predicts "
        f"{predicted:.2f}x for rho={rho}, m={m}. A ratio near 1.0 would mean "
        f"bootstrap_ci is resampling windows, not participants."
    )


def test_whole_participants_enter_or_leave_together():
    """A cluster bootstrap must be able to omit a participant entirely.

    Constructed so it cannot pass by accident: one participant carries a huge
    offset, so a replicate that drops it is far from the point estimate. Under a
    window bootstrap that participant's windows are always partly present and
    the replicate spread collapses.
    """
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(0, 0.01, 200), rng.normal(50.0, 0.01, 200)])
    g = np.array(["A"] * 200 + ["B"] * 200)
    _, clo, chi = bootstrap_ci(v, g, n_boot=4000, seed=0)
    _, wlo, whi = bootstrap_ci(v, None, n_boot=4000, seed=0)
    assert (chi - clo) > 10.0, (
        f"clustered interval width {chi - clo:.3f} on a two-participant sample "
        f"whose participants differ by 50 units: the resampler is not drawing "
        f"whole participants"
    )
    assert (whi - wlo) < 10.0


def test_paired_statistics_share_bootstrap_draws():
    """``compare`` pairs models through a common draw matrix; verify it is common."""
    from scwbd.foundation.baselines import _boot_draws

    a = _boot_draws(27, 500, 0)
    b = _boot_draws(27, 500, 0)
    assert np.array_equal(a, b), "draws are not reproducible from the seed alone"
    assert not np.array_equal(a, _boot_draws(27, 500, 1))


def test_the_check_can_fail():
    """Recommendation 1: break it on purpose and confirm the alarm sounds.

    With ``rho = 0`` the design effect is 1 and the clustered and window
    intervals must agree; if this assertion did not hold, the widening test
    above would be measuring something other than clustering.
    """
    v, g, _, _ = _clustered_values(rho=1e-9, seed=5)
    _, clo, chi = bootstrap_ci(v, g, n_boot=4000, seed=0)
    _, wlo, whi = bootstrap_ci(v, None, n_boot=4000, seed=0)
    assert (chi - clo) / (whi - wlo) == pytest.approx(1.0, rel=0.25)
