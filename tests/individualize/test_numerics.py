"""Numerical guards, each watched firing on an input built to break it.

The delay-line guard is the one with a measured failure behind it: at
``n_delay_taps = 10`` and ``tau = 12 ms`` the conduction-delay information reads
``1.79e+25`` instead of ``2.21``, silently, and the inflated reading is the one
that says "spectacularly identifiable".

The Schur tests make the weaker claim honestly: our pseudo-inverse profile is a
genuine Schur complement in the exactly-singular limit, and it **agrees** with
``scwbd.infer.fisher.schur_information`` wherever that one is well posed.  No
defect in the shipped helper was found; the local version exists for its rank
diagnostics and for the nuisance-prior variant.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.individualize.profile import (
    InadequateDelayLine,
    assert_delay_line_adequate,
    benchmark_config,
    profiled_information,
)
from scwbd.infer.fisher import schur_information
from scwbd.infer.linear_gaussian import prior_mean_u


def _singular_nuisance_information(eps: float) -> np.ndarray:
    """A PSD information matrix whose nuisance block has an ``eps`` direction.

    Built as ``J^T J`` so it is PSD by construction, with one nuisance direction
    scaled down to ``eps`` -- the shape an EEG-only design has, where the
    haemodynamic parameters carry no information.
    """
    rng = np.random.default_rng(0)
    n = 5
    J = rng.normal(size=(40, n))
    scale = np.array([1.0, 1.0, 1.0, 1.0, eps])
    J = J * scale.reshape(1, -1)
    return J.T @ J


def test_our_schur_is_a_valid_schur_complement_on_a_singular_block():
    keep = [0, 1]
    I = _singular_nuisance_information(1e-9)

    S_ours, diag = profiled_information(I, keep)
    lam_ours = float(np.linalg.eigvalsh(0.5 * (S_ours + S_ours.T)).min())

    # PSD, and never larger than the unprofiled block: profiling can only
    # remove information, never add it.
    assert lam_ours >= -1e-9, lam_ours
    assert lam_ours <= float(np.linalg.eigvalsh(I[np.ix_(keep, keep)]).min()) + 1e-9
    # the null direction was dropped explicitly and the fact is recorded, so a
    # rank-deficient profile is visible rather than inferred
    assert diag["nuisance_dropped"] >= 1
    assert diag["nuisance_rank"] == 5 - len(keep) - diag["nuisance_dropped"]


@pytest.mark.parametrize("eps", [1.0, 1e-3, 1e-6])
def test_our_schur_agrees_with_the_shipped_helper_when_well_conditioned(eps):
    """Where the block is well posed, the two must not disagree."""
    I = _singular_nuisance_information(eps)
    keep = [0, 1]
    S_ours, _ = profiled_information(I, keep)
    S_ref = schur_information(I, keep)
    assert np.allclose(S_ours, S_ref, rtol=1e-8, atol=1e-10)


def test_exactly_zero_nuisance_block_profiles_to_the_block_itself():
    """The correct limit: profiling out a direction the data ignore removes nothing."""
    n = 4
    I = np.zeros((n, n))
    I[:2, :2] = np.array([[3.0, 1.0], [1.0, 2.0]])
    S, diag = profiled_information(I, [0, 1])
    assert np.allclose(S, I[:2, :2])
    assert diag["nuisance_rank"] == 0


def test_nuisance_prior_variant_is_never_smaller():
    I = _singular_nuisance_information(0.3)
    keep = [0, 1]
    a, _ = profiled_information(I, keep, nuisance_prior=0.0)
    b, _ = profiled_information(I, keep, nuisance_prior=1.0)
    ea = np.linalg.eigvalsh(0.5 * (a + a.T)).min()
    eb = np.linalg.eigvalsh(0.5 * (b + b.T)).min()
    assert eb >= ea - 1e-12


# ------------------------------------------------------------------ delay line
def test_delay_line_guard_fires_on_a_too_short_delay_line():
    """Measured consequence of not having this: lambda_min read 1.9e+25."""
    bad = benchmark_config(n_delay_taps=10)  # tau/dt = 12 > 10
    with pytest.raises(InadequateDelayLine) as e:
        assert_delay_line_adequate(bad, prior_mean_u())
    assert "n_delay_taps" in str(e.value)


def test_delay_line_guard_passes_the_benchmark_configuration():
    assert_delay_line_adequate(benchmark_config(), prior_mean_u())


def test_delay_line_guard_permits_the_deliberate_zero_tap_control():
    """D=0 is the naive-resampling control, where d mu / d tau == 0 honestly."""
    assert_delay_line_adequate(benchmark_config(n_delay_taps=0), prior_mean_u())


def test_delay_line_guard_discriminates_across_the_boundary():
    """The guard must read differently on either side, not always one way."""
    cfg_ok = benchmark_config(n_delay_taps=26)
    cfg_bad = benchmark_config(n_delay_taps=17)  # 12 + 3*2 = 18 needed
    assert_delay_line_adequate(cfg_ok, prior_mean_u())
    with pytest.raises(InadequateDelayLine):
        assert_delay_line_adequate(cfg_bad, prior_mean_u())
