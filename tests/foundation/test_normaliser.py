"""The window normaliser must not let silence inflate amplitude.

`SimCorpus` originally scaled each window by the median per-region standard
deviation. On this corpus ~6 % of windows have a majority of parcels near-silent,
so the median collapsed while the peak did not, and the window inflated by up to
2427x in real data. The signal that dominates training (`wilson_cowan`, ~76 % of
post-normalisation batch variance) was the worst affected at 11.7 % of its
windows.

These tests are written so that reverting to the bare median makes them fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.foundation.simulate import SCALE_PEAK_FLOOR, normalise_window

T, N = 72, 454


def _window(n_active: int, quiet: float, seed: int = 0) -> np.ndarray:
    """A window with `n_active` normal parcels and the rest near-silent."""
    rng = np.random.default_rng(seed)
    a = np.empty((T, N), dtype=np.float32)
    a[:, :n_active] = rng.standard_normal((T, n_active))
    a[:, n_active:] = rng.standard_normal((T, N - n_active)) * quiet
    return a


# ----------------------------------------------------------------------
# the failure mode itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_active", [1, 5, 10, 50])
def test_silence_cannot_inflate_amplitude(n_active):
    """The exact pathology: most parcels silent, a few normal.

    Under the old rule the median sd is ~1e-6 and `max|z|` reaches ~3.4e6.
    """
    z = normalise_window(_window(n_active, quiet=1e-7))
    assert np.isfinite(z).all()
    assert np.abs(z).max() < 100.0, (
        f"{n_active} active parcels among {N} inflated to max|z|="
        f"{np.abs(z).max():.1f}; the scale collapsed toward the silent majority"
    )


def test_the_old_rule_would_have_failed_this():
    """Guard the premise -- if this stops holding, the tests above go vacuous."""
    a = _window(10, quiet=1e-7)
    ac = a - a.mean(0, keepdims=True)
    sd = ac.std(0, keepdims=True) + 1e-6
    old = np.abs(ac / max(float(np.median(sd)), 1e-6)).max()
    new = np.abs(normalise_window(a)).max()
    assert old > 1e4, f"expected the bare-median rule to blow up, got max|z|={old:.1f}"
    assert new < 100.0
    assert old / new > 1e3


def test_scale_never_falls_below_the_peak_floor():
    """The invariant, stated directly."""
    for n_active in (1, 3, 25, 200, N):
        a = _window(n_active, quiet=1e-7, seed=n_active)
        ac = a - a.mean(0, keepdims=True)
        sd = ac.std(0, keepdims=True) + 1e-6
        implied = np.abs(ac).max() / max(np.abs(normalise_window(a)).max(), 1e-12)
        assert implied >= SCALE_PEAK_FLOOR * float(sd.max()) * 0.999


# ----------------------------------------------------------------------
# it must not disturb windows that were never pathological
# ----------------------------------------------------------------------
def test_homogeneous_windows_are_unchanged_by_the_floor():
    """When parcels are comparably active the floor must not bind.

    This is the other half of the requirement: the fix bounds the tail *and*
    leaves ordinary data alone. Measured shift on non-pathological corpus
    windows was x1.000.
    """
    rng = np.random.default_rng(3)
    a = rng.standard_normal((T, N)).astype(np.float32)
    ac = a - a.mean(0, keepdims=True)
    sd = ac.std(0, keepdims=True) + 1e-6
    assert float(np.median(sd)) > SCALE_PEAK_FLOOR * float(sd.max()), (
        "premise: on homogeneous data the median should exceed the peak floor"
    )
    np.testing.assert_allclose(normalise_window(a), ac / float(np.median(sd)), rtol=1e-6)


def test_output_is_zero_mean_and_finite():
    for quiet in (1e-7, 1e-3, 1.0):
        z = normalise_window(_window(10, quiet=quiet, seed=7))
        assert np.isfinite(z).all()
        assert np.abs(z.mean(0)).max() < 1e-4


def test_scale_invariance():
    """Multiplying the input by a constant must not change the output.

    Measured against **peak amplitude**, not element-wise: a relative comparison
    of individual elements is meaningless where the true value is ~0, and doing
    it that way suggested a 0.4 % violation that does not exist. The deviation is
    ~1e-6 of peak in both a near-`1e-6` and a far-from-`1e-6` regime, i.e. float32
    rounding rather than the additive epsilon in `sd`.
    """
    def dev(a):
        x, y = normalise_window(a * 1000.0), normalise_window(a)
        return float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-12))

    for quiet in (1e-3, 1.0):
        d = dev(_window(10, quiet=quiet, seed=11))
        assert d < 1e-5, f"scale invariance violated at quiet={quiet}: {d:.2e}"


def test_all_silent_window_does_not_produce_nan():
    """Degenerate input must degrade gracefully, not poison the batch."""
    z = normalise_window(np.zeros((T, N), dtype=np.float32))
    assert np.isfinite(z).all()
    assert np.abs(z).max() == 0.0
