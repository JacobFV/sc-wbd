"""The anatomical prior must be real, or the failure must be loud.

SC-WBD-001-beta trained on a synthetic ellipsoid. `_from_agent_c` looked for
`obj.weights` where `BrainPrior` exposes `obj.structural.weights`; the
`AttributeError` was swallowed by a bare `except Exception` and `_synthetic_prior`
was substituted. Every provenance record said `synthetic_fallback` correctly and
nothing read them.

The second defect mattered more: the adapter read `ei_prior`, while `BrainPrior`
exposes `ei_ratio_prior()`. That returned **None rather than raising**, so a
rename-only fix would have produced a real ENIGMA connectome with **no receptor
E/I at all** — silently, again.

Every test here is written to fail against the pre-fix code.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy

anatomy_pkg = pytest.importorskip("scwbd.anatomy")


@pytest.fixture(scope="module")
def real():
    return load_anatomy(device="cpu")


# ----------------------------------------------------------------------
# the primary defect
# ----------------------------------------------------------------------
def test_load_anatomy_is_biological_when_assets_are_present(real):
    """The whole finding, as one assertion.

    Fails against pre-fix code, which returned `synthetic_fallback` here.
    """
    assert real.is_biological(), (
        f"load_anatomy() returned provenance={real.provenance!r} with scwbd.anatomy "
        "importable — the synthetic fallback is in force and this model carries no "
        "biological content"
    )
    assert real.provenance != "synthetic_fallback"


def test_parcel_count_is_the_real_atlas_not_the_synthetic_spec(real):
    """454 is the synthetic spec; the real atlas is 414.

    The count was the visible tell that went unread for an entire run.
    """
    assert real.n_regions == 414, (
        f"expected the real 414-parcel atlas, got {real.n_regions}"
        + (" — 454 is the synthetic prior's spec" if real.n_regions == 454 else "")
    )


# ----------------------------------------------------------------------
# the second defect: a prior that is absent must not become a constant
# ----------------------------------------------------------------------
def test_ei_prior_is_present_and_not_constant(real):
    """`ei_ratio_prior()` must actually reach the model.

    A rename-only fix yields a real connectome with a *constant* E/I prior,
    because the old adapter substituted `torch.ones(n)` when the lookup missed.
    A constant carries no regional information and is indistinguishable from
    "not requested" downstream.
    """
    # This test is about the REAL prior. Without this guard it passes against the
    # synthetic fallback too -- which also has a non-constant E/I gradient by
    # construction -- so it would assert nothing about the defect it exists for.
    assert real.is_biological(), "prior is synthetic; this test cannot discriminate"
    ei = real.ei_prior
    assert ei is not None and ei.numel() == real.n_regions
    assert torch.isfinite(ei).all()
    assert ei.min() != ei.max(), (
        "E/I prior is constant — the receptor-derived prior did not reach the "
        "model even though the connectome did"
    )


def test_timescale_prior_is_present_and_not_constant(real):
    # This test is about the REAL prior. Without this guard it passes against the
    # synthetic fallback too -- which also has a non-constant E/I gradient by
    # construction -- so it would assert nothing about the defect it exists for.
    assert real.is_biological(), "prior is synthetic; this test cannot discriminate"
    ts = real.timescale_prior
    assert ts is not None and ts.numel() == real.n_regions
    assert torch.isfinite(ts).all()
    assert ts.min() != ts.max(), "timescale prior is constant; it did not reach the model"


def test_connectome_and_tract_lengths_are_real(real):
    # This test is about the REAL prior. Without this guard it passes against the
    # synthetic fallback too -- which also has a non-constant E/I gradient by
    # construction -- so it would assert nothing about the defect it exists for.
    assert real.is_biological(), "prior is synthetic; this test cannot discriminate"
    assert real.weights.shape == (real.n_regions, real.n_regions)
    assert real.tract_length.shape == (real.n_regions, real.n_regions)
    assert float(real.weights.sum()) > 0
    assert float(real.tract_length.max()) > 1.0, "tract lengths look degenerate"
    assert 0.0 < real.density() < 1.0


# ----------------------------------------------------------------------
# the fallback must require a decision
# ----------------------------------------------------------------------
def test_fallback_requires_an_explicit_flag(real):
    """`force_fallback=True` is the only way to get the synthetic prior."""
    synth = load_anatomy(device="cpu", force_fallback=True)
    assert synth.provenance == "synthetic_fallback"
    assert not synth.is_biological()
    assert synth.n_regions != real.n_regions, (
        "the synthetic and real priors should differ in parcel count (454 vs 414); "
        "if they match, the tell that exposed this defect no longer exists"
    )


def test_an_adapter_failure_raises_rather_than_degrading(monkeypatch):
    """Guard the premise: a broken adapter must not silently fall back.

    This is the exact shape of the original defect — an exception inside the
    adapter, with the package importable.
    """
    import scwbd.foundation.anatomy as A

    def boom(obj, device):
        raise AttributeError("BrainPrior exposes no weights/connectome")

    monkeypatch.setattr(A, "_from_agent_c", boom)
    with pytest.raises(RuntimeError, match="could not be adapted"):
        A.load_anatomy(device="cpu")

    # ...and the escape hatch still works, because it is explicit
    assert A.load_anatomy(device="cpu", force_fallback=True).provenance == "synthetic_fallback"


def test_missing_ei_prior_raises_rather_than_defaulting_to_ones(monkeypatch):
    """The second defect, asserted directly.

    Pre-fix, a `BrainPrior` without a resolvable E/I prior yielded
    `torch.ones(n)` — a silent constant. It must raise instead.
    """
    import scwbd.foundation.anatomy as A

    class Structural:
        weights = torch.rand(8, 8).numpy()
        distance_mm = (torch.rand(8, 8) * 50).numpy()

    class Stub:
        structural = Structural()
        labels = [f"p{i}" for i in range(8)]
        timescale_prior = staticmethod(lambda: [type("P", (), {"mean": 0.1 + 0.01 * i})() for i in range(8)])
        # no ei_ratio_prior / ei_prior at all

    with pytest.raises(AttributeError, match="no E/I prior"):
        A._from_agent_c(Stub(), torch.device("cpu"))
