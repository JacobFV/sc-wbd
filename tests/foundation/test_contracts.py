"""Contract tests for ``scwbd.foundation``: shapes, units, determinism, refusals.

These are the executable form of the claim gates for agent I's module
(ARCHITECTURE.md §4).  A test here is not a formality: each one corresponds to a
statement in the thesis that would otherwise be unverifiable prose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from scwbd.foundation.anatomy import CONTROL_GRAPHS, EVIDENCE_CLASSES, load_anatomy
from scwbd.foundation.config import FoundationConfig, load_config
from scwbd.foundation.heads import build_lead_field, gaussian_nll
from scwbd.foundation.model import SCWBD
from scwbd.foundation.simulate import THETA_NAMES, ThetaPrior
from scwbd.foundation.state import default_layout, scalar_layout
from scwbd.foundation.util import set_determinism

REPO = Path(__file__).resolve().parents[2]


def tiny_config() -> FoundationConfig:
    cfg = load_config(REPO / "configs" / "scwbd_ci_smoke.yaml")
    cfg.model.n_regions = 60
    return cfg


@pytest.fixture(scope="module")
def anat():
    """The small synthetic prior these contract tests were written against.

    ``force_fallback=True`` is **required**, not decoration. Without it
    ``load_anatomy`` now finds the real ``scwbd.anatomy`` and returns the
    414-parcel Schaefer/Tian prior, silently ignoring ``n_cortex=40`` and the
    rest — so every test in this module was running against a different object
    than the one it names, and
    :func:`test_fallback_anatomy_is_labelled_as_not_biological` was asserting a
    provenance that could no longer occur.
    """
    return load_anatomy(
        device="cpu",
        n_cortex=40,
        n_subcortex=12,
        n_cerebellum=8,
        density=0.15,
        seed=7,
        force_fallback=True,
    )


@pytest.fixture(scope="module")
def real_anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def model(anat):
    set_determinism(0)
    return SCWBD(tiny_config().model, anat)


# ----------------------------------------------------------------------
# state layout
# ----------------------------------------------------------------------
def test_state_is_structured_not_a_scalar():
    """ARCHITECTURE.md §5: one scalar per region is the *named failure mode*."""
    lay = default_layout()
    assert lay.dim > 1
    for required in ("rate_e", "rate_i", "adaptation", "spectral", "hemo", "uncertainty"):
        assert required in lay, f"structured state is missing {required!r}"
    assert scalar_layout().dim == 1  # retained only as an ablation baseline


def test_layout_offsets_are_disjoint_and_cover_the_state():
    lay = default_layout()
    seen = set()
    for c in lay:
        a, b = lay.span(c.name)
        assert b > a
        assert not (seen & set(range(a, b))), f"{c.name} overlaps another component"
        seen |= set(range(a, b))
    assert seen == set(range(lay.dim))


def test_layout_round_trips_through_json():
    lay = default_layout()
    again = type(lay).from_dict(json.loads(json.dumps(lay.as_dict())))
    assert again.dim == lay.dim
    assert [c.name for c in again] == [c.name for c in lay]
    assert [c.units for c in again] == [c.units for c in lay]


def test_every_component_declares_units_and_a_clock():
    for c in default_layout():
        assert c.units, f"{c.name} has no units"
        assert c.clock in ("fast", "slow", "meta")


# ----------------------------------------------------------------------
# anatomy
# ----------------------------------------------------------------------
def test_fallback_anatomy_is_labelled_as_not_biological(anat):
    """A synthetic connectome must never be able to masquerade as anatomy."""
    assert anat.n_regions == 60, "the fixture must actually build the synthetic prior it asks for"
    assert anat.provenance == "synthetic_fallback"
    assert anat.is_biological() is False
    assert "NOT ANATOMY" in anat.source_note.upper()


def test_real_anatomy_is_labelled_as_biological(real_anat):
    """The other direction, without which the guard above cannot discriminate.

    A test that only ever sees the synthetic prior passes whether or not
    ``is_biological()`` can return True, so it does not distinguish anatomy from
    a stand-in — which is the exact distinction the synthetic-prior incident was
    about. This asserts the real prior is recognised, so the pair fails if either
    label sticks to the wrong object.
    """
    assert real_anat.n_regions != 60
    assert real_anat.provenance != "synthetic_fallback"
    assert real_anat.is_biological() is True


def test_a_real_prior_cannot_be_silently_replaced_by_the_fallback(monkeypatch):
    """``load_anatomy`` must REFUSE, not substitute, when the real prior breaks.

    This is the incident itself: the adapter raised, the exception was swallowed,
    and SC-WBD-001-beta trained on a synthetic ellipsoid while every provenance
    field said so correctly and nothing read them.
    """
    import importlib

    import scwbd.foundation.anatomy as fa

    def boom(name):
        if name == "scwbd.anatomy":
            raise AttributeError("simulated adapter breakage")
        return importlib.import_module(name)

    monkeypatch.setattr(fa.importlib, "import_module", boom)
    with pytest.raises(RuntimeError, match="Refusing to substitute the synthetic prior"):
        fa.load_anatomy(device="cpu")
    # ...and the escape hatch still works, because it is a DECLARATION
    assert fa.load_anatomy(device="cpu", force_fallback=True).is_biological() is False


def test_delays_follow_length_over_velocity(anat):
    d_fast = anat.delay_matrix(10.0)
    d_slow = anat.delay_matrix(2.0)
    m = anat.weights > 0
    assert torch.allclose(d_slow[m], d_fast[m] * 5.0, rtol=1e-5)
    assert float(d_fast[m].min()) >= 0.0


def test_control_graphs_preserve_density(anat):
    n = anat.n_regions
    base = (anat.weights > 0).sum().item()
    for kind in CONTROL_GRAPHS:
        g = anat.control_graph(kind, seed=1)
        assert g.shape == (n, n)
        assert float(torch.diagonal(g).abs().max()) == 0.0, f"{kind} has self-loops"
        if kind in ("randomized", "distance_matched", "local_only"):
            got = (g > 0).sum().item()
            assert 0.5 * base <= got <= 1.5 * base, f"{kind} density {got} vs {base}"


def test_edges_carry_an_evidence_class(anat):
    present = anat.weights > 0
    cls = anat.evidence_class[present]
    assert cls.min() >= 0 and cls.max() < len(EVIDENCE_CLASSES)
    assert (anat.evidence_class[~present] == -1).all()


# ----------------------------------------------------------------------
# shapes and units
# ----------------------------------------------------------------------
def test_rollout_shapes_and_dtypes(model, anat):
    B, T, C = 3, 6, 4
    y = torch.randn(B, C, anat.n_regions)
    th = ThetaPrior().sample(B, seed=1)
    r = model.rollout(y_context=y, theta=th, n_steps=T, with_hemo=True)
    assert r.state.shape == (B, T, anat.n_regions, model.layout.dim)
    assert r.activity.shape == (B, T, anat.n_regions)
    assert r.activity_logvar.shape == r.activity.shape
    assert torch.isfinite(r.state).all()
    assert r.state.dtype == torch.float32


def test_eeg_head_shapes_and_leadfield_is_a_buffer_not_a_parameter(model):
    B, T = 2, 5
    x = torch.randn(B, T, model.n_regions, model.layout.dim)
    mu, lv = model.eeg(x)
    assert mu.shape == (B, T, len(model.eeg.channel_names))
    assert lv.shape == mu.shape
    names = [n for n, _ in model.eeg.named_parameters()]
    assert "L" not in names, "physics is compiled, not fitted: the lead field must be a buffer"
    assert model.eeg.lead_field_meta["individual_head_model"] is False


def test_bold_head_is_stable_and_shrinks_to_literature_values(model):
    B, N = 2, model.n_regions
    h = model.bold.initial(B, N, torch.device("cpu"))
    assert h.shape == (B, N, 4)
    drive = torch.randn(B, N) * 0.1
    for _ in range(50):
        h = model.bold.step(h, drive)
    assert torch.isfinite(h).all()
    y, lv = model.bold.signal(h)
    assert y.shape == (B, N) and lv.shape == (B, N)
    assert float(model.bold.prior_penalty()) == pytest.approx(0.0, abs=1e-6)


def test_gaussian_nll_masks_rather_than_imputes():
    """Rule 1: missing data is never imputed as zero."""
    y = torch.randn(4, 8, 3)
    mu = torch.zeros_like(y)
    lv = torch.zeros_like(y)
    mask = torch.zeros_like(y)
    mask[:, :4] = 1.0
    masked = gaussian_nll(y, mu, lv, mask=mask)
    only_obs = gaussian_nll(y[:, :4], mu[:, :4], lv[:, :4])
    assert float(masked) == pytest.approx(float(only_obs), rel=1e-5)


# ----------------------------------------------------------------------
# causality and delays
# ----------------------------------------------------------------------
def test_coupling_never_reads_the_future(model):
    """The delay line is causal: lag 0 reads the newest message, never a later one."""
    cpl = model.coupling
    B, N, C = 2, model.n_regions, cpl.message_dim
    hist = torch.zeros(B, cpl.max_lag_steps + 1, N, C)
    hist[:, -1] = 1.0  # only the newest slice is non-zero
    lags = torch.zeros(B, cpl.n_bins, dtype=torch.long)
    out0 = cpl(hist, lags)
    lags_far = torch.full((B, cpl.n_bins), cpl.max_lag_steps, dtype=torch.long)
    out_far = cpl(hist, lags_far)
    assert float(out0.abs().sum()) > 0.0
    assert float(out_far.abs().sum()) == pytest.approx(0.0, abs=1e-6)


def test_delay_bins_scale_inversely_with_velocity(model):
    slow = model.coupling.lags_for_velocity(torch.tensor([2.0]))
    fast = model.coupling.lags_for_velocity(torch.tensor([10.0]))
    assert (slow >= fast).all()
    assert int(slow.max()) > int(fast.max())


def test_connectome_mask_is_respected(model, anat):
    """An absent structural edge must contribute exactly zero coupling."""
    W = model.coupling.effective_weights().reshape(model.n_regions, model.coupling.n_bins, model.n_regions)
    total = W.abs().sum(1)
    absent = (anat.weights <= 0) & ~torch.eye(anat.n_regions, dtype=torch.bool)
    assert float(total[absent].abs().max()) == pytest.approx(0.0, abs=1e-8)


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------
def test_rollout_is_deterministic_given_a_seed(anat):
    cfg = tiny_config()
    outs = []
    for _ in range(2):
        set_determinism(1234, strict=True)
        m = SCWBD(cfg.model, anat)
        y = torch.randn(2, 5, anat.n_regions, generator=torch.Generator().manual_seed(9))
        th = ThetaPrior().sample(2, seed=3)
        outs.append(m.rollout(y_context=y, theta=th, n_steps=4).activity)
    assert torch.equal(outs[0], outs[1]), "identical seeds must give bit-identical rollouts"


def test_theta_prior_sampling_is_deterministic():
    a = ThetaPrior().sample(16, seed=5)
    b = ThetaPrior().sample(16, seed=5)
    c = ThetaPrior().sample(16, seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_theta_prior_bounds_and_round_trip():
    p = ThetaPrior()
    th = p.sample(64, seed=0)
    b = p.bounds()
    assert (th >= b[:, 0] - 1e-5).all() and (th <= b[:, 1] + 1e-5).all()
    assert torch.allclose(p.denormalise(p.normalise(th)), th, atol=1e-4)
    assert len(THETA_NAMES) == th.shape[1]


# ----------------------------------------------------------------------
# ablation configurations must actually build
# ----------------------------------------------------------------------
@pytest.mark.parametrize("field", ["scalar_state_ablation", "dense_coupling_ablation"])
def test_required_ablations_build(anat, field):
    cfg = tiny_config()
    setattr(cfg.model, field, True)
    m = SCWBD(cfg.model, anat)
    y = torch.randn(2, 4, anat.n_regions)
    r = m.rollout(y_context=y, theta=ThetaPrior().sample(2, seed=0), n_steps=3)
    assert torch.isfinite(r.activity).all()
    if field == "scalar_state_ablation":
        assert m.layout.dim == 1


@pytest.mark.parametrize("graph", ["randomized", "distance_matched", "local_only", "dense"])
def test_g2_control_graphs_build(anat, graph):
    cfg = tiny_config()
    cfg.model.control_graph = graph
    m = SCWBD(cfg.model, anat)
    r = m.rollout(y_context=torch.randn(2, 4, anat.n_regions), theta=ThetaPrior().sample(2, seed=0), n_steps=3)
    assert torch.isfinite(r.activity).all()


def test_mechanistic_core_is_a_configuration_switch(anat):
    """"Interchangeable backends compared, not assumed" must be one config field."""
    cfg = tiny_config()
    cfg.model.local_core = "wilson_cowan"
    m = SCWBD(cfg.model, anat)
    assert "mech" in m.layout
    th = ThetaPrior().sample(2, seed=0)
    m.set_mechanistic_theta(th, anat)
    r = m.rollout(y_context=torch.randn(2, 4, anat.n_regions), theta=th, n_steps=3, enforce_r05=False)
    assert torch.isfinite(r.activity).all()
    assert r.diagnostics["rho_enforced"] is True
