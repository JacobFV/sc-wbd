"""Hippocampal backends and the benchmark that is supposed to discriminate them."""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import (
    ModernHopfield,
    SparseDistributedMemory,
    SuccessorRepresentation,
    VectorHaSH,
    compare_backends,
    get_hippocampal_backend,
    signature,
)
from scwbd.dynamics.hippocampus import (
    HIPPOCAMPAL_BACKENDS,
    capacity_curve,
    cue_degradation_curve,
    interference_curve,
    pattern_separation,
    replay_order_score,
)

D_K, D_V = 48, 48


def factories(device):
    return {
        "modern_hopfield": lambda: ModernHopfield(D_K, D_V, device=device),
        "vector_hash": lambda: VectorHaSH(D_K, D_V, d_hidden=256, device=device),
        "sdm": lambda: SparseDistributedMemory(D_K, D_V, n_locations=512, device=device),
        "successor_representation": lambda: SuccessorRepresentation(D_K, D_V, n_states=64, device=device),
    }


@pytest.mark.parametrize("name", sorted(HIPPOCAMPAL_BACKENDS))
def test_backend_shares_the_H_t_interface(name, device):
    mem = factories(device)[name]()
    torch.manual_seed(0)
    k = torch.randn(8, D_K, device=device)
    v = torch.randn(8, D_V, device=device)
    mem.write(k, v)
    out = mem.read(k)
    assert out.value.shape == (8, D_V)
    assert out.rho.shape == (8,) and float(out.rho.min()) >= 0.0
    assert mem.state.n_items == 8
    assert mem.encode(k).ndim == 2
    assert mem.describe()["hypothesis"], "a backend must state the hypothesis it encodes"
    mem.reset()
    assert mem.state.n_items == 0


def test_unknown_backend_is_refused():
    with pytest.raises(KeyError, match="unknown hippocampal backend"):
        get_hippocampal_backend("holographic_vibes")


def test_modern_hopfield_recalls_stored_content(device):
    mem = ModernHopfield(D_K, D_V, device=device, beta=20.0)
    torch.manual_seed(1)
    k = torch.randn(16, D_K, device=device)
    v = torch.randn(16, D_V, device=device)
    mem.write(k, v)
    out = mem.read(k)
    err = (out.value - v).norm(dim=-1) / v.norm(dim=-1)
    assert float(err.mean()) < 0.1


def test_capacity_curve_degrades_with_load(device):
    curve = capacity_curve(
        factories(device)["modern_hopfield"], loads=(4, 16, 64, 256), d_key=D_K, d_value=D_V, seed=0
    )
    acc = curve["accuracy"]
    assert acc[0] >= acc[-1], f"accuracy should not improve with load: {acc}"
    assert curve["capacity_at_criterion"] > 0


def test_interference_accumulates_with_episodes(device):
    curve = interference_curve(
        factories(device)["sdm"], n_episodes=128, chunk=16, d_key=D_K, d_value=D_V, seed=0
    )
    ys = curve["first_chunk_accuracy"]
    assert ys[0] >= ys[-1], f"retention of the first chunk should not improve: {ys}"


def test_cue_degradation_is_monotone(device):
    curve = cue_degradation_curve(
        factories(device)["modern_hopfield"],
        n_items=16,
        fractions=(0.0, 0.2, 0.4, 0.6, 0.8),
        d_key=D_K,
        d_value=D_V,
        seed=0,
    )
    acc = curve["accuracy"]
    assert acc[0] > acc[-1], f"accuracy must fall as the cue degrades: {acc}"
    assert acc[0] > 0.9


def test_pattern_separation_slope_is_measured(device):
    sep = pattern_separation(
        factories(device)["vector_hash"],
        n_pairs=32,
        input_similarities=(0.0, 0.3, 0.6, 0.9, 0.99),
        d_key=D_K,
        d_value=D_V,
        seed=0,
    )
    assert len(sep["output_similarity"]) == 5
    assert sep["output_similarity"][-1] > sep["output_similarity"][0], "identical cues must map together"
    assert sep["slope"] > 0


def test_successor_representation_derives_its_replay_order(device):
    """Replay order is a *prediction* of the SR, not a stipulation."""
    sr = replay_order_score(
        factories(device)["successor_representation"], n_items=12, d_key=D_K, d_value=D_V, seed=0
    )
    ep = replay_order_score(
        factories(device)["modern_hopfield"], n_items=12, d_key=D_K, d_value=D_V, seed=0
    )
    assert sr["derived"] is True
    assert ep["derived"] is False
    # the episodic store's "forward" order is stipulated and therefore perfect
    assert ep["forward_corr"] == pytest.approx(1.0, abs=1e-6)
    assert ep["reverse_corr"] == pytest.approx(-1.0, abs=1e-6)


def test_backends_have_distinguishable_signatures(device):
    """The point of the benchmark: the backends must actually differ on it.

    If every backend produced the same signature, "a backend is selected by these
    signatures" would be empty — so this test fails when the benchmark stops
    discriminating, which is a result about the benchmark.
    """
    sigs = compare_backends(factories(device), d_key=D_K, d_value=D_V, seed=0, loads=(4, 16, 64))
    assert set(sigs) == set(factories(device))
    caps = {n: s.capacity["capacity_at_criterion"] for n, s in sigs.items()}
    seps = {n: round(s.separation["slope"], 3) for n, s in sigs.items()}
    cues = {n: tuple(round(a, 3) for a in s.cue_degradation["accuracy"]) for n, s in sigs.items()}
    assert len(set(caps.values())) > 1 or len(set(seps.values())) > 1, (
        f"backends are indistinguishable on capacity {caps} and separation {seps}"
    )
    assert len(set(cues.values())) == len(cues), "cue-degradation curves collapse across backends"
    for s in sigs.values():
        assert s.summary()


def test_signature_is_seed_deterministic(device):
    a = signature(factories(device)["vector_hash"], d_key=D_K, d_value=D_V, seed=3, loads=(8, 16))
    b = signature(factories(device)["vector_hash"], d_key=D_K, d_value=D_V, seed=3, loads=(8, 16))
    assert a.capacity == b.capacity and a.cue_degradation == b.cue_degradation
