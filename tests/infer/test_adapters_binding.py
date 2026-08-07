"""The cross-agent binding contract for ``expected_fisher``.

Gate G4 (agent J / Popper) does not implement Fisher information itself -- it
consumes ours through ``scwbd.bench.adapters.fisher_design_map``, which resolves
``scwbd.infer.fisher.expected_fisher`` and calls it as
``fn(u, cfg, proto, design=<name>)``.  These tests pin that call shape so an
internal refactor here cannot silently break a gate over there.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.infer.adapters import PARAMETER_BLOCKS, design_information_map
from scwbd.infer.fisher import expected_fisher


def test_expected_fisher_is_importable_by_every_name_the_gate_probes():
    import scwbd.infer as pkg
    import scwbd.infer.fisher as mod

    # fisher_backend() walks these candidates in order and takes the first hit.
    assert callable(getattr(mod, "expected_fisher"))
    assert callable(getattr(pkg, "expected_fisher"))


def test_gate_call_shape_u_cfg_proto_design(tiny_setup):
    """``fn(u, cfg, proto, design=...)`` -- positional trio, keyword design."""
    cfg, proto, u0 = tiny_setup
    rep = expected_fisher(u0, cfg, proto, design="joint_native")
    m = rep.metrics
    # the statistic the decision rule reads out of every design
    assert "theta_profile_min_eigenvalue_nonprior" in m
    assert "rank_likelihood" in m and "condition_number_total" in m
    assert np.isfinite(m["theta_profile_min_eigenvalue_nonprior"])


def test_design_information_map_exposes_stable_block_structure(tiny_cfg):
    from scwbd.infer.identifiability import DESIGNS

    picked = [d for d in DESIGNS if d.name in ("eeg_only", "joint_native")]
    out = design_information_map(cfg=tiny_cfg, designs=picked)
    assert set(out) == {"eeg_only", "joint_native"}
    # eta = (theta, ell, rho) partition must stay contiguous and complete
    idx = PARAMETER_BLOCKS["theta"] + PARAMETER_BLOCKS["ell"] + PARAMETER_BLOCKS["rho"]
    assert sorted(idx) == list(range(len(idx)))
    for di in out.values():
        assert di.theta_profile_information.shape == (len(PARAMETER_BLOCKS["theta"]),) * 2


def test_bench_gate_binding_resolves_if_agent_j_module_is_present(tiny_setup):
    """End-to-end through Popper's adapter, skipped if bench has not landed."""
    adapters = pytest.importorskip("scwbd.bench.adapters")
    if not hasattr(adapters, "fisher_design_map"):
        pytest.skip("scwbd.bench.adapters.fisher_design_map not present")
    cfg, proto, u0 = tiny_setup
    dep = adapters.fisher_design_map(u0, cfg, proto)
    assert dep.available, dep.reason if hasattr(dep, "reason") else dep
    rep = dep.obj("joint_native")
    assert np.isfinite(rep.metrics["theta_profile_min_eigenvalue_nonprior"])
