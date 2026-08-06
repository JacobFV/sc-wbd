"""`base:` inheritance must resolve transitively, and cycles must raise.

Resolution used to stop after one level: the parent was read raw, so a
grandparent's `base` key survived the merge and `from_dict` rejected it with
`unknown config key 'base'` -- an error naming the symptom, not the cause. Any
two-level hierarchy was unbuildable, which is exactly the shape run 2 needs
(pilot override -> arm config -> run config).
"""

from __future__ import annotations

import pytest
import yaml

from scwbd.foundation.config import load_config


def _w(p, d):
    p.write_text(yaml.safe_dump(d))
    return p


def _root(tmp_path):
    return _w(
        tmp_path / "root.yaml",
        {
            "model": {"n_regions": 12, "hidden": 32, "compile": False},
            "data": {"real_test_fraction": 0.25, "seed": 1},
            "train": {"run_name": "root", "out_dir": "ck/root"},
        },
    )


def test_two_level_chain_resolves(tmp_path):
    _root(tmp_path)
    _w(tmp_path / "mid.yaml", {"base": "root.yaml", "model": {"hidden": 64}})
    _w(tmp_path / "leaf.yaml", {"base": "mid.yaml", "data": {"real_test_fraction": 0.5}})

    cfg = load_config(tmp_path / "leaf.yaml")
    assert cfg.data.real_test_fraction == 0.5   # leaf wins
    assert cfg.model.hidden == 64               # mid wins over root
    assert cfg.model.n_regions == 12            # root survives two hops
    assert cfg.data.seed == 1
    assert cfg.train.run_name == "root"


def test_three_level_chain_resolves(tmp_path):
    _root(tmp_path)
    _w(tmp_path / "a.yaml", {"base": "root.yaml", "model": {"hidden": 64}})
    _w(tmp_path / "b.yaml", {"base": "a.yaml", "model": {"hidden": 128}})
    _w(tmp_path / "c.yaml", {"base": "b.yaml", "train": {"run_name": "leaf"}})

    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.model.hidden == 128
    assert cfg.model.n_regions == 12
    assert cfg.train.run_name == "leaf"


def test_cycle_raises_rather_than_recursing(tmp_path):
    _w(tmp_path / "x.yaml", {"base": "y.yaml", "model": {"hidden": 32}})
    _w(tmp_path / "y.yaml", {"base": "x.yaml", "model": {"hidden": 64}})
    with pytest.raises(ValueError, match="cyclic config inheritance"):
        load_config(tmp_path / "x.yaml")


def test_self_cycle_raises(tmp_path):
    _w(tmp_path / "s.yaml", {"base": "s.yaml"})
    with pytest.raises(ValueError, match="cyclic config inheritance"):
        load_config(tmp_path / "s.yaml")


def test_run2_pilot_pair_is_matched():
    """The pair that actually launches: same split, same seed, same budget."""
    t = load_config("configs/run2/pilot-families.yaml")
    c = load_config("configs/run2/pilot-pooled-param-matched.yaml")

    # A1 §3.6: the arms differ ONLY in state structure.
    assert t.model.family_state is True and c.model.family_state is False
    assert t.model.local_core == c.model.local_core, "operator assignment must be identical"
    assert t.model.state_dependent_variance == c.model.state_dependent_variance  # RL-4

    # Paired in the participants, not just the windows.
    assert t.data.real_test_fraction == c.data.real_test_fraction == 0.5
    assert t.data.seed == c.data.seed
    assert t.data.sim_index_fast == c.data.sim_index_fast

    # B3: matched optimiser steps.
    assert sum(s.steps for s in t.train.stages) == sum(s.steps for s in c.train.stages)

    # The pilot is the pilot: it must not collide with the endpoint's outputs.
    assert t.train.out_dir != c.train.out_dir
    assert "pilot" in t.train.run_name and "pilot" in c.train.run_name
