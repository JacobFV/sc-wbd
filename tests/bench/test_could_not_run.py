"""A missing dependency yields COULD_NOT_RUN — never a pass, never a silent skip."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench import adapters
from scwbd.bench.gates import run_all_gates, run_g1, run_g2, run_g4, run_g5
from scwbd.bench.leakage import run_all_audits
from scwbd.bench.numerics import run_numerics_suite
from scwbd.bench.report import ClaimReport
from scwbd.bench.synthetic import (
    RidgeGaussian,
    SyntheticFisher,
    make_fusion_dataset,
    make_graph_dataset,
)

from .conftest import FIXTURE_THRESHOLDS, g1_arms


def test_bare_run_produces_only_could_not_run_and_never_a_pass():
    reports = run_all_gates()
    assert len(reports) == 5
    for r in reports:
        assert r.status == "COULD_NOT_RUN"
        assert r.blocking_reasons, f"{r.manifest.claim_id} gave no reason"
        assert r.consequence is None


def test_every_could_not_run_names_the_blocking_agent_or_input():
    for r in run_all_gates() + run_all_audits() + run_numerics_suite():
        if r.status != "COULD_NOT_RUN":
            continue
        joined = " ".join(r.blocking_reasons).lower()
        assert any(w in joined for w in ("agent", "missing", "no ", "not supplied",
                                         "out of scope", "off by default", "unavailable")), \
            f"{r.manifest.claim_id}: unhelpful reason {r.blocking_reasons}"


def test_g1_without_the_resampling_baseline_cannot_run():
    d = make_fusion_dataset(seed=1)
    cand, base = g1_arms()
    base.pop("naive_resampling")
    rep = run_g1(train=d["train"], test=d["test"], candidate=cand, baselines=base,
                 thresholds=FIXTURE_THRESHOLDS, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert any("naive_resampling" in r for r in rep.blocking_reasons)


def test_g1_without_the_intervention_holdout_cannot_pass():
    d = make_fusion_dataset(seed=1, bold_informative=True)
    cand, base = g1_arms()
    rep = run_g1(train=d["train"], test=d["test"], candidate=cand, baselines=base,
                 delay_true=3.0,
                 delay_estimates={"candidate": [3.0], "naive_resampling": [4.4],
                                  "single_modality_eeg": [3.9]},
                 thresholds=FIXTURE_THRESHOLDS, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert any("intervention" in r for r in rep.blocking_reasons)


def test_g2_refuses_to_invent_the_anatomy_controls():
    d = make_graph_dataset(seed=2)
    rep = run_g2(train=d["train"], test=d["test"], ood=d["ood"],
                 model_for_graph=lambda A: RidgeGaussian(mask=A),
                 anatomy=d["anatomy"], controls=None, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    joined = " ".join(rep.blocking_reasons)
    assert "agent C" in joined
    assert "control is the experiment" in joined


def test_g2_with_a_missing_single_control_cannot_run():
    d = make_graph_dataset(seed=2)
    controls = dict(d["controls"])
    controls.pop("distance_matched")
    rep = run_g2(train=d["train"], test=d["test"], ood=d["ood"],
                 model_for_graph=lambda A: RidgeGaussian(mask=A),
                 anatomy=d["anatomy"], controls=controls, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert any("distance_matched" in r for r in rep.blocking_reasons)


def test_g4_without_agent_h_fisher_cannot_run():
    rep = run_g4(seed=0)
    assert rep.status == "COULD_NOT_RUN"
    joined = " ".join(rep.blocking_reasons)
    assert "agent H" in joined
    assert "will not reimplement it" in joined


def test_g4_without_the_parameter_partition_cannot_run():
    f = SyntheticFisher()
    rep = run_g4(fisher=f, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert any("nuisance" in r for r in rep.blocking_reasons)


def test_g5_without_a_mandatory_baseline_cannot_run():
    from scwbd.bench.synthetic import make_individualization_dataset

    d = make_individualization_dataset(seed=5)
    cand, base = __import__("tests.bench.conftest", fromlist=["g5_arms"]).g5_arms()
    base.pop("anatomy_only")
    rep = run_g5(train=d["train"], new_session=d["new_session"],
                 unseen_task=d["unseen_task"], candidate=cand, baselines=base, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert any("anatomy_only" in r for r in rep.blocking_reasons)


def test_adapters_report_a_missing_module_with_its_owner():
    dep = adapters.probe("scwbd.definitely_not_a_module")
    assert not dep.available
    assert "unavailable" in dep.blocker
    dep2 = adapters.probe_attr("scwbd.bench.gates", "not_a_symbol")
    assert not dep2.available
    assert "defines no" in dep2.reason


def test_dependency_table_lists_every_sibling_module():
    rows = adapters.dependency_table()
    mods = {r["module"] for r in rows}
    assert {"scwbd.anatomy", "scwbd.infer", "scwbd.foundation", "scwbd.sources"} <= mods


def test_agent_b_splitter_is_consumed_not_reimplemented():
    """Leakage audits must use scwbd.sources.splits, which agent B owns."""
    import scwbd.bench.leakage as leakage

    src = (leakage.__file__)
    text = open(src, encoding="utf-8").read()
    assert "GroupedSplitter" in text and "leakage_audit" in text
    assert "class GroupedSplitter" not in text, "bench must not reimplement the splitter"
