"""The scoreboard must be blunt: no pass may appear that was not earned."""

from __future__ import annotations

import json

import pytest

from scwbd.bench.gates import CLAIMS
from scwbd.bench.runner import build_summary, run_everything


@pytest.fixture(scope="module")
def bare():
    return run_everything(write=False, seed=0)


def test_bare_repository_claims_nothing(bare):
    flat = [r for group in bare.values() for r in group]
    assert flat, "no checks were run at all"
    assert not any(r.status == "PASS" for r in flat), \
        "a check passed without any model, data or dependency being supplied"


def test_summary_lists_every_gate_ablation_audit_and_numerical_check(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    for r in [x for g in bare.values() for x in g]:
        assert f"`{r.manifest.claim_id}`" in md
    for gid, c in CLAIMS.items():
        assert c["falsified_by"] in md
        assert c["consequence"] in md


def test_summary_has_the_what_we_cannot_claim_section(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "## 6. What we cannot yet claim" in md
    assert "may not make" in md
    assert "No digital-twin claim" in md
    assert "No clinical, wellness or treatment claim" in md
    assert "No consciousness or Phi claim" in md
    assert "A gate that cannot run is **not** a gate that passed" in md


def test_summary_tells_the_reader_not_to_move_the_threshold(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "not a smaller threshold" in md
    assert "changes the claim class" in md


def test_reports_are_written_as_json_and_markdown(tmp_path, bare):
    rep = bare["gates"][0]
    jp, mp = rep.write(tmp_path)
    payload = json.loads(jp.read_text())
    assert payload["claim_id"] == "G1"
    assert payload["status"] == "COULD_NOT_RUN"
    assert payload["manifest"]["thesis_reference"].startswith("thesis_contract.tex")
    assert "This gate did **not** pass" in mp.read_text()


def test_every_gate_declares_the_out_of_scope_non_goals(bare):
    for r in bare["gates"]:
        joined = " ".join(r.manifest.non_goals)
        assert "digital twin" in joined
        assert "no IRB" in joined
