"""The scoreboard must be blunt: no pass may appear that was not earned."""

from __future__ import annotations

import json

import pytest

from scwbd.bench.gates import CLAIMS
from scwbd.bench.runner import build_summary, run_everything


@pytest.fixture(scope="module")
def bare():
    return run_everything(write=False, seed=0)


def test_no_claim_bearing_gate_passes_without_a_model(bare):
    """Gates, ablations and audits have no subject yet, so none may pass.

    Numerical checks are exempt only because they *do* have a subject once the
    module they audit has landed (N1 compiles agent A's reference example), and
    a numerics PASS licenses a statement about code, never about a brain.
    """
    for kind in ("gates", "ablations", "leakage"):
        passing = [r.manifest.claim_id for r in bare[kind] if r.status == "PASS"]
        assert not passing, f"{kind} passed with no model or data supplied: {passing}"


def test_any_numerics_pass_declares_its_subject(bare):
    for r in bare["numerics"]:
        if r.status == "PASS":
            assert r.artifacts.get("subject"), \
                f"{r.manifest.claim_id} passed without recording what it checked"
            assert any("not evidence" in n for n in r.notes)


def test_summary_lists_every_gate_ablation_audit_and_numerical_check(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    for r in [x for g in bare.values() for x in g]:
        assert f"`{r.manifest.claim_id}`" in md
    for gid, c in CLAIMS.items():
        assert c["falsified_by"] in md
        assert c["consequence"] in md


def test_summary_has_the_what_we_cannot_claim_section(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "## 7. What we cannot yet claim" in md
    assert "## 6. What is licensed so far" in md
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


def test_summary_names_the_modality_additivity_tautology(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "A tautology this scoreboard refuses to report as a result" in md
    assert "I_{EEG+BOLD} = I_EEG + I_BOLD" in md
    assert "cannot fail" in md
    # and it must say where the falsifiable content actually lives
    assert "joint_whitening=True" in md
    assert "native-clock versus naively resampled" in md
    assert "prior_standardised" in md


def test_summary_does_not_quote_g4_numbers_before_the_preregistered_run(bare):
    """G4 has no subject yet; the scoreboard must not carry smoke numbers."""
    g4 = next(r for r in bare["gates"] if r.manifest.claim_id == "G4")
    assert g4.status == "COULD_NOT_RUN"
    assert not g4.artifacts.get("fisher"), "G4 reported information numbers with no run"


def test_summary_states_what_a_numerical_pass_does_not_unblock(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "What a passing numerical gate does and does not unblock" in md
    assert "licenses no claim on its own" in md
    assert "N3` validates **conduction**" in md
    assert "does **not** cover" in md
    assert "`N6`" in md


def test_summary_records_the_provenance_lesson(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"], bare["numerics"])
    assert "Provenance is part of the discipline" in md
    assert "stale artifact" in md
    assert "refused unless it records its subject or its solver provenance" in md
