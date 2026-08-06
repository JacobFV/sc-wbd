"""The instrument audit must itself be able to fail."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.instruments import (
    KNOWN_UNINFORMATIVE,
    Instrument,
    audit_instruments,
    default_instruments,
)
from scwbd.bench.report import source_dirty_entries


def test_every_bench_guard_can_read_differently():
    rep = audit_instruments()
    assert rep.status == "PASS", rep.blocking_reasons
    names = {s.name for s in rep.subchecks}
    assert {"source_dirty_flag", "capacity_matching", "interval_strict_threshold",
            "smoothing_check", "report_provenance_rule"} <= names


def test_audit_fails_on_an_instrument_that_cannot_vary():
    """The negative control: a constant reading is decoration, and must FAIL."""
    constant = Instrument(
        name="always_green",
        description="A flag that reads the same no matter what happened.",
        read=lambda _x: "ok",
        inputs={"nothing_wrong": 0, "everything_wrong": 1},
        consequence="it would be reported as evidence while being incapable of alarm",
    )
    rep = audit_instruments([constant])
    assert rep.status == "FAIL"
    assert rep.consequence.startswith("Stop reporting the affected field")
    m = rep.subchecks[0].metrics[0]
    assert m.value == 1.0 and m.passed is False


def test_audit_needs_at_least_two_inputs_to_conclude_anything():
    one = Instrument(name="single", description="d", read=lambda x: x,
                     inputs={"only": 1})
    rep = audit_instruments([one])
    assert rep.status == "COULD_NOT_RUN"
    assert "at least two" in " ".join(rep.blocking_reasons)


def test_scoped_dirty_flag_separates_source_edits_from_a_run_writing_its_log():
    """The exact discrimination the whole-tree -dirty flag could not make."""
    rep = audit_instruments()
    sub = next(s for s in rep.subchecks if s.name == "source_dirty_flag")
    note = sub.metrics[0].note
    assert "run_wrote_its_own_log -> []" in note      # writing tracked output: clean
    assert "source_modified -> ['scwbd/mod.py']" in note   # editing source: dirty
    assert sub.status == "PASS"


def test_bench_provenance_does_not_record_or_gate_on_the_whole_tree_dirty_flag():
    from scwbd.bench.report import provenance

    p = provenance()
    assert "git_dirty" not in p
    assert "dirty" not in str(p.get("git_rev", ""))
    assert "git_dirty_whole_tree" in p["known_uninformative_fields"]
    # what IS recorded is actionable: which paths, not a boolean
    assert isinstance(p["source_dirty_paths"], list)
    assert "source_clean" in p["not_gated_on"]


def test_source_scoped_status_is_a_path_list_not_a_boolean():
    """In a shared worktree, whose edit it was is the useful question."""
    mine = source_dirty_entries(("scwbd/bench", "tests/bench"))
    assert isinstance(mine, list)
    assert all(isinstance(x, str) for x in mine)


def test_known_uninformative_fields_are_registered_with_remedies():
    assert len(KNOWN_UNINFORMATIVE) >= 5
    for u in KNOWN_UNINFORMATIVE:
        assert u.remedy and u.why_it_cannot_discriminate and u.found_by
    names = " ".join(u.name for u in KNOWN_UNINFORMATIVE)
    assert "git_sha" in names
    assert "torch.compile" in names
    assert "MemoryMax" in names
    assert "OOM" in names
    # the bench's own bug is registered too, not quietly fixed
    mine = [u for u in KNOWN_UNINFORMATIVE if u.owner.startswith("bench")]
    assert mine and "not the actual reason" in mine[0].why_it_cannot_discriminate


def test_default_instruments_all_declare_a_consequence():
    for inst in default_instruments():
        assert inst.consequence, f"{inst.name} does not say what a constant reading means"
        assert len(inst.inputs) >= 2
