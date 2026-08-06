"""The report schema must refuse to produce a flattering report."""

from __future__ import annotations

import json

import pytest

from scwbd.bench.report import (
    BaselineResult,
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    ReportDisciplineError,
    SubCheck,
    could_not_run,
)


def _manifest(**kw):
    base = dict(
        claim_id="TEST",
        claim_text="a claim",
        falsified_by="a falsifier",
        consequence_if_failed="do the thing",
    )
    base.update(kw)
    return ClaimManifest(**base)


def _ok_metric(name="m", value=1.0, threshold=0.0):
    return Metric(name=name, value=value, kind="accuracy", interval=Interval(0.5, 1.5),
                  threshold=threshold)


def _cal_metric():
    return Metric(name="coverage_error", value=0.01, kind="calibration",
                  interval=Interval(0.0, 0.02), threshold=0.05, direction="less_is_better")


def test_manifest_requires_a_consequence():
    with pytest.raises(ReportDisciplineError):
        _manifest(consequence_if_failed="")


def test_manifest_requires_a_falsifier():
    with pytest.raises(ReportDisciplineError):
        _manifest(falsified_by="")


def test_estimated_metric_requires_an_interval():
    with pytest.raises(ReportDisciplineError):
        Metric(name="m", value=1.0, kind="accuracy")
    # ... unless it is exact (a count, a rank, a boolean)
    Metric(name="rank", value=3.0, kind="identifiability", exact=True)


def test_accuracy_without_calibration_is_refused():
    rep = ClaimReport(
        manifest=_manifest(),
        subchecks=[SubCheck(name="s", description="d", metrics=[_ok_metric()])],
        baselines_run=[BaselineResult(name="b", role="control")],
    )
    with pytest.raises(ReportDisciplineError, match="calibration"):
        rep.finalize()


def test_one_could_not_run_mandatory_check_blocks_the_whole_pass():
    rep = ClaimReport(
        manifest=_manifest(),
        subchecks=[
            SubCheck(name="ok", description="d", metrics=[_ok_metric(), _cal_metric()]),
            could_not_run("blocked", "d", "agent H has not landed"),
        ],
        baselines_run=[BaselineResult(name="b", role="control")],
    ).finalize()
    assert rep.status == "COULD_NOT_RUN"
    assert any("agent H" in r for r in rep.blocking_reasons)
    assert rep.consequence is None


def test_a_gate_cannot_pass_without_baselines():
    rep = ClaimReport(
        manifest=_manifest(),
        subchecks=[SubCheck(name="ok", description="d",
                            metrics=[_ok_metric(), _cal_metric()])],
        kind="gate",
    )
    with pytest.raises(ReportDisciplineError, match="baselines"):
        rep.finalize()


def test_failure_carries_the_implementation_consequence():
    rep = ClaimReport(
        manifest=_manifest(consequence_if_failed="demote anatomy to a weak prior"),
        subchecks=[SubCheck(name="bad", description="d",
                            metrics=[_ok_metric(value=-1.0), _cal_metric()])],
        baselines_run=[BaselineResult(name="b", role="control")],
    ).finalize()
    assert rep.status == "FAIL"
    assert rep.consequence == "demote anatomy to a weak prior"
    assert "demote anatomy to a weak prior" in rep.to_markdown()


def test_interval_strict_metric_rejects_a_noisy_win():
    noisy = Metric(name="delta", value=0.02, kind="accuracy", interval=Interval(-0.05, 0.09),
                   threshold=0.0, require_interval_beats_threshold=True)
    assert noisy.passed is False
    clean = Metric(name="delta", value=0.02, kind="accuracy", interval=Interval(0.01, 0.03),
                   threshold=0.0, require_interval_beats_threshold=True)
    assert clean.passed is True


def test_could_not_run_requires_a_reason():
    with pytest.raises(ReportDisciplineError):
        could_not_run("x", "d", "")


def test_json_round_trip_is_machine_readable(tmp_path):
    rep = ClaimReport(
        manifest=_manifest(),
        subchecks=[SubCheck(name="ok", description="d",
                            metrics=[_ok_metric(), _cal_metric()])],
        baselines_run=[BaselineResult(name="b", role="control")],
    ).finalize()
    jp, mp = rep.write(tmp_path)
    payload = json.loads(jp.read_text())
    assert payload["status"] == "PASS"
    assert payload["manifest"]["consequence_if_failed"]
    assert payload["subchecks"][0]["metrics"][0]["passed"] is True
    assert mp.read_text().startswith("# TEST")
