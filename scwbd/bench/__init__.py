"""``scwbd.bench`` — the falsification machinery (agent J).

Contents:

* :mod:`~scwbd.bench.report` — the one canonical claim manifest / claim report
  schema used by everything else, with the §11.2 reporting discipline enforced
  at construction time;
* :mod:`~scwbd.bench.gates` — claim gates G1--G5 (``tab:claim-gates``);
* :mod:`~scwbd.bench.ablations` — every required ablation from §11.4;
* :mod:`~scwbd.bench.leakage` — Appendix D ``tab:mixture-evaluation`` as audits;
* :mod:`~scwbd.bench.numerics` — §11.1 numerical / representational / physical
  tests, including the adaptive-resolution permit;
* :mod:`~scwbd.bench.statistics` — intervals, calibration, selection optimism,
  stratified bias, and the "smoothed away the effect" check;
* :mod:`~scwbd.bench.matching` — matched parameter/compute accounting;
* :mod:`~scwbd.bench.harness` — the thin model/dataset protocol gates measure through;
* :mod:`~scwbd.bench.synthetic` — fixtures used to prove the gates can fail;
* :mod:`~scwbd.bench.runner` — writes ``reports/gates/SUMMARY.md``.

Standing rule: **a gate that fails is a result, not a bug**, and a gate that
cannot run reports ``COULD_NOT_RUN`` — never a pass.
"""

from __future__ import annotations

from .report import (
    BENCH_SCHEMA_VERSION,
    BaselineResult,
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    ReportDisciplineError,
    SubCheck,
    could_not_run,
)

__all__ = [
    "BENCH_SCHEMA_VERSION",
    "BaselineResult",
    "ClaimManifest",
    "ClaimReport",
    "Interval",
    "Metric",
    "ReportDisciplineError",
    "SubCheck",
    "could_not_run",
]
