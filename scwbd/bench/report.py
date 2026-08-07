"""Canonical claim manifest / claim report schema (agent J).

One schema, used by every gate (:mod:`scwbd.bench.gates`), every ablation
(:mod:`scwbd.bench.ablations`), every leakage audit
(:mod:`scwbd.bench.leakage`) and every numerical check
(:mod:`scwbd.bench.numerics`).

The schema is deliberately *unfriendly to optimism*.  ``ClaimReport.finalize``
enforces the reporting discipline of ``body.tex`` §11.2 and
``thesis_contract.tex`` Table ``tab:claim-gates`` as hard invariants:

* a report may only be ``PASS`` when **every mandatory sub-check ran and
  passed**.  One ``COULD_NOT_RUN`` mandatory sub-check makes the whole report
  ``COULD_NOT_RUN`` — a gate that cannot run is never a pass;
* a ``FAIL`` report must carry the *implementation consequence* from the
  thesis table.  Failure without a consequence is not reportable;
* a metric of kind ``"accuracy"`` may not appear without a companion metric of
  kind ``"calibration"`` ("aggregate accuracy cannot substitute for
  calibration within the intended deployment population");
* a metric that claims to be an estimate must carry an interval unless it is
  explicitly declared exact (counts, ranks, booleans);
* a passing numerical check must record **what it measured** — the subject or
  the solver provenance. An artifact that does not record how it was produced
  is how a stale output gets compared against new code and mistaken for a
  result, which has already happened twice in this repository.

These are raised as :class:`ReportDisciplineError` at construction time, so a
non-compliant report cannot be written to ``reports/``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "BENCH_SCHEMA_VERSION",
    "THESIS_VERSION",
    "SCHEMA_VERSION",
    "MODEL_DESIGNATION",
    "Status",
    "ReportDisciplineError",
    "Interval",
    "Metric",
    "SubCheck",
    "BaselineResult",
    "ClaimManifest",
    "ClaimReport",
    "REPORTS_ROOT",
    "GATES_DIR",
    "ABLATIONS_DIR",
    "provenance",
    "source_dirty_entries",
    "SOURCE_PATHS",
    "write_reports",
]

BENCH_SCHEMA_VERSION = "scwbd-bench-report/1.0.0"
THESIS_VERSION = "V6"
SCHEMA_VERSION = "scwbd-schema/1.0.0"
from ..schema.designation import MODEL_DESIGNATION  # one definition, imported

Status = Literal["PASS", "FAIL", "COULD_NOT_RUN"]
ReportKind = Literal["gate", "ablation", "leakage", "numerics", "instrument",
                     "adjudication"]

_STATUS_ORDER = {"PASS": 0, "FAIL": 1, "COULD_NOT_RUN": 2}

_REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = _REPO_ROOT / "reports"
GATES_DIR = REPORTS_ROOT / "gates"
ABLATIONS_DIR = REPORTS_ROOT / "ablations"


class ReportDisciplineError(RuntimeError):
    """Raised when a report violates the §11.2 reporting discipline."""


# --------------------------------------------------------------------------
# intervals and metrics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    """An uncertainty interval.  Every estimated number ships with one."""

    lo: float
    hi: float
    level: float = 0.95
    method: str = "bootstrap-percentile"

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"interval hi<lo: [{self.lo}, {self.hi}]")

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)

    def __str__(self) -> str:
        return f"[{self.lo:.4g}, {self.hi:.4g}]_{self.level:.0%}"


MetricKind = Literal[
    "accuracy",       # aggregate predictive accuracy / likelihood / error
    "calibration",    # coverage, ECE, overconfidence
    "identifiability",
    "efficiency",     # data efficiency / compute
    "systematic",     # bias / worst-stratum / systematic error
    "capacity",       # parameter and compute accounting
    "audit",          # leakage / provenance audit outcome
    "numerical",      # solver / compiler correctness
    "diagnostic",     # informational only, never gates a claim
]


@dataclass(frozen=True)
class Metric:
    """One reported number, with its interval and its preregistered threshold."""

    name: str
    value: float
    units: str = "dimensionless"
    kind: MetricKind = "diagnostic"
    interval: Interval | None = None
    threshold: float | None = None
    direction: Literal["greater_is_better", "less_is_better", "two_sided"] = "greater_is_better"
    #: require the *interval* to sit entirely on the good side of the
    #: threshold, not just the point estimate.  This is what stops a noisy
    #: win from being reported as a win.
    require_interval_beats_threshold: bool = False
    #: exact quantities (counts, ranks, booleans) do not need an interval
    exact: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.exact and self.interval is None and self.kind != "diagnostic":
            raise ReportDisciplineError(
                f"metric {self.name!r} (kind={self.kind}) has no interval and is not "
                "declared exact; §11.2 requires sampling variance / bootstrap or "
                "posterior intervals on every reported estimate"
            )

    @property
    def passed(self) -> bool | None:
        """``None`` when the metric carries no threshold (informational)."""
        if self.threshold is None:
            return None
        if self.direction == "greater_is_better":
            if self.require_interval_beats_threshold:
                if self.interval is None:
                    return None
                return bool(self.interval.lo > self.threshold)
            return bool(self.value > self.threshold)
        if self.direction == "less_is_better":
            if self.require_interval_beats_threshold:
                if self.interval is None:
                    return None
                return bool(self.interval.hi < self.threshold)
            return bool(self.value < self.threshold)
        # two_sided: |value| <= threshold
        if self.require_interval_beats_threshold and self.interval is not None:
            return bool(abs(self.interval.lo) <= self.threshold and abs(self.interval.hi) <= self.threshold)
        return bool(abs(self.value) <= self.threshold)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["passed"] = self.passed
        return d

    def render(self) -> str:
        iv = f" {self.interval}" if self.interval is not None else ""
        thr = ""
        if self.threshold is not None:
            arrow = {"greater_is_better": ">", "less_is_better": "<", "two_sided": "|.|<="}[
                self.direction
            ]
            thr = f" (threshold {arrow} {self.threshold:.4g}"
            thr += ", interval-strict)" if self.require_interval_beats_threshold else ")"
        return f"{self.name} = {self.value:.6g} {self.units}{iv}{thr}"


# --------------------------------------------------------------------------
# sub-checks
# --------------------------------------------------------------------------
@dataclass
class SubCheck:
    """One falsifiable component of a claim.

    ``mandatory=True`` means the claim *cannot pass* without this check.  A
    mandatory check that could not run forces the whole report to
    ``COULD_NOT_RUN``: this is how a missing dependency becomes an honest
    "we do not know" instead of a silent pass.
    """

    name: str
    description: str
    metrics: list[Metric] = field(default_factory=list)
    mandatory: bool = True
    #: set explicitly to COULD_NOT_RUN with a reason; otherwise derived from
    #: the metrics' thresholds.
    forced_status: Status | None = None
    reason: str = ""
    #: what result would falsify this specific sub-check (thesis column 3)
    falsified_by: str = ""

    @property
    def status(self) -> Status:
        if self.forced_status is not None:
            return self.forced_status
        verdicts = [m.passed for m in self.metrics if m.passed is not None]
        if not verdicts:
            return "COULD_NOT_RUN"
        return "PASS" if all(verdicts) else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "mandatory": self.mandatory,
            "reason": self.reason,
            "falsified_by": self.falsified_by,
            "metrics": [m.as_dict() for m in self.metrics],
        }


def could_not_run(name: str, description: str, reason: str, *, mandatory: bool = True,
                  falsified_by: str = "") -> SubCheck:
    """Build a loud COULD_NOT_RUN sub-check.  Never returns a pass."""
    if not reason:
        raise ReportDisciplineError("COULD_NOT_RUN requires an explicit reason")
    return SubCheck(
        name=name,
        description=description,
        metrics=[],
        mandatory=mandatory,
        forced_status="COULD_NOT_RUN",
        reason=reason,
        falsified_by=falsified_by,
    )


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """A baseline actually run (not merely named).

    ``n_parameters`` / ``compute`` are mandatory because "matched compute and
    parameter count" is part of the claim, not a footnote.
    """

    name: str
    role: str
    n_parameters: int | None = None
    compute_flops: float | None = None
    train_steps: int | None = None
    metrics: list[Metric] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "n_parameters": self.n_parameters,
            "compute_flops": self.compute_flops,
            "train_steps": self.train_steps,
            "note": self.note,
            "metrics": [m.as_dict() for m in self.metrics],
        }


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
@dataclass
class ClaimManifest:
    """Preregistration block: what is claimed, on what data, against what."""

    claim_id: str
    claim_text: str
    #: verbatim from thesis_contract.tex tab:claim-gates, column 3
    falsified_by: str
    #: verbatim from thesis_contract.tex tab:claim-gates, column 4
    consequence_if_failed: str
    thesis_version: str = THESIS_VERSION
    schema_version: str = SCHEMA_VERSION
    bench_schema_version: str = BENCH_SCHEMA_VERSION
    model_designation: str = MODEL_DESIGNATION
    thesis_reference: str = ""
    permitted_source_cards: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    acceptance_thresholds: dict[str, Any] = field(default_factory=dict)
    refusal_fixtures: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    seed: int = 0
    preregistered: bool = True

    def __post_init__(self) -> None:
        if not self.consequence_if_failed:
            raise ReportDisciplineError(
                f"claim {self.claim_id!r} declares no implementation consequence; "
                "a claim without a consequence for failure is not falsifiable"
            )
        if not self.falsified_by:
            raise ReportDisciplineError(
                f"claim {self.claim_id!r} declares no falsification condition"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
#: Paths that hold *source*. A run writing its own logs into ``reports/`` must
#: not make a provenance field read "modified".
SOURCE_PATHS: tuple[str, ...] = ("scwbd", "tests", "configs", "benchmarks")


def _git_rev(cwd: str | os.PathLike[str] = _REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def source_dirty_entries(
    paths: Sequence[str] = SOURCE_PATHS,
    cwd: str | os.PathLike[str] = _REPO_ROOT,
) -> list[str]:
    """Porcelain entries **scoped to source paths**, as a list of paths.

    Why scoped, and why a list rather than a boolean:

    A whole-tree dirty flag is structurally incapable of ever reading clean
    during a run that writes tracked output — every SC-WBD training run writes
    ``reports/training/*.log`` and ``*.jsonl``, both tracked — so the flag
    stamps ``-dirty`` on every checkpoint the project has ever produced and
    cannot distinguish "source was modified" from "the run wrote its own log".
    A field that cannot vary is not evidence. (Found by agent Turing.)

    Scoping to source fixes that discrimination. It does **not** fix a second
    one: in a shared multi-agent worktree the scoped flag still cannot tell
    *whose* edit made it dirty. So this returns the offending paths, letting a
    reader see that the dirt is another module's in-flight work rather than the
    source under test.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - environment dependent
        return ["<git unavailable>"]
    entries = [ln[3:].strip() for ln in out.stdout.splitlines() if ln.strip()]
    return sorted(entries)


def provenance() -> dict[str, Any]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            versions[mod] = "absent"
    dirty = source_dirty_entries()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "host": platform.node(),
        "platform": platform.platform(),
        "versions": versions,
        # scoped to source; the offending paths are listed so a reader can see
        # whose edit it was, not merely that "something" changed
        "source_paths_checked": list(SOURCE_PATHS),
        "source_dirty_paths": dirty,
        "source_clean": not dirty,
        # Fields recorded but KNOWN NOT TO DISCRIMINATE. Nothing may gate on
        # these, and a reader must not infer anything from them.
        "known_uninformative_fields": {
            "git_dirty_whole_tree": (
                "a whole-tree dirty flag always reads dirty during a run, because the run "
                "writes tracked output (reports/training/*.log, *.jsonl). It cannot "
                "distinguish modified source from a run writing its own log, so it is not "
                "recorded here and must not be gated on. Use source_dirty_paths."
            ),
        },
        "not_gated_on": ["source_clean", "source_dirty_paths", "git_rev", "timestamp_utc"],
    }


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
@dataclass
class ClaimReport:
    """Machine-readable result of one gate / ablation / audit."""

    manifest: ClaimManifest
    subchecks: list[SubCheck] = field(default_factory=list)
    baselines_run: list[BaselineResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: free-form structured payload (curves, tables) for figures
    artifacts: dict[str, Any] = field(default_factory=dict)
    kind: ReportKind = "gate"
    _provenance: dict[str, Any] = field(default_factory=provenance)

    # -- status ---------------------------------------------------------
    @property
    def status(self) -> Status:
        mandatory = [s for s in self.subchecks if s.mandatory]
        if not mandatory:
            return "COULD_NOT_RUN"
        stats = [s.status for s in mandatory]
        if any(s == "FAIL" for s in stats):
            return "FAIL"
        if any(s == "COULD_NOT_RUN" for s in stats):
            return "COULD_NOT_RUN"
        return "PASS"

    @property
    def blocking_reasons(self) -> list[str]:
        out: list[str] = []
        for s in self.subchecks:
            if not s.mandatory:
                continue
            if s.status == "COULD_NOT_RUN":
                out.append(f"{s.name}: could not run — {s.reason or 'no reason given'}")
            elif s.status == "FAIL":
                failed = [m.render() for m in s.metrics if m.passed is False]
                detail = "; ".join(failed) or s.reason
                out.append(f"{s.name}: FAILED — {detail}")
        return out

    @property
    def consequence(self) -> str | None:
        """The implementation consequence, present exactly when it applies."""
        if self.status == "FAIL":
            return self.manifest.consequence_if_failed
        return None

    # -- discipline -----------------------------------------------------
    def finalize(self) -> "ClaimReport":
        """Enforce reporting discipline.  Call before writing anything out."""
        all_metrics = [m for s in self.subchecks for m in s.metrics]
        all_metrics += [m for b in self.baselines_run for m in b.metrics]

        acc = [m for m in all_metrics if m.kind == "accuracy"]
        cal = [m for m in all_metrics if m.kind == "calibration"]
        if acc and not cal:
            raise ReportDisciplineError(
                f"{self.manifest.claim_id}: report contains accuracy metrics "
                f"({', '.join(m.name for m in acc)}) but no calibration metric. "
                "body.tex §11.2: aggregate accuracy cannot substitute for "
                "calibration within the intended deployment population."
            )
        for m in all_metrics:
            if m.kind == "diagnostic":
                continue
            if m.interval is None and not m.exact:
                raise ReportDisciplineError(
                    f"{self.manifest.claim_id}: metric {m.name!r} has no interval"
                )
        if self.status == "PASS":
            bad = [s.name for s in self.subchecks if s.mandatory and s.status != "PASS"]
            if bad:  # pragma: no cover - defensive, status property forbids it
                raise ReportDisciplineError(
                    f"{self.manifest.claim_id}: PASS with non-passing mandatory checks {bad}"
                )
            if self.kind == "gate" and not self.baselines_run:
                raise ReportDisciplineError(
                    f"{self.manifest.claim_id}: a gate may not PASS without baselines run "
                    "(ARCHITECTURE.md §4: baseline comparisons are part of 'done')"
                )
            if self.kind in ("numerics", "instrument", "adjudication") and not (
                self.artifacts.get("subject") or self.artifacts.get("solver_provenance")
            ):
                raise ReportDisciplineError(
                    f"{self.manifest.claim_id}: a passing numerical check must record what "
                    "it measured (artifacts['subject'] or artifacts['solver_provenance']). "
                    "An artifact that does not record how it was produced is how a stale "
                    "output gets compared against new code and mistaken for a result."
                )
        if self.status == "COULD_NOT_RUN":
            if not self.blocking_reasons:
                raise ReportDisciplineError(
                    f"{self.manifest.claim_id}: COULD_NOT_RUN without a reason"
                )
        return self

    # -- serialisation --------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "bench_schema_version": BENCH_SCHEMA_VERSION,
            "kind": self.kind,
            "claim_id": self.manifest.claim_id,
            "status": self.status,
            "manifest": self.manifest.as_dict(),
            "consequence_applied": self.consequence,
            "blocking_reasons": self.blocking_reasons,
            "subchecks": [s.as_dict() for s in self.subchecks],
            "baselines_run": [b.as_dict() for b in self.baselines_run],
            "notes": list(self.notes),
            "artifacts": _jsonable(self.artifacts),
            "provenance": self._provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False, default=str)

    def to_markdown(self) -> str:
        m = self.manifest
        badge = {"PASS": "PASS", "FAIL": "**FAIL**", "COULD_NOT_RUN": "COULD_NOT_RUN"}[self.status]
        L: list[str] = []
        L.append(f"# {m.claim_id} — {badge}")
        L.append("")
        L.append(f"**Claim.** {m.claim_text}")
        L.append("")
        L.append(f"**Falsified by (thesis).** {m.falsified_by}")
        L.append("")
        L.append(
            f"*thesis {m.thesis_version} · schema {m.schema_version} · bench "
            f"{m.bench_schema_version} · {m.model_designation} · seed {m.seed} · "
            f"git {self._provenance.get('git_rev')} · {self._provenance.get('timestamp_utc')}*"
        )
        L.append("")
        if self.status == "FAIL":
            L.append("## Implementation consequence (mandatory)")
            L.append("")
            L.append(f"> {m.consequence_if_failed}")
            L.append("")
        elif self.status == "COULD_NOT_RUN":
            L.append("## Could not run")
            L.append("")
            L.append(
                "> This gate did **not** pass. It did not run. Nothing may be "
                "claimed on its basis."
            )
            L.append("")
        L.append("## Sub-checks")
        L.append("")
        L.append("| check | mandatory | status | detail |")
        L.append("|---|---|---|---|")
        for s in self.subchecks:
            detail = "; ".join(mm.render() for mm in s.metrics) or s.reason or "-"
            L.append(
                f"| {s.name} | {'yes' if s.mandatory else 'no'} | {s.status} | "
                f"{detail.replace('|', '/')} |"
            )
        L.append("")
        if self.blocking_reasons:
            L.append("## Blocking reasons")
            L.append("")
            for r in self.blocking_reasons:
                L.append(f"- {r}")
            L.append("")
        L.append("## Baselines run")
        L.append("")
        if self.baselines_run:
            L.append("| baseline | role | params | flops | key metrics |")
            L.append("|---|---|---|---|---|")
            for b in self.baselines_run:
                km = "; ".join(mm.render() for mm in b.metrics) or "-"
                L.append(
                    f"| {b.name} | {b.role} | {b.n_parameters if b.n_parameters is not None else '?'} "
                    f"| {b.compute_flops if b.compute_flops is not None else '?'} "
                    f"| {km.replace('|', '/')} |"
                )
        else:
            L.append("_none run_ — no baseline, no claim.")
        L.append("")
        if m.acceptance_thresholds:
            L.append("## Preregistered acceptance thresholds")
            L.append("")
            for k, v in m.acceptance_thresholds.items():
                L.append(f"- `{k}`: {v}")
            L.append("")
        if m.permitted_source_cards:
            L.append("## Permitted source cards")
            L.append("")
            for c in m.permitted_source_cards:
                L.append(f"- `{c}`")
            L.append("")
        if m.refusal_fixtures:
            L.append("## Refusal fixtures exercised")
            L.append("")
            for c in m.refusal_fixtures:
                L.append(f"- `{c}`")
            L.append("")
        if m.non_goals:
            L.append("## Explicit non-goals")
            L.append("")
            for c in m.non_goals:
                L.append(f"- {c}")
            L.append("")
        if self.notes:
            L.append("## Notes")
            L.append("")
            for n in self.notes:
                L.append(f"- {n}")
            L.append("")
        return "\n".join(L)

    def write(self, directory: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
        self.finalize()
        base = Path(directory) if directory is not None else (
            ABLATIONS_DIR if self.kind == "ablation" else GATES_DIR
        )
        base.mkdir(parents=True, exist_ok=True)
        stem = self.manifest.claim_id.replace("/", "_")
        jp = base / f"{stem}.json"
        mp = base / f"{stem}.md"
        jp.write_text(self.to_json(), encoding="utf-8")
        mp.write_text(self.to_markdown(), encoding="utf-8")
        return jp, mp


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_reports(reports: Iterable[ClaimReport],
                  directory: str | os.PathLike[str] | None = None) -> list[Path]:
    """Write a batch of reports; returns the JSON paths."""
    out: list[Path] = []
    for r in reports:
        jp, _ = r.write(directory)
        out.append(jp)
    return out


def worst_status(statuses: Sequence[Status]) -> Status:
    """COULD_NOT_RUN > FAIL > PASS (worst wins)."""
    if not statuses:
        return "COULD_NOT_RUN"
    return max(statuses, key=lambda s: _STATUS_ORDER[s])


def load_report(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
