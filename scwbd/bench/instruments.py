"""Do my own instruments discriminate? (agent J)

``tests/bench`` already answers the gate-level question: *a gate that cannot
fail is worthless*, so every gate ships with a world in which its claim is
false and it must say so.  This module generalises that discipline one level
down, to the **guards and provenance fields the gates themselves rely on**.

The generalisation was forced by evidence.  Five times in this project an
instrument reported a discrimination it was structurally incapable of making,
and every one of them looked green:

1. ``torch.compile`` renamed parameters, so exact-name gradient permissions
   matched *nothing* on CUDA while passing on CPU — the permission check was
   reading a name that no longer existed;
2. ``systemd-run MemoryMax`` did not charge CUDA allocations on unified
   memory, so ``memory.current`` read a reassuring 8 GB against a 40 GB cap
   that was not binding anything;
3. at OOM, "allocated by PyTorch" always equalled the ceiling, so the number
   could not distinguish batch-linear growth from batch-independent growth;
4. ``git_sha()`` appended ``-dirty`` whenever the whole tree was dirty — which
   it always is during a run, because the run writes tracked logs — so every
   checkpoint ever produced carried the same flag, and it could not
   distinguish modified source from a run writing its own output;
5. **this module's own owner did it too**: ``run_g4`` gated its parameter-partition
   probe on how the Fisher map arrived, so a caller passing a bound map got a
   ``COULD_NOT_RUN`` whose stated reason was not the actual reason.

The fourth appeared *inside the mechanism built to catch stale artifacts*, and
the fifth inside the gate machinery itself — which is the sharpest possible
argument that this belongs in the falsification machinery rather than in a
comment.  A variant worth naming: an instrument can also fail by reporting a
**reason that is not the actual reason**, which is a discrimination failure
about causes rather than about values.

The rule this module enforces:

    **For every guard or provenance field a claim relies on, there must exist
    an input under which it reads differently. If there is not, it is
    decoration, and it must be labelled as such rather than reported.**

:func:`audit_instruments` runs each registered instrument over at least two
labelled inputs and FAILs any whose reads are all identical.  The audit has its
own negative control in ``tests/bench``: a deliberately constant instrument
must make it fail.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .report import (
    ClaimManifest,
    ClaimReport,
    Metric,
    SubCheck,
    could_not_run,
    source_dirty_entries,
)

__all__ = [
    "Instrument",
    "UninformativeField",
    "KNOWN_UNINFORMATIVE",
    "default_instruments",
    "audit_instruments",
]


# ==========================================================================
# the register of fields known not to discriminate
# ==========================================================================
@dataclass(frozen=True)
class UninformativeField:
    """A field that is recorded but cannot vary, with the fix."""

    name: str
    reads: str
    why_it_cannot_discriminate: str
    remedy: str
    found_by: str
    owner: str
    still_reported: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reads": self.reads,
            "why_it_cannot_discriminate": self.why_it_cannot_discriminate,
            "remedy": self.remedy,
            "found_by": self.found_by,
            "owner": self.owner,
            "still_reported": self.still_reported,
        }


KNOWN_UNINFORMATIVE: tuple[UninformativeField, ...] = (
    UninformativeField(
        name="git_sha() -dirty suffix (whole-tree scope)",
        reads="always '-dirty' during any run",
        why_it_cannot_discriminate=(
            "the run writes tracked output (reports/training/train_main.log, "
            "reports/training/scwbd-001-beta_train.jsonl), so git status --porcelain is "
            "never empty while a run is in flight; the flag therefore cannot separate "
            "'source was modified' from 'the run wrote its own log', and every checkpoint "
            "the project has produced carries it"
        ),
        remedy=(
            "scope the check to source paths and record the offending PATHS rather than a "
            "boolean: scwbd.bench.report.source_dirty_entries(SOURCE_PATHS)"
        ),
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
    UninformativeField(
        name="run_g4 parameter_partition COULD_NOT_RUN reason (agent J's own bug)",
        reads="'theta_index / nuisance_index not supplied' whenever fisher was passed bound",
        why_it_cannot_discriminate=(
            "the partition probe was gated on auto_probed, which is only set when fisher is "
            "None; a caller passing a BOUND fisher map -- the only usable form, since bare "
            "expected_fisher needs u/cfg/proto -- skipped the probe entirely, so the gate "
            "reported a reason that was not the actual reason and never reached "
            "fisher_information at all"
        ),
        remedy=(
            "drop the auto_probed guard: the probe returns a Dependency that reports its own "
            "unavailability, so nothing is lost. Regression-tested by "
            "test_g4_resolves_the_partition_from_agent_h_even_for_a_bound_fisher"
        ),
        found_by="agent Fisher (running the gate end to end)",
        owner="bench (agent J)",
        still_reported=False,
    ),
    UninformativeField(
        name="exact-name gradient permission matching under torch.compile",
        reads="'permission matched' on CPU, silently matches nothing on CUDA",
        why_it_cannot_discriminate=(
            "torch.compile renames parameters, so a permission keyed on an exact parameter "
            "name matches an empty set and the source appears authorised while updating "
            "nothing"
        ),
        remedy=(
            "N1's gradient.unmatched_permission_patterns metric fails when a permission "
            "pattern matches nothing; run it against the compiled module, not only the "
            "eager one"
        ),
        found_by="agent Turing",
        owner="foundation (agent I) / compiler (agent A)",
        still_reported=True,
    ),
    UninformativeField(
        name="systemd-run MemoryMax against CUDA unified memory",
        reads="memory.current ~8 GB against a 40 GB cap",
        why_it_cannot_discriminate=(
            "CUDA allocations on unified memory are not charged to the cgroup, so the cap "
            "is not binding and the reassuring number is measuring the wrong pool"
        ),
        remedy="measure the allocator's own accounting, and prove the cap binds by exceeding it",
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
    UninformativeField(
        name="'allocated by PyTorch' at OOM",
        reads="always equal to the ceiling",
        why_it_cannot_discriminate=(
            "at the moment of OOM the allocated figure is pinned to the limit by "
            "construction, so it cannot distinguish batch-linear from batch-independent "
            "growth — the question the number was consulted to answer"
        ),
        remedy="sweep batch size and fit the growth curve; a single reading at OOM cannot",
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
)


# ==========================================================================
# instruments
# ==========================================================================
@dataclass
class Instrument:
    """A guard, plus inputs under which it must read differently."""

    name: str
    description: str
    read: Callable[[Any], Any]
    inputs: Mapping[str, Any]
    #: what a non-discriminating reading would mean
    consequence: str = "the field is decoration and must not be reported as evidence"
    mandatory: bool = True

    def evaluate(self) -> tuple[dict[str, str], int]:
        reads: dict[str, str] = {}
        for label, value in self.inputs.items():
            try:
                reads[label] = repr(self.read(value))
            except Exception as exc:
                reads[label] = f"<raised {type(exc).__name__}: {exc}>"
        return reads, len(set(reads.values()))


# -- the concrete instruments this bench relies on -------------------------
def _tmp_git_repo(*, modify_source: bool, write_output: bool) -> str:
    """A throwaway repo in one of three states, for the dirty-flag instrument."""
    d = tempfile.mkdtemp(prefix="scwbd-instr-")
    src = Path(d) / "scwbd"
    out = Path(d) / "reports" / "training"
    src.mkdir(parents=True)
    out.mkdir(parents=True)
    (src / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (out / "train.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=d, env=env, capture_output=True, timeout=30)
    if modify_source:
        (src / "mod.py").write_text("x = 2\n", encoding="utf-8")
    if write_output:                      # the run writing its own tracked log
        (out / "train.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    return d


def default_instruments() -> list[Instrument]:
    """Guards this bench relies on, each with a discriminating input pair."""
    from .matching import check_matched
    from .report import Interval, Metric as M
    from .statistics import smoothing_check

    class _Sized:
        def __init__(self, n: int) -> None:
            self.n = n

        def n_parameters(self) -> int:
            return self.n

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(200, 2))
    honest = y + rng.normal(0, 0.2, size=y.shape)
    flat = np.zeros_like(y) + y.mean(axis=0)
    effect = lambda a: float(np.std(a.reshape(a.shape[0], -1), axis=0).mean())

    return [
        Instrument(
            name="source_dirty_flag",
            description=(
                "Scoped source-dirty provenance: must read clean when the run writes its "
                "own tracked output, and dirty when source is modified. The whole-tree "
                "flag it replaces could not tell those apart."
            ),
            read=lambda repo: source_dirty_entries(("scwbd",), cwd=repo),
            inputs={
                "clean": _tmp_git_repo(modify_source=False, write_output=False),
                "run_wrote_its_own_log": _tmp_git_repo(modify_source=False,
                                                       write_output=True),
                "source_modified": _tmp_git_repo(modify_source=True, write_output=False),
            },
            consequence=(
                "provenance cannot distinguish a modified source tree from a run writing "
                "its own log; no reader may infer anything from it"
            ),
        ),
        Instrument(
            name="capacity_matching",
            description="Matched-capacity guard must reject an over-budget candidate.",
            read=lambda pair: check_matched(_Sized(pair[0]), {"b": _Sized(pair[1])}).matched,
            inputs={"equal": (100, 100), "candidate_3x_larger": (300, 100),
                    "candidate_smaller": (50, 100)},
            consequence="an unmatched comparison would be reported as a win",
        ),
        Instrument(
            name="interval_strict_threshold",
            description=(
                "Interval-strict metrics must reject a win whose interval straddles the "
                "threshold and accept one that clears it."
            ),
            read=lambda iv: M(name="d", value=0.02, kind="accuracy", interval=iv,
                              threshold=0.0,
                              require_interval_beats_threshold=True).passed,
            inputs={"noisy": Interval(-0.05, 0.09), "clean": Interval(0.01, 0.03)},
            consequence="a noisy difference would be reported as a reproducible gain",
        ),
        Instrument(
            name="smoothing_check",
            description=(
                "The §11.4 smoothing guard must fire on a flat predictor and not on an "
                "honest one."
            ),
            read=lambda pred: smoothing_check(
                arm_name="a", reference_name="b", y_true=y, pred_arm=pred,
                pred_reference=honest, effect=effect, seed=0, n_boot=80).smoothed_away,
            inputs={"flat_predictor": flat, "honest_predictor": honest},
            consequence="a model that won by destroying the effect would be preferred",
        ),
        Instrument(
            name="report_provenance_rule",
            description=(
                "finalize() must refuse a passing numerical report that does not record "
                "what it measured, and accept one that does."
            ),
            read=_provenance_rule_reads,
            inputs={"without_subject": False, "with_subject": True},
            consequence="a stale artifact could pass without recording how it was produced",
        ),
    ]


def _provenance_rule_reads(with_subject: bool) -> str:
    from .report import ReportDisciplineError

    rep = ClaimReport(
        manifest=ClaimManifest(
            claim_id="INSTRUMENT_PROBE", claim_text="probe", falsified_by="probe",
            consequence_if_failed="probe",
        ),
        subchecks=[SubCheck(name="ok", description="d", metrics=[
            Metric(name="x", value=1.0, kind="numerical", exact=True, threshold=0.5)])],
        kind="numerics",
    )
    if with_subject:
        rep.artifacts["subject"] = "probe"
    try:
        rep.finalize()
        return "accepted"
    except ReportDisciplineError:
        return "refused"


# ==========================================================================
def audit_instruments(instruments: Sequence[Instrument] | None = None, *,
                      seed: int = 0) -> ClaimReport:
    """FAIL any guard or provenance field that cannot read differently."""
    man = ClaimManifest(
        claim_id="N7_instrument_discrimination",
        claim_text=(
            "Every guard and provenance field this bench relies on has an input under "
            "which it reads differently, so a green reading is evidence rather than "
            "decoration."
        ),
        falsified_by=(
            "an instrument that returns the same reading on every input it was given — it "
            "is structurally incapable of reporting the discrimination it is consulted for"
        ),
        consequence_if_failed=(
            "Stop reporting the affected field, and stop gating on it. Replace it with a "
            "measurement that varies with the thing it claims to measure, or label it "
            "explicitly as decoration in the report where it appears."
        ),
        thesis_reference=(
            "generalises ARCHITECTURE.md §4 ('a gate that fails is a result') and this "
            "bench's negative-control discipline from gates to their instruments"
        ),
        acceptance_thresholds={"min_distinct_reads": 2},
        non_goals=[
            "This audit does not check that an instrument is CORRECT, only that it is "
            "capable of varying. A field that varies can still be wrong.",
        ],
        seed=seed,
    )
    try:
        instruments = list(instruments) if instruments is not None else default_instruments()
    except Exception as exc:  # pragma: no cover - environment dependent
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run("instruments", "Build the instrument set.",
                                     f"could not construct instruments: "
                                     f"{type(exc).__name__}: {exc}")],
            kind="instrument",
            artifacts={"subject": "scwbd.bench guards"},
        ).finalize()

    subs: list[SubCheck] = []
    table: dict[str, Any] = {}
    for inst in instruments:
        if len(inst.inputs) < 2:
            subs.append(could_not_run(
                inst.name, inst.description,
                f"only {len(inst.inputs)} input supplied; discrimination needs at least two",
                mandatory=inst.mandatory))
            continue
        reads, n_distinct = inst.evaluate()
        table[inst.name] = {"reads": reads, "distinct": n_distinct,
                            "consequence_if_constant": inst.consequence}
        subs.append(SubCheck(
            name=inst.name,
            description=inst.description,
            metrics=[Metric(
                name=f"{inst.name}.distinct_reads", value=float(n_distinct),
                kind="audit", exact=True, threshold=1.5,
                direction="greater_is_better",
                note="; ".join(f"{k} -> {v}" for k, v in reads.items())[:400],
            )],
            mandatory=inst.mandatory,
            falsified_by=f"identical reading on every input: {inst.consequence}",
        ))

    return ClaimReport(
        manifest=man, subchecks=subs, kind="instrument",
        artifacts={
            "subject": "the guards and provenance fields of scwbd.bench",
            "instrument_reads": table,
            "known_uninformative_fields": [u.as_dict() for u in KNOWN_UNINFORMATIVE],
        },
        notes=[
            "A green reading from an instrument that cannot vary is not evidence. Four such "
            "instruments have already been found in this project; the fourth was inside the "
            "mechanism built to catch stale artifacts.",
            "This audit checks capability to vary, not correctness. An instrument that "
            "varies can still be measuring the wrong thing.",
        ],
    ).finalize()
