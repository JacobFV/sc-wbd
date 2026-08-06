"""Scoreboard generator: ``reports/gates/SUMMARY.md`` (agent J).

Runs every gate, ablation, Appendix D audit and §11.1 numerical check with
whatever inputs exist, writes one JSON + one Markdown per claim, and then
writes the top-level scoreboard the architect reads to decide **what
SC-WBD-001-beta is allowed to claim**.

Run it with::

    .venv/bin/python -m scwbd.bench

With no configuration, a claim reports ``COULD_NOT_RUN`` unless its subject has
actually landed — the only current exception is ``N1``, which compiles agent
A's reference three-region example and checks *that*.  The blank result is
deliberately not softened: an unrun gate supports nothing, and a passing
numerical check licenses a statement about code, never about a brain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import adapters
from .ablations import ABLATIONS, run_all_ablations
from .gates import CLAIMS, run_all_gates
from .leakage import APPENDIX_D_ROWS, run_all_audits
from .numerics import run_numerics_suite
from .report import (
    ABLATIONS_DIR,
    BENCH_SCHEMA_VERSION,
    GATES_DIR,
    MODEL_DESIGNATION,
    SCHEMA_VERSION,
    THESIS_VERSION,
    ClaimReport,
    provenance,
)

__all__ = ["build_summary", "run_everything", "main"]

_BADGE = {"PASS": "PASS", "FAIL": "**FAIL**", "COULD_NOT_RUN": "could-not-run"}


def _headline(rep: ClaimReport) -> str:
    """One number or one reason, whichever is the honest summary."""
    if rep.status == "COULD_NOT_RUN":
        r = rep.blocking_reasons
        return (r[0][:150] + ("…" if len(r[0]) > 150 else "")) if r else "no reason recorded"
    failed = [m for s in rep.subchecks for m in s.metrics if m.passed is False]
    if failed:
        m = failed[0]
        iv = f" {m.interval}" if m.interval else ""
        return f"`{m.name}` = {m.value:.4g}{iv} vs threshold {m.threshold:.4g}"
    keyed = [m for s in rep.subchecks for m in s.metrics if m.kind == "accuracy"]
    if keyed:
        m = keyed[0]
        iv = f" {m.interval}" if m.interval else ""
        return f"`{m.name}` = {m.value:.4g}{iv}"
    mand = [s for s in rep.subchecks if s.mandatory]
    tally = f"{sum(1 for s in mand if s.status == 'PASS')}/{len(mand)} mandatory sub-checks"
    subject = rep.artifacts.get("subject")
    return f"{tally}; subject: {subject}" if subject else tally


def _negative_controls() -> list[str]:
    """Names of the tests that prove the gates can fail.

    Read off disk rather than hard-coded, so this section cannot drift away
    from what is actually tested.
    """
    root = Path(__file__).resolve().parents[2] / "tests" / "bench"
    markers = ("fail", "fires", "catches", "refus")
    out: list[str] = []
    for path in sorted(root.glob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover
            continue
        for line in text.splitlines():
            if line.startswith("def test_") and any(m in line for m in markers):
                out.append(f"{path.name}::{line[4:].split('(')[0]}")
    return out or ["(no negative-control tests found — the gates are unverified)"]


def _table(reports: Sequence[ClaimReport], *, title: str, describe: Mapping[str, str]) -> list[str]:
    L = [f"## {title}", ""]
    L.append("| id | status | headline number or blocker | consequence if failed |")
    L.append("|---|---|---|---|")
    for r in reports:
        cid = r.manifest.claim_id
        cons = r.manifest.consequence_if_failed if r.status == "FAIL" else "—"
        L.append(
            f"| `{cid}` | {_BADGE[r.status]} | {_headline(r).replace('|', '/')} | "
            f"{cons.replace('|', '/')} |"
        )
    L.append("")
    return L


def build_summary(
    gates: Sequence[ClaimReport],
    ablations: Sequence[ClaimReport],
    audits: Sequence[ClaimReport],
    numerics: Sequence[ClaimReport],
) -> str:
    prov = provenance()
    all_reports = list(gates) + list(ablations) + list(audits) + list(numerics)
    n_pass = sum(1 for r in all_reports if r.status == "PASS")
    n_fail = sum(1 for r in all_reports if r.status == "FAIL")
    n_cnr = sum(1 for r in all_reports if r.status == "COULD_NOT_RUN")

    L: list[str] = []
    L.append("# SC-WBD-001-beta — claim gate scoreboard")
    L.append("")
    L.append(
        f"*thesis {THESIS_VERSION} · schema {SCHEMA_VERSION} · bench {BENCH_SCHEMA_VERSION} · "
        f"{MODEL_DESIGNATION} · git {prov['git_rev']} · {prov['timestamp_utc']}*"
    )
    L.append("")
    L.append(
        f"**{n_pass} PASS · {n_fail} FAIL · {n_cnr} COULD_NOT_RUN** "
        f"out of {len(all_reports)} claim-bearing checks."
    )
    L.append("")
    L.append(
        "> A gate that cannot run is **not** a gate that passed. Nothing in this repository "
        "may be claimed on the basis of a `could-not-run` row. Engineering breadth, parameter "
        "count, plausible diagrams, and in-sample fit are not substitutes for these tests "
        "(`thesis_contract.tex`)."
    )
    L.append("")

    L.append("## 0. Is the machinery itself trustworthy?")
    L.append("")
    L.append(
        "A gate that cannot fail is worthless. Each gate therefore ships with a "
        "negative control: a synthetic world in which its claim is false by construction, "
        "and in which the gate is required to report `FAIL`. These are the negative "
        "controls currently in `tests/bench`:"
    )
    L.append("")
    for name in _negative_controls():
        L.append(f"- `{name}`")
    L.append("")
    L.append(
        "Positive controls (worlds where the effect is present, and the gate must `PASS`) "
        "live in `tests/bench/test_gates_can_pass.py`; a gate that can never pass is not a "
        "measurement either."
    )
    L.append("")

    L += _table(list(gates), title="1. Claim gates G1–G5 (`tab:claim-gates`)",
                describe={k: v["claim"] for k, v in CLAIMS.items()})
    L.append("### What each gate is testing")
    L.append("")
    for gid, c in CLAIMS.items():
        L.append(f"- **{gid}** — {c['claim']}")
        L.append(f"  - falsified by: *{c['falsified_by']}*")
        L.append(f"  - if it fails: {c['consequence']}")
    L.append("")

    L.append("### A tautology this scoreboard refuses to report as a result")
    L.append("")
    L.append(
        "Under the modality-block-diagonal form of T4, the joint expected Fisher "
        "information is the sum of the per-modality informations **identically**: "
        "`I_{EEG+BOLD} = I_EEG + I_BOLD`. \"Joint fusion beats single-modality\" is "
        "therefore *arithmetic* in that form — it cannot fail, so it is not evidence "
        "for claim G1 and no gate here reports it as such. G4 measures the additivity "
        "residual explicitly (`modality_additivity_declaration`) so that the identity is "
        "named rather than exploited."
    )
    L.append("")
    L.append("The comparisons that **can** fail, and which therefore carry the claims:")
    L.append("")
    L.append(
        "- **G4** — intervention design versus baseline design, on the theta block with "
        "the observation nuisances profiled out. This is what G4 tests."
    )
    L.append(
        "- **G1** — native-clock versus naively resampled inference (agent H's design "
        "benchmark), and held-out *predictive* log score between fitted models, where a "
        "fusion model with more inputs can and does lose out of sample."
    )
    L.append(
        "- **G1 (information side)** — the non-additive joint information that appears "
        "only under `joint_whitening=True`, carried by the EEG/BOLD cross-covariance from "
        "shared process noise. That excess over the modality sum is the honest "
        "information-theoretic content of the typed-fusion claim, and it can be zero."
    )
    L.append("")
    L.append(
        "Every eigenvalue and condition number in a G4 report travels with its basis "
        "(default `prior_standardised`, in which `I_prior` is the identity). A condition "
        "number without a declared basis is not interpretable."
    )
    L.append("")

    L += _table(list(ablations), title="2. Required ablations (`body.tex` §11.4)",
                describe={k: v.thesis_clause for k, v in ABLATIONS.items()})
    L += _table(list(audits), title="3. Leakage and evaluation audits (Appendix D, "
                                   "`tab:mixture-evaluation`)",
                describe={k: v["failure_mode"] for k, v in APPENDIX_D_ROWS.items()})
    L += _table(list(numerics), title="4. Numerical, representational and physical tests "
                                     "(§11.1)",
                describe={})

    # dependency state
    L.append("## 5. Dependency state (who is blocking what)")
    L.append("")
    L.append("| module | owner | available |")
    L.append("|---|---|---|")
    for row in adapters.dependency_table():
        L.append(f"| `{row['module']}` | agent {row['agent']} | "
                 f"{'yes' if row['available'] else 'no'} |")
    L.append("")

    # what has actually been earned, stated at its true scope
    L.append("## 6. What is licensed so far")
    L.append("")
    passed = [r for r in all_reports if r.status == "PASS"]
    if not passed:
        L.append(
            "**Nothing.** No claim-bearing check has passed, so no statement in this "
            "repository is currently supported by a gate."
        )
    else:
        L.append(
            "Only the following, and only at the scope stated. A passing check licenses "
            "exactly its own sentence — not a generalisation of it:"
        )
        L.append("")
        for r in passed:
            L.append(f"- **{r.manifest.claim_id}**: {r.manifest.claim_text}")
            subject = r.artifacts.get("subject")
            if subject:
                L.append(f"  - subject: {subject}")
            for n in r.notes:
                if "not evidence" in n or "not a statement" in n:
                    L.append(f"  - scope limit: {n}")
    L.append("")

    # the section that matters
    L.append("## 7. What we cannot yet claim")
    L.append("")
    blocked = [r for r in all_reports if r.status != "PASS"]
    if not blocked:
        L.append("Every claim-bearing check passed. Each claim is licensed only at the "
                 "strength its gate measured, and only inside the declared validity domain.")
    else:
        L.append("Each line below is a claim SC-WBD-001-beta **may not make** in text, "
                 "figures, abstracts, or a model card, because the gate that would license it "
                 "did not pass:")
        L.append("")
        for r in blocked:
            why = "FAILED" if r.status == "FAIL" else "did not run"
            L.append(f"- **{r.manifest.claim_id}** ({why}): {r.manifest.claim_text}")
            if r.status == "FAIL":
                L.append(f"  - required action: {r.manifest.consequence_if_failed}")
            else:
                for reason in r.blocking_reasons[:2]:
                    L.append(f"  - blocked by: {reason}")
    L.append("")
    L.append("### Standing exclusions (independent of any result)")
    L.append("")
    L.append(
        "- **No digital-twin claim.** SC-WBD-001-beta is not a validated model of any specific "
        "person, and no gate here can make it one."
    )
    L.append(
        "- **No clinical, wellness or treatment claim.** Appendix D row `D10` is a standing "
        "refusal: prospective human TMS/tFUS is out of scope (no IRB, no consent, no "
        "participants), so decision validity is unmeasured and unmeasurable in this release."
    )
    L.append(
        "- **No mechanism claim without its gate.** A mechanistic label is earned only by "
        "predictions an equal-capacity generic surrogate misses, on a held-out perturbation."
    )
    L.append(
        "- **No consciousness or Phi claim.** There is no ground truth and no estimate here "
        "(ARCHITECTURE.md rule 4)."
    )
    L.append("")
    L.append("## 8. How to change a row in this table")
    L.append("")
    L.append(
        "Supply the missing evidence to the gate, not a smaller threshold. Thresholds are "
        "preregistered in each report's manifest; changing one changes the claim class and "
        "must be recorded as an override in the `ClaimManifest`, where it stays visible in the "
        "artifact's provenance."
    )
    L.append("")
    return "\n".join(L)


def run_everything(config: Mapping[str, Any] | None = None, *, seed: int = 0,
                   write: bool = True) -> dict[str, list[ClaimReport]]:
    cfg = dict(config or {})
    gates = run_all_gates(cfg.get("gates"), seed=seed)
    ablations = run_all_ablations(cfg.get("ablations"), seed=seed)
    audits = run_all_audits(cfg.get("leakage"), seed=seed)
    numerics = run_numerics_suite(seed=seed, **dict(cfg.get("numerics", {})))

    if write:
        GATES_DIR.mkdir(parents=True, exist_ok=True)
        ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        for r in gates:
            r.write(GATES_DIR)
        for r in ablations:
            r.write(ABLATIONS_DIR)
        for r in audits:
            r.write(GATES_DIR / "leakage")
        for r in numerics:
            r.write(GATES_DIR / "numerics")
        (GATES_DIR / "SUMMARY.md").write_text(
            build_summary(gates, ablations, audits, numerics), encoding="utf-8"
        )
    return {"gates": gates, "ablations": ablations, "leakage": audits, "numerics": numerics}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run SC-WBD claim gates and write the scoreboard.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="do not write reports/")
    args = ap.parse_args(argv)
    out = run_everything(seed=args.seed, write=not args.no_write)
    flat = [r for group in out.values() for r in group]
    n_pass = sum(1 for r in flat if r.status == "PASS")
    n_fail = sum(1 for r in flat if r.status == "FAIL")
    n_cnr = len(flat) - n_pass - n_fail
    print(f"{n_pass} PASS, {n_fail} FAIL, {n_cnr} COULD_NOT_RUN ({len(flat)} checks)")
    if not args.no_write:
        print(f"scoreboard: {GATES_DIR / 'SUMMARY.md'}")
    for r in flat:
        print(f"  {r.status:>14}  {r.manifest.claim_id}")
    # exit non-zero when something FAILED: a failing gate is a result that must
    # be seen, and CI should stop on it rather than scroll past it.
    return 1 if n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
