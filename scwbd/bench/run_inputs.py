"""The seam between a trained run and the claim gates.

`run_all_gates` has always taken a config -- `run_g5(**cfg["G5"])` -- and nothing has ever passed
one. So every gate has been constructed with no candidate, no datasets and no baselines, and has
correctly reported `COULD_NOT_RUN` naming what it was not given. That is not a stale report and it
is not a broken adapter; it is an empty input set, faithfully described.
(`notes/findings/2026-08-14-the-gate-reports-are-not-stale-nothing-supplies-inputs.md`)

This module builds that config from a run's artifacts. What it will not do is more important than
what it does.

THE THREE REFUSALS
------------------

**It never fabricates an input.** If a gate wants a baseline nobody trained, the key is absent from
the config and the gate says so. An absent input produces `COULD_NOT_RUN`, which is the honest
answer; a fabricated one produces a verdict about a different question in the same words.

**It never aliases one input as another.** `new_session` and `unseen_task` are both holdouts and
both exist as `Dataset` objects, so passing one for the other costs nothing mechanically and is
caught by nothing downstream -- the gate would run, report a number, and that number would answer
"does the person effect predict a held-out NIGHT" while the report says "an unseen TASK". Aliasing
is refused explicitly, by name, in `_refuse_aliases`.

**It never relaxes a mandatory sub-check.** If G5 wants `population`, `anatomy_only` and
`session_adapted`, supplying two of them yields `COULD_NOT_RUN` naming the third. That is the
designed behaviour and this module must not route around it.

WHY THE REFUSALS ARE THE PRODUCT

The gates exist so that "we have not measured this" cannot quietly become "we measured this". The
easiest way to break that is a well-meaning adapter that fills a slot with the nearest available
object, because the resulting green gate is indistinguishable from a real one. G5's own docstring
carries the sharpest instance: *"Including the person's scan is not personalization"* -- the
`anatomy_only` baseline is given the person's anatomy deliberately, and a candidate that beats a
population baseline but not `anatomy_only` supports the claim that anatomy is informative, which is
a weaker and different claim than the gate's name.

WHAT IS AND IS NOT BUILT HERE

Built: the seam, the refusals, the artifact discovery, and the reporting of what a run genuinely
holds. Not built: the candidate wrapper (a `predict(Dataset) -> Prediction` around a trained
checkpoint) and the baseline arms, because those are training and evaluation work rather than
wiring. Each is named in `missing` with what would supply it, and `scratch/CLAIM_GATES.md` carries
the cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["GateInputs", "RunArtifacts", "discover", "gates_config", "AliasRefusal"]

ROOT = Path(__file__).resolve().parents[2]

#: Inputs that are datasets, per gate. Two of these being interchangeable at the type level is
#: exactly why `_refuse_aliases` exists.
DATASET_SLOTS: Mapping[str, tuple[str, ...]] = {
    "G1": ("train", "test"),
    "G2": ("train", "test"),
    "G3": ("train", "test", "coarse_eval"),
    "G5": ("train", "new_session", "unseen_task"),
}

#: Baselines each gate declares mandatory. Read from the gates' own `required` tuples; kept here so
#: this module can report what a run is short of without importing and half-running a gate.
MANDATORY_BASELINES: Mapping[str, tuple[str, ...]] = {
    "G1": ("naive_resampling",),
    "G2": ("dense", "randomized", "distance_matched"),
    "G3": ("coarse_only",),
    "G5": ("population", "anatomy_only", "session_adapted"),
}


class AliasRefusal(Exception):
    """Raised when two distinct gate inputs would be given the same object.

    Not a ValueError, and not a warning. This is the failure this module exists to prevent, and it
    must be impossible to catch by accident alongside ordinary argument validation.
    """


@dataclass
class GateInputs:
    """What one gate can be given from a run, and what it is short of.

    `present` goes into the config verbatim. `missing` is not passed anywhere -- the gate rediscovers
    it and reports it, which keeps one authority for what a gate requires. It is carried here so a
    caller can print the shortfall without running anything.
    """

    gate: str
    present: dict[str, Any] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        if self.complete:
            return f"{self.gate}: all declared inputs present"
        lines = [f"{self.gate}: {len(self.missing)} input(s) short"]
        for k, why in sorted(self.missing.items()):
            lines.append(f"  {k}: {why}")
        return "\n".join(lines)


@dataclass
class RunArtifacts:
    """What a run left on disk, resolved once so nothing below guesses at paths."""

    run: str
    checkpoint_dir: Path | None = None
    evaluation: Path | None = None
    ablation: Path | None = None

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_dir is not None and any(self.checkpoint_dir.glob("*.pt"))


def discover(run: str = "scwbd-004", *, root: Path | None = None) -> RunArtifacts:
    """Resolve a run's artifacts. Absent is absent -- nothing here invents a path."""
    base = root or ROOT
    ckpt = base / "checkpoints" / run
    ev = base / "reports" / "training" / f"evaluation_{_run_key(run)}.json"
    abl = base / "reports" / "training" / f"evaluation_{_run_key(run)}_ablation.json"
    return RunArtifacts(
        run=run,
        checkpoint_dir=ckpt if ckpt.is_dir() else None,
        evaluation=ev if ev.is_file() else None,
        ablation=abl if abl.is_file() else None,
    )


def _run_key(run: str) -> str:
    """`scwbd-004` -> `run4`, which is what the evaluation artifacts are named after."""
    tail = run.rsplit("-", 1)[-1]
    return f"run{int(tail)}" if tail.isdigit() else run


def _refuse_aliases(gate: str, present: Mapping[str, Any]) -> None:
    """Two dataset slots must never hold the same object.

    Identity, not equality: two datasets built separately from the same rows are a different
    (and legitimate) thing from one object handed to two slots. What is refused is the shortcut --
    `unseen_task=new_session` -- which type-checks, runs, and produces a number that answers a
    question the report does not ask.
    """
    slots = [s for s in DATASET_SLOTS.get(gate, ()) if s in present]
    for i, a in enumerate(slots):
        for b in slots[i + 1:]:
            if present[a] is present[b]:
                raise AliasRefusal(
                    f"{gate}: {a!r} and {b!r} were given the same object. They are different "
                    f"holdouts and answer different questions; supplying one for the other would "
                    f"produce a verdict whose label does not match what was measured. Build the "
                    f"missing one or leave it out and let the gate report COULD_NOT_RUN."
                )


def g5_inputs(art: RunArtifacts) -> GateInputs:
    """G5 -- individualization improves future prediction.

    The closest gate to reachable: a run with a session split already holds the candidate and the
    new-session holdout. It is short an unseen-task holdout and all three baselines.
    """
    out = GateInputs(gate="G5")

    if art.has_checkpoint:
        # The checkpoint exists; a `predict(Dataset) -> Prediction` wrapper around it does not.
        # Reporting the path as if it were a model would be the fabrication this module refuses.
        out.missing["candidate"] = (
            f"checkpoint present at {art.checkpoint_dir}, but no arm wrapper exposes "
            "predict(Dataset) -> Prediction over it (scwbd.bench.harness.evaluate needs that "
            "protocol, not a state dict)"
        )
    else:
        out.missing["candidate"] = f"no checkpoint for {art.run}"

    out.missing["train"] = (
        "the session split exists in scwbd.foundation.evaluate.session_split, but is not exported "
        "as a bench Dataset (targets + inputs + groups)"
    )
    out.missing["new_session"] = (
        "run 4 measured this holdout (75 sleep-EDFx participants recorded twice, scored on night "
        "2) but it is not exported as a bench Dataset"
    )
    out.missing["unseen_task"] = (
        "NO SUCH HOLDOUT EXISTS for this run. The claim needs an unseen task or intervention for "
        "the same people; the session split provides a new NIGHT, which is a different question. "
        "Do not alias one for the other -- see _refuse_aliases"
    )
    for b in MANDATORY_BASELINES["G5"]:
        out.missing[f"baseline:{b}"] = "not trained; one arm, and it is mandatory for this gate"
    return out


#: Gates whose inputs this module knows how to look for. The others are not silently skipped --
#: `gates_config` reports them as unmapped, which is a different and honest statement from "short".
_BUILDERS = {"G5": g5_inputs}


def gates_config(run: str = "scwbd-004", *, root: Path | None = None
                 ) -> tuple[dict[str, dict[str, Any]], dict[str, GateInputs]]:
    """Build the `config["gates"]` mapping for `run_everything`, plus the shortfall per gate.

    The config carries ONLY inputs that genuinely exist. A gate with nothing available contributes
    an empty dict, which is exactly what it gets today -- so wiring this in can never turn a
    `COULD_NOT_RUN` into a verdict on its own. That property is the point, and
    `tests/bench/test_run_inputs_cannot_flip_a_gate.py` holds it.
    """
    art = discover(run, root=root)
    report: dict[str, GateInputs] = {}
    cfg: dict[str, dict[str, Any]] = {}
    for gate, build in _BUILDERS.items():
        gi = build(art)
        _refuse_aliases(gate, gi.present)
        report[gate] = gi
        if gi.present:
            cfg[gate] = dict(gi.present)
    return cfg, report
