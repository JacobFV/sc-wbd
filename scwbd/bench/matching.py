"""Matched capacity and compute accounting (agent J).

``thesis_contract.tex`` Table ``tab:claim-gates`` requires the fusion claim to
be demonstrated "at matched compute and parameter count", and Appendix D
requires the connectome controls "at matched parameter/compute budgets".

This module makes that mandatory rather than rhetorical:

* :func:`budget_of` extracts a parameter/compute budget from an arm by duck
  typing (``n_parameters()``, ``parameters()`` for torch modules, an explicit
  ``budget`` attribute);
* :func:`check_matched` compares the candidate against every baseline on
  **every binding field either side declares**, and returns a verdict;
* :func:`matched_subcheck` turns that verdict into a **mandatory** sub-check
  which is ``COULD_NOT_RUN`` when the candidate is over budget, when the
  accounting is asymmetric, or when the budgets are unknown.  It is never a
  ``PASS`` by default.

The asymmetry is deliberate.  A candidate that wins with *fewer* parameters
than its baselines has produced evidence; a candidate that wins with more has
produced a bigger model.

Which fields bind, and the one that does not
--------------------------------------------
Until 2026-08-06 this module compared ``n_parameters`` **and nothing else**,
while :class:`Budget` declared ``flops``, ``train_steps`` and ``wall_seconds``
and ``ablations.py`` promised arms "at matched capacity **and compute**".  A
comparison could be compute-unmatched by any factor and still carry a green
``matched_capacity`` check.  That is a guard that cannot fire, in the module
whose job is to make the §11.4 comparison honest.  Closed by 🛡️ Popper; see
``reports/ablations/PREREG_A1_run2.md`` §3.4, which named it before it was hit.

Three rules now hold, and the third is the one that keeps the fix from being
cosmetic:

1. **Every field in :data:`BINDING_FIELDS` binds when both sides declare it.**
   Over budget on any of them is over budget.
2. **Declared on one side only is a defect, not a skip.** Comparing arms whose
   accounting does not even cover the same quantities cannot support "at
   matched compute and parameter count", so it is ``COULD_NOT_RUN``.
3. **A field no arm declares is named in the verdict.** It does not silently
   vanish into a ``matched=True``.  The reason string of a *passing* check says
   which fields went unchecked, so a green row can never be read as more than
   it is.  Callers that need a field to be present pass ``require=(...)``.

``wall_seconds`` is deliberately **advisory, never binding** — reported and
never enforced.  This build runs four concurrent agents over one shared
~121 GB pool, so wall-clock measures contention, not an arm's capacity.
``thesis_contract.tex`` asks for matched *compute*, which ``flops`` and
``train_steps`` carry; wall seconds are neither compute nor capacity, so this
is a scoping decision rather than a divergence from the thesis.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from dataclasses import fields as _fields
from typing import Any, Mapping, Sequence

from .report import Interval, Metric, SubCheck

__all__ = [
    "Budget",
    "BINDING_FIELDS",
    "ADVISORY_FIELDS",
    "budget_of",
    "MatchVerdict",
    "check_matched",
    "matched_subcheck",
    "ArmPath",
    "ParityVerdict",
    "check_path_parity",
    "parity_subcheck",
    "VarianceConvergenceVerdict",
    "check_variance_convergence",
]

#: Budget fields that BIND a comparison: a mismatch on any one makes the
#: comparison unmatched.  ``state_width`` and ``n_configs_trained`` exist for
#: PREREG_A1_run2 §3.1 budgets B2 and B4 — total state width, and the number of
#: hyperparameter configurations trained per arm.  B4 is the budget most often
#: violated in practice and least often counted: an arm tuned over twenty
#: configurations against a control given one is not a matched comparison,
#: whatever the parameter counts say.
BINDING_FIELDS: tuple[str, ...] = (
    "n_parameters",
    "n_parameters_effective",
    "flops",
    "train_steps",
    "state_width",
    "n_configs_trained",
)

#: Reported, never enforced.  See the module docstring for why wall-clock is
#: not a capacity fact about a model on this machine.
ADVISORY_FIELDS: tuple[str, ...] = ("wall_seconds",)


@dataclass(frozen=True)
class Budget:
    n_parameters: int | None = None
    flops: float | None = None
    train_steps: int | None = None
    wall_seconds: float | None = None
    source: str = "unknown"
    #: total state width summed over regions (PREREG_A1_run2 §3.1 B2).  For a
    #: heterogeneous arm this is ``sum(D_f for each region's family)``; for a
    #: uniform arm ``N * D``.  Padding does not count — narrowing N-1 pads
    #: family state to the max family dimension and padded cells are not
    #: capacity.
    state_width: int | None = None
    #: distinct hyperparameter configurations trained and val-scored for this
    #: arm (PREREG_A1_run2 §3.1 B4).
    n_configs_trained: int | None = None
    #: trainable parameters that actually MOVED between initialisation and the
    #: scored checkpoint (PREREG_A1_run2 §3.1 B1e).  A parameter that never
    #: received a gradient is not capacity, and this project has now found three
    #: of them: `z_session` (2,616 params bit-identical to init),
    #: `eeg.source_proj` under the freeze control, and `bold.log_noise` (454
    #: values all exactly -4.0, its initialiser).  §3.2d of CLAIM_BOUNDARY
    #: already showed dead parameters making a capacity confound 5x worse than
    #: reported, so they corrupt the BUDGET, not only the model.
    n_parameters_effective: int | None = None

    @property
    def known(self) -> bool:
        return self.n_parameters is not None

    @property
    def declared(self) -> tuple[str, ...]:
        """Fields this budget actually carries a number for."""
        return tuple(
            f for f in BINDING_FIELDS + ADVISORY_FIELDS if getattr(self, f) is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_parameters": self.n_parameters,
            "flops": self.flops,
            "train_steps": self.train_steps,
            "wall_seconds": self.wall_seconds,
            "state_width": self.state_width,
            "n_configs_trained": self.n_configs_trained,
            "source": self.source,
        }


def budget_of(model: Any) -> Budget:
    """Best-effort parameter/compute budget for an arm."""
    b = getattr(model, "budget", None)
    if isinstance(b, Budget):
        return b

    n: int | None = None
    src = "unknown"
    fn = getattr(model, "n_parameters", None)
    if callable(fn):
        try:
            n = int(fn())
            src = "n_parameters()"
        except Exception:
            n = None
    if n is None and hasattr(model, "parameters"):
        try:  # torch.nn.Module
            n = int(sum(int(p.numel()) for p in model.parameters()))
            src = "torch.parameters()"
        except Exception:
            n = None
    if n is None:
        v = getattr(model, "n_params", None)
        if isinstance(v, int):
            n, src = v, "n_params attribute"

    def _num(attr: str) -> float | None:
        val = getattr(model, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                return None
        try:
            return None if val is None else float(val)
        except Exception:
            return None

    def _int(attr: str) -> int | None:  # noqa: D401
        v = _num(attr)
        return None if v is None else int(v)

    return Budget(
        n_parameters=n,
        flops=_num("compute_flops"),
        train_steps=_int("train_steps"),
        wall_seconds=_num("wall_seconds"),
        state_width=_int("state_width"),
        n_configs_trained=_int("n_configs_trained"),
        n_parameters_effective=_int("n_parameters_effective"),
        source=src,
    )


@dataclass
class MatchVerdict:
    candidate: str
    candidate_budget: Budget
    baseline_budgets: dict[str, Budget]
    tol: float
    matched: bool
    unknown: list[str] = field(default_factory=list)
    over_budget: list[str] = field(default_factory=list)
    ratios: dict[str, float] = field(default_factory=dict)
    favourable_to_null: bool = False
    #: ``{field: {baseline: candidate/baseline}}`` for every binding field both
    #: sides declared.  ``ratios`` is the ``n_parameters`` row of this.
    field_ratios: dict[str, dict[str, float]] = field(default_factory=dict)
    #: ``"baseline.field"`` entries over tolerance.
    over_budget_fields: list[str] = field(default_factory=list)
    #: ``"baseline.field"`` declared by exactly one side — an accounting
    #: mismatch, which is a defect rather than a field to skip.
    asymmetric: list[str] = field(default_factory=list)
    #: binding fields that NO arm declared.  Not a failure; named so that a
    #: passing check cannot be read as having checked them.
    unchecked_fields: list[str] = field(default_factory=list)
    #: ``"arm.field"`` for fields the caller passed in ``require``.
    undeclared_required: list[str] = field(default_factory=list)
    required: tuple[str, ...] = ()
    #: advisory ratios, reported and never enforced.
    advisory_ratios: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def unchecked_note(self) -> str:
        if not self.unchecked_fields:
            return "all binding budget fields were declared and checked"
        return (
            "NOT CHECKED (no arm declared them): "
            + ", ".join(self.unchecked_fields)
            + " — this comparison is matched on "
            + ", ".join(f for f in BINDING_FIELDS if f not in self.unchecked_fields)
            + " only"
        )

    @property
    def reason(self) -> str:
        if self.unknown:
            return (
                "capacity accounting unavailable for "
                f"{', '.join(self.unknown)}; a comparison whose budgets are unknown cannot "
                "support 'at matched compute and parameter count'"
            )
        if self.undeclared_required:
            return (
                "required budget fields not declared: "
                f"{', '.join(self.undeclared_required)}; the caller asked for these to be "
                "matched and no number was supplied, so they cannot have been"
            )
        if self.asymmetric:
            return (
                "budget accounting is asymmetric: "
                f"{', '.join(self.asymmetric)} declared on one side only. Arms whose "
                "accounting does not cover the same quantities cannot be compared 'at "
                "matched compute and parameter count'"
            )
        if self.over_budget_fields:
            worst_field, worst_key, worst_r = "", "", 0.0
            for f, row in self.field_ratios.items():
                for k, r in row.items():
                    if f"{k}.{f}" in self.over_budget_fields and r > worst_r:
                        worst_field, worst_key, worst_r = f, k, r
            return (
                f"candidate {self.candidate!r} exceeds the budget of "
                f"{', '.join(self.over_budget)} on {', '.join(self.over_budget_fields)} "
                f"(worst {worst_r:.2f}x on {worst_field} vs {worst_key!r}, "
                f"tolerance {1 + self.tol:.2f}x); an unmatched win is not a win"
            )
        return f"matched; {self.unchecked_note}"

    def metrics(self) -> list[Metric]:
        out = [
            Metric(
                name="capacity.candidate_n_parameters",
                value=float(self.candidate_budget.n_parameters or float("nan")),
                units="parameters",
                kind="capacity",
                exact=True,
            ),
            # A green matched_capacity row means nothing without this: it says
            # how many of the binding budgets were actually compared.
            Metric(
                name="capacity.binding_fields_checked",
                value=float(len(BINDING_FIELDS) - len(self.unchecked_fields)),
                units=f"of {len(BINDING_FIELDS)}",
                kind="capacity",
                exact=True,
                direction="greater_is_better",
                note=self.unchecked_note,
            ),
        ]
        for f in BINDING_FIELDS:
            for k, r in sorted(self.field_ratios.get(f, {}).items()):
                out.append(
                    Metric(
                        name=(
                            f"capacity.param_ratio_vs_{k}"
                            if f == "n_parameters"
                            else f"capacity.{f}_ratio_vs_{k}"
                        ),
                        value=float(r),
                        kind="capacity",
                        exact=True,
                        threshold=1.0 + self.tol,
                        direction="less_is_better",
                        note=f"candidate {f} / baseline {f}",
                    )
                )
        for f, row in sorted(self.advisory_ratios.items()):
            for k, r in sorted(row.items()):
                out.append(
                    Metric(
                        name=f"capacity.{f}_ratio_vs_{k}",
                        value=float(r),
                        kind="capacity",
                        exact=True,
                        direction="less_is_better",
                        note=f"ADVISORY, never enforced: candidate {f} / baseline {f}",
                    )
                )
        return out


def check_matched(
    candidate: Any,
    baselines: Mapping[str, Any],
    *,
    tol: float = 0.10,
    candidate_name: str | None = None,
    require: Sequence[str] = (),
) -> MatchVerdict:
    """Compare candidate capacity against every baseline, on every declared field.

    ``require`` names binding fields that every arm **must** declare; a field
    in ``require`` that no arm carries makes the verdict unmatched rather than
    unchecked.  Use it where a preregistration commits to a budget — A1's run 2
    requires ``state_width``, ``train_steps`` and ``n_configs_trained``.
    """
    bad = [f for f in require if f not in BINDING_FIELDS]
    if bad:
        raise ValueError(
            f"require names non-binding budget field(s) {bad}; binding fields are "
            f"{list(BINDING_FIELDS)} and {list(ADVISORY_FIELDS)} is advisory by design"
        )
    cname = candidate_name or getattr(candidate, "name", "candidate")
    cb = budget_of(candidate)
    bbs = {k: budget_of(v) for k, v in baselines.items()}
    unknown: list[str] = []
    if not cb.known:
        unknown.append(cname)
    unknown += [k for k, b in bbs.items() if not b.known]

    field_ratios: dict[str, dict[str, float]] = {}
    advisory_ratios: dict[str, dict[str, float]] = {}
    over_fields: list[str] = []
    asymmetric: list[str] = []
    unchecked: list[str] = []

    for f in BINDING_FIELDS + ADVISORY_FIELDS:
        cv = getattr(cb, f)
        row: dict[str, float] = {}
        any_declared = cv is not None
        for k, b in bbs.items():
            bv = getattr(b, f)
            any_declared = any_declared or bv is not None
            if cv is None and bv is None:
                continue
            if (cv is None) != (bv is None):
                # Declared on one side only.  Skipping this is how a
                # compute-unmatched comparison used to read as matched.
                if f not in ADVISORY_FIELDS:
                    asymmetric.append(f"{k}.{f}")
                continue
            if float(bv) == 0.0:  # type: ignore[arg-type]
                continue
            r = float(cv) / float(bv)  # type: ignore[arg-type]
            row[k] = r
            if f not in ADVISORY_FIELDS and r > 1.0 + tol:
                over_fields.append(f"{k}.{f}")
        if f in ADVISORY_FIELDS:
            if row:
                advisory_ratios[f] = row
            continue
        if row:
            field_ratios[f] = row
        if not any_declared:
            unchecked.append(f)

    undeclared_required: list[str] = []
    for f in require:
        if getattr(cb, f) is None:
            undeclared_required.append(f"{cname}.{f}")
        undeclared_required += [f"{k}.{f}" for k, b in bbs.items() if getattr(b, f) is None]

    ratios = dict(field_ratios.get("n_parameters", {}))
    over = sorted({e.rsplit(".", 1)[0] for e in over_fields})
    favourable = bool(ratios) and all(r < 1.0 - tol for r in ratios.values())
    return MatchVerdict(
        candidate=cname,
        candidate_budget=cb,
        baseline_budgets=bbs,
        tol=float(tol),
        matched=(
            not unknown and not over_fields and not asymmetric and not undeclared_required
        ),
        unknown=unknown,
        over_budget=over,
        ratios=ratios,
        favourable_to_null=favourable,
        field_ratios=field_ratios,
        over_budget_fields=sorted(over_fields),
        asymmetric=sorted(asymmetric),
        unchecked_fields=unchecked,
        undeclared_required=sorted(set(undeclared_required)),
        required=tuple(require),
        advisory_ratios=advisory_ratios,
    )


def matched_subcheck(verdict: MatchVerdict, *, name: str = "matched_capacity") -> SubCheck:
    """Mandatory sub-check: an unmatched comparison cannot support a claim."""
    if verdict.matched:
        note = (
            "candidate is strictly smaller than every baseline (favourable to the null)"
            if verdict.favourable_to_null
            else "within tolerance"
        )
        # A passing row must carry what it did NOT check, or it reads as more
        # than it is.  This is the whole difference between a guard and a
        # decoration.
        note = f"{note}; {verdict.unchecked_note}"
        return SubCheck(
            name=name,
            description=(
                "Parameter/compute matching against every baseline, on every binding "
                "budget field either side declares (tab:claim-gates: 'at matched compute "
                "and parameter count')."
            ),
            metrics=verdict.metrics(),
            mandatory=True,
            reason=note,
            falsified_by="candidate needs more capacity than the baselines to win",
        )
    return SubCheck(
        name=name,
        description="Parameter/compute matching against every baseline.",
        metrics=verdict.metrics() if verdict.candidate_budget.known else [],
        mandatory=True,
        forced_status="COULD_NOT_RUN",
        reason=verdict.reason,
        falsified_by="candidate needs more capacity than the baselines to win",
    )


# ---------------------------------------------------------------------------
# PATH PARITY: the second matching axis.
#
# Budgets guard the MODEL.  Nothing guarded the path from the model to the
# number -- and every between-arm defect this project has found so far lives on
# that path, not in the budgets:
#
#   * 🌊 Hodgkin, 2026-08-06, caught before shipping: the A1 treatment arm's
#     EEGHead received a shared interface view exporting ("rate_e","rate_i") = 2
#     dims against the control arm's ("rate_e","rate_i","spectral") = 18.  Every
#     field `Budget` declares could have matched exactly.  A1 would have
#     concluded that heterogeneous regional state does not help, with a green
#     harness, because the treatment arm was handicapped at the OBSERVATION
#     BOUNDARY rather than at the hypothesis.
#   * run 1's §11.2 comparison: five baselines received a held-out residual
#     variance calibration and SC-WBD-001-beta received none
#     (CLAIM_BOUNDARY.md §3.5.5).  Same shape, on the SCORE rather than the
#     interface.
#   * `subject_specific_ar` under a participant-disjoint split: every test
#     window routed to the pooled fallback, making a "subject-specific"
#     baseline arithmetically identical to `ar16` (§3.5.6a).  Same shape, on
#     the SPLIT.
#
# The general statement, which is the transferable part: **a between-arm
# comparison is interpretable only if everything except the manipulated
# variable is identical, and "everything" includes the whole path from state to
# scalar.**  Capacity is one term on that path.  These are the others.
#
# Two deliberate differences from :func:`check_matched`:
#
# 1. **Equality, not a budget.**  Budgets are one-sided -- a candidate winning
#    with FEWER parameters has produced evidence.  Interface width has no such
#    direction: a narrower candidate that WINS is strengthened, a narrower
#    candidate that LOSES is confounded.  Since the winner is unknown when the
#    rule is fixed, the only preregisterable rule is exact equality.
# 2. **Undeclared blocks.**  An unchecked budget field passes and is named
#    (legacy: nothing has ever declared them).  Path parity is new and has no
#    legacy, so an undeclared field is COULD_NOT_RUN: parity that was not
#    verified is not parity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPath:
    """Everything between an arm's manipulated variable and its scalar score.

    Every field must be identical across arms.  Fields are compared
    **generically**, so adding one here automatically extends the check rather
    than requiring the comparison loop to be remembered and updated.
    """

    #: ordered ``(port_name, width)`` each head actually reads, per head.
    #: Ordered and named, not just totalled: two arms can present 18 dims each
    #: and still export different quantities.
    observation_ports: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] | None = None
    #: how the predictive variance is produced, e.g.
    #: ``"per_channel_scalar_broadcast"`` vs ``"state_dependent_logvar"``.
    variance_model: str | None = None
    #: post-hoc variance calibration, e.g. ``"as_emitted"`` or
    #: ``"held_out_training_windows"``.  Run 1's §11.2 defect is exactly a
    #: difference here.
    calibration_protocol: str | None = None
    #: metric identity INCLUDING units, e.g. ``"gaussian_nll_raw_units"``.
    score_metric: str | None = None
    #: fingerprint of the scored split.  Different splits, different populations.
    split_fingerprint: str | None = None
    #: context length fed to the arm.
    context_length: int | None = None
    #: input normalisation applied before the arm sees the data.
    input_normalisation: str | None = None
    #: anatomy artifact provenance consumed by the arm.
    anatomy_provenance: str | None = None

    @property
    def undeclared(self) -> tuple[str, ...]:
        return tuple(f.name for f in _fields(self) if getattr(self, f.name) is None)


@dataclass
class ParityVerdict:
    candidate: str
    paths: dict[str, ArmPath]
    matched: bool
    #: ``"field: candidate=<x> vs <arm>=<y>"``
    mismatches: list[str] = field(default_factory=list)
    #: ``"arm.field"`` not declared by some arm
    undeclared: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.undeclared:
            return (
                "path parity not verifiable: "
                f"{', '.join(self.undeclared)} undeclared. Parity that was not checked is "
                "not parity, and an arm handicapped between its state and its score is "
                "indistinguishable from an arm whose hypothesis is wrong"
            )
        if self.mismatches:
            return (
                "arms differ OFF the manipulated variable: "
                f"{'; '.join(self.mismatches)}. The comparison would attribute this "
                "difference to the hypothesis"
            )
        return "every arm presents the same path from state to score"

    def metrics(self) -> list[Metric]:
        total = len(_fields(ArmPath))
        return [
            Metric(
                name="path_parity.fields_verified",
                value=float(total - len({u.split(".", 1)[1] for u in self.undeclared})),
                units=f"of {total}",
                kind="capacity",
                exact=True,
                direction="greater_is_better",
                note=self.reason,
            ),
            Metric(
                name="path_parity.mismatches",
                value=float(len(self.mismatches)),
                kind="capacity",
                exact=True,
                threshold=0.0,
                # Parity is an EQUALITY constraint, not a budget: `two_sided`
                # at zero is "exactly none", where `less_is_better` at zero
                # could never pass (the comparison is strict).
                direction="two_sided",
                note="; ".join(self.mismatches) or "none",
            ),
        ]


def check_path_parity(
    paths: Mapping[str, ArmPath], *, candidate: str
) -> ParityVerdict:
    """Every arm must present the same path from state to scalar score."""
    if candidate not in paths:
        raise ValueError(f"candidate {candidate!r} has no ArmPath; got {sorted(paths)}")
    undeclared = sorted(
        f"{arm}.{f}" for arm, p in paths.items() for f in p.undeclared
    )
    ref = paths[candidate]
    mismatches: list[str] = []
    for f in _fields(ArmPath):
        cv = getattr(ref, f.name)
        if cv is None:
            continue
        for arm, p in paths.items():
            if arm == candidate:
                continue
            av = getattr(p, f.name)
            if av is not None and av != cv:
                mismatches.append(f"{f.name}: {candidate}={cv!r} vs {arm}={av!r}")
    return ParityVerdict(
        candidate=candidate,
        paths=dict(paths),
        matched=not undeclared and not mismatches,
        mismatches=mismatches,
        undeclared=undeclared,
    )


def parity_subcheck(verdict: ParityVerdict, *, name: str = "path_parity") -> SubCheck:
    """Mandatory sub-check: an arm handicapped off-hypothesis cannot be compared."""
    common = dict(
        name=name,
        description=(
            "Every arm presents the same observation interface, variance model, "
            "calibration protocol, metric, split, context and normalisation. Capacity "
            "matching guards the model; this guards the path from the model to the number."
        ),
        metrics=verdict.metrics(),
        mandatory=True,
        falsified_by=(
            "an arm is handicapped or advantaged somewhere other than the manipulated "
            "variable, and the comparison attributes it to the hypothesis"
        ),
    )
    if verdict.matched:
        return SubCheck(**common, reason=verdict.reason)
    return SubCheck(**common, forced_status="COULD_NOT_RUN", reason=verdict.reason)


# ---------------------------------------------------------------------------
# VARIANCE-SCALE CONVERGENCE: a stage-5 parity failure that ArmPath cannot see.
#
# 🔥 Turing's P0 result, 2026-08-06: run 1's variance defect is NOT structural.
# `eeg.log_noise` is trainable in stage V only (`train.py:78`), stage V ran 900
# steps in 134 seconds, and SGD reached 19.8% of a CLOSED-FORM optimum -- the
# fitted scalar asserts variance exp(0.2732)=1.3142 against a held-out residual
# variance of 3.9697, uniformly overconfident by 3.02x.
#
# The consequence for A1 is the part that is easy to miss.  `ArmPath` compares
# DECLARED configuration, so two arms can both declare `state_dependent_logvar`
# -- passing path parity -- while one arm's variance scale is 20% converged and
# the other's is 90%.  That is an unmatched **stage 5** hiding inside a matched
# stage 5, and the difference would be attributed to the hypothesis.
#
# It is cheap to close precisely because the optimum is closed-form: for a
# Gaussian score the best global log-variance is `log(MSE)` on the same held-out
# data.  So convergence is VERIFIABLE, not hoped for.
# ---------------------------------------------------------------------------


@dataclass
class VarianceConvergenceVerdict:
    tol: float
    matched: bool
    #: ``{arm: fitted_mean_log_variance - log(mse)}``; 0 is converged, negative
    #: is overconfident, positive is underconfident.
    gaps: dict[str, float] = field(default_factory=dict)
    unconverged: list[str] = field(default_factory=list)
    #: worst pairwise difference between arms' gaps
    spread: float = 0.0

    @property
    def reason(self) -> str:
        if self.unconverged:
            worst = max(self.gaps.items(), key=lambda kv: abs(kv[1]))
            return (
                f"variance scale not converged for {', '.join(self.unconverged)} "
                f"(worst {worst[0]}: log-variance off its closed-form optimum by "
                f"{worst[1]:+.4f}, i.e. {math.exp(-worst[1]):.2f}x mis-scaled; tolerance "
                f"{self.tol}). An arm whose variance scale did not converge is not "
                "reporting a measurement of its predictive quality"
            )
        if self.spread > self.tol:
            return (
                f"arms' variance scales converged to DIFFERENT degrees (spread "
                f"{self.spread:.4f} > {self.tol}); the difference would be attributed to "
                "the hypothesis"
            )
        return f"every arm's variance scale is within {self.tol} of its closed-form optimum"

    def metrics(self) -> list[Metric]:
        out = [
            Metric(
                name=f"variance_convergence.gap.{arm}",
                value=float(g),
                units="log-variance",
                kind="calibration",
                exact=True,
                threshold=self.tol,
                direction="two_sided",
                note="fitted mean log-variance minus log(held-out MSE); 0 is converged",
            )
            for arm, g in sorted(self.gaps.items())
        ]
        out.append(
            Metric(
                name="variance_convergence.between_arm_spread",
                value=float(self.spread),
                units="log-variance",
                kind="calibration",
                exact=True,
                threshold=self.tol,
                direction="two_sided",
            )
        )
        return out


def check_variance_convergence(
    arms: Mapping[str, tuple[float, float]], *, tol: float = 0.1
) -> VarianceConvergenceVerdict:
    """``arms`` maps arm name to ``(fitted_mean_log_variance, heldout_mse)``.

    The closed-form optimum for a single global predictive variance is
    ``log(MSE)``.  An arm far from it has a fitting failure, not a modelling
    result, and its NLL is not a measurement of its predictive quality.
    """
    gaps = {a: float(lv) - math.log(float(mse)) for a, (lv, mse) in arms.items()}
    unconverged = sorted(a for a, g in gaps.items() if abs(g) > tol)
    spread = (max(gaps.values()) - min(gaps.values())) if len(gaps) > 1 else 0.0
    return VarianceConvergenceVerdict(
        tol=float(tol),
        matched=not unconverged and spread <= tol,
        gaps=gaps,
        unconverged=unconverged,
        spread=float(spread),
    )
