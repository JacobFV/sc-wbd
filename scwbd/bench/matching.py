"""Matched capacity and compute accounting (agent J).

``thesis_contract.tex`` Table ``tab:claim-gates`` requires the fusion claim to
be demonstrated "at matched compute and parameter count", and Appendix D
requires the connectome controls "at matched parameter/compute budgets".

This module makes that mandatory rather than rhetorical:

* :func:`budget_of` extracts a parameter/compute budget from an arm by duck
  typing (``n_parameters()``, ``parameters()`` for torch modules, an explicit
  ``budget`` attribute);
* :func:`check_matched` compares the candidate against every baseline and
  returns a verdict;
* :func:`matched_subcheck` turns that verdict into a **mandatory** sub-check
  which is ``COULD_NOT_RUN`` when the candidate is over budget or when the
  budgets are unknown.  It is never a ``PASS``.

The asymmetry is deliberate.  A candidate that wins with *fewer* parameters
than its baselines has produced evidence; a candidate that wins with more has
produced a bigger model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .report import Interval, Metric, SubCheck

__all__ = [
    "Budget",
    "budget_of",
    "MatchVerdict",
    "check_matched",
    "matched_subcheck",
]


@dataclass(frozen=True)
class Budget:
    n_parameters: int | None = None
    flops: float | None = None
    train_steps: int | None = None
    wall_seconds: float | None = None
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return self.n_parameters is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_parameters": self.n_parameters,
            "flops": self.flops,
            "train_steps": self.train_steps,
            "wall_seconds": self.wall_seconds,
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

    steps = _num("train_steps")
    return Budget(
        n_parameters=n,
        flops=_num("compute_flops"),
        train_steps=None if steps is None else int(steps),
        wall_seconds=_num("wall_seconds"),
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

    @property
    def reason(self) -> str:
        if self.unknown:
            return (
                "capacity accounting unavailable for "
                f"{', '.join(self.unknown)}; a comparison whose budgets are unknown cannot "
                "support 'at matched compute and parameter count'"
            )
        if self.over_budget:
            worst = max(self.ratios.items(), key=lambda kv: kv[1])
            return (
                f"candidate {self.candidate!r} exceeds the budget of "
                f"{', '.join(self.over_budget)} (worst ratio {worst[1]:.2f}x vs {worst[0]!r}, "
                f"tolerance {1 + self.tol:.2f}x); an unmatched win is not a win"
            )
        return "matched"

    def metrics(self) -> list[Metric]:
        out = [
            Metric(
                name="capacity.candidate_n_parameters",
                value=float(self.candidate_budget.n_parameters or float("nan")),
                units="parameters",
                kind="capacity",
                exact=True,
            )
        ]
        for k, r in sorted(self.ratios.items()):
            out.append(
                Metric(
                    name=f"capacity.param_ratio_vs_{k}",
                    value=float(r),
                    kind="capacity",
                    exact=True,
                    threshold=1.0 + self.tol,
                    direction="less_is_better",
                    note="candidate params / baseline params",
                )
            )
        return out


def check_matched(
    candidate: Any,
    baselines: Mapping[str, Any],
    *,
    tol: float = 0.10,
    candidate_name: str | None = None,
) -> MatchVerdict:
    """Compare candidate capacity against every baseline."""
    cname = candidate_name or getattr(candidate, "name", "candidate")
    cb = budget_of(candidate)
    bbs = {k: budget_of(v) for k, v in baselines.items()}
    unknown: list[str] = []
    if not cb.known:
        unknown.append(cname)
    unknown += [k for k, b in bbs.items() if not b.known]
    ratios: dict[str, float] = {}
    over: list[str] = []
    if cb.known:
        for k, b in bbs.items():
            if not b.known or b.n_parameters == 0:
                continue
            r = float(cb.n_parameters) / float(b.n_parameters)  # type: ignore[arg-type]
            ratios[k] = r
            if r > 1.0 + tol:
                over.append(k)
    favourable = bool(ratios) and all(r < 1.0 - tol for r in ratios.values())
    return MatchVerdict(
        candidate=cname,
        candidate_budget=cb,
        baseline_budgets=bbs,
        tol=float(tol),
        matched=(not unknown and not over),
        unknown=unknown,
        over_budget=over,
        ratios=ratios,
        favourable_to_null=favourable,
    )


def matched_subcheck(verdict: MatchVerdict, *, name: str = "matched_capacity") -> SubCheck:
    """Mandatory sub-check: an unmatched comparison cannot support a claim."""
    if verdict.matched:
        note = (
            "candidate is strictly smaller than every baseline (favourable to the null)"
            if verdict.favourable_to_null
            else "within tolerance"
        )
        return SubCheck(
            name=name,
            description=(
                "Parameter/compute matching against every baseline "
                "(tab:claim-gates: 'at matched compute and parameter count')."
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
