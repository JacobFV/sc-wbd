"""The posterior sweep must measure what the trainer and the evaluator do.

``scripts/sweep_posterior.py`` exists to decide, in ~90 minutes, whether a 38-hour
run is worth launching -- specifically whether ISSUE-012's remedies move
``posterior_r2`` on ``log_G`` past 0.4. A sweep that trains or scores under
conditions the run does not use answers a different question confidently, which
is worse than answering none.

It already happened once. The first version evaluated with ``n_samples=128`` and
applied the training slice mask at evaluation; production
(``evaluate.posterior_calibration``) uses 256 bins and an unmasked context.
Reproducing run 3's exact setting returned ``sbc_ks_pvalue_min`` 0.000 where
``reports/training/evaluation_run3.json`` records 0.0976 -- R^2 agreed to 0.01
and calibration did not, because the calibration column was measuring the
harness. These tests pin the three places the two can drift apart.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "scripts/sweep_posterior.py"

pytestmark = pytest.mark.skipif(not SWEEP.is_file(), reason="sweep script absent")


def _sweep_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sweep_under_test", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_sweep_slice_mask_is_the_trainers_slice_mask() -> None:
    """Same observed-subgraph rule, asserted on behaviour rather than on source.

    The sweep duplicates ``FoundationTrainer._slice_mask`` rather than importing
    it, because importing would mean constructing the 26.3M-parameter model and
    the anatomy prior to get a Bernoulli mask. Duplication is the right call and
    it is exactly the kind that rots, so the two are compared by running them.
    """
    from scwbd.foundation.train import FoundationTrainer

    sweep = _sweep_module()
    B, N, p = 64, 414, 0.65

    torch.manual_seed(0)
    mine = sweep._slice_mask(B, N, "cpu", p_observed=p)
    torch.manual_seed(0)
    theirs = FoundationTrainer._slice_mask.__wrapped__(  # type: ignore[attr-defined]
        None, B, N, p_observed=p
    ) if hasattr(FoundationTrainer._slice_mask, "__wrapped__") else None

    if theirs is None:
        # `_slice_mask` reads only `self.device`; bind a stand-in rather than
        # building a trainer.
        class _Stub:
            device = "cpu"

        torch.manual_seed(0)
        theirs = FoundationTrainer._slice_mask(_Stub(), B, N, p_observed=p)

    assert torch.equal(mine, theirs), (
        "the sweep's slice mask has drifted from FoundationTrainer._slice_mask. "
        "The sweep would then train the posterior on a different observed subgraph "
        "than the run does, and its R^2 would not predict the run's."
    )
    assert (mine.sum(1) > 0).all(), "every episode must observe something"


def test_the_sweep_scores_calibration_the_way_production_does() -> None:
    """256 SBC bins, matching ``evaluate.posterior_calibration``'s default.

    Pinned on both sides: if either moves, this fails and names which.
    """
    from scwbd.foundation.evaluate import posterior_calibration

    sweep = _sweep_module()
    prod = inspect.signature(posterior_calibration).parameters["n_samples"].default
    assert sweep.PRODUCTION_SBC_BINS == prod, (
        f"the sweep scores SBC with {sweep.PRODUCTION_SBC_BINS} bins and "
        f"evaluate.posterior_calibration uses {prod}. The KS p-value depends on the "
        "bin count, so the sweep's calibration column would not be comparable to the "
        "number a run publishes -- which is the comparison the sweep exists to make."
    )


def test_the_sweep_evaluates_on_an_unmasked_context() -> None:
    """Training masks the context; production evaluation does not.

    ``sim_losses`` shows the posterior ``ctx_y * ctx_mask``;
    ``posterior_calibration`` passes ``b["activity"][:, :context]`` with no mask.
    The sweep must do BOTH, in the right places, or its numbers describe a
    regime the run never enters.
    """
    sweep = _sweep_module()
    x = torch.ones(8, 24, 414)

    torch.manual_seed(0)
    masked = sweep._context(x, _Cfg(), "cpu", mask=True)
    unmasked = sweep._context(x, _Cfg(), "cpu", mask=False)

    assert torch.equal(unmasked, x[:, :24]), "mask=False must pass the context through"
    assert not torch.equal(masked, unmasked), "mask=True must actually mask"
    frac = float((masked != 0).float().mean())
    assert 0.55 < frac < 0.75, f"observed fraction {frac:.2f} is not the 0.65 the trainer uses"

    # And the calls in the script are the right way round: train masked, eval not.
    tree = ast.parse(SWEEP.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_context"
    ]
    assert calls, "no _context calls found; this guard has stopped guarding"
    kinds = {
        next((k.value.value for k in c.keywords if k.arg == "mask"), None) for c in calls
    }
    assert kinds == {True, False}, (
        f"_context is called with mask in {kinds}; the sweep must train on the masked "
        "context and evaluate on the unmasked one, as the trainer and the evaluator do."
    )


class _Cfg:
    class data:
        context = 24


def test_the_discharge_condition_is_the_one_the_issue_states() -> None:
    """R^2 > 0.4 AND calibration surviving. Either alone is not the condition.

    ISSUE-012: "Recovering calibration by widening back to the prior does not
    discharge it." The converse binds equally -- an informative posterior that
    has lost its calibration has not discharged it either, and the sweep must not
    be able to report a winner on R^2 alone.
    """
    sweep = _sweep_module()
    assert sweep.DISCHARGE["log_G_r2"] == 0.4
    assert sweep.DISCHARGE["sbc_ks_pvalue_min"] == 0.01
    assert sweep.DISCHARGE["coverage_mae"] == 0.05

    informative_but_uncalibrated = {
        "log_G_r2": 0.80,
        "sbc_ks_pvalue_min": 0.0,
        "coverage_mae": 0.039,
    }
    assert not sweep.discharges(informative_but_uncalibrated), (
        "a cell with R^2 0.80 and a failed SBC was reported as discharging ISSUE-012. "
        "The issue requires the calibration to survive the recovery."
    )
    calibrated_but_uninformative = {
        "log_G_r2": 0.001,
        "sbc_ks_pvalue_min": 0.098,
        "coverage_mae": 0.021,
    }
    assert not sweep.discharges(calibrated_but_uninformative), (
        "run 3's own numbers were reported as discharging ISSUE-012."
    )
    assert sweep.discharges({"log_G_r2": 0.62, "sbc_ks_pvalue_min": 0.05, "coverage_mae": 0.03})
