"""E9 -- ``_scwbd_scores`` conditions on one posterior draw, not on the posterior.

``evaluate._scwbd_scores`` line 65::

    th = trainer.posterior.sample(ctx_e, 1)[:, 0][:, : len(THETA_NAMES)]

The quantity the report claims is a held-out predictive log-score,
``-log p(y_target | y_context)``.  With an amortized posterior ``q(theta | y_ctx)``
that is

    -log E_{theta ~ q} p(y_target | theta, y_ctx)

and a single draw estimates it with ``K = 1``.  By Jensen's inequality the
Monte-Carlo estimator of ``-log (1/K) sum_k p_k`` is biased **upward** for finite
``K``, so the single draw always makes SC-WBD look worse than it is -- the
opposite direction to the units defect, and equally invalid.

The bias is not the strongest objection.  The stronger one is *reproducibility*:
the reported headline number carries a run-to-run standard deviation from the
draw alone.  Measured on this checkpoint (see reports/evaluation_audit.md) that
sd is 0.0075 nats, while the gap between ``ar16`` and ``var4`` on the same fold
is 0.0016-0.0053 nats.  A statistic whose seed-noise exceeds the differences it
is used to rank cannot rank them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_scwbd_score_is_a_predictive_not_a_single_draw():
    """E9 verdict test."""
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate._scwbd_scores)
    single_draw = ".sample(ctx_e, 1)" in src
    assert not single_draw, (
        "_scwbd_scores conditions the rollout on ONE posterior draw. The reported "
        "quantity is then -log p(y | theta_1), not -log E_q p(y | theta), so it "
        "(a) is biased upward by Jensen, penalising SC-WBD, and (b) is not "
        "reproducible: re-running the same checkpoint on the same windows moves "
        "the headline number by more than the gap between the two closest "
        "baselines. Use a marginalisation over K draws (log-mean-exp of the "
        "per-element likelihood); the posterior mean is a weaker second choice "
        "because it reports a plug-in score, not a predictive."
    )


def test_scwbd_score_is_deterministic_given_the_checkpoint():
    """A headline number must not depend on an unseeded draw.

    ``evaluate.evaluate_model`` never calls ``set_determinism``, so the draw is
    governed by whatever RNG state the process happens to hold. Two invocations
    of ``evaluate.main`` on one checkpoint are not required to agree.
    """
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate)
    seeds_the_scoring = (
        "set_determinism" in inspect.getsource(evaluate.evaluate_model)
        or "generator=" in inspect.getsource(evaluate._scwbd_scores)
    )
    assert seeds_the_scoring, (
        "evaluate_model seeds nothing before scoring, and _scwbd_scores draws "
        "theta from the global RNG. set_determinism is imported by this module "
        "and used only inside source_ablation."
    )


def test_jensen_direction_of_a_single_draw_estimator():
    """The sign of the bias, as arithmetic -- so the ruling is checkable.

    E[-log p_k] >= -log E[p_k]: a one-sample estimate of a predictive log-score
    is never optimistic.  This must always pass; it is the derivation, not the
    defect.
    """
    rng = np.random.default_rng(0)
    # per-draw likelihoods with the spread a wide posterior produces
    p = np.exp(rng.normal(-2.0, 0.6, size=(20000,)))
    single = float(np.mean(-np.log(p)))
    marginal = float(-np.log(p.mean()))
    assert single > marginal, (single, marginal)
