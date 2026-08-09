"""E13-E15 -- audit of 🔥 Turing's four patches (`2e70ecd`..`a385c7a`, `wt/turing`).

These assert invariants the evaluation path must satisfy in **either** version, so
they are meaningful run from `master` (pre-patch) or `wt/turing` (post-patch).
`checkpoint.py` and `baselines.py` are byte-identical across the two
(`git diff master HEAD -- ...` is empty), so E14 applies to both unchanged.

E13 — the estimator class must match between SC-WBD and the baselines. Patch 3
      fixed the *units*; patch 4 then broke the *kind*, in the opposite direction.
E14 — patch 2's guard reads `load_report`, which `load_checkpoint` populates for
      the model only. Posterior mismatches remain silent.
E15 — no evaluation path loads or applies the individualizer, so a Stage-V
      checkpoint is scored as the population model.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch


# ------------------------------------------------------------------ E13
def test_scwbd_and_baselines_use_the_same_kind_of_estimator():
    """Both sides must be plug-in, or both marginal. Not one of each.

    Every baseline is a **plug-in** score: ``ARBaseline`` predicts from
    point-estimated coefficients and supplies a calibrated predictive variance;
    it does not integrate over coefficient uncertainty. Marginalising SC-WBD
    over ``theta`` while the baselines stay plug-in reintroduces exactly the
    like-for-like violation the units patch removed -- this time favouring
    SC-WBD.

    Measured on the live checkpoint over 54 participant-balanced test windows:
    the K=64 marginal is 2.4616 against a posterior-mean plug-in of 2.4993, a
    gift of 0.0377 nats -- 7x the ar16<->var4 gap of 0.0053 and larger than the
    entire 0.035-nat spread of the non-trivial baselines.

    **This test used to assert the defect rather than the fix.** It grepped
    ``_scwbd_scores`` for ``logsumexp`` and failed on any occurrence. That WAS
    the right proxy while the marginal was the headline, and it was repaired:
    the headline is now the plug-in at the posterior mean (the function's own
    docstring leads with it, and the return dict labels it in ``estimator``),
    while the marginal is retained under the distinct key
    ``nll_per_window_marginal`` with its ESS and K/2->K drift diagnostics. The
    grep was never updated, so it kept reporting a repaired defect as live --
    the marginal's continued existence as a labelled secondary is the fix, not
    the defect.

    It now checks which key is the headline. A red means the marginal has been
    promoted back into the column that is compared against a baseline.
    """
    from scwbd.foundation import baselines, evaluate

    base_src = inspect.getsource(baselines._LinearForecaster)
    assert "logsumexp" not in base_src, (
        "a baseline started marginalising; re-derive this test before trusting it"
    )

    src = inspect.getsource(evaluate._scwbd_scores)
    # The headline column is the one `real_eeg_holdout` hands to bootstrap_ci and
    # ranks against the baselines.
    holdout = inspect.getsource(evaluate.real_eeg_holdout)
    assert 'scw["nll_per_window"]' in holdout, (
        "real_eeg_holdout no longer ranks on scw['nll_per_window']; find the "
        "column it now compares against the baselines and re-point this test"
    )

    # `nll_per_window` must be built from the plug-in pass, and the marginal must
    # stay in its own key. `logsumexp` is legitimate here -- it is how the
    # SECONDARY is computed -- so the check is where its result lands.
    marginal_sinks = [
        ln.strip()
        for ln in src.splitlines()
        if "logsumexp" in ln or "nll_per_window_marginal" in ln
    ]
    assert marginal_sinks, (
        "the marginal and its diagnostics have disappeared from _scwbd_scores. "
        "It is a legitimate secondary and its removal is not obviously wrong, "
        "but this test can no longer tell a plug-in headline from a marginal one"
    )
    assert 'out["nll_per_window_marginal"]' in src, (
        "_scwbd_scores computes a logsumexp over theta draws but does not write "
        "it to the separately labelled nll_per_window_marginal key. Both sides "
        "of a comparison must be the same kind of estimator: the headline "
        "nll_per_window must be the plug-in at the posterior mean, and the "
        "marginal must be reported beside it under its own name with K, the "
        "ESS and the K/2->K drift."
    )
    assert "plug-in" in src and "estimator" in src, (
        "_scwbd_scores no longer labels its headline estimator. A number "
        "compared against a plug-in baseline must say that it is one."
    )


def test_posterior_marginal_has_a_usable_effective_sample_size(
    cfg, compiled_checkpoint, real_eeg, real_split
):
    """A marginal whose ESS is 1 is a best-of-K draw wearing a predictive's name.

    ``-log E_q[p(y|theta)]`` over a 3072-element joint likelihood with a diffuse
    proposal concentrates essentially all mass on one draw. Measured on the live
    checkpoint: **median ESS 1.049 of 64 draws**, 89% of windows below 2, the
    single best draw holding **97.6%** of the mass -- and the estimate still
    drifting -0.0036 nats between K=32 and K=64, which is 68% of the gap that
    decides a rank. The reported number is a function of K.
    """
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.model import SCWBD
    from scwbd.foundation.posterior import AmortizedPosterior
    from scwbd.foundation.simulate import THETA_NAMES, ThetaPrior
    from scwbd.foundation.train import SensorToParcel

    path, payload = compiled_checkpoint
    d = cfg.data
    from scwbd.runtime.predict import rebuild_anatomy
    anat = rebuild_anatomy((payload.get("extra") or {}).get("anatomy") or {}, device="cpu")
    model = SCWBD(cfg.model, anat)
    miss, unexp = model.load_state_dict(
        {k.replace("._orig_mod.", "."): v for k, v in payload["model"].items()}, strict=False
    )
    assert not miss and not unexp, "prefix reconciliation failed; fix before reading ESS"
    post = AmortizedPosterior(
        cfg.posterior, len(THETA_NAMES), prior=ThetaPrior(), fs=d.fs_hz,
        nuisance_dim=cfg.posterior.nuisance_dim,
    )
    post.load_state_dict(payload["posterior"])
    model.eval(), post.eval()
    s2p = SensorToParcel(model.eeg.L)

    # REAL held-out windows. An earlier version of this test used torch.randn and
    # passed -- white noise gives the posterior a flat likelihood landscape and an
    # ESS near K, so the test read clean on data the evaluation never sees. That is
    # the decorative-guard failure this directory exists to catch, committed by its
    # own author; the fixture is the defect's habitat, not a convenience.
    K, c = 8, d.context
    idx = np.asarray(real_split["test"])
    subs = np.asarray(real_eeg.window_subjects)[idx]
    keep = [int(idx[subs == s][0]) for s in sorted(set(subs))[:8]]
    eeg = torch.stack([torch.as_tensor(real_eeg[i]["eeg"]) for i in keep])
    torch.manual_seed(0)
    ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
    src = s2p(ctx_e)
    src = src / src.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        th_all = post.sample(ctx_e, K)
        L = []
        for k in range(K):
            roll = model.rollout(
                y_context=src, theta=th_all[:, k][:, : len(THETA_NAMES)],
                n_steps=tgt_e.shape[1], enforce_r05=False,
            )
            mu, lv = model.eeg(roll.state)
            v = lv.float().clamp(-14, 14)
            nll = 0.5 * (math.log(2 * math.pi) + v + (tgt_e - mu.float()) ** 2 * torch.exp(-v))
            L.append(-nll.sum(dim=(1, 2)))
    t = torch.stack(L, dim=1)
    ess = torch.exp(2 * torch.logsumexp(t, 1) - torch.logsumexp(2 * t, 1))
    med = float(ess.median())
    assert med >= K / 4, (
        f"median effective sample size {med:.2f} of {K} draws. The 'marginal' is "
        f"the best single draw; increasing K buys ~log K, so the estimate never "
        f"converges at any affordable K and the reported number depends on the K "
        f"that was chosen. Report the plug-in at the posterior mean instead, or "
        f"publish K, the ESS and the K/2->K drift beside every marginal."
    )


# ------------------------------------------------------------------ E14
def test_load_report_covers_the_posterior_not_only_the_model(compiled_checkpoint):
    """Patch 2 reads ``load_report``; ``load_checkpoint`` writes it for the model only."""
    from scwbd.foundation.checkpoint import load_checkpoint

    src = inspect.getsource(load_checkpoint)
    model_block = "missing, unexpected = model.load_state_dict"
    assert model_block in src
    assert "posterior.load_state_dict" in src
    post_line = [ln for ln in src.splitlines() if "posterior.load_state_dict" in ln][0]
    assert "missing" in post_line or "load_report" in post_line, (
        "load_checkpoint discards posterior.load_state_dict's return value, so "
        "load_report records the model only. evaluate.main's fail-closed guard "
        "therefore cannot see a posterior key mismatch -- the guard is complete "
        "for one of the two modules it appears to cover."
    )


def test_absent_posterior_is_recorded_rather_than_skipped(compiled_checkpoint, tmp_path):
    """The stated fail-closed worry, tested: an absent posterior does NOT raise.

    ``load_checkpoint`` guards with ``if posterior is not None and
    payload.get("posterior")``, so a checkpoint with no posterior is silently
    skipped and writes nothing to ``load_report``. The guard cannot trip on it.
    That makes the fail-closed concern unfounded -- and means absence is
    indistinguishable from a clean load, which is the defect that matters.
    """
    from scwbd.foundation.checkpoint import load_checkpoint

    path, payload = compiled_checkpoint
    stripped = dict(payload)
    stripped["posterior"] = None
    stripped["model"] = {k.replace("._orig_mod.", "."): v for k, v in payload["model"].items()}
    p = tmp_path / "no_posterior.pt"
    torch.save(stripped, p)

    class _Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(3))

    out = load_checkpoint(str(p), posterior=_Probe(), map_location="cpu", strict=False,
                          restore_rng=False)
    rep = out.get("load_report", {})
    assert rep.get("posterior_absent") or rep.get("posterior"), (
        "a checkpoint carrying no posterior loads silently and writes nothing to "
        "load_report. A field only ever written on failure of one module is not a "
        "record of the load."
    )


# ------------------------------------------------------------------ E15
def test_evaluation_loads_and_applies_the_individualizer():
    """Stage V is scored as the population model, so G5 cannot be measured here.

    ``train.real_losses`` applies ``self.individualizer(participant=pid,
    base=th)`` before rolling out. ``evaluate._scwbd_scores`` does not, and
    ``evaluate.main`` never passes ``individualizer=`` to ``load_checkpoint``.
    An individualised checkpoint and a population checkpoint therefore produce
    the same held-out number, which is exactly the comparison G5 rests on.
    """
    from scwbd.foundation import evaluate

    main_src = inspect.getsource(evaluate.main)
    score_src = inspect.getsource(evaluate._scwbd_scores)
    loads = "individualizer=" in main_src
    applies = "individualizer" in score_src and "#" not in score_src.split("individualizer")[0][-2:]
    assert loads and applies, (
        f"evaluate.main passes individualizer to load_checkpoint: {loads}; "
        f"_scwbd_scores applies it: {applies}. train.real_losses does both. "
        f"Until the evaluation matches the training forward pass, a Stage-V "
        f"checkpoint is scored as if Stage V had not run."
    )
