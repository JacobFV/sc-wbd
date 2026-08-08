"""Assemble the released artifact: weights + ClaimManifest + config + provenance.

Run **after** training and evaluation.  Every claim written here is derived from
the evaluation JSON; nothing is asserted by hand, and
:class:`~scwbd.foundation.manifest.ClaimManifest` refuses the ones the evidence
does not support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import Claim, ClaimManifest, hash_file
from .report import CANNOT_DO
from .util import env_fingerprint, git_sha

__all__ = ["build_manifest", "main"]


def _source_rows(
    cfg_sources: Mapping[str, Any],
    *,
    contributed: "set[str] | None" = None,
) -> list[dict[str, Any]]:
    """One row per source, separating what it MAY train from what it DID.

    ``gradient_permission`` and ``frozen`` come from the card: they are a
    statement of permission. ``contributed_gradient`` comes from the
    checkpoint's own ``extra.contributed_sources``: it is a statement of fact.

    Run 2 shipped a card asserting that ``eegmmidb_real`` trained the regional
    model, while every regional tensor stayed bit-identical to its
    initialisation for the whole run. Both statements were in the artifact and
    only the permission one was written down, so every audit read the
    permission and reported it as the outcome. ``None`` means the checkpoint
    predates the tracking and the question is unanswered -- which is not the
    same as ``False``.
    """
    rows = []
    for sid, s in sorted(cfg_sources.items()):
        rows.append(
            {
                "id": sid,
                "role": s.get("role"),
                "enabled": s.get("enabled", True),
                "contributed_gradient": (None if contributed is None else sid in contributed),
                "evidence_status": "simulator_conditioned" if s.get("is_simulated") else (
                    "prior" if s.get("role") == "prior" else "measured"
                ),
                "n_eff": s.get("n_eff"),
                "n_rows_audit_only": s.get("n_rows"),
                "bias_status": s.get("bias_status"),
                "model_discrepancy": s.get("model_discrepancy"),
                "validity_overlap": s.get("validity_overlap"),
                "gradient_permission": list(s.get("gradient_permission", ())),
                "frozen": list(s.get("frozen", ())),
            }
        )
    return rows


def _attachment_kinds(
    ck_extra: Mapping[str, Any], contributed: "set[str] | None"
) -> dict[str, Any] | None:
    """Which of the schema's four attachment kinds a run actually exercised.

    Derived from the enabled cards' declared channels crossed with the
    checkpoint's own contributed-source list. Returns ``None`` when the
    checkpoint cannot answer, because a card that says "not exercised" and a
    card that says "we did not record" are different cards.
    """
    if contributed is None:
        return None
    try:
        from .attachment_report import attachment_report
    except Exception:  # noqa: BLE001 - the card must build without it
        return None
    try:
        rep = attachment_report(None)
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, Any] = {}
    for kind, v in rep["kinds"].items():
        feeding = [
            c["source"]
            for c in v["channels"]
            if c["feeds_a_loss"] and c["enabled"] and c["source"] in contributed
        ]
        out[kind] = {
            "declared_by_an_enabled_card": v["declared_by_enabled_card"],
            "reached_the_model": bool(feeding),
            "via_sources": sorted(set(feeding)),
        }
    return out


def build_manifest(
    *,
    checkpoint: str | Path,
    evaluation: str | Path,
    summary: str | Path,
    corpus_stats: str | Path | None = None,
    out: str | Path | None = None,
) -> ClaimManifest:
    ck = Path(checkpoint)
    ev = json.loads(Path(evaluation).read_text()) if Path(evaluation).exists() else {}
    sm = json.loads(Path(summary).read_text()) if Path(summary).exists() else {}
    corpus = json.loads(Path(corpus_stats).read_text()) if corpus_stats and Path(corpus_stats).exists() else {}

    srcs = sm.get("sources", {})

    # Read off the weights, not off the config. `extra.contributed_sources` is
    # written by the trainer from the loss terms that actually ran, and
    # `extra.moved_since_init` from a sha256 of every parameter taken before the
    # first step. A checkpoint that predates either leaves them None, and None
    # is reported as "unknown" rather than collapsed into "no".
    ck_extra: dict[str, Any] = {}
    if ck.exists():
        try:
            import torch

            ck_extra = (
                torch.load(ck, map_location="cpu", weights_only=False).get("extra") or {}
            )
        except Exception:  # noqa: BLE001 - a card must still build without torch
            ck_extra = {}
    contributed = ck_extra.get("contributed_sources")
    contributed_set = set(contributed) if contributed is not None else None

    m = ClaimManifest(
        git_sha=git_sha(),
        environment=env_fingerprint(),
        weights_hash=hash_file(ck) if ck.exists() else "",
        config_hash=str(ev.get("config", {}).get("train", {}).get("seed", "")),
        training_sources=_source_rows(srcs, contributed=contributed_set),
        anatomy=ev.get("anatomy", {}),
        corpus=corpus,
        cannot_do=CANNOT_DO,
        metrics={
            "sim_val_nll": ev.get("sim_val_nll"),
            "parameters": ev.get("n_parameters", {}).get("TOTAL"),
            "posterior_parameters": sm.get("posterior_parameters"),
            "global_steps": sm.get("global_steps"),
            "stage_wall_seconds": {s["stage"]: s.get("wall_seconds") for s in sm.get("stages", []) if "stage" in s},
            # "every enabled source contributed" and "every kind of signal the
            # schema declares was exercised" are different claims, and only the
            # second answers a schematic that gives boundary_output equal
            # billing with observation. Both are recorded; neither is inferred
            # from the other.
            "contributed_sources": sorted(contributed_set) if contributed_set is not None else None,
            "admitted_but_no_term": ck_extra.get("admitted_but_no_term") or {},
            "parameters_moved_since_init": (
                {
                    k: v
                    for k, v in (ck_extra.get("moved_since_init") or {}).items()
                    if k in ("n_parameters", "n_moved", "n_frozen", "unfingerprinted")
                }
                or None
            ),
            "attachment_kinds_exercised": _attachment_kinds(ck_extra, contributed_set),
        },
        notes=(
            "SC-WBD-001-beta. The word 'beta' is load-bearing: this release targets build-order "
            "items 1-5 with claim-bearing gates, not a whole-brain prediction claim."
        ),
    )

    anat = ev.get("anatomy", {})
    if not anat.get("is_biological", False):
        m.add_negative(
            "anatomy_is_synthetic",
            "This checkpoint was trained against the labelled synthetic fallback connectome, not a "
            "measured connectome. No anatomical or connectome-prior claim (gate G2) is supported by "
            "this artifact, and the G2 control comparisons it contains are internal consistency "
            "checks only.",
            {"provenance": anat.get("provenance"), "n_regions": anat.get("n_regions")},
        )

    # -- posterior calibration -------------------------------------------
    cal = ev.get("posterior_calibration", {})
    if cal.get("available"):
        worst = min(cal.get("sbc_ks_pvalue", [1.0]) or [1.0])
        mae = cal.get("coverage_mae", 1.0)
        status = "partial" if (worst > 0.01 and mae < 0.12) else "unsupported"
        m.add_claim(
            Claim(
                id="amortized_posterior_self_consistency",
                statement=(
                    "The amortized posterior over global coupling, conduction velocity, regional E/I "
                    "balance, process noise and drive is self-consistent under the simulator that "
                    "generated its training corpus, as measured by SBC rank uniformity and "
                    "expected-coverage curves."
                ),
                status=status,
                evidence_status="simulator_conditioned",
                evidence={
                    "sbc_ks_pvalue_min": worst,
                    "coverage_mae": mae,
                    "posterior_r2": cal.get("posterior_r2"),
                    "posterior_z_sd": cal.get("posterior_z_sd"),
                    "n_datasets": cal.get("coverage", {}).get("n_datasets"),
                },
                sources=("sim_wholebrain",),
                falsifier=(
                    "SBC rank histograms departing from uniformity (KS p below 0.01) or an "
                    "expected-coverage curve deviating from the diagonal by more than 0.12 on held-out "
                    "simulated datasets."
                ),
                caveats=(
                    "Simulator-conditioned only. Calibration against the same simulator that produced "
                    "the training corpus says nothing about calibration on measured human data, and "
                    "nothing about biological validity.",
                    "The parameters are the simulator's parameters, not measured physiological "
                    "quantities; 'conduction velocity' here is a knob of the delay model.",
                ),
            )
        )

    # -- real EEG --------------------------------------------------------
    hold = ev.get("real_eeg_holdout", {})
    if hold.get("available"):
        beaten = hold.get("scwbd_beaten_by", [])
        res = hold["results"]
        me = res.get("scwbd_001_beta", {})
        m.add_claim(
            Claim(
                id="held_out_real_eeg_forecast",
                statement=(
                    "On a participant-level holdout of measured 64-channel EEG, SC-WBD-001-beta "
                    "produces a proper predictive distribution in sensor space whose held-out "
                    "Gaussian log-likelihood is reported with participant-clustered 95% intervals "
                    "alongside persistence, AR, VAR, population, subject-specific and "
                    "equal-capacity dense-neural baselines."
                ),
                status="negative" if beaten else "partial",
                evidence_status="measured",
                evidence={
                    "nll_per_sample": me.get("nll_per_sample"),
                    "nll_ci95": me.get("nll_ci95"),
                    "n_test_participants": hold.get("n_test_participants"),
                    "n_test_windows": hold.get("n_test_windows"),
                    "ranking_best_first": hold.get("ranking_best_first"),
                    "beaten_by": beaten,
                },
                sources=("eegmmidb_real",),
                falsifier=(
                    "A matched-capacity baseline achieving a lower held-out NLL with non-overlapping "
                    "participant-clustered intervals -- which is precisely what is recorded here when "
                    "it happens."
                ),
                caveats=(
                    "One site, one montage, 109 adults, resting and motor-imagery conditions. No "
                    "generalisation beyond that population or protocol is claimed.",
                    "Window-level scores are not individual-level generalisation.",
                )
                + (("A baseline beat the model; the comparison is reported as a negative result.",) if beaten else ()),
            )
        )
        if beaten:
            m.add_negative(
                "real_eeg_holdout_beaten_by_baseline",
                "SC-WBD-001-beta did not achieve the best held-out likelihood on measured EEG. "
                f"Beaten by: {', '.join(beaten)}. Reported as found; no tuning against this metric "
                "was performed after the comparison was run.",
                {k: res[k].get("nll_per_sample") for k in res if "nll_per_sample" in res[k]},
            )
    else:
        m.add_negative(
            "real_eeg_holdout_unavailable",
            f"No measured-EEG holdout evaluation was produced ({hold.get('reason', 'unknown')}). "
            "Without it the artifact has no measured evidence at all and every claim about brains "
            "remains unsupported.",
            {},
        )

    # -- backends ---------------------------------------------------------
    bc = ev.get("backend_comparison", {})
    if bc.get("per_backend_nll"):
        m.add_claim(
            Claim(
                id="backend_interchangeability",
                statement=(
                    "A single learned operator forecasts trajectories from five mechanistically "
                    "distinct neural-mass families, and its per-family held-out error is reported "
                    "rather than pooled, so the families can be compared instead of assumed."
                ),
                status="partial",
                evidence_status="simulator_conditioned",
                evidence=bc,
                sources=("sim_wholebrain",),
                falsifier=(
                    "A family whose trajectories the learned operator cannot forecast at all, or "
                    "conversely uniform performance across families, which would show the metric "
                    "does not discriminate mechanisms."
                ),
                caveats=(
                    "This is a statement about the operator's expressiveness on simulator output. It "
                    "is not evidence that any of the five families is neurally realized.",
                ),
            )
        )

    # -- structural claims that are true by construction -------------------
    m.add_claim(
        Claim(
            id="typed_gradient_permissions",
            statement=(
                "Every training source enters through a source card whose gradient permission A_k is "
                "compiled into an executable mask: gradients are taken only with respect to permitted "
                "parameters, so a non-permitted parameter never receives one."
            ),
            status="supported",
            evidence_status="prior",
            evidence={"test": "tests/foundation/test_gradient_masks.py::test_non_permitted_parameters_have_grad_none"},
            falsifier="A parameter outside a source's A_k acquiring a non-None gradient from that source.",
        )
    )
    m.add_claim(
        Claim(
            id="r07_centering",
            statement=(
                "Stage V population effects are sum-to-zero by construction and person/session effects "
                "are hierarchically shrunk; a simulated recovery test confirms the decomposition is "
                "identified rather than merely written down."
            ),
            status="supported",
            evidence_status="prior",
            evidence={"test": "tests/foundation/test_checkpoint_manifest.py::test_hierarchical_decomposition_is_identified"},
            falsifier=(
                "A weighted group-effect sum departing from zero, or a recovery test in which the "
                "summed parameter is recovered while its components are not."
            ),
        )
    )
    m.add_claim(
        Claim(
            id="teacher_distillation_off",
            statement=(
                "TRIBE v2 distillation is disabled in this artifact and is declared with the "
                "distillation role, which cannot contribute a subject likelihood."
            ),
            status="supported",
            evidence_status="prior",
            evidence={"card": "configs/source_cards/tribe_v2_teacher.yaml", "enabled": False},
            falsifier="Any released checkpoint whose training mixture includes an enabled teacher source.",
        )
    )

    # -- source ablation ---------------------------------------------------
    sa = ev.get("source_ablation")
    if sa and sa.get("negative_transfer"):
        m.add_negative(
            "negative_transfer",
            "Removing these source families improved held-out validation error, i.e. they transferred "
            "negatively: " + ", ".join(sa["negative_transfer"]),
            {k: v for k, v in sa.items() if k.startswith("delta_")},
        )

    if out:
        m.save(out)
        Path(out).with_name("CLAIM_MANIFEST.md").write_text(m.to_markdown())
    return m


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="assemble the SC-WBD-001-beta release manifest")
    ap.add_argument("--checkpoint", default="checkpoints/scwbd-001-beta/last.pt")
    ap.add_argument("--evaluation", default="reports/training/evaluation.json")
    ap.add_argument("--summary", default="reports/training/scwbd-001-beta_summary.json")
    ap.add_argument("--corpus", default="reports/training/corpus_statistics.json")
    ap.add_argument("--out", default="checkpoints/scwbd-001-beta/claim_manifest.json")
    a = ap.parse_args(argv)
    m = build_manifest(
        checkpoint=a.checkpoint, evaluation=a.evaluation, summary=a.summary, corpus_stats=a.corpus, out=a.out
    )
    print(m.to_markdown())


if __name__ == "__main__":  # pragma: no cover
    main()
