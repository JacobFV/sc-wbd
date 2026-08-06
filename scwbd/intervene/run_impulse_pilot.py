"""Run the preregistered pose-contrast analysis against a trained checkpoint.

Staged **before** any checkpoint exists, so it runs the moment one does.  The
criterion it implements is fixed in
``reports/intervene/impulse_pilot_preregistration.md``, committed at ``007bee2``
while ``checkpoints/`` was empty.  This module implements that document and
must not quietly diverge from it: :func:`main` re-states each threshold as a
named constant and the report records both SHAs.

Usage::

    PYTHONPATH=. python -m scwbd.intervene.run_impulse_pilot
    PYTHONPATH=. python -m scwbd.intervene.run_impulse_pilot --checkpoint path.pt

With no checkpoint present it reports ``awaiting_checkpoint`` and **exits 0**.
That is not a failure and must not be treated as one: a staged analysis whose
input has not arrived is a different state from an analysis that ran and found
nothing, and collapsing the two is how a pipeline reports success for work it
never did (``reports/decorative_guards.md``, the exit-code entry).

What this is not
----------------
A forward-model measurement.  It does not optimise a coil position, rank
poses, or recommend anything: the two poses are fixed constants from the
preregistration.  ``trained_on_perturbation_data`` stays ``False`` whatever the
result -- the model will have seen resting dynamics, never a TMS-evoked
response, so a surviving contrast means the trained dynamics propagate a focal
input pose-dependently, not correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .impulse_response import UNTRAINED_PREDICTION_NOTICE, parcel_drive, predict_impulse_response

__all__ = [
    "PREREGISTRATION",
    "PREREG_SHA",
    "COIL_A",
    "COIL_B",
    "COLLAPSE_CRR",
    "ATTENUATION_FRACTION",
    "N_PERMUTATIONS",
    "PilotResult",
    "contrast_to_response_ratio",
    "run_pilot",
    "main",
]

#: The document this module implements. Any change to a threshold here without
#: a matching change there is a defect, and the report names both.
PREREGISTRATION = "reports/intervene/impulse_pilot_preregistration.md"
#: The commit that fixed the criterion, made while ``checkpoints/`` was empty.
PREREG_SHA = "007bee2"

# -- fixed configuration, copied from the preregistration -------------------
COIL_A = (0.00, 0.00, 0.10)
COIL_B = (0.00, 0.10, 0.00)
N_STEPS = 64
GAIN = 50.0
BATCH = 4
CONTEXT_SEED = 1
THETA_SEED = 1
N_PERMUTATIONS = 200
PERMUTATION_SEED = 20260806

#: `CRR` below this is reported as **collapsed**.
COLLAPSE_CRR = 0.10
#: `CRR` below this fraction of the untrained value is **attenuated**.
ATTENUATION_FRACTION = 0.5
#: One-sided permutation alpha for the shuffled-normal null.
ALPHA = 0.05

_DT = torch.float32


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------


def _rms(x: Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.to(torch.float64) ** 2)))


def contrast_to_response_ratio(response_a: Any, response_b: Any) -> float:
    """``RMS(eeg_A - eeg_B) / mean RMS evoked``. Dimensionless, scale-free.

    Scale-free deliberately: a trained model may have a wholly different output
    scale from an untrained one, and comparing raw differences across the two
    arms would confound "the pose matters more" with "the outputs got bigger".
    """
    num = _rms(response_a.eeg - response_b.eeg)
    den = 0.5 * (_rms(response_a.evoked) + _rms(response_b.evoked))
    if den <= 0.0:
        return 0.0
    return num / den


@dataclass(frozen=True)
class PilotResult:
    status: str
    crr: dict[str, float]
    reading: str
    null: dict[str, Any]
    control: dict[str, Any]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "crr": self.crr,
            "reading": self.reading,
            "shuffled_normal_null": self.null,
            "control": self.control,
            "provenance": self.provenance,
            "preregistration": PREREGISTRATION,
            "preregistration_sha": PREREG_SHA,
            "notice": UNTRAINED_PREDICTION_NOTICE,
        }


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def _positions(prior: Any) -> Tensor:
    pos = getattr(prior, "position", None)
    n = int(prior.n_regions)
    if pos is not None:
        p = torch.as_tensor(pos, dtype=_DT).reshape(n, 3)
        if torch.isfinite(p).all():
            return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9) * 0.07
    g = torch.Generator().manual_seed(11)
    p = torch.randn(n, 3, generator=g)
    return p / p.norm(dim=-1, keepdim=True) * 0.07


def _efield(centre: Tensor, pos: Tensor) -> Tensor:
    d = pos - centre.reshape(1, 3)
    r = d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    return 1e4 * d / r.pow(3) * 1e-6


def _make_context(prior: Any) -> tuple[Tensor, Tensor]:
    from ..foundation.simulate import ThetaPrior

    g = torch.Generator().manual_seed(CONTEXT_SEED)
    y = torch.randn(BATCH, 4, int(prior.n_regions), generator=g)
    th = ThetaPrior().sample(BATCH, seed=THETA_SEED)
    return y, th


def _baseline_eeg(model: Any, prior: Any) -> Tensor:
    """The unperturbed readout. Depends on model/context/theta only.

    Computed once and reused across poses and permutations. Exact, not an
    approximation -- the baseline is the ``u=None`` rollout and the drive does
    not enter it -- and it halves the cost of the permutation null.
    """
    y, th = _make_context(prior)
    with torch.no_grad():
        _b = getattr(model, "set_mechanistic_theta", None)
        if _b is not None and getattr(model, "family_layout", None) is not None:
            from scwbd.foundation.anatomy import load_anatomy

            _b(th, load_anatomy())
        roll = model.rollout(y_context=y, theta=th, n_steps=N_STEPS)
        mu, _ = model.eeg(roll.state)
    return mu


def _arm(
    model: Any,
    prior: Any,
    pos: Tensor,
    normal: Tensor,
    coherence: Tensor | None,
    baseline: Tensor | None = None,
):
    y, th = _make_context(prior)
    out = {}
    for name, centre in (("A", COIL_A), ("B", COIL_B)):
        d = parcel_drive(_efield(torch.tensor(centre, dtype=_DT), pos), normal, coherence=coherence)
        out[name] = predict_impulse_response(
            model, d, y_context=y, theta=th, n_steps=N_STEPS, gain=GAIN,
            baseline_eeg=baseline,
        )
    return out


def _shuffled_normal_null(
    model: Any, prior: Any, pos: Tensor, normal: Tensor, coherence: Tensor | None, crr_real: float
) -> dict[str, Any]:
    """Permute normals across covered parcels; keep field and parcel identity.

    Isolates orientation: field magnitude and which parcel is which are
    untouched, only the ``E.n`` projection structure is destroyed.
    """
    n = torch.as_tensor(normal, dtype=_DT)
    covered = torch.isfinite(n).all(dim=-1)
    idx = torch.nonzero(covered).reshape(-1)
    g = torch.Generator().manual_seed(PERMUTATION_SEED)
    baseline = _baseline_eeg(model, prior)   # invariant across permutations

    draws: list[float] = []
    for _ in range(N_PERMUTATIONS):
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        shuffled = n.clone()
        shuffled[idx] = n[perm]
        arm = _arm(model, prior, pos, shuffled, coherence, baseline=baseline)
        draws.append(contrast_to_response_ratio(arm["A"], arm["B"]))

    t = torch.tensor(draws, dtype=torch.float64)
    # One-sided: predicted direction is real > shuffled. +1 for the observed
    # value, the standard finite-sample correction -- a p of exactly 0 would
    # claim more resolution than 200 draws can carry.
    p = float((t >= crr_real).sum().add(1).div(N_PERMUTATIONS + 1))
    return {
        "k": N_PERMUTATIONS,
        "seed": PERMUTATION_SEED,
        "crr_real": crr_real,
        "null_mean": float(t.mean()),
        "null_std": float(t.std()),
        "null_max": float(t.max()),
        "percentile_of_real": float((t < crr_real).to(torch.float64).mean() * 100.0),
        "p_one_sided": p,
        "alpha": ALPHA,
        "orientation_carries_the_contrast": bool(p < ALPHA),
        "direction_predicted_in_advance": "crr_real > crr_shuffled",
    }


def run_pilot(
    checkpoint: Path | None = None,
    *,
    device: str = "cpu",
    permutations: bool = True,
) -> PilotResult:
    """Run the preregistered analysis. Never raises for a missing checkpoint."""
    from ..foundation.anatomy import load_anatomy
    from ..foundation.checkpoint import load_checkpoint
    from ..foundation.config import ModelConfig
    from ..foundation.model import SCWBD
    from ..foundation.util import set_determinism

    prior = load_anatomy(device=device)
    normal = getattr(prior, "normal", None)
    if normal is None:
        return PilotResult(
            status="unavailable",
            crr={}, reading="anatomy prior carries no cortical normals",
            null={}, control={},
            provenance={"reason": "prior.normal is None; a fallback prior cannot run this"},
        )
    coherence = getattr(prior, "normal_coherence", None)
    pos = _positions(prior)

    ckpt_meta: dict[str, Any] = {"found": False}
    if checkpoint is None:
        found = sorted(Path("checkpoints").rglob("*.pt")) if Path("checkpoints").exists() else []
        checkpoint = found[0] if found else None

    # -- untrained arm, always available ------------------------------------
    set_determinism(0)
    untrained_cfg = ModelConfig(n_regions=int(prior.n_regions))
    untrained = SCWBD(untrained_cfg, prior).eval()

    # -- trained arm, if a checkpoint exists --------------------------------
    #
    # Three states, deliberately distinct. "Not arrived" and "arrived and
    # unreadable" are different facts and a harness that reports the same thing
    # for both is the silent-load-failure pattern (reports/decorative_guards.md,
    # the `strict=False` plus discarded load report entry). The first version of
    # this function had exactly that: it captured `load_report` into provenance
    # and never looked at it, so a checkpoint whose keys did not match would
    # have loaded nothing and still reported `ran`.
    trained = None
    load_failure: str | None = None
    if checkpoint is not None and Path(checkpoint).exists():
        try:
            payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            payload, load_failure = None, f"torch.load failed: {type(exc).__name__}: {exc}"
        if payload is not None and payload.get("format") != "scwbd-foundation-checkpoint/1":
            load_failure = f"unrecognised checkpoint format {payload.get('format')!r}"
            payload = None

        if payload is not None:
            cfg_dict = (payload.get("config") or {}).get("model", payload.get("config") or {})
            try:
                cfg = ModelConfig(
                    **{k: v for k, v in cfg_dict.items() if k in ModelConfig.__annotations__}
                )
            except Exception:  # noqa: BLE001
                cfg = untrained_cfg
            set_determinism(0)
            candidate = SCWBD(cfg, prior)
            # Snapshot the init so "did the weights actually change" is
            # answerable. A load report can be empty for the wrong reason;
            # this cannot.
            before = {k: v.detach().clone() for k, v in candidate.state_dict().items()}
            report: dict[str, Any] = {}
            strict_ok = True
            try:
                load_checkpoint(Path(checkpoint), model=candidate, strict=True)
            except Exception:  # noqa: BLE001 - retry loosely, but SAY so
                strict_ok = False
                try:
                    report = load_checkpoint(Path(checkpoint), model=candidate, strict=False)
                except Exception as exc:  # noqa: BLE001
                    load_failure = f"load_checkpoint failed: {type(exc).__name__}: {exc}"

            if load_failure is None:
                after = candidate.state_dict()
                changed = sum(
                    1 for k, v in after.items()
                    if k in before and not torch.equal(before[k], v)
                )
                if changed == 0:
                    # The decisive check. Whatever the load report said, no
                    # weight moved, so this is the untrained model wearing a
                    # checkpoint's name and every "trained" number would be a
                    # relabelled untrained one.
                    load_failure = (
                        "checkpoint loaded but not one weight tensor changed; "
                        "the trained arm would be the untrained model relabelled"
                    )
                else:
                    trained = candidate.eval()
                    ckpt_meta = {
                        "found": True,
                        "path": str(checkpoint),
                        "step": payload.get("step"),
                        "stage": payload.get("stage"),
                        "saved_utc": payload.get("saved_utc"),
                        "git_sha": payload.get("git_sha"),
                        "strict_load": strict_ok,
                        "load_report": (report or {}).get("load_report", {}),
                        "tensors_changed_by_load": changed,
                        "tensors_total": len(before),
                    }

    if load_failure is not None:
        return PilotResult(
            status="checkpoint_unreadable",
            crr={}, reading=load_failure, null={}, control={},
            provenance={
                "checkpoint": {"found": True, "path": str(checkpoint), "error": load_failure},
                "note": (
                    "A checkpoint exists and could not be used. This is NOT "
                    "`awaiting_checkpoint` -- reporting the same state for "
                    "'not arrived' and 'arrived and unreadable' would hide a "
                    "real failure behind a legitimate one."
                ),
            },
        )

    if trained is None:
        return PilotResult(
            status="awaiting_checkpoint",
            crr={}, reading="staged; no trained checkpoint exists yet",
            null={}, control={},
            provenance={
                "checkpoint": ckpt_meta,
                "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": (
                    "This is not a failure. The analysis is fixed and will run "
                    "unchanged when a checkpoint lands."
                ),
            },
        )

    # -- the control that voids everything else -----------------------------
    a_arm = _arm(trained, prior, pos, normal, coherence)
    repeat = _arm(trained, prior, pos, normal, coherence)
    control_crr = contrast_to_response_ratio(a_arm["A"], repeat["A"])
    control_ok = control_crr == 0.0

    crr_trained = contrast_to_response_ratio(a_arm["A"], a_arm["B"])
    u_arm = _arm(untrained, prior, pos, normal, coherence)
    crr_untrained = contrast_to_response_ratio(u_arm["A"], u_arm["B"])

    if crr_trained < COLLAPSE_CRR:
        reading = "collapsed"
    elif crr_trained < ATTENUATION_FRACTION * crr_untrained:
        reading = "attenuated"
    else:
        reading = "survived"

    null: dict[str, Any] = {"skipped": True}
    if permutations and control_ok:
        null = _shuffled_normal_null(trained, prior, pos, normal, coherence, crr_trained)

    return PilotResult(
        status="ran",
        crr={
            "trained": crr_trained,
            "untrained": crr_untrained,
            "ratio_trained_over_untrained": (
                crr_trained / crr_untrained if crr_untrained > 0 else float("nan")
            ),
        },
        reading=reading,
        null=null,
        control={
            "same_pose_crr": control_crr,
            "must_be_zero": True,
            "ok": control_ok,
            "note": (
                "if this is non-zero the statistic is measuring nondeterminism "
                "and no other number here means anything"
            ),
        },
        provenance={
            "checkpoint": ckpt_meta,
            "coil_a": list(COIL_A),
            "coil_b": list(COIL_B),
            "n_steps": N_STEPS,
            "gain": GAIN,
            "batch": BATCH,
            "n_regions": int(prior.n_regions),
            "trained_on_perturbation_data": False,
            "response_mapping_validated": False,
            "claim": (
                "a prediction about this model's dynamics under a computed "
                "field; the model has seen resting dynamics and no TMS-evoked "
                "response, so a surviving contrast means focal input propagates "
                "pose-dependently, not correctly"
            ),
        },
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _markdown(res: PilotResult) -> str:
    d = res.as_dict()
    L = [
        "# Pose contrast under training: measured",
        "",
        f"Implements `{PREREGISTRATION}`, criterion fixed at `{PREREG_SHA}` "
        "while `checkpoints/` was empty.",
        "",
        f"**Status: `{res.status}`**",
        "",
    ]
    if res.status != "ran":
        L += [res.reading, "", "```json", json.dumps(d, indent=2), "```", ""]
        return "\n".join(L)

    c = res.crr
    L += [
        f"## Reading: **{res.reading}**",
        "",
        "| | CRR |",
        "|---|---|",
        f"| trained | {c['trained']:.4f} |",
        f"| untrained | {c['untrained']:.4f} |",
        f"| ratio | {c['ratio_trained_over_untrained']:.4f} |",
        "",
        f"Thresholds fixed in advance: collapsed `< {COLLAPSE_CRR}`; "
        f"attenuated `< {ATTENUATION_FRACTION} x` untrained; else survived.",
        "",
        "## Control",
        "",
        f"same-pose CRR = {res.control['same_pose_crr']:.6g} "
        f"(must be 0; ok = {res.control['ok']})",
        "",
    ]
    if not res.null.get("skipped"):
        n = res.null
        L += [
            "## Shuffled-normal null (orientation)",
            "",
            f"- K = {n['k']}, seed {n['seed']}",
            f"- null mean {n['null_mean']:.4f}, sd {n['null_std']:.4f}",
            f"- real CRR at the {n['percentile_of_real']:.1f}th percentile",
            f"- one-sided p = {n['p_one_sided']:.4f} (alpha {n['alpha']}), "
            f"direction predicted in advance: {n['direction_predicted_in_advance']}",
            "",
            f"**Orientation carries the contrast: "
            f"{n['orientation_carries_the_contrast']}**",
            "",
        ]
    L += [
        "## What this does not establish",
        "",
        "`trained_on_perturbation_data` remains **False**. The model has seen "
        "resting dynamics and no TMS-evoked response, so a surviving contrast "
        "means the trained dynamics propagate a focal input pose-dependently, "
        "not that they do so correctly. No held-out TEP exists to check against.",
        "",
        "```json",
        json.dumps(d, indent=2),
        "```",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("reports/intervene"))
    ap.add_argument("--no-permutations", action="store_true")
    a = ap.parse_args(argv)

    res = run_pilot(
        a.checkpoint, device=a.device, permutations=not a.no_permutations
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "impulse_pilot.json").write_text(json.dumps(res.as_dict(), indent=2))
    (a.out / "impulse_pilot.md").write_text(_markdown(res))

    print(f"status: {res.status}")
    if res.status == "ran":
        print(f"reading: {res.reading}")
        print(f"CRR trained={res.crr['trained']:.4f} untrained={res.crr['untrained']:.4f}")
        if not res.null.get("skipped"):
            print(f"shuffled-normal p = {res.null['p_one_sided']:.4f}")
    else:
        print(res.reading)
    print(f"wrote {a.out / 'impulse_pilot.json'}")
    # A staged analysis whose input has not arrived exits 0. It is a state, not
    # a failure, and conflating the two reports success for work never done.
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
