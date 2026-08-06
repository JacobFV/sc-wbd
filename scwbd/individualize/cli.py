"""``python -m scwbd.individualize.cli`` -- regenerate everything from source.

Three subcommands, and none of them reads a table:

``verify``
    recompute ``theta_profile_min_eigenvalue_nonprior`` for every (regime,
    design) at the committed benchmark configuration and diff it against
    ``reports/identifiability/results.json``.  Exit code 1 on any mismatch.
``table``
    the per-parameter-group identifiability table across modality combinations
    and regimes, at the committed benchmark configuration.
``patients``
    end-to-end per-patient reports for the four clinically interesting
    availabilities, including a pure-noise negative control.

Everything runs on **CPU**.  A 12 h training run owns the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DEFAULT_OUT = Path("reports/individualize")
COMMITTED = Path("reports/identifiability/results.json")

#: The modality combinations a clinic actually presents.
PATIENTS: tuple[tuple[str, list[str], dict[str, Any]], ...] = (
    ("P01-mri-only", ["structural_mri", "dmri"], {}),
    ("P02-eeg-only", ["structural_mri", "eeg"], {}),
    ("P03-fmri-only", ["structural_mri", "fmri"], {}),
    ("P04-joint", ["structural_mri", "eeg", "fmri"], {}),
    ("P05-noise-control", ["structural_mri", "eeg"], {"pure_noise": True}),
)


def _environment() -> dict[str, Any]:
    import torch

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # pragma: no cover
        rev = ""
    return {
        "git_revision": rev,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }


# --------------------------------------------------------------------------
def cmd_verify(args) -> int:
    from scwbd.individualize.profile import (
        _fisher_for_design,
        benchmark_config,
        profiled_information,
    )
    from scwbd.infer.identifiability import REGIMES
    from scwbd.infer.linear_gaussian import PARAM_INDEX, THETA_NAMES

    path = Path(args.committed)
    if not path.exists():
        print(f"[error] {path} not found", file=sys.stderr)
        return 1
    committed = json.loads(path.read_text())["decision"]["per_regime"]
    cfg = benchmark_config()
    idx = [PARAM_INDEX[t] for t in THETA_NAMES]

    rows: list[dict[str, Any]] = []
    bad = 0
    for regime in REGIMES:
        for design in ("eeg_only", "fmri_only", "joint_native"):
            t0 = time.time()
            I, _ = _fisher_for_design(
                design, cfg=cfg, regime=regime, eta=None, seed=args.seed
            )
            S, _ = profiled_information(I, idx, nuisance_prior=0.0)
            lam = float(np.linalg.eigvalsh(0.5 * (S + S.T)).min())
            want = committed[regime.name][
                "theta_profile_min_eigenvalue_nonprior"
            ][design]
            rel = abs(lam - want) / max(abs(want), 1e-300)
            absd = abs(lam - want)
            # Two tolerances, because two regimes of conditioning.  The
            # well-determined designs agree to ~1e-13 relative.  The fMRI-only
            # value is ~3e-06 against an information matrix with entries up to
            # ~25: it is the residue of a Schur complement that cancelled seven
            # orders of magnitude, so its *relative* precision is ~1e-7 while
            # its *absolute* precision is ~1e-13 -- and it varies at that level
            # between runs on the same machine with different BLAS thread
            # counts.  Demanding 1e-9 relative there would be demanding a
            # precision float64 does not have.
            ok = rel < args.rtol or absd < args.atol
            bad += 0 if ok else 1
            rows.append(
                {
                    "regime": regime.name,
                    "design": design,
                    "recomputed": lam,
                    "committed": want,
                    "relative_error": rel,
                    "absolute_error": absd,
                    "agrees": ok,
                    "criterion": (
                        "relative" if rel < args.rtol else
                        "absolute (cancellation-limited)" if absd < args.atol else
                        "neither"
                    ),
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"[{'ok ' if ok else 'BAD'}] {regime.name}/{design}: "
                f"{lam!r} vs {want!r} (rel {rel:.3g}, abs {absd:.3g})",
                flush=True,
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_fisher.json").write_text(
        json.dumps(
            {
                "purpose": (
                    "regenerated from source on CPU; the committed run used "
                    "CUDA, so agreement is also a device-parity check"
                ),
                "committed_file": str(path),
                "config": {
                    "epoch_seconds": cfg.epoch_seconds,
                    "n_epochs": cfg.n_epochs,
                    "n_delay_taps": cfg.n_delay_taps,
                    "hrf_stages": cfg.hrf_stages,
                    "dtype": cfg.dtype,
                    "seed": args.seed,
                    "source": "reports/identifiability/manifest.json extra.command",
                },
                "rtol": args.rtol,
                "atol": args.atol,
                "rows": rows,
                "n_disagreements": bad,
                "environment": _environment(),
            },
            indent=1,
        )
    )
    print(f"[written] {out / 'verify_fisher.json'}  disagreements={bad}")
    return 1 if bad else 0


# --------------------------------------------------------------------------
def cmd_table(args) -> int:
    from scwbd.individualize.availability import ModalityAvailability
    from scwbd.individualize.groups import GROUPS, LIKELIHOOD_GROUPS
    from scwbd.individualize.profile import (
        IdentifiabilityThresholds,
        benchmark_config,
        profile_identifiability,
    )
    from scwbd.infer.identifiability import REGIMES

    cfg = benchmark_config()
    th = IdentifiabilityThresholds()
    combos = [
        ("mri_only", ["structural_mri"]),
        ("dmri_only", ["dmri"]),
        ("eeg_only", ["structural_mri", "eeg"]),
        ("meg_only", ["structural_mri", "meg"]),
        ("fmri_only", ["structural_mri", "fmri"]),
        ("eeg_fmri", ["structural_mri", "eeg", "fmri"]),
        ("behavior_only", ["behavior"]),
        ("nothing", []),
    ]
    payload: dict[str, Any] = {
        "purpose": (
            "per-parameter-group identifiability by modality combination, at "
            "the committed benchmark configuration; regenerated from source"
        ),
        "config": {
            "epoch_seconds": cfg.epoch_seconds,
            "n_epochs": cfg.n_epochs,
            "n_delay_taps": cfg.n_delay_taps,
            "hrf_stages": cfg.hrf_stages,
            "dtype": cfg.dtype,
        },
        "thresholds": th.to_dict(),
        "groups": [g.to_dict() for g in GROUPS],
        "regimes": {},
        "environment": _environment(),
    }
    for regime in REGIMES:
        if args.regimes and regime.name not in args.regimes:
            continue
        block: dict[str, Any] = {}
        for label, mods in combos:
            av = ModalityAvailability.from_modalities(f"T-{label}", mods)
            p = profile_identifiability(av, cfg=cfg, regime=regime)
            block[label] = {
                "modalities": mods,
                "design": p.design,
                "groups": {k: v.to_dict() for k, v in p.groups.items()},
            }
            print(
                f"[{regime.name}/{label}] "
                + " ".join(
                    f"{g.name}={block[label]['groups'][g.name]['lambda_min_likelihood']:.6g}"
                    f"({block[label]['groups'][g.name]['status'][:4]})"
                    for g in LIKELIHOOD_GROUPS
                ),
                flush=True,
            )
        payload["regimes"][regime.name] = block

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identifiability_by_modality.json").write_text(json.dumps(payload, indent=1))
    (out / "identifiability_by_modality.md").write_text(_table_markdown(payload))
    print(f"[written] {out / 'identifiability_by_modality.json'}")
    return 0


def _table_markdown(payload: dict[str, Any]) -> str:
    from scwbd.individualize.groups import LIKELIHOOD_GROUPS

    L = ["# Identifiability by modality combination", ""]
    L.append(
        "Minimum eigenvalue of the Schur complement of the **likelihood-only** "
        "expected Fisher information on each parameter group, other parameters "
        "profiled out, prior-standardised basis. Units are prior precision: "
        "`lambda_min = 1` means the data are worth as much as the prior."
    )
    L.append("")
    L.append(f"Configuration: `{json.dumps(payload['config'], sort_keys=True)}`")
    L.append(f"Thresholds: `{json.dumps(payload['thresholds'], sort_keys=True)}`")
    L.append("")
    for regime, block in payload["regimes"].items():
        L.append(f"## regime `{regime}`")
        L.append("")
        head = "| available data | design | " + " | ".join(
            f"`{g.name}`" for g in LIKELIHOOD_GROUPS
        ) + " |"
        L.append(head)
        L.append("|" + "---|" * (2 + len(LIKELIHOOD_GROUPS)))
        for label, row in block.items():
            cells = []
            for g in LIKELIHOOD_GROUPS:
                e = row["groups"][g.name]
                lam = e["lambda_min_likelihood"]
                cells.append(f"{lam:.6g} ({e['status'].replace('_identifiable','')})")
            L.append(
                f"| `{label}` ({', '.join(row['modalities']) or 'none'}) | "
                f"`{row['design']}` | " + " | ".join(cells) + " |"
            )
        L.append("")
    L.append("Anatomical groups are presence-determined and carry no lambda_min:")
    L.append("")
    for g in payload["groups"]:
        if g["kind"] == "anatomical":
            L.append(f"- `{g['name']}` <- {g['informed_by']}: {g['clinical_meaning']}")
    return "\n".join(L)


# --------------------------------------------------------------------------
def cmd_patients(args) -> int:
    from scwbd.individualize import (
        ModalityAvailability,
        PopulationModel,
        answer,
        coupling_gain_query,
        individualize,
        patient_report,
        profile_identifiability,
        result_json,
        simulate_patient,
    )
    from scwbd.individualize.profile import benchmark_config
    from scwbd.infer.linear_gaussian import PARAM_INDEX

    cfg = benchmark_config(
        epoch_seconds=args.epoch_seconds, n_epochs=args.n_epochs
    )
    out = Path(args.out)
    (out / "patients").mkdir(parents=True, exist_ok=True)
    pop = PopulationModel.reference()

    def predicted_response(theta):
        return float(
            theta[PARAM_INDEX["a21"]]
            * theta[PARAM_INDEX["a32"]]
            * np.exp(theta[PARAM_INDEX["tau"]])
        )

    index: list[dict[str, Any]] = []
    for pid, mods, kw in PATIENTS:
        t0 = time.time()
        av = ModalityAvailability.from_modalities(pid, mods)
        prof = profile_identifiability(
            av, cfg=cfg, counterfactual_modalities=("eeg", "fmri")
        )
        data = None
        if av.channels:
            # NOT ``hash(pid)``: str hashing is salted per process, so that
            # would make this report irreproducible across runs.
            seed = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:4], "big")
            data = simulate_patient(av, cfg=cfg, seed=seed % 100000, **kw)
        res = individualize(
            pop, av, data, profile=prof, cfg=cfg, n_newton=args.newton
        )
        q = answer(res, coupling_gain_query(), predicted_response)
        md = patient_report(res)
        md += "\n\n## 7. A coupling-dependent query\n\n"
        md += f"```\n{q}\n```\n"
        (out / "patients" / f"{pid}.md").write_text(md)
        (out / "patients" / f"{pid}.json").write_text(result_json(res))
        row = {
            "patient": pid,
            "modalities": mods,
            "design": av.design,
            "individualized": list(res.individualized_groups),
            "population_prior": list(res.population_prior_groups),
            "coupling_lambda_min": prof["coupling"].lambda_min,
            "query_outcome": type(q).__name__,
            "consistency_passed": (
                res.consistency.passed if res.consistency else None
            ),
            "seconds": time.time() - t0,
        }
        index.append(row)
        print(f"[{pid}] {json.dumps(row)}", flush=True)

    (out / "patients" / "index.json").write_text(
        json.dumps(
            {
                "config": {
                    "epoch_seconds": cfg.epoch_seconds,
                    "n_epochs": cfg.n_epochs,
                    "newton": args.newton,
                },
                "patients": index,
                "environment": _environment(),
            },
            indent=1,
        )
    )
    print(f"[written] {out / 'patients'}")
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser("scwbd.individualize")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="regenerate agent Fisher's committed numbers")
    v.add_argument("--committed", default=str(COMMITTED))
    v.add_argument("--rtol", type=float, default=1e-9)
    v.add_argument("--atol", type=float, default=1e-11,
                   help="absolute fallback for cancellation-limited values")
    v.add_argument("--seed", type=int, default=20260805)
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("table", help="per-group identifiability by modality")
    t.add_argument("--regimes", nargs="*", default=None)
    t.set_defaults(func=cmd_table)

    q = sub.add_parser("patients", help="end-to-end per-patient reports")
    q.add_argument("--epoch-seconds", type=float, default=3.0)
    q.add_argument("--n-epochs", type=int, default=8)
    q.add_argument("--newton", type=int, default=3)
    q.set_defaults(func=cmd_patients)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
