"""Entry points: ``python -m scwbd.infer.cli benchmark|slice``.

The benchmark writes ``reports/identifiability/manifest.json`` **before** it
runs anything (preregistration) and ``results.json`` / ``summary.md`` /
``figures/`` afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .identifiability import DESIGNS, REGIMES, run_benchmark
from .linear_gaussian import SystemConfig
from .report import evaluate_decision_rule, make_figures, write_manifest, write_report
from .types import cap_gpu_memory, gpu_memory_report

DEFAULT_OUT = Path("reports/identifiability")


def _default_gpu_gib() -> float:
    """Default CUDA cap, overridable by ``SCWBD_CUDA_MAX_GB``.

    This box has **one** unified memory pool: ``total_memory`` (~130.6 GB) and
    ``free -h`` (~121 GiB) describe the same physical RAM in different units.
    A ``systemd-run -p MemoryMax=`` scope charges host allocations but *not*
    CUDA allocations -- measured here at 6.8 GB of CUDA against 2.4 GB of
    cgroup-visible RSS -- so a cgroup cap alone will not stop a runaway kernel
    from taking the machine down.  The allocator-enforced cap in
    :func:`scwbd.infer.types.cap_gpu_memory` closes that gap; this function only
    decides its default, so a run launched under the project's memory discipline
    inherits the same bound without having to pass ``--gpu-gib``.
    """
    gb = os.environ.get("SCWBD_CUDA_MAX_GB")
    return float(gb) if gb else 20.0


def _cfg(args) -> SystemConfig:
    return SystemConfig(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=args.dtype,
        epoch_seconds=args.epoch_seconds,
        n_epochs=args.n_epochs,
    )


def cmd_benchmark(args) -> int:
    cap = cap_gpu_memory(args.gpu_gib)
    print(f"[gpu cap] {cap}", flush=True)
    cfg = _cfg(args)
    out = Path(args.out)
    regimes = REGIMES if not args.regimes else [
        r for r in REGIMES if r.name in set(args.regimes)
    ]
    designs = DESIGNS if not args.primary_only else [d for d in DESIGNS if d.primary]
    # The manifest must record what this run will *actually* compute, not the
    # defaults of the flags it ignored: an arm that is switched off has zero
    # replicates, and the recovery arm may be restricted to a subset.
    recovery_designs = (
        [] if args.no_recovery
        else list(args.recovery_designs or [d.name for d in designs if d.primary])
    )
    write_manifest(
        out, cfg=cfg, regimes=regimes, designs=designs, seed=args.seed,
        n_replicates=0 if args.no_recovery else args.replicates,
        mc_replicates=0 if args.no_monte_carlo else args.mc_replicates,
        extra={"command": {k: v for k, v in vars(args).items() if k != "func"},
               "recovery_designs": recovery_designs,
               "arms_computed": {
                   "recovery": not args.no_recovery,
                   "profile_likelihood": not args.no_profiles,
                   "monte_carlo_fisher": not args.no_monte_carlo,
               },
               "profile_and_monte_carlo_regime": regimes[0].name,
               "gpu_memory_cap": cap},
    )
    print(f"[preregistered] {out/'manifest.json'}")
    t0 = time.time()
    res = run_benchmark(
        cfg, regimes=regimes, designs=designs, seed=args.seed,
        n_replicates=args.replicates, n_newton=args.newton,
        newton_tol=args.newton_tol,
        with_recovery=not args.no_recovery,
        with_profiles=not args.no_profiles,
        with_monte_carlo_fisher=not args.no_monte_carlo,
        mc_replicates=args.mc_replicates,
        profile_grid=args.profile_grid,
        profile_newton=args.profile_newton,
        recovery_designs=(
            args.recovery_designs or [d.name for d in designs if d.primary]
        ),
        heavy_regimes=[regimes[0].name],
        checkpoint_path=str(out / 'results_partial.json'),
        resume=not args.no_resume,
    )
    res["wall_seconds"] = time.time() - t0
    res["gpu_memory"] = gpu_memory_report()
    res["gpu_memory_cap"] = cap
    decision = evaluate_decision_rule(res)
    figs = make_figures(res, out)
    paths = write_report(res, out, decision=decision, figures=figs)
    print(f"[verdict] {decision['verdict']}")
    print(f"[written] {paths['json']}  {paths['markdown']}")
    return 0


def cmd_slice(args) -> int:
    cap_gpu_memory(args.gpu_gib)
    from .synthetic_slice import run_synthetic_slice

    cfg = _cfg(args)
    rep = run_synthetic_slice(cfg=cfg, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_slice.json").write_text(json.dumps(rep.to_dict(), indent=1))
    for k, v in rep.criteria.items():
        if v.get("evaluable") is False:
            tag = "N/EVAL"
        else:
            tag = "PASS" if v["pass"] else "FAIL"
        print(f"  {tag:6s}  {k}")
    print(f"[written] {out/'synthetic_slice.json'}")
    return 0


def cmd_report(args) -> int:
    """Regenerate summary.md / figures from an existing results.json.

    Useful when the decision-rule *presentation* changes; the preregistered
    criteria in manifest.json are re-applied unchanged to the stored results.
    """
    out = Path(args.out)
    payload = json.loads((out / "results.json").read_text())
    res = payload.get("results", payload)
    decision = evaluate_decision_rule(res)
    figs = make_figures(res, out)
    paths = write_report(res, out, decision=decision, figures=figs)
    print(f"[verdict] {decision['verdict']}")
    print(f"[written] {paths['json']}  {paths['markdown']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser("scwbd.infer")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float64")
    p.add_argument("--gpu-gib", type=float, default=_default_gpu_gib(),
                   help="device-side CUDA allocation cap for this process "
                        "(default: $SCWBD_CUDA_MAX_GB, else 20)")
    p.add_argument("--epoch-seconds", type=float, default=6.0)
    p.add_argument("--n-epochs", type=int, default=32)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--replicates", type=int, default=96)
    b.add_argument("--mc-replicates", type=int, default=256)
    b.add_argument("--newton", type=int, default=4,
                   help="maximum Newton steps; the loop stops earlier once every "
                        "replicate is within --newton-tol of the optimum")
    b.add_argument("--newton-tol", type=float, default=0.05,
                   help="convergence tolerance, in posterior standard deviations")
    b.add_argument("--profile-grid", type=int, default=7)
    b.add_argument("--profile-newton", type=int, default=5)
    b.add_argument("--regimes", nargs="*", default=None)
    b.add_argument("--primary-only", action="store_true")
    b.add_argument("--no-recovery", action="store_true")
    b.add_argument("--no-profiles", action="store_true")
    b.add_argument("--no-monte-carlo", action="store_true")
    b.add_argument(
        "--recovery-designs", nargs="*", default=None,
        help="restrict the expensive MAP-recovery arm to these designs "
             "(default: every primary design)",
    )
    b.add_argument(
        "--no-resume", action="store_true",
        help="recompute designs already present in results_partial.json",
    )
    b.set_defaults(func=cmd_benchmark)
    s = sub.add_parser("slice")
    s.set_defaults(func=cmd_slice)
    r = sub.add_parser("report")
    r.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
