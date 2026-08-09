"""Generate ``reports/training/``: curves, corpus statistics, calibration plots, and
an explicit statement of what SC-WBD-001-beta cannot do.

Deliverable 5.  Every figure is generated from the run's own JSONL log and the
evaluation JSON -- nothing is typed in by hand, so a number in the report and a
number in the artifact cannot drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["build_report", "CANNOT_DO", "plot_training_curves", "plot_calibration", "plot_corpus"]

#: The required explicit statement of limits.  Written into both the report and
#: the checkpoint's ClaimManifest, so the artifact carries its own limits.
CANNOT_DO: tuple[str, ...] = (
    "It cannot predict any specific person's brain activity, state, or behaviour. It is a "
    "population model; the individualization machinery is fitted, not validated, and no "
    "individual-level generalisation claim is made.",
    "It cannot localise sources. The EEG head uses an analytic spherical lead field with no "
    "individual head model, no measured electrode positions and no tissue segmentation.",
    "It cannot establish that any mechanism is neurally realized. The learned operator is the "
    "equal-capacity control for exactly that comparison, and matching a mechanistic family's "
    "trajectories is not evidence for that family.",
    "It cannot be used for any clinical, diagnostic or therapeutic purpose, and it emits no "
    "stimulation protocol, joint command, trajectory or actuation authority of any kind.",
    "It cannot claim anatomical validity for anything trained against the synthetic fallback "
    "connectome. A synthetic connectome carries no biological information.",
    "It cannot treat its simulated corpus as evidence about brains. Every simulated trajectory is "
    "simulator-conditioned evidence and enters the mixture with the 'prior' role only.",
    "It cannot measure consciousness. No integrated-information quantity is estimated and no "
    "consciousness ground truth exists to validate one against.",
    "It cannot generalise beyond the montage, task set and population of its measured sources: "
    "one site, one 64-channel montage, resting/motor-imagery EEG from 109 adults.",
    "It cannot resolve anything above ~45 Hz, below ~0.5 Hz, or faster than its 8 ms model step; "
    "the corpus itself is stored at 125 Hz after an 8 ms box-average.",
    "Its haemodynamic head is fitted to simulated neural drive only: no BOLD, fMRI or fNIRS "
    "measurement entered training, so the BOLD output is a structural placeholder, not a "
    "predictor of measured haemodynamics.",
    "It cannot infer the parameters it is built to infer. The amortized posterior explains no "
    "variance in any of them on held-out simulated data, and it is well calibrated because it "
    "returns the prior -- an uninformative posterior is calibrated by construction, so the "
    "calibration curves qualify nothing. ISSUE-012.",
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    return plt


# ----------------------------------------------------------------------
def plot_training_curves(records: Sequence[Mapping[str, Any]], out: Path) -> list[str]:
    if not records:
        return []
    plt = _mpl()
    keys = ["loss", "sim_forecast_nll", "npe_loss", "real_eeg_nll", "rho", "kl"]
    have = [k for k in keys if any(k in r for k in [k] for r in records)]
    have = [k for k in keys if any(k in r for r in records)]
    stages = []
    seen = set()
    for r in records:
        s = r.get("stage")
        if s not in seen:
            seen.add(s)
            stages.append((s, r.get("global_step", 0)))
    n = len(have)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(10, 2.4 * math.ceil(n / 2)), squeeze=False)
    for ax, k in zip(axes.ravel(), have):
        xs = [r["global_step"] for r in records if k in r]
        ys = [r[k] for r in records if k in r]
        ax.plot(xs, ys, lw=0.9)
        if ys and min(ys) > 0 and max(ys) / max(min(ys), 1e-12) > 50:
            ax.set_yscale("log")
        for s, gs in stages:
            ax.axvline(gs, color="0.6", lw=0.6, ls=":")
        ax.set_title(k)
        ax.set_xlabel("global step")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("SC-WBD-001-beta training (dotted lines = stage boundaries)", y=1.0)
    p = out / "training_curves.png"
    fig.savefig(p)
    plt.close(fig)
    return [p.name]


def plot_calibration(cal: Mapping[str, Any], out: Path) -> list[str]:
    if not cal or not cal.get("available"):
        return []
    plt = _mpl()
    names = cal.get("param_names", [])
    ranks = np.asarray(cal.get("sbc_ranks", []), dtype=float)
    files: list[str] = []
    if ranks.size:
        P = ranks.shape[1]
        nb = int(cal.get("sbc_n_bins", 256))
        fig, axes = plt.subplots(1, P, figsize=(2.1 * P, 2.3), squeeze=False)
        for j, ax in enumerate(axes[0]):
            ax.hist(ranks[:, j], bins=20, range=(0, nb), color="#4477aa", edgecolor="white", lw=0.4)
            exp = ranks.shape[0] / 20
            ax.axhline(exp, color="#cc6677", lw=1.0)
            ks = cal.get("sbc_ks_pvalue", [])
            ax.set_title(f"{names[j] if j < len(names) else j}\nKS p={ks[j]:.3f}" if j < len(ks) else str(j), fontsize=8)
            ax.set_yticks([])
        fig.suptitle("SBC rank histograms (flat = calibrated; U = over-confident)", y=1.06)
        p = out / "sbc_ranks.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p.name)
    cov = cal.get("coverage", {})
    if cov.get("levels"):
        fig, ax = plt.subplots(figsize=(4, 3.6))
        lv = np.asarray(cov["levels"], dtype=float)
        ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="nominal")
        ax.plot(lv, cov["coverage_mean"], "o-", lw=1.2, label="marginal mean")
        if cov.get("coverage_joint"):
            ax.plot(lv, cov["coverage_joint"], "s-", lw=1.2, label="joint")
        ax.set_xlabel("nominal credible level")
        ax.set_ylabel("realised coverage")
        ax.set_title(f"Expected coverage (MAE={cov.get('coverage_mae', float('nan')):.3f})")
        ax.legend(fontsize=7)
        p = out / "expected_coverage.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p.name)
    return files


def plot_corpus(index_paths: Iterable[str | Path], out: Path) -> tuple[list[str], dict[str, Any]]:
    plt = _mpl()
    stats: dict[str, Any] = {"tiers": {}}
    shards_all = []
    for ip in index_paths:
        p = Path(ip)
        if not p.exists():
            continue
        idx = json.loads(p.read_text())
        tier = idx.get("spec", {}).get("tier", p.stem)
        by_backend: dict[str, float] = {}
        for s in idx.get("shards", []):
            by_backend[s["backend"]] = by_backend.get(s["backend"], 0.0) + s["traj_seconds"]
            shards_all.append({**s, "tier": tier})
        stats["tiers"][tier] = {
            "trajectories": idx.get("total_trajectories", 0),
            "trajectory_seconds": idx.get("total_trajectory_seconds", 0.0),
            "wall_seconds": idx.get("wall_seconds", 0.0),
            "gigabytes": sum(s["bytes"] for s in idx.get("shards", [])) / 1e9,
            "by_backend_trajectory_seconds": by_backend,
            "fs_hz": idx.get("spec", {}).get("fs_hz"),
            "duration_s": idx.get("spec", {}).get("duration_s"),
            "anatomy_provenance": idx.get("anatomy", {}).get("provenance"),
            "n_shards": len(idx.get("shards", [])),
            "delay_quantisation_rmse_s_mean": float(
                np.mean([s["delay_quantisation_rmse_s"] for s in idx.get("shards", [])] or [0.0])
            ),
        }
    stats["total_trajectory_seconds"] = sum(v["trajectory_seconds"] for v in stats["tiers"].values())
    stats["total_trajectories"] = sum(v["trajectories"] for v in stats["tiers"].values())
    stats["total_gigabytes"] = sum(v["gigabytes"] for v in stats["tiers"].values())
    stats["total_wall_seconds"] = sum(v["wall_seconds"] for v in stats["tiers"].values())
    if not shards_all:
        return [], stats
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    agg: dict[str, float] = {}
    for s in shards_all:
        agg[s["backend"]] = agg.get(s["backend"], 0.0) + s["traj_seconds"]
    axes[0].bar(list(agg), [v / 1e3 for v in agg.values()], color="#4477aa")
    axes[0].set_ylabel("kilo trajectory-seconds")
    axes[0].set_title("corpus by backend family")
    axes[0].tick_params(axis="x", rotation=35, labelsize=7)
    vel = [s["velocity_m_s"] for s in shards_all]
    axes[1].hist(vel, bins=20, color="#88ccee", edgecolor="white")
    axes[1].set_xlabel("conduction velocity (m/s)")
    axes[1].set_title("velocity sweep")
    rms = [s["delay_quantisation_rmse_s"] * 1e3 for s in shards_all]
    axes[2].hist(rms, bins=20, color="#ddcc77", edgecolor="white")
    axes[2].set_xlabel("delay quantisation RMSE (ms)")
    axes[2].set_title("reported numerical budget")
    fig.suptitle("Simulated corpus (SIMULATOR-CONDITIONED EVIDENCE)", y=1.04)
    p = out / "corpus_statistics.png"
    fig.savefig(p)
    plt.close(fig)
    return [p.name], stats


def plot_holdout(hold: Mapping[str, Any], out: Path) -> list[str]:
    if not hold or not hold.get("available"):
        return []
    plt = _mpl()
    rows = [(k, v) for k, v in hold["results"].items() if "nll_per_sample" in v]
    rows.sort(key=lambda kv: kv[1]["nll_per_sample"])
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 1.4))
    ys = np.arange(len(rows))
    pts = [r[1]["nll_per_sample"] for r in rows]
    los = [r[1]["nll_ci95"][0] for r in rows]
    his = [r[1]["nll_ci95"][1] for r in rows]
    colors = ["#cc6677" if r[0] == "scwbd_001_beta" else "#4477aa" for r in rows]
    ax.errorbar(
        pts, ys, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
        fmt="none", ecolor="0.5", capsize=3, lw=1,
    )
    ax.scatter(pts, ys, c=colors, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("held-out EEG NLL (nats / channel / sample), lower is better")
    ax.set_title("Participant-level holdout, participant-clustered 95% CI")
    p = out / "real_eeg_holdout.png"
    fig.savefig(p)
    plt.close(fig)
    return [p.name]


# ----------------------------------------------------------------------
def build_report(
    *,
    report_dir: str | Path = "reports/training",
    run_name: str = "scwbd-001-beta",
    corpus_indexes: Sequence[str] = ("/data/scwbd/sim_corpus/index_fast.json", "/data/scwbd/sim_corpus/index_slow.json"),
    evaluation: str | Path | None = None,
    summary: str | Path | None = None,
    mixture_glob: str = "mixture_*.json",
) -> Path:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(out / f"{run_name}_train.jsonl")
    summ = json.loads(Path(summary).read_text()) if summary and Path(summary).exists() else (
        json.loads((out / f"{run_name}_summary.json").read_text()) if (out / f"{run_name}_summary.json").exists() else {}
    )
    ev_path = Path(evaluation) if evaluation else out / "evaluation.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    figs: list[str] = []
    figs += plot_training_curves(records, out)
    corpus_figs, corpus_stats = plot_corpus(corpus_indexes, out)
    figs += corpus_figs
    figs += plot_calibration(ev.get("posterior_calibration", {}), out)
    figs += plot_holdout(ev.get("real_eeg_holdout", {}), out)

    mixes = []
    for f in sorted(out.glob(mixture_glob)):
        try:
            mixes.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue

    md: list[str] = []
    A = md.append
    A("# SC-WBD-001-beta — training report")
    A("")
    A(f"- git: `{summ.get('git_sha', ev.get('git_sha', 'unknown'))}`")
    env = summ.get("environment", ev.get("environment", {}))
    if env:
        A(f"- device: `{env.get('device')}` (sm {env.get('capability')}, {env.get('total_memory_gb')} GB unified)")
        A(f"- torch `{env.get('torch')}`, python `{env.get('python')}`")
    A("")
    A("> SC-WBD-001-beta is a **population/general adult** model. It is not a validated model of "
      "any person, not a clinical device, and not evidence that any admitted operator is neurally "
      "realized. Simulated trajectories are simulator-conditioned evidence throughout.")
    A("")

    # -- parameters
    pr = summ.get("model_parameters") or ev.get("n_parameters") or {}
    if pr:
        A("## Parameter count")
        A("")
        A("| module | parameters |")
        A("|---|---:|")
        for k, v in sorted(pr.items(), key=lambda kv: -kv[1]):
            A(f"| `{k}` | {v:,} |")
        if summ.get("posterior_parameters"):
            A(f"| `amortized_posterior` | {summ['posterior_parameters']:,} |")
        A("")

    # -- corpus
    A("## Simulated corpus (Phase 2)")
    A("")
    A(f"- **{corpus_stats.get('total_trajectory_seconds', 0):,.0f} trajectory-seconds** "
      f"over {corpus_stats.get('total_trajectories', 0):,} trajectories, "
      f"{corpus_stats.get('total_gigabytes', 0):.1f} GB on disk, "
      f"{corpus_stats.get('total_wall_seconds', 0)/3600:.2f} h of GPU wall clock.")
    for tier, v in corpus_stats.get("tiers", {}).items():
        A(f"- tier `{tier}`: {v['trajectory_seconds']:,.0f} traj-s, {v['trajectories']:,} trajectories, "
          f"{v['duration_s']} s each at {v['fs_hz']} Hz, {v['gigabytes']:.1f} GB, "
          f"delay-quantisation RMSE {v['delay_quantisation_rmse_s_mean']*1e3:.2f} ms, "
          f"anatomy provenance `{v['anatomy_provenance']}`.")
    A("")

    # -- throughput
    if records:
        last = [r for r in records if "traj_s_per_s" in r]
        if last:
            A("## Achieved training throughput (GB10)")
            A("")
            by_stage: dict[str, list[float]] = {}
            for r in last:
                by_stage.setdefault(r["stage"], []).append(r["traj_s_per_s"])
            A("| stage | steps logged | trajectory-seconds/s (median) |")
            A("|---|---:|---:|")
            for s, v in by_stage.items():
                A(f"| {s} | {len(v)} | {np.median(v):.1f} |")
            A("")
    if summ.get("stages"):
        A("| stage | steps | wall (s) | steps/s | best normalised loss |")
        A("|---|---:|---:|---:|---:|")
        for s in summ["stages"]:
            if s.get("skipped"):
                continue
            A(f"| {s['stage']} | {s['steps']} | {s['wall_seconds']:.0f} | {s['steps_per_second']:.2f} | {s['best_loss']:.4f} |")
        A("")

    # -- posterior calibration
    cal = ev.get("posterior_calibration", {})
    if cal.get("available"):
        A("## Amortized posterior calibration (simulator-conditioned)")
        A("")
        A("| parameter | SBC KS p | posterior R^2 | RMSE | z-sd (1.0 = honest width) |")
        A("|---|---:|---:|---:|---:|")
        for i, nm in enumerate(cal.get("param_names", [])):
            A(f"| `{nm}` | {cal['sbc_ks_pvalue'][i]:.3f} | {cal['posterior_r2'][i]:.3f} | "
              f"{cal['posterior_rmse'][i]:.3f} | {cal['posterior_z_sd'][i]:.2f} |")
        A("")
        A(f"- expected-coverage MAE (marginal): **{cal['coverage_mae']:.3f}**; "
          f"joint: {cal['coverage'].get('coverage_mae_joint', float('nan')):.3f}")
        # The verdict goes above the calibration line, not below it: a reader who
        # stops at "coverage MAE 0.021" has read a pass, and on run 3 that pass
        # was produced by a posterior that returns the prior.  ISSUE-012.
        if cal.get("informativeness_note"):
            A(f"- **{cal['informativeness_note']}**")
        A(f"- {cal['note']}")
        A("")

    # -- held-out real EEG
    hold = ev.get("real_eeg_holdout", {})
    A("## Held-out real EEG (Phase 4)")
    A("")
    if hold.get("available"):
        A(f"- {hold['n_test_windows']} windows from {hold['n_test_participants']} held-out participants; "
          f"train {hold['n_train_windows']} windows / {hold['n_train_participants']} participants.")
        A(f"- metric: {hold['metric']}")
        A("")
        A("| model | NLL | 95% CI (participant-clustered) | MSE | parameters |")
        A("|---|---:|---|---:|---:|")
        for k, v in sorted(hold["results"].items(), key=lambda kv: kv[1].get("nll_per_sample", 1e30)):
            if "nll_per_sample" not in v:
                A(f"| {k} | — | ERROR: {v.get('error')} | — | — |")
                continue
            ci = v["nll_ci95"]
            A(f"| {'**'+k+'**' if k=='scwbd_001_beta' else k} | {v['nll_per_sample']:.4f} | "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] | {v['mse']:.4f} | {v['n_parameters']:,} |")
        A("")
        A(f"**Verdict:** {hold['verdict']}.")
        if hold.get("scwbd_beaten_by"):
            A(f"**Beaten by:** {', '.join(hold['scwbd_beaten_by'])}. This is reported, not tuned away.")
        A("")
        A(f"_{hold['interpretation']}_")
    else:
        A(f"- not available: {hold.get('reason', 'unknown')}")
    A("")

    # -- backends
    bc = ev.get("backend_comparison", {})
    if bc:
        A("## Backend comparison (held-out simulated forecast NLL)")
        A("")
        A("| backend family | forecast NLL | n windows |")
        A("|---|---:|---:|")
        for k, v in sorted((bc.get("per_backend_nll") or {}).items(), key=lambda kv: (kv[1] is None, kv[1])):
            A(f"| {k} | {'—' if v is None else f'{v:.4f}'} | {bc['per_backend_n'].get(k, 0)} |")
        A("")
        A(f"_{bc.get('note', '')}_")
        A("")

    # -- mixture / conflict
    if mixes:
        A("## Mixture weights, per-source contribution and gradient conflict")
        A("")
        m = mixes[-1]
        A("| source | reliability weight | share of the objective |")
        A("|---|---:|---:|")
        w = m["mixture_weights"]["weights"]
        contrib = m["per_source_contribution"]
        for k in sorted(set(w) | set(contrib)):
            A(f"| `{k}` | {w.get(k, 0):.4f} | {contrib.get(k, 0):.4f} |")
        A("")
        A(f"_{m['mixture_weights']['note']}_")
        A("")
        pairs = m.get("gradient_conflict", {}).get("pairs", [])
        if pairs:
            A("### Gradient conflict by module and source (most opposed first)")
            A("")
            A("| source A | source B | module | mean cosine | min | fraction negative | n |")
            A("|---|---|---|---:|---:|---:|---:|")
            for r in pairs[:20]:
                A(f"| {r['source_a']} | {r['source_b']} | `{r['module']}` | {r['mean_cosine']:.3f} | "
                  f"{r['min_cosine']:.3f} | {r['frac_negative']:.2f} | {r['n']} |")
            A("")
        if m.get("conflict_decisions"):
            A("### Conflict actions taken (adapter / partial pooling / freeze / reject)")
            A("")
            for d in m["conflict_decisions"]:
                A(f"- `{d['pair'][0]}` vs `{d['pair'][1]}` on `{d['module']}`: **{d['action']}** "
                  f"(yielding: `{d['yielding_source']}`, mean cosine {d['mean_cosine']:.3f})")
            A("")
        else:
            A("_No source pair crossed the incompatibility threshold; no adapters, freezes or "
              "rejections were triggered._")
            A("")
        audit = m.get("gradient_permission_audit", {})
        if audit:
            A("### Gradient permission audit (A_k)")
            A("")
            A("| source | permitted parameter tensors | blocked |")
            A("|---|---:|---:|")
            for k, v in sorted(audit.items()):
                A(f"| `{k}` | {v['n_permitted']} | {v['n_blocked']} |")
            A("")

    # -- source ablation
    sa = ev.get("source_ablation")
    if sa:
        A("## Per-source contribution / negative transfer")
        A("")
        A("| removed family | validation NLL | delta vs all sources |")
        A("|---|---:|---:|")
        A(f"| (none) | {sa['with_all_sources']:.4f} | — |")
        for f in sa["families"]:
            A(f"| `{f}` | {sa[f'without_{f}']:.4f} | {sa[f'delta_{f}']:+.4f} |")
        A("")
        A(f"_{sa['interpretation']}_")
        if sa.get("negative_transfer"):
            A(f"**Negative transfer detected for: {', '.join(sa['negative_transfer'])}.**")
        A("")

    # -- figures
    if figs:
        A("## Figures")
        A("")
        for f in figs:
            A(f"![{f}]({f})")
            A("")

    # -- limits
    A("## What SC-WBD-001-beta cannot do")
    A("")
    for line in CANNOT_DO:
        A(f"- {line}")
    A("")
    A("A gate that fails is a result, not a bug. Negative results above are reported as found and "
      "were not tuned against. The claim gates themselves are owned by agent J; nothing in this "
      "report decides whether one passes.")

    p = out / "REPORT.md"
    p.write_text("\n".join(md))
    (out / "corpus_statistics.json").write_text(json.dumps(corpus_stats, indent=2))
    return p


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="build reports/training")
    ap.add_argument("--report-dir", default="reports/training")
    ap.add_argument("--run-name", default="scwbd-001-beta")
    ap.add_argument("--evaluation", default=None)
    a = ap.parse_args(argv)
    p = build_report(report_dir=a.report_dir, run_name=a.run_name, evaluation=a.evaluation)
    print(f"wrote {p}")


if __name__ == "__main__":  # pragma: no cover
    main()
