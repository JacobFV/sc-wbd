"""Bounded cost measurement for SC-WBD-004's BOLD path. NOT a training run.

ISSUE-008's fix rolls the neural clock for the duration a BOLD frame actually
covers: 500 steps per window at `bold_predict_frames: 2` and TR = 2 s, against
run 3's 8 for the whole window. Run 3's `data.batch: 8` was measured against the
old path and cannot be carried over -- so this runs a fixed, small number of
optimiser steps of ONE configuration and reports seconds/step and peak CUDA
reserve.

HANDOFF-004 and CLAUDE.md both say not to probe the scaling with a sweep; a
sweep that reached 8 sources x batch 32 took this machine down. One
configuration per invocation, and the caller decides what to do with the
number.

Two disciplines are built in rather than remembered:

* **Nothing touches production.** `run_name`, `out_dir` and `report_dir` are all
  redirected. `--out` moves checkpoints only -- the JSONL log is keyed by
  `run_name` -- which is how a scratch run appended to run 2's production log.
* **Contention is recorded, not assumed absent.** A number measured beside
  another job is not a measurement, and this box runs several agents. The
  harness samples competing processes throughout and reports the maximum, so a
  contaminated run says so in its own output instead of being quoted clean.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())

from scwbd.foundation.config import load_config  # noqa: E402
from scwbd.foundation.train import FoundationTrainer  # noqa: E402


def _free_gb() -> dict[str, float]:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()
    row = [l for l in out if l.startswith("Mem:")][0].split()
    return {"total_gb": int(row[1]) / 1024, "used_gb": int(row[2]) / 1024, "available_gb": int(row[6]) / 1024}


class _Contention:
    """Samples other heavy processes for the life of the measurement.

    `ps -eo pid,args | grep "[p]attern"` rather than `pgrep -f`, which matches
    the shell running it -- that killed a turn five separate times.
    """

    PATTERNS = ("pytest", "train.py", "evaluate.py", "measure_run4_cost.py")

    def __init__(self) -> None:
        self.max_others = 0
        self.samples = 0
        self.witnessed: set[str] = set()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _snapshot(self) -> list[str]:
        """Competing processes, excluding this measurement's own process group.

        Excluding only `os.getpid()` counted this run's own DataLoader workers
        as contention -- eight to ten of them -- and reported
        `max_competing_processes: 16` for a box carrying one foreign job. A
        contention meter that fires on its own children makes every measurement
        look dirty, which is the same as making none of them look dirty.
        """
        mine = os.getpgid(0)
        out = subprocess.run(["ps", "-eo", "pid,pgid,args"], capture_output=True, text=True).stdout
        hits = []
        for line in out.splitlines()[1:]:
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            _, pgid, args = parts
            if "ps -eo" in args:
                continue
            try:
                if int(pgid) == mine:
                    continue
            except ValueError:
                continue
            if any(p in args for p in self.PATTERNS):
                hits.append(args[:90])
        return hits

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            hits = self._snapshot()
            self.samples += 1
            self.max_others = max(self.max_others, len(hits))
            self.witnessed.update(hits)

    def __enter__(self) -> "_Contention":
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=10)

    def report(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "max_competing_processes": self.max_others,
            "clean": self.max_others == 0,
            "witnessed": sorted(self.witnessed)[:10],
            "note": (
                "clean=false means another job ran during the timed steps. The "
                "step time is then an upper bound and the peak reserve is still "
                "this process's own -- torch.cuda.max_memory_reserved is "
                "per-process. Quote neither as a clean measurement."
            ),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run4/scwbd-004.yaml")
    ap.add_argument("--stage", default="T1_measured_founding")
    ap.add_argument("--steps", type=int, default=23)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--bold-frames", type=int, default=None)
    ap.add_argument("--bold-every", type=int, default=None)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--out", default="reports/run4_cost")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg.train.run_name = f"scwbd-004-cost-{a.tag}"
    cfg.train.out_dir = f"checkpoints/scwbd-004-cost-{a.tag}"
    cfg.train.report_dir = a.out
    cfg.train.resume = False
    cfg.train.max_wall_seconds = 3600
    if a.batch is not None:
        cfg.data.batch = a.batch
    if a.bold_frames is not None:
        cfg.model.bold_predict_frames = a.bold_frames
    if a.bold_every is not None:
        cfg.model.bold_every = a.bold_every

    stage = next(s for s in cfg.train.stages if s.name == a.stage)
    stage.steps = a.steps
    stage.warmup = max(1, a.steps // 4)
    stage.log_every = 1
    stage.ckpt_every = 10**9
    cfg.train.stages = [stage]

    print(json.dumps({"before": _free_gb()}, indent=2), flush=True)
    torch.cuda.reset_peak_memory_stats()

    with _Contention() as watch:
        t = FoundationTrainer(cfg, resume=False)
        t.build_data()
        build_reserved = torch.cuda.max_memory_reserved() / 1024**3

        t0 = time.time()
        rep = t.run_stage(stage)
        wall = time.time() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_res = torch.cuda.max_memory_reserved() / 1024**3

    # Per-step seconds from consecutive `wall_s` in the trainer's own rows, so
    # it is the same clock the run reports rather than a second one that can
    # disagree with it. log_every is 1 here, so every step has a row.
    rows = sorted(
        (r for r in t.history if r.get("stage") == a.stage), key=lambda r: int(r["step"])
    )
    times = [
        (float(b["wall_s"]) - float(x["wall_s"])) / (int(b["step"]) - int(x["step"]))
        for x, b in zip(rows, rows[1:])
        if int(b["step"]) > int(x["step"])
    ]
    timed = times[a.warmup:] if len(times) > a.warmup else times

    result = {
        "config": a.config,
        "stage": a.stage,
        "batch": cfg.data.batch,
        "bold_predict_frames": cfg.model.bold_predict_frames,
        "bold_every": cfg.model.bold_every,
        "hemo_ratio": cfg.model.hemo_ratio,
        "dt_model": cfg.model.dt_model,
        "steps_run": rep.get("steps"),
        "warmup_discarded": a.warmup,
        "n_timed_steps": len(timed),
        "seconds_per_step_mean": statistics.fmean(timed) if timed else None,
        "seconds_per_step_median": statistics.median(timed) if timed else None,
        "seconds_per_step_min": min(timed) if timed else None,
        "seconds_per_step_max": max(timed) if timed else None,
        "wall_seconds_total": wall,
        "peak_cuda_allocated_gb": peak_alloc,
        "peak_cuda_reserved_gb": peak_res,
        "reserved_after_build_gb": build_reserved,
        "cuda_reserve_cap_gb": cfg.train.cuda_reserve_gb,
        "contention": watch.report(),
        "host_after": _free_gb(),
        "loss_keys": sorted(rows[-1].keys()) if rows else [],
        "last_row": rows[-1] if rows else None,
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"cost_{a.tag}.json"
    p.write_text(json.dumps(result, indent=2, default=str))
    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "last_row"}, indent=2, default=str))
    print(f"written to {p}")


if __name__ == "__main__":
    main()
