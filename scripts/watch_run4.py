"""Emit one line per event worth acting on during run 4. Silence means nothing crossed a line.

Armed after `scripts/launch_run4.sh`. It finds the training process by CONFIG
rather than by a PID passed in, so it survives a relaunch and cannot latch onto
a probe or a smoke -- the same mistake `health.sh` made on 2026-08-10, when it
paired run 4's stale log with a probe's live process and called it a hang.

Coverage is the point. A monitor that watches only the two things you are worried
about goes SILENT if the run crashes, and silence looks exactly like healthy. So
it emits on death and on failure signatures as well as on the trends.

What it watches, and why each:

* `real_bold_nll` RISING. ISSUE-016 killed the first launch: 1.99 -> 12.96 over
  400 steps while `eeg_nll` improved. The signature is the CLIMB, not a level, so
  a trend test fires before any threshold would. A hard alarm at 12 sits well
  below the 21.7 ISSUE-008 reached.
* `eeg_nll` rising. ISSUE-016's remedy was declined for this run, so a degrading
  BOLD term keeps feeding the shared trunk. If the damage spreads to EEG, that is
  the signal to stop -- and it is the risk the plan accepted explicitly.
* `gpu_reserved_gb` against the 80 GB cap (measured peak 59.95 at T5).
* MemAvailable on the 121.6 GB UNIFIED pool. This box has OOM'd.
* stage transitions, death, and CUDA/traceback signatures in the run log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = os.environ.get("WATCH_CONFIG", "configs/run4/scwbd-004.yaml")
JSONL = REPO / os.environ.get("WATCH_JSONL", "reports/training/scwbd-004_train.jsonl")
RUNLOG = REPO / os.environ.get("WATCH_RUNLOG", "reports/training/run004.log")

BOLD_ALARM = 12.0
TREND_RISE = 1.6          # last-N mean this many x the prior-N mean
EEG_ALARM = 4.0           # sustained eeg_nll well above its ~1.6 working range

#: Rows per trend window. `log_every` is 20, so a window of 20 ROWS is 400 STEPS
#: -- the monitor would have been slower than the thing it watches. The first
#: launch's EEG transient hit at step 120 and had fully recovered by 140; a test
#: that could not speak until step 400 is not a monitor, it is a post-mortem.
TREND_ROWS = 5

#: A SINGLE row above this is reported immediately, separately from the trend.
#: Both launches produced one large isolated EEG spike early in T1 (21.23 at step
#: 120, then 2.38; 34.41 at step 140) and both are consistent with one bad batch.
#: Worth seeing, not worth stopping for -- so it is labelled as a spike and the
#: trend test remains the thing that decides.
EEG_SPIKE = 10.0
BOLD_SPIKE = 8.0
GPU_CAP_GB = 80.0
GPU_WARN_GB = 72.0
MEM_FLOOR_GB = 12.0
MEM_WARN_GB = 16.0
GRACE_S = 900             # the trainer builds anatomy and datasets before step 1


def pids() -> list[int]:
    """PIDs training THIS config. Not `pgrep -f` on the module: that matches any run."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        if "scwbd.foundation.train" in line and CONFIG in line and "grep" not in line:
            try:
                found.append(int(line.split()[0]))
            except (ValueError, IndexError):
                pass
    return found


def rows() -> list[dict]:
    try:
        return [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    except Exception:
        return []


def mem_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 999.0


def trend(vals: list[float]) -> tuple[float, float] | None:
    n = TREND_ROWS
    if len(vals) < 2 * n:
        return None
    return sum(vals[-n:]) / n, sum(vals[-2 * n : -n]) / n


def main() -> int:
    started = time.time()
    seen_stage: str | None = None
    warned = {"mem": False, "bold": False, "eeg": False, "gpu": False}
    n_rows = 0
    ever_alive = False

    while True:
        alive = pids()
        if alive:
            ever_alive = True
        elif ever_alive or time.time() - started > GRACE_S:
            tail = ""
            try:
                tail = "\n".join(RUNLOG.read_text().splitlines()[-25:])
            except Exception:
                pass
            print(
                f"RUN4 GONE: no process training {CONFIG}. Last 25 lines of the run log:\n{tail}",
                flush=True,
            )
            return 0

        m = mem_gb()
        if m < MEM_FLOOR_GB:
            print(
                f"RUN4 MEMORY CRITICAL: MemAvailable {m:.1f} GB on a 121.6 GB UNIFIED "
                "pool. This box has OOM'd before.",
                flush=True,
            )
        elif m < MEM_WARN_GB and not warned["mem"]:
            print(f"RUN4 memory warning: MemAvailable {m:.1f} GB", flush=True)
            warned["mem"] = True

        r = rows()
        if len(r) > n_rows:
            n_rows = len(r)
            last = r[-1]

            st = last.get("stage")
            if st != seen_stage:
                seen_stage = st
                print(
                    f"RUN4 stage -> {st} at global_step {last.get('global_step')} "
                    f"(gpu_reserved {last.get('gpu_reserved_gb')} GB, MemAvailable {m:.1f} GB)",
                    flush=True,
                )

            g = last.get("gpu_reserved_gb")
            if isinstance(g, (int, float)) and g > GPU_WARN_GB and not warned["gpu"]:
                print(
                    f"RUN4 gpu_reserved {g} GB is within {GPU_CAP_GB - g:.1f} GB of the "
                    f"{GPU_CAP_GB} GB cap (measured peak was 59.95 at T5).",
                    flush=True,
                )
                warned["gpu"] = True

            # Isolated spikes, reported on the NEWEST row only so a past one is
            # not re-announced every minute.
            for key, lim, label in (
                ("eegmmidb_real_eeg_nll", EEG_SPIKE, "eeg_nll"),
                ("real_bold_nll", BOLD_SPIKE, "real_bold_nll"),
            ):
                v = last.get(key)
                if isinstance(v, (int, float)) and v > lim:
                    print(
                        f"RUN4 {label} SPIKE {v:.2f} at step {last.get('global_step')} "
                        f"(> {lim}). Both launches produced one isolated early-T1 EEG "
                        "spike that recovered; watch the next rows before concluding.",
                        flush=True,
                    )

            b = [x["real_bold_nll"] for x in r if isinstance(x.get("real_bold_nll"), (int, float))]
            t = trend(b)
            if t:
                recent, prior = t
                if recent > BOLD_ALARM:
                    print(
                        f"RUN4 real_bold_nll ALARM: last-10 mean {recent:.2f} > {BOLD_ALARM}. "
                        "ISSUE-016 killed the first launch at 12.96; ISSUE-008 reached 21.7.",
                        flush=True,
                    )
                elif prior > 0 and recent / prior >= TREND_RISE and not warned["bold"]:
                    print(
                        f"RUN4 real_bold_nll RISING: last-10 {recent:.2f} vs prior-10 "
                        f"{prior:.2f} ({recent / prior:.2f}x) at step {last.get('global_step')}. "
                        "EXPECTED under ISSUE-016; watch eeg_nll for spread.",
                        flush=True,
                    )
                    warned["bold"] = True

            e = [
                x["eegmmidb_real_eeg_nll"]
                for x in r
                if isinstance(x.get("eegmmidb_real_eeg_nll"), (int, float))
            ]
            te = trend(e)
            if te:
                recent, prior = te
                if recent > EEG_ALARM and not warned["eeg"]:
                    print(
                        f"RUN4 eeg_nll ALARM: last-10 mean {recent:.2f} > {EEG_ALARM}. The BOLD "
                        "degradation may be spreading to the shared trunk -- this is the risk "
                        "RUN4_LAUNCH_PLAN.md §6 accepted explicitly. Consider stopping.",
                        flush=True,
                    )
                    warned["eeg"] = True

        try:
            t = RUNLOG.read_text()[-4000:]
            for sig in ("Traceback", "OutOfMemoryError", "CUDA out of memory", "Killed"):
                if sig in t:
                    print(f"RUN4 FAILURE SIGNATURE in the run log: {sig!r}", flush=True)
                    break
        except Exception:
            pass

        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
