#!/usr/bin/env python
"""Regenerate the LR-rescale comparison from the raw logs.

For 🛡️ Popper. Do not take the table in `rescale_adjudication_packet.md` on
trust -- run this and check it reproduces.

The pre-committed metric is `sim_forecast_nll` at matched steps. The composite
`loss` is emitted alongside it only so its disagreement is visible; it is **not**
admissible for the verdict (see decorative_guards.md row 7).

Usage:
    python reports/training/compare_rescale.py \
        --original <archived superseded jsonl> \
        --rescaled reports/training/scwbd-001-beta_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC = "sim_forecast_nll"  # pre-committed at commit a52ccf2


def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "step" in d and METRIC in d:
            rows[int(d["step"])] = d
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, type=Path, help="LR 6.0e-4 run (7f18528)")
    ap.add_argument("--rescaled", required=True, type=Path, help="LR 3.46e-4 run (4be98fc)")
    a = ap.parse_args()

    o, n = load(a.original), load(a.rescaled)
    steps = sorted(set(o) & set(n))
    if not steps:
        print("no matched steps; cannot compare")
        return 2

    print(f"metric: {METRIC}   (pre-committed; composite loss shown but inadmissible)")
    print(f"matched steps: {len(steps)}  range {steps[0]}..{steps[-1]}")
    print()
    print(f"{'step':>6} | {'orig':>9} {'rescaled':>9} {'delta':>9} | {'orig loss':>9} {'resc loss':>9}")
    print("-" * 72)

    worse = better = 0
    for s in steps:
        om, nm = float(o[s][METRIC]), float(n[s][METRIC])
        d = nm - om
        if s > 20:  # skip the warmup transient where schedules differ most
            better += d < 0
            worse += d > 0
        print(
            f"{s:>6} | {om:9.3f} {nm:9.3f} {d:+9.3f} | "
            f"{float(o[s]['loss']):9.4f} {float(n[s]['loss']):9.4f}"
        )

    print()
    print(f"excluding steps <= 20 (warmup transient):")
    print(f"  rescaled better at {better} matched steps, worse at {worse}")
    tot = better + worse
    if tot:
        # SIGNED, not absolute. Reporting mean|difference| counts a step where
        # the treatment was BETTER as though it were worse, inflating the figure
        # (2.21% vs the correct 1.72% on this data). An absolute value discards
        # the sign that carries the direction of the effect -- which is the
        # entire quantity of interest in a treatment/control comparison.
        rel = [
            (float(n[s][METRIC]) - float(o[s][METRIC])) / max(abs(float(o[s][METRIC])), 1e-9)
            for s in steps if s > 20
        ]
        print(f"  signed mean relative difference: {sum(rel) / len(rel) * 100:+.2f}%"
              "   (positive = rescaled worse)")
        print(f"  largest single |difference|:     {max(abs(x) for x in rel) * 100:.2f}%")
        print(f"  steps where rescaled was better: "
              f"{[s for s, x in zip([s for s in steps if s > 20], rel) if x < 0]}")
    print()
    print("NOTE: the superseded run stopped at step 260, so the pre-committed")
    print("      end-of-Stage-I (step 900) comparison CANNOT be evaluated on this")
    print("      data. Whether a verdict on the available prefix is admissible is")
    print("      Popper's call, not the author's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
