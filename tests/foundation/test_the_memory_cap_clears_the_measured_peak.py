"""``cuda_reserve_gb`` must clear the measured peak, on a UNIFIED pool.

Run 4's cap was 56 GB, fitted to a cost run that measured T1's loss set only.
The config said so in as many words -- *"T4 admits the simulator as well and is
not this loss set ... a planning number, not a measurement of the run"* -- and
the pre-launch smoke then killed T4 at that cap, wanting 352 MiB more. In a real
run that is ~5,400 steps and ~14 hours in.

Re-measured across all six stages, the peak is **T5 at 59.95 GB**, not T4's
57.98. A cap fitted to T4 would have died one stage later, which is the same
mistake twice.

Two properties are pinned here, and they pull in opposite directions:

* the cap must exceed the measured peak with real headroom, or the run dies
  partway through;
* the cap must leave the HOST a substantial share, because the GB10's ~121.6 GB
  is ONE UNIFIED POOL. ``systemd MemoryMax`` does not bound CUDA, and reporting
  the pool as "RAM plus GPU" is what OOM'd this machine.

The cap is the only thing bounding the caching allocator, so "just raise it" is
not free and "never raise it" is not safe either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.foundation.config import load_config

REPO = Path(__file__).resolve().parents[2]

#: Measured on 2026-08-10, one smoke walking all six stages at 4 steps each,
#: `gpu_reserved_gb` at each stage's first step. Raw log in reports/RUN4.md.
MEASURED_PEAK_GB = {
    "T1_measured_founding": 47.41,
    "T2_calibration": 49.17,
    "T3_population_prior": 49.17,
    "T4_simulator": 57.98,
    "T5_measured_return": 59.95,  # the peak, and NOT the stage anyone expected
    "T6_individual": 59.95,
}

#: The machine, not a guess: `MemTotal` reads 121.6 GB and the pool is unified.
UNIFIED_POOL_GB = 121.6

#: Host share that must survive the cap. The measurement itself ran with a
#: watchdog armed at 10 GB `MemAvailable` and never came close to it.
MIN_HOST_GB = 30.0


def test_run4s_cap_clears_the_measured_peak() -> None:
    cfg = load_config(str(REPO / "configs/run4/scwbd-004.yaml"))
    cap = cfg.train.cuda_reserve_gb
    peak = max(MEASURED_PEAK_GB.values())
    assert cap > peak, (
        f"cuda_reserve_gb={cap} is at or below the measured peak {peak} GB "
        f"({max(MEASURED_PEAK_GB, key=MEASURED_PEAK_GB.get)}). The run would die at "
        "that stage. This is exactly how 56 GB failed: it was fitted to T1."
    )
    headroom = (cap - peak) / peak
    assert headroom >= 0.15, (
        f"cuda_reserve_gb={cap} leaves only {headroom:.0%} over the measured peak "
        f"{peak} GB. Those figures are step-1 readings from a 4-step-per-stage smoke "
        "and reserve drifts upward across thousands of steps, so a thin margin is a "
        "late failure rather than an early one."
    )


def test_the_cap_leaves_the_host_a_share_of_the_unified_pool() -> None:
    """Raising the ceiling is bounded from above, not only from below."""
    cap = load_config(str(REPO / "configs/run4/scwbd-004.yaml")).train.cuda_reserve_gb
    host = UNIFIED_POOL_GB - cap
    assert host >= MIN_HOST_GB, (
        f"cuda_reserve_gb={cap} leaves the host {host:.1f} GB of the {UNIFIED_POOL_GB} GB "
        "UNIFIED pool. GPU and host share one pool here; systemd MemoryMax does not "
        "bound CUDA, and treating the pool as 'RAM plus GPU' is what OOM'd this box. "
        "The dataloader workers, the OS and anything else on the machine live in what "
        "is left."
    )


def test_the_cap_is_still_a_cap() -> None:
    """``0`` disables the ceiling entirely, which is not a fix for an OOM."""
    cap = load_config(str(REPO / "configs/run4/scwbd-004.yaml")).train.cuda_reserve_gb
    assert cap and cap > 0, (
        "cuda_reserve_gb is 0 or unset, which DISABLES the cap. It is the only thing "
        "bounding the caching allocator on this box -- it reserves freed blocks rather "
        "than returning them -- and it is what stopped the two-frame BOLD arm from "
        "taking the machine down. Raise it against a measurement; do not remove it."
    )


@pytest.mark.parametrize("stage", sorted(MEASURED_PEAK_GB))
def test_every_stage_was_measured_not_just_the_convenient_one(stage: str) -> None:
    """All six stages appear in the record backing the cap.

    The blocker existed because the cost run measured T1 and the cap was set from
    it. If a stage is added later and never measured, this fails and names it
    rather than the run finding out at hour fourteen.
    """
    cfg = load_config(str(REPO / "configs/run4/scwbd-004.yaml"))
    names = [s.name for s in cfg.train.stages]
    assert stage in names, f"{stage} is in the measured record but not in the config"
    missing = [n for n in names if n not in MEASURED_PEAK_GB]
    assert not missing, (
        f"stage(s) {missing} have no measured peak. The 56 GB cap was set from T1 "
        "alone and T4 died at it; the peak turned out to be T5. Measure the new "
        "stage with a smoke before trusting the cap."
    )
