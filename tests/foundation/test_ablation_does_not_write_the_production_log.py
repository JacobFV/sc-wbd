"""The ablation retrains, so it must not log into the run's own transcript.

`FoundationTrainer.logger` is keyed by `cfg.train.run_name`, and
`source_ablation` retrains one arm per source family on the same trainer. So
every arm appended to the production log. It did: `make release-003-ablate` put
a `global_step=1` row on the end of run 3's completed 13,400-step record, and
`make health-run3` immediately reported *"log ends at global_step=1 but the
checkpoint records 13400"*.

`short_train` sets `log_every = 10**9`, which is why the leak was one row per
arm rather than hundreds — step 1 logs regardless. A bound on the damage is not
the same as not doing it. CLAUDE.md carries this trap by name (*"`--out` moves
checkpoints, not logs. Logs are keyed by `train.run_name`, so a scratch run
appends to the production log"*) because losing part of run 2's log cost real,
unrecoverable data.

The check is on the file the logger points at, not on the rows it happens to
write: a future change to `log_every` must not be able to reopen this.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class _StubLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def log(self, **kw: object) -> None:  # pragma: no cover - never called here
        raise AssertionError("the stub production logger was written to")


class _StubTrainer:
    """Only the attributes `source_ablation` touches before it starts training."""

    def __init__(self, tmp: Path) -> None:
        self.report_dir = tmp
        self.logger = _StubLogger(tmp / "scwbd-003_train.jsonl")

        class _T:
            run_name = "scwbd-003"

        class _Cfg:
            train = _T()

        self.cfg = _Cfg()


def test_the_ablation_redirects_the_logger_away_from_the_run_log(tmp_path: Path) -> None:
    """The arms log somewhere else, and the production path is never reopened."""
    from scwbd.foundation import evaluate as ev

    trainer = _StubTrainer(tmp_path)
    production = trainer.logger.path
    seen: dict[str, Path] = {}

    def _fake_inner(tr, *, steps, seed):
        seen["path"] = Path(tr.logger.path)
        return {"ok": True}

    monkey = ev._source_ablation_inner
    ev._source_ablation_inner = _fake_inner
    try:
        out = ev.source_ablation(trainer, steps=1, seed=0)
    finally:
        ev._source_ablation_inner = monkey

    assert out == {"ok": True}
    assert seen["path"] != production, (
        f"the ablation logged to the production run log {production.name}. Every "
        "retraining arm appends to the transcript of the run being evaluated."
    )
    assert "ablation" in seen["path"].name, (
        f"expected an ablation-specific log, got {seen['path'].name}"
    )


def test_the_original_logger_is_restored_afterwards(tmp_path: Path) -> None:
    """A redirect that leaks would silently move all later logging."""
    from scwbd.foundation import evaluate as ev

    trainer = _StubTrainer(tmp_path)
    before = trainer.logger

    monkey = ev._source_ablation_inner
    ev._source_ablation_inner = lambda tr, *, steps, seed: {}
    try:
        ev.source_ablation(trainer, steps=1, seed=0)
    finally:
        ev._source_ablation_inner = monkey

    assert trainer.logger is before, "the trainer kept the ablation's logger"


def test_the_logger_is_restored_even_when_an_arm_raises(tmp_path: Path) -> None:
    """The `finally` is the point: an OOM mid-ablation must not leave it redirected."""
    from scwbd.foundation import evaluate as ev

    trainer = _StubTrainer(tmp_path)
    before = trainer.logger

    def _boom(tr, *, steps, seed):
        raise RuntimeError("CUDA out of memory")

    monkey = ev._source_ablation_inner
    ev._source_ablation_inner = _boom
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            ev.source_ablation(trainer, steps=1, seed=0)
    finally:
        ev._source_ablation_inner = monkey

    assert trainer.logger is before, (
        "after a failed arm the trainer is still pointed at the ablation log, so "
        "any later training in this process would write to the wrong file"
    )
