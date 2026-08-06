"""Small shared utilities: determinism, provenance, timing, logging.

Determinism is a *test*, not an aspiration (ARCHITECTURE.md §3), so the seeding
helper is the only sanctioned way to start a stochastic run.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

__all__ = [
    "set_determinism",
    "git_sha",
    "repo_root",
    "env_fingerprint",
    "Timer",
    "JsonlLogger",
    "count_parameters",
    "human_bytes",
    "flatten_dict",
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "ARCHITECTURE.md").exists():
            return p
    return here.parents[2]


def set_determinism(seed: int, *, strict: bool = False) -> None:
    """Seed every RNG this process can reach.

    ``strict=True`` also forces deterministic cuDNN/cuBLAS kernels, which costs
    throughput.  Training runs at ``strict=False`` and record the fact; the
    determinism *tests* run at ``strict=True``.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


_SHA: str | None = None


def git_sha() -> str:
    global _SHA
    if _SHA is None:
        try:
            _SHA = subprocess.check_output(
                ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(repo_root()), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if dirty:
                _SHA += "-dirty"
        except Exception:  # noqa: BLE001
            _SHA = "unknown"
    return _SHA


def env_fingerprint() -> dict[str, Any]:
    d: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_sha": git_sha(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(
            {
                "device": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_memory_gb": round(p.total_memory / 1e9, 2),
                "multi_processor_count": p.multi_processor_count,
            }
        )
    return d


class Timer:
    """Wall-clock timer that also accumulates a FLOP counter, honestly."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.flops = 0.0
        self.marks: dict[str, float] = {}

    def add_flops(self, n: float) -> None:
        self.flops += float(n)

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def tflops(self) -> float:
        return self.flops / max(self.elapsed, 1e-9) / 1e12

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        t = time.time()
        yield
        self.marks[name] = self.marks.get(name, 0.0) + (time.time() - t)


class JsonlLogger:
    """Append-only JSONL log.  Every training run writes one; nothing is lost."""

    def __init__(self, path: str | Path, *, echo: bool = True, echo_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)
        self.echo = echo
        self.echo_every = max(1, echo_every)
        self._n = 0

    def log(self, **kw: Any) -> None:
        kw.setdefault("t", round(time.time(), 3))
        self._f.write(json.dumps(kw, default=float) + "\n")
        self._n += 1
        if self.echo and self._n % self.echo_every == 0:
            keys = [k for k in kw if k not in ("t",)]
            msg = "  ".join(
                f"{k}={kw[k]:.4g}" if isinstance(kw[k], (int, float)) and not isinstance(kw[k], bool) else f"{k}={kw[k]}"
                for k in keys[:12]
            )
            print(msg, flush=True)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key + "."))
        else:
            out[key] = v
    return out
