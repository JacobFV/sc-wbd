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
    "logical_param_name",
    "cap_cuda_reserve",
    "cuda_reserved_gb",
]


def cuda_reserved_gb(device: Any = None) -> float:
    """GB the CUDA caching allocator currently holds (reserved, not live).

    This is the number that matters for whether the machine survives: the
    allocator keeps freed blocks, so *reserved* is the machine's exposure while
    *allocated* is merely what the model is using right now.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(device) / 1024**3


def cap_cuda_reserve(device: Any, limit_gb: float) -> float | None:
    """Bound the CUDA caching allocator to ``limit_gb``.  Returns the fraction set.

    **This is not redundant with a systemd MemoryMax cgroup, and assuming it was
    is what killed the machine.**  The GB10 is one unified physical pool, but
    device allocations are not charged to the cgroup: on 2026-08-06 a training
    run held 97.9 GB of device memory (``nvidia-smi --query-compute-apps``)
    while its own cgroup reported ``memory.current = 8.17 GB`` against a 40 GB
    ``MemoryMax`` that never fired.  The cgroup was measuring host pages and
    reporting reassuring numbers about a limit it was not enforcing.

    PyTorch's caching allocator has no default ceiling -- it reserves and holds
    rather than returning -- so on a shared pool it will grow until something
    dies.  ``set_per_process_memory_fraction`` is the only bound that applies to
    the thing actually doing the allocating.

    Call before the first allocation.  ``limit_gb <= 0`` disables the cap and
    says so out loud, because silently unbounded is how this happened.
    """
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    # ``torch.device("cuda")`` carries no index and set_per_process_memory_fraction
    # refuses it; resolve to the concrete device the work will land on.
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    if limit_gb <= 0:
        print(
            f"[mem] CUDA reserve UNCAPPED on a {total / 1024**3:.0f} GB shared pool; "
            "the caching allocator may grow until the machine OOMs",
            flush=True,
        )
        return None
    fraction = min(max(limit_gb * 1024**3 / total, 0.01), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    print(
        f"[mem] CUDA reserve capped at {limit_gb:.1f} GB "
        f"(fraction={fraction:.3f} of {total / 1024**3:.1f} GB unified pool)",
        flush=True,
    )
    return fraction

#: Name segments ``torch.compile`` inserts into ``named_parameters()``.
#: ``torch.compile(mod)`` returns an ``OptimizedModule`` holding the original
#: under ``_orig_mod``, so ``local.embed`` silently becomes
#: ``local._orig_mod.embed`` the moment a submodule is compiled.
_COMPILE_WRAPPER_SEGMENTS = frozenset({"_orig_mod", "_orig_module"})


def logical_param_name(name: str) -> str:
    """A parameter's name with ``torch.compile`` wrapper segments removed.

    Every permission in this codebase -- source cards, stage allowlists, the
    compiler's parameter groups -- is written against the *logical* module tree
    the architecture describes.  ``torch.compile`` is an execution detail that
    rewrites that tree's names, and a glob written as ``local.embed`` stops
    matching when it does.  The failure is silent and asymmetric: prefix globs
    like ``local.*`` keep matching, exact names do not, so a permission set can
    half-apply and still look enforced.

    This ran in production on 2026-08-05.  ``cfg.model.compile`` is true on CUDA,
    which compiled ``model.local`` and ``model.residual``; every per-region
    binding (``local.embed``, ``residual.embed``, ``local.films.*.region_scale``)
    then matched nothing, while the coarser operator globs still matched -- so
    the run trained with per-region permissions that governed no tensor.
    Normalising here, at the single point where names meet patterns, is what
    keeps the declaration true of the model that actually runs.
    """
    if "_orig_mod" not in name:
        return name
    return ".".join(p for p in name.split(".") if p not in _COMPILE_WRAPPER_SEGMENTS)


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
