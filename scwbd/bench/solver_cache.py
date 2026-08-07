"""Content-keyed cache for expensive, deterministic solver calls (agent J).

Why this exists, stated as the defect it repairs. Auto-wiring the field gates
into :func:`~scwbd.bench.numerics.run_numerics_suite` bought a real correctness
property -- a default run can no longer silently revert a verified verdict to
``COULD_NOT_RUN`` -- and paid for it by running FDTD marches and BEM solves on
every invocation of the test suite. That is a trade nobody measured at the time
it was made. **A suite people stop running is a check that will not be invoked,
which reaches the same end state as a guard that cannot fire.** The remedy
differs, though: you fix a guard that cannot fire by redesigning the check, and
one that will not be invoked by making it cheap enough to run.

The guarantee is kept and the cost is dropped. It is not weakened to fit the
clock.

Two properties this cache must have, because it is now a provenance object:

1. **Keyed on the solver's CONTENT, never its path or import name.** A key that
   can stay constant while the solver changes would serve a stale verdict --
   precisely the failure the auto-wiring was introduced to prevent, reintroduced
   one layer down. The key includes a sha256 over the *source text* of the
   module that defines the callable, so any edit to that module misses.
2. **A cache hit must be VISIBLE in the report.** An invisible hit is
   indistinguishable from a fresh solve, and this project has a register full of
   readings that could not be distinguished from their own absence.

A callable whose source cannot be read is never cached. Refusing to cache is
always safe; guessing at content is not.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "CACHE_DIR",
    "solver_content_hash",
    "CacheStats",
    "CachedSolver",
    "cached_solver",
]

CACHE_DIR = Path(
    os.environ.get("SCWBD_BENCH_CACHE", Path(__file__).resolve().parents[2] / ".cache"
                   / "scwbd-bench")
)
#: bump when the cache's own encoding changes, so old entries cannot be read
_CACHE_FORMAT = "v1"
#: sentinel returned when a callable's source is unreadable
UNCACHEABLE = "uncacheable:no-source"


def solver_content_hash(fn: Callable[..., Any]) -> str:
    """sha256 over the SOURCE TEXT of the module defining ``fn``, plus its qualname.

    Deliberately not the module path, the import name, the file mtime or the
    function's ``id``: every one of those can stay constant across a change to
    the solver, which is the whole failure mode.  Whole-module source is used
    rather than the function body because a solver's behaviour depends on the
    helpers beside it.
    """
    try:
        module = inspect.getmodule(fn)
        src = inspect.getsource(module) if module is not None else None
    except (OSError, TypeError):
        src = None
    if not src:
        return UNCACHEABLE
    h = hashlib.sha256()
    h.update(_CACHE_FORMAT.encode())
    h.update(src.encode("utf-8", "surrogatepass"))
    h.update(getattr(fn, "__qualname__", getattr(fn, "__name__", "?")).encode())
    return h.hexdigest()


def _hash_value(obj: Any, h: "hashlib._Hash") -> None:
    if isinstance(obj, np.ndarray):
        h.update(b"ndarray")
        h.update(str(obj.dtype).encode())
        h.update(str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
        return
    if isinstance(obj, Mapping):
        h.update(b"map")
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _hash_value(obj[k], h)
        return
    if isinstance(obj, (list, tuple)):
        h.update(b"seq")
        for v in obj:
            _hash_value(v, h)
        return
    h.update(repr(obj).encode())


@dataclass
class CacheStats:
    """What a run actually did. Carried into the report; never inferred."""

    solver: str = ""
    content_hash: str = ""
    hits: int = 0
    misses: int = 0
    uncacheable: int = 0

    @property
    def served_from_cache(self) -> bool:
        return self.hits > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "solver_content_hash": self.content_hash[:16],
            "hits": self.hits,
            "misses": self.misses,
            "uncacheable_calls": self.uncacheable,
            "served_from_cache": self.served_from_cache,
            "note": (
                "a HIT means the physics was NOT recomputed on this run; the stored value "
                "was produced by a solver whose module source hashes identically. A MISS "
                "re-solved."
                if self.served_from_cache else
                "every call re-solved: no stored result matched this solver's content hash"
            ),
        }


@dataclass
class CachedSolver:
    """Memoising proxy around a deterministic solver. Refuses to guess."""

    fn: Callable[..., Any]
    seed: int = 0
    stats: CacheStats = field(default_factory=CacheStats)
    cache_dir: Path = CACHE_DIR
    enabled: bool = True
    _mem: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.stats.solver = (f"{getattr(self.fn, '__module__', '?')}."
                             f"{getattr(self.fn, '__qualname__', '?')}")
        self.stats.content_hash = solver_content_hash(self.fn)
        # The report records WHAT WAS MEASURED, so the proxy must present the
        # wrapped solver's identity rather than its own. A cache that renamed
        # the subject in the report would be a provenance defect of exactly the
        # kind this module exists to avoid.
        self.__dict__["__module__"] = getattr(self.fn, "__module__", "?")
        self.__dict__["__qualname__"] = getattr(self.fn, "__qualname__", "?")
        self.__dict__["__name__"] = getattr(self.fn, "__name__", "?")

    def _key(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(self.stats.content_hash.encode())
        h.update(str(self.seed).encode())
        _hash_value(list(args), h)
        _hash_value(dict(kwargs), h)
        return h.hexdigest()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled or self.stats.content_hash == UNCACHEABLE:
            self.stats.uncacheable += 1
            return self.fn(*args, **kwargs)
        key = self._key(args, kwargs)
        if key in self._mem:
            self.stats.hits += 1
            return self._mem[key]
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            try:
                with path.open("rb") as fh:
                    value = pickle.load(fh)
            except Exception:
                value = None
            if value is not None:
                self.stats.hits += 1
                self._mem[key] = value
                return value
        value = self.fn(*args, **kwargs)
        self.stats.misses += 1
        self._mem[key] = value
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except Exception:
            pass          # a cache that cannot write is slow, never wrong
        return value


def cached_solver(fn: Callable[..., Any] | None, *, seed: int = 0,
                  enabled: bool = True) -> Any:
    """Wrap ``fn`` so identical calls re-use the stored result. ``None`` passes through."""
    if fn is None:
        return None
    return CachedSolver(fn=fn, seed=seed, enabled=enabled)
