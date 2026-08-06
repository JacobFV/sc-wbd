"""Hashed, licensed asset manifest.

Every file this package downloads or derives is registered in
``assets/MANIFEST.json`` with a sha256, a source URL, a license and a version.
The manifest is git-tracked; the binaries are not.  A build that cannot state
where a file came from and under what terms it may be used is not a build we
ship.

Entries are keyed by path relative to the asset root, so the manifest is
portable across machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .paths import assets_root, manifest_path

__all__ = ["AssetEntry", "Manifest", "sha256_file", "sha256_tree"]

_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: str | os.PathLike[str], *, exclude: Iterable[str] = (".git",)) -> tuple[str, int, int]:
    """Hash a directory tree deterministically.

    Returns ``(sha256, n_files, total_bytes)``.  The hash covers relative paths
    and file contents in sorted order, so it is stable across clones.
    """
    root = Path(root)
    ex = set(exclude)
    h = hashlib.sha256()
    n = 0
    total = 0
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ex)
        for fn in sorted(filenames):
            files.append(Path(dirpath) / fn)
    for f in files:
        rel = f.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha256_file(f).encode())
        n += 1
        total += f.stat().st_size
    return h.hexdigest(), n, total


@dataclass
class AssetEntry:
    """One tracked asset.

    Attributes
    ----------
    path
        Path relative to the asset root.
    kind
        ``"upstream"`` for verbatim third-party data, ``"derived"`` for build
        products of this package.
    sha256
        Content hash.  For directories this is the deterministic tree hash from
        :func:`sha256_tree`.
    n_bytes
        Total size in bytes.
    source_url
        Canonical download location.
    license
        SPDX-ish license identifier or a short prose statement.  ``"unknown"``
        is a legitimate value and must be treated as restrictive.
    version
        Upstream version, git commit, or our build stamp.
    citation
        Primary reference the upstream asset asks to be cited.
    notes
        Known biases, redistribution constraints, or processing caveats.
    """

    path: str
    kind: str
    sha256: str
    n_bytes: int
    source_url: str
    license: str
    version: str
    citation: str = ""
    notes: str = ""
    is_dir: bool = False
    n_files: int = 1
    recorded_at: str = ""
    produced_by: str = ""
    inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Manifest:
    """Read/modify/write ``assets/MANIFEST.json``."""

    SCHEMA = "scwbd-asset-manifest/1.0.0"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else manifest_path()
        self.root = assets_root()
        self.entries: dict[str, AssetEntry] = {}
        self.meta: dict[str, Any] = {}
        if self.path.exists():
            self.load()

    # -- io --------------------------------------------------------------
    def load(self) -> "Manifest":
        with open(self.path) as fh:
            doc = json.load(fh)
        self.meta = doc.get("meta", {})
        self.entries = {k: AssetEntry(**v) for k, v in doc.get("assets", {}).items()}
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema": self.SCHEMA,
            "meta": {
                **self.meta,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_assets": len(self.entries),
                "total_bytes": sum(e.n_bytes for e in self.entries.values()),
            },
            "assets": {k: self.entries[k].to_dict() for k in sorted(self.entries)},
        }
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=False)
            fh.write("\n")
        tmp.replace(self.path)
        return self.path

    # -- mutation --------------------------------------------------------
    def relpath(self, path: str | os.PathLike[str]) -> str:
        p = Path(path).resolve()
        root = self.root.resolve()
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            # assets/ subdirectories are symlinks onto /data; compare targets
            for sub in ("src", "cache", "derived"):
                target = (root / sub).resolve()
                try:
                    return (Path(sub) / p.relative_to(target)).as_posix()
                except ValueError:
                    continue
            raise

    def register(
        self,
        path: str | os.PathLike[str],
        *,
        kind: str,
        source_url: str,
        license: str,
        version: str,
        citation: str = "",
        notes: str = "",
        produced_by: str = "",
        inputs: Iterable[str] = (),
    ) -> AssetEntry:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            digest, n_files, n_bytes = sha256_tree(p)
            is_dir = True
        else:
            digest = sha256_file(p)
            n_files, n_bytes, is_dir = 1, p.stat().st_size, False
        rel = self.relpath(p)
        entry = AssetEntry(
            path=rel,
            kind=kind,
            sha256=digest,
            n_bytes=n_bytes,
            source_url=source_url,
            license=license,
            version=version,
            citation=citation,
            notes=notes,
            is_dir=is_dir,
            n_files=n_files,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            produced_by=produced_by,
            inputs=list(inputs),
        )
        self.entries[rel] = entry
        return entry

    # -- verification ----------------------------------------------------
    def verify(self, *, kinds: Iterable[str] | None = None) -> dict[str, str]:
        """Re-hash every registered asset.

        Returns a mapping ``relpath -> status`` where status is one of
        ``"ok"``, ``"missing"``, ``"hash_mismatch"``.
        """
        kinds = set(kinds) if kinds is not None else None
        out: dict[str, str] = {}
        for rel, e in self.entries.items():
            if kinds is not None and e.kind not in kinds:
                continue
            p = self.root / rel
            if not p.exists():
                out[rel] = "missing"
                continue
            digest = sha256_tree(p)[0] if p.is_dir() else sha256_file(p)
            out[rel] = "ok" if digest == e.sha256 else "hash_mismatch"
        return out


def git_commit(path: str | os.PathLike[str]) -> str:
    """Best-effort ``git rev-parse HEAD`` for a cloned upstream directory."""
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
