"""Content hashing and file manifests.

Appendix B, "Identity and lineage": a source card must carry *file hashes* and
a reproducible relation to its parent.  A mutable download that cannot be
re-verified is a rejection condition, so every dataset on disk gets a manifest

``scwbd/sources/manifests/<dataset_id>__<version>.json``

listing every file's sha256 and byte count, plus a ``manifest_sha256`` root
hash over the canonical JSON of that listing.  The source card records the root
hash, the file/byte counts and a handful of exemplar file hashes; the manifest
itself is the full record and is tracked in git.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_DIR = Path(__file__).parent / "manifests"
CHUNK = 1 << 20

#: Never hashed: our own bookkeeping and filesystem noise.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    "*.log",
    ".DS_Store",
    "*.part",
    # ``download_ranged`` writes a sidecar ``<name>.part.done`` recording which
    # byte ranges completed, so a transfer resumes at chunk granularity. The
    # pattern above does not match it (fnmatch is not a prefix match), so an
    # interrupted fetch left these in the tree and they were hashed into the
    # manifest as if they were data. Added 2026-08-06 (🗄️ Ada) after finding
    # them under ds000113 mid-download.
    "*.part.done",
    "*.tmp",
    ".scwbd_*",
)


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def iter_files(root: str | Path, excludes: Sequence[str] = DEFAULT_EXCLUDES) -> list[Path]:
    root = Path(root)
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            if any(fnmatch.fnmatch(fn, pat) for pat in excludes):
                continue
            p = Path(dirpath) / fn
            if p.is_symlink() and not p.exists():
                continue
            out.append(p)
    return sorted(out)


@dataclass
class Manifest:
    dataset_id: str
    version: str
    root: str
    generated_utc: str
    algorithm: str
    files: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return int(sum(int(v["bytes"]) for v in self.files.values()))

    def manifest_sha256(self) -> str:
        payload = {k: v["sha256"] for k, v in self.files.items()}
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    def exemplars(self, n: int = 5) -> dict[str, str]:
        """Deterministic sample of file hashes to embed in the source card.

        The largest files plus the lexicographically first ones: big binaries
        are the payload, small ones are the metadata that pins the release.
        """
        by_size = sorted(self.files.items(), key=lambda kv: -int(kv[1]["bytes"]))
        picks: dict[str, str] = {}
        for k, v in by_size[: max(1, n // 2)]:
            picks[k] = str(v["sha256"])
        for k, v in sorted(self.files.items())[: n - len(picks)]:
            picks.setdefault(k, str(v["sha256"]))
        return dict(sorted(picks.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "root": self.root,
            "generated_utc": self.generated_utc,
            "algorithm": self.algorithm,
            "n_files": self.n_files,
            "total_bytes": self.total_bytes,
            "manifest_sha256": self.manifest_sha256(),
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            dataset_id=d["dataset_id"],
            version=d["version"],
            root=d["root"],
            generated_utc=d["generated_utc"],
            algorithm=d.get("algorithm", "sha256"),
            files=d["files"],
        )

    def path(self, manifest_dir: str | Path = MANIFEST_DIR) -> Path:
        return Path(manifest_dir) / f"{self.dataset_id}__{self.version}.json"

    def write(self, manifest_dir: str | Path = MANIFEST_DIR) -> Path:
        p = self.path(manifest_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")
        return p


def build_manifest(
    root: str | Path,
    dataset_id: str,
    version: str,
    *,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    workers: int = 8,
    progress: bool = False,
) -> Manifest:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root {root} does not exist")
    paths = iter_files(root, excludes)

    def one(p: Path) -> tuple[str, dict[str, object]]:
        return str(p.relative_to(root)), {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    files: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (rel, rec) in enumerate(ex.map(one, paths)):
            files[rel] = rec
            if progress and i % 200 == 0:
                print(f"  hashed {i}/{len(paths)}", flush=True)
    return Manifest(
        dataset_id=dataset_id,
        version=version,
        root=str(root),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        algorithm="sha256",
        files=dict(sorted(files.items())),
    )


def load_manifest(
    dataset_id: str, version: str, manifest_dir: str | Path = MANIFEST_DIR
) -> Manifest:
    p = Path(manifest_dir) / f"{dataset_id}__{version}.json"
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p}")
    return Manifest.from_dict(json.loads(p.read_text()))


@dataclass
class VerifyReport:
    dataset_id: str
    version: str
    n_checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched

    def summary(self) -> str:
        head = "OK" if self.ok else "FAIL"
        return (
            f"verify {self.dataset_id}@{self.version}: {head} "
            f"({self.n_checked} checked, {len(self.missing)} missing, "
            f"{len(self.mismatched)} mismatched, {len(self.extra)} untracked)"
        )


def verify_manifest(
    manifest: Manifest,
    root: str | Path | None = None,
    *,
    sample: int | None = None,
    seed: int = 0,
    workers: int = 8,
) -> VerifyReport:
    """Re-hash files on disk and compare against ``manifest``.

    ``sample`` checks a deterministic random subset (for a 100k-file tree);
    ``None`` checks everything.
    """
    import random

    root = Path(root or manifest.root)
    rep = VerifyReport(dataset_id=manifest.dataset_id, version=manifest.version)
    items = sorted(manifest.files.items())
    if sample is not None and sample < len(items):
        items = random.Random(seed).sample(items, sample)

    def one(kv: tuple[str, dict]) -> tuple[str, str | None]:
        rel, rec = kv
        p = root / rel
        if not p.exists():
            return rel, None
        return rel, sha256_file(p)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rel, digest in ex.map(one, items):
            rep.n_checked += 1
            if digest is None:
                rep.missing.append(rel)
            elif digest != manifest.files[rel]["sha256"]:
                rep.mismatched.append(rel)
    if sample is None and root.exists():
        on_disk = {str(p.relative_to(root)) for p in iter_files(root)}
        rep.extra = sorted(on_disk - set(manifest.files))
    return rep


def hashes_for(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths}


__all__ = (
    "MANIFEST_DIR",
    "Manifest",
    "VerifyReport",
    "build_manifest",
    "load_manifest",
    "verify_manifest",
    "sha256_file",
    "iter_files",
)
