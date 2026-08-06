"""Resumable downloaders + checksum verification + CLI.

::

    python -m scwbd.sources.download --list
    python -m scwbd.sources.download eegmmidb
    python -m scwbd.sources.download eegmmidb --manifest      # (re)hash on disk
    python -m scwbd.sources.download eegmmidb --verify        # re-hash vs manifest
    python -m scwbd.sources.download --all --verify --sample 200

Three fetchers are implemented:

``S3PrefixFetcher``
    Anonymous (``--no-sign-request``-equivalent) S3 listing + parallel GET with
    size-based skip.  Used for the PhysioNet open mirror
    (``s3://physionet-open``) and OpenNeuro (``s3://openneuro.org``).
``HttpFetcher``
    Resumable single-file HTTP with ``Range`` restart, for direct URLs.
``MneDatasetFetcher``
    Delegates to ``mne.datasets.<name>.data_path`` (which ships its own pooch
    checksums) and then re-hashes the tree ourselves.

Nothing here writes into a dataset directory that already verifies, and
nothing fabricates a hash: a fetcher that fails leaves the registry entry
``unavailable`` and the reason is written into the source card by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .manifest import Manifest, build_manifest, load_manifest, verify_manifest

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------
@dataclass
class FetchResult:
    dataset_id: str
    dest: Path
    n_files: int = 0
    n_bytes: int = 0
    n_skipped: int = 0
    ok: bool = True
    error: str = ""

    def summary(self) -> str:
        state = "ok" if self.ok else f"FAILED: {self.error}"
        return (
            f"fetch {self.dataset_id} -> {self.dest}: {state} "
            f"({self.n_files} new files, {self.n_bytes / 1e9:.3f} GB, "
            f"{self.n_skipped} already present)"
        )


class Fetcher:
    """Base class.  ``fetch(dest)`` must be resumable and idempotent.

    Subclasses are dataclasses and declare ``dataset_id`` themselves; the base
    deliberately does not provide a default (a class-level default here would
    become an implicit dataclass default in every subclass).
    """

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:  # pragma: no cover
        raise NotImplementedError


@dataclass
class S3PrefixFetcher(Fetcher):
    """Anonymous parallel sync of an S3 prefix.

    Files whose local size already matches the object size are skipped, which
    makes the fetch resumable at file granularity.  Partial files are written
    to ``<name>.part`` and renamed on completion, so an interrupted transfer
    never leaves a truncated file that would later hash "successfully" into a
    manifest.
    """

    dataset_id: str
    bucket: str
    prefix: str
    includes: tuple[str, ...] = ()  # fnmatch patterns on the key suffix; () = all
    excludes: tuple[str, ...] = ()
    workers: int = 12

    def _client(self):
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        return boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=self.workers * 2,
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )

    def _wanted(self, rel: str) -> bool:
        import fnmatch

        if any(fnmatch.fnmatch(rel, pat) for pat in self.excludes):
            return False
        if not self.includes:
            return True
        return any(fnmatch.fnmatch(rel, pat) for pat in self.includes)

    def list_objects(self) -> list[tuple[str, int]]:
        s3 = self._client()
        out: list[tuple[str, int]] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(self.prefix) :].lstrip("/")
                if not rel or rel.endswith("/"):
                    continue
                if self._wanted(rel):
                    out.append((obj["Key"], int(obj["Size"])))
        return sorted(out)

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        res = FetchResult(dataset_id=self.dataset_id, dest=dest)
        try:
            objs = self.list_objects()
        except Exception as exc:  # network / bucket gone
            res.ok = False
            res.error = f"listing s3://{self.bucket}/{self.prefix} failed: {exc}"
            return res
        if dry_run:
            res.n_files = len(objs)
            res.n_bytes = sum(s for _, s in objs)
            return res
        dest.mkdir(parents=True, exist_ok=True)
        s3 = self._client()

        def one(item: tuple[str, int]) -> tuple[int, int]:
            key, size = item
            rel = key[len(self.prefix) :].lstrip("/")
            out = dest / rel
            if out.exists() and out.stat().st_size == size:
                return (0, 0)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".part")
            s3.download_file(self.bucket, key, str(tmp))
            os.replace(tmp, out)
            return (1, size)

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(one, o): o for o in objs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    new, nbytes = fut.result()
                except Exception as exc:
                    errors.append(f"{futs[fut][0]}: {exc}")
                    continue
                res.n_files += new
                res.n_bytes += nbytes
                res.n_skipped += 1 - new
                if done % 200 == 0:
                    _log(f"  [{self.dataset_id}] {done}/{len(objs)} objects")
        if errors:
            res.ok = False
            res.error = f"{len(errors)} object(s) failed, first: {errors[0]}"
        return res


@dataclass
class HttpFetcher(Fetcher):
    """Resumable HTTP fetch of a list of (url, relative_path) pairs."""

    dataset_id: str
    urls: tuple[tuple[str, str], ...]
    workers: int = 4
    timeout: float = 60.0

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        import requests

        res = FetchResult(dataset_id=self.dataset_id, dest=dest)
        if dry_run:
            res.n_files = len(self.urls)
            return res
        dest.mkdir(parents=True, exist_ok=True)

        def one(item: tuple[str, str]) -> tuple[int, int]:
            url, rel = item
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".part")
            if out.exists():
                return (0, 0)
            pos = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, stream=True, headers=headers, timeout=self.timeout) as r:
                if pos and r.status_code == 200:  # server ignored Range
                    pos = 0
                r.raise_for_status()
                mode = "ab" if pos else "wb"
                got = 0
                with open(tmp, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        got += len(chunk)
            os.replace(tmp, out)
            return (1, got)

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(one, u): u for u in self.urls}
            for fut in as_completed(futs):
                try:
                    new, nbytes = fut.result()
                except Exception as exc:
                    errors.append(f"{futs[fut][0]}: {exc}")
                    continue
                res.n_files += new
                res.n_bytes += nbytes
                res.n_skipped += 1 - new
        if errors:
            res.ok = False
            res.error = f"{len(errors)} url(s) failed, first: {errors[0]}"
        return res


def download_ranged(
    url: str,
    out: Path,
    *,
    total: int,
    workers: int = 8,
    chunk_size: int = 16 << 20,
    timeout: float = 300.0,
    tries: int = 4,
) -> int:
    """Parallel byte-range download with per-chunk resume.

    Some servers (the OpenNeuro CRN file endpoint among them) throttle each
    connection hard while happily serving ``206 Partial Content``, so a single
    stream runs at ~0.2 MB/s while eight ranges run at ~2 MB/s.  Chunks are
    written into a sparse ``.part`` file at their true offsets and a sidecar
    ``.part.done`` records which chunks completed, so an interrupted transfer
    resumes at chunk granularity instead of restarting.
    """
    import requests

    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    done_path = part.with_suffix(part.suffix + ".done")
    n_chunks = max(1, (total + chunk_size - 1) // chunk_size)
    done: set[int] = set()
    if part.exists() and done_path.exists():
        try:
            done = {int(x) for x in done_path.read_text().split() if x}
        except ValueError:
            done = set()
    else:
        with open(part, "wb") as fh:
            fh.truncate(total)
        done_path.write_text("")

    lock = threading.Lock()

    def one(i: int) -> int:
        if i in done:
            return 0
        start = i * chunk_size
        end = min(start + chunk_size, total) - 1
        last: Exception | None = None
        for _ in range(tries):
            try:
                r = requests.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout
                )
                r.raise_for_status()
                buf = r.content
                if len(buf) != end - start + 1:
                    raise OSError(f"short chunk {i}: {len(buf)} != {end - start + 1}")
                with lock:
                    with open(part, "r+b") as fh:
                        fh.seek(start)
                        fh.write(buf)
                    with open(done_path, "a") as fh:
                        fh.write(f"{i}\n")
                return len(buf)
            except Exception as exc:  # transient network failure
                last = exc
        raise OSError(f"chunk {i} failed after {tries} tries: {last}")

    got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(one, range(n_chunks)):
            got += n
    if part.stat().st_size != total:
        raise OSError(f"{out.name}: assembled {part.stat().st_size} of {total} bytes")
    os.replace(part, out)
    done_path.unlink(missing_ok=True)
    return got


@dataclass
class OpenNeuroSnapshotFetcher(Fetcher):
    """Resumable fetch of named files from a **pinned OpenNeuro snapshot**.

    Needed because the ``s3://openneuro.org`` mirror does not carry every
    git-annex object: for ds000117 the MEG ``.fif`` binaries are absent from
    S3 but served by the CRN snapshot endpoint.  Pinning ``tag`` (not "draft")
    is what makes the download reproducible.
    """

    dataset_id: str
    accession: str
    tag: str
    relpaths: tuple[str, ...]
    workers: int = 2            # files in flight
    range_workers: int = 8      # parallel byte ranges per file (the endpoint throttles per connection)
    timeout: float = 300.0

    @property
    def base(self) -> str:
        return (
            f"https://openneuro.org/crn/datasets/{self.accession}"
            f"/snapshots/{self.tag}/files/"
        )

    def url_for(self, rel: str) -> str:
        return self.base + rel.replace("/", ":")

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        import requests

        res = FetchResult(dataset_id=self.dataset_id, dest=dest)
        if dry_run:
            res.n_files = len(self.relpaths)
            return res
        dest.mkdir(parents=True, exist_ok=True)

        def one(rel: str) -> tuple[int, int]:
            url = self.url_for(rel)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            head = requests.head(url, allow_redirects=True, timeout=self.timeout)
            head.raise_for_status()
            total = int(head.headers.get("content-length", 0))
            if out.exists() and (total == 0 or out.stat().st_size == total):
                return (0, 0)
            if total == 0:
                raise OSError(f"{rel}: server did not report a content length")
            got = download_ranged(
                url, out, total=total, workers=self.range_workers, timeout=self.timeout
            )
            _log(f"  [{self.dataset_id}] {rel} ({got / 1e6:.0f} MB)")
            return (1, out.stat().st_size)

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(one, r): r for r in self.relpaths}
            for fut in as_completed(futs):
                try:
                    new, nbytes = fut.result()
                except Exception as exc:
                    errors.append(f"{futs[fut]}: {exc}")
                    continue
                res.n_files += new
                res.n_bytes += nbytes
                res.n_skipped += 1 - new
        if errors:
            res.ok = False
            res.error = f"{len(errors)} file(s) failed, first: {errors[0]}"
        return res


@dataclass
class CompositeFetcher(Fetcher):
    """Run several fetchers into one destination (e.g. S3 bulk + snapshot MEG)."""

    dataset_id: str
    parts: tuple[Fetcher, ...]

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        total = FetchResult(dataset_id=self.dataset_id, dest=dest)
        for part in self.parts:
            r = part.fetch(dest, dry_run=dry_run)
            total.n_files += r.n_files
            total.n_bytes += r.n_bytes
            total.n_skipped += r.n_skipped
            if not r.ok:
                total.ok = False
                total.error = (total.error + " | " + r.error).strip(" |")
        return total


@dataclass
class MneDatasetFetcher(Fetcher):
    """``mne.datasets.<name>.data_path`` with its own pooch checksums."""

    dataset_id: str
    mne_name: str

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        res = FetchResult(dataset_id=self.dataset_id, dest=dest)
        if dry_run:
            return res
        try:
            import mne
        except Exception as exc:
            res.ok = False
            res.error = f"mne not importable: {exc}"
            return res
        dest.mkdir(parents=True, exist_ok=True)
        try:
            mod = getattr(mne.datasets, self.mne_name)
            path = mod.data_path(path=str(dest), download=True, verbose="warning")
        except Exception as exc:
            res.ok = False
            res.error = f"mne.datasets.{self.mne_name}.data_path failed: {exc}"
            return res
        got = Path(str(path))
        n, b = 0, 0
        for p in got.rglob("*"):
            if p.is_file():
                n += 1
                b += p.stat().st_size
        res.n_files, res.n_bytes, res.dest = n, b, got
        return res


@dataclass
class UnavailableFetcher(Fetcher):
    """A registered dataset we deliberately do not download.

    Used for DUA/credential-gated registers (UK Biobank, HCP, TUH, ADNI, RAM).
    Calling ``fetch`` never touches the network; it returns the honest reason.
    """

    dataset_id: str
    reason: str

    def fetch(self, dest: Path, *, dry_run: bool = False) -> FetchResult:
        return FetchResult(
            dataset_id=self.dataset_id, dest=dest, ok=False, error=f"unavailable: {self.reason}"
        )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def fetch_dataset(dataset_id: str, *, dry_run: bool = False) -> FetchResult:
    from .registry import REGISTRY

    entry = REGISTRY[dataset_id]
    if entry.fetcher is None:
        return FetchResult(
            dataset_id=dataset_id,
            dest=entry.local_path,
            ok=False,
            error="no fetcher registered (see the source card's governance block)",
        )
    return entry.fetcher.fetch(entry.local_path, dry_run=dry_run)


def make_manifest(dataset_id: str, *, write: bool = True, progress: bool = True) -> Manifest:
    from .registry import REGISTRY

    entry = REGISTRY[dataset_id]
    man = build_manifest(
        entry.local_path, entry.dataset_id, entry.version, workers=12, progress=progress
    )
    if write:
        p = man.write()
        _log(f"wrote {p} ({man.n_files} files, {man.total_bytes / 1e9:.3f} GB)")
    return man


def verify_dataset(dataset_id: str, *, sample: int | None = None):
    from .registry import REGISTRY

    entry = REGISTRY[dataset_id]
    man = load_manifest(entry.dataset_id, entry.version)
    return verify_manifest(man, entry.local_path, sample=sample)


def _main(argv: Sequence[str] | None = None) -> int:
    from .registry import REGISTRY

    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.download")
    ap.add_argument("dataset_id", nargs="*", help="registry ids to act on")
    ap.add_argument("--all", action="store_true", help="act on every downloadable dataset")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--manifest", action="store_true", help="(re)build the sha256 manifest")
    ap.add_argument("--verify", action="store_true", help="verify bytes against the manifest")
    ap.add_argument("--sample", type=int, default=None, help="verify only N files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for e in REGISTRY.values():
            print(e.describe())
        return 0

    ids = list(args.dataset_id)
    if args.all:
        ids = [e.dataset_id for e in REGISTRY.values() if e.fetcher is not None]
    if not ids:
        ap.error("give at least one dataset_id, or --all, or --list")

    rc = 0
    for did in ids:
        if did not in REGISTRY:
            print(f"unknown dataset id {did!r}; try --list", file=sys.stderr)
            rc = 2
            continue
        if not (args.manifest or args.verify):
            res = fetch_dataset(did, dry_run=args.dry_run)
            print(res.summary())
            rc = rc or (0 if res.ok else 1)
        if args.manifest:
            make_manifest(did)
        if args.verify:
            rep = verify_dataset(did, sample=args.sample)
            print(rep.summary())
            if not rep.ok:
                rc = 1
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
