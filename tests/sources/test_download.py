"""Downloaders: resumable, checksum-verified, and refusing what they must."""

from __future__ import annotations

import http.server
import socket
import threading
from pathlib import Path

import pytest

from scwbd.sources.download import (
    CompositeFetcher,
    FetchResult,
    HttpFetcher,
    OpenNeuroSnapshotFetcher,
    S3PrefixFetcher,
    UnavailableFetcher,
    download_ranged,
)
from scwbd.sources.manifest import sha256_file


# --------------------------------------------------------------------------
# a local server, so the resume logic is tested without the network
# --------------------------------------------------------------------------
class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def log_message(self, *a):  # silence
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng:
            spec = rng.split("=")[1]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            body = self.payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        else:
            body = self.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    import os

    _RangeHandler.payload = os.urandom(5_000_000)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RangeHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/file.bin", _RangeHandler.payload
    httpd.shutdown()


def test_ranged_download_reassembles_exactly(server, tmp_path):
    url, payload = server
    out = tmp_path / "file.bin"
    got = download_ranged(url, out, total=len(payload), workers=4, chunk_size=1 << 20)
    assert got == len(payload)
    assert out.read_bytes() == payload
    assert not out.with_suffix(".bin.part").exists()


def test_ranged_download_resumes_from_completed_chunks(server, tmp_path):
    url, payload = server
    out = tmp_path / "file.bin"
    part = out.with_suffix(".bin.part")
    done = part.with_suffix(".part.done")
    # simulate an interrupted transfer: chunk 0 written, chunks 1..n missing
    chunk = 1 << 20
    with open(part, "wb") as fh:
        fh.truncate(len(payload))
        fh.write(payload[:chunk])
    done.write_text("0\n")
    download_ranged(url, out, total=len(payload), workers=3, chunk_size=chunk)
    assert out.read_bytes() == payload


def test_http_fetcher_is_idempotent(server, tmp_path):
    url, payload = server
    f = HttpFetcher(dataset_id="toy", urls=((url, "a/b.bin"),))
    r1 = f.fetch(tmp_path)
    assert r1.ok and r1.n_files == 1
    digest = sha256_file(tmp_path / "a" / "b.bin")
    r2 = f.fetch(tmp_path)
    assert r2.n_files == 0 and r2.n_skipped == 1
    assert sha256_file(tmp_path / "a" / "b.bin") == digest


def test_unavailable_fetcher_never_touches_the_network(tmp_path):
    f = UnavailableFetcher(dataset_id="ukbiobank-brain-imaging", reason="MTA required")
    res = f.fetch(tmp_path)
    assert not res.ok
    assert "MTA required" in res.error
    assert list(tmp_path.iterdir()) == []


def test_composite_fetcher_aggregates_failures(tmp_path):
    class _Boom:
        def fetch(self, dest, *, dry_run=False):
            return FetchResult(dataset_id="x", dest=dest, ok=False, error="nope")

    c = CompositeFetcher(dataset_id="x", parts=(_Boom(), _Boom()))
    res = c.fetch(tmp_path)
    assert not res.ok and res.error.count("nope") == 2


def test_s3_include_exclude_filtering():
    f = S3PrefixFetcher(
        dataset_id="x",
        bucket="b",
        prefix="p/",
        includes=("*.json", "sub-01/**"),
        excludes=(".datalad/*",),
    )
    assert f._wanted("a.json")
    assert not f._wanted("a.nii.gz")
    assert not f._wanted(".datalad/config")


def test_openneuro_url_is_snapshot_pinned():
    f = OpenNeuroSnapshotFetcher(
        dataset_id="ds000117", accession="ds000117", tag="1.1.0", relpaths=("sub-01/a/b.fif",)
    )
    url = f.url_for("sub-01/a/b.fif")
    assert "/snapshots/1.1.0/" in url and "draft" not in url
    assert url.endswith("sub-01:a:b.fif")


@pytest.mark.parametrize("dataset_id", ["eegmmidb", "sleep-edfx"])
def test_registered_fetcher_dry_run_lists_real_objects(dataset_id):
    """Network test: the pinned prefix still exists upstream."""
    pytest.importorskip("boto3")
    from scwbd.sources.registry import get

    f = get(dataset_id).fetcher
    try:
        res = f.fetch(Path("/nonexistent"), dry_run=True)
    except Exception as exc:  # offline
        pytest.skip(f"network unavailable: {exc}")
    if not res.ok:
        pytest.skip(f"upstream listing failed: {res.error}")
    assert res.n_files > 100
    assert res.n_bytes > 1e9
