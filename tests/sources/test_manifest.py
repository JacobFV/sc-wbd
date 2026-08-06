"""Hashes match the bytes on disk, and a corrupted byte is detected."""

from __future__ import annotations

import json

import pytest

from scwbd.sources import registry
from scwbd.sources.manifest import (
    build_manifest,
    load_manifest,
    sha256_file,
    verify_manifest,
)

from .conftest import require_dataset, require_manifest


def test_manifest_roundtrip_and_corruption_detection(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    (root / "sub-01" / "a.bin").write_bytes(b"hello world" * 100)
    (root / "b.txt").write_text("metadata\n")

    man = build_manifest(root, "toy", "1.0.0")
    assert man.n_files == 2
    assert man.total_bytes == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    root_hash = man.manifest_sha256()
    assert len(root_hash) == 64

    rep = verify_manifest(man, root)
    assert rep.ok, rep.summary()

    # flip one byte
    p = root / "sub-01" / "a.bin"
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF
    p.write_bytes(bytes(data))
    rep2 = verify_manifest(man, root)
    assert not rep2.ok
    assert "sub-01/a.bin" in rep2.mismatched

    # a missing file is caught too
    p.unlink()
    rep3 = verify_manifest(man, root)
    assert "sub-01/a.bin" in rep3.missing


def test_manifest_root_hash_is_content_addressed(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "a").write_text("x")
    m1 = build_manifest(root, "toy", "1")
    (root / "a").write_text("y")
    m2 = build_manifest(root, "toy", "1")
    assert m1.manifest_sha256() != m2.manifest_sha256()


def test_manifest_write_and_load(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "a").write_text("x")
    man = build_manifest(root, "toy", "1.0.0")
    p = man.write(tmp_path)
    loaded = json.loads(p.read_text())
    assert loaded["manifest_sha256"] == man.manifest_sha256()
    assert loaded["n_files"] == 1


@pytest.mark.parametrize("dataset_id", sorted(registry.REGISTRY))
def test_recorded_hashes_match_the_downloaded_bytes(dataset_id):
    """Every exemplar hash in a live card is re-derived from the real file."""
    from scwbd.sources.cards import load_card

    entry = registry.get(dataset_id)
    card = load_card(entry.card_path)
    if card.status == "unavailable":
        pytest.skip(f"{dataset_id} is registered as unavailable")
    require_dataset(dataset_id)
    fm = card.data["identity"]["file_manifest"]
    assert isinstance(fm, dict), f"{dataset_id}: card was never refreshed"
    root = entry.local_path
    checked = 0
    for rel, digest in fm["exemplars"].items():
        p = root / rel
        if not p.exists():
            pytest.fail(f"{dataset_id}: card names {rel} but it is not on disk")
        assert sha256_file(p) == digest, f"{dataset_id}: {rel} does not match the recorded hash"
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("dataset_id", sorted(registry.REGISTRY))
def test_manifest_sample_verifies_against_disk(dataset_id):
    from scwbd.sources.cards import load_card

    entry = registry.get(dataset_id)
    if load_card(entry.card_path).status == "unavailable":
        pytest.skip("unavailable by design")
    require_manifest(dataset_id)
    man = load_manifest(dataset_id, entry.version)
    rep = verify_manifest(man, entry.local_path, sample=25, seed=0)
    assert rep.ok, rep.summary()


@pytest.mark.parametrize("dataset_id", sorted(registry.REGISTRY))
def test_card_manifest_hash_matches_the_stored_manifest(dataset_id):
    from scwbd.sources.cards import load_card

    entry = registry.get(dataset_id)
    card = load_card(entry.card_path)
    if card.status == "unavailable":
        pytest.skip("unavailable by design")
    require_manifest(dataset_id)
    man = load_manifest(dataset_id, entry.version)
    assert card.data["identity"]["file_manifest"]["manifest_sha256"] == man.manifest_sha256()
    assert card.data["identity"]["file_manifest"]["n_files"] == man.n_files
    assert card.data["identity"]["file_manifest"]["total_bytes"] == man.total_bytes
