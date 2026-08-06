"""The asset manifest: every tracked file exists, hashes, and states its license."""

from __future__ import annotations

import json

import pytest

from scwbd.anatomy import sources as S
from scwbd.anatomy.manifest import Manifest, sha256_file, sha256_tree
from scwbd.anatomy.paths import assets_root, manifest_path


@pytest.fixture(scope="module")
def manifest():
    if not manifest_path().exists():
        pytest.skip("assets/MANIFEST.json not built; run python -m scwbd.anatomy.build")
    return Manifest()


def test_manifest_is_valid_json_with_a_schema_version():
    if not manifest_path().exists():
        pytest.skip("manifest not built")
    doc = json.loads(manifest_path().read_text())
    assert doc["schema"].startswith("scwbd-asset-manifest/")
    assert doc["meta"]["n_assets"] == len(doc["assets"])
    assert doc["meta"]["total_bytes"] > 0


def test_every_asset_exists_and_hashes_correctly(manifest):
    status = manifest.verify()
    bad = {k: v for k, v in status.items() if v != "ok"}
    assert not bad, f"manifest entries failed verification: {bad}"


def test_every_asset_declares_a_license_and_a_source(manifest):
    for rel, e in manifest.entries.items():
        assert e.license and e.license != "", f"{rel} has no license"
        assert e.source_url, f"{rel} has no source URL"
        assert e.version, f"{rel} has no version"
        assert e.sha256 and len(e.sha256) == 64
        assert e.n_bytes > 0


def test_upstream_assets_cite_their_paper(manifest):
    ups = [e for e in manifest.entries.values() if e.kind == "upstream"]
    assert ups, "no upstream assets registered"
    for e in ups:
        assert e.citation, f"{e.path} does not cite anything"


def test_derived_assets_name_their_inputs_and_producer(manifest):
    der = [e for e in manifest.entries.values() if e.kind == "derived"]
    assert der, "no derived assets registered"
    for e in der:
        assert e.produced_by.startswith("scwbd.anatomy"), e.path
        assert e.inputs, f"{e.path} does not name its inputs"


def test_the_non_commercial_license_is_recorded_not_laundered(manifest):
    """The Hansen receptor atlas is CC-BY-NC-SA; that must survive into the manifest."""
    hits = [e for e in manifest.entries.values() if "hansen_receptors" in e.path]
    assert hits, "hansen_receptors is not in the manifest"
    assert any("NC" in e.license for e in hits)


def test_derived_products_of_a_restricted_source_say_so(manifest):
    maps = [e for e in manifest.entries.values() if "/maps/" in e.path]
    assert maps
    for e in maps:
        assert "hansen_receptors" in e.inputs
        assert "restrictive" in e.license or "inherits" in e.license


def test_the_expected_artifact_families_are_present(manifest):
    kinds = {p.split("/")[1] for p in manifest.entries if p.startswith("derived/")}
    assert {"parcellations", "geometry", "maps", "connectome"} <= kinds


def test_source_registry_entries_are_complete():
    for key, src in S.SRC.items():
        for f in ("name", "url", "license", "version", "citation"):
            assert src.get(f), f"SRC[{key!r}] is missing {f}"
        assert src.get("bias"), f"SRC[{key!r}] does not state its known bias"


def test_source_registry_flags_cross_species_transfer():
    assert "CROSS-SPECIES" in S.SRC["markov2014"]["bias"]


def test_hashing_helpers(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert len(sha256_file(f)) == 64
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"world")
    h1, n, total = sha256_tree(tmp_path)
    assert n == 2 and total == 10
    h2, _, _ = sha256_tree(tmp_path)
    assert h1 == h2, "tree hashing must be deterministic"


def test_register_and_roundtrip(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    (root / "derived").mkdir(parents=True)
    f = root / "derived" / "x.npz"
    f.write_bytes(b"\x00" * 32)
    monkeypatch.setenv("SCWBD_ASSETS", str(root))
    m = Manifest(root / "MANIFEST.json")
    m.root = root
    m.register(f, kind="derived", source_url="u", license="MIT", version="1")
    p = m.save()
    m2 = Manifest(p)
    m2.root = root
    assert "derived/x.npz" in m2.entries
    assert m2.verify()["derived/x.npz"] == "ok"
    f.write_bytes(b"\x01" * 32)
    assert m2.verify()["derived/x.npz"] == "hash_mismatch"
    f.unlink()
    assert m2.verify()["derived/x.npz"] == "missing"
