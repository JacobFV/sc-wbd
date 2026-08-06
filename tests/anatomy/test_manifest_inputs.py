"""``MANIFEST.json`` ``inputs`` must be what the loaders read. SIMULATION ONLY.

Why this file exists
--------------------
``inputs`` is not decoration.  ``scwbd.release.licence.anatomy_nc_inputs``
resolves it against ``scwbd.anatomy.sources.SRC`` to decide which assets
inherit a non-commercial or share-alike term, so **a wrong ``inputs`` produces
a wrong licence answer for the release**, silently and with no other symptom.

It was wrong.  ``build.py`` registered every connectome asset with the same
four-key literal and every maps asset with the same five-key literal,
regardless of what that build actually loaded.  Measured on this tree:

===================  ==========================================  ==============
asset                literal said                                actually reads
===================  ==========================================  ==============
Schaefer400x7 conn.  + ``hansen_schaefer_sc``                    ``enigma_hcp_sc`` only
DesikanKilliany conn ``hansen_schaefer_sc``                      ``hansen_lausanne_sc``
every ``*maps.npz``  ``neuromaps``                               ``hill2010``, ``raichle_metabolism``
===================  ==========================================  ==============

The Schaefer-400 row claimed a CC-BY-NC-SA-4.0 dependency the asset does not
have.  The DesikanKilliany row credited a Schaefer-100 matrix it never read
while omitting the Lausanne matrix it did -- CC-BY requires crediting the right
work, so that is an attribution defect and not only a bookkeeping one.  The
maps rows omitted two unknown-licence sources entirely, which an audit driven
off ``inputs`` could not see at all.

**This is attribution, not a gate.**  Nothing here refuses to build, refuses to
train, or restructures anything to keep a licence out.  Per ``ARCHITECTURE.md``
§7a a checkpoint may carry whatever it inherits provided it says so; these
tests only check that what it says is true.

The register's rule these tests exist to satisfy
------------------------------------------------
``reports/decorative_guards.md``: a guard that cannot fail is decoration.  So
each check below is paired with a **mutation** that proves it can fail --
``test_*_would_catch_*`` feeds the comparison a deliberately wrong value and
asserts it is caught.  Without those, a green run here would be worth nothing.
"""

from __future__ import annotations

import json

import pytest

from scwbd.anatomy import sources as S
from scwbd.anatomy.build import _connectome_inputs, _maps_inputs
from scwbd.anatomy.atlases import load_parcellation
from scwbd.anatomy.connectome import load_structural_prior
from scwbd.anatomy.maps import load_maps
from scwbd.anatomy.paths import manifest_path

#: The literals that used to sit in ``build.py``.  Kept so the mutation tests
#: can replay the exact historical defect rather than an invented one.
OLD_CONNECTOME_LITERAL = [
    "enigma_hcp_sc",
    "hansen_schaefer_sc",
    "netneuro_lausanne_sc",
    "markov2014",
]
OLD_MAPS_LITERAL = [
    "hansen_receptors",
    "neuromaps",
    "margulies2016",
    "hcps1200_maps",
    "sydnor2021",
]

CONNECTOME_ATLASES = ["Schaefer100x7", "Schaefer400x7", "DesikanKilliany"]


def _manifest_assets() -> dict:
    mp = manifest_path()
    if not mp.exists():  # pragma: no cover - assets not present
        pytest.skip("assets/MANIFEST.json not present")
    payload = json.loads(mp.read_text())
    return payload.get("assets", payload)


def _entry(assets: dict, needle: str) -> dict | None:
    for k, v in assets.items():
        if needle in str(k) and isinstance(v, dict):
            return v
    return None


# ---------------------------------------------------------------------------
# 1. inputs are derived from the object, not declared beside the call
# ---------------------------------------------------------------------------


class TestInputsAreComputedFromWhatLoaded:
    @pytest.mark.parametrize("atlas", CONNECTOME_ATLASES)
    def test_connectome_inputs_match_the_loaded_streams(self, atlas):
        sp = load_structural_prior(atlas, include_subcortex=True)
        derived = _connectome_inputs(sp)
        stream_keys = {st["source_key"] for st in sp.provenance["streams"]}
        assert stream_keys <= set(derived)
        # the base weight matrix is a real input and is not among the streams
        assert "enigma_hcp_sc" in derived, (
            f"{atlas}: the ENIGMA/HCP base weights are an input and must be "
            "recorded even when no stream names them"
        )

    def test_the_base_source_survives_the_cache_round_trip(self):
        """The bug that was in the first version of ``_connectome_inputs``.

        ``provenance["source"]`` is JSON round-tripped through the ``.npz``
        cache, so an identity check against ``S.SRC`` fails for every prior
        loaded from disk -- which is every prior in a normal build.  Matching on
        ``name`` is what makes it survive.  On ``DesikanKilliany`` the identity
        version silently dropped ``enigma_hcp_sc``: it under-reported a real
        dependency while looking like it worked.
        """
        sp = load_structural_prior("DesikanKilliany", include_subcortex=True)
        base = sp.provenance["source"]
        assert base is not S.SRC["enigma_hcp_sc"], "round trip no longer happens"
        assert base == S.SRC["enigma_hcp_sc"]
        assert "enigma_hcp_sc" in _connectome_inputs(sp)

    def test_maps_inputs_match_the_loaded_source_keys(self):
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        ms = load_maps(p)
        derived = _maps_inputs(ms)
        assert derived == sorted({m.source_key for m in ms.maps.values()})
        assert derived, "a MapSet with no source keys is not a MapSet"

    def test_every_derived_key_resolves_in_the_source_registry(self):
        """An unresolvable key silently drops out of the licence resolution."""
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        for key in _maps_inputs(load_maps(p)):
            assert key in S.SRC, f"{key!r} is not in scwbd.anatomy.sources.SRC"
        for atlas in CONNECTOME_ATLASES:
            sp = load_structural_prior(atlas, include_subcortex=True)
            for key in _connectome_inputs(sp):
                assert key in S.SRC, f"{atlas}: {key!r} is not in SRC"


# ---------------------------------------------------------------------------
# 2. the specific historical defects, each asserted by name
# ---------------------------------------------------------------------------


class TestTheHistoricalDefectsAreGone:
    def test_schaefer400_does_not_claim_a_hansen_dependency_it_lacks(self):
        """It over-claimed CC-BY-NC-SA-4.0. Over-claiming a restriction is not
        a safe error: it is the reason a permissive asset gets treated as
        encumbered."""
        derived = _connectome_inputs(
            load_structural_prior("Schaefer400x7", include_subcortex=True)
        )
        assert "hansen_schaefer_sc" not in derived
        assert "hansen_lausanne_sc" not in derived
        assert derived == ["enigma_hcp_sc"]

    def test_desikankilliany_credits_the_matrix_it_actually_read(self):
        """CC-BY requires crediting the right work. This row credited a
        Schaefer-100 matrix it never loaded and omitted the Lausanne one it
        did."""
        derived = _connectome_inputs(
            load_structural_prior("DesikanKilliany", include_subcortex=True)
        )
        assert "hansen_lausanne_sc" in derived
        assert "hansen_schaefer_sc" not in derived

    def test_maps_record_the_two_unknown_licence_sources(self):
        """``hill2010`` and ``raichle_metabolism`` are loaded and were invisible
        to any audit driven off ``inputs``."""
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        derived = _maps_inputs(load_maps(p))
        assert "hill2010" in derived
        assert "raichle_metabolism" in derived
        assert "neuromaps" not in derived, "no loaded map carries this source key"

    def test_hansen_is_still_recorded_where_it_is_real(self):
        """The fix must not become a way to lose Hansen from the record.

        Correcting over-claims is only honest if the true claims survive.
        """
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        assert "hansen_receptors" in _maps_inputs(load_maps(p))
        assert "hansen_schaefer_sc" in _connectome_inputs(
            load_structural_prior("Schaefer100x7", include_subcortex=True)
        )


# ---------------------------------------------------------------------------
# 3. THE MUTATIONS -- proof that section 1 and 2 can fail
# ---------------------------------------------------------------------------


class TestTheseChecksCanActuallyFail:
    """Without this class the file above is decoration.

    Each test replays the exact historical literal and asserts the comparison
    rejects it.  If someone reverts ``build.py`` to a hardcoded list, these are
    what go red.
    """

    @pytest.mark.parametrize(
        "atlas,wrong_key",
        [("Schaefer400x7", "hansen_schaefer_sc"), ("DesikanKilliany", "hansen_schaefer_sc")],
    )
    def test_it_would_catch_the_old_connectome_literal(self, atlas, wrong_key):
        sp = load_structural_prior(atlas, include_subcortex=True)
        derived = _connectome_inputs(sp)
        assert sorted(OLD_CONNECTOME_LITERAL) != derived
        assert wrong_key in OLD_CONNECTOME_LITERAL and wrong_key not in derived

    def test_it_would_catch_the_old_maps_literal(self):
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        derived = _maps_inputs(load_maps(p))
        assert sorted(OLD_MAPS_LITERAL) != derived
        missed = set(derived) - set(OLD_MAPS_LITERAL)
        assert missed == {"hill2010", "raichle_metabolism"}

    def test_it_would_catch_a_dropped_base_source(self):
        """Simulates the identity-match bug: base source silently absent."""
        sp = load_structural_prior("DesikanKilliany", include_subcortex=True)
        streams_only = sorted({st["source_key"] for st in sp.provenance["streams"]})
        assert streams_only == ["hansen_lausanne_sc"]
        assert "enigma_hcp_sc" not in streams_only
        assert "enigma_hcp_sc" in _connectome_inputs(sp)

    def test_it_would_catch_an_unregistered_source_key(self):
        """An unregistered base is recorded, never dropped.

        Silence is the failure mode this whole file exists to remove, so a
        source the registry does not know must still appear in ``inputs`` --
        visibly wrong beats invisibly absent.
        """

        class _FakePrior:
            provenance = {
                "streams": [{"source_key": "enigma_hcp_sc"}],
                "source": {"name": "a source nobody registered"},
            }

        out = _connectome_inputs(_FakePrior())
        assert "unregistered:a source nobody registered" in out


# ---------------------------------------------------------------------------
# 4. the manifest on disk agrees with the loaders
# ---------------------------------------------------------------------------


class TestTheManifestOnDiskAgrees:
    """Runs against the checked-in ``MANIFEST.json``.

    Section 1 tests the function; this tests the artifact the release actually
    reads.  They are different claims and both are needed: a correct function
    whose output was never written to disk changes nothing.
    """

    @pytest.mark.parametrize("atlas", CONNECTOME_ATLASES)
    def test_recorded_connectome_inputs_match_the_loader(self, atlas):
        assets = _manifest_assets()
        entry = _entry(assets, f"connectome/{atlas}__enigma_hcp__withsctx")
        if entry is None:
            pytest.skip(f"no manifest entry for {atlas} connectome")
        recorded = sorted(entry.get("inputs") or [])
        derived = _connectome_inputs(
            load_structural_prior(atlas, include_subcortex=True)
        )
        assert recorded == derived, (
            f"{atlas}: manifest records {recorded}, loader reads {derived}. "
            "Re-run `python -m scwbd.anatomy.build` to bring them back in step."
        )

    def test_recorded_maps_inputs_match_the_loader(self):
        assets = _manifest_assets()
        entry = _entry(assets, "maps/Schaefer100x7__fsLR-32k__maps")
        if entry is None:
            pytest.skip("no manifest entry for Schaefer100x7 maps")
        recorded = sorted(entry.get("inputs") or [])
        p = load_parcellation("Schaefer100x7", "fsLR", "32k")
        derived = _maps_inputs(load_maps(p))
        assert recorded == derived, (
            f"manifest records {recorded}, loader reads {derived}. "
            "Re-run `python -m scwbd.anatomy.build`."
        )
