"""Attribution in assets/MANIFEST.json must be derived and must never be empty.

The failure this file exists for: `Aseg14T` -- the DEFAULT subcortical atlas and
half of what makes the prior 414 parcels -- was absent from a hand-maintained
`{name: [keys]}` literal, fell through to `[]`, and shipped unattributed. The
attribution gate correctly refused the whole anatomy prior.
"""

from __future__ import annotations

import pytest

from scwbd.anatomy import load_parcellation
from scwbd.anatomy.build import _atlas_inputs
from scwbd.anatomy.connectome import DEFAULT_SUBCORTICAL_ATLAS, structural_cache_tag


def test_default_subcortical_atlas_is_attributed():
    """Mutation: drop the Tian branch from _atlas_inputs -> raises, not [].

    The Melbourne licence's single condition is that work using the atlas cites
    Tian 2020, so an empty list here is a licence gap, not just a tidiness one.
    """
    assert "tian2020" in _atlas_inputs(DEFAULT_SUBCORTICAL_ATLAS)


def test_atlas_inputs_refuses_to_return_nothing():
    with pytest.raises(ValueError, match="NO source keys|unattributed"):
        _atlas_inputs("NotAnAtlasWeShip")


def test_redistributor_never_replaces_the_defining_atlas():
    """Schaefer labels arrive via the ENIGMA toolbox; both must be recorded.

    Deriving the key from `provenance.source_url` alone matched `enigmatoolbox`
    and silently dropped `schaefer2018` -- losing the citation the parcellation's
    own licence asks for while still looking derived and principled.
    """
    p = load_parcellation("Schaefer400x7", "fsLR", "32k")
    got = _atlas_inputs("Schaefer400x7", p)
    assert "schaefer2018" in got and "enigmatoolbox" in got


def test_structural_tag_is_produced_by_one_function():
    """The artifact stem encodes a licence-bearing choice of subcortical atlas.

    build.py used to rebuild this string by hand as `...withsctx...`, hardcoding
    the Aseg14/Harvard-Oxford variant while the loader defaulted to Aseg14T, so
    the manifest described a different artifact than production loads.
    """
    assert structural_cache_tag("Schaefer400x7", True, None, "euclidean") == (
        f"Schaefer400x7__enigma_hcp__with-{DEFAULT_SUBCORTICAL_ATLAS}sctx__euclidean"
    )
    assert structural_cache_tag("Schaefer400x7", False, None, "euclidean").endswith("nosctx__euclidean")
    assert "with-Aseg14T" not in structural_cache_tag("Schaefer400x7", True, "Aseg14", "euclidean")


def test_manifest_has_no_unattributed_derived_asset():
    from scwbd.anatomy.manifest import Manifest

    m = Manifest().load()
    bad = [
        k for k, e in m.entries.items()
        if k.startswith("derived/")
        and not (e.to_dict() if hasattr(e, "to_dict") else e).get("inputs")
    ]
    assert not bad, f"derived assets with no attribution: {bad}"
