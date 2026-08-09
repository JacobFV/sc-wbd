"""The run carries the object R02 and R12 validate, not a config key.

`cortical_source_dipole <= parcel` has been declared, measured and validated for
two runs -- R o P = I on the coarse support to 4.4e-16, landmark-tested at 94%
coverage, recorded in `reports/transforms/resolution_pair.json` -- and no
training run ever built it. `scwbd/foundation/` imported the poset in exactly one
place, `compiler_bridge`, and the trainer touched none of it. body.tex's second
differentiator rested on a machine that was never switched on.

It could not be switched on before. R12's control test read
`model.scale_prolongations`, so declaring a prolongation bought an exemption from
the refusal that polices overclaiming, and `test_resolution_pair_r02.py` had to
pin that field empty to keep R12 firing. R12 now requires the compiled poset, so
a run can carry its prolongations honestly and still be refused if they are not
real.
"""

from __future__ import annotations

import pytest


def test_the_measured_pair_exists_and_is_validated() -> None:
    """Not "declared". Measured, with the residual on the record."""
    from scwbd.transforms import resolution_pair as rp

    m = rp.load_measurement()
    assert m is not None, (
        f"{rp.MEASUREMENT_RELPATH} is absent, so the prolongation is declared and "
        "unmeasured. R02 refuses that and R12 must not count it."
    )
    assert m.coarse_roundtrip_residual <= m.coarse_roundtrip_tolerance, (
        f"R o P = I fails on the coarse support: residual "
        f"{m.coarse_roundtrip_residual:g} > tolerance {m.coarse_roundtrip_tolerance:g}"
    )


def test_the_compiled_poset_carries_a_tested_prolongation() -> None:
    from scwbd.foundation.compiler_bridge import _poset

    p = _poset()
    pairs = list(p.prolongations())
    assert pairs, "the compiled poset declares no prolongation"
    for pair in p.maps:
        assert pair.roundtrip_tested, (
            f"{pair.fine} <= {pair.coarse} is in the poset with roundtrip_tested "
            "False. An untested map is a declaration, and R12 must not accept a "
            "declaration as evidence."
        )


def test_r12_condition_two_is_met_by_the_poset_and_not_by_the_config() -> None:
    """The exemption is closed and the real map still counts.

    Both halves matter. If only the first held, R12 could never be satisfied and
    the multiresolution claim would be unmakeable; if only the second, editing a
    YAML key would switch the refusal off again.
    """
    from scwbd.foundation.compiler_bridge import _poset
    from scwbd.schema.designation import check_r12

    control_shaped = {
        "model": {"family_state": False, "local_core": "learned", "n_regions": 414}
    }

    assert not list(check_r12(config=control_shaped, poset=_poset())), (
        "the measured, validated prolongation no longer satisfies R12's second "
        "condition, so a genuinely multiresolution run cannot be emitted"
    )

    faked = {
        "model": {
            **control_shaped["model"],
            "scale_prolongations": ["cortical_source_dipole<=parcel"],
        }
    }
    assert list(check_r12(config=faked)), (
        "model.scale_prolongations alone switched R12 off -- the exemption is back"
    )


def test_the_trainer_builds_and_records_it() -> None:
    """Read off the source, because constructing a trainer needs the corpus.

    The point is that `scwbd/foundation/train.py` reaches for the poset at all;
    for two runs it did not, and nothing in the training path could have told
    anyone.
    """
    import inspect

    from scwbd.foundation import train as tr

    src = inspect.getsource(tr)
    assert "self.resolution_poset" in src, (
        "the trainer no longer holds a resolution poset, so the run makes no "
        "multiresolution claim it can substantiate"
    )
    assert "resolution_prolongations" in src, (
        "the checkpoint no longer records which prolongations the run carried, so "
        "R12 has to be answered from a config key again"
    )
