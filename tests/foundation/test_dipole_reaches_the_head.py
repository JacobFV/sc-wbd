"""O-5b: the dipole moment must be addressable by an observation head.

The defect this closes was not a missing component and not a missing head. Both
existed:

* ``build_lead_field`` emits ``matrix_vec`` with shape ``(64, 414, 3)``;
* ``EEGHead.source_moment()`` reads a ``dipole`` component and projects it;
* every cortical family declared ``dipole``, dim 3, in ``Hz*m``.

And ``source_moment()`` returned ``None``, every time, because ``dipole`` was
declared *per cortical family* and therefore lived inside the ``private`` block
that ``SCWBD.build_layout`` forbids a head from addressing. Two correct halves,
pointed at different address spaces — the same shape as the source-card rename
that left 88.7% of run 2 untrainable, in the place it costs the most:

    a per-parcel scalar carries  5.6% of the whitened EEG lead field
    a 3-vector moment carries   51.7%

``ARCHITECTURE.md`` O-5b deferred this to run 3 because changing the shared
interface changes every offset and would have invalidated the checkpoints of the
run then training. That run is finished, evaluated and published, so the reason
expired rather than being overridden.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.families import shared_components
from scwbd.foundation.model import SCWBD

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = "configs/run2/pilot-families.yaml"


@pytest.fixture(scope="module")
def model() -> SCWBD:
    return SCWBD(load_config(CONFIG).model, load_anatomy())


def test_dipole_is_in_the_shared_interface_not_the_private_block() -> None:
    """The interface commitment: every family exposes it at the same offset."""
    names = [c.name for c in shared_components()]
    assert "dipole" in names, (
        "dipole is not in shared_components(), so it lives in `private` and no "
        "observation head can address it -- which is the entire O-5b defect"
    )
    spec = next(c for c in shared_components() if c.name == "dipole")
    assert spec.dim == 3, f"dipole must be a 3-vector moment, got dim={spec.dim}"
    assert spec.units == "Hz*m", (
        f"dipole declares units {spec.units!r}. It is a MOMENT, not a rate: the "
        "dipole_out port refuses to be wired to anything carrying Hz, and the "
        "units field is what makes that refusal possible."
    )


def test_every_family_declares_the_dipole_exactly_once(model: SCWBD) -> None:
    """Twice is as broken as never, and quieter.

    The cortical families used to declare ``dipole`` themselves. Adding it to the
    shared prefix without removing that gives them two dipole spans -- the head
    reads the shared one, the dynamics write the private one, and nothing raises.
    """
    for fam in model.family_layout.families:
        names = [c.name for c in fam.layout.components]
        assert names.count("dipole") == 1, (
            f"family {fam.name!r} declares dipole {names.count('dipole')} times: {names}"
        )
        assert names.index("dipole") == 4, (
            f"family {fam.name!r} has dipole at offset index {names.index('dipole')}; "
            "the shared prefix must be at identical offsets in every family or the "
            "head reads a different quantity depending on the region"
        )


def test_source_moment_returns_a_three_vector(model: SCWBD) -> None:
    """The check that would have failed for the whole of run 2.

    ``source_moment`` returns ``None`` when the state declares no ``dipole``.
    That is a deliberate compatibility path for a scalar-state model, and it is
    also exactly how this went unnoticed: the head reported "no dipole here" and
    the caller took it as "this model does not use one".
    """
    an = load_anatomy()
    width = sum(c.dim for c in model.layout.components)
    x = torch.zeros(2, 3, an.n_regions, width)
    out = model.eeg.source_moment(x)
    assert out is not None, (
        "source_moment() returned None -- the head cannot see the dipole. This is "
        "the O-5b defect and it is silent by construction."
    )
    assert tuple(out.shape) == (2, 3, an.n_regions, 3), tuple(out.shape)


def test_subcortex_has_no_orientation_but_may_have_a_zero_moment() -> None:
    """The one place a zero fill is correct, and the reason it is.

    ``ARCHITECTURE.md`` §7 rule 1 forbids imputing an unobserved value. O-5b
    writes zero into the subcortical dipole anyway, and the distinction that
    makes it sound rather than an exception is:

    * absent **orientation** stays ``NaN`` -- a direction of zero length is a
      lie, and averaging it in would tilt the field;
    * absent **moment** genuinely *is* the zero vector -- a parcel with no
      cortical sheet contributes no current dipole, and zero contributes exactly
      zero through ``L_vec`` rather than contributing a fabricated small one.

    So this asserts the NaN survives. If a future change "fixes" the NaN normals
    by filling them, the zero-moment argument silently stops holding.
    """
    an = load_anatomy()
    normal = np.asarray(an.normal)
    div = list(an.division)
    sub = [i for i, d in enumerate(div) if d != "cortex"]
    cortex = [i for i, d in enumerate(div) if d == "cortex"]
    assert len(sub) == 14 and len(cortex) == 400, (len(sub), len(cortex))
    assert np.all(np.isnan(normal[sub])), (
        "a subcortical region has a finite cortical normal. Either the anatomy "
        "gained real orientation data -- in which case O-5b's zero-moment "
        "argument needs revisiting -- or a NaN was filled in, which is the "
        "imputation §7 rule 1 forbids."
    )
    assert np.all(np.isfinite(normal[cortex])), "a cortical region lost its normal"


def test_the_padding_cost_is_recorded_rather_than_absorbed(model: SCWBD) -> None:
    """Giving 14 regions a zero dipole widens the padded plane, and that is a cost.

    O-6 already argues the padded layout stores roughly twice the state it uses
    and should be replaced by a ragged one. This change makes that slightly
    worse, deliberately, and the number is pinned so the trade stays visible
    instead of being discovered later as drift.
    """
    frac = float(model.family_layout.padding_fraction())
    assert 0.49 <= frac <= 0.51, (
        f"padding fraction is {frac:.4f}; it was 0.4734 before dipole entered the "
        "shared prefix and 0.4973 after. A large move means the family widths "
        "changed for some other reason -- find it rather than widening this bound."
    )


def test_the_published_checkpoint_still_loads_in_its_own_layout() -> None:
    """O-5b widened D from 59 to 62, and run 2's weights are 59 wide.

    ``ARCHITECTURE.md`` deferred this change precisely because it "invalidates
    the checkpoints of the run currently training". The run finished, so the
    deferral expired — but the breakage did not. The first strict load after the
    change failed on thirteen tensors, which means the *published* artifact could
    no longer be evaluated from the tree that documents it.

    A published model that its own repository cannot load is a broken artifact,
    not a completed migration. The checkpoint records its own ``state_layout``,
    so the era does not have to be guessed.
    """
    import torch

    from scwbd.foundation.families import layout_of_checkpoint

    ckpt = REPO / "checkpoints/scwbd-002-pilot/last.pt"
    if not ckpt.is_file():
        pytest.skip("run-2 checkpoint not on disk")

    an = load_anatomy()
    cfg = load_config(CONFIG)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    with layout_of_checkpoint(ckpt):
        old = SCWBD(cfg.model, an)
        assert sum(c.dim for c in old.layout.components) == 59
        old.load_state_dict(state)  # strict: raises on any mismatch

    # And the switch is scoped, not global -- a process must be able to hold both
    # eras, or evaluating run 2 would silently downgrade every model built after
    # it in the same session.
    new = SCWBD(cfg.model, an)
    assert sum(c.dim for c in new.layout.components) == 62
    with pytest.raises(RuntimeError):
        new.load_state_dict(state)
