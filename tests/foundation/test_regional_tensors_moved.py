"""The regional tensors must have MOVED, measured on the weights themselves.

`test_card_patterns_reach_the_model.py` is the mechanism half: it asserts that
some enabled card's glob matches every module's parameter names. This file is
the measurement half: it asserts that the tensors are no longer bit-identical to
their initialisation.

They are independent, and run 2 failed both without either being checked. The
cards granted `local.*` against modules called `family_local`, so 88.8% of the
parameters could not receive a gradient -- and nothing measured whether they
had, because an unmatched glob is an empty permission set rather than an error
and the loss fell anyway.

Two properties this checks that a loss curve cannot:

* a module that is entirely bit-identical to its initialisation while sitting on
  the forward pass;
* `residual_ratio == 0.0` **exactly**. The residual output projections ship
  zero-initialised, so an exact zero is not a small effect -- it is the
  signature of a residual that never trained. A tolerance would hide it.

The comparison is a bit comparison via sha256, on purpose. The question is "did
this tensor receive a gradient at all", and any answer involving a threshold
invites the threshold to be tuned until the answer is yes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._runs import parametrize_runs, raw_stages, training_runs

REPO = Path(__file__).resolve().parents[2]
RUN3 = REPO / "checkpoints/scwbd-003"
SMOKE = REPO / "checkpoints/scwbd-003-smoke"

#: Modules that carry the regional model. These are the ones run 2 froze.
REGIONAL = ("family_local", "family_residual", "family_readout")

#: Modules added in run 3 that were unreachable before it, so a regression to
#: "declared but frozen" is the specific thing worth catching.
RUN3_NEW = ("behaviour", "eeg_montages", "tms_drive")


def _checkpoints() -> list[tuple[object, Path]]:
    """``(run, checkpoint)`` for every discovered run, plus the run-3 smoke dir.

    Was `(RUN3, SMOKE)`. The run whose regional tensors most need checking is the
    one about to launch, and pinning this to run 3 meant run 4 could not reach
    the gate HANDOFF-003 calls a launch precondition. The run is carried beside
    the path because `_founding_exempt` has to read THAT run's stage block --
    resolving a run-4 stage name against run 3's config would silently return
    the wrong exemption set.
    """
    out: list[tuple[object, Path]] = []
    for run in training_runs():
        cks = [c for c in run.checkpoints if c.name.startswith("stage_")] or [
            c for c in run.checkpoints if c.name == "last.pt"
        ]
        out += [(run, c) for c in cks]
    if SMOKE.is_dir():
        smoke = sorted(SMOKE.glob("stage_*.pt")) or sorted(SMOKE.glob("last.pt"))
        run3 = next((r for r in training_runs() if r.run_id == "run3"), None)
        out += [(run3, c) for c in smoke]
    return out


def _ckpt_id(pair) -> str:
    run, path = pair
    return f"{path.parent.name}/{path.name}"


def _load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _moved(ck: dict) -> dict:
    # An architecture-only checkpoint is written before any step, purely to give
    # `test_card_patterns_reach_the_model` a set of parameter names to match the
    # cards against. Nothing in it has moved and nothing should have; asking
    # this question of it is a category error, not a finding. It says so in its
    # own `stage` field rather than being recognised by filename.
    if ck.get("stage") == "architecture_only":
        pytest.skip("architecture-only checkpoint: written before any training step")
    rep = (ck.get("extra") or {}).get("moved_since_init")
    if not rep:
        pytest.skip(
            f"checkpoint carries no `moved_since_init`; it predates the "
            "fingerprint and cannot answer this question"
        )
    return rep


def _founding_exempt(run, stage: str) -> set[str]:
    """Modules a stage is *permitted* to leave frozen, derived not listed.

    Two independent sources, both auditable, neither a hard-coded allowance in
    this file:

    * ``configs/curriculum/tiers.yaml`` ``founding_exemptions`` — a module the
      policy says no tier-1 source can found. ``posterior.*`` was always there;
      ``family_readout.*`` was added when SC-WBD-003 measured it at 0 of 36 in a
      tier-1-only stage and the reason turned out to be structural: nothing in
      tier 1 reads ``rollout.activity``, so nothing in tier 1 can score a
      per-parcel activity readout.
    * ``configs/run3/scwbd-003.yaml`` — a module the stage's own
      ``tier_permissions`` do not grant at all. ``tms_drive.*`` is granted in
      T5 alone, deliberately, so a simulator stage cannot fit the pulse to
      synthetic dynamics.

    Deriving it keeps the test sharp: ``family_readout`` frozen in T4 or
    ``tms_drive`` frozen in T5 would still fail, because by then both are
    reachable.
    """
    exempt: set[str] = set()
    tiers = REPO / "configs/curriculum/tiers.yaml"
    if not tiers.is_file() or run is None:
        return exempt
    import yaml

    block = next(
        (
            ((s.get("extra") or {}).get("curriculum") or {})
            for s in raw_stages(run)
            if s.get("name") == stage
        ),
        None,
    )
    if block is None:
        return exempt
    admits = set(block.get("admits") or [])
    perms = block.get("tier_permissions") or {}
    granted = [str(g) for gs in perms.values() for g in gs]

    for entry in (yaml.safe_load(tiers.read_text()).get("policy") or {}).get(
        "founding_exemptions", []
    ):
        if int(entry.get("granted_to_tier", 0)) not in admits:
            for g in entry.get("globs", []):
                exempt.add(str(g).split(".")[0])

    for mod in (*REGIONAL, *RUN3_NEW):
        if not any(g.split(".")[0] == mod for g in granted):
            exempt.add(mod)
    return exempt


@pytest.mark.parametrize("run_ckpt", _checkpoints(), ids=_ckpt_id)
def test_the_regional_modules_are_not_at_their_initialisation(run_ckpt) -> None:
    """The check HANDOFF-003 makes a launch precondition."""
    run, ckpt = run_ckpt
    ck = _load(ckpt)
    rep = _moved(ck)
    by = rep["by_module"]
    exempt = _founding_exempt(run, str(ck.get("stage") or ""))
    dead = []
    for m in REGIONAL:
        e = by.get(m)
        if e is None or m in exempt:
            continue  # pooled arm, or a documented founding exemption
        if e["moved"] == 0:
            dead.append(f"{m} ({e['frozen']} tensors, all bit-identical to init)")
    assert not dead, (
        f"{ckpt.name}: the regional model did not train: {dead}. This is run 2's "
        "defect measured directly on the weights. Check which card is expected to "
        "grant these modules and whether its glob matches the names they actually "
        "have -- `fnmatch('family_local.ports.out_proj.weight', 'local.*')` is False."
    )


@pytest.mark.parametrize("run_ckpt", _checkpoints(), ids=_ckpt_id)
def test_the_attachment_kinds_added_in_run3_received_gradient(run_ckpt) -> None:
    """`behaviour`, the montage heads and the TMS drive are new and reachable.

    Each existed on the forward path before it had a source. The point of run 3
    is that each now has one, so "present and frozen" is exactly the regression
    to catch.
    """
    run, ckpt = run_ckpt
    ck = _load(ckpt)
    rep = _moved(ck)
    by = rep["by_module"]
    exempt = _founding_exempt(run, str(ck.get("stage") or ""))
    dead = [
        f"{m} ({by[m]['frozen']} tensors)"
        for m in RUN3_NEW
        if m in by and by[m]["moved"] == 0 and m not in exempt
    ]
    assert not dead, (
        f"{ckpt.name}: {dead} are on the forward path and bit-identical to their "
        "initialisation. Either the source that should reach them is not admitted "
        "by this stage, or its grant pattern names nothing."
    )


@pytest.mark.parametrize("run_ckpt", _checkpoints(), ids=_ckpt_id)
def test_most_of_the_model_moved(run_ckpt) -> None:
    """The headline number run 2 got wrong: 88.8% could not receive a gradient.

    A generous floor. It is not a quality bar -- a stage may legitimately freeze
    a head whose source it does not admit -- but a run in which most of the model
    is at its initialisation is the failure this whole file exists for.
    """
    run, ckpt = run_ckpt
    rep = _moved(_load(ckpt))
    frac = rep["n_moved"] / max(rep["n_parameters"], 1)
    assert frac > 0.5, (
        f"{ckpt.name}: only {rep['n_moved']}/{rep['n_parameters']} tensors "
        f"({frac:.1%}) differ from their initialisation. Run 2 shipped at 11.2%. "
        f"Frozen by module: "
        f"{ {k: v['frozen'] for k, v in rep['by_module'].items() if v['frozen']} }"
    )


@pytest.mark.parametrize("run_ckpt", _checkpoints(), ids=_ckpt_id)
def test_the_residual_ratio_is_not_exactly_zero(run_ckpt) -> None:
    """`residual_ratio == 0.0` exactly is a frozen residual, not a small effect.

    The residual output projections are zero-initialised, so the ratio starts at
    exactly zero and stays there if the module never trains. Any real training
    moves it off zero; the assertion is therefore against the exact value and
    needs no tolerance to choose.
    """
    run, ckpt = run_ckpt
    ck = _load(ckpt)
    metrics = ck.get("metrics") or {}
    extra = ck.get("extra") or {}
    ratio = metrics.get("residual_ratio", extra.get("residual_ratio"))
    if ratio is None:
        pytest.skip("this checkpoint records no residual_ratio")
    assert float(ratio) != 0.0, (
        f"{ckpt.name}: residual_ratio is exactly 0.0. The residual output "
        "projections ship zero-initialised, so this is the signature of a "
        "residual that never received a gradient -- not a residual with a small "
        "effect."
    )


@pytest.mark.parametrize("run_ckpt", _checkpoints(), ids=_ckpt_id)
def test_every_parameter_has_a_recorded_initialisation(run_ckpt) -> None:
    """A module built after the fingerprint has no baseline, and reads as moved.

    ``tms_drive`` is constructed by ``build_data`` -- it only exists when the
    perturbation corpus is on disk -- so it is absent from the fingerprint taken
    in ``__init__``. ``moved_since_init`` then compared a hash against ``None``,
    which is unequal, so the drive reported as MOVED on a checkpoint that had
    never taken a step. Measured exactly that: ``tms_drive moved 4/4`` with
    every other module frozen, on the architecture-only checkpoint.

    That is a false pass in the guard run 3 exists for, on the module carrying
    its novel claim. The trainer now registers late-built modules; this asserts
    the set stays empty so the same thing cannot come back through another one.
    """
    run, ckpt = run_ckpt
    rep = _moved(_load(ckpt))
    unf = rep.get("unfingerprinted")
    if unf is None:
        pytest.skip("checkpoint predates the unfingerprinted field")
    assert unf == [], (
        f"{ckpt.name}: {unf} have no recorded initialisation, so their "
        "moved/frozen status is not evidence either way. Register the module "
        "with `_fingerprint_late_module` where it is built."
    )


@parametrize_runs
def test_the_initialisation_is_reproducible_so_a_resume_does_not_inflate_it(run) -> None:
    """`moved_since_init` must survive an interrupted run, and this is why it does.

    The fingerprint is taken in ``FoundationTrainer.__init__``, before any
    checkpoint is loaded. So after a resume it fingerprints a **fresh**
    initialisation and then has trained weights loaded over it — and the whole
    measurement is only meaningful if that fresh initialisation is bit-identical
    to the original one.

    It is, because ``set_determinism(cfg.train.seed)`` runs before the model is
    constructed. Remove or reorder that and every resumed run would report
    ~100% of tensors "moved" the instant it started, silently inflating run 3's
    central published claim in the direction that flatters it.

    A 13,400-step run over ~25 hours will plausibly be resumed at least once,
    which is what makes this worth asserting rather than assuming.
    """
    import hashlib

    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.config import load_config
    from scwbd.foundation.model import SCWBD
    from scwbd.foundation.util import set_determinism

    cfg = load_config(str(run.config))
    anat = load_anatomy(device="cpu")

    def fingerprint() -> str:
        set_determinism(cfg.train.seed)
        m = SCWBD(cfg.model, anat)
        h = hashlib.sha256()
        for name, p in sorted(m.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().numpy().tobytes())
        return h.hexdigest()

    assert fingerprint() == fingerprint(), (
        f"{run.run_id}: two constructions at the same seed produced different weights, so the "
        "initialisation this run is compared against is not reproducible. After "
        "any resume, `moved_since_init` would report nearly everything as moved "
        "regardless of training."
    )


def test_there_is_something_to_check() -> None:
    """Guards the guard: a parametrisation over an empty list passes vacuously.

    ``reports/decorative_guards.md`` names exactly this shape. With no run-3
    checkpoint on disk every test above collects zero cases and the file reports
    green while measuring nothing.
    """
    if not _checkpoints():
        pytest.skip("no run-3 checkpoint on disk yet")
    assert _checkpoints()
