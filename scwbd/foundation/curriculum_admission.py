"""Stage admission read from the **config**, not from the stage's name.

``FoundationTrainer.run_stage`` currently decides what a stage sees with six
separate tests on ``stage.name``.  Two of them gate source admission and are the
reason the integrity ordering cannot be corrected by editing a config
(:mod:`scwbd.curriculum.spec`, refusal ``X06``).  The other four were found while
preparing run 2 and are listed in :data:`NAME_GATES` because *an unexercised code
path has no bug count, only a lower bound of one* --- a run whose stages are not
called ``I_regional`` … ``V_individual`` silently takes the default branch of all
six, and one of those defaults is **permissive**.

This module is the replacement decision.  It is deliberately a *pure function of
the config and the source cards*, so that:

* the admission a run performed can be read out of the artifact it wrote, and
* an undeclared stage **raises** rather than defaulting to "everything".

Nothing here imports :mod:`scwbd.foundation.train`, so it can be unit-tested and
validated while a training job is live.

The trainer-side change is carried as an unapplied patch --- see
``configs/run2/patches/`` --- because run 1 was on the GPU when this was written
and modifying source under a running job is how a provenance claim gets
destroyed (``reports/decorative_guards.md``, "A corollary about fixing things").

Why not just fix the two admission gates
----------------------------------------
Because the four remaining ones are not cosmetic for a renamed curriculum:

``STAGE_PERMISSIONS.get(stage.name, ("*",))``
    the stage allowlist.  An unknown stage name gets ``("*",)``, i.e. **no
    restriction at all**.  Run 2's stages are named ``T1_measured_founding`` …,
    so every per-stage mask in the corrected curriculum would silently widen to
    the union of the source cards' own permissions.  This is the register's
    absence variant with the sign flipped: the *default* is the permissive case,
    so a missing entry reads as "allowed".
``stage.name == "I_regional"``
    boundary randomisation / corrupted inputs.  Silently off for run 2.
``stage.name in ("IV_assembly",)``
    ``with_hemo`` in the rollout.  Silently off for run 2.
``stage.name == "V_individual"``
    constructs the :class:`Individualizer` and narrows the optimiser's parameter
    list to it.  Silently never constructed for run 2, so the individualisation
    stage would train nothing it was supposed to.

Each becomes an explicit declaration in ``extra.curriculum`` and each has a
recorded default that matches run 1's behaviour for the stage names run 1 used,
so the reconstruction of an old config is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NAME_GATES",
    "UndeclaredStage",
    "StageAdmission",
    "stage_admission",
    "admitted_source_ids",
    "legacy_stage_flags",
    "assert_region_count",
]


#: Every ``stage.name`` test in ``FoundationTrainer.run_stage`` and its callees,
#: at ``3400cee`` (branch ``master``).  Recorded as data so a test can assert the
#: list is exhaustive rather than trusting that someone read the function
#: carefully.  ``(line, expression, what it decides, default for an unknown name)``
NAME_GATES: tuple[tuple[int, str, str, str], ...] = (
    (303, 'STAGE_PERMISSIONS.get(stage.name, ("*",))', "the stage's gradient allowlist", "PERMISSIVE: ('*',)"),
    (498, 'stage.name == "I_regional"', "boundary randomisation of the sim inputs", "off"),
    (510, 'stage.name in ("IV_assembly",)', "haemodynamic state in the rollout", "off"),
    (585, 'stage.name == "V_individual"', "construct the Individualizer; narrow the optimiser", "off"),
    (613, 'stage.name != "V_individual"', "admit the SIMULATED sources", "admitted"),
    (617, 'stage.name in ("III_sliced", "IV_assembly", "V_individual")', "admit the MEASURED sources", "refused"),
)


class UndeclaredStage(RuntimeError):
    """A stage carries no ``extra.curriculum`` block, so its admission is unknown.

    Raised rather than defaulted.  ``STAGE_PERMISSIONS.get(name, ("*",))``
    answers this question with *everything*, which is the one answer that cannot
    be wrong-looking: an unwired stage and a fully-permitted stage produce the
    same reading.
    """


@dataclass(frozen=True)
class StageAdmission:
    """What one stage admits, and with what permissions --- all from the config."""

    stage: str
    #: integrity tiers this stage admits, as declared
    admits: tuple[int, ...] = ()
    #: source ids resolved from the cards, sorted; the mixture keys
    source_ids: tuple[str, ...] = ()
    #: per admitted tier, the stage's declared glob restriction (may be empty =
    #: "do not restrict beyond the card")
    tier_permissions: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    #: tiers admitted for which no enabled card exists.  Not an error here --- the
    #: validator's X08 refuses an *unrecorded* absence; this field is what makes
    #: the absence available to the trainer's own report.
    absent_tiers: tuple[int, ...] = ()
    #: the four non-admission behaviours that used to be keyed on the stage name
    boundary_randomisation: bool = False
    with_hemo: bool = False
    individualize: bool = False
    provenance: str = "config:extra.curriculum"

    # -- queries the trainer makes ------------------------------------------
    def admits_simulated(self, sources: Mapping[str, Any]) -> bool:
        return any(getattr(sources[s], "is_simulated", False) for s in self.source_ids if s in sources)

    def admits_measured(self, sources: Mapping[str, Any]) -> bool:
        return any(
            not getattr(sources[s], "is_simulated", False)
            and getattr(sources[s], "role", "") == "likelihood"
            for s in self.source_ids
            if s in sources
        )

    def allow_globs(self) -> tuple[str, ...]:
        """The stage allowlist, replacing ``STAGE_PERMISSIONS[stage.name]``.

        The union over admitted tiers, because the allowlist is intersected with
        each *source's own* ``A_k`` afterwards; a per-tier restriction that is
        tighter than the union is still enforced by
        :meth:`permits_for_tier`.  An empty union means "the config declared no
        restriction", which is returned as ``("*",)`` **only** when the stage
        declared at least one tier --- never as a default for a missing block.
        """
        out: list[str] = []
        for t in sorted(self.tier_permissions):
            out.extend(self.tier_permissions[t])
        if not out:
            return ("*",)
        return tuple(dict.fromkeys(out))

    def permits_for_tier(self, tier: int) -> tuple[str, ...]:
        return tuple(self.tier_permissions.get(tier, ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "admits": list(self.admits),
            "source_ids": list(self.source_ids),
            "tier_permissions": {k: list(v) for k, v in sorted(self.tier_permissions.items())},
            "absent_tiers": list(self.absent_tiers),
            "boundary_randomisation": self.boundary_randomisation,
            "with_hemo": self.with_hemo,
            "individualize": self.individualize,
            "provenance": self.provenance,
        }


# ----------------------------------------------------------------------
#: What each of run 1's stage names did, so an *undeclared* legacy config keeps
#: its behaviour exactly.  This is a transcription of the six gates in
#: :data:`NAME_GATES`, and ``tests/foundation/test_curriculum_admission.py``
#: checks it against ``inspect.getsource`` rather than trusting the transcription.
_LEGACY_FLAGS: dict[str, dict[str, bool]] = {
    "I_regional": {"boundary_randomisation": True, "with_hemo": False, "individualize": False},
    "II_interface": {"boundary_randomisation": False, "with_hemo": False, "individualize": False},
    "III_sliced": {"boundary_randomisation": False, "with_hemo": False, "individualize": False},
    "IV_assembly": {"boundary_randomisation": False, "with_hemo": True, "individualize": False},
    "V_individual": {"boundary_randomisation": False, "with_hemo": False, "individualize": True},
}


def legacy_stage_flags(name: str) -> dict[str, bool] | None:
    """Run 1's behaviour for a run-1 stage name, or ``None`` if unknown."""
    d = _LEGACY_FLAGS.get(name)
    return dict(d) if d is not None else None


def _curriculum_block(stage: Any) -> Mapping[str, Any] | None:
    extra = getattr(stage, "extra", None) or {}
    blk = extra.get("curriculum") if isinstance(extra, Mapping) else None
    return blk if isinstance(blk, Mapping) else None


def stage_admission(
    stage: Any,
    cards: Mapping[str, Any] | None = None,
    *,
    cards_dir: str | Path | None = None,
    strict: bool = True,
) -> StageAdmission:
    """Resolve one stage's admission from its ``extra.curriculum`` block.

    ``stage`` is a :class:`scwbd.foundation.config.StageConfig`.  ``cards`` maps
    source id -> :class:`scwbd.curriculum.tiers.RawCard`; when omitted they are
    loaded from ``cards_dir``.  Tier assignment goes through
    :func:`scwbd.curriculum.tiers.tier_of`, the *same* derivation the validator
    uses --- comparing a tier computed here against a tier computed there would
    repeat defect 1 of the register, where a permission set was compared in one
    name space and enforced in another.

    ``strict=False`` falls back to :func:`legacy_stage_flags` for a stage that
    declares nothing but whose name run 1 knew.  That path exists so an old
    config keeps running unchanged; it is **not** available to a new stage name,
    which raises.
    """
    from scwbd.curriculum.tiers import TIER_NONE, load_mixture_cards, tier_of

    name = str(getattr(stage, "name", "stage"))
    blk = _curriculum_block(stage)
    if blk is None:
        legacy = legacy_stage_flags(name)
        if strict or legacy is None:
            raise UndeclaredStage(
                f"stage {name!r} declares no `extra.curriculum` block, so which sources it "
                "admits is unknown. Refusing to default: the previous default was "
                "STAGE_PERMISSIONS.get(name, ('*',)), which grants everything, so an "
                "unwired stage and a fully-permitted stage read identically."
            )
        return StageAdmission(stage=name, provenance="legacy:run_stage stage-name gates", **legacy)

    if cards is None:
        cards = load_mixture_cards(cards_dir or "configs/curriculum/source_cards")

    admits = tuple(int(t) for t in (blk.get("admits") or ()))
    perms = {int(k): tuple(v) for k, v in (blk.get("tier_permissions") or {}).items()}

    live: dict[int, list[str]] = {}
    for sid, card in sorted(cards.items()):
        if not card.spec.enabled:
            continue
        a = tier_of(card)
        if a.tier in (None, TIER_NONE):
            continue
        live.setdefault(a.tier, []).append(sid)

    source_ids = tuple(sorted({s for t in admits for s in live.get(t, ())}))
    absent = tuple(t for t in admits if not live.get(t))

    return StageAdmission(
        stage=name,
        admits=admits,
        source_ids=source_ids,
        tier_permissions=perms,
        absent_tiers=absent,
        boundary_randomisation=bool(blk.get("boundary_randomisation", False)),
        with_hemo=bool(blk.get("with_hemo", False)),
        individualize=bool(blk.get("individualize", False)),
    )


def admitted_source_ids(
    stage: Any, cards: Mapping[str, Any] | None = None, *, cards_dir: str | Path | None = None
) -> tuple[str, ...]:
    """Shorthand: the source ids a stage admits.  See :func:`stage_admission`."""
    return stage_admission(stage, cards, cards_dir=cards_dir).source_ids


# ----------------------------------------------------------------------
def assert_region_count(declared: int, anat: Any) -> None:
    """Refuse a config whose ``model.n_regions`` disagrees with the anatomy.

    ``SCWBD.__init__`` sets ``self.n_regions = anat.n_regions`` and **never
    reads** ``cfg.model.n_regions``; at ``3400cee`` the only consumer of that
    field anywhere in the run path is a provenance *string* in
    ``scwbd.curriculum.validate.parameter_universe``.  So the field is a comment
    that formats itself into evidence: it will report 414 while a 454-region
    prior is loaded, and nothing anywhere disagrees.

    That is exactly how ``n_regions: 454  # Schaefer-400 + 32 + 22`` survived a
    whole night of run 1 --- the number was read as a specification when it was
    only a label.  Making the two agree *by construction* is cheaper than
    remembering to check (``reports/decorative_guards.md``, recommendation 7).
    """
    actual = int(getattr(anat, "n_regions", -1))
    if int(declared) != actual:
        raise ValueError(
            f"config declares model.n_regions={int(declared)} but the loaded anatomy has "
            f"{actual} regions (provenance={getattr(anat, 'provenance', '?')!r}, "
            f"is_biological={getattr(anat, 'is_biological', lambda: '?')()}). "
            "SCWBD takes its shape from the anatomy, so the config field is otherwise "
            "inert and would have recorded a parcellation the model never loaded."
        )
