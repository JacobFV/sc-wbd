"""Integrity tiers: the thesis's source-role hierarchy read as an ordering.

Appendix B fixes seven roles and says they are "deliberately non-equivalent".
It does **not** say in which order a curriculum should admit them --- that is the
gap this module closes.  The ordering below is not a new axiom; it is the
existing non-equivalence sorted by *how much of the representation a source is
entitled to found*:

===== ================================ ==================================
tier   what it is                       why it sits there
===== ================================ ==================================
1      likelihood, measured             measured evidence about a real brain
                                        through a declared observation model
2      boundary target / calibration    supervises a sensor/body/environment
                                        port; estimates gains, coordinates and
                                        nuisances --- never biology
3      prior, population                shapes population parameters *before*
                                        individual evidence
4      simulator-conditioned            "cannot establish biological validity"
                                        (body.tex 6.3); exercises regimes the
                                        measured data does not cover
5      distillation                     teacher-derived, discrepancy-inflated;
                                        "never a subject likelihood"
===== ================================ ==================================

Two roles carry no tier at all because they never contribute a gradient:
``negative_control`` and ``evaluation_only``.  They are recorded as
:data:`TIER_NONE` rather than omitted --- an unranked source and an absent source
must not look alike (``reports/decorative_guards.md``, "the absence variant").

**The tier is derived from the card, never from a table.**  In particular tier 3
and tier 4 share the *same* Appendix-B role (``prior``): the simulated corpus
carries ``role: prior`` because a simulator may not hold ``likelihood`` over
measured observables (``mixture.SourceSpec.__post_init__``).  What separates them
is ``is_simulated``, whose Python default is ``False``.  A card that simply omits
the field would therefore be silently promoted from tier 4 to tier 3 --- absence
reading as clean.  :func:`tier_of` refuses that case instead
(:data:`REFUSAL_UNDECLARED_PROVENANCE`), which is why it needs the card's *raw*
keys and not only its :class:`~scwbd.foundation.mixture.SourceSpec` projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from scwbd.foundation.mixture import ROLES, SourceSpec

__all__ = [
    "IntegrityTier",
    "TIERS",
    "TIER_NONE",
    "TIER_BY_NUMBER",
    "RawCard",
    "TierAssignment",
    "REFUSAL_UNDECLARED_PROVENANCE",
    "load_mixture_cards",
    "tier_of",
    "tier_table",
]

REFUSAL_UNDECLARED_PROVENANCE = "X07_undeclared_provenance"

#: A role that contributes no gradient has no place in an *ordering* of
#: gradient authority.  ``None`` would be indistinguishable from "not computed",
#: so the unranked case gets its own value.
TIER_NONE = 0


@dataclass(frozen=True)
class IntegrityTier:
    number: int
    name: str
    roles: tuple[str, ...]
    #: What this tier is entitled to do to the representation, in one sentence.
    entitlement: str
    #: Appendix B / body.tex sentence the placement rests on.
    citation: str


TIERS: tuple[IntegrityTier, ...] = (
    IntegrityTier(
        number=1,
        name="likelihood_measured",
        roles=("likelihood",),
        entitlement="may found the representation: it is the only kind of evidence about a real brain",
        citation="Appendix B: 'A likelihood factor contributes measured evidence about a latent "
        "variable through an observation model.'",
    ),
    IntegrityTier(
        number=2,
        name="boundary_and_calibration",
        roles=("boundary_target", "calibration"),
        entitlement="may fix ports, gains, coordinates and nuisances; may not supervise biology",
        citation="Appendix B: 'A boundary target supervises a sensor, body or environment port "
        "without licensing gradients through the unobserved brain.' Rejection condition: 'uses a "
        "calibration source as biological supervision.'",
    ),
    IntegrityTier(
        number=3,
        name="population_prior",
        roles=("prior",),
        entitlement="may shape population parameters, but only before individual evidence and only within its prior",
        citation="Appendix B: 'A prior shapes population parameters before individual evidence.' "
        "body.tex 6.4: 'soft constraints update only within their priors.'",
    ),
    IntegrityTier(
        number=4,
        name="simulator_conditioned",
        roles=("prior", "boundary_target"),
        entitlement="may extend into regimes the measured data does not cover; may not found anything",
        citation="body.tex 6.3: simulated data 'remain simulator-conditioned evidence and cannot "
        "establish biological validity.'",
    ),
    IntegrityTier(
        number=5,
        name="distillation",
        roles=("distillation",),
        entitlement="a discrepancy-inflated interface regulariser, off by default",
        citation="body.tex 6.3: 'It is a discrepancy-inflated interface regulariser, never a "
        "subject likelihood.'",
    ),
)

TIER_BY_NUMBER: dict[int, IntegrityTier] = {t.number: t for t in TIERS}


@dataclass
class RawCard:
    """A mixture card as it is written on disk, plus its typed projection.

    Both halves are kept because they answer different questions.  ``spec``
    says what the trainer will do; ``raw`` says what the *author* actually
    committed to, which is the only way to tell a declared ``false`` from an
    omitted field.
    """

    path: Path
    raw: dict[str, Any]
    spec: SourceSpec

    @property
    def id(self) -> str:
        return self.spec.id

    def declares(self, key: str) -> bool:
        return key in self.raw


@dataclass
class TierAssignment:
    source_id: str
    tier: int | None
    reason: str
    refusal: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.refusal is None


#: Curriculum-only fields, held in a sidecar rather than in the card.
#:
#: ``SourceSpec.load_dir`` constructs ``SourceSpec(**yaml)``, so a card carrying
#: any key the dataclass does not declare raises ``TypeError`` and the *trainer*
#: cannot load it.  A card the trainer cannot load is a worse failure than a
#: split record, so these live in ``card_metadata.yaml`` next to the card
#: directory and are merged into :attr:`RawCard.raw` on load.  ``declares()``
#: therefore reports them, and the split is invisible to every check.
SIDECAR_NAME = "card_metadata.yaml"
SIDECAR_FIELDS = ("observes", "dataset_id", "runtime_provenance_check")


def load_mixture_cards(path: str | Path, *, metadata: str | Path | None = None) -> dict[str, RawCard]:
    """Load a ``**/source_cards`` directory keeping the raw YAML keys.

    Merges ``<parent>/card_metadata.yaml`` when present.  A sidecar entry for an
    id with no card is an error, not a silent no-op: it would otherwise be
    indistinguishable from a field that was never written.
    """
    out: dict[str, RawCard] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mixture card directory {p} does not exist")

    side_path = Path(metadata) if metadata is not None else p.parent / SIDECAR_NAME
    side: dict[str, dict[str, Any]] = {}
    if side_path.exists():
        doc = yaml.safe_load(side_path.read_text()) or {}
        side = {str(k): dict(v or {}) for k, v in (doc.get("sources") or {}).items()}

    for f in sorted(p.glob("*.y*ml")):
        d = yaml.safe_load(f.read_text()) or {}
        kw = dict(d)
        for k in ("gradient_permission", "frozen", "losses", "compiler_permission"):
            if k in kw and isinstance(kw[k], list):
                kw[k] = tuple(kw[k])
        spec = SourceSpec(**kw)
        merged = dict(d)
        merged.update(side.get(spec.id, {}))
        out[spec.id] = RawCard(path=f, raw=merged, spec=spec)
    if not out:
        raise FileNotFoundError(f"no source cards found under {p}")

    orphan = sorted(set(side) - set(out))
    if orphan:
        raise KeyError(
            f"{side_path}: metadata for {orphan} but no card of that id under {p}. "
            "A sidecar entry that matches nothing is indistinguishable from a field nobody wrote."
        )
    return out


def tier_of(card: RawCard) -> TierAssignment:
    """Derive a card's integrity tier from its own declared fields.

    The order of the tests is load-bearing: ``is_simulated`` and ``is_teacher``
    are checked *before* ``role``, because a simulated corpus legitimately holds
    ``role: prior`` and a teacher legitimately holds ``role: distillation`` ---
    reading the role first would collapse tier 4 into tier 3.
    """
    s = card.spec
    if s.role not in ROLES:  # pragma: no cover - SourceSpec already refuses this
        return TierAssignment(s.id, None, f"unknown role {s.role!r}", refusal="X07_unknown_role")

    if s.role in ("negative_control", "evaluation_only"):
        return TierAssignment(
            s.id,
            TIER_NONE,
            f"role={s.role} contributes an audit, never a gradient; it takes no place in the "
            "integrity ordering",
            evidence={"role": s.role},
        )

    if s.is_teacher or s.role == "distillation":
        return TierAssignment(
            s.id, 5, "teacher-derived", evidence={"role": s.role, "is_teacher": s.is_teacher}
        )

    if s.is_simulated:
        return TierAssignment(
            s.id,
            4,
            "declares is_simulated: true",
            evidence={"role": s.role, "is_simulated": True, "model_discrepancy": s.model_discrepancy},
        )

    if s.role == "prior":
        # The tier-3/tier-4 discriminator has a False default.  An omitted field
        # must not be read as "measured".
        if not card.declares("is_simulated"):
            return TierAssignment(
                s.id,
                None,
                "role=prior with no declared provenance: tier 3 (population prior) and tier 4 "
                "(simulator-conditioned) share this role, and the field that separates them "
                "defaults to False. An omitted declaration would silently promote a simulated "
                "source one tier.",
                refusal=REFUSAL_UNDECLARED_PROVENANCE,
                evidence={"role": "prior", "declared_keys": sorted(card.raw)},
            )
        return TierAssignment(
            s.id, 3, "declares is_simulated: false", evidence={"role": "prior", "is_simulated": False}
        )

    if s.role in ("boundary_target", "calibration"):
        return TierAssignment(s.id, 2, f"role={s.role}", evidence={"role": s.role})

    if s.role == "likelihood":
        if s.is_simulated:  # pragma: no cover - SourceSpec refuses this combination
            return TierAssignment(
                s.id, None, "simulated likelihood", refusal="X07_simulated_likelihood"
            )
        return TierAssignment(s.id, 1, "role=likelihood on measured data", evidence={"role": "likelihood"})

    return TierAssignment(s.id, None, f"unhandled role {s.role!r}", refusal="X07_unhandled_role")


def tier_table(cards: Mapping[str, RawCard]) -> dict[str, TierAssignment]:
    return {sid: tier_of(c) for sid, c in sorted(cards.items())}
