"""Quarantined teacher cards and the guard that keeps them quarantined.

A teacher is not a dataset.  It measures nothing, it adds no participants, and
its output carries a model discrepancy that must be *measured* rather than
assumed (Appendix B, row "Bias, variance and discrepancy"; Appendix D, row
"Teacher/simulator domination").  Cards in this package are therefore kept out
of ``scwbd/sources/cards/`` so that no directory glob can sweep a teacher into
the training mixture, and they are loaded only through :func:`load_teacher_card`.

The guard implemented here, :func:`check_teacher_quarantine`, refuses a card
whose declarations would let a teacher behave like evidence.  Every refusal it
can raise is exercised by ``tests/sources/teachers/test_tribe_v2_card.py``
against a deliberately broken copy of the card: a guard nobody has watched fire
is indistinguishable from one that cannot fire.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Iterator

from scwbd.sources.cards import CARD_DIR, CardError, SourceCardDoc, load_card

TEACHER_DIR = Path(__file__).parent

#: The only role a teacher may hold.  Appendix B: a *distillation* factor
#: transfers representational organisation and carries teacher discrepancy; a
#: *likelihood* factor contributes measured evidence through an observation
#: model.  A teacher has no observation model, so the likelihood role is not
#: merely disallowed by policy -- it is unavailable in fact.
TEACHER_ROLE = "distillation"

#: Module/port families a teacher may never update, whatever a future
#: measurement says.  Preregistered here rather than in the card alone so that
#: editing one file cannot widen the boundary.  body.tex sec. 6.3: the factor
#: "receives no authority over latent mechanisms, subcortical or bodily state".
PERMANENTLY_FORBIDDEN: tuple[str, ...] = (
    "dynamics.coupling",
    "dynamics.conduction_delay",
    "dynamics.excitation_inhibition",
    "anatomy.subcortical.*",
    "physiology.bodily_state.*",
    "anatomy.connectome_prior",
    "intervene.*",
    "infer.population_prior",
)

#: Controls that must all be present in the preregistered branch before any arm
#: runs (body.tex sec. 6.3; Appendix D "Teacher/simulator domination").
REQUIRED_ABLATION_ARMS: tuple[str, ...] = (
    "no_teacher",
    "output_distillation",
    "intermediate_feature_distillation",
    "matched_generic_features",
    "generic_smoothness",
    "time_shuffled_teacher",
    "stimulus_mismatched_teacher",
    "perception_vs_imagery",
    "empirical_only",
)


def _is_unknown(value: Any) -> bool:
    """A field declares itself unresolved.

    Mirrors ``scwbd.sources.cards._is_unknown``: the literal ``unknown`` or a
    sentence that *begins* with it.  ``None`` is **not** unknown -- a null is a
    positive assertion of absence and must not be confused with an unmeasured
    quantity.
    """
    if isinstance(value, str):
        return value.strip().lower().startswith("unknown")
    return value == "unknown"


def iter_teacher_card_paths(directory: str | Path = TEACHER_DIR) -> Iterator[Path]:
    yield from sorted(Path(directory).glob("*.yaml"))


def load_teacher_card(path: str | Path, *, check: bool = True) -> SourceCardDoc:
    """Load a teacher card, validate it, and enforce the quarantine."""
    doc = load_card(path)
    if check:
        check_teacher_quarantine(doc)
    return doc


def check_teacher_quarantine(doc: SourceCardDoc) -> SourceCardDoc:
    """Refuse a teacher card that could act as evidence.

    Raises :class:`~scwbd.sources.cards.CardError` on any of:

    ``role``
        not ``distillation``.
    ``likelihood_kind``
        anything but ``none`` -- a teacher with a likelihood is a subject
        likelihood by another name.
    ``population.n_participants``
        non-zero.  A teacher adds no participants (Appendix A).
    ``ledger.model_discrepancy``
        numerically zero.  An unmeasured discrepancy is *unknown*, never 0.
    unmeasured discrepancy with open gradients
        ``model_discrepancy`` unknown while ``gradient_permission.allow`` is
        non-empty, or ``enabled`` true.  Appendix B: the field remains unknown
        and the corresponding gradient path is disabled.
    permanently forbidden targets
        any :data:`PERMANENTLY_FORBIDDEN` family appearing in ``allow`` or in
        ``preregistered_allow_if_validated``.
    incomplete preregistration
        a missing :data:`REQUIRED_ABLATION_ARMS` entry.
    location
        the card living in ``scwbd/sources/cards/``, where a glob would find it.
    """
    cid = doc.id
    data = doc.data

    if doc.path is not None and doc.path.resolve().parent == CARD_DIR.resolve():
        raise CardError(
            cid,
            "a teacher card must not live in scwbd/sources/cards/: load_all_cards() "
            "globs that directory and would place a teacher in the training mixture",
        )

    if doc.role != TEACHER_ROLE:
        raise CardError(
            cid, f"teacher role must be {TEACHER_ROLE!r}, not {doc.role!r} "
                 "(Appendix B: a teacher has no observation model and cannot be a likelihood)"
        )

    obs = data.get("observation") or {}
    kind = obs.get("likelihood_kind", "none")
    if kind != "none":
        raise CardError(
            cid, f"observation.likelihood_kind must be 'none' for a teacher, got {kind!r}"
        )

    pop = data.get("population") or {}
    n = pop.get("n_participants", 0)
    if not isinstance(n, int) or n != 0:
        raise CardError(
            cid, f"population.n_participants must be 0 for a teacher, got {n!r} "
                 "(Appendix A: distillation models add no empirical participants)"
        )

    ledger = data.get("ledger") or {}
    disc = ledger.get("model_discrepancy", "unknown")
    if isinstance(disc, (int, float)) and not isinstance(disc, bool) and disc == 0:
        raise CardError(
            cid, "ledger.model_discrepancy is 0 -- Appendix B lists zero teacher "
                 "variance as a rejection condition; an unmeasured discrepancy is 'unknown'"
        )

    gp = data.get("gradient_permission") or {}
    allow = list(gp.get("allow") or [])
    discrepancy_unmeasured = _is_unknown(disc) or disc is None

    if discrepancy_unmeasured and allow:
        raise CardError(
            cid, "ledger.model_discrepancy is unmeasured but gradient_permission.allow "
                 f"has {len(allow)} entr{'y' if len(allow) == 1 else 'ies'}: Appendix B "
                 "disables the gradient path of any field that remains unknown",
        )
    if discrepancy_unmeasured and gp.get("enabled", False):
        raise CardError(
            cid, "gradient_permission.enabled is true while ledger.model_discrepancy "
                 "is unmeasured (body.tex sec. 6.3: the experiment remains off by default)",
        )

    prereg = list(gp.get("preregistered_allow_if_validated") or [])
    for bucket, entries in (("allow", allow), ("preregistered_allow_if_validated", prereg)):
        for entry in entries:
            target = str((entry or {}).get("target", ""))
            for pattern in PERMANENTLY_FORBIDDEN:
                if target == pattern or fnmatch.fnmatch(target, pattern):
                    raise CardError(
                        cid, f"gradient_permission.{bucket} names {target!r}, which is "
                             "permanently forbidden to a teacher (body.tex sec. 6.3: no "
                             "authority over latent mechanisms, subcortical or bodily state)",
                    )

    branch = data.get("preregistered_ablation_branch") or {}
    arms = {str((a or {}).get("id")) for a in (branch.get("arms") or [])}
    missing = [a for a in REQUIRED_ABLATION_ARMS if a not in arms]
    if missing:
        raise CardError(
            cid, "preregistered_ablation_branch is missing required control arm(s): "
                 f"{missing} (Appendix D: distillation is retained only when it improves "
                 "empirical prediction beyond matched computation)",
        )

    return doc


__all__ = [
    "PERMANENTLY_FORBIDDEN",
    "REQUIRED_ABLATION_ARMS",
    "TEACHER_DIR",
    "TEACHER_ROLE",
    "check_teacher_quarantine",
    "iter_teacher_card_paths",
    "load_teacher_card",
]
