"""``scwbd.curriculum`` --- the integrity-ordered curriculum and its validator.

The training curriculum is ordered by **data integrity**: measured evidence
founds the representation, and each lower-integrity tier enters afterwards with a
narrower gradient mask.  The ordering is the thesis's own source-role hierarchy
(Appendix B, "the permitted roles are deliberately non-equivalent") read as a
gradient of authority --- see :mod:`scwbd.curriculum.tiers`.

Nothing here trains.  It expresses a curriculum as
*(admitted tiers, gradient permissions, objective terms)*
(:mod:`scwbd.curriculum.spec`), derives each source's tier from its own card
rather than from a table, grounds per-tier permission narrowing in the measured
Fisher information (:mod:`scwbd.curriculum.information`), and refuses a
configuration whose ordering is inverted (:mod:`scwbd.curriculum.validate`).

    python -m scwbd.curriculum validate configs/scwbd_001_beta.yaml
"""

from __future__ import annotations

from .information import BlindRule, derive_blind_rules, load_modality_information
from .spec import Curriculum, FoundingExemption, StageCurriculum, TierPolicy, load_tier_policy
from .tiers import TIERS, IntegrityTier, RawCard, load_mixture_cards, tier_of, tier_table
from .validate import Refusal, Verdict, expand, parameter_universe, validate, validate_config

__all__ = [
    "TIERS",
    "IntegrityTier",
    "RawCard",
    "load_mixture_cards",
    "tier_of",
    "tier_table",
    "Curriculum",
    "StageCurriculum",
    "TierPolicy",
    "FoundingExemption",
    "load_tier_policy",
    "BlindRule",
    "derive_blind_rules",
    "load_modality_information",
    "Refusal",
    "Verdict",
    "validate",
    "validate_config",
    "parameter_universe",
    "expand",
]
