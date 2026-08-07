"""Participant-grouped splits for the perturbation records, and their bound.

R10: group before splitting.  All six spTMS runs of a participant belong to one
person and must land in one fold; four sessions of one person are not four
subjects.  That part is mechanical and :class:`~scwbd.sources.splits.GroupedSplitter`
already does it.

The part that is not mechanical is what a split over **two** participants can
support.  A leave-one-participant-out split over N=2 is constructible — it is
two folds, each training on one person and testing on the other — and it is
genuinely leakage-free.  It is also incapable of estimating between-participant
variance, because each fold's test set contains exactly one person and there is
no replication to average over.  Any statistic it produces is a pair of numbers,
not a population estimate with an interval.

:func:`participant_split` therefore returns the split *and*
:func:`split_bound` states what it cannot support, so a consumer records the
bound rather than rediscovering it.  ``ds004024.yaml`` ``split_policy.notes``
already forbids pooling the Fisher-rank claim across participants; at N=2 that
rule is not a stylistic preference, it is the only thing confining the residual
MNAR risk in upstream session attendance (``known_issues.md`` ISSUE-005).
"""

from __future__ import annotations

from typing import Any, Sequence

from ..lineage import Record
from ..splits import GroupedSplitter, Split, leakage_audit

#: Below this many participants, a held-out-person split yields no
#: between-participant variance estimate.
MIN_PARTICIPANTS_FOR_VARIANCE = 3


def participant_split(records: Sequence[Record], *, seed: int = 0) -> Split:
    """Leave-one-participant-out over the perturbation records.

    Groups on ``family`` (declared ``singleton:<participant>``) so that a
    participant's runs never straddle a fold.  Raises
    :class:`~scwbd.sources.lineage.LineageError` when only one group is present,
    which is the correct outcome: a held-out-person split over one person is not
    evaluable and must not be silently downgraded to a run-level split.
    """
    n_groups = len({r.lineage.group_key("family") for r in records})
    return GroupedSplitter("participant", n_folds=max(2, n_groups), seed=seed).split(records)


def split_bound(records: Sequence[Record]) -> dict[str, Any]:
    """What this split can and cannot support, given the participants present."""
    participants = sorted({r.lineage.participant for r in records if r.lineage.participant})
    n = len(participants)
    return {
        "n_participants": n,
        "participants": participants,
        "n_records": len(records),
        "participant_holdout_constructible": n >= 2,
        "between_participant_variance_estimable": n >= MIN_PARTICIPANTS_FOR_VARIANCE,
        "pooling_permitted": False,
        "bound": (
            f"N={n} participants. A leave-one-participant-out split is "
            f"{'constructible' if n >= 2 else 'NOT constructible'}, but each test "
            "fold holds "
            f"{'exactly one person' if n == 2 else f'{max(n // 2, 1)} person(s)'}"
            ", so between-participant variance is "
            f"{'estimable' if n >= MIN_PARTICIPANTS_FOR_VARIANCE else 'NOT estimable'}"
            ". Report per participant; never pool. Pooling across participants "
            "would re-open the residual MNAR component of upstream session "
            "attendance that ds004024.yaml confines by per-participant reporting "
            "(known_issues.md ISSUE-005)."
        ),
    }


def audit(split: Split, records: Sequence[Record]) -> dict[str, Any]:
    """Run Ada's leakage audit and return it as a dict alongside the bound."""
    rep = leakage_audit(split, records)
    return {
        "ok": rep.ok,
        "summary": rep.summary(),
        "violations": [str(v) for v in rep.violations],
        "bound": split_bound(records),
    }
