"""Leakage-safe splitting (Appendix D, ``tab:mixture-evaluation``).

Rules implemented here, in the order they are enforced:

1. **Group before you split.**  Records are grouped by immutable lineage
   (participant -> family -> site -> device -> session -> run -> trial) and by
   derivation root *before* any randomisation.  Derived records (tractograms,
   re-references, augmentations) inherit their parent's group so that
   "different algorithms over one scan" never count as independent
   participants.
2. **Fail loudly on unresolved parentage.**  This is refusal ``R10``:
   :class:`~scwbd.sources.lineage.LineageError` is raised, never silently
   worked around.
3. **Audit after you split.**  :func:`leakage_audit` re-derives the grouping
   from the record set and checks duplicate participants/families, duplicate
   content hashes, derived-data duplication, stimulus overlap and residual
   site predictability of fold membership.

Nothing here fits a model, and nothing here resamples data.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

from .lineage import LINEAGE_ORDER, Lineage, LineageError, Record, resolve_parentage

SplitMode = Literal["participant", "site", "stimulus"]

#: Above this normalised mutual information between site and fold membership a
#: *non site-level* split is flagged: fold identity is partly predictable from
#: the scanner/device, so pooled accuracy may be a site shortcut.
SITE_NMI_WARN = 0.20


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    test_groups: tuple[str, ...]
    held_out_stimuli: tuple[str, ...] = ()

    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_ids)


@dataclass(frozen=True)
class Split:
    mode: SplitMode
    level: str
    seed: int
    folds: tuple[Fold, ...]
    group_of: Mapping[str, str]
    #: True when the holdout cannot be enforced by dropping whole records
    #: (stimulus mode over records that mix held-out and kept stimuli).
    requires_trial_masking: bool = False
    notes: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)


@dataclass(frozen=True)
class Violation:
    kind: str
    message: str
    code: str = "R10"
    offending: object = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}:{self.kind}] {self.message}"


@dataclass
class LeakageReport:
    split_mode: str
    level: str
    n_folds: int
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_leaky(self) -> None:
        if self.violations:
            raise LineageError(
                "leakage audit failed: " + "; ".join(str(v) for v in self.violations),
                offending_object=[v.offending for v in self.violations],
                remedy="Regroup by the parent-level key and re-split; do not reweight.",
            )

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"LeakageReport({self.split_mode}/{self.level}, {self.n_folds} folds): {head}"]
        lines += [f"  violation: {v}" for v in self.violations]
        lines += [f"  warning:   {w}" for w in self.warnings]
        for k, v in sorted(self.stats.items()):
            lines.append(f"  stat: {k} = {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# splitter
# --------------------------------------------------------------------------
class GroupedSplitter:
    """Group-then-split with participant-, site-, and stimulus-level holdout.

    Parameters
    ----------
    mode
        ``"participant"`` groups on the highest resolvable identity key
        (``family`` when the source declares families, else ``participant``)
        and assigns whole groups to folds.  ``"site"`` performs
        leave-site(-or-device)-out: one fold per site.  ``"stimulus"`` holds
        out stimuli/semantic families.
    n_folds
        Number of folds.  Ignored in ``"site"`` mode, where the number of
        folds is the number of distinct sites.
    seed
        Explicit seed (ARCHITECTURE.md §3: determinism is a test).
    level
        Override the grouping level.  Defaults to ``"family"`` for participant
        mode when any record declares a family, else ``"participant"``; and to
        ``"site"`` / ``"device"`` for site mode.
    """

    def __init__(
        self,
        mode: SplitMode = "participant",
        *,
        n_folds: int = 5,
        seed: int = 0,
        level: str | None = None,
    ) -> None:
        if mode not in ("participant", "site", "stimulus"):
            raise ValueError(f"unknown split mode {mode!r}")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.mode = mode
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.level = level

    # -- grouping ------------------------------------------------------
    def _resolve_level(self, records: Sequence[Record]) -> str:
        if self.level is not None:
            if self.level not in LINEAGE_ORDER:
                raise ValueError(f"unknown lineage level {self.level!r}")
            return self.level
        if self.mode == "site":
            return "site"
        if self.mode == "stimulus":
            return "trial"
        has_family = any(r.lineage.family is not None for r in records)
        if has_family:
            missing = [r for r in records if r.lineage.family is None]
            if missing:
                raise LineageError(
                    f"{len(missing)} record(s) have no family key while others do; "
                    "relatives could be split across folds",
                    offending_object=missing[0],
                    remedy=(
                        "Declare family='singleton:<participant>' for unrelated "
                        "participants so the grouping level is uniform."
                    ),
                )
            return "family"
        return "participant"

    def group_keys(self, records: Sequence[Record]) -> dict[str, str]:
        """Return record id -> immutable group key.  Raises R10 on failure."""
        records = list(records)
        if not records:
            raise ValueError("no records to split")
        level = self._resolve_level(records)
        root = resolve_parentage(records)
        by_id = {r.id: r for r in records}
        keys: dict[str, str] = {}
        # site and device are shared across participants, so they are grouped by
        # their bare value; identity levels are grouped by their full ancestry
        # so that two participants' "ses-01" never merge.
        flat = level in ("site", "device")
        for rec in records:
            ancestor = by_id[root[rec.id]]
            # the derivation root owns the group; a derivative can never
            # obtain a group of its own (Appendix D "Derived-data duplication")
            lin = ancestor.lineage
            keys[rec.id] = lin.value_key(level) if flat else lin.group_key(level)
        return keys

    # -- splitting -----------------------------------------------------
    def split(self, records: Iterable[Record]) -> Split:
        records = list(records)
        level = self._resolve_level(records)
        if self.mode == "stimulus":
            return self._split_stimulus(records)
        keys = self.group_keys(records)
        groups = sorted(set(keys.values()))
        if self.mode == "site":
            fold_of = {g: i for i, g in enumerate(groups)}
            n_folds = len(groups)
            if n_folds < 2:
                raise LineageError(
                    "site-level holdout requested but the record set contains "
                    f"{n_folds} site(s); leave-site-out is not evaluable",
                    offending_object=groups,
                    remedy="Add a second site/device or use participant-level holdout.",
                )
        else:
            n_folds = min(self.n_folds, len(groups))
            if n_folds < 2:
                raise LineageError(
                    f"only {len(groups)} independent group(s) at level {level!r}; "
                    "a held-out-person split is not evaluable",
                    offending_object=groups,
                )
            rng = random.Random(self.seed)
            shuffled = list(groups)
            rng.shuffle(shuffled)
            fold_of = {g: i % n_folds for i, g in enumerate(shuffled)}

        folds: list[Fold] = []
        for f in range(n_folds):
            test = tuple(sorted(r.id for r in records if fold_of[keys[r.id]] == f))
            train = tuple(sorted(r.id for r in records if fold_of[keys[r.id]] != f))
            tgroups = tuple(sorted({keys[i] for i in test}))
            folds.append(Fold(index=f, train_ids=train, test_ids=test, test_groups=tgroups))
        notes = ()
        if self.mode == "site":
            notes = (
                "leave-site-out: fold index == site index; site predictability of "
                "fold membership is 1.0 by construction and is not a violation",
            )
        return Split(
            mode=self.mode,
            level=level,
            seed=self.seed,
            folds=tuple(folds),
            group_of=keys,
            notes=notes,
        )

    def _split_stimulus(self, records: Sequence[Record]) -> Split:
        stimuli = sorted({s for r in records for s in r.stimulus_ids})
        if not stimuli:
            raise LineageError(
                "stimulus-level holdout requested but no record declares stimulus_ids",
                offending_object=records[0] if records else None,
                remedy="Populate Record.stimulus_ids from the source's events table.",
            )
        if "unknown" in stimuli:
            raise LineageError(
                "stimulus id 'unknown' present; refusing to hold out stimuli whose "
                "identity is unresolved",
                offending_object=[r.id for r in records if "unknown" in r.stimulus_ids][:5],
            )
        n_folds = min(self.n_folds, len(stimuli))
        if n_folds < 2:
            raise LineageError(
                f"only {len(stimuli)} distinct stimulus/stimulus-family; "
                "cross-stimulus generalisation is not evaluable",
                offending_object=stimuli,
            )
        rng = random.Random(self.seed)
        shuffled = list(stimuli)
        rng.shuffle(shuffled)
        fold_of = {s: i % n_folds for i, s in enumerate(shuffled)}

        # a record is "pure" if all its stimuli fall in one fold
        mixed = [r for r in records if len({fold_of[s] for s in r.stimulus_ids}) > 1]
        folds: list[Fold] = []
        for f in range(n_folds):
            held = tuple(sorted(s for s in stimuli if fold_of[s] == f))
            test = tuple(
                sorted(r.id for r in records if r.stimulus_ids and set(r.stimulus_ids) <= set(held))
            )
            train = tuple(
                sorted(
                    r.id
                    for r in records
                    if r.stimulus_ids and not (set(r.stimulus_ids) & set(held))
                )
            )
            folds.append(
                Fold(
                    index=f,
                    train_ids=train,
                    test_ids=test,
                    test_groups=held,
                    held_out_stimuli=held,
                )
            )
        notes: tuple[str, ...] = ()
        if mixed:
            notes = (
                f"{len(mixed)} record(s) contain stimuli from >1 fold; those records are "
                "excluded from both train and test unless the trial-level mask is applied",
            )
        return Split(
            mode="stimulus",
            level="stimulus",
            seed=self.seed,
            folds=tuple(folds),
            group_of={s: f"stim/{fold_of[s]}" for s in stimuli},
            requires_trial_masking=bool(mixed),
            notes=notes,
        )


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def _normalised_mutual_information(a: Sequence[str], b: Sequence[str]) -> float:
    """NMI(a; b) normalised by H(b), in nats-free [0, 1]. 0 when H(b)==0."""
    n = len(a)
    if n == 0:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    cab = Counter(zip(a, b))
    hb = -sum((c / n) * math.log(c / n) for c in cb.values())
    if hb <= 1e-12:
        return 0.0
    mi = 0.0
    for (x, y), c in cab.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((ca[x] / n) * (cb[y] / n)))
    return max(0.0, min(1.0, mi / hb))


def leakage_audit(split: Split, records: Iterable[Record] | None = None) -> LeakageReport:
    """Audit a :class:`Split` for the leakage modes of Appendix D.

    Checks
    ------
    * **participant/family duplication** — any group appearing in more than one
      fold's test set, or appearing in both train and test of the same fold.
    * **derived-data duplication** — identical ``content_hash`` or a shared
      derivation root spanning folds.
    * **stimulus overlap** — held-out stimuli reappearing in a fold's training
      records.
    * **site predictability** — normalised mutual information between site and
      fold membership (a warning outside site mode, expected in site mode).

    ``records`` may be omitted only for the group-level checks; the derived-data
    and site checks need them.
    """
    rep = LeakageReport(split_mode=split.mode, level=split.level, n_folds=len(split.folds))
    recs = list(records) if records is not None else []
    by_id = {r.id: r for r in recs}

    # ---- 1. group disjointness ---------------------------------------
    if split.mode != "stimulus":
        seen_group_fold: dict[str, int] = {}
        for fold in split.folds:
            for rid in fold.test_ids:
                g = split.group_of.get(rid)
                if g is None:
                    rep.violations.append(
                        Violation(
                            "unresolved_group",
                            f"record {rid!r} has no group key in the split",
                            offending=rid,
                        )
                    )
                    continue
                prev = seen_group_fold.setdefault(g, fold.index)
                if prev != fold.index:
                    rep.violations.append(
                        Violation(
                            "participant_across_folds",
                            f"group {g!r} appears in test folds {prev} and {fold.index}",
                            offending=g,
                        )
                    )
            train_groups = {split.group_of.get(r) for r in fold.train_ids}
            overlap = sorted({g for g in fold.test_groups if g in train_groups})
            if overlap:
                rep.violations.append(
                    Violation(
                        "train_test_group_overlap",
                        f"fold {fold.index}: group(s) {overlap[:5]} in both train and test",
                        offending=overlap,
                    )
                )
        rep.stats["n_groups"] = len(set(split.group_of.values()))

    # ---- 2. record duplication across folds --------------------------
    fold_of_record: dict[str, list[int]] = defaultdict(list)
    for fold in split.folds:
        for rid in fold.test_ids:
            fold_of_record[rid].append(fold.index)
    dup = {k: v for k, v in fold_of_record.items() if len(v) > 1}
    if dup:
        rep.violations.append(
            Violation(
                "record_in_multiple_test_folds",
                f"{len(dup)} record(s) appear in more than one test fold, e.g. "
                f"{sorted(dup)[:3]}",
                offending=sorted(dup)[:10],
            )
        )

    # ---- 3. derived-data / duplicate-archive audit -------------------
    if recs:
        # (a) identical bytes under different ids
        by_hash: dict[str, list[str]] = defaultdict(list)
        for r in recs:
            if r.lineage.content_hash:
                by_hash[r.lineage.content_hash].append(r.id)
        n_dupe_hash = 0
        for h, ids in by_hash.items():
            if len(ids) < 2:
                continue
            n_dupe_hash += 1
            folds_hit = {f for i in ids for f in fold_of_record.get(i, [])}
            train_hit = any(
                i in set(fold.train_ids) for fold in split.folds for i in ids
            ) and bool(folds_hit)
            if len(folds_hit) > 1 or (folds_hit and train_hit and _crosses(split, ids)):
                rep.violations.append(
                    Violation(
                        "duplicate_content_across_folds",
                        f"content hash {h[:12]}… shared by {ids} which cross fold "
                        f"boundaries {sorted(folds_hit)}",
                        offending=ids,
                    )
                )
        rep.stats["n_duplicate_content_hashes"] = n_dupe_hash

        # (b) derivation roots
        try:
            root = resolve_parentage(recs)
        except LineageError as exc:
            rep.violations.append(
                Violation("unresolved_parentage", str(exc), offending=exc.offending_object)
            )
            root = {}
        if root:
            by_root: dict[str, list[str]] = defaultdict(list)
            for rid, rt in root.items():
                by_root[rt].append(rid)
            for rt, ids in by_root.items():
                if len(ids) < 2:
                    continue
                if _crosses(split, ids):
                    rep.violations.append(
                        Violation(
                            "derived_data_crosses_split",
                            f"records {sorted(ids)[:5]} share derivation root {rt!r} "
                            "but are not in the same split partition",
                            offending=sorted(ids),
                        )
                    )
            rep.stats["n_derivation_roots"] = len(by_root)

        # ---- 4. site predictability ---------------------------------
        fold_label: dict[str, str] = {}
        for fold in split.folds:
            for rid in fold.test_ids:
                fold_label[rid] = f"fold{fold.index}"
        sites = [by_id[r].lineage.site or "unknown" for r in fold_label if r in by_id]
        labels = [fold_label[r] for r in fold_label if r in by_id]
        nmi = _normalised_mutual_information(sites, labels)
        rep.stats["site_fold_nmi"] = round(nmi, 4)
        rep.stats["n_sites"] = len(set(sites))
        if len(set(sites)) > 1 and split.mode != "site" and nmi > SITE_NMI_WARN:
            rep.warnings.append(
                f"site predicts fold membership (NMI={nmi:.2f} > {SITE_NMI_WARN}); "
                "pooled performance may be a site/device shortcut - run leave-site-out"
            )
        if len(set(sites)) == 1 and split.mode != "site":
            rep.warnings.append(
                "all records come from one site: this split cannot falsify a "
                "site/device shortcut (Appendix D 'Site/device shortcuts')"
            )
        if "unknown" in set(sites):
            rep.warnings.append(
                "some records have site='unknown'; leave-site-out evaluation is "
                "not available for them and the site gradient path stays disabled"
            )

    # ---- 5. stimulus overlap ----------------------------------------
    if split.mode == "stimulus" and recs:
        for fold in split.folds:
            held = set(fold.held_out_stimuli)
            bad = [
                rid
                for rid in fold.train_ids
                if rid in by_id and set(by_id[rid].stimulus_ids) & held
            ]
            if bad:
                rep.violations.append(
                    Violation(
                        "stimulus_in_train_and_test",
                        f"fold {fold.index}: {len(bad)} training record(s) contain "
                        f"held-out stimuli, e.g. {bad[:3]}",
                        offending=bad[:10],
                    )
                )
        if split.requires_trial_masking:
            rep.warnings.append(
                "stimulus split requires trial-level masking: some records mix "
                "held-out and retained stimuli and were dropped from both sides"
            )

    rep.stats["n_records_in_test"] = sum(f.n_test for f in split.folds)
    return rep


def _crosses(split: Split, ids: Sequence[str]) -> bool:
    """True if ``ids`` are not all on the same side of every fold."""
    idset = set(ids)
    for fold in split.folds:
        in_train = idset & set(fold.train_ids)
        in_test = idset & set(fold.test_ids)
        if in_train and in_test:
            return True
    return False


def records_from_lineages(
    source_id: str, lineages: Iterable[Mapping[str, object]], *, prefix: str = ""
) -> list[Record]:
    """Convenience: build :class:`Record` objects from lineage mappings."""
    out: list[Record] = []
    for i, m in enumerate(lineages):
        m = dict(m)
        rid = str(m.pop("id", f"{prefix}{i:05d}"))
        stim = tuple(m.pop("stimulus_ids", ()) or ())
        path = m.pop("path", None)
        out.append(
            Record(
                id=rid,
                source_id=source_id,
                lineage=Lineage.from_mapping(m),
                stimulus_ids=stim,
                path=str(path) if path else None,
            )
        )
    return out
