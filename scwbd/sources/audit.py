"""Card-versus-disk audit: a card may not claim data that is not there.

``scwbd/sources/cards/*.yaml`` already carries exemplar hashes, and
``tests/sources/test_manifest.py`` re-derives those from the real files.  That
covers *corruption*.  It does not cover *coverage*: a card can declare
``signal.modalities: [eeg, fmri, dwi]`` while the tree on disk holds one T1w
and no BOLD and no diffusion at all, and every existing check stays green,
because none of them ever asks whether a declared modality has a file behind
it.  A card that reads as coverage we do not have is worse than a missing
card — it is the exact failure catalogued in ``reports/decorative_guards.md``,
where ~26 guards looked green and could not fire.

This module is the check that fires.  Five claims are re-derived from the
filesystem, never from the manifest and never from another report:

``A1 status``
    A card declaring ``live``/``partial`` must have its ``local_path`` present
    and non-empty.  The registry already *computes* status; nothing asserted
    that a card's own declaration survives contact with the disk.

``A2 file_manifest``
    ``n_files`` and ``total_bytes`` are re-counted by walking the tree.  The
    stored manifest is not consulted: it was generated from disk and therefore
    cannot disagree with it, which is why comparing card-to-manifest proves
    less than it appears to.

``A3 participants``
    Every id in ``population.participant_ids`` (when it is a list rather than
    a range description) must appear on disk as a directory or as a filename
    prefix.  ``population.n_participants`` must equal the number found.

``A4 modality evidence``
    **The one that catches phantom coverage.**  Every entry of
    ``signal.modalities`` must have a matching entry in
    ``signal.modality_evidence`` — a list of globs, relative to the dataset
    root, that shows where that modality physically lives — and at least one
    file must match.  A modality with no evidence block is a failure, not a
    pass: silence is how a phantom modality survives.  An optional
    ``channel_type`` narrows the claim further by reading one file's header.

``A5 licence``
    The free-text ``governance.license`` is classified by
    ``scwbd.release.licence`` and must agree with the card's own declared
    ``governance.license_spdx``.  Two independent statements of the same fact:
    if the regex classifier ever mis-reads a licence, the disagreement fires
    here instead of silently routing an NC source into a permissive tier.

**What this module does not do.**  A glob is evidence that a *file* exists, not
proof of what is inside it.  ``channel_type`` closes that for header-bearing
formats (FIFF, EDF, BrainVision) and nothing closes it for NIfTI, where the
declared modality is carried by the BIDS suffix and not by the bytes.  Stated
because an audit that overclaims its own reach is the thing it exists to stop.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cards import SourceCardDoc, load_all_cards, load_card
from .manifest import DEFAULT_EXCLUDES, iter_files

__all__ = [
    "CardAudit",
    "Finding",
    "audit_card",
    "audit_all",
    "AVAILABLE_STATUS",
]

#: Statuses that assert bytes are present.  ``unavailable`` asserts the
#: opposite and is audited for the opposite property.
AVAILABLE_STATUS: frozenset[str] = frozenset({"live", "partial"})


@dataclass(frozen=True)
class Finding:
    """One disagreement between a card and the filesystem."""

    card_id: str
    check: str
    claim: str
    on_disk: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        tail = f" — {self.detail}" if self.detail else ""
        return (
            f"[{self.card_id}] {self.check}: card claims {self.claim}; "
            f"disk has {self.on_disk}{tail}"
        )


@dataclass
class CardAudit:
    card_id: str
    root: Path | None
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        head = "OK" if self.ok else "FAIL"
        return (
            f"audit {self.card_id}: {head} "
            f"({len(self.checks_run)} checks, {len(self.findings)} findings, "
            f"{len(self.skipped)} not applicable)"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel_files(root: Path) -> list[str]:
    """Every tracked file under ``root``, as a POSIX path relative to it.

    Uses ``manifest.iter_files`` so the audit and the manifest agree on what
    counts as a file (same exclude list, same symlink handling).  A divergence
    here would make the two disagree for a reason that has nothing to do with
    the data.
    """
    return sorted(str(p.relative_to(root)) for p in iter_files(root, DEFAULT_EXCLUDES))


def _match_any(rels: Sequence[str], globs: Iterable[str]) -> list[str]:
    pats = list(globs)
    hits: list[str] = []
    for rel in rels:
        for pat in pats:
            if fnmatch.fnmatch(rel, pat):
                hits.append(rel)
                break
    return hits


def _header_info(path: Path):
    """MNE ``Info`` read from a header only, or ``None`` if not readable.

    Header-only: no sample data is loaded, so this stays cheap on a 300 MB
    BrainVision file.  ``None`` (rather than an empty result) when the format
    carries no channel concept, so "cannot tell" never reads as "nothing
    there".
    """
    name = path.name.lower()
    try:
        import mne

        if name.endswith(".fif"):
            return mne.io.read_info(path, verbose="ERROR")
        if name.endswith(".edf"):
            return mne.io.read_raw_edf(path, preload=False, verbose="ERROR").info
        if name.endswith(".vhdr"):
            return mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR").info
    except Exception:
        return None
    return None


def _channel_types(path: Path) -> set[str] | None:
    """The set of channel types this file *states* it holds.

    Two authorities, and they are not the same authority:

    * a ``*_channels.tsv`` states the types in its ``type`` column — this is
      BIDS metadata written by the acquirer;
    * a FIFF/EDF/BrainVision header states what the acquisition software
      recorded.

    They disagree in practice.  ds004024's BrainVision header reports 69
    undifferentiated ``eeg`` channels while its ``channels.tsv`` names 64 EEG,
    2 EOG, 2 EMG and 1 ECG; ds000117's FIFF reports 74 ``eeg`` while its
    ``channels.tsv`` separates ECG, HEOG and VEOG.  A card claiming EOG must
    therefore point at whichever file actually carries the distinction, and
    this function reads whichever it was pointed at.  Neither is "the truth" —
    but a card that points at neither is claiming nothing at all.
    """
    if path.name.lower().endswith(".tsv"):
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            return None
        if not lines:
            return None
        header = [c.strip() for c in lines[0].split("\t")]
        if "type" not in header:
            return None
        col = header.index("type")
        out: set[str] = set()
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > col:
                out.add(parts[col].strip().upper())
        return out or None
    info = _header_info(path)
    if info is None:
        return None
    import mne

    return {str(t).upper() for t, idx in mne.channel_indices_by_type(info).items() if idx}


def _channel_names(path: Path) -> list[str] | None:
    if path.name.lower().endswith(".tsv"):
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            return None
        if not lines:
            return None
        header = [c.strip() for c in lines[0].split("\t")]
        if "name" not in header:
            return None
        col = header.index("name")
        return [ln.split("\t")[col].strip() for ln in lines[1:] if len(ln.split("\t")) > col]
    info = _header_info(path)
    return None if info is None else [str(c) for c in info["ch_names"]]


def _declared_participants(card: SourceCardDoc) -> list[str] | None:
    """Participant ids **only** when the card lists them explicitly.

    Cards that describe a range in prose (``"S001..S109 (see manifest)"``) are
    not audited here: inventing a parser for prose would turn a stated
    approximation into a false precision.  Those cards are reported as skipped
    so the gap is visible rather than absent.
    """
    ids = card.data.get("population", {}).get("participant_ids")
    if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
        return list(ids)
    return None


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def audit_card(card: SourceCardDoc | str | Path, root: Path | str | None = None) -> CardAudit:
    """Re-derive a card's claims from the filesystem.

    ``root`` defaults to the registry's path for this dataset id, so the audit
    checks the tree the loaders would actually read, not the one the card
    happens to name (a card's ``identity.local_path`` is an absolute path from
    whichever worktree last refreshed it and is not portable).
    """
    if not isinstance(card, SourceCardDoc):
        card = load_card(card)
    cid = card.id

    if root is None:
        from .registry import REGISTRY

        entry = REGISTRY.get(cid)
        root = entry.local_path if entry is not None else None
    root = Path(root) if root is not None else None
    audit = CardAudit(card_id=cid, root=root)
    gov = card.data.get("governance", {}) or {}
    status = str(gov.get("status", "unknown"))

    # ---- A5 licence: free text and SPDX must agree ------------------------
    audit.checks_run.append("A5_licence")
    _audit_licence(card, audit)

    # ---- unavailable cards assert the opposite ---------------------------
    if status not in AVAILABLE_STATUS:
        audit.checks_run.append("A0_unavailable_claims_nothing")
        if root is not None and root.exists() and any(root.iterdir()):
            audit.findings.append(
                Finding(cid, "A0_unavailable_claims_nothing", f"status={status!r} (no bytes)",
                        f"{root} exists and is non-empty")
            )
        audit.skipped += ["A1_status", "A2_file_manifest", "A3_participants",
                          "A4_modality_evidence"]
        return audit

    # ---- A1 the declared root is really there ----------------------------
    audit.checks_run.append("A1_status")
    if root is None:
        audit.findings.append(
            Finding(cid, "A1_status", f"status={status!r}", "no registry entry names a local path")
        )
        audit.skipped += ["A2_file_manifest", "A3_participants", "A4_modality_evidence"]
        return audit
    if not root.exists():
        audit.findings.append(
            Finding(cid, "A1_status", f"status={status!r} at {root}", "path does not exist")
        )
        audit.skipped += ["A2_file_manifest", "A3_participants", "A4_modality_evidence"]
        return audit

    rels = _rel_files(root)
    if not rels:
        audit.findings.append(
            Finding(cid, "A1_status", f"status={status!r} at {root}", "0 files")
        )
        audit.skipped += ["A2_file_manifest", "A3_participants", "A4_modality_evidence"]
        return audit

    # ---- A2 file counts re-derived from the tree -------------------------
    audit.checks_run.append("A2_file_manifest")
    fm = card.data.get("identity", {}).get("file_manifest")
    if isinstance(fm, dict):
        n_disk = len(rels)
        b_disk = sum((root / r).stat().st_size for r in rels)
        if int(fm.get("n_files", -1)) != n_disk:
            audit.findings.append(
                Finding(cid, "A2_file_manifest", f"n_files={fm.get('n_files')}", f"{n_disk} files")
            )
        if int(fm.get("total_bytes", -1)) != b_disk:
            audit.findings.append(
                Finding(cid, "A2_file_manifest", f"total_bytes={fm.get('total_bytes')}",
                        f"{b_disk} bytes")
            )
        for rel in (fm.get("exemplars") or {}):
            if not (root / rel).exists():
                audit.findings.append(
                    Finding(cid, "A2_file_manifest", f"exemplar {rel}", "absent")
                )
    else:
        audit.findings.append(
            Finding(cid, "A2_file_manifest", "an available card", "no file_manifest block")
        )

    # ---- A3 participants -------------------------------------------------
    audit.checks_run.append("A3_participants")
    declared = _declared_participants(card)
    pop = card.data.get("population", {}) or {}
    if declared is None:
        audit.skipped.append("A3_participants(prose participant_ids)")
    else:
        present = [p for p in declared
                   if (root / p).is_dir() or any(r.startswith(p) or f"/{p}" in r for r in rels)]
        missing = sorted(set(declared) - set(present))
        if missing:
            audit.findings.append(
                Finding(cid, "A3_participants", f"{len(declared)} participant ids",
                        f"{len(present)} found", f"missing: {', '.join(missing[:8])}")
            )
        n_claim = pop.get("n_participants")
        if isinstance(n_claim, int) and n_claim != len(present):
            audit.findings.append(
                Finding(cid, "A3_participants", f"n_participants={n_claim}",
                        f"{len(present)} on disk")
            )

    # ---- A4 modality evidence -------------------------------------------
    audit.checks_run.append("A4_modality_evidence")
    sig = card.data.get("signal", {}) or {}
    modalities = list(sig.get("modalities") or [])
    evidence = sig.get("modality_evidence")
    if not isinstance(evidence, dict):
        audit.findings.append(
            Finding(cid, "A4_modality_evidence", f"{len(modalities)} modalities",
                    "no signal.modality_evidence block",
                    "a declared modality with no evidence entry is unfalsifiable; "
                    "add one glob per modality showing where it lives on disk")
        )
    else:
        for mod in modalities:
            spec = evidence.get(mod)
            if spec is None:
                audit.findings.append(
                    Finding(cid, "A4_modality_evidence", f"modality {mod!r}",
                            "no evidence entry", "modality declared without a file behind it")
                )
                continue
            if isinstance(spec, (str, list, tuple)):
                globs = [spec] if isinstance(spec, str) else list(spec)
                ctype: Any = None
                cname: Any = None
            else:
                globs = list(spec.get("globs") or [])
                ctype = spec.get("channel_type")
                cname = spec.get("channel_name")
            hits = _match_any(rels, globs)
            if not hits:
                audit.findings.append(
                    Finding(cid, "A4_modality_evidence", f"modality {mod!r} at {globs}",
                            "0 matching files")
                )
                continue
            if ctype:
                want = {str(x).upper() for x in ([ctype] if isinstance(ctype, str) else ctype)}
                types = _channel_types(root / hits[0])
                if types is None:
                    audit.findings.append(
                        Finding(cid, "A4_modality_evidence",
                                f"modality {mod!r} with channel_type {sorted(want)}",
                                f"{hits[0]} states no channel types",
                                "the evidence file cannot support the claim made of it")
                    )
                elif not (want & types):
                    audit.findings.append(
                        Finding(cid, "A4_modality_evidence",
                                f"modality {mod!r} with channel_type {sorted(want)}",
                                f"{hits[0]} states {sorted(types)}")
                    )
            if cname:
                want_n = [cname] if isinstance(cname, str) else list(cname)
                names = _channel_names(root / hits[0])
                if names is None:
                    audit.findings.append(
                        Finding(cid, "A4_modality_evidence",
                                f"modality {mod!r} with channel_name {want_n}",
                                f"{hits[0]} states no channel names")
                    )
                else:
                    absent = [w for w in want_n
                              if not any(w.lower() in n.lower() for n in names)]
                    if absent:
                        audit.findings.append(
                            Finding(cid, "A4_modality_evidence",
                                    f"modality {mod!r} with channel_name {absent}",
                                    f"{hits[0]} has no such channel")
                        )
        extra = sorted(set(evidence) - set(modalities))
        if extra:
            audit.findings.append(
                Finding(cid, "A4_modality_evidence", f"modalities {sorted(modalities)}",
                        f"evidence for undeclared {extra}",
                        "evidence without a declaration is as misleading as the reverse")
            )
    return audit


def _audit_licence(card: SourceCardDoc, audit: CardAudit) -> None:
    """``governance.license`` (free text) must agree with ``license_spdx``."""
    from ..release.licence import term_from_dataset_card

    cid = card.id
    gov = card.data.get("governance", {}) or {}
    spdx = gov.get("license_spdx")
    if not spdx:
        audit.findings.append(
            Finding(cid, "A5_licence", "a licence the policy can route",
                    "no governance.license_spdx",
                    "free text alone routes through a regex; an SPDX id (or the "
                    "literal 'UNKNOWN' / 'DUA-<name>') is the second, independent "
                    "statement that lets a mis-parse be detected")
        )
        return
    # Read through the *release* module's own entry point, on the card file
    # itself, so this check exercises the path the checkpoint policy uses. A
    # check that re-implements the classifier would agree with itself and
    # prove nothing.
    term = term_from_dataset_card(card.path)
    spdx_u = str(spdx).upper()
    if spdx_u in ("UNKNOWN", "NONE") or spdx_u.startswith("DUA-"):
        expect_nc: bool | None = None
    else:
        expect_nc = "-NC" in spdx_u
    expect_sa = None if expect_nc is None else ("-SA" in spdx_u)
    if expect_nc is not None and term.noncommercial is not expect_nc:
        audit.findings.append(
            Finding(cid, "A5_licence", f"license_spdx={spdx!r} (noncommercial={expect_nc})",
                    f"free-text classifier says noncommercial={term.noncommercial}")
        )
    if expect_sa is not None and term.share_alike is not expect_sa:
        audit.findings.append(
            Finding(cid, "A5_licence", f"license_spdx={spdx!r} (share_alike={expect_sa})",
                    f"free-text classifier says share_alike={term.share_alike}")
        )


def audit_all(card_dir: str | Path | None = None) -> dict[str, CardAudit]:
    cards = load_all_cards() if card_dir is None else load_all_cards(card_dir)
    return {cid: audit_card(doc) for cid, doc in sorted(cards.items())}


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.audit")
    ap.add_argument("card_id", nargs="*")
    args = ap.parse_args(argv)
    audits = audit_all()
    if args.card_id:
        audits = {k: v for k, v in audits.items() if k in set(args.card_id)}
    rc = 0
    for cid, a in audits.items():
        print(a.summary())
        for f in a.findings:
            print(f"    {f}")
        rc = rc or (0 if a.ok else 1)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
