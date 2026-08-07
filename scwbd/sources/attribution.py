"""The attribution block a checkpoint must be able to print about itself.

Why this is a compliance surface and not documentation
-----------------------------------------------------
``ARCHITECTURE.md`` §7a: inherited data attribution and permissions are the
project's licence concern.  For several inputs **citation is the licence
condition itself** — ``scwbd.anatomy.sources.SRC["tian2020"]`` is usable
without restriction *subject to citation*, and the PhysioNet/ODC-By sources
carry attribution as their only obligation.  An artifact that cannot state what
it was built from is therefore not merely undocumented; it is out of compliance
with the terms it inherited.

Two rules, both of which exist because the opposite failure is easy and quiet.

**Derive, never restate.**  Every line of the block is read from the registry
that already holds the fact: ``scwbd/sources/cards/*.yaml`` for data sources
(``identity.citation``, ``governance.license``, ``governance.license_spdx``)
and ``scwbd.anatomy.sources.SRC`` for anatomy inputs.  Nothing here contains a
citation string of its own.  A citation typed into this module would drift from
the card the moment the card changed, and the drift would be invisible.

**Absence is loud.**  A contributing source that cannot produce a citation
raises :class:`AttributionError`.  It does not emit "unknown" and it does not
omit the row.  Under a licence whose condition is citation, a missing citation
is a violation, and a block that silently skips it is a decorative control:
it renders, it looks complete, and it cannot fail.  The negative controls in
``tests/sources/test_attribution.py`` are what make that claim checkable.

Scope
-----
This module derives attribution for **sources**.  It deliberately does not
decide the licence — that is ``scwbd.release.licence.union_of``, and having two
modules answer the same question is how they come to disagree.  The block
reports the licence each source *states* so a reader can audit the union; the
union remains the authority.
"""

from __future__ import annotations

import os

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AttributionError",
    "Attribution",
    "AttributionBlock",
    "attribution_for_datasets",
    "attribution_for_anatomy",
    "attribution_for_checkpoint",
]


class AttributionError(RuntimeError):
    """A contributing source cannot state what it must be cited as."""


@dataclass(frozen=True)
class Attribution:
    """One input's attribution, every field read from a registry."""

    key: str
    kind: str                 # "dataset" | "anatomy"
    citation: str
    licence: str
    licence_spdx: str | None
    url: str | None
    doi: str | None
    #: The file this was read from. A citation with no provenance is a claim.
    provenance: str

    def render(self) -> str:
        bits = [f"  {self.key} ({self.kind})", f"    cite:    {self.citation}"]
        spdx = f" [{self.licence_spdx}]" if self.licence_spdx else ""
        bits.append(f"    licence:{spdx} {self.licence}")
        if self.doi:
            bits.append(f"    doi:     {self.doi}")
        elif self.url:
            bits.append(f"    url:     {self.url}")
        bits.append(f"    from:    {_repo_relative(self.provenance)}")
        return "\n".join(bits)



def _repo_relative(provenance: str) -> str:
    """A provenance path as it appears in the repository, never on this machine.

    Registries differ in whether they record card paths absolute or relative, so
    the same attribution block printed both ``configs/source_cards/...`` and
    ``/home/<user>/Documents/.../scwbd/sources/cards/...`` -- and that block is
    published verbatim on the model card. An absolute path there is a reader's
    dead end (it names a directory only the author has) and it discloses the
    author's home directory to everyone who downloads the model.

    Normalising at render time rather than at each registry means a registry
    added later cannot reintroduce it.
    """
    if not provenance or not os.path.isabs(provenance):
        return provenance
    root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(provenance).resolve().relative_to(root))
    except ValueError:
        # Outside the repository entirely: report the basename rather than a
        # path into somebody's filesystem, and say that is what happened.
        return f"<outside repo>/{Path(provenance).name}"


@dataclass
class AttributionBlock:
    entries: tuple[Attribution, ...] = ()
    #: Sources that were asked for and had no registry entry at all. Recorded
    #: rather than dropped: a source nobody can attribute is the finding.
    unattributable: tuple[tuple[str, str], ...] = ()
    header: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unattributable

    def citations(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e.citation, None)
        return tuple(seen)

    def render(self) -> str:
        lines = ["ATTRIBUTION"]
        if self.header:
            lines.append(self.header)
        lines.append("=" * 60)
        for kind in ("dataset", "anatomy"):
            rows = [e for e in self.entries if e.kind == kind]
            if not rows:
                continue
            lines.append(f"\n{kind.upper()} INPUTS ({len(rows)})")
            lines += [e.render() for e in rows]
        if self.unattributable:
            lines.append("\nUNATTRIBUTABLE — THIS ARTIFACT IS NOT COMPLIANT")
            lines += [f"  {k}: {why}" for k, why in self.unattributable]
        for n in self.notes:
            lines.append(f"\nNOTE {n}")
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "scwbd-attribution/1.0.0",
            "ok": self.ok,
            "entries": [asdict(e) for e in self.entries],
            "unattributable": [list(x) for x in self.unattributable],
            "citations": list(self.citations()),
            "notes": list(self.notes),
        }

    def require_complete(self) -> "AttributionBlock":
        """Raise unless every contributing source is attributable.

        Call this on the release path.  Citation is a licence condition for at
        least one input, so shipping an artifact whose attribution has holes is
        a licence failure, not a documentation gap.
        """
        if self.unattributable:
            detail = "; ".join(f"{k}: {why}" for k, why in self.unattributable)
            raise AttributionError(
                f"{len(self.unattributable)} contributing source(s) cannot be "
                f"attributed: {detail}. Citation is a condition of use for at "
                f"least one input (scwbd.anatomy.sources.SRC['tian2020']), so "
                f"this artifact may not be released."
            )
        return self


# ---------------------------------------------------------------------------
# dataset side: read scwbd/sources/cards/*.yaml
# ---------------------------------------------------------------------------
def _dataset_attribution(dataset_id: str, card_path: Path) -> Attribution:
    import yaml

    data = yaml.safe_load(card_path.read_text()) or {}
    ident = data.get("identity", {}) or {}
    gov = data.get("governance", {}) or {}
    citation = str(ident.get("citation") or "").strip()
    if not citation or citation.lower().startswith("unknown"):
        raise AttributionError(
            f"{dataset_id}: {card_path} states no identity.citation. A dataset "
            f"card without a citation cannot satisfy an attribution obligation."
        )
    return Attribution(
        key=dataset_id,
        kind="dataset",
        citation=" ".join(citation.split()),
        licence=" ".join(str(gov.get("license") or "unknown").split())[:400],
        licence_spdx=(str(gov["license_spdx"]) if gov.get("license_spdx") else None),
        url=gov.get("license_url") or ident.get("source_url"),
        doi=ident.get("doi"),
        provenance=str(card_path),
    )


def attribution_for_datasets(
    dataset_ids: Iterable[str], *, card_dir: str | Path | None = None
) -> AttributionBlock:
    """Attribution for dataset ids, read from their source cards."""
    from .cards import CARD_DIR

    d = Path(card_dir) if card_dir else CARD_DIR
    entries: list[Attribution] = []
    missing: list[tuple[str, str]] = []
    for did in dataset_ids:
        p = d / f"{did}.yaml"
        if not p.is_file():
            missing.append((did, f"no source card at {p}"))
            continue
        try:
            entries.append(_dataset_attribution(did, p))
        except AttributionError as exc:
            missing.append((did, str(exc)))
    return AttributionBlock(entries=tuple(entries), unattributable=tuple(missing))


# ---------------------------------------------------------------------------
# anatomy side: read scwbd.anatomy.sources.SRC
# ---------------------------------------------------------------------------
def attribution_for_anatomy(keys: Iterable[str]) -> AttributionBlock:
    """Attribution for anatomy inputs, read from the anatomy source registry.

    ``keys`` must come from the prior's **own provenance** — the pattern
    ``reports/anatomy_prior.md`` establishes after a hardcoded atlas key
    flagged every subcortical field non-commercial. Passing a hand-written list
    here reintroduces exactly that defect one module over, so the caller is the
    one that must derive it.
    """
    try:
        from ..anatomy.sources import SRC
    except Exception as exc:  # pragma: no cover - anatomy is an optional import
        return AttributionBlock(
            unattributable=tuple((k, f"anatomy source registry unavailable: {exc}")
                                 for k in keys)
        )
    entries: list[Attribution] = []
    missing: list[tuple[str, str]] = []
    src_path = "scwbd/anatomy/sources.py::SRC"
    for k in keys:
        rec = SRC.get(k)
        if rec is None:
            missing.append((k, f"{k!r} is not in {src_path}"))
            continue
        citation = str(rec.get("citation") or "").strip()
        if not citation:
            missing.append((k, f"{src_path}[{k!r}] states no citation"))
            continue
        entries.append(
            Attribution(
                key=k,
                kind="anatomy",
                citation=" ".join(citation.split()),
                licence=" ".join(str(rec.get("license") or "unknown").split())[:400],
                licence_spdx=None,   # the anatomy registry states prose, not SPDX
                url=rec.get("url"),
                doi=None,
                provenance=src_path,
            )
        )
    return AttributionBlock(entries=tuple(entries), unattributable=tuple(missing))


# ---------------------------------------------------------------------------
# the checkpoint-level block
# ---------------------------------------------------------------------------
def attribution_for_checkpoint(
    *,
    dataset_ids: Sequence[str],
    anatomy_keys: Sequence[str] = (),
    tag: str = "",
    notes: Sequence[str] = (),
) -> AttributionBlock:
    """The block a released checkpoint prints about itself.

    ``dataset_ids`` should be derived from the run's own manifest — the
    dataset cards its *contributing* mixture sources link to
    (``scwbd.release.manifest.SourceFamilyManifest.dataset_links``), not a list
    anybody typed. ``anatomy_keys`` likewise from the prior's provenance.
    """
    ds = attribution_for_datasets(dataset_ids)
    an = attribution_for_anatomy(anatomy_keys)
    block = AttributionBlock(
        entries=ds.entries + an.entries,
        unattributable=ds.unattributable + an.unattributable,
        header=f"checkpoint: {tag}" if tag else "",
        notes=list(notes),
    )
    if not block.entries:
        block.unattributable = block.unattributable + (
            ("<no inputs>", "no contributing source was supplied; an artifact "
                            "with no attributable input has no provenance"),
        )
    return block


def attribution_from_manifest(manifest: Any, *, tag: str = "") -> AttributionBlock:
    """Derive the block from a run's own ``SourceFamilyManifest``.

    This is the entry point the release path should use, because it takes the
    dataset ids from ``manifest.dataset_links`` — the same objects
    ``licence_terms()`` unions — so the attribution block and the licence union
    can never be computed over different sets of sources.  Passing a list by
    hand is what lets them drift.

    A **contributing** source whose ``dataset_links`` entry is ``None`` is
    reported as unattributable *unless* it is a known non-dataset source
    (the simulator, the prior, the calibration and the negative control are not
    datasets and never will be).  That distinction is read from
    ``scwbd.release.manifest.NON_DATASET_TERMS`` rather than hardcoded here.
    """
    try:
        from ..release.manifest import NON_DATASET_TERMS
        non_dataset = set(NON_DATASET_TERMS) | {"anatomical_prior"}
    except Exception:  # pragma: no cover - release module optional
        non_dataset = {"anatomical_prior"}

    links: Mapping[str, Any] = getattr(manifest, "dataset_links", {}) or {}
    contributing = [r.id for r in getattr(manifest, "contributing", ())]
    dataset_ids: list[str] = []
    orphans: list[tuple[str, str]] = []
    for sid in contributing:
        info = links.get(sid)
        if info is not None:
            dataset_ids.append(info.dataset_id)
        elif sid not in non_dataset:
            orphans.append((
                sid,
                "contributes to the checkpoint but links to no dataset card, and "
                "is not a registered non-dataset source; nothing states what it "
                "must be cited as",
            ))
    block = attribution_for_checkpoint(dataset_ids=dataset_ids, tag=tag)
    block.unattributable = tuple(orphans) + tuple(
        x for x in block.unattributable if x[0] != "<no inputs>"
    ) + (() if (block.entries or orphans) else (
        ("<no inputs>", "no contributing source was supplied"),))
    return block


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.attribution")
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--anatomy", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if incomplete")
    args = ap.parse_args(argv)
    block = attribution_for_checkpoint(
        dataset_ids=args.datasets, anatomy_keys=args.anatomy, tag=args.tag
    )
    print(json.dumps(block.as_dict(), indent=1) if args.json else block.render())
    return 1 if (args.strict and not block.ok) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
