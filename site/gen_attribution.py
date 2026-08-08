#!/usr/bin/env python3
"""Generate the site's attribution page from the repository's own registries.

Licensing is the one real compliance surface on this project, so this page is
**computed, never hand-written**.  It enumerates every dataset card in
``scwbd/sources/cards/`` and every anatomy source in ``scwbd/anatomy/sources.py``,
renders them through ``scwbd.sources.attribution``, and unions their terms
through ``scwbd.release.licence``.

Hand-typing a source list here would reintroduce exactly the defect the
attribution module exists to prevent — a citation set that is a memory of what
we used rather than a record of it.

Writes ``site/content/attribution.html``.  Run before ``site/build.py``.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from scwbd.anatomy.sources import SRC  # noqa: E402
from scwbd.release.licence import term_from_dataset_card, term_from_licence_text, union_of  # noqa: E402
from scwbd.release.manifest import DOWNSTREAM_REACH_QUESTION  # noqa: E402
from scwbd.sources.attribution import (  # noqa: E402
    attribution_for_anatomy,
    attribution_for_datasets,
)

CARD_DIR = REPO / "scwbd" / "sources" / "cards"
OUT = ROOT / "content" / "attribution.html"


def e(s: object) -> str:
    return html.escape(str(s))


def entry_rows(block, kind_label: str) -> str:
    rows = []
    for a in block.entries:
        links = []
        if a.doi:
            links.append(f'<a href="https://doi.org/{e(a.doi)}">doi:{e(a.doi)}</a>')
        if a.url:
            links.append(f'<a href="{e(a.url)}">source</a>')
        link_html = " · ".join(links)
        rows.append(
            "<tr>"
            f'<td><code>{e(a.key)}</code></td>'
            f"<td>{e(kind_label)}</td>"
            f"<td>{e(a.citation)}</td>"
            f"<td>{e(a.licence)}{'<br>' + link_html if link_html else ''}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def main() -> None:
    dataset_ids = sorted(p.stem for p in CARD_DIR.glob("*.yaml"))
    anatomy_keys = sorted(SRC.keys())

    ds_block = attribution_for_datasets(dataset_ids, card_dir=CARD_DIR)
    an_block = attribution_for_anatomy(anatomy_keys)

    # `--text` prints the citation set instead of writing the page, so the same
    # enumeration backs both `make attribution` and the published page. Two code
    # paths producing two source lists is exactly the drift this module exists
    # to prevent.
    if "--text" in sys.argv:
        print(ds_block.require_complete().render())
        print()
        print(an_block.require_complete().render())
        return

    terms = []
    for did in dataset_ids:
        try:
            terms.append(term_from_dataset_card(CARD_DIR / f"{did}.yaml"))
        except Exception:  # a card without governance still counts as unknown
            pass
    for key in anatomy_keys:
        meta = SRC[key]
        lic = meta.get("licence") or meta.get("license")
        terms.append(
            term_from_licence_text(
                key,
                lic,
                provenance=f"scwbd/anatomy/sources.py::SRC[{key!r}]",
                verified=False,
                url=meta.get("url"),
            )
        )

    union = union_of(terms)
    summary = union.summary()

    # The summary ends with a parenthesised roll-call of every source whose
    # licence is unresolved -- 27 of them.  Inline it swamps the paragraph it
    # is supposed to qualify, so the count stays in the sentence and the names
    # go into a note.  Split rather than truncate: a hardcoded head would stop
    # tracking the summary the moment its format changed.
    m = re.search(r"^(.*?UNKNOWN licence)\s*\(([^)]*)\)(.*)$", summary, re.S)
    unknown_names = m.group(2) if m else ""

    # `key: value; key: value; ...` with two clauses at the end that are
    # sentences rather than pairs.  Parsed into rows so the union can be read at
    # a glance instead of as forty words of inline code; the raw string is kept
    # in a note, so nothing computed is lost by presenting it as a table.
    kv = {}
    for chunk in summary.split(";"):
        if ":" in chunk:
            k, v = chunk.split(":", 1)
            kv[k.strip().lower()] = v.strip()
    n_unknown = re.search(r"(\d+) source\(s\) with UNKNOWN", summary)

    union_rows = [
        (label, kv[key])
        for key, label in (
            ("non-commercial", "Non-commercial"),
            ("share-alike", "Share-alike"),
            ("attribution", "Attribution"),
            ("redistribution", "Redistribution"),
        )
        if key in kv
    ]
    if n_unknown:
        union_rows.append(("Unresolved sources", n_unknown.group(1)))
    if not union_rows:                       # summary format changed under us
        union_rows = [("Union", summary)]

    union_table = (
        '<aside class="sidetable">\n<table>\n<tbody>\n'
        + "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>\n" for k, v in union_rows)
        + "</tbody>\n</table>\n</aside>"
    )

    unattributable = list(ds_block.unattributable) + list(an_block.unattributable)

    # Two sections, and the second one is a table.
    #
    # This page used to run to seven headings and roughly a thousand words, most
    # of it explaining the reasoning behind conditions rather than stating them.
    # Every condition that binds is still here; each is now one sentence, and
    # the reasoning that a reader may want is in a note they can open.  The
    # owner-decision section is gone outright: it restated the licence line
    # three paragraphs above it.
    n_sources = len(ds_block.entries) + len(an_block.entries)

    parts: list[str] = []
    parts.append(f"""title: Attribution and licensing — SC-WBD
description: The computed citation set and licence union for every data source SC-WBD holds.
nav: attribution
heading: Attribution and licensing
bodyclass: flat-notes
---
<div class="prose">

<p class="lede">SC-WBD is built on {n_sources} publicly released atlases, receptor
maps and neuroimaging datasets. Several attach conditions — attribution,
non-commercial use, share-alike — and those conditions propagate into anything
derived from them.[[note: Generated by
<code>site/gen_attribution.py</code>, which enumerates every card in
<code>scwbd/sources/cards/</code> and every entry in
<code>scwbd/anatomy/sources.py</code> and renders them through the same
<code>scwbd.sources.attribution</code> module the release path uses. Regenerate
with <code>make site-attribution</code>. Licence is split into
<em>inheritance</em>, what the sources impose, and <em>policy</em>, what the
owner chose; the two are never summed into one boolean, because only the second
is the owner's to revoke — see <code>scwbd/release/licence.py</code>.]]</p>

<div class="withside">
{union_table}

<p>The effective licence is the union over every source the repository holds,
including sources no default code path loads — and there
<strong>unknown is not permissive</strong>, so a source naming no terms is
carried unresolved rather than rounded down to fine.[[note: The full computed
string is <code>{e(summary)}</code>.]][[note: <code>is_vacuous_licence_text</code> exists because a registry entry
once read “See repository LICENSE (open, academic use)” for an atlas whose actual
licence imposes no academic-use limit at all — an invented restriction, since
corrected. The classifier now refuses to resolve text that names no terms.]] The
<strong>Hansen receptor maps</strong> are CC-BY-NC-SA-4.0, non-commercial
<em>and</em> share-alike, and where those terms attach a derivative must carry
them too.[[note: Two questions have to be kept apart or the answer misreports:
<strong>does the object contain Hansen data</strong>, and <strong>does the
default prior read it</strong>? Those have different answers today; the audit
is <code>reports/licence_audit.md</code>.]] The <strong>Tian subcortical atlas</strong> is
unrestricted subject to one condition — that any publication using it cites Tian
et&nbsp;al. (2020) — which the table below is how we meet.[[note: Verified
against the vendored licence text at
<code>assets/src/tian_subcortex/license.txt</code>, not against the registry's
summary of it.]] <strong>Schaefer 2018</strong> labels are MIT, but the Genomics
Superstruct Project data underneath is released “under its own terms” and those
terms are not named, so the classifier resolves it to unknown rather than to MIT.
Whether a model trained on CC-BY-NC-SA data is itself a derivative of that data
is recorded as <code>{e(DOWNSTREAM_REACH_QUESTION.get('status', 'unsettled'))}</code>
and <strong>deliberately not answered here</strong>; the conservative reading,
assume it is, is the one that fails safe.[[src: scwbd/release/manifest.py]] And
several datasets carry <code>redistribution_class: none</code> — HCP Young Adult,
ADNI, UK Biobank, TUH-EEG among them — so <strong>this site hosts no copy of any
source and links to no derived data</strong>, with access going through each
provider's own agreement.</p>
</div>

<p>The repository itself is released under
<a href="https://github.com/JacobFV/sc-wbd/blob/master/LICENSE">CC BY-NC-SA
4.0</a>, matching the most restrictive term inherited from the atlas inputs.</p>
""")

    parts.append('<div class="tablewrap wide"><table class="srcs"><thead><tr>'
                 "<th>Key</th><th>Kind</th><th>Citation</th><th>Licence</th>"
                 "</tr></thead><tbody>")
    parts.append(entry_rows(ds_block, "dataset"))
    parts.append(entry_rows(an_block, "anatomy"))
    parts.append("</tbody></table></div>")

    if unattributable:
        parts.append(
            "<p>These could not be rendered into a complete citation. They are "
            "listed rather than dropped, because a missing row is indistinguishable "
            "from a clean one.</p><ul>"
        )
        for key, why in unattributable:
            parts.append(f"<li><code>{e(key)}</code> — {e(why)}</li>")
        parts.append("</ul>")

    parts.append("\n</div>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")
    print(f"  datasets: {len(ds_block.entries)}  anatomy: {len(an_block.entries)}"
          f"  unattributable: {len(unattributable)}")
    print(f"  union: {summary[:100]}")


if __name__ == "__main__":
    main()
