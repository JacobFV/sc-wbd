"""Every number on the public site must still be true of the repository.

The site is the one artifact a stranger reads without being able to check it,
so a figure that drifts there is worse than one that drifts in a report: the
report has a reader who can re-run it.  ``reports/decorative_guards.md``
catalogues eleven relayed figures this project had to withdraw; one dropped
qualifier inverted a claim's meaning.

These tests are deliberately narrow.  They do **not** re-derive the science —
they check that a claim on a page is still traceable to the file it came from,
which is the property that actually rots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "content"
DOCS = ROOT / "docs"


def _all_site_text() -> str:
    return "\n".join(p.read_text() for p in SITE.rglob("*.html"))


def test_the_site_and_the_deployed_copy_agree():
    """``docs/`` is what the world sees; ``site/_build`` is what we built.

    A rebuild that was never copied is a site that silently serves the previous
    claim.  Compare page *count* rather than bytes: the generator stamps a
    build time, so byte equality would be a test that can never pass.
    """
    built = ROOT / "site" / "_build"
    if not built.exists() or not DOCS.exists():
        pytest.skip("no build on disk")
    n_built = len(list(built.rglob("*.html")))
    n_docs = len(list(DOCS.rglob("*.html")))
    assert n_built == n_docs, f"built {n_built} pages, deployed {n_docs}"


def test_parcel_count_matches_the_anatomy_prior():
    """414 appears on the landing page as a load-bearing figure."""
    from scwbd.foundation.anatomy import load_anatomy

    n = load_anatomy().n_regions
    text = _all_site_text()
    if "414" not in text:
        pytest.skip("the site does not cite a parcel count")
    assert n == 414, f"anatomy says {n}, the site says 414"


def test_the_orientation_ratio_is_the_measured_one_not_the_escalated_one():
    """We published 2.64x after measuring it; ~9x was another forward model.

    This is the figure most likely to drift back, because 9x is the more
    exciting number and it appears in the reports as an escalation that was
    later corrected.
    """
    text = _all_site_text()
    if "2.64" not in text and "9×" not in text and "9x" not in text:
        pytest.skip("the site does not cite an orientation ratio")
    # If the site cites 9x as OUR result it must also carry the correction.
    if re.search(r"\b9(\.0)?\s*[x×]", text):
        assert "2.64" in text, "the site cites ~9x without the measured 2.64x"


def test_the_impulse_null_p_value_matches_the_artifact():
    """The site quotes p = 0.005; the harness wrote the real value."""
    f = ROOT / "reports/intervene/impulse_pilot.json"
    if not f.exists():
        pytest.skip("no impulse result on disk")
    d = json.loads(f.read_text())
    null = d.get("shuffled_normal_null", {})
    if null.get("skipped"):
        pytest.skip("the null was skipped in the current artifact")
    p = null["p_one_sided"]
    text = _all_site_text()
    if "0.005" not in text and "0.00498" not in text:
        pytest.skip("the site does not cite the null p-value")
    assert p < 0.01, f"site says p ~ 0.005, artifact says {p}"


def test_no_page_claims_a_trained_model_beats_a_baseline():
    """The one claim we must not make before the evaluation says so.

    Run 1 lost to five baselines. Run 2 is not scored yet. A page asserting
    otherwise would be the single most damaging thing on the site, so it is
    tested for directly rather than trusted to review.
    """
    text = _all_site_text().lower()
    # Self-referential claims only.  The first version of this test banned the
    # bare phrase "state of the art" and fired on
    #
    #   "honesty requires saying that the published state of the art is far
    #    from continuous open-vocabulary decoding from EEG"
    #
    # -- a sentence about the *field*, and an honest one.  An instrument that
    # cannot tell a claim from a disclaimer is the exact defect
    # reports/decorative_guards.md catalogues, and this is the fourth time it
    # has been written in this repository by someone who had just read that
    # entry.  Each pattern below must name us.
    # Second correction, for the same reason.  The page says
    #
    #   "on the conditional mean, with the paired intervals restored, SC-WBD
    #    beats every baseline"
    #
    # which is TRUE, measured, and carries its qualifier.  A test that cannot
    # tell a qualified claim from an unqualified one bans honest writing --
    # so the rule is not "never say beats", it is "never say it bare".
    QUALIFIERS = ("conditional mean", "mse", "on the mean", "variance channel")
    for m in re.finditer(r"\b(sc-wbd|our model|the model)\b[^.]{0,90}?\b(beats|outperforms)\b", text):
        window = text[max(0, m.start() - 220) : m.end() + 220]
        assert any(q in window for q in QUALIFIERS), (
            f"unqualified superiority claim: ...{text[m.start():m.end()+90]}..."
        )

    # 002 has no score yet.  Any claim about it beating anything is unsupported
    # regardless of qualifier, until the evaluation exists on disk.
    ev = ROOT / "reports/training/evaluation_run2.json"
    if not ev.exists():
        bad = re.search(r"\b002\b[^.]{0,80}\b(beats|outperforms|best)\b", text)
        assert bad is None, f"002 claim before it is scored: {bad.group(0) if bad else ''}"


def test_jacob_prompts_are_visible_not_just_commented():
    """A placeholder only in an HTML comment is a placeholder nobody sees."""
    text = _all_site_text()
    if "JACOB" not in text:
        pytest.skip("no operator prompts left")
    # every commented prompt should have a rendered counterpart somewhere
    n_comment = len(re.findall(r"<!--\s*JACOB", text))
    n_rendered = len(re.findall(r"class=\"[^\"]*todo[^\"]*\"", text))
    assert n_rendered > 0 or n_comment == 0, (
        f"{n_comment} JACOB comments and no rendered TODO block"
    )
