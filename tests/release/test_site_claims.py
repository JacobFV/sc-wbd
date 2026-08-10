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


def _declared_pair() -> dict:
    from scwbd.transforms import resolution_pair as rp

    art = ROOT / rp.MEASUREMENT_RELPATH
    if not art.is_file():
        pytest.skip("no declared resolution-pair measurement on disk")
    return json.loads(art.read_text())


def test_the_orientation_figures_are_the_declared_pairs_own():
    """The site's eta figures must come from the pair the model declares.

    They did not, for two runs.  ``reports/transforms/resolution_pair.json``
    measures ``cortical_source_dipole <= parcel`` on the 68-parcel
    Desikan-Killiany atlas: 5.6% for a scalar per parcel against 51.7% for a
    3-vector, a factor of 9.2.  SC-WBD runs Schaefer400x7, where the same pair
    on the same head and the same BEM lead field gives 32.1% against 83.4%, a
    factor of 2.6.  Both measurements are real; only one is about this model,
    and the site published the other one (ISSUE-015).

    So the figures are read out of ``rp.MEASUREMENT_RELPATH`` -- whichever
    artefact ``DECLARED_PARCELLATION`` names -- rather than written here.
    Switching the declared parcellation and not the site fails this test, which
    is the coupling that was missing.
    """
    d = _declared_pair()
    scalar = d["lead_field_energy_retained"]
    vector = d["specificity"]["net_dipole_moment"]["lead_field_energy_retained"]
    text = _all_site_text()
    for what, value in (("scalar per parcel", scalar), ("3-vector per parcel", vector)):
        pct = f"{100 * value:.1f}%"
        assert pct in text, (
            f"the declared pair retains {pct} for a {what} and no page says so; "
            f"the site is quoting some other measurement"
        )


def test_a_superseded_parcellations_figures_are_never_stated_bare():
    """5.6% and 51.7% may appear -- attributed, and only attributed.

    The defect ISSUE-015 records is not a wrong number.  Every affected sentence
    quoted a true measurement of the Desikan-Killiany atlas as a fact about
    SC-WBD.  A figure whose parcellation is unstated is the failure mode, so the
    guard is proximity to an attribution rather than absence of the digits.

    An earlier version of this test searched only for "9x" and "9×" while the
    page said "nine times" and "nine-fold", and skipped silently for weeks.  The
    pattern below carries every spelling that has appeared on this site.
    """
    d = _declared_pair()
    dk = d["specificity"]["aparc"]
    superseded = [
        f"{100 * dk['lead_field_energy_retained']:.1f}%",  # 5.6%
        "51.7%",
        "51.7 %",
    ]
    pattern = "|".join(re.escape(s) for s in superseded)
    pattern += r"|\b9(\.\d)?\s*[x×]|\bnine[- ](?:fold|times)"
    ATTRIBUTED = re.compile(r"desikan|dk-68|68[- ]parcel|68 parcels", re.I)
    text = _all_site_text()
    found = list(re.finditer(pattern, text, re.I))
    if not found:
        pytest.skip("the site no longer cites the Desikan-Killiany pair")
    for m in found:
        window = text[max(0, m.start() - 500) : m.end() + 500]
        assert ATTRIBUTED.search(window), (
            f"{m.group(0)!r} is on the site with no parcellation named within "
            f"500 characters: ...{text[max(0, m.start() - 120) : m.end() + 120]}..."
        )


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


def test_the_prose_layout_matches_what_the_checkpoint_says_it_is():
    """Run 2 was described as *ragged* in three places. It is padded.

    ``reports/RUN2.md``, the landing page, and ``paper/body.tex`` all said run 2
    used a segment/ragged state layout. The checkpoint's own ``state_layout``
    says ``family_padded``, and the ragged layout -- while built and tested -- is
    not what these weights use.

    All three errors ran the same direction: toward the design the project
    argues for. That is the drift-toward-intent class in
    ``reports/decorative_guards.md``, and the reason it survived is that nobody
    asked the checkpoint what it was.

    So this test asks. It is deliberately narrow -- one claim, checked against
    the artifact's self-report rather than against anyone's memory.
    """
    import torch

    ckpt = ROOT / "checkpoints/scwbd-002-pilot/last.pt"
    if not ckpt.is_file():
        pytest.skip("no run-2 checkpoint on disk")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    layout = (ck.get("regional_state") or {}).get("layout")
    if not layout:
        pytest.skip("the checkpoint records no layout")

    surfaces = {
        "reports/RUN2.md": (ROOT / "reports/RUN2.md").read_text(),
        "site": _all_site_text(),
    }
    for where, text in surfaces.items():
        # Only sentences that describe *this run's* state layout, not the ones
        # discussing the alternative -- which we legitimately built and write
        # about. Anchor on the pairing of "002"/"run 2" with a layout word.
        for m in re.finditer(
            r"(?:002|run[- ]2)[^.\n]{0,120}?\b(ragged|segment)\b[^.\n]{0,40}\blayout\b",
            text,
            re.I,
        ):
            window = text[max(0, m.start() - 200) : m.end() + 200]
            if "family_padded" in window or "padded" in window.lower():
                continue  # the sentence names the real layout too
            pytest.fail(
                f"{where} calls run 2's layout {m.group(1)!r}, "
                f"but the checkpoint records {layout!r}: ...{m.group(0)[:110]}..."
            )


def test_every_engineering_page_is_linked_from_its_index():
    """A published page nothing links to is a page nobody reads.

    Three pages sat unlinked for a day: 10, 11 and 12 were written, built,
    deployed, and reachable only by typing the URL. The section index still
    listed nine.

    This is the same shape as the failing-test list that was assumed complete
    and was not -- an enumeration nobody checked against the thing it
    enumerates. Cheap to test, invisible otherwise, because the index renders
    perfectly with entries missing.
    """
    eng = SITE / "engineering"
    if not eng.is_dir():
        pytest.skip("no engineering section")
    index = eng / "index.html"
    if not index.is_file():
        pytest.skip("no engineering index")
    text = index.read_text()
    pages = sorted(p.name for p in eng.glob("*.html") if p.name != "index.html")
    assert pages, "no engineering pages found -- this test would pass vacuously"
    missing = [p for p in pages if p not in text]
    assert not missing, f"engineering pages not linked from index.html: {missing}"
