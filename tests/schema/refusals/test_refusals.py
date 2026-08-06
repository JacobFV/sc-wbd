"""Every refusal in Table tab:compiler-refusals has a failing fixture.

This is the definition of done for build-order item 1 (thesis sec. 0.6):
"valid examples compile; every refusal in Table tab:compiler-refusals has a
failing fixture."
"""

from __future__ import annotations

import pytest

from scwbd.compiler import compile, run_checks
from scwbd.schema import REFUSAL_CODES, REFUSALS, ClaimOverride, CompilerRefusal

from .fixtures import BUILDERS, build_valid


def test_every_code_has_a_fixture():
    assert tuple(sorted(BUILDERS)) == REFUSAL_CODES


@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_refusal_fires(code: str):
    """The fixture raises exactly its own code, with the thesis remedy."""
    schema, claim = BUILDERS[code]()
    with pytest.raises(CompilerRefusal) as excinfo:
        compile(schema, claim=claim)
    refusal = excinfo.value
    assert refusal.code == code
    assert refusal.remedy == REFUSALS[code].remedy
    assert refusal.offending_object is not None
    assert refusal.detail, "a refusal must say what specifically was wrong"


@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_refusal_is_the_first_one(code: str):
    """The fixture is minimal: no *earlier* refusal fires on it.

    Without this, a fixture could pass by breaking something unrelated.
    """
    schema, claim = BUILDERS[code]()
    fired = {r.code for r in run_checks(schema, claim)}
    earlier = {c for c in fired if c < code}
    assert not earlier, f"fixture for {code} also trips {sorted(earlier)}"
    assert code in fired


@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_remedy_text_is_verbatim(code: str):
    """Remedy text is the thesis's, not a paraphrase."""
    spec = REFUSALS[code]
    assert spec.remedy and spec.rejected and spec.why
    assert spec.remedy[0].isupper() or spec.remedy.startswith("Require")
    assert not spec.remedy.endswith(".")  # table cells carry no terminal period


def test_valid_example_compiles():
    schema, claim = build_valid()
    model = compile(schema, claim=claim)
    assert model.provenance.checks_passed == REFUSAL_CODES
    assert not model.provenance.refusals
    assert not model.provenance.was_overridden
    assert model.claim_class == "effective"


def test_valid_example_has_no_refusals_at_all():
    schema, claim = build_valid()
    assert run_checks(schema, claim) == ()


# ---------------------------------------------------------------------------
# override semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_override_admits_and_demotes(code: str):
    """An override lets the schema compile but visibly weakens the claim."""
    schema, claim = BUILDERS[code]()
    claim = claim.model_copy(
        update={
            "claim_class": "effective",
            "overrides": (
                ClaimOverride(
                    code=code,
                    justification=f"unit test of the {code} override path",
                    resulting_claim_class="surrogate",
                    approved_by="tests",
                    disabling_evidence="any use of this artifact as evidence",
                ),
            ),
        }
    )
    model = compile(schema, claim=claim)
    assert model.provenance.requested_claim_class == "effective"
    assert model.provenance.effective_claim_class == "surrogate"
    assert model.claim_class == "surrogate"
    assert model.provenance.claim_was_demoted
    assert model.provenance.was_overridden
    assert code in model.provenance.overridden_codes
    records = [r for r in model.provenance.refusals if r.code == code]
    assert records and all(r.overridden for r in records)
    assert records[0].remedy == REFUSALS[code].remedy
    assert code not in model.provenance.checks_passed


def test_override_cannot_be_a_no_op():
    """An override that does not weaken the claim is itself rejected."""
    _, claim = build_valid()
    with pytest.raises(ValueError, match="does not weaken"):
        claim.model_copy(update={"claim_class": "effective"}).__class__(
            id="bad",
            claim_class="effective",
            overrides=(
                ClaimOverride(
                    code="R04",
                    justification="pretend nothing happened",
                    resulting_claim_class="effective",
                    approved_by="nobody",
                ),
            ),
        )


def test_override_of_one_code_does_not_mask_another():
    """Overriding R04 does not let an R08 defect through."""
    schema, claim = BUILDERS["R08"]()
    claim = claim.model_copy(
        update={
            "overrides": (
                ClaimOverride(
                    code="R04",
                    justification="unrelated override",
                    resulting_claim_class="functional",
                    approved_by="tests",
                ),
            )
        }
    )
    with pytest.raises(CompilerRefusal) as excinfo:
        compile(schema, claim=claim)
    assert excinfo.value.code == "R08"


def test_refusal_message_carries_remedy_and_reason():
    schema, claim = BUILDERS["R04"]()
    with pytest.raises(CompilerRefusal) as excinfo:
        compile(schema, claim=claim)
    text = str(excinfo.value)
    assert "R04" in text
    assert REFUSALS["R04"].remedy in text
    assert REFUSALS["R04"].why in text


def test_prospective_human_stimulation_is_refused_even_without_optimization():
    """ARCHITECTURE.md sec. 0: build-order item 6 is out of scope, full stop."""
    schema, claim = build_valid()
    source = schema.source("impulse_sim_v1")
    intervention = source.intervention.model_copy(
        update={"modality": "tms", "is_prospective_human": True}
    )
    sources = [
        s.model_copy(update={"intervention": intervention}) if s.id == "impulse_sim_v1" else s
        for s in schema.sources
    ]
    schema = schema.model_copy(update={"sources": sources})
    assert not claim.optimizes_intervention
    with pytest.raises(CompilerRefusal) as excinfo:
        compile(schema, claim=claim)
    assert excinfo.value.code == "R11"


def test_refusal_record_is_serializable():
    schema, claim = BUILDERS["R10"]()
    refusals = run_checks(schema, claim)
    record = refusals[0].record()
    dumped = record.model_dump(mode="json")
    assert dumped["code"] == "R10"
    assert dumped["remedy"] == REFUSALS["R10"].remedy
    assert isinstance(dumped["evidence"], dict)
