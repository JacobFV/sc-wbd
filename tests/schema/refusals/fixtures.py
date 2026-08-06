"""One failing fixture per refusal R01..R11, plus the positive fixture.

Each ``build_rNN()`` returns ``(schema, claim)`` that MUST raise exactly that
code.  Every fixture is a *minimal* mutation of the valid three-region example,
so the test proves the check fires on the specific defect rather than on some
unrelated incompleteness elsewhere in the schema.

Mutations use ``model_copy(update=...)``, which deliberately bypasses
validation - that is how an invalid program gets built at all.  Real schemas
cannot be constructed this way through the public constructors.
"""

from __future__ import annotations

from typing import Callable

from scwbd.schema import (
    BrainSchema,
    ClaimManifest,
    CocycleCheck,
    GluingPolicy,
    Identification,
    ResidualPolicy,
    ScaleId,
    UncertaintyLedger,
)
from scwbd.schema.examples import build_three_region_claim, build_three_region_schema

__all__ = ["BUILDERS", "build_valid", "build_r01", "build_r02", "build_r03",
           "build_r04", "build_r05", "build_r06", "build_r07", "build_r08",
           "build_r09", "build_r10", "build_r11"]

Fixture = tuple[BrainSchema, ClaimManifest]


def build_valid() -> Fixture:
    """The positive fixture: compiles cleanly, no overrides."""
    return build_three_region_schema(), build_three_region_claim()


# ---------------------------------------------------------------------------
def _replace_operator(schema: BrainSchema, op_id: str, **updates) -> BrainSchema:
    ops = [
        op.model_copy(update=updates) if op.id == op_id else op
        for op in schema.operators
    ]
    return schema.model_copy(update={"operators": ops})


def _replace_source(schema: BrainSchema, source_id: str, **updates) -> BrainSchema:
    sources = [
        s.model_copy(update=updates) if s.id == source_id else s for s in schema.sources
    ]
    return schema.model_copy(update={"sources": sources})


# ---------------------------------------------------------------------------
# R01 - unknown units, clock, support, frame, handedness, transform lineage
# ---------------------------------------------------------------------------
def build_r01() -> Fixture:
    """The EEG amplifier clock is never declared, so ``eeg_out`` has no clock."""
    schema, claim = build_valid()
    clocks = [c for c in schema.clocks if str(c.id) != "eeg_amp"]
    return schema.model_copy(update={"clocks": clocks}), claim


# ---------------------------------------------------------------------------
# R02 - prolongation without a declared restriction partner / tested coverage
# ---------------------------------------------------------------------------
def build_r02() -> Fixture:
    """A parcel -> vertex prolongation with its restriction partner removed."""
    schema, claim = build_valid()
    poset = schema.resolution_poset
    maps = tuple(
        p.model_copy(update={"restriction": None, "landmark_tested": False})
        if str(p.fine) == "surface_vertex"
        else p
        for p in poset.maps
    )
    return (
        schema.model_copy(
            update={"resolution_poset": poset.model_copy(update={"maps": maps})}
        ),
        claim,
    )


# ---------------------------------------------------------------------------
# R03 - global cross-scale state with an out-of-tolerance cocycle residual
# ---------------------------------------------------------------------------
def build_r03() -> Fixture:
    """A global section is requested while the vertex/parcel/network path fails."""
    schema, claim = build_valid()
    poset = schema.resolution_poset
    gluing = GluingPolicy(
        materialize_global=True,
        cocycle_checks=(
            CocycleCheck(
                path=(ScaleId("surface_vertex"), ScaleId("parcel"), ScaleId("network")),
                residual=1.9,
                tolerance=0.4,
                n_samples=512,
            ),
        ),
        on_failure="preserve_sections",
    )
    schema = schema.model_copy(
        update={"resolution_poset": poset.model_copy(update={"gluing": gluing})}
    )
    return schema, claim.model_copy(update={"requires_global_section": True})


# ---------------------------------------------------------------------------
# R04 - effective/causal operator from passive correlation alone
# ---------------------------------------------------------------------------
def build_r04() -> Fixture:
    """``couple_c_a`` is relabelled effective while still correlation-only."""
    schema, claim = build_valid()
    return (
        _replace_operator(schema, "couple_c_a", mechanistic_status="effective"),
        claim,
    )


# ---------------------------------------------------------------------------
# R05 - learned residual dominating a mechanistic term silently
# ---------------------------------------------------------------------------
def build_r05() -> Fixture:
    """A learned residual on an effective term with no preregistered rho_max."""
    schema, claim = build_valid()
    return (
        _replace_operator(
            schema,
            "residual_b_a",
            mechanistic_status="effective",
            residual=ResidualPolicy(
                rho_max=None, measured_ratio=0.9, validity_set=None,
                report_violations=True,
            ),
        ),
        claim,
    )


# ---------------------------------------------------------------------------
# R06 - adaptive stepping for a learned propagator without semigroup testing
# ---------------------------------------------------------------------------
def build_r06() -> Fixture:
    """The sim clock enables adaptive stepping; the surrogate is untested."""
    schema, claim = build_valid()
    clocks = [
        c.model_copy(update={"adaptive_stepping": True}) if str(c.id) == "sim" else c
        for c in schema.clocks
    ]
    schema = schema.model_copy(update={"clocks": clocks})
    return _replace_operator(schema, "residual_b_a", semigroup=None), claim


# ---------------------------------------------------------------------------
# R07 - hierarchical effects without centering or shrinkage
# ---------------------------------------------------------------------------
def build_r07() -> Fixture:
    """The EEG session effect becomes unconstrained with no shrinkage prior."""
    schema, claim = build_valid()
    source = schema.source("eeg_sim_v1")
    effects = tuple(
        e.model_copy(
            update={
                "parameterization": "unconstrained",
                "shrinkage_prior": None,
                "recovery_tested": False,
            }
        )
        for e in source.population.effects
    )
    population = source.population.model_copy(update={"effects": effects})
    return _replace_source(schema, "eeg_sim_v1", population=population), claim


# ---------------------------------------------------------------------------
# R08 - bias point estimate with no estimator or external bound
# ---------------------------------------------------------------------------
def build_r08() -> Fixture:
    """The fMRI ledger asserts a bias of exactly 0.3 with nothing behind it."""
    schema, claim = build_valid()
    ledger = UncertaintyLedger(
        variance={"measurement": 0.25},
        bias_interval=(0.3, 0.3),
        bias_status="prior_specified_sensitivity",
        units="dimensionless",
    )
    return _replace_source(schema, "fmri_sim_v1", ledger=ledger), claim


# ---------------------------------------------------------------------------
# R09 - pseudo-likelihood treated as a calibrated posterior likelihood
# ---------------------------------------------------------------------------
def build_r09() -> Fixture:
    """An uncalibrated agreement penalty enters as a likelihood factor."""
    schema, claim = build_valid()
    source = schema.source("eeg_sim_v1")
    observation = source.observation.model_copy(
        update={"likelihood_kind": "agreement_penalty", "calibration_status": "uncalibrated"}
    )
    schema = _replace_source(schema, "eeg_sim_v1", observation=observation)
    return schema, claim.model_copy(update={"posterior_class": "calibrated_bayesian"})


# ---------------------------------------------------------------------------
# R10 - derived sessions crossing a parent-level holdout
# ---------------------------------------------------------------------------
def build_r10() -> Fixture:
    """``sub-01_ses-02`` is moved to test while its parent stays in train."""
    schema, claim = build_valid()
    source = schema.source("eeg_sim_v1")
    folds = dict(source.split_policy.fold_assignments)
    folds["sub-01_ses-02"] = "test"
    policy = source.split_policy.model_copy(update={"fold_assignments": folds})
    return _replace_source(schema, "eeg_sim_v1", split_policy=policy), claim


# ---------------------------------------------------------------------------
# R11 - intervention optimization outside a validated feasible set
# ---------------------------------------------------------------------------
def build_r11() -> Fixture:
    """Optimization is requested over an intervention whose A_safe is gone."""
    schema, claim = build_valid()
    source = schema.source("impulse_sim_v1")
    intervention = source.intervention.model_copy(update={"a_safe": None})
    schema = _replace_source(schema, "impulse_sim_v1", intervention=intervention)
    return schema, claim.model_copy(update={"optimizes_intervention": True})


BUILDERS: dict[str, Callable[[], Fixture]] = {
    "R01": build_r01,
    "R02": build_r02,
    "R03": build_r03,
    "R04": build_r04,
    "R05": build_r05,
    "R06": build_r06,
    "R07": build_r07,
    "R08": build_r08,
    "R09": build_r09,
    "R10": build_r10,
    "R11": build_r11,
}
