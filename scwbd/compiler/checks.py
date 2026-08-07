"""The eleven mandatory refusal checks (thesis Table tab:compiler-refusals).

Each ``check_rNN`` is a generator of :class:`CompilerRefusal` instances.  The
driver in :mod:`scwbd.compiler.compile` runs them in code order and, for each
refusal, either raises it (fail closed) or - if and only if
``ClaimManifest.overrides`` names that code - records it and *demotes the claim
class recorded in provenance*.

"The compiler earns credibility by rejecting programs" (ARCHITECTURE.md sec. 7,
rule 7).  A check that is hard to satisfy is doing its job; the remedy text
tells the author exactly what the thesis requires instead.
"""

from __future__ import annotations

from typing import Iterator

from ..schema.authorization import AuthorizationVerdict
from ..schema.claims import ClaimManifest
from ..schema.clocks import UNVERIFIED_SYNC
from ..schema.lineage import LineageError
from ..schema.refusals import CompilerRefusal
from ..schema.schema import BrainSchema
from .graphs import CompiledClockGraph, CompiledFrameGraph

__all__ = [
    "check_r01",
    "check_r02",
    "check_r03",
    "check_r04",
    "check_r05",
    "check_r06",
    "check_r07",
    "check_r08",
    "check_r09",
    "check_r10",
    "check_r11",
    "ALL_CHECKS",
]


# ---------------------------------------------------------------------------
# R01 - unknown units, clock, physical support, frame, handedness, transform
#       lineage
# ---------------------------------------------------------------------------
def check_r01(
    schema: BrainSchema,
    claim: ClaimManifest,
    frame_graph: CompiledFrameGraph,
    clock_graph: CompiledClockGraph,
) -> Iterator[CompilerRefusal]:
    poset = schema.resolution_poset

    def refuse(obj, detail: str, **evidence):
        return CompilerRefusal("R01", offending_object=obj, detail=detail, evidence=evidence)

    # -- every declared support must name a frame and a clock that exist ----
    for region in schema.regions:
        for name, comp in region.state.components.items():
            where = f"{region.id}.{name}"
            if not frame_graph.has(str(comp.support.frame)):
                yield refuse(comp.support, f"state {where} names undeclared frame "
                             f"{str(comp.support.frame)!r}", object=where)
            if not schema.has_clock(str(comp.temporal.clock)):
                yield refuse(comp.temporal, f"state {where} names undeclared clock "
                             f"{str(comp.temporal.clock)!r}", object=where)
            if comp.resolution is not None and not poset.has(str(comp.resolution)):
                yield refuse(comp, f"state {where} names undeclared resolution "
                             f"{str(comp.resolution)!r}", object=where)
        for port in region.ports:
            where = f"{region.id}.{port.name}"
            if not frame_graph.has(str(port.support.frame)):
                yield refuse(port, f"port {where} names undeclared frame "
                             f"{str(port.support.frame)!r}", object=where)
            if not schema.has_clock(str(port.temporal.clock)):
                yield refuse(port, f"port {where} names undeclared clock "
                             f"{str(port.temporal.clock)!r}", object=where)
            if port.role in ("observation", "intervention") and port.support.psf is None:
                yield refuse(
                    port,
                    f"{port.role} port {where} declares no point-spread function; a "
                    "nominal coordinate is not a physical support",
                    object=where,
                )

    # -- frames used on a live path need lineage back to the root ----------
    for frame in sorted(frame_graph.used_frames):
        if not frame_graph.has(frame):
            yield refuse(frame, f"frame {frame!r} is referenced but never declared")
            continue
        node = frame_graph.node(frame)
        if frame_graph.path(frame_graph.root, frame) is None:
            yield refuse(
                node,
                f"frame {frame!r} has no transform lineage to root "
                f"{frame_graph.root!r}",
            )
        if node.validity_interval is None:
            yield refuse(node, f"frame {frame!r} declares no validity interval")

    for src, dst in frame_graph.spec.handedness_conflicts():
        yield refuse(
            frame_graph.spec, f"transform {src}->{dst} joins frames of opposite "
            "handedness without a declared reflection", edge=f"{src}->{dst}",
        )
    for src, dst in frame_graph.spec.unit_conflicts():
        yield refuse(
            frame_graph.spec,
            f"transform {src}->{dst} emits units incompatible with the target frame",
            edge=f"{src}->{dst}",
        )

    # -- every transform edge needs lineage and a calibrated manifest ------
    for edge in frame_graph.edges():
        tag = f"{edge.src}->{edge.dst}"
        if edge.transform == "identity":
            continue
        if not edge.lineage.strip():
            yield refuse(edge, f"transform {tag} declares no lineage (how it was obtained)")
        if edge.calibration is None:
            yield refuse(edge, f"transform {tag} has no calibration manifest")
        elif not edge.calibration.is_complete:
            yield refuse(
                edge,
                f"transform {tag} calibration is incomplete: missing "
                f"{list(edge.calibration.missing_fields())}",
                missing=list(edge.calibration.missing_fields()),
            )
        if not edge.roundtrip_ok:
            yield refuse(
                edge,
                f"transform {tag} round-trip residual "
                f"{edge.roundtrip_residual} exceeds tolerance {edge.roundtrip_tolerance}",
            )

    # -- clock relations must be evidenced ---------------------------------
    for cid in clock_graph.orphans():
        yield refuse(
            clock_graph.spec(cid),
            f"clock {cid!r} references undeclared clock "
            f"{str(clock_graph.spec(cid).reference)!r}",
        )
    for cid in clock_graph.unverified():
        spec = clock_graph.spec(cid)
        yield refuse(
            spec,
            f"clock {cid!r} relation to {str(spec.reference)!r} is "
            f"{spec.sync_evidence!r}; a physical synchronization event or an "
            "independent cross-correlation target is required",
        )
    if clock_graph.master is None and schema.clocks:
        yield refuse(schema, "no master clock declared; every clock references another")

    # -- source cards ------------------------------------------------------
    for source in schema.sources:
        if not frame_graph.has(str(source.spatial.frame)):
            yield refuse(source, f"source {source.id!r} names undeclared frame "
                         f"{str(source.spatial.frame)!r}")
        if not schema.has_clock(str(source.temporal.clock)):
            yield refuse(source, f"source {source.id!r} names undeclared clock "
                         f"{str(source.temporal.clock)!r}")
        if source.spatial.psf is None:
            yield refuse(
                source,
                f"source {source.id!r} declares no point-spread function for its "
                "spatial support",
            )
        if not source.calibration.is_complete:
            yield refuse(
                source,
                f"source {source.id!r} calibration manifest is incomplete: missing "
                f"{list(source.calibration.missing_fields())}",
                missing=list(source.calibration.missing_fields()),
            )
        if source.signal is not None and not source.signal.units_are_meaningful:
            yield refuse(
                source,
                f"source {source.id!r} reports arbitrary units "
                f"{str(source.signal.units)!r} with no amplitude calibration",
            )
        if source.signal is not None and source.signal.irreversible_aggregation:
            yield refuse(
                source,
                f"source {source.id!r} declares irreversible aggregation that removes "
                "the target signal",
            )


# ---------------------------------------------------------------------------
# R02 - prolongation without a declared restriction partner and tested coverage
# ---------------------------------------------------------------------------
def check_r02(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    poset = schema.resolution_poset
    for pair in poset.orphan_prolongations():
        reasons: list[str] = []
        if pair.restriction is None:
            reasons.append("no restriction partner")
        if not pair.roundtrip_tested:
            reasons.append("no round-trip test")
        if not pair.landmark_tested:
            reasons.append("no held-out landmark test")
        if pair.landmark_coverage is None:
            reasons.append("no landmark coverage")
        elif pair.landmark_coverage < pair.required_coverage:
            reasons.append(
                f"landmark coverage {pair.landmark_coverage} < required "
                f"{pair.required_coverage}"
            )
        if pair.out_of_support_policy is None:
            reasons.append("no out-of-support uncertainty policy")
        yield CompilerRefusal(
            "R02",
            offending_object=pair,
            detail=(
                f"prolongation {pair.coarse}->{pair.fine}: " + "; ".join(reasons)
            ),
            evidence={"fine": str(pair.fine), "coarse": str(pair.coarse), "reasons": reasons},
        )
    for pair in poset.unordered_maps():
        yield CompilerRefusal(
            "R02",
            offending_object=pair,
            detail=(
                f"map declared between {pair.fine} and {pair.coarse}, which are not "
                "ordered in the resolution poset; no defensible map exists"
            ),
        )


# ---------------------------------------------------------------------------
# R03 - global cross-scale state when overlap/cocycle residual exceeds tolerance
# ---------------------------------------------------------------------------
def check_r03(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    gluing = schema.resolution_poset.gluing
    wants_global = claim.requires_global_section or gluing.materialize_global
    if not wants_global:
        return
    if not gluing.cocycle_checks:
        yield CompilerRefusal(
            "R03",
            offending_object=gluing,
            detail=(
                "a global cross-scale section was requested but no overlap/cocycle "
                "residual was measured; the residual cannot be shown to be within "
                "tolerance"
            ),
        )
        return
    failures = gluing.failures()
    if failures:
        cert = gluing.certificate()
        yield CompilerRefusal(
            "R03",
            offending_object=gluing,
            detail=(
                "cocycle residual exceeds tolerance on "
                + ", ".join(
                    f"{'->'.join(str(s) for s in c.path)} ({c.residual} > {c.tolerance})"
                    for c in failures
                )
            ),
            evidence={
                "obstruction_certificate": cert.model_dump(mode="json"),
                "on_failure": gluing.on_failure,
            },
        )


# ---------------------------------------------------------------------------
# R04 - effective/causal operator estimated from passive correlation alone
# ---------------------------------------------------------------------------
def check_r04(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    for op in schema.operators:
        if not op.claims_causality:
            continue
        ident = op.identification
        if ident.supports_causal_claim:
            continue
        if not ident.basis:
            detail = (
                f"operator {op.key} claims {op.mechanistic_status!r} status but "
                "declares no identification basis at all"
            )
        else:
            detail = (
                f"operator {op.key} claims {op.mechanistic_status!r} status from "
                f"basis {list(ident.basis)} with no intervention and no assumption set"
            )
        yield CompilerRefusal(
            "R04",
            offending_object=op,
            detail=detail,
            evidence={
                "basis": list(ident.basis),
                "mechanistic_status": op.mechanistic_status,
                "evidence_sources": list(ident.evidence_source_ids),
            },
        )


# ---------------------------------------------------------------------------
# R05 - learned residual allowed to dominate a mechanistic term silently
# ---------------------------------------------------------------------------
def check_r05(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    for op in schema.operators:
        res = op.residual
        if res is None:
            continue
        if op.mechanistic_status == "surrogate":
            continue  # nothing mechanistic is being claimed
        if not res.is_preregistered:
            yield CompilerRefusal(
                "R05",
                offending_object=op,
                detail=(
                    f"operator {op.key} carries a learned residual on a "
                    f"{op.mechanistic_status!r} term with no preregistered rho_max "
                    "on a declared validity set"
                ),
                evidence={"rho_max": res.rho_max, "validity_set": res.validity_set},
            )
            continue
        if res.measured_ratio is None:
            yield CompilerRefusal(
                "R05",
                offending_object=op,
                detail=(
                    f"operator {op.key} declares rho_max={res.rho_max} but the ratio "
                    f"was never measured on {res.validity_set!r}"
                ),
            )
            continue
        if res.violates and not res.reclassified_as_surrogate:
            yield CompilerRefusal(
                "R05",
                offending_object=op,
                detail=(
                    f"operator {op.key} residual ratio {res.measured_ratio} exceeds "
                    f"rho_max {res.rho_max} on {res.validity_set!r} and the module was "
                    "not reclassified as a surrogate"
                ),
                evidence={
                    "measured_ratio": res.measured_ratio,
                    "rho_max": res.rho_max,
                    "validity_set": res.validity_set,
                },
            )
        if not res.report_violations:
            yield CompilerRefusal(
                "R05",
                offending_object=op,
                detail=(
                    f"operator {op.key} suppresses residual violation reporting; the "
                    "named mechanism becomes unfalsifiable"
                ),
            )


# ---------------------------------------------------------------------------
# R06 - adaptive step sizes for a learned propagator without semigroup testing
# ---------------------------------------------------------------------------
def check_r06(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    from .dispatch import resolve_operator_clock

    for op in schema.operators:
        if not op.is_learned:
            continue
        clock_id = resolve_operator_clock(schema, op)
        if not schema.has_clock(clock_id):
            continue  # R01 already reports this
        clock = schema.clock(clock_id)
        if not clock.adaptive_stepping:
            continue
        sg = op.semigroup
        if sg is None or not sg.is_tested:
            yield CompilerRefusal(
                "R06",
                offending_object=op,
                detail=(
                    f"learned propagator {op.key} runs on clock {clock_id!r} with "
                    "adaptive stepping enabled but reports no semigroup residual"
                ),
                evidence={"clock": clock_id},
            )
            continue
        untested = sg.untested_permitted_pairs()
        if untested:
            yield CompilerRefusal(
                "R06",
                offending_object=op,
                detail=(
                    f"learned propagator {op.key} permits step pairs {list(untested)} "
                    "that were never tested for the semigroup property"
                ),
                evidence={"untested_pairs": [list(p) for p in untested]},
            )
            continue
        if not sg.within_tolerance and not sg.folded_into_intervals:
            yield CompilerRefusal(
                "R06",
                offending_object=op,
                detail=(
                    f"learned propagator {op.key} semigroup residual "
                    f"{sg.max_residual} exceeds tolerance {sg.tolerance}; restrict the "
                    "scheduler or include the discrepancy in prediction intervals"
                ),
                evidence={"max_residual": sg.max_residual, "tolerance": sg.tolerance},
            )


# ---------------------------------------------------------------------------
# R07 - population/subject/session effects without centering or shrinkage
# ---------------------------------------------------------------------------
def check_r07(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    for source in schema.sources:
        for effect in source.population.unidentified_effects():
            yield CompilerRefusal(
                "R07",
                offending_object=effect,
                detail=(
                    f"source {source.id!r} effect {effect.name!r} at level "
                    f"{effect.level!r}: {effect.deficiency()}"
                ),
                evidence={
                    "source": source.id,
                    "effect": effect.name,
                    "level": effect.level,
                    "parameterization": effect.parameterization,
                },
            )


# ---------------------------------------------------------------------------
# R08 - bias point estimate without an estimator or external bound
# ---------------------------------------------------------------------------
def check_r08(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    for path, obj in schema.all_ledgers():
        ledger = obj  # UncertaintyLedger
        if ledger.has_estimator():  # type: ignore[union-attr]
            continue
        status = ledger.bias_status  # type: ignore[union-attr]
        if status == "design_estimable":
            why = "declared design-estimable but names no estimator"
        elif status == "externally_bounded":
            why = "declared externally bounded but names no external bound source"
        else:
            why = (
                "declared prior-specified sensitivity but carries a point estimate "
                f"{ledger.bias_interval}; it must be propagated as a range"  # type: ignore[union-attr]
            )
        yield CompilerRefusal(
            "R08",
            offending_object=ledger,
            detail=f"{path}: {why}",
            evidence={
                "object": path,
                "bias_status": status,
                "bias_interval": list(ledger.bias_interval),  # type: ignore[union-attr]
            },
        )


# ---------------------------------------------------------------------------
# R09 - pseudo-likelihood treated as a calibrated posterior likelihood
# ---------------------------------------------------------------------------
def check_r09(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    for source in schema.sources:
        obs = source.observation
        if obs is None or source.role != "likelihood":
            continue
        if obs.is_generative:
            continue
        problems: list[str] = []
        if claim.posterior_class not in ("generalized", "pseudo"):
            problems.append(
                f"the claim manifest reports posterior_class="
                f"{claim.posterior_class!r} rather than a generalized/pseudo-posterior"
            )
        if not obs.is_calibrated_pseudo:
            problems.append(
                f"the factor is {obs.calibration_status!r} rather than calibrated "
                "empirically"
            )
        if not problems:
            continue
        yield CompilerRefusal(
            "R09",
            offending_object=source,
            detail=(
                f"source {source.id!r} contributes a {obs.likelihood_kind!r} factor as "
                "a likelihood: " + "; ".join(problems)
            ),
            evidence={
                "source": source.id,
                "likelihood_kind": obs.likelihood_kind,
                "calibration_status": obs.calibration_status,
                "posterior_class": claim.posterior_class,
            },
        )


# ---------------------------------------------------------------------------
# R10 - derived scans/sessions/relatives/replicas crossing a parent-level
#       holdout
# ---------------------------------------------------------------------------
def check_r10(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    registry = schema.lineage_registry()

    for source in schema.sources:
        ident = source.identity
        unresolved = ident.unresolved_parents(registry)
        if unresolved:
            yield CompilerRefusal(
                "R10",
                offending_object=ident,
                detail=(
                    f"source {source.id!r} declares parents {list(unresolved)} that are "
                    "not in the lineage registry; parentage is unresolved"
                ),
                evidence={"source": source.id, "unresolved_parents": list(unresolved)},
            )
        if ident.mutable_download:
            yield CompilerRefusal(
                "R10",
                offending_object=ident,
                detail=(
                    f"source {source.id!r} is an unversioned mutable download; it has "
                    "no immutable lineage identifier to group by"
                ),
                evidence={"source": source.id},
            )
        policy = source.split_policy
        if policy.leakage_barrier != "parent_lineage":
            yield CompilerRefusal(
                "R10",
                offending_object=policy,
                detail=(
                    f"source {source.id!r} splits with leakage_barrier="
                    f"{policy.leakage_barrier!r}; grouping must be by immutable lineage"
                ),
                evidence={"source": source.id, "barrier": policy.leakage_barrier},
            )
        if policy.stimuli_shared_across_folds:
            yield CompilerRefusal(
                "R10",
                offending_object=policy,
                detail=(
                    f"source {source.id!r} repeats stimuli across folds of a held-out "
                    "identity test"
                ),
                evidence={"source": source.id},
            )

    units = schema.lineage_units()
    if not units:
        return
    graph = schema.lineage_graph()

    for unit_id, missing in graph.unresolved_parents():
        yield CompilerRefusal(
            "R10",
            offending_object=graph.units[unit_id],
            detail=(
                f"lineage unit {unit_id!r} references {missing!r}, which is not "
                "declared; the run must fail when parentage is unresolved"
            ),
            evidence={"unit": unit_id, "missing_parent": missing},
        )
    for unit in units:
        if unit.is_synthetic_replica and not unit.parent_ids:
            yield CompilerRefusal(
                "R10",
                offending_object=unit,
                detail=(
                    f"simulator replica {unit.id!r} declares no parent; a replica "
                    "counted as an independent subject duplicates evidence"
                ),
                evidence={"unit": unit.id},
            )

    assignments: dict[str, str] = {}
    conflicting: list[str] = []
    for source in schema.sources:
        for uid, fold in source.split_policy.fold_assignments.items():
            if uid in assignments and assignments[uid] != fold:
                conflicting.append(uid)
            assignments[uid] = fold
    for uid in sorted(set(conflicting)):
        yield CompilerRefusal(
            "R10",
            offending_object=uid,
            detail=f"lineage unit {uid!r} is assigned to different folds by different sources",
            evidence={"unit": uid},
        )
    if not assignments:
        return
    try:
        crossings = graph.crossing_assignments(assignments)
        unassigned = graph.unassigned(assignments)
    except LineageError as exc:
        yield CompilerRefusal("R10", offending_object=schema, detail=str(exc))
        return
    for group, folds in crossings:
        yield CompilerRefusal(
            "R10",
            offending_object=graph.units[group],
            detail=(
                f"lineage group rooted at {group!r} crosses folds {list(folds)}; "
                f"members: {list(graph.groups()[group])}"
            ),
            evidence={"group": group, "folds": list(folds), "members": list(graph.groups()[group])},
        )
    if unassigned:
        yield CompilerRefusal(
            "R10",
            offending_object=schema,
            detail=(
                f"lineage units {list(unassigned)} have no fold assignment; parentage "
                "is unresolved with respect to the holdout"
            ),
            evidence={"unassigned": list(unassigned)},
        )


# ---------------------------------------------------------------------------
# R11 - intervention optimization outside an independently validated feasible
#       set
# ---------------------------------------------------------------------------
def _authorization_refusal(
    verdict: AuthorizationVerdict, offending: object, detail_prefix: str, evidence: dict
) -> CompilerRefusal:
    """Turn a refused authorization verdict into R11 with its specific reason."""
    return CompilerRefusal(
        "R11",
        remedy=(
            "supply a complete, in-date AuthorizationRecord whose consent scope "
            "covers the requested intervention class and whose A_safe is "
            "attributable to the named protocol; "
            + "; ".join(f.remedy for f in verdict.failures if f.remedy)
        ),
        offending_object=offending,
        detail=f"{detail_prefix}: {verdict.reason()}",
        evidence={
            **evidence,
            "authorization_failures": [
                f.model_dump(mode="json") for f in verdict.failures
            ],
            "authorization_failure_codes": list(verdict.failure_codes),
            "authorization_record_id": verdict.record_id,
            "authorization_record_hash": verdict.record_hash,
            "claim_scope": verdict.claim_scope,
        },
    )


def check_r11(schema: BrainSchema, claim: ClaimManifest) -> Iterator[CompilerRefusal]:
    """R11: intervention optimization outside an independently validated A_safe.

    Governance is **gated, not hard-coded**.  A prospective human stimulation
    protocol is refused *unless* ``claim.authorization`` is a validated
    :class:`~scwbd.schema.authorization.AuthorizationRecord` that covers the
    declared intervention class at ``claim.request_time_s``.  An unconditional
    refusal would be a constant, and a constant cannot discriminate between a
    project holding a current IRB approval and one holding nothing; it would
    encode a belief about the world rather than check one.

    Every other reason R11 ever refused still refuses, authorization or not:
    no ``A_safe``, an ``A_safe`` that is not independently validated, an
    uncalibrated dose or pose, or optimization requested with no intervention
    model at all.  Authorization admits operating *within* declared limits; it
    never widens them.

    Claim limit: a validated record is a *recorded declaration*.  Nothing here
    establishes that an ethics approval exists in the world.
    """
    interventions = [(s, s.intervention) for s in schema.sources if s.intervention is not None]

    # -- the governance gate -------------------------------------------------
    for source, iv in interventions:
        if not (iv.is_prospective_human and iv.is_physical_stimulation):
            continue
        axes = tuple(sorted(iv.a_safe.constraints)) if iv.a_safe is not None else ()
        verdict = claim.authorization_for(
            iv.modality,
            a_safe_id=iv.a_safe.id if iv.a_safe is not None else None,
            required_a_safe_axes=axes,
            what=f"source card {source.id!r}",
        )
        if not verdict.admitted:
            yield _authorization_refusal(
                verdict,
                iv,
                (
                    f"source {source.id!r} declares prospective human "
                    f"{iv.modality} stimulation and no validated authorization "
                    "admits it"
                ),
                {"source": source.id, "modality": iv.modality},
            )

    if claim.prospective_human:
        classes = sorted(
            {
                iv.modality
                for _, iv in interventions
                if iv.is_prospective_human and iv.is_physical_stimulation
            }
        )
        if not classes:
            yield CompilerRefusal(
                "R11",
                offending_object=claim,
                detail=(
                    "the claim manifest requests a prospective human protocol but "
                    "no source card declares a prospective human physical "
                    "intervention, so there is no intervention class to authorise "
                    "and nothing an approval could be checked against"
                ),
                evidence={"claim": claim.id},
            )
        for modality in classes:
            verdict = claim.authorization_for(modality, what="claim manifest")
            if not verdict.admitted:
                yield _authorization_refusal(
                    verdict,
                    claim,
                    (
                        "the claim manifest requests a prospective human "
                        f"{modality} protocol and no validated authorization "
                        "admits it"
                    ),
                    {"claim": claim.id, "modality": modality},
                )

    if not claim.optimizes_intervention:
        return

    if not interventions:
        yield CompilerRefusal(
            "R11",
            offending_object=claim,
            detail=(
                "intervention optimization was requested but no source card declares "
                "an intervention model, so no feasible set exists"
            ),
        )
        return

    for source, iv in interventions:
        safe = iv.a_safe
        if safe is None:
            yield CompilerRefusal(
                "R11",
                offending_object=iv,
                detail=(
                    f"source {source.id!r} is optimized over but declares no A_safe"
                ),
                evidence={"source": source.id},
            )
            continue
        if not safe.is_usable:
            reasons: list[str] = []
            if not safe.constraints:
                reasons.append("no declared constraints")
            if not safe.independently_validated:
                reasons.append("not independently validated")
            if safe.validator is None:
                reasons.append("no named validator")
            if safe.deferral_policy not in ("defer", "refuse"):
                reasons.append(f"deferral policy is {safe.deferral_policy!r}")
            yield CompilerRefusal(
                "R11",
                offending_object=safe,
                detail=(
                    f"source {source.id!r} A_safe {safe.id!r} is not an independently "
                    "validated feasible set: " + "; ".join(reasons)
                ),
                evidence={"source": source.id, "reasons": reasons},
            )
        if not iv.dose_independently_calibrated:
            yield CompilerRefusal(
                "R11",
                offending_object=iv,
                detail=(
                    f"source {source.id!r} is optimized over but its dose and pose are "
                    "not independently calibrated"
                ),
                evidence={"source": source.id},
            )
        if iv.is_physical_stimulation and iv.ethics_review is None:
            # An AuthorizationRecord is the machine-readable form of the same
            # thing, so it satisfies this requirement -- but only if it
            # validates for *this* modality, at the requested time, against
            # *this* A_safe.  A record that merely exists does not.
            verdict = claim.authorization_for(
                iv.modality,
                a_safe_id=safe.id,
                required_a_safe_axes=tuple(sorted(safe.constraints)),
                what=f"source card {source.id!r}",
            )
            if not verdict.admitted:
                yield _authorization_refusal(
                    verdict,
                    iv,
                    (
                        f"source {source.id!r} optimizes a {iv.modality} exposure "
                        "with no applicable ethics and regulatory review on record"
                    ),
                    {"source": source.id, "modality": iv.modality},
                )


#: Checks in table order.  R01 needs the compiled graphs; the rest do not.
ALL_CHECKS: tuple[str, ...] = (
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11",
)
