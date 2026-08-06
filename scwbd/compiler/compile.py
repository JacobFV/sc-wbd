"""``compile(schema, *, claim) -> CompiledModel``.

The driver runs the eleven checks in table order and fails closed on the first
refusal that is not explicitly overridden.  An override does not make the
refusal disappear: it is recorded in ``CompiledModel.provenance.refusals`` with
``overridden=True``, and the artifact's ``effective_claim_class`` is demoted to
the class the override declared.  A compiled model can therefore never report a
stronger claim than the evidence its schema supplies.
"""

from __future__ import annotations

from typing import Iterator

from ..schema.claims import ClaimManifest, is_demotion
from ..schema.refusals import REFUSAL_CODES, CompilerRefusal, RefusalRecord
from ..schema.schema import BrainSchema
from . import checks as _checks
from .adjacency import build_adjacency
from .dispatch import build_dispatch
from .graphs import build_clock_graph, build_frame_graph
from .layout import build_state_layout
from .ledger import build_ledger
from .masks import build_gradient_masks
from .model import COMPILER_VERSION, CompiledModel, Provenance
from .schedule import build_schedule

__all__ = ["compile", "run_checks", "first_refusal"]


def _iter_refusals(
    code: str, schema: BrainSchema, claim: ClaimManifest, frame_graph, clock_graph
) -> Iterator[CompilerRefusal]:
    if code == "R01":
        yield from _checks.check_r01(schema, claim, frame_graph, clock_graph)
    else:
        yield from getattr(_checks, f"check_{code.lower()}")(schema, claim)


def run_checks(
    schema: BrainSchema, claim: ClaimManifest
) -> tuple[CompilerRefusal, ...]:
    """Every refusal the schema triggers, in table order. Raises nothing.

    Useful for tooling and for tests that want the whole picture; ``compile``
    itself stops at the first non-overridden refusal.
    """
    frame_graph = build_frame_graph(schema)
    clock_graph = build_clock_graph(schema)
    out: list[CompilerRefusal] = []
    for code in REFUSAL_CODES:
        out.extend(_iter_refusals(code, schema, claim, frame_graph, clock_graph))
    return tuple(out)


def first_refusal(
    schema: BrainSchema, claim: ClaimManifest
) -> CompilerRefusal | None:
    for refusal in run_checks(schema, claim):
        if claim.override_for(refusal.code) is None:
            return refusal
    return None


def compile(schema: BrainSchema, *, claim: ClaimManifest) -> CompiledModel:
    """Compile a ``BrainSchema`` under a ``ClaimManifest``.

    Raises :class:`~scwbd.schema.refusals.CompilerRefusal` on the first refusal
    the claim manifest does not explicitly override.
    """
    if not isinstance(schema, BrainSchema):  # pragma: no cover - misuse guard
        raise TypeError(f"expected a BrainSchema, got {type(schema).__name__}")
    if not isinstance(claim, ClaimManifest):  # pragma: no cover
        raise TypeError(f"expected a ClaimManifest, got {type(claim).__name__}")

    frame_graph = build_frame_graph(schema)
    clock_graph = build_clock_graph(schema)

    records: list[RefusalRecord] = []
    passed: list[str] = []
    effective_class = claim.claim_class

    for code in REFUSAL_CODES:
        fired = False
        for refusal in _iter_refusals(code, schema, claim, frame_graph, clock_graph):
            fired = True
            override = claim.override_for(refusal.code)
            if override is None:
                # Fail closed.  Record nothing: the artifact does not exist.
                raise refusal
            # An override is visible and costly: the recorded claim class moves.
            if is_demotion(effective_class, override.resulting_claim_class):
                effective_class = override.resulting_claim_class
            records.append(
                refusal.record(
                    overridden=True, resulting_claim_class=override.resulting_claim_class
                )
            )
        if not fired:
            passed.append(code)

    # -- emit -------------------------------------------------------------
    layout = build_state_layout(schema)
    adjacency = build_adjacency(schema)
    dispatch = build_dispatch(schema, layout)
    schedule = build_schedule(schema, layout)
    masks = build_gradient_masks(schema)
    ledger = build_ledger(schema)

    warnings = _collect_warnings(schema, layout, masks, ledger, claim, records)
    authorizations, claim_scope = _authorization_provenance(schema, claim, records)

    provenance = Provenance(
        schema_id=schema.id,
        schema_version=schema.version,
        schema_hash=schema.content_hash(),
        claim_id=claim.id,
        requested_claim_class=claim.claim_class,
        effective_claim_class=effective_class,
        posterior_class=claim.posterior_class,
        compiler_version=COMPILER_VERSION,
        refusals=tuple(records),
        checks_passed=tuple(passed),
        warnings=tuple(warnings),
        target_gates=tuple(claim.target_gates),
        disabling_evidence=claim.disabling_evidence,
        claim_scope=claim_scope,
        authorizations=authorizations,
        extra={
            "abi_digest": layout.abi_digest(),
            "boundary_abi_digest": layout.boundary_abi_digest(),
            "n_regions": len(schema.regions),
            "n_operators": len(schema.operators),
            "n_sources": len(schema.sources),
            "hyperperiod_s": schedule.hyperperiod,
        },
    )

    return CompiledModel(
        schema=schema,
        claim=claim,
        state_layout=layout,
        adjacency=adjacency,
        dispatch=dispatch,
        schedule=schedule,
        gradient_masks=masks,
        frame_graph=frame_graph,
        clock_graph=clock_graph,
        ledger=ledger,
        provenance=provenance,
    )


def _authorization_provenance(
    schema: BrainSchema, claim: ClaimManifest, records: list[RefusalRecord]
) -> tuple[tuple[dict, ...], str]:
    """Every authorization verdict this artifact was compiled under, and its scope.

    Reached only after ``check_r11`` has passed, so any verdict recorded here
    is an admitted one.  Compiling under a validated authorization **changes
    the claim the artifact carries**: its scope moves from ``simulation_only``
    to ``protocol:<id>@<version>``, the record's content hash is pinned, and
    both travel in ``CompiledModel.provenance`` for every downstream consumer.
    The thesis requires exactly this ("remains visible in its provenance"); a
    silently-authorized artifact would be indistinguishable from an
    unauthorized one, which is the defect this gate exists to remove.
    """
    if claim.authorization is None:
        return (), "simulation_only"

    classes: list[str] = sorted(
        {
            s.intervention.modality
            for s in schema.sources
            if s.intervention is not None
            and s.intervention.is_prospective_human
            and s.intervention.is_physical_stimulation
        }
    )
    if not classes:
        # A record was declared but nothing prospective was requested. Record
        # it as carried-but-unexercised rather than dropping it: a reader must
        # be able to see that a declaration was attached to this build.
        return (
            (
                {
                    "admitted": False,
                    "record_id": claim.authorization.id,
                    "record_hash": claim.authorization.content_hash(),
                    "protocol": claim.authorization.protocol_reference,
                    "claim_scope": "simulation_only",
                    "note": (
                        "an authorization record was declared but no source card "
                        "requests a prospective human physical intervention, so "
                        "nothing was authorized and the artifact stays "
                        "simulation-only"
                    ),
                },
            ),
            "simulation_only",
        )

    # An overridden R11 is not an authorized R11.  If the gate fired and the
    # manifest paid for it with a claim demotion, the artifact keeps the
    # demotion *and* keeps simulation-only scope: an override buys visibility,
    # never a protocol.
    overridden_r11 = any(r.code == "R11" and r.overridden for r in records)

    entries: list[dict] = []
    scope = "simulation_only"
    for modality in classes:
        verdict = claim.authorization_for(modality, what="compiled artifact")
        entry = verdict.as_provenance()
        if overridden_r11:
            entry["note"] = (
                "R11 fired and was overridden; the artifact carries the "
                "override's demoted claim class and stays simulation-only"
            )
        entries.append(entry)
        if verdict.admitted and not overridden_r11:
            scope = verdict.claim_scope
    return tuple(entries), scope


def _collect_warnings(
    schema: BrainSchema,
    layout,
    masks,
    ledger,
    claim: ClaimManifest,
    records: list[RefusalRecord],
) -> list[str]:
    """Non-fatal observations worth surfacing in a claim report.

    These are deliberately not refusals: none of them is in the thesis table.
    They are the things a reviewer should still see.
    """
    warnings: list[str] = []

    unreachable = masks.unreachable_groups()
    if unreachable:
        warnings.append(
            f"{len(unreachable)} parameter groups are updatable by no source and "
            f"will never train: {list(unreachable[:5])}"
            + ("..." if len(unreachable) > 5 else "")
        )
    for source_id in masks.keys():
        unmatched = masks[source_id].unmatched_patterns
        if unmatched:
            warnings.append(
                f"source {source_id!r} grants {list(unmatched)}, which match no "
                "parameter group"
            )

    # ARCHITECTURE.md sec. 3: bfloat16 is permitted only inside learned
    # operators, never in solvers, Fisher information, or covariance.
    bf16 = [e.qualified for e in layout.entries if e.dtype == "bfloat16"]
    if bf16:
        warnings.append(
            f"state components stored as bfloat16: {bf16}; solvers, Fisher "
            "information and covariance propagation must not consume them"
        )

    if ledger.unknown_components:
        warnings.append(
            f"variance keys outside the declared taxonomy: "
            f"{list(ledger.unknown_components)}"
        )

    sensitivity = ledger.sensitivity_terms()
    if sensitivity:
        warnings.append(
            f"{len(sensitivity)} objects carry prior-specified-sensitivity bias and "
            "must be swept, not advertised as empirically estimated"
        )

    for source in schema.sources:
        if source.role == "calibration" and any(
            g.startswith("region:") or g.startswith("operator:")
            for g in masks[source.id].allowed_groups()
        ):
            warnings.append(
                f"calibration source {source.id!r} updates biological parameter "
                "groups; a calibration factor should estimate nuisance terms"
            )
        if not source.split_policy.role_locked:
            warnings.append(
                f"source {source.id!r} has not locked its role before evaluation"
            )
        if not source.governance.is_resolved:
            warnings.append(
                f"source {source.id!r} governance is unresolved "
                f"(withdrawal={source.governance.withdrawal_status!r}, "
                f"redistribution={source.governance.redistribution!r})"
            )
        if not source.identity.is_reproducible:
            warnings.append(
                f"source {source.id!r} is not bit-reproducible: missing "
                f"{list(source.identity.missing_reproducibility_fields())}"
            )

    if not claim.disabling_evidence.strip():
        warnings.append(
            "claim manifest states no empirical finding that would disable it "
            "(ARCHITECTURE.md sec. 4)"
        )

    if records:
        warnings.append(
            f"{len(records)} refusals were overridden; the artifact's claim class is "
            "demoted accordingly and this is not reversible by re-compiling"
        )
    return warnings
