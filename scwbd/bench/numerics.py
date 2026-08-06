"""Numerical, representational and physical tests -- ``body.tex`` §11.1 (agent J).

    "The compiler is tested for shape, unit, coordinate, delay, and mask
    correctness. Modules must pass solver-convergence, stability,
    conservation, random-seed, and boundary-consistency tests where
    applicable. Electromagnetic and acoustic solvers are validated
    independently of neural-response models. Fine and coarse regional backends
    must agree within a declared tolerance on boundary observables **before**
    adaptive resolution is used for inference."

The last sentence is implemented as a *permit*
(:func:`permit_adaptive_resolution`).  Adaptive resolution is not a setting a
module may turn on; it is a capability that has to be earned by passing the
boundary-consistency test, and the permit is a machine-readable artifact that
records whether it was.

The solver validators take the solver and an analytic reference; the built-in
references (infinite-homogeneous-medium current dipole for EM, free-field
spherical spreading + Helmholtz residual for acoustics) contain no neural
response model at all, which is the point: a field solver that is only ever
checked through a neural read-out has not been validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import adapters
from .report import (
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    SubCheck,
    could_not_run,
)
from .statistics import bootstrap_ci

__all__ = [
    "check_compiler_correctness",
    "convergence_order",
    "check_solver_convergence",
    "check_stability",
    "check_conservation",
    "check_seed_reproducibility",
    "boundary_consistency",
    "AdaptiveResolutionPermit",
    "permit_adaptive_resolution",
    "analytic_dipole_potential",
    "analytic_free_field_pressure",
    "helmholtz_residual",
    "validate_em_solver",
    "validate_acoustic_solver",
    "validate_induced_efield_solver",
    "validate_induced_efield_contact",
    "run_numerics_suite",
]


def _manifest(claim_id: str, claim: str, falsified_by: str, consequence: str,
              *, seed: int = 0, thresholds: Mapping[str, Any] | None = None) -> ClaimManifest:
    return ClaimManifest(
        claim_id=claim_id,
        claim_text=claim,
        falsified_by=falsified_by,
        consequence_if_failed=consequence,
        thesis_reference="body.tex §11.1",
        acceptance_thresholds=dict(thresholds or {}),
        non_goals=[
            "These checks establish code correctness only. Numerical correctness is "
            "necessary, never sufficient: agreement with recorded signals is stronger, "
            "held-out perturbation stronger still (thesis §0.2).",
        ],
        seed=seed,
    )


# ==========================================================================
# compiler: shape / unit / coordinate / delay / mask correctness
# ==========================================================================
def _reference_compiled():
    """Agent A's worked example, compiled — the subject of the N1 check."""
    return adapters.reference_compiled()


def _dense_mask(adjacency: Any, cls: str) -> np.ndarray:
    """Dense boolean adjacency for one evidence class, whatever the storage."""
    m = adjacency.masks[cls]
    if hasattr(m, "to_dense"):
        try:
            m = m.to_dense()
        except Exception:  # pragma: no cover - already dense
            pass
    return np.asarray(m.numpy() if hasattr(m, "numpy") else m).astype(bool)


def check_compiler_correctness(compiled: Any = None, *, schema: Any = None,
                               use_reference_example: bool = True,
                               seed: int = 0) -> ClaimReport:
    """Shape, unit, coordinate, delay and mask correctness of a ``CompiledModel``.

    Written against agent A's compiler API (``ARCHITECTURE.md`` §2):
    ``.state_layout``, ``.adjacency``, ``.dispatch``, ``.schedule``,
    ``.gradient_masks``, ``.frame_graph``, ``.clock_graph``, ``.ledger``,
    ``.provenance``.  Every accessor is guarded, so a backend that omits one
    structure yields ``COULD_NOT_RUN`` for that sub-check rather than a crash
    or a free pass.

    With no argument the subject is the reference three-region schema
    (``scwbd.schema.examples.three_region``).  **That is what a PASS here
    means**: the compiler emits an internally consistent artifact *for the
    reference example*.  It is not a statement about a whole-brain schema,
    and the report says so.
    """
    subject = "caller-supplied CompiledModel"
    if compiled is None and use_reference_example:
        dep = _reference_compiled()
        if dep.available:
            compiled = dep.obj
            subject = "reference example: scwbd.schema.examples.three_region"
        else:
            reference_reason = dep.reason
    man = _manifest(
        "N1_compiler_correctness",
        "The compiler produces a model whose shapes, units, frames, clocks, delays, masks "
        "and gradient permissions are internally consistent, and whose recorded claim class "
        "is the one it was compiled for.",
        "any offset overlap or gap, an undeclared unit/frame/clock, a negative or "
        "unrepresentable delay, a mask that disagrees with the dispatched operator set, a "
        "gradient permission naming a module that does not exist, an unbacked bias term, or "
        "a silently demoted claim class",
        "Fix the compiler before any claim-bearing run; a numerically inconsistent "
        "compilation invalidates every downstream gate that consumes it.",
        seed=seed,
    )
    if compiled is None:
        return ClaimReport(
            manifest=man,
            subchecks=[
                could_not_run(
                    "compiled_model", "A CompiledModel to inspect.",
                    "no CompiledModel supplied and the reference example could not be "
                    f"compiled: {locals().get('reference_reason', 'reference disabled')}",
                    falsified_by=man.falsified_by,
                )
            ],
            kind="numerics",
        ).finalize()

    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {"subject": subject}
    schema = schema if schema is not None else getattr(compiled, "schema", None)

    # -- 1. state layout: disjoint, gapless, byte-consistent --------------
    layout = getattr(compiled, "state_layout", None)
    entries = tuple(getattr(layout, "entries", ()) or ())
    if layout is None or not entries:
        subs.append(could_not_run(
            "state_layout", "Packed state offsets per region/component.",
            "the compiled model exposes no .state_layout entries",
            falsified_by="overlapping or unaddressed state"))
    else:
        ordered = sorted(entries, key=lambda e: int(e.elem_offset))
        overlaps = gaps = byte_mismatch = 0
        cursor = 0
        for e in ordered:
            off, n = int(e.elem_offset), int(e.numel)
            if off < cursor:
                overlaps += 1
            elif off > cursor:
                gaps += 1
            cursor = max(cursor, off + n)
            width = int(e.nbytes) // max(n, 1)
            if int(e.nbytes) != width * n or int(e.byte_offset) % max(width, 1) != 0:
                byte_mismatch += 1
        total = int(getattr(layout, "total_elements", cursor))
        tail_gap = int(total != cursor)
        total_bytes = int(getattr(layout, "total_bytes", 0))
        byte_sum = sum(int(e.nbytes) for e in entries)
        artifacts["state_layout"] = {
            "n_entries": len(entries), "total_elements": total,
            "total_bytes": total_bytes, "sum_entry_bytes": byte_sum,
        }
        subs.append(SubCheck(
            name="state_layout",
            description="Region/component blocks tile the state vector exactly, and the "
                        "byte view agrees with the element view.",
            metrics=[
                Metric(name="layout.overlaps", value=float(overlaps), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better"),
                Metric(name="layout.gaps", value=float(gaps + tail_gap), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better",
                       note="state allocated but unaddressed by any region/component"),
                Metric(name="layout.byte_view_mismatches", value=float(byte_mismatch),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better",
                       note="nbytes not an integer multiple of numel, or a misaligned offset"),
                Metric(name="layout.total_bytes_consistent",
                       value=float(total_bytes == byte_sum), kind="numerical", exact=True,
                       threshold=0.5, direction="greater_is_better",
                       note=f"declared {total_bytes} vs summed {byte_sum}"),
            ],
            mandatory=True,
            falsified_by="overlapping, unaddressed, or byte-inconsistent state",
        ))

    # -- 2. units, frames and clocks declared and resolvable --------------
    clock_graph = getattr(compiled, "clock_graph", None)
    frame_graph = getattr(compiled, "frame_graph", None)
    if not entries or clock_graph is None or frame_graph is None:
        subs.append(could_not_run(
            "units_frames_clocks", "Units, frames and clocks declared and resolvable.",
            "the compiled model exposes no state entries, frame graph or clock graph",
            falsified_by="an undeclared unit, frame or clock (refusal R01)"))
    else:
        missing_units = sum(1 for e in entries if not str(getattr(e, "units", "")).strip())
        missing_clock = sum(1 for e in entries if not str(getattr(e, "clock", "")).strip())
        unknown_clock = sum(1 for e in entries
                            if getattr(e, "clock", None) and not clock_graph.has(e.clock))
        unverified = tuple(getattr(clock_graph, "unverified", lambda: ())())
        orphans = tuple(getattr(clock_graph, "orphans", lambda: ())())
        # every frame the schema uses must be reachable from the graph root
        bad_frames: list[str] = []
        used = sorted(getattr(frame_graph, "used_frames", ()) or ())
        root = getattr(frame_graph, "root", None)
        for f in used:
            if not frame_graph.has(f):
                bad_frames.append(f)
            elif root is not None and not frame_graph.path_is_valid(root, f):
                bad_frames.append(f)
        artifacts["frames"] = {"root": root, "used": used, "unreachable": bad_frames}
        artifacts["clocks"] = {"ids": list(getattr(clock_graph, "ids", lambda: ())()),
                               "master": getattr(clock_graph, "master", None),
                               "unverified": list(unverified), "orphans": list(orphans)}
        subs.append(SubCheck(
            name="units_frames_clocks",
            description="Every state block declares units and a known clock; every used "
                        "frame has a valid path from the graph root (refusal R01).",
            metrics=[
                Metric(name="ports.missing_units", value=float(missing_units),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better"),
                Metric(name="ports.missing_clock", value=float(missing_clock),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better"),
                Metric(name="clocks.unknown_referenced", value=float(unknown_clock),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better"),
                Metric(name="clocks.unverified_or_orphaned",
                       value=float(len(unverified) + len(orphans)), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better",
                       note=f"unverified={unverified}, orphans={orphans}"),
                Metric(name="frames.unreachable_from_root", value=float(len(bad_frames)),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better",
                       note=f"root={root}; unreachable={bad_frames}"),
                Metric(name="state.n_blocks", value=float(len(entries)),
                       kind="diagnostic", exact=True),
            ],
            mandatory=True,
            falsified_by="an undeclared unit, an unknown clock, or an unreachable frame",
        ))

    # -- 3. delays finite, non-negative, representable --------------------
    dispatch = getattr(compiled, "dispatch", None)
    schedule = getattr(compiled, "schedule", None)
    ops = list(dispatch or [])
    if not ops:
        subs.append(could_not_run(
            "delays", "Delay validity against the multirate schedule.",
            "the compiled model dispatches no operators",
            falsified_by="a negative, non-finite or sub-step delay"))
    else:
        base_dt = float(getattr(schedule, "base_dt", 0.0) or 0.0)
        hyper = float(getattr(schedule, "hyperperiod", float("inf")) or float("inf"))
        neg = nonfinite = sub_step = beyond_hyper = 0
        delays: list[float] = []
        for o in ops:
            try:
                d = float(o.delay_seconds())
            except Exception:
                nonfinite += 1
                continue
            delays.append(d)
            if not math.isfinite(d):
                nonfinite += 1
            elif d < 0:
                neg += 1
            else:
                if base_dt > 0 and 0 < d < base_dt:
                    sub_step += 1
                if d > hyper:
                    beyond_hyper += 1
        artifacts["delays"] = {"base_dt": base_dt, "hyperperiod": hyper,
                               "max": max(delays) if delays else None,
                               "n_operators": len(ops)}
        subs.append(SubCheck(
            name="delays",
            description="Delays are finite, non-negative, at least one base step, and "
                        "buffered within the schedule hyperperiod.",
            metrics=[
                Metric(name="delays.negative", value=float(neg), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better"),
                Metric(name="delays.nonfinite", value=float(nonfinite), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better"),
                Metric(name="delays.below_base_dt", value=float(sub_step), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better",
                       note=f"base dt = {base_dt}s; a delay the scheduler cannot represent "
                            "is silently rounded and the dynamics are not the declared ones"),
                Metric(name="delays.beyond_hyperperiod", value=float(beyond_hyper),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better", note=f"hyperperiod = {hyper}s"),
            ],
            mandatory=True,
            falsified_by="a negative, non-finite, sub-step or unbufferable delay",
        ))

    # -- 4. masks agree with the dispatched operator set ------------------
    adjacency = getattr(compiled, "adjacency", None)
    if adjacency is None or not getattr(adjacency, "masks", None):
        subs.append(could_not_run(
            "masks", "Block-sparse masks per evidence class.",
            "the compiled model exposes no .adjacency masks",
            falsified_by="a mask inconsistent with the dispatched operator set"))
    else:
        try:
            classes = sorted(adjacency.masks)
            dense = {c: _dense_mask(adjacency, c) for c in classes}
            shapes = {tuple(v.shape) for v in dense.values()}
            n_regions = len(getattr(adjacency, "region_ids", ()) or ())
            shape_ok = len(shapes) == 1 and (not n_regions or
                                             shapes == {(n_regions, n_regions)})
            nonbinary = sum(1 for v in dense.values()
                            if v.dtype != bool and not np.all(np.isin(v, (0, 1))))
            # every dispatched operator must be present in its class's mask,
            # and no mask may assert an edge nobody dispatches
            unmasked = 0
            dispatched: dict[str, set[tuple[int, int]]] = {c: set() for c in classes}
            for o in ops:
                cls = getattr(o, "evidence_class", None)
                if cls not in dense:
                    unmasked += 1
                    continue
                i, j = adjacency.index_of(o.src), adjacency.index_of(o.dst)
                dispatched[cls].add((i, j))
                if not bool(dense[cls][i, j]):
                    unmasked += 1
            phantom = sum(int(np.count_nonzero(dense[c])) - len(dispatched[c])
                          for c in classes)
            artifacts["masks"] = {
                "classes": classes,
                "density": {c: float(dense[c].mean()) for c in classes},
                "n_regions": n_regions,
            }
            subs.append(SubCheck(
                name="masks",
                description="Evidence-class masks share the region shape, are binary, and "
                            "describe exactly the operators the dispatcher will run.",
                metrics=[
                    Metric(name="masks.consistent_shape", value=float(shape_ok),
                           kind="numerical", exact=True, threshold=0.5,
                           direction="greater_is_better", note=f"shapes: {shapes}"),
                    Metric(name="masks.nonbinary_blocks", value=float(nonbinary),
                           kind="numerical", exact=True, threshold=0.5,
                           direction="less_is_better",
                           note="a soft mask is a parameter, not a mask; declare it as one"),
                    Metric(name="masks.dispatched_edges_not_masked", value=float(unmasked),
                           kind="numerical", exact=True, threshold=0.5,
                           direction="less_is_better",
                           note="an operator the mask does not permit"),
                    Metric(name="masks.masked_edges_not_dispatched", value=float(phantom),
                           kind="numerical", exact=True, threshold=0.5,
                           direction="less_is_better",
                           note="a permitted edge that no operator implements"),
                ],
                mandatory=True,
                falsified_by="masks that disagree with the dispatched operator set",
            ))
        except Exception as exc:
            subs.append(could_not_run(
                "masks", "Block-sparse masks per evidence class.",
                f"could not inspect .adjacency: {type(exc).__name__}: {exc}",
                falsified_by="a mask inconsistent with the dispatched operator set"))

    # -- 5. gradient permissions name modules that exist ------------------
    masks_obj = getattr(compiled, "gradient_masks", None)
    if masks_obj is None:
        subs.append(could_not_run(
            "gradient_permissions", "Per-source gradient masks (rule 2).",
            "the compiled model exposes no .gradient_masks",
            falsified_by="a source permitted to update a module that does not exist"))
    else:
        try:
            keys = tuple(masks_obj.keys())
            declared = tuple(getattr(s, "id", getattr(s, "identity", None))
                             for s in (getattr(schema, "sources", ()) or ()))
            unmatched = 0
            for k in keys:
                unmatched += len(tuple(getattr(masks_obj[k], "unmatched_patterns", ()) or ()))
            unreachable = tuple(getattr(masks_obj, "unreachable_groups", lambda: ())())
            missing_sources = max(len(declared) - len(keys), 0) if declared else 0
            artifacts["gradient_masks"] = {
                "sources": list(keys), "unmatched_patterns": unmatched,
                "unreachable_groups": list(unreachable),
            }
            subs.append(SubCheck(
                name="gradient_permissions",
                description="Every source card compiles to a mask, and no permission names a "
                            "parameter group that does not exist (a silent no-op).",
                metrics=[
                    Metric(name="gradient.sources_without_a_mask",
                           value=float(missing_sources), kind="numerical", exact=True,
                           threshold=0.5, direction="less_is_better"),
                    Metric(name="gradient.unmatched_permission_patterns",
                           value=float(unmatched), kind="numerical", exact=True,
                           threshold=0.5, direction="less_is_better",
                           note="a source naming a module that does not exist updates "
                                "nothing while appearing to be authorised"),
                    Metric(name="gradient.unreachable_parameter_groups",
                           value=float(len(unreachable)), kind="diagnostic", exact=True,
                           note="groups no source may update; not an error, but they will "
                                "never train and must not be described as learned"),
                ],
                mandatory=True,
                falsified_by="a gradient permission that matches nothing",
            ))
        except Exception as exc:
            subs.append(could_not_run(
                "gradient_permissions", "Per-source gradient masks.",
                f"could not inspect .gradient_masks: {type(exc).__name__}: {exc}",
                falsified_by="a source permitted to update a module that does not exist"))

    # -- 6. every bias term is backed (refusal R08) -----------------------
    led = getattr(compiled, "ledger", None)
    if led is None:
        subs.append(could_not_run(
            "uncertainty_ledger", "Bias status backing on every compiled object (R08).",
            "the compiled model exposes no .ledger",
            falsified_by="a bias point estimate with no estimator and no external bound"))
    else:
        unbacked = tuple(getattr(led, "unbacked_bias", ()) or ())
        sens = tuple(getattr(led, "sensitivity_terms", lambda: ())())
        counts = dict(getattr(led, "bias_status_counts", {}) or {})
        artifacts["ledger"] = {"n_objects": len(led), "status_counts": counts,
                               "unbacked": list(unbacked),
                               "n_sensitivity_terms": len(sens)}
        subs.append(SubCheck(
            name="uncertainty_ledger",
            description="Every bias term is design-estimable, externally bounded, or "
                        "declared prior-specified sensitivity (refusal R08).",
            metrics=[
                Metric(name="ledger.unbacked_bias_terms", value=float(len(unbacked)),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better", note=f"{list(unbacked)[:5]}"),
                Metric(name="ledger.prior_specified_sensitivity_terms",
                       value=float(len(sens)), kind="diagnostic", exact=True,
                       note="swept over a declared range; never advertised as empirically "
                            "estimated"),
            ],
            mandatory=True,
            falsified_by="a bias point estimate with no estimator and no external bound",
        ))

    # -- 7. the artifact carries the claim it was compiled for ------------
    prov = getattr(compiled, "provenance", None)
    if prov is None:
        subs.append(could_not_run(
            "claim_class_integrity", "Recorded claim class versus the requested one.",
            "the compiled model exposes no .provenance",
            falsified_by="a silently demoted or overridden claim class"))
    else:
        demoted = bool(getattr(prov, "claim_was_demoted", False))
        overridden = tuple(getattr(prov, "overridden_codes", ()) or ())
        artifacts["provenance"] = {
            "requested": getattr(prov, "requested_claim_class", None),
            "effective": getattr(prov, "effective_claim_class", None),
            "overridden_codes": list(overridden),
            "checks_passed": list(getattr(prov, "checks_passed", ()) or ()),
            "warnings": list(getattr(prov, "warnings", ()) or ()),
        }
        subs.append(SubCheck(
            name="claim_class_integrity",
            description="No refusal was overridden, so the artifact carries the claim class "
                        "it was compiled for.",
            metrics=[
                Metric(name="claim.was_demoted", value=float(demoted), kind="numerical",
                       exact=True, threshold=0.5, direction="less_is_better",
                       note=f"requested={getattr(prov, 'requested_claim_class', None)!r}, "
                            f"effective={getattr(prov, 'effective_claim_class', None)!r}"),
                Metric(name="claim.overridden_refusals", value=float(len(overridden)),
                       kind="numerical", exact=True, threshold=0.5,
                       direction="less_is_better", note=f"codes: {list(overridden)}"),
                Metric(name="claim.refusal_checks_passed",
                       value=float(len(getattr(prov, "checks_passed", ()) or ())),
                       kind="diagnostic", exact=True),
            ],
            mandatory=True,
            falsified_by="an override fired, so the artifact's claim class is weaker than "
                         "the one the run intends to report",
        ))

    notes = [
        f"Subject of this check: {subject}.",
        "A PASS here means the compiler emits an internally consistent artifact for this "
        "subject. It is not evidence about any other schema, and it is not evidence that "
        "any compiled operator is neurally realized.",
    ]
    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts, kind="numerics",
                       notes=notes).finalize()



# ==========================================================================
# solver tests
# ==========================================================================
def convergence_order(errors: Sequence[float], dts: Sequence[float]) -> float:
    """Observed order of accuracy: slope of log(error) versus log(dt)."""
    e = np.asarray(list(errors), dtype=float)
    h = np.asarray(list(dts), dtype=float)
    good = np.isfinite(e) & (e > 0) & (h > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[good]), np.log(e[good]), 1)[0])


def check_solver_convergence(
    solve: Callable[[float], np.ndarray] | None,
    *,
    dts: Sequence[float],
    reference: np.ndarray | None = None,
    expected_order: float = 1.0,
    tol: float = 0.5,
) -> SubCheck:
    """Refine the step and require the error to fall at the advertised rate."""
    if solve is None:
        return could_not_run("solver_convergence", "Step-refinement convergence study.",
                             "no solver supplied (agent E dynamics / agent G field solvers)",
                             falsified_by="observed order below the advertised order")
    dts = sorted(dts, reverse=True)
    try:
        sols = [np.asarray(solve(dt), dtype=float) for dt in dts]
        ref = np.asarray(reference, dtype=float) if reference is not None else \
            np.asarray(solve(dts[-1] / 2.0), dtype=float)
        errs = [float(np.linalg.norm(s.ravel() - ref.ravel()) /
                      (np.linalg.norm(ref.ravel()) + 1e-30)) for s in sols]
    except Exception as exc:
        return could_not_run("solver_convergence", "Step-refinement convergence study.",
                             f"solver raised {type(exc).__name__}: {exc}",
                             falsified_by="observed order below the advertised order")
    p = convergence_order(errs, dts)
    return SubCheck(
        name="solver_convergence",
        description="Observed order of accuracy under step refinement.",
        metrics=[
            Metric(name="solver.observed_order", value=p, kind="numerical", exact=True,
                   threshold=expected_order - tol, direction="greater_is_better",
                   note=f"errors {['%.3g' % e for e in errs]} at dts {dts}"),
            Metric(name="solver.finest_relative_error", value=float(errs[-1]),
                   kind="numerical", exact=True, direction="less_is_better"),
        ],
        mandatory=True,
        falsified_by="observed order materially below the advertised order",
    )


def check_stability(trajectory: np.ndarray | None, *, bound: float | None = None,
                    name: str = "solver_stability") -> SubCheck:
    """Long-horizon boundedness: no NaN, no blow-up, no silent clipping."""
    if trajectory is None:
        return could_not_run(name, "Long-horizon stability.", "no trajectory supplied",
                             falsified_by="non-finite or unbounded state")
    x = np.asarray(trajectory, dtype=float)
    n_nan = int(np.sum(~np.isfinite(x)))
    amax = float(np.nanmax(np.abs(x))) if x.size else float("nan")
    growth = float("nan")
    if x.ndim >= 1 and x.shape[0] >= 4:
        head = np.nanmean(np.abs(x[: x.shape[0] // 4]))
        tail = np.nanmean(np.abs(x[-x.shape[0] // 4:]))
        growth = float(tail / (head + 1e-30))
    metrics = [
        Metric(name="stability.nonfinite_entries", value=float(n_nan), kind="numerical",
               exact=True, threshold=0.5, direction="less_is_better"),
        Metric(name="stability.tail_over_head_amplitude", value=growth, kind="numerical",
               exact=True, threshold=10.0, direction="less_is_better"),
    ]
    if bound is not None:
        metrics.append(
            Metric(name="stability.max_abs_state", value=amax, kind="numerical", exact=True,
                   threshold=float(bound), direction="less_is_better")
        )
    return SubCheck(name=name, description="Boundedness over a long horizon.",
                    metrics=metrics, mandatory=True,
                    falsified_by="non-finite or unbounded state")


def check_conservation(trajectory: np.ndarray | None,
                       invariant: Callable[[np.ndarray], float] | None,
                       *, tol: float = 1e-3, name: str = "conservation") -> SubCheck:
    """Relative drift of a declared invariant along the trajectory."""
    if trajectory is None or invariant is None:
        return could_not_run(name, "Conservation of a declared invariant.",
                             "no trajectory or no invariant function supplied; a module that "
                             "declares no invariant cannot claim a conservation property",
                             falsified_by="the invariant drifts beyond tolerance")
    x = np.asarray(trajectory, dtype=float)
    vals = np.array([float(invariant(x[t])) for t in range(x.shape[0])])
    v0 = vals[0] if abs(vals[0]) > 1e-30 else 1.0
    drift = np.abs(vals - vals[0]) / abs(v0)
    pt, iv = bootstrap_ci(drift, statistic=np.max, seed=0, n_boot=200)
    return SubCheck(
        name=name,
        description="Maximum relative drift of the declared invariant.",
        metrics=[
            Metric(name="conservation.max_relative_drift", value=float(np.max(drift)),
                   kind="numerical", interval=iv, threshold=float(tol),
                   direction="less_is_better")
        ],
        mandatory=True,
        falsified_by="the invariant drifts beyond the declared tolerance",
    )


def check_seed_reproducibility(fn: Callable[[int], Any] | None, *, seed: int = 0,
                               name: str = "seed_reproducibility") -> SubCheck:
    """Determinism is a test, not an aspiration (ARCHITECTURE.md §3)."""
    if fn is None:
        return could_not_run(name, "Bitwise reproducibility for a fixed seed.",
                             "no stochastic entry point supplied",
                             falsified_by="two runs with the same seed differ")
    try:
        a = np.asarray(fn(seed))
        b = np.asarray(fn(seed))
        c = np.asarray(fn(seed + 1))
    except Exception as exc:
        return could_not_run(name, "Bitwise reproducibility for a fixed seed.",
                             f"entry point raised {type(exc).__name__}: {exc}",
                             falsified_by="two runs with the same seed differ")
    identical = bool(a.shape == b.shape and np.array_equal(a, b))
    differs = bool(a.shape != c.shape or not np.array_equal(a, c))
    return SubCheck(
        name=name,
        description="Same seed -> identical output; different seed -> different output.",
        metrics=[
            Metric(name="determinism.same_seed_identical", value=float(identical),
                   kind="numerical", exact=True, threshold=0.5, direction="greater_is_better"),
            Metric(name="determinism.different_seed_differs", value=float(differs),
                   kind="numerical", exact=True, threshold=0.5, direction="greater_is_better",
                   note="a 'stochastic' function that ignores its seed is also a bug"),
        ],
        mandatory=True,
        falsified_by="two runs with the same seed differ, or the seed is ignored",
    )


# ==========================================================================
# boundary consistency and the adaptive-resolution permit
# ==========================================================================
def boundary_consistency(fine_observable: np.ndarray | None,
                         coarse_observable: np.ndarray | None,
                         *, tol: float = 0.05, seed: int = 0,
                         name: str = "boundary_consistency") -> SubCheck:
    """Fine and coarse backends must agree on the boundary observable."""
    if fine_observable is None or coarse_observable is None:
        return could_not_run(
            name, "Fine/coarse agreement on the declared boundary observable.",
            "the fine and/or coarse boundary observable was not supplied by the backends "
            "(agent E dynamics / agent D restriction maps); adaptive resolution may not be "
            "used for inference until both produce it (§11.1)",
            falsified_by="disagreement beyond the declared tolerance",
        )
    f = np.asarray(fine_observable, dtype=float).ravel()
    c = np.asarray(coarse_observable, dtype=float).ravel()
    n = min(f.size, c.size)
    scale = float(np.mean(np.abs(c[:n]))) + 1e-30
    rel = np.abs(f[:n] - c[:n]) / scale
    pt, iv = bootstrap_ci(rel, seed=seed, n_boot=500)
    return SubCheck(
        name=name,
        description="Relative disagreement between fine and coarse boundary observables.",
        metrics=[
            Metric(name="boundary.mean_relative_disagreement", value=pt, kind="numerical",
                   interval=iv, threshold=float(tol), direction="less_is_better",
                   require_interval_beats_threshold=True),
            Metric(name="boundary.max_relative_disagreement", value=float(np.max(rel)),
                   kind="numerical", exact=True, direction="less_is_better"),
        ],
        mandatory=True,
        falsified_by="disagreement beyond the declared tolerance",
    )


@dataclass(frozen=True)
class AdaptiveResolutionPermit:
    """Machine-readable permission to use adaptive resolution for inference."""

    granted: bool
    tolerance: float
    observed: float | None
    reason: str
    report_id: str = "N2_boundary_consistency"

    def require(self) -> None:
        if not self.granted:
            raise PermissionError(
                f"adaptive resolution is not permitted: {self.reason} "
                "(body.tex §11.1: fine and coarse backends must agree within a declared "
                "tolerance on boundary observables BEFORE adaptive resolution is used for "
                "inference)"
            )


def permit_adaptive_resolution(fine_observable: np.ndarray | None = None,
                               coarse_observable: np.ndarray | None = None,
                               *, tol: float = 0.05, seed: int = 0
                               ) -> tuple[AdaptiveResolutionPermit, ClaimReport]:
    """Run the boundary test and issue (or refuse) the permit."""
    sub = boundary_consistency(fine_observable, coarse_observable, tol=tol, seed=seed)
    man = _manifest(
        "N2_boundary_consistency",
        "Fine and coarse regional backends agree within the declared tolerance on boundary "
        "observables, so adaptive resolution may be used for inference.",
        "disagreement beyond the declared tolerance, or a backend that cannot produce the "
        "boundary observable at all",
        "Adaptive resolution is refused for inference; run single-resolution backends or "
        "narrow the tolerance claim (§11.1).",
        seed=seed,
        thresholds={"boundary_rel_tol": tol},
    )
    rep = ClaimReport(
        manifest=man, subchecks=[sub], kind="numerics",
        artifacts={"subject": (
            "fine observable n="
            f"{np.asarray(fine_observable).size if fine_observable is not None else 0}"
            ", coarse observable n="
            f"{np.asarray(coarse_observable).size if coarse_observable is not None else 0}"
            f", declared tolerance {tol}")},
    ).finalize()
    observed = next((m.value for m in sub.metrics
                     if m.name == "boundary.mean_relative_disagreement"), None)
    granted = rep.status == "PASS"
    reason = ("boundary observables agree within tolerance" if granted
              else "; ".join(rep.blocking_reasons) or "boundary test did not pass")
    return AdaptiveResolutionPermit(granted=granted, tolerance=tol, observed=observed,
                                    reason=reason), rep


# ==========================================================================
# independent physical-solver validation
# ==========================================================================
def analytic_dipole_potential(points: np.ndarray, dipole_pos: np.ndarray,
                              dipole_moment: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Quasi-static potential of a current dipole in an infinite homogeneous medium.

    ``phi(r) = (p . (r - r0)) / (4 pi sigma |r - r0|^3)``, units V for ``p`` in
    A*m and ``sigma`` in S/m.  No neural response model is involved: this is
    the reference an EM solver must reproduce before it is allowed anywhere
    near a lead field.
    """
    r = np.asarray(points, dtype=float) - np.asarray(dipole_pos, dtype=float)
    p = np.asarray(dipole_moment, dtype=float)
    d = np.linalg.norm(r, axis=-1)
    d = np.where(d < 1e-9, np.nan, d)
    return (r @ p) / (4.0 * math.pi * float(sigma) * d**3)


def analytic_free_field_pressure(points: np.ndarray, source_pos: np.ndarray,
                                 *, amplitude: float = 1.0, k: float = 100.0) -> np.ndarray:
    """Free-field pressure of a monopole: ``p(r) = A exp(i k r) / r`` (complex)."""
    r = np.linalg.norm(np.asarray(points, dtype=float) -
                       np.asarray(source_pos, dtype=float), axis=-1)
    r = np.where(r < 1e-9, np.nan, r)
    return amplitude * np.exp(1j * k * r) / r


def helmholtz_residual(field: np.ndarray, *, dx: float, k: float) -> float:
    """Relative residual of ``lap(p) + k^2 p = 0`` on a uniform 3-D grid.

    **Refinement warning (this bit is load-bearing).**  Evaluated with the
    scheme's *own* second-difference Laplacian, the spatial truncation error
    cancels: a discrete steady state satisfies the discrete Helmholtz equation
    to round-off.  What remains is *temporal* dispersion, which for a leapfrog
    march is

    ``|k^2 - kappa^2| / k^2 = (omega*dt)^2 / 12 + O(dt^4)``,
    ``kappa = (2/(c*dt)) * sin(omega*dt/2)``.

    So this residual falls when **dt** is refined, and sits flat when ``h`` is
    refined at fixed ``dt``.  A convergence study that refines only ``h`` will
    show no improvement and can be misread as "the residual does not vanish
    under refinement" — a false falsification.  Refine ``dt`` with ``h`` at
    fixed CFL.  (Caught by agent Faraday on the first N4 sweep; the criterion
    in this module's N4 manifest was reworded because of it.)
    """
    p = np.asarray(field)
    lap = np.zeros_like(p)
    for ax in range(p.ndim):
        lap = lap + (np.roll(p, 1, axis=ax) - 2.0 * p + np.roll(p, -1, axis=ax)) / dx**2
    core = tuple(slice(1, -1) for _ in range(p.ndim))
    res = lap[core] + (k**2) * p[core]
    return float(np.linalg.norm(res) / (np.linalg.norm((k**2) * p[core]) + 1e-30))


def _maybe_call(obj: Any) -> Any:
    """Read a value that may be an attribute or a zero-arg method."""
    if obj is None:
        return None
    if callable(obj):
        try:
            return obj()
        except Exception:
            return None
    return obj


def _subject_of(fn: Any) -> str:
    """Which callable produced these numbers (recorded on every report)."""
    return (f"{getattr(fn, '__module__', '?')}."
            f"{getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))}")


def _validate_against_analytic(numeric: np.ndarray, analytic: np.ndarray, *, tol: float,
                               label: str, seed: int = 0) -> SubCheck:
    a = np.asarray(numeric).ravel()
    b = np.asarray(analytic).ravel()
    n = min(a.size, b.size)
    good = np.isfinite(a[:n]) & np.isfinite(b[:n])
    if good.sum() == 0:
        return could_not_run(label, "Comparison against the analytic reference.",
                             "no finite overlapping samples between solver and reference",
                             falsified_by="relative error above tolerance")
    rel = np.abs(a[:n][good] - b[:n][good]) / (np.abs(b[:n][good]).mean() + 1e-30)
    pt, iv = bootstrap_ci(rel, seed=seed, n_boot=500)
    return SubCheck(
        name=label,
        description="Relative error against a closed-form solution, with no neural model in "
                    "the loop.",
        metrics=[
            Metric(name=f"{label}.mean_relative_error", value=pt, kind="numerical",
                   interval=iv, threshold=float(tol), direction="less_is_better",
                   require_interval_beats_threshold=True),
            Metric(name=f"{label}.max_relative_error",
                   value=float(np.max(rel)), kind="numerical", exact=True,
                   direction="less_is_better"),
        ],
        mandatory=True,
        falsified_by="relative error above tolerance against the analytic solution",
    )


def validate_em_solver(solver: Callable[..., np.ndarray] | None = None, *,
                       points: np.ndarray | None = None,
                       dipole_pos: Sequence[float] = (0.0, 0.0, 0.0),
                       dipole_moment: Sequence[float] = (0.0, 0.0, 1e-8),
                       sigma: float = 0.33, tol: float = 0.05, seed: int = 0) -> ClaimReport:
    """Validate an electromagnetic solver *independently of neural-response models*."""
    man = _manifest(
        "N3_em_solver",
        "The quasi-static CONDUCTION solver reproduces the closed-form potential of a "
        "current dipole in an unbounded homogeneous conductor, validated independently of "
        "any neural-response model. This is the EEG/lead-field forward problem; it is NOT "
        "the magnetically induced TMS field, which has a different source term and boundary "
        "condition and needs its own gate (N6).",
        "relative error above tolerance against the analytic dipole solution",
        "The conduction solver may not be used for lead fields or source modelling; every "
        "downstream conduction-dependent claim is suspended.",
        seed=seed,
        thresholds={"relative_tol": tol, "sigma_S_per_m": sigma},
    )
    if solver is None:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "em_solver", "EM solver versus the analytic dipole potential.",
                "no EM solver supplied (agent G scwbd.intervene / agent F lead fields); the "
                "field model is unvalidated and no E-field claim may be made",
                falsified_by=man.falsified_by)],
            kind="numerics",
        ).finalize()
    if points is None:
        rng = np.random.default_rng(seed)
        pts = rng.normal(0, 0.08, size=(512, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.02]
        points = pts
    ref = analytic_dipole_potential(points, np.asarray(dipole_pos, dtype=float),
                                    np.asarray(dipole_moment, dtype=float), sigma=sigma)
    try:
        num = np.asarray(solver(points=points, dipole_pos=dipole_pos,
                                dipole_moment=dipole_moment, sigma=sigma), dtype=float)
    except TypeError:
        try:
            num = np.asarray(solver(points), dtype=float)
        except Exception as exc:
            return ClaimReport(
                manifest=man,
                subchecks=[could_not_run("em_solver", "EM solver versus analytic dipole.",
                                         f"solver raised {type(exc).__name__}: {exc}",
                                         falsified_by=man.falsified_by)],
                kind="numerics").finalize()
    except Exception as exc:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run("em_solver", "EM solver versus analytic dipole.",
                                     f"solver raised {type(exc).__name__}: {exc}",
                                     falsified_by=man.falsified_by)],
            kind="numerics").finalize()
    sub = _validate_against_analytic(num, ref, tol=tol, label="em_solver", seed=seed)
    return ClaimReport(
        manifest=man, subchecks=[sub], kind="numerics",
        artifacts={"subject": _subject_of(solver),
                   "reference": "current dipole in an unbounded homogeneous conductor "
                                "(conduction / volume-current problem)",
                   "does_not_cover": "magnetically induced E-field of a TMS coil "
                                     "(induction); see gate N6"},
        notes=[
            "SCOPE: conduction, not induction. A PASS licenses the quasi-static conduction "
            "discretisation used for EEG lead fields. It does NOT license the magnetically "
            "induced TMS field: different source term, different boundary condition, "
            "separate gate (N6_induced_efield).",
            "A verification gate is destroyed if the reference leaks into the solver. Check "
            "that the boundary data is homogeneous, not the analytic value, before reading "
            "this PASS as evidence.",
            "Field accuracy, target engagement, network effect and clinical utility remain "
            "separate quantities (thesis §0.5).",
        ],
    ).finalize()


def validate_acoustic_solver(solver: Callable[..., np.ndarray] | None = None, *,
                             points: np.ndarray | None = None,
                             source_pos: Sequence[float] = (0.0, 0.0, 0.0),
                             k: float = 100.0, tol: float = 0.05,
                             grid: np.ndarray | None = None, dx: float | None = None,
                             seed: int = 0) -> ClaimReport:
    """Validate an acoustic solver against free-field spreading and the Helmholtz residual."""
    man = _manifest(
        "N4_acoustic_solver",
        "The acoustic solver reproduces free-field spreading and satisfies the Helmholtz "
        "equation, validated independently of any neural-response model.",
        "relative error above tolerance, or a Helmholtz residual that does not fall as the "
        "TIME step is refined at fixed CFL (see the refinement note: refining h alone leaves "
        "the residual flat for reasons unrelated to solver quality, so 'flat under h "
        "refinement' is NOT a falsification of this gate)",
        "The acoustic solver may not be used for tFUS exposure or targeting; every downstream "
        "acoustic claim is suspended.",
        seed=seed,
        thresholds={"relative_tol": tol, "wavenumber_per_m": k},
    )
    if solver is None:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "acoustic_solver", "Acoustic solver versus free-field reference.",
                "no acoustic solver supplied (agent G scwbd.intervene); the exposure model is "
                "unvalidated and no tFUS claim may be made",
                falsified_by=man.falsified_by)],
            kind="numerics").finalize()
    if points is None:
        rng = np.random.default_rng(seed)
        pts = rng.normal(0, 0.05, size=(512, 3))
        points = pts[np.linalg.norm(pts, axis=1) > 0.01]
    ref = np.abs(analytic_free_field_pressure(points, np.asarray(source_pos, dtype=float), k=k))
    try:
        num = np.abs(np.asarray(solver(points=points, source_pos=source_pos, k=k)))
    except TypeError:
        num = np.abs(np.asarray(solver(points)))
    except Exception as exc:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run("acoustic_solver", "Acoustic solver versus free field.",
                                     f"solver raised {type(exc).__name__}: {exc}",
                                     falsified_by=man.falsified_by)],
            kind="numerics").finalize()
    subs = [_validate_against_analytic(num, ref, tol=tol, label="acoustic_solver", seed=seed)]
    if grid is not None and dx is not None:
        res = helmholtz_residual(np.asarray(grid), dx=float(dx), k=float(k))
        subs.append(
            SubCheck(
                name="helmholtz_residual",
                description="Relative residual of lap(p) + k^2 p on the solver's own grid.",
                metrics=[Metric(name="acoustic.helmholtz_relative_residual", value=res,
                                kind="numerical", exact=True, threshold=float(tol),
                                direction="less_is_better")],
                mandatory=True,
                falsified_by="the discrete field does not satisfy its own governing equation",
            )
        )
    else:
        subs.append(
            SubCheck(name="helmholtz_residual",
                     description="Relative residual of the Helmholtz operator.",
                     metrics=[], mandatory=False, forced_status="COULD_NOT_RUN",
                     reason="no solver grid/dx supplied; only the free-field comparison was run")
        )
    return ClaimReport(
        manifest=man, subchecks=subs, kind="numerics",
        artifacts={"subject": _subject_of(solver)},
        notes=[
            "REFINEMENT RULE: the Helmholtz residual here is set by TEMPORAL dispersion, "
            "not by h. Measured with the scheme's own Laplacian the spatial error cancels, "
            "leaving (omega*dt)^2/12. Refining h at fixed dt leaves the residual flat, which "
            "reads like a failure and is not one. Refine dt with h at fixed CFL.",
            "Amplitude calibration is part of what is under test when the source strength is "
            "fixed a priori rather than fitted to the reference; a residual amplitude bias "
            "must be reported, not divided out.",
        ],
    ).finalize()


def validate_induced_efield_solver(
    solver: Callable[..., np.ndarray] | None = None,
    *,
    analytic: Callable[..., np.ndarray] | None = None,
    points: np.ndarray | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    tol: float = 0.05,
    convergence: Sequence[Mapping[str, float]] | None = None,
    expected_order: float = 1.5,
    convergence_ratio: float | None = None,
    reference_degree: int | None = None,
    max_convergence_ratio: float = 0.90,
    max_bound_over_error: float = 0.10,
    geometry: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> ClaimReport:
    """Validate the **magnetically induced** E-field solver (N6).

    N3 validates *conduction*: a current dipole in an unbounded homogeneous
    conductor, which is the EEG/lead-field forward problem.  A TMS coil's
    induced field is a different problem — different source term (``-dA/dt``
    plus the secondary charge field), different boundary condition — and a
    conduction PASS licenses nothing about it.  This gate exists so that gap is
    visible on the scoreboard rather than implicit in a caveat.

    The analytic reference (Sarvas / Heller--van Hulsteyn closed form for a
    spherically symmetric conductor) must be **supplied**, not assumed: agent J
    does not implement induction physics.  When the reference and the solver
    come from the same module the report says so, because a solver checked
    against its own module's closed form is a weaker test than one checked
    against an independent implementation.
    """
    man = _manifest(
        "N6_induced_efield",
        "The magnetically induced E-field solver reproduces the closed-form "
        "(Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, "
        "validated independently of any neural-response model.",
        "relative error above tolerance against the closed form, or a mesh-refinement study "
        "that does not converge at the advertised order",
        "The induced-field solver may not be used for TMS E-field prediction, target "
        "engagement or pose ranking; every downstream induction-dependent claim is "
        "suspended (N3 does not cover this: it validates conduction, not induction).",
        seed=seed,
        thresholds={"relative_tol": tol, "expected_order": expected_order},
    )
    missing: list[str] = []
    if solver is None:
        missing.append("induced-field solver (agent Faraday: scwbd.intervene.tms.efield)")
    if analytic is None:
        missing.append(
            "closed-form reference (Sarvas / Heller-van Hulsteyn); agent J does not "
            "implement induction physics and will not substitute the conduction reference "
            "from N3, which is a different problem"
        )
    if missing:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "induced_efield", "Induced E-field versus the closed form.",
                "missing: " + "; ".join(missing),
                falsified_by=man.falsified_by)],
            kind="numerics",
            notes=["Opened because N3 passed for CONDUCTION only. Any claim that depends on "
                   "the induced TMS field remains suspended until this gate runs."],
        ).finalize()

    kw = dict(solver_kwargs or {})
    if points is None:
        rng = np.random.default_rng(seed)
        pts = rng.normal(0, 0.05, size=(512, 3))
        points = pts[np.linalg.norm(pts, axis=1) > 0.02]
    try:
        num = np.asarray(solver(points=points, **kw), dtype=float)
        ref = np.asarray(analytic(points=points, **kw), dtype=float)
    except TypeError:
        try:
            num = np.asarray(solver(points), dtype=float)
            ref = np.asarray(analytic(points), dtype=float)
        except Exception as exc:
            return ClaimReport(
                manifest=man,
                subchecks=[could_not_run(
                    "induced_efield", "Induced E-field versus the closed form.",
                    f"solver or reference raised {type(exc).__name__}: {exc}",
                    falsified_by=man.falsified_by)],
                kind="numerics").finalize()
    except Exception as exc:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "induced_efield", "Induced E-field versus the closed form.",
                f"solver or reference raised {type(exc).__name__}: {exc}",
                falsified_by=man.falsified_by)],
            kind="numerics").finalize()

    subs = [_validate_against_analytic(num, ref, tol=tol, label="induced_efield", seed=seed)]
    artifacts: dict[str, Any] = {"subject": _subject_of(solver),
                                 "reference": _subject_of(analytic)}

    shared = getattr(solver, "__module__", "?") == getattr(analytic, "__module__", "?")
    subs.append(SubCheck(
        name="reference_provenance",
        description="Does the closed-form reference come from the same module as the solver?",
        metrics=[Metric(
            name="induced_efield.reference_shares_module_with_solver",
            value=float(shared), kind="audit", exact=True,
            threshold=0.5, direction="less_is_better",
            note=(f"solver={getattr(solver, '__module__', '?')}, "
                  f"reference={getattr(analytic, '__module__', '?')}; shared provenance is "
                  "not disqualifying, but it is a weaker test than an independent "
                  "implementation and must not be described as independent validation"),
        )],
        mandatory=False,
    ))

    # -- the reference's own validity domain, self-declaring ---------------
    measured = next((m.value for m in subs[0].metrics
                     if m.name.endswith("mean_relative_error")), float("nan"))
    if convergence_ratio is None:
        convergence_ratio = _maybe_call(getattr(analytic, "convergence_ratio", None))
    if convergence_ratio is None or reference_degree is None:
        subs.append(could_not_run(
            "reference_validity_domain",
            "Is the reference more accurate than the solver, at THIS geometry?",
            "the reference's convergence ratio and/or expansion degree were not declared. A "
            "reference whose own accuracy at the geometry under test is unknown is not a "
            "reference, and the gate cannot say whether it measured the solver or the "
            "reference. Pass convergence_ratio= and reference_degree=.",
            falsified_by="the reference is no more accurate than the solver it measures",
        ))
    else:
        ratio = float(convergence_ratio)
        bound = ratio ** float(reference_degree)
        rel = bound / measured if measured and math.isfinite(measured) and measured > 0 \
            else float("inf")
        if math.isfinite(measured) and measured <= bound:
            # At or below the reference's own error bound the comparison is at
            # the reference's noise floor: it cannot separate solver error from
            # reference error. That is "cannot conclude", not "fail".
            subs.append(could_not_run(
                "reference_validity_domain",
                "Is the reference more accurate than the solver, at THIS geometry?",
                f"the measured solver error ({measured:.3g}) is at or below the reference's "
                f"own a-priori bound ({bound:.3g} = {ratio:.4g}**{reference_degree}); this "
                "comparison is at the reference's noise floor and cannot separate solver "
                "error from reference error. Refine the reference or state a weaker claim.",
                falsified_by="the reference is no more accurate than the solver it measures",
            ))
            artifacts["reference_validity_domain"] = {
                "convergence_ratio": ratio, "reference_degree": reference_degree,
                "a_priori_bound": bound, "measured_solver_error": measured,
                "at_reference_noise_floor": True,
            }
        else:
          subs.append(SubCheck(
            name="reference_validity_domain",
            description=(
                "The multipole/series reference converges like ratio**degree. Its a-priori "
                "error bound at this geometry must sit well below the solver error being "
                "measured, or the gate is measuring the reference."
            ),
            metrics=[
                Metric(name="reference.convergence_ratio", value=ratio,
                       kind="numerical", exact=True, threshold=max_convergence_ratio,
                       direction="less_is_better",
                       note=("a/R_c at the validated geometry; the series converges like "
                             "ratio**degree, so a ratio approaching 1 is not fixable by "
                             "raising the degree")),
                Metric(name="reference.a_priori_bound", value=float(bound),
                       kind="numerical", exact=True,
                       note=f"ratio**degree with degree={reference_degree}"),
                Metric(name="reference.bound_over_measured_error", value=float(rel),
                       kind="numerical", exact=True, threshold=max_bound_over_error,
                       direction="less_is_better",
                       note=(f"a-priori reference bound {bound:.3g} / measured solver error "
                             f"{measured:.3g}; the reference must be the more accurate of "
                             "the two by a clear margin")),
            ],
            mandatory=True,
            falsified_by="the reference is no more accurate than the solver it measures",
        ))
        artifacts["reference_validity_domain"] = {
            "convergence_ratio": ratio, "reference_degree": reference_degree,
            "a_priori_bound": bound, "measured_solver_error": measured,
            "bound_over_measured_error": rel,
            "does_not_cover": (
                "contact geometry. A coil element in contact with the scalp has a/R_c ~ 0.955, "
                "where no feasible degree brings the series bound below the solver error. "
                "This gate validates a STANDOFF equivalent dipole; the contact regime is "
                "gate N8_induced_efield_contact and has not run."
            ),
        }
    if geometry:
        artifacts["geometry"] = dict(geometry)

    if convergence:
        errs = [float(r["error"]) for r in convergence]
        sizes = [float(r.get("h", r.get("n_elements", i + 1)))
                 for i, r in enumerate(convergence)]
        p = convergence_order(errs, sizes)
        subs.append(SubCheck(
            name="mesh_convergence",
            description="Observed order of the induced-field discretisation.",
            metrics=[Metric(
                name="induced_efield.observed_order", value=p, kind="numerical", exact=True,
                threshold=expected_order, direction="greater_is_better",
                note=f"errors {['%.3g' % e for e in errs]}")],
            mandatory=True,
            falsified_by="refinement does not converge at the advertised order",
        ))
    return ClaimReport(
        manifest=man, subchecks=subs, artifacts=artifacts, kind="numerics",
        notes=[
            "Induction, not conduction: this gate is what N3 does NOT cover.",
            "STANDOFF ONLY. The reference series converges like (a/R_c)**degree. At a "
            "contact geometry (a/R_c ~ 0.955 for a coil element 4 mm off an 85 mm scalp) no "
            "feasible degree brings its bound below the solver error, so this gate validates "
            "the discretisation against a STANDOFF equivalent dipole, not against a contact "
            "coil. tms-robotics positions a coil in contact; that regime is gate "
            "N8_induced_efield_contact and it has not run.",
            "The validity domain is a metric in this report, not a footnote: a reader who "
            "checks only the headline error still sees reference.convergence_ratio.",
        ],
    ).finalize()


def validate_induced_efield_contact(
    solver: Callable[..., np.ndarray] | None = None,
    *,
    reference: Callable[..., np.ndarray] | None = None,
    self_convergence: Sequence[Mapping[str, float]] | None = None,
    points: np.ndarray | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    convergence_ratio: float | None = None,
    min_contact_ratio: float = 0.95,
    tol: float | None = None,
    expected_order: float = 1.5,
    seed: int = 0,
) -> ClaimReport:
    """N8 — the induced field in the **contact** regime, which N6 does not cover.

    N6 validates the induced-field discretisation against a spectral reference
    whose series converges like ``(a/R_c)**degree``.  At the standoff geometry
    N6 uses (``a/R_c = 0.7727``) that reference is orders of magnitude more
    accurate than the solver, which is what makes it a reference.  A coil in
    **contact** has ``a/R_c ~ 0.955``, where no feasible degree brings the
    bound below the solver error.  N6 therefore validates a standoff equivalent
    dipole, not a contact coil.

    This gap is load-bearing rather than academic: ``tms-robotics`` positions a
    coil against a registered scalp target, i.e. contact geometry, and
    near-surface accuracy is precisely what the BEM exists for.  So it gets its
    own row rather than a caveat inside N6's.

    **The contract this gate requires** (report shape, for whoever builds it):

    * ``solver`` and ``points`` at a contact geometry, with
      ``convergence_ratio >= min_contact_ratio`` — a gate handed standoff
      geometry is not this gate and will refuse;
    * **either** an independent contact-regime ``reference`` (a boundary-integral
      reference with graded panels, say) **or** ``self_convergence``: a
      Richardson study of the solver against itself under refinement, which
      does not need an external reference but proves only self-consistency and
      is reported as such;
    * a declared ``tol``.  The tolerance is preregistered here, not chosen after
      seeing the error.

    If the contact regime turns out not to be validatable to a defensible
    tolerance, this gate FAILs, and that is a result: the runtime must then
    surface targeting in the contact regime as unvalidated rather than
    returning a confident number.
    """
    man = _manifest(
        "N8_induced_efield_contact",
        "The induced-field solver is validated in the CONTACT regime (a coil element at "
        "clinical standoff from the scalp, a/R_c >= 0.95) to a preregistered tolerance — "
        "the geometry the downstream targeting consumer actually uses.",
        "no reference or self-convergence study achieves a defensible tolerance at contact "
        "geometry, or the solver's error there exceeds the preregistered tolerance",
        "Targeting in the contact regime is UNVALIDATED. scwbd.runtime must surface that as "
        "Unresolved/Defer rather than returning a confident E-field or engagement number, "
        "and no pose ranking may be reported as validated at contact geometry.",
        seed=seed,
        thresholds={"min_contact_ratio": min_contact_ratio, "relative_tol": tol,
                    "expected_order": expected_order},
    )
    missing: list[str] = []
    if solver is None:
        missing.append("induced-field solver at contact geometry (agent Faraday)")
    if reference is None and not self_convergence:
        missing.append(
            "either an independent contact-regime reference (e.g. boundary-integral with "
            "graded panels) or a Richardson self-convergence study of the solver under "
            "refinement; the N6 spectral reference does NOT extend here, since its series "
            "bound at a/R_c ~ 0.955 exceeds the solver error it would be measuring"
        )
    if convergence_ratio is None:
        missing.append(
            "the geometry ratio a/R_c, so the gate can confirm it was handed CONTACT "
            "geometry rather than a standoff case relabelled"
        )
    if tol is None:
        missing.append("a preregistered tolerance (chosen before seeing the error)")
    if missing:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "contact_regime", "Induced field validated where the coil actually sits.",
                "missing: " + "; ".join(missing),
                falsified_by=man.falsified_by)],
            kind="numerics",
            artifacts={"subject": "not yet run — contract stated in the docstring",
                       "why_this_row_exists": (
                           "N6's reference is accurate only for standoff geometry "
                           "(a/R_c = 0.7727). The consumer uses contact geometry "
                           "(a/R_c ~ 0.955). Folding this into N6 would let a standoff "
                           "PASS be read as covering contact.")},
            notes=["Opened at agent J's request after agent Faraday disclosed the validity "
                   "domain of the N6 reference. Visible and unrun beats implicit."],
        ).finalize()

    ratio = float(convergence_ratio)
    subs: list[SubCheck] = [SubCheck(
        name="is_contact_geometry",
        description="Confirm the gate was handed contact geometry, not standoff relabelled.",
        metrics=[Metric(name="contact.a_over_Rc", value=ratio, kind="numerical", exact=True,
                        threshold=min_contact_ratio, direction="greater_is_better",
                        note="a standoff geometry belongs in N6, not here")],
        mandatory=True,
        falsified_by="the geometry is not in the contact regime this gate exists for",
    )]
    kw = dict(solver_kwargs or {})
    if points is None:
        missing_pts = True
    else:
        missing_pts = False
    if reference is not None and not missing_pts:
        try:
            num = np.asarray(solver(points=points, **kw), dtype=float)
            ref = np.asarray(reference(points=points, **kw), dtype=float)
            subs.append(_validate_against_analytic(num, ref, tol=float(tol),
                                                   label="contact_efield", seed=seed))
        except Exception as exc:
            subs.append(could_not_run(
                "contact_efield", "Contact-regime error against an independent reference.",
                f"solver or reference raised {type(exc).__name__}: {exc}",
                falsified_by=man.falsified_by))
    elif reference is None:
        subs.append(SubCheck(
            name="contact_efield",
            description="No independent contact reference supplied; self-convergence only.",
            metrics=[], mandatory=False, forced_status="COULD_NOT_RUN",
            reason=("self-convergence proves the discretisation converges to SOMETHING, not "
                    "that it converges to the right answer; an independent contact reference "
                    "is still owed"),
        ))
    if self_convergence:
        errs = [float(r["error"]) for r in self_convergence]
        hs = [float(r.get("h", r.get("n_elements", i + 1)))
              for i, r in enumerate(self_convergence)]
        subs.append(SubCheck(
            name="self_convergence",
            description="Richardson refinement of the solver against itself at contact.",
            metrics=[Metric(name="contact.self_convergence_order",
                            value=convergence_order(errs, hs), kind="numerical", exact=True,
                            threshold=expected_order, direction="greater_is_better",
                            note="self-consistency only: converging is not converging to the "
                                 "right answer")],
            mandatory=True,
            falsified_by="the solver does not converge under refinement at contact geometry",
        ))
    return ClaimReport(
        manifest=man, subchecks=subs, kind="numerics",
        artifacts={"subject": _subject_of(solver), "a_over_Rc": ratio},
        notes=["Contact geometry is what tms-robotics uses. A PASS here still licenses no "
               "claim about target engagement, network effect or clinical utility."],
    ).finalize()


# ==========================================================================
def run_numerics_suite(
    *,
    compiled: Any = None,
    schema: Any = None,
    solver: Callable[[float], np.ndarray] | None = None,
    solver_dts: Sequence[float] = (0.02, 0.01, 0.005, 0.0025),
    expected_order: float = 1.0,
    trajectory: np.ndarray | None = None,
    invariant: Callable[[np.ndarray], float] | None = None,
    stochastic_entry_point: Callable[[int], Any] | None = None,
    fine_observable: np.ndarray | None = None,
    coarse_observable: np.ndarray | None = None,
    boundary_tol: float = 0.05,
    em_solver: Callable[..., np.ndarray] | None = None,
    acoustic_solver: Callable[..., np.ndarray] | None = None,
    acoustic_grid: np.ndarray | None = None,
    acoustic_dx: float | None = None,
    induced_efield_solver: Callable[..., np.ndarray] | None = None,
    induced_efield_analytic: Callable[..., np.ndarray] | None = None,
    seed: int = 0,
) -> list[ClaimReport]:
    """Run §11.1 end to end; every absent input yields a loud COULD_NOT_RUN."""
    reports: list[ClaimReport] = [check_compiler_correctness(compiled, schema=schema, seed=seed)]

    man = _manifest(
        "N5_solver_suite",
        "Solvers converge at their advertised order, remain stable, conserve their declared "
        "invariants, and are bitwise reproducible for a fixed seed.",
        "an observed order below the advertised order, non-finite or unbounded state, "
        "invariant drift beyond tolerance, or non-determinism at a fixed seed",
        "The solver may not be used for claim-bearing inference until it converges; results "
        "produced with it are withdrawn.",
        seed=seed,
        thresholds={"expected_order": expected_order, "dts": list(solver_dts)},
    )
    reports.append(
        ClaimReport(
            manifest=man,
            subchecks=[
                check_solver_convergence(solver, dts=solver_dts, expected_order=expected_order),
                check_stability(trajectory),
                check_conservation(trajectory, invariant),
                check_seed_reproducibility(stochastic_entry_point, seed=seed),
            ],
            artifacts={"subject": _subject_of(solver) if solver is not None
                       else "no solver supplied"},
            kind="numerics",
        ).finalize()
    )

    _, permit_report = permit_adaptive_resolution(fine_observable, coarse_observable,
                                                  tol=boundary_tol, seed=seed)
    reports.append(permit_report)
    if em_solver is None or acoustic_solver is None:
        # agent Faraday's reference-problem solvers, once they are importable
        dep = adapters.field_solvers()
        if dep.available:
            em_fn, ac_run = dep.obj
            em_solver = em_solver or em_fn
            if acoustic_solver is None:
                def acoustic_solver(points, source_pos=(0.0, 0.0, 0.0), k=100.0, **kw):
                    return ac_run(points, source_pos, k, **kw).pressure
                if acoustic_grid is None:
                    probe = ac_run(np.full((1, 3), 0.02), (0.0, 0.0, 0.0), 100.0)
                    acoustic_grid, acoustic_dx = probe.grid_block, probe.spacing_m
    reports.append(validate_em_solver(em_solver, seed=seed))
    reports.append(validate_acoustic_solver(acoustic_solver, grid=acoustic_grid,
                                            dx=acoustic_dx, seed=seed))
    reports.append(validate_induced_efield_solver(induced_efield_solver,
                                                  analytic=induced_efield_analytic,
                                                  seed=seed))
    reports.append(validate_induced_efield_contact(seed=seed))
    return reports
