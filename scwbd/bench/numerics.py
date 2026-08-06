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
    rep = ClaimReport(manifest=man, subchecks=[sub], kind="numerics").finalize()
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
    """Relative residual of ``lap(p) + k^2 p = 0`` on a uniform 3-D grid."""
    p = np.asarray(field)
    lap = np.zeros_like(p)
    for ax in range(p.ndim):
        lap = lap + (np.roll(p, 1, axis=ax) - 2.0 * p + np.roll(p, -1, axis=ax)) / dx**2
    core = tuple(slice(1, -1) for _ in range(p.ndim))
    res = lap[core] + (k**2) * p[core]
    return float(np.linalg.norm(res) / (np.linalg.norm((k**2) * p[core]) + 1e-30))


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
        "The electromagnetic solver reproduces a closed-form quasi-static reference, "
        "validated independently of any neural-response model.",
        "relative error above tolerance against the analytic dipole solution",
        "The EM solver may not be used for lead fields, E-field prediction or targeting; "
        "every downstream field-dependent claim is suspended.",
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
    return ClaimReport(manifest=man, subchecks=[sub], kind="numerics",
                       notes=["Field accuracy, target engagement, network effect and clinical "
                              "utility remain separate quantities (thesis §0.5)."]).finalize()


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
        "relative error above tolerance, or a Helmholtz residual that does not vanish under "
        "grid refinement",
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
    return ClaimReport(manifest=man, subchecks=subs, kind="numerics").finalize()


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
            kind="numerics",
        ).finalize()
    )

    _, permit_report = permit_adaptive_resolution(fine_observable, coarse_observable,
                                                  tol=boundary_tol, seed=seed)
    reports.append(permit_report)
    reports.append(validate_em_solver(em_solver, seed=seed))
    reports.append(validate_acoustic_solver(acoustic_solver, seed=seed))
    return reports
