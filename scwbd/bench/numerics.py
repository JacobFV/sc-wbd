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
def check_compiler_correctness(compiled: Any = None, *, schema: Any = None,
                               seed: int = 0) -> ClaimReport:
    """Shape, unit, coordinate, delay and mask correctness of a CompiledModel.

    Duck-typed against ``ARCHITECTURE.md`` §2: ``.state_layout``,
    ``.adjacency``, ``.dispatch``, ``.schedule``, ``.gradient_masks``,
    ``.frame_graph``, ``.clock_graph``, ``.ledger``.
    """
    man = _manifest(
        "N1_compiler_correctness",
        "The compiler produces a model whose shapes, units, frames, delays and masks are "
        "internally consistent.",
        "any offset overlap, undeclared unit/frame/clock, negative or unrepresentable delay, "
        "or a mask that does not match the declared operator set",
        "Fix the compiler before any claim-bearing run; a numerically inconsistent compilation "
        "invalidates every downstream gate.",
        seed=seed,
    )
    if compiled is None:
        return ClaimReport(
            manifest=man,
            subchecks=[
                could_not_run(
                    "compiled_model", "A CompiledModel to inspect.",
                    "no CompiledModel supplied (agent A's scwbd.compiler.compile has not been "
                    "run or has not landed); compiler correctness is unverified",
                    falsified_by=man.falsified_by,
                )
            ],
            kind="numerics",
        ).finalize()

    subs: list[SubCheck] = []

    # -- state layout: complete, non-overlapping, shape-consistent -------
    layout = getattr(compiled, "state_layout", None)
    if layout is None:
        subs.append(could_not_run("state_layout", "Packed state offsets per region/component.",
                                  "CompiledModel exposes no .state_layout",
                                  falsified_by="overlapping or incomplete state offsets"))
    else:
        spans: list[tuple[int, int, str]] = []
        try:
            items = layout.items() if hasattr(layout, "items") else list(layout)
            for key, v in (items if hasattr(layout, "items") else enumerate(items)):
                off = getattr(v, "offset", None)
                size = getattr(v, "size", None)
                if off is None and isinstance(v, (tuple, list)) and len(v) >= 2:
                    off, size = int(v[0]), int(v[1])
                if off is None or size is None:
                    continue
                spans.append((int(off), int(off) + int(size), str(key)))
        except Exception as exc:
            spans = []
        if not spans:
            subs.append(could_not_run("state_layout", "Packed state offsets.",
                                      "could not read (offset, size) pairs from .state_layout",
                                      falsified_by="overlapping or incomplete state offsets"))
        else:
            spans.sort()
            overlaps = sum(1 for a, b in zip(spans, spans[1:]) if a[1] > b[0])
            gaps = sum(1 for a, b in zip(spans, spans[1:]) if a[1] < b[0])
            subs.append(
                SubCheck(
                    name="state_layout",
                    description="Region/component offsets are disjoint and contiguous.",
                    metrics=[
                        Metric(name="layout.overlaps", value=float(overlaps), kind="numerical",
                               exact=True, threshold=0.5, direction="less_is_better"),
                        Metric(name="layout.gaps", value=float(gaps), kind="numerical",
                               exact=True, threshold=0.5, direction="less_is_better",
                               note="a gap means state is allocated but unaddressed"),
                    ],
                    mandatory=True,
                    falsified_by="overlapping or unaddressed state",
                )
            )

    # -- units / frames / clocks declared everywhere --------------------
    def _walk_ports(obj: Any) -> list[Any]:
        out: list[Any] = []
        regions = getattr(obj, "regions", None) or getattr(schema, "regions", None) or []
        for r in regions:
            out += list(getattr(r, "ports", []) or [])
        return out

    ports = _walk_ports(compiled) or _walk_ports(schema)
    if not ports:
        subs.append(could_not_run("units_frames_clocks",
                                  "Units, frames and clocks declared on every port.",
                                  "no ports reachable from the compiled model or schema",
                                  falsified_by="an undeclared unit, frame or clock"))
    else:
        bad_units = bad_frames = bad_clocks = 0
        for p in ports:
            sup = getattr(p, "support", None)
            tmp = getattr(p, "temporal", None)
            if sup is None or not getattr(sup, "units", None):
                bad_units += 1
            if sup is None or not getattr(sup, "frame", None):
                bad_frames += 1
            if tmp is None or not getattr(tmp, "clock", None):
                bad_clocks += 1
        subs.append(
            SubCheck(
                name="units_frames_clocks",
                description="Every port declares units, a frame and a clock (refusal R01).",
                metrics=[
                    Metric(name="ports.missing_units", value=float(bad_units), kind="numerical",
                           exact=True, threshold=0.5, direction="less_is_better"),
                    Metric(name="ports.missing_frame", value=float(bad_frames), kind="numerical",
                           exact=True, threshold=0.5, direction="less_is_better"),
                    Metric(name="ports.missing_clock", value=float(bad_clocks), kind="numerical",
                           exact=True, threshold=0.5, direction="less_is_better"),
                    Metric(name="ports.total", value=float(len(ports)), kind="diagnostic",
                           exact=True),
                ],
                mandatory=True,
                falsified_by="an undeclared unit, frame or clock (R01)",
            )
        )

    # -- delays representable on the schedule ---------------------------
    ops = list(getattr(compiled, "dispatch", []) or getattr(schema, "operators", []) or [])
    sched = getattr(compiled, "schedule", None)
    dt_min = None
    for attr in ("dt_min", "base_dt", "dt"):
        v = getattr(sched, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            dt_min = float(v)
            break
    if not ops:
        subs.append(could_not_run("delays", "Delay validity against the multirate schedule.",
                                  "no operators reachable from the compiled model or schema",
                                  falsified_by="a negative or unrepresentable delay"))
    else:
        neg = nonfinite = unrepresentable = 0
        for o in ops:
            d = getattr(o, "delay", None)
            if d is None:
                pr = getattr(o, "delay_prior", None)
                d = getattr(pr, "mean", None) if pr is not None else None
            if d is None:
                continue
            try:
                dv = float(d)
            except Exception:
                continue
            if not math.isfinite(dv):
                nonfinite += 1
            elif dv < 0:
                neg += 1
            elif dt_min is not None and 0 < dv < dt_min:
                unrepresentable += 1
        subs.append(
            SubCheck(
                name="delays",
                description="Delays are finite, non-negative and representable on the schedule.",
                metrics=[
                    Metric(name="delays.negative", value=float(neg), kind="numerical",
                           exact=True, threshold=0.5, direction="less_is_better"),
                    Metric(name="delays.nonfinite", value=float(nonfinite), kind="numerical",
                           exact=True, threshold=0.5, direction="less_is_better"),
                    Metric(name="delays.below_base_dt", value=float(unrepresentable),
                           kind="numerical", exact=True, threshold=0.5,
                           direction="less_is_better",
                           note=f"base dt = {dt_min}" if dt_min else "no base dt exposed"),
                ],
                mandatory=True,
                falsified_by="a negative, non-finite or sub-step delay",
            )
        )

    # -- masks -----------------------------------------------------------
    adj = getattr(compiled, "adjacency", None)
    if adj is None:
        subs.append(could_not_run("masks", "Block-sparse masks per evidence class.",
                                  "CompiledModel exposes no .adjacency",
                                  falsified_by="a mask inconsistent with the operator set"))
    else:
        try:
            blocks = adj.items() if hasattr(adj, "items") else {"all": adj}.items()
            shapes, densities, nonbinary = [], [], 0
            for name, M in blocks:
                A = np.asarray(M)
                shapes.append(tuple(A.shape))
                densities.append(float((A != 0).mean()))
                if not np.all(np.isin(A, (0, 1))) and A.dtype != bool:
                    nonbinary += 1
            same_shape = len(set(shapes)) <= 1
            subs.append(
                SubCheck(
                    name="masks",
                    description="Evidence-class masks share a shape and are binary.",
                    metrics=[
                        Metric(name="masks.consistent_shape", value=float(same_shape),
                               kind="numerical", exact=True, threshold=0.5,
                               direction="greater_is_better", note=f"shapes: {set(shapes)}"),
                        Metric(name="masks.nonbinary_blocks", value=float(nonbinary),
                               kind="numerical", exact=True, threshold=0.5,
                               direction="less_is_better",
                               note="a soft mask is a parameter, not a mask; declare it as one"),
                        Metric(name="masks.density", value=float(np.mean(densities))
                               if densities else float("nan"), kind="diagnostic", exact=True),
                    ],
                    mandatory=True,
                    falsified_by="masks that disagree in shape or are silently continuous",
                )
            )
        except Exception as exc:
            subs.append(could_not_run("masks", "Block-sparse masks per evidence class.",
                                      f"could not inspect .adjacency: {type(exc).__name__}: {exc}",
                                      falsified_by="a mask inconsistent with the operator set"))

    return ClaimReport(manifest=man, subchecks=subs, kind="numerics").finalize()


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
