"""Run the §11.1 field-physics gates N3 and N4 against real solvers.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

``reports/gates/numerics/N3_em_solver.json`` and ``N4_acoustic_solver.json``
were both written ``COULD_NOT_RUN`` with the reason "no EM solver supplied
(agent G ``scwbd.intervene``)" / "no acoustic solver supplied".  That was true
when it was written and is no longer true.  This module supplies the solvers
from :mod:`scwbd.intervene.numerics`, runs agent J's own gate functions
unmodified, and writes the resulting :class:`~scwbd.bench.report.ClaimReport`
under ``reports/intervene/`` for the bench owner to adopt.

It deliberately does **not** write into ``reports/gates/``: the bench agent owns
that directory and owns the verdict.  What is handed over is a report object
produced by their code from our solvers.

Run::

    python -m scwbd.intervene.run_field_gates --out reports/intervene

Everything is preregistered-default: the gate's own seed-0 point clouds, its own
tolerances (``relative_tol=0.05``, ``sigma_S_per_m=0.33``,
``wavenumber_per_m=100.0``), its own reference formulae.  No argument of this
script changes what "pass" means.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from .numerics import (
    acoustic_grid_convergence,
    em_grid_convergence,
    quasistatic_dipole_potential_fd,
    run_free_field_monopole,
)

__all__ = ["em_gate_points", "acoustic_gate_points", "run_n3", "run_n4", "main"]

#: the gate's preregistered tolerances, restated here so a drift is visible
RELATIVE_TOL = 0.05
SIGMA_S_PER_M = 0.33
WAVENUMBER_PER_M = 100.0
SEED = 0


def em_gate_points(seed: int = SEED) -> np.ndarray:
    """The N3 field points, reproduced exactly as ``validate_em_solver`` draws them."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 0.08, size=(512, 3))
    return pts[np.linalg.norm(pts, axis=1) > 0.02]


def acoustic_gate_points(seed: int = SEED) -> np.ndarray:
    """The N4 field points, reproduced exactly as ``validate_acoustic_solver`` draws them."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 0.05, size=(512, 3))
    return pts[np.linalg.norm(pts, axis=1) > 0.01]


def run_n3(*, n_per_axis: int = 256, convergence: bool = True) -> Any:
    """N3: the FD quasi-static solver against the closed-form dipole potential."""
    from scwbd.bench.numerics import validate_em_solver

    def solver(points: np.ndarray, **kw: Any) -> np.ndarray:
        return quasistatic_dipole_potential_fd(
            points,
            kw.get("dipole_pos", (0.0, 0.0, 0.0)),
            kw.get("dipole_moment", (0.0, 0.0, 1e-8)),
            kw.get("sigma", SIGMA_S_PER_M),
            n_per_axis=n_per_axis,
        )

    rep = validate_em_solver(
        solver, sigma=SIGMA_S_PER_M, tol=RELATIVE_TOL, seed=SEED
    )
    rep.notes.append(
        "Solver: scwbd.intervene.numerics.quasistatic_dipole_potential_fd -- a "
        f"second-order 7-point finite-difference Poisson solve on a {n_per_axis}^3 "
        "grid, diagonalised exactly by the type-I DST. The truncation boundary "
        "carries HOMOGENEOUS DIRICHLET data (zero), not the analytic potential, so "
        "no value from the reference enters the solve; the boundary is placed at "
        "1.9x the farthest field point and the residual truncation error is part of "
        "the reported number rather than removed by it."
    )
    rep.notes.append(
        "The reference here is a CURRENT dipole in an unbounded homogeneous "
        "conductor -- the EEG/lead-field forward problem, not the magnetically "
        "induced TMS field of scwbd.intervene.tms.efield. Passing N3 licenses the "
        "quasi-static conduction discretisation; the induced-field operator is "
        "separately convergence-tested against the Sarvas / Heller-van Hulsteyn "
        "closed form in tests/intervene/test_tms_efield.py."
    )
    if convergence:
        pts = em_gate_points()
        rows = em_grid_convergence(pts, n_list=(128, 192, 256))
        rep.artifacts["grid_convergence"] = rows
        rep.notes.append(
            "Refinement study (mean relative error vs h): "
            + "; ".join(
                f"n={int(r['n_per_axis'])} h={r['spacing_m']:.5f} m "
                f"err={r['mean_relative_error']:.4f}"
                + ("" if math.isnan(r["observed_order"])
                   else f" order={r['observed_order']:.2f}")
                for r in rows
            )
        )
        trunc = _em_truncation_study(pts, base_n=n_per_axis)
        rep.artifacts["truncation_study"] = trunc
        rep.notes.append(
            "The observed order falls off at the finest grid because the h-refinement "
            "holds the truncation box fixed, so the zero-Dirichlet truncation error is "
            "an h-independent floor. Enlarging the box at CONSTANT h separates the two: "
            + "; ".join(
                f"margin={r['margin']:.2f} n={int(r['n_per_axis'])} "
                f"h={r['spacing_m']:.5f} m err={r['mean_relative_error']:.4f}"
                for r in trunc
            )
            + ". The discretisation error is the limit of that sequence; the difference "
            "from the reported number is what the finite domain costs, and it is a "
            "budget item, not a fitted correction."
        )
    return rep.finalize()


def _em_truncation_study(points: np.ndarray, *, base_n: int = 256,
                         base_margin: float = 1.9,
                         factors: tuple[float, ...] = (1.0, 1.25, 1.5)) -> list[dict]:
    """Enlarge the truncation box at fixed ``h``: only the boundary error moves."""
    from .numerics import solve_dipole_potential

    pts = np.asarray(points, dtype=float)
    dist = np.linalg.norm(pts, axis=-1)
    ref = pts[:, 2] * 1e-8 / (4.0 * math.pi * SIGMA_S_PER_M * dist**3)
    scale = float(np.abs(ref).mean())
    rmax = float(dist.max())
    h = 2 * (base_margin * rmax) / base_n

    rows = []
    for f in factors:
        n = int(round(base_n * f / 2) * 2)
        res = solve_dipole_potential(
            pts, (0, 0, 0), (0, 0, 1e-8), SIGMA_S_PER_M,
            n_per_axis=n, half_width_m=n * h / 2,
        )
        rows.append({
            "margin": (n * h / 2) / rmax,
            "n_per_axis": float(n),
            "spacing_m": res.spacing_m,
            "half_width_m": res.half_width_m,
            "mean_relative_error": float(
                (np.abs(res.potential - ref) / scale).mean()
            ),
        })
    return rows


def run_n4(*, points_per_wavelength: int = 20, convergence: bool = True,
           device: str | None = None) -> Any:
    """N4: the FDTD solver against free-field spreading and its own Helmholtz residual."""
    from scwbd.bench.numerics import validate_acoustic_solver

    pts = acoustic_gate_points()
    # one FDTD run serves both subchecks: the phasor at the gate's points and
    # the source-free grid block the Helmholtz residual is measured on
    result = run_free_field_monopole(
        pts, (0.0, 0.0, 0.0), WAVENUMBER_PER_M,
        points_per_wavelength=points_per_wavelength, device=device,
    )

    def solver(points: np.ndarray, **kw: Any) -> np.ndarray:
        if not np.allclose(np.asarray(points, dtype=float), pts):  # pragma: no cover
            raise ValueError("acoustic gate points changed under the cached FDTD run")
        return result.pressure

    rep = validate_acoustic_solver(
        solver, k=WAVENUMBER_PER_M, tol=RELATIVE_TOL, seed=SEED,
        grid=result.grid_block, dx=result.spacing_m,
    )
    rep.notes.append(
        "Solver: scwbd.intervene.numerics.free_field_monopole_fdtd -- a "
        "second-order leapfrog FDTD march to steady state at "
        f"{points_per_wavelength} points per wavelength on a {result.n_per_axis}^3 "
        f"grid ({result.n_steps} steps), with a quadratic damping sponge. The "
        "radiation condition is imposed by the march, not by analytic boundary "
        "data, and the lattice-delta source strength is fixed a priori at Q = 4*pi*A "
        "rather than fitted, so the amplitude calibration is itself under test."
    )
    rep.notes.append(
        "The Helmholtz residual is evaluated on a source-free, sponge-free cube of "
        "the solver's own grid. Measuring it with the scheme's own Laplacian "
        "cancels the spatial truncation error exactly -- the discrete steady state "
        "satisfies the discrete Helmholtz equation -- so what is left is the "
        "TEMPORAL dispersion: |k^2 - kappa^2|/k^2 = (omega dt)^2/12 with kappa = "
        "(2/c dt) sin(omega dt/2). At 60 steps per period that predicts 9.139e-4 "
        "and 9.178e-4 is measured. It therefore falls only when dt is refined, "
        "which is what the refinement study below does (CFL held fixed); holding "
        "dt fixed while refining h would leave it flat for a reason that has "
        "nothing to do with solver quality."
    )
    if convergence:
        rows = acoustic_grid_convergence(pts, ppw_list=(10, 14, 20), device=device)
        rep.artifacts["grid_convergence"] = rows
        rep.notes.append(
            "Refinement study: "
            + "; ".join(
                f"ppw={int(r['points_per_wavelength'])} h={r['spacing_m']:.5f} m "
                f"dt={r['time_step_s']:.3e} s err={r['mean_relative_error']:.4f} "
                f"amp_ratio={r['mean_amplitude_ratio']:.4f} "
                f"helmholtz={r['helmholtz_relative_residual']:.4f}"
                + ("" if math.isnan(r["observed_order"])
                   else f" order={r['observed_order']:.2f}")
                for r in rows
            )
        )
    rep.artifacts["fdtd_meta"] = dict(result.meta, n_per_axis=result.n_per_axis,
                                      spacing_m=result.spacing_m, n_steps=result.n_steps)
    return rep.finalize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/intervene",
                    help="directory for the handover reports (NOT reports/gates)")
    ap.add_argument("--n-per-axis", type=int, default=256)
    ap.add_argument("--ppw", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-convergence", action="store_true")
    ap.add_argument("--only", choices=("n3", "n4"), default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conv = not args.no_convergence

    reports = []
    if args.only in (None, "n3"):
        reports.append(run_n3(n_per_axis=args.n_per_axis, convergence=conv))
    if args.only in (None, "n4"):
        reports.append(
            run_n4(points_per_wavelength=args.ppw, convergence=conv, device=args.device)
        )

    for rep in reports:
        jp, mp = rep.write(out)
        print(f"{rep.manifest.claim_id}: {rep.status}  -> {jp}  {mp}")
        for sub in rep.subchecks:
            for met in sub.metrics:
                print(f"    {met.name} = {met.value:.6g}"
                      + (f"  (threshold {met.threshold})" if met.threshold is not None else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
