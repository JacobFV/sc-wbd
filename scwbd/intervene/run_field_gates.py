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
from typing import Any, Sequence

import numpy as np

from .numerics import (
    acoustic_grid_convergence,
    em_grid_convergence,
    quasistatic_dipole_potential_fd,
    run_free_field_monopole,
)

__all__ = ["em_gate_points", "acoustic_gate_points", "n6_points",
           "n8_points", "n8_dipole_pos",
           "N9_PINNED_BOUND", "bound_has_moved",
           "run_n3", "run_n4", "run_n6", "run_n8", "run_n9", "main"]

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
        rep.artifacts["truncation_study_note"] = "box enlarged at constant h"
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


# ---------------------------------------------------------------------------
# N6: the magnetically induced field (what N3 does NOT cover)
# ---------------------------------------------------------------------------

#: N6 geometry.  A single equivalent magnetic dipole standing off an 85 mm
#: sphere.  The standoff is not cosmetic: the spectral reference converges like
#: (a/R_c)^L, so 110 mm buys a reference five orders more accurate than the
#: solver it measures.  See the disclosure note in :func:`run_n6`.
N6_SPHERE_RADIUS = 0.085
N6_CORTEX_RADIUS = 0.070
N6_DIPOLE_POS = ((0.0, 0.0, 0.11),)
N6_DIPOLE_MDOT = ((0.0, 1e6, 0.0),)


def n6_points(n: int = 512, seed: int = SEED) -> np.ndarray:
    """Interior field points for N6.

    The gate's default cloud (``N(0, 0.05)``, ``|r| > 0.02``) reaches ~0.2 m and
    would put most points *outside* an 85 mm head, where the interior solution
    is not the field and the solver correctly refuses. These are drawn inside
    the modelled cortical shell instead, which is where an induced-field claim
    would actually be made.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = N6_CORTEX_RADIUS * rng.uniform(0.25, 1.0, size=(n, 1)) ** (1.0 / 3.0)
    return direction * radius


def run_n6(*, convergence: bool = True, subdivisions: Sequence[int] = (1, 2, 3, 4)) -> Any:
    """N6: the charge BEM against an independent spectral Sarvas / H-vH reference."""
    import torch

    from scwbd.bench.numerics import validate_induced_efield_solver

    from .spectral_reference import (
        SphericalInductionReference,
        reference_degree_convergence,
        spectral_induced_efield,
    )
    from .tms.efield import ChargeBEM, TriMesh, charge_bem_induced_efield, icosphere

    pts = n6_points()
    kw = {
        "dipole_pos": N6_DIPOLE_POS,
        "dipole_mdot": N6_DIPOLE_MDOT,
        "sphere_radius": N6_SPHERE_RADIUS,
    }

    rows: list[dict[str, float]] = []
    if convergence:
        ref_field = torch.as_tensor(spectral_induced_efield(pts, **kw))
        t_pts = torch.as_tensor(pts, dtype=torch.float64)
        t_pos = torch.as_tensor(np.asarray(N6_DIPOLE_POS, dtype=float))
        t_mdot = torch.as_tensor(np.asarray(N6_DIPOLE_MDOT, dtype=float))
        for k in subdivisions:
            verts, faces = icosphere(int(k))
            bem = ChargeBEM(
                [TriMesh(verts * N6_SPHERE_RADIUS, faces)], [0.33], [0.0]
            )
            got = bem.total_field(t_pts, t_pos, t_mdot)
            rows.append({
                "subdivision": float(k),
                "n_elements": float(bem.n_faces),
                # nominal panel edge: sqrt(area per panel) on the sphere
                "h": math.sqrt(4 * math.pi * N6_SPHERE_RADIUS**2 / bem.n_faces),
                "error": float((got - ref_field).norm() / ref_field.norm()),
            })

    ref = SphericalInductionReference(radius=N6_SPHERE_RADIUS, degree=48)
    pos_t = torch.as_tensor(np.asarray(N6_DIPOLE_POS, dtype=float))

    rep = validate_induced_efield_solver(
        charge_bem_induced_efield,
        analytic=spectral_induced_efield,
        points=pts,
        solver_kwargs=kw,
        tol=RELATIVE_TOL,
        convergence=rows or None,
        # the reference must declare its own accuracy AT THIS geometry, or the
        # gate cannot tell whether it measured the solver or the reference
        convergence_ratio=ref.convergence_ratio(pos_t),
        reference_degree=ref.degree,
        seed=SEED,
    )
    rep.artifacts["solver_provenance"] = {
        "solver": "scwbd.intervene.tms.efield.ChargeBEM "
                  f"(icosphere subdiv 4, {20 * 4 ** 4} panels, single shell)",
        "reference": "scwbd.intervene.spectral_reference.SphericalInductionReference "
                     "(multipole degree 48, Neumann problem by spherical-harmonic "
                     "transform, gradient by autograd)",
        "geometry": {
            "sphere_radius_m": N6_SPHERE_RADIUS,
            "field_points": f"{len(pts)} inside the {N6_CORTEX_RADIUS} m cortical shell",
            "dipole_pos_m": [list(p) for p in N6_DIPOLE_POS],
            "dipole_mdot_A_m2_per_s": [list(m) for m in N6_DIPOLE_MDOT],
        },
    }
    rep.artifacts["mesh_refinement"] = rows
    rep.artifacts["reference_self_convergence"] = reference_degree_convergence(
        pts, **kw
    )
    rep.artifacts["reference_truncation_estimate"] = ref.truncation_estimate(pos_t)
    rep.artifacts["reference_convergence_ratio"] = ref.convergence_ratio(pos_t)

    rep.notes.append(
        "The reference is INDEPENDENT of the solver, in code and in derivation. "
        "The solver is a surface-charge BEM; the reference solves the interior "
        "Neumann problem spectrally -- multipole expansion in regular solid "
        "harmonics, coefficients by quadrature on the sphere, gradient by "
        "automatic differentiation. They share no function and no formula. "
        "`reference_shares_module_with_solver` is 0 for that reason, and it is "
        "the reason rather than an accident of file layout."
    )
    rep.notes.append(
        "The reference is validated in its own right, not asserted: it converges "
        "geometrically to the closed form (9.7e-9 relative at degree 48), it "
        "reproduces the Heller-van Hulsteyn theorem r.E = 0 to the same order, "
        "and in the far-source limit it reduces to the elementary Faraday "
        "solution E = -(1/2) Bdot x r. Those three checks are in "
        "tests/intervene/test_spectral_reference.py."
    )
    rep.notes.append(
        "DISCLOSURE -- validity domain of the reference. The multipole series "
        f"converges like (a/R_c)^L; here a/R_c = {ref.convergence_ratio(pos_t):.4f}, so "
        f"the a-priori bound at degree 48 is {ref.truncation_estimate(pos_t):.2e} and the "
        "MEASURED agreement with the closed form is 9.7e-9 -- respectively three "
        "and six orders below the ~4.8e-3 solver error being measured, which is "
        "the condition for calling this a reference. A coil element a few "
        "millimetres off the scalp has a/R_c -> 1 and would need a prohibitive "
        "degree: this reference is a yardstick for a standoff source, NOT for a "
        "coil in contact. Near-surface geometry is exactly what the BEM exists "
        "for, and validating it there needs a different reference -- N6 does not "
        "claim to have done so."
    )
    rep.notes.append(
        "Field accuracy is not target engagement, network effect or clinical "
        "utility (thesis Sec. 0.5). A numerical PASS lifts a precondition and "
        "licenses no claim."
    )
    return rep.finalize()


# ---------------------------------------------------------------------------
# N8: the CONTACT regime -- the geometry tms-robotics actually uses
# ---------------------------------------------------------------------------

# Preregistered before the N8 run.  These are the repo's standing numerics
# tolerances, identical to N3/N4/N6 -- not numbers picked to fit a result.
# Full disclosure of the ordering, since the gate asks for preregistration: the
# graded-mesh refinement study below was measured while scoping whether contact
# was reachable at all, BEFORE N8's contract existed. The tolerance was not
# adjusted afterwards; it is the same 0.05 every other numerics gate uses.
N8_TOL = 0.05
N8_EXPECTED_ORDER = 1.5

#: 4 mm standoff on an 85 mm sphere: a/R_c = 0.9551, above the gate's 0.95 floor.
N8_SPHERE_RADIUS = 0.085
N8_STANDOFF_M = 0.004
#: off-axis on purpose, so the reference's rotation path is exercised rather
#: than the one symmetric case where it is trivially correct
N8_DIRECTION = (0.03, -0.04, 0.0715)
N8_DIPOLE_MDOT = ((0.0, 1e6, 0.0),)
#: graded family: near-source panels held at ~1 mm while the GLOBAL mesh refines
N8_GRADED_FAMILY = ((1, 5), (2, 4), (3, 3), (4, 2))


def n8_dipole_pos() -> tuple[tuple[float, float, float], ...]:
    d = np.asarray(N8_DIRECTION, dtype=float)
    d = d / np.linalg.norm(d)
    return (tuple((d * (N8_SPHERE_RADIUS + N8_STANDOFF_M)).tolist()),)


def n8_points(n: int = 256, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * N8_CORTEX_RADIUS


N8_CORTEX_RADIUS = 0.070


def run_n8(*, self_convergence: bool = True, audits: bool = True) -> Any:
    """N8: the induced field where the coil actually sits."""
    import torch

    from scwbd.bench.numerics import validate_induced_efield_contact

    from .spectral_reference import AxialInductionReference, contact_induced_efield
    from .tms.efield import (
        ChargeBEM,
        TriMesh,
        contact_bem_induced_efield,
        graded_icosphere,
        icosphere,
    )

    pts = n8_points()
    pos = n8_dipole_pos()
    kw = {
        "dipole_pos": pos,
        "dipole_mdot": N8_DIPOLE_MDOT,
        "sphere_radius": N8_SPHERE_RADIUS,
    }
    t_pts = torch.as_tensor(pts, dtype=torch.float64)
    t_pos = torch.as_tensor(np.asarray(pos, dtype=float))
    t_mdot = torch.as_tensor(np.asarray(N8_DIPOLE_MDOT, dtype=float))
    ratio = N8_SPHERE_RADIUS / float(np.linalg.norm(np.asarray(pos[0], dtype=float)))
    direction = t_pos[0] / t_pos[0].norm()

    # --- self-convergence: successive differences, no reference involved -----
    rows: list[dict[str, float]] = []
    if self_convergence:
        fields = []
        for base, levels in N8_GRADED_FAMILY:
            mesh = graded_icosphere(
                N8_SPHERE_RADIUS, base, direction, levels, half_angle_rad=0.35
            )
            bem = ChargeBEM([mesh], [0.33], [0.0])
            fields.append((
                math.sqrt(4 * math.pi * N8_SPHERE_RADIUS**2 / (20 * 4**base)),
                bem.total_field(t_pts, t_pos, t_mdot),
                mesh.n_faces,
            ))
        for i in range(len(fields) - 1):
            h_coarse, e_coarse, n_coarse = fields[i]
            _, e_fine, _ = fields[i + 1]
            rows.append({
                "h": h_coarse,
                "n_elements": float(n_coarse),
                "error": float((e_coarse - e_fine).norm() / e_fine.norm()),
            })

    rep = validate_induced_efield_contact(
        contact_bem_induced_efield,
        reference=contact_induced_efield,
        self_convergence=rows or None,
        points=pts,
        solver_kwargs=kw,
        convergence_ratio=ratio,
        tol=N8_TOL,
        expected_order=N8_EXPECTED_ORDER,
        seed=SEED,
    )

    # --- declare the reference's own accuracy AT THIS geometry --------------
    ref = AxialInductionReference(radius=N8_SPHERE_RADIUS, degree=400)
    from .tms.efield import analytic_sphere_efield

    measured = float(
        (ref.induced_field(t_pts, t_pos, t_mdot)
         - analytic_sphere_efield(t_pts, t_pos, t_mdot)).norm()
        / analytic_sphere_efield(t_pts, t_pos, t_mdot).norm()
    )
    solver_err = None
    for sub in rep.subchecks:
        for met in sub.metrics:
            if met.name == "contact_efield.mean_relative_error":
                solver_err = float(met.value)
    bound = ref.truncation_estimate(t_pos)
    rep.artifacts["reference_validity_domain"] = {
        "a_over_Rc": ratio,
        "reference_degree": 400,
        "a_priori_bound_ratio_pow_degree": bound,
        "measured_vs_closed_form": measured,
        "solver_error": solver_err,
        "bound_over_solver_error": (bound / solver_err) if solver_err else None,
        "requirement": "bound/solver_error <= 0.1",
    }
    rep.artifacts["solver_provenance"] = {
        "solver": "scwbd.intervene.tms.efield.contact_bem_induced_efield "
                  "(graded icosphere, base subdiv 4 + 2 levels under the source)",
        "reference": "scwbd.intervene.spectral_reference.AxialInductionReference "
                     "(m = +-1 multipole, degree 400, per-element rotation)",
        "geometry": {
            "sphere_radius_m": N8_SPHERE_RADIUS,
            "standoff_m": N8_STANDOFF_M,
            "a_over_Rc": ratio,
            "field_points": f"{len(pts)} on the {N8_CORTEX_RADIUS} m cortical shell",
        },
    }
    rep.artifacts["self_convergence_family"] = rows

    if audits:
        uniform: list[dict[str, float]] = []
        ref_field = ref.induced_field(t_pts, t_pos, t_mdot)
        for k in (1, 2, 3, 4):
            verts, faces = icosphere(k)
            bem = ChargeBEM(
                [TriMesh(verts * N8_SPHERE_RADIUS, faces)], [0.33], [0.0]
            )
            got = bem.total_field(t_pts, t_pos, t_mdot)
            uniform.append({
                "subdivision": float(k),
                "n_elements": float(bem.n_faces),
                "panel_to_standoff": bem.near_source_resolution(t_pos)[
                    "panel_to_standoff"
                ],
                "error": float((got - ref_field).norm() / ref_field.norm()),
            })
        rep.artifacts["uniform_mesh_audit"] = uniform
        monotone = all(
            uniform[i]["error"] > uniform[i + 1]["error"]
            for i in range(len(uniform) - 1)
        )
        rep.artifacts["uniform_mesh_refinement_is_monotone"] = monotone
        rep.notes.append(
            "AUDIT -- why grading is required, not merely preferable. On UNIFORM meshes at "
            "this geometry the errors are "
            + ", ".join(
                f"{r['error']:.4f} ({int(r['n_elements'])} panels, ratio "
                f"{r['panel_to_standoff']:.2f})" for r in uniform
            )
            + f". Monotone under refinement: {monotone}. Refining one level makes the answer "
            "WORSE at the coarse end, so a user watching the error would have no way to "
            "distinguish that from convergence. scwbd.intervene.tms.efield refuses beyond "
            "panel_to_standoff = 1.0 for this reason."
        )

    rep.notes.append(
        "The N8 contract says the N6 spectral reference does not extend to contact, since "
        "its bound at a/R_c = 0.955 exceeds the solver error. That is correct for the "
        "GENERAL reference (degree 48, O(L^2) basis): its bound there is 1.10e-1. It is not "
        "correct for the reference used here. Rotating a single source onto the axis makes "
        "the Neumann data exactly azimuthal order one, so the basis is O(L) instead of "
        "O(L^2) and degree 400 is cheap; the bound becomes "
        f"{bound:.2e} and the measured agreement with the closed form {measured:.1e}. The "
        "contact regime is therefore validatable against an INDEPENDENT reference, not only "
        "by self-consistency. Suggest amending the N8 docstring."
    )
    rep.notes.append(
        "Both branches of the contract are supplied: an independent contact reference AND a "
        "Richardson self-convergence study. The self-convergence subcheck proves the "
        "discretisation converges to something; the reference subcheck is what says it "
        "converges to the right thing."
    )
    rep.notes.append(
        "Tolerance provenance: 0.05 is the repo's standing numerics tolerance, identical to "
        "N3/N4/N6. Disclosure, since this gate asks for preregistration: the graded "
        "refinement study was measured while scoping whether contact was reachable at all, "
        "before N8's contract was written. The tolerance was not adjusted afterwards."
    )
    rep.notes.append(
        "A PASS here licenses no claim about target engagement, network effect or clinical "
        "utility, and no claim-bearing run has been made."
    )
    return rep.finalize()


# ---------------------------------------------------------------------------
# N9: the runtime's fallback approximation, gated rather than labelled
# ---------------------------------------------------------------------------

#: geometry envelope for N9.  Standoffs up to A_safe's own
#: ``tms.coil_scalp_distance_mm`` maximum of 40 mm, and head radii down to
#: **60 mm** -- not because a 60 mm head is adult (it is not; adult radii are
#: ~80-100 mm, where the error is 0.74-0.89) but because nothing in the runtime's
#: ``HeadModel`` enforces an adult radius.  The bound has to cover what the code
#: admits, not what biology suggests.  Agent Asimov made that argument when they
#: declined to justify a smaller bound the easy way, and it is the same
#: discipline as calibrating the BEM envelope on the concentrated source: pick
#: the worst case the system can actually reach.
N9_HEAD_RADII_M = (0.060, 0.065, 0.070, 0.075, 0.085, 0.092, 0.100)
N9_STANDOFFS_M = (0.000, 0.005, 0.010, 0.020, 0.030, 0.040)
N9_REFERENCE_DEGREE = 250

#: The bound this gate is *pinned* to, with its justification and date.
#:
#: Reading the threshold from ``scwbd.runtime`` at run time is what stops this
#: gate going stale against a snapshot -- but it opens a second hole, which agent
#: Popper named: a threshold that can move is weaker than one that cannot, and
#: nothing would announce a silent loosening.  So the run-time read stays and the
#: value is pinned here as well.  Any movement fails
#: ``bound_provenance`` until this record is updated, and updating it is a commit
#: -- dated, attributed and reviewable by construction.
#:
#: A tightening fails too.  That is deliberate: the check is "did this move
#: without anyone saying why", not "did this get worse".
N9_PINNED_BOUND: dict[str, Any] = {
    "solution_discrepancy_fraction": (0.0, 1.35),
    "justified_on": "2026-08-06",
    "justified_by": "agent Asimov (scwbd.runtime), reviewed by agent Popper (scwbd.bench)",
    "source": "scwbd.runtime.backends.AnalyticSphericalEField",
    "justification": (
        "Split from the former composite discrepancy_fraction=(-0.8,+0.8), which "
        "declared one interval for two physically distinct things: this "
        "approximation's error against the exact solution of the same geometry, and "
        "that geometry's error against a real head. Upper bound 1.35 covers the "
        "measured worst case 1.3204 at a 60 mm head radius with 40 mm standoff. "
        "60 mm is not an adult head -- it is the smallest radius the runtime's "
        "HeadModel admits, and a bound must cover what the code admits. Lower bound "
        "0.0 is attained (circular coil, where the approximation is exact) and the "
        "approximation never underestimates."
    ),
}


def bound_has_moved(
    observed: tuple[float, float], pinned: tuple[float, float] | None = None
) -> bool:
    """Has the judged party's declared bound moved away from the pinned value?

    Extracted so the provenance guard can be exercised without running the full
    sweep. A guard nobody can show firing is indistinguishable from one that
    cannot fire.
    """
    ref = tuple(
        float(v) for v in (pinned or N9_PINNED_BOUND["solution_discrepancy_fraction"])
    )
    return not (
        math.isclose(float(observed[0]), ref[0], rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(observed[1]), ref[1], rel_tol=0.0, abs_tol=1e-12)
    )


def _n9_sweep(coil_kind: str = "figure8") -> list[dict[str, float]]:
    """Measure the approximation against the independent reference, geometry by geometry."""
    from .spectral_reference import AxialInductionReference
    from .tms.coil import CircularCoil, FigureEightCoil, biphasic
    from .tms.efield import (
        SphericalHeadModel,
        coil_dipoles_in_head_frame,
        primary_tangential_projection,
    )
    from .tms.pose import coil_pose_on_sphere

    rows: list[dict[str, float]] = []
    for radius in N9_HEAD_RADII_M:
        for standoff in N9_STANDOFFS_M:
            head = SphericalHeadModel(radius=radius, cortex_radius=radius - 0.015)
            coil = (
                FigureEightCoil(n_azimuth=32, n_radial=4)
                if coil_kind == "figure8"
                else CircularCoil(n_azimuth=32, n_radial=4)
            )
            pose = coil_pose_on_sphere(
                head, [-0.55, 0.68, 0.48], standoff_m=standoff
            )
            pos, mdot = coil_dipoles_in_head_frame(
                coil, pose.matrix(), float(biphasic().peak_didt)
            )
            pts, normals = head.cortical_shell(642)
            reference = AxialInductionReference(
                radius=radius, degree=N9_REFERENCE_DEGREE
            ).induced_field(pts, pos, mdot)
            approx = primary_tangential_projection(pts, normals, pos, mdot)
            peak_ratio = float(
                approx.norm(dim=-1).max() / reference.norm(dim=-1).max()
            )
            peak = int(reference.norm(dim=-1).argmax())
            cosine = float(
                (approx[peak] @ reference[peak])
                / (approx[peak].norm() * reference[peak].norm()).clamp_min(1e-30)
            )
            rows.append({
                "head_radius_m": radius,
                "standoff_m": standoff,
                "peak_ratio": peak_ratio,
                "relative_overestimate": peak_ratio - 1.0,
                "mean_relative_error": float(
                    (approx - reference).norm(dim=-1).mean()
                    / reference.norm(dim=-1).mean()
                ),
                "peak_direction_cosine": cosine,
            })
    return rows


def run_n9(*, include_axisymmetric: bool = True) -> Any:
    """N9: does the fallback's declared discrepancy interval cover its real error?"""
    from scwbd.bench.report import ClaimManifest, ClaimReport, Metric, SubCheck, could_not_run

    rows = _n9_sweep("figure8")
    worst = max(rows, key=lambda r: r["relative_overestimate"])
    best = min(rows, key=lambda r: r["relative_overestimate"])
    worst_mean_rel = max(r["mean_relative_error"] for r in rows)
    min_cos = min(r["peak_direction_cosine"] for r in rows)

    # Read the declared interval from the runtime AT RUN TIME rather than copying
    # it, so this gate tracks the real value instead of a snapshot that goes
    # stale.  Point it at ``solution_discrepancy_fraction`` -- the term that
    # claims to cover THIS error -- and not at the composite
    # ``discrepancy_fraction``, which also carries the sphere-vs-head geometry
    # prior.  Testing an approximation's own error against an interval that also
    # holds something else is the conflation this gate exists to catch, one level
    # up; agent Asimov asked for the change and they are right.
    solution: tuple[float, float] | None = None
    composite: tuple[float, float] | None = None
    declared_reason = ""
    try:
        from scwbd.runtime.backends import AnalyticSphericalEField

        backend = AnalyticSphericalEField()
        composite = tuple(float(v) for v in backend.discrepancy_fraction)
        raw = getattr(backend, "solution_discrepancy_fraction", None)
        if raw is not None:
            solution = (float(raw[0]), float(raw[1]))
    except Exception as exc:  # pragma: no cover - depends on sibling module
        declared_reason = f"{type(exc).__name__}: {exc}"

    subs: list[SubCheck] = []

    if solution is None:
        subs.append(could_not_run(
            "solution_bound_covers_error",
            "Does the term that claims to cover THIS error actually cover it?",
            "scwbd.runtime.backends.AnalyticSphericalEField exposes no "
            "solution_discrepancy_fraction"
            + (f" ({declared_reason})" if declared_reason else "")
            + ". The composite discrepancy_fraction cannot stand in: it also carries the "
            "sphere-vs-head geometry prior, so a pass against it would be the same "
            "conflation this gate exists to catch, one level up.",
            falsified_by="the declared solution-discrepancy interval does not cover the "
                         "approximation's measured error",
        ))
    else:
        subs.append(SubCheck(
            name="solution_bound_covers_error",
            description=(
                "solution_discrepancy_fraction is the backend's claim about its error "
                "against the exact solution of the SAME geometry. That is exactly what "
                "this gate measures, so it is the interval to test against."
            ),
            metrics=[
                Metric(
                    name="fallback.max_relative_overestimate",
                    value=float(worst["relative_overestimate"]),
                    kind="numerical", exact=True,
                    threshold=float(solution[1]), direction="less_is_better",
                    note=(f"worst over the envelope, at head radius "
                          f"{worst['head_radius_m'] * 1e3:.0f} mm and standoff "
                          f"{worst['standoff_m'] * 1e3:.0f} mm; declared solution "
                          f"interval {solution}"),
                ),
                Metric(
                    name="fallback.composite_bound_would_have_been",
                    value=float(max(abs(composite[0]), abs(composite[1])))
                    if composite else float("inf"),
                    kind="audit", exact=True,
                    note="the composite discrepancy_fraction, recorded to show what this "
                         "check would have been graded against had it kept pointing at "
                         "the wrong term -- a threshold twice as loose, passing trivially",
                ),
            ],
            mandatory=True,
            falsified_by="the declared solution-discrepancy interval does not cover the "
                         "approximation's own measured error over its declared envelope",
        ))
        subs.append(SubCheck(
            name="approximation_never_underestimates",
            description=(
                "The declared interval is one-sided (lower bound 0.0). That is a claim "
                "about the physics, not a convenience, so it gets tested: the tangential "
                "projection drops a term and must therefore never come out low."
            ),
            metrics=[Metric(
                name="fallback.min_relative_overestimate",
                value=float(best["relative_overestimate"]),
                kind="numerical", exact=True,
                threshold=float(solution[0]), direction="greater_is_better",
                note=f"minimum over the envelope, at head radius "
                     f"{best['head_radius_m'] * 1e3:.0f} mm and standoff "
                     f"{best['standoff_m'] * 1e3:.0f} mm",
            )],
            mandatory=True,
            falsified_by="the approximation underestimates somewhere in the envelope, so "
                         "a one-sided interval is the wrong shape for it",
        ))

    if solution is not None:
        pinned = tuple(float(v) for v in N9_PINNED_BOUND["solution_discrepancy_fraction"])
        moved = bound_has_moved(solution, pinned)
        direction = (
            "LOOSENED" if solution[1] > pinned[1] or solution[0] < pinned[0]
            else "tightened"
        )
        note = (
            f"runtime declares {solution}, matching the value pinned and justified on "
            f"{N9_PINNED_BOUND['justified_on']} by {N9_PINNED_BOUND['justified_by']}"
            if not moved else
            f"pinned {pinned} justified on {N9_PINNED_BOUND['justified_on']}; runtime "
            f"now declares {solution} ({direction}). Update N9_PINNED_BOUND in "
            "scwbd.intervene.run_field_gates with the new value, a date and a "
            "justification, in the same commit that moves it."
        )
        subs.append(SubCheck(
            name="bound_provenance",
            description=(
                "Has the threshold moved since it was last justified? The bound is read "
                "from the judged party at run time, which is what stops this gate going "
                "stale -- and that is exactly what would let it be loosened silently. "
                "So it is also pinned here, and any movement fails until the pin is "
                "updated: a commit, hence dated and attributable by construction. A "
                "tightening fails too; the question is 'did this move without anyone "
                "saying why', not 'did this get worse'."
            ),
            metrics=[Metric(
                name="fallback.bound_moved_since_justified",
                value=1.0 if moved else 0.0,
                kind="numerical", exact=True,
                # boolean indicator; the comparison is strict, so 0.5 is the
                # threshold that separates "did not move" from "moved"
                threshold=0.5, direction="less_is_better",
                note=note,
            )],
            mandatory=True,
            falsified_by="the judged party's declared bound changed without a dated "
                         "justification",
        ))

    subs.append(SubCheck(
        name="error_is_characterised",
        description="Is the error bounded and orderly over the envelope, or wild?",
        metrics=[
            Metric(name="fallback.peak_direction_cosine_min", value=min_cos,
                   kind="numerical", exact=True, threshold=0.99,
                   direction="greater_is_better",
                   note="the approximation gets the direction right and the magnitude "
                        "wrong; that is worth stating precisely, because a direction-only "
                        "consumer is affected differently from a magnitude consumer"),
        ],
        mandatory=True,
        falsified_by="the approximation's error is not characterised over its envelope",
    ))

    subs.append(SubCheck(
        name="not_the_induced_field",
        description="Recorded so nobody reads a bounded approximation as a solver.",
        metrics=[Metric(
            name="fallback.max_mean_relative_error", value=float(worst_mean_rel),
            kind="audit", exact=True,
            note="elementwise error against the closed form, of order 100 % of the mean "
                 "field magnitude. This is an approximation with a bound, not a field "
                 "solver, and no gate here makes it one.",
        )],
        mandatory=False,
    ))

    axisym: list[dict[str, float]] = []
    if include_axisymmetric:
        axisym = _n9_sweep("circular")
        subs.append(SubCheck(
            name="exact_for_axisymmetric_sources",
            description=(
                "A circular coil coaxial with the head radius has a purely azimuthal "
                "primary vector potential, so r.E_p vanishes, the Neumann data is zero, "
                "and there is no secondary field: the approximation is EXACT."
            ),
            metrics=[Metric(
                name="fallback.axisymmetric_max_relative_error",
                value=float(max(r["mean_relative_error"] for r in axisym)),
                kind="audit", exact=True,
                note="round-off. This is the trap: validated on a circular coil the "
                     "approximation looks perfect. The error is a function of source "
                     "SYMMETRY, not of any resolution parameter, so refining nothing "
                     "finds it -- only changing the coil does.",
            )],
            mandatory=False,
        ))

    man = ClaimManifest(
        claim_id="N9_fallback_field_approximation",
        claim_text=(
            "The runtime's fallback field model (tangential projection of the primary "
            "field) has a measured error over its declared geometry envelope, and the "
            "discrepancy interval it declares covers that error."
        ),
        falsified_by=(
            "the declared discrepancy interval does not cover the approximation's own "
            "measured error over the declared envelope"
        ),
        consequence_if_failed=(
            "The fallback backend's ledger understates its own model discrepancy. Any "
            "dose, engagement or ranking obtained through it carries an uncertainty "
            "interval narrower than the physics justifies, and scwbd.runtime must widen "
            "the declared interval or refuse to use the backend for claim-bearing output."
        ),
        thesis_reference="body.tex §11.1",
        acceptance_thresholds={
            "solution_discrepancy_fraction": list(solution) if solution else None,
            "composite_discrepancy_fraction": list(composite) if composite else None,
            "envelope_head_radii_m": list(N9_HEAD_RADII_M),
            "envelope_standoffs_m": list(N9_STANDOFFS_M),
        },
        non_goals=[
            "This gate bounds an approximation. It does not make it a field solver, and "
            "a PASS would license no claim about target engagement or clinical utility.",
        ],
        seed=SEED,
    )

    rep = ClaimReport(
        manifest=man, subchecks=subs, kind="numerics",
        artifacts={
            "subject": "scwbd.intervene.tms.efield.primary_tangential_projection",
            "reference": "scwbd.intervene.spectral_reference.AxialInductionReference "
                         f"(degree {N9_REFERENCE_DEGREE})",
            "solver_provenance": {
                "consumer": "scwbd.runtime.backends.AnalyticSphericalEField",
                "solution_discrepancy_fraction": list(solution) if solution else None,
                "composite_discrepancy_fraction": list(composite) if composite else None,
                "equivalence_pinned_by":
                    "tests/intervene/test_fallback_approximation.py -- asserts the runtime "
                    "backend computes this same expression, so the gate's subject is the "
                    "object actually in the runtime path",
            },
            "bound_provenance": {
                "pinned": list(N9_PINNED_BOUND["solution_discrepancy_fraction"]),
                "observed": list(solution) if solution else None,
                "justified_on": N9_PINNED_BOUND["justified_on"],
                "justified_by": N9_PINNED_BOUND["justified_by"],
                "justification": N9_PINNED_BOUND["justification"],
            },
            "figure_eight_sweep": rows,
            "axisymmetric_sweep": axisym,
        },
        notes=[
            "The approximation is EXACT for a coil whose windings are circular loops "
            "coaxial with the head radius: the primary vector potential is then purely "
            "azimuthal, so r.E_p = 0, the Neumann data vanishes and there is no secondary "
            "field. Measured error for a circular coil is round-off. A gate that tested "
            "only that case would report a perfect PASS and be worthless.",
            "For a figure-eight the wings are opposed and sit off the radial axis, the "
            "Neumann data does not vanish, and the approximation is high by "
            f"{best['relative_overestimate'] + 1:.2f}x to "
            f"{worst['relative_overestimate'] + 1:.2f}x at the peak across the envelope. "
            "The overestimate grows monotonically as the head gets smaller and the "
            "standoff larger.",
            "Direction is essentially unaffected (peak cosine >= "
            f"{min_cos:.4f}); the error is almost purely in magnitude. A consumer that "
            "uses only the field DIRECTION is affected far less than one that uses its "
            "magnitude, and the two should not inherit the same bound.",
            "The declared interval is read from scwbd.runtime at run time rather than "
            "copied into this gate, so it tracks the real value instead of a snapshot "
            "that silently goes stale. It is read from solution_discrepancy_fraction, "
            "the term that claims to cover THIS error, not from the composite "
            "discrepancy_fraction which also carries the geometry prior -- grading "
            "against the composite would repeat, one level up, the conflation the gate "
            "was built to catch.",
            "The envelope reaches down to a 60 mm head radius. That is not an adult "
            "head; adult radii are ~80-100 mm, where the overestimate is 0.74-0.89. It "
            "is here because nothing in the runtime's HeadModel enforces an adult "
            "radius, and a bound has to cover what the code admits rather than what "
            "biology suggests.",
            "The threshold is read at run time AND pinned here with a dated "
            "justification. The run-time read stops the gate going stale against a "
            "snapshot; the pin stops the bound moving without anyone saying why. Agent "
            "Popper raised the second problem as the cost of my solution to the first, "
            "and both properties are obtainable together -- a hardcoded snapshot has "
            "neither.",
        ],
    )
    return rep.finalize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/intervene",
                    help="directory for the handover reports (NOT reports/gates)")
    ap.add_argument("--n-per-axis", type=int, default=256)
    ap.add_argument("--ppw", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-convergence", action="store_true")
    ap.add_argument("--only", choices=("n3", "n4", "n6", "n8", "n9"), default=None)
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
    if args.only in (None, "n6"):
        reports.append(run_n6(convergence=conv))
    if args.only in (None, "n8"):
        reports.append(run_n8(self_convergence=conv, audits=conv))
    if args.only in (None, "n9"):
        reports.append(run_n9(include_axisymmetric=conv))

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
