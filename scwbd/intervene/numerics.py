"""Independent numerical solvers for the §11.1 field-physics gates (N3, N4).

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Thesis §11.1 requires the electromagnetic and acoustic solvers to be validated
*independently of any neural-response model*.  ``scwbd.bench.numerics`` states
the two references that must be reproduced:

``N3``  the quasi-static potential of a **current dipole** in an infinite
        homogeneous conductor, :math:`\\phi = p\\cdot(r-r_0)/(4\\pi\\sigma|r-r_0|^3)`;
``N4``  the **free-field monopole** :math:`p = A\\,e^{ikr}/r`, i.e. geometric
        spreading, together with a Helmholtz residual that vanishes under grid
        refinement.

Neither reference is the problem the *production* operators solve.
:mod:`scwbd.intervene.tms.efield` computes the **magnetically induced** field of
a coil (a different source term), and :mod:`scwbd.intervene.tfus.acoustics`
propagates an aperture forward (a different boundary condition).  So this module
supplies two purpose-built discretisations of exactly the reference problems.
They are deliberately *bare*: a second-order finite-difference Poisson solve and
a second-order FDTD wave solve, sharing no code path with the closed forms they
are checked against.

The point of a verification gate is destroyed if the reference leaks into the
solver, so:

* :func:`quasistatic_dipole_potential_fd` uses **homogeneous Dirichlet data on a
  far truncation boundary** -- zero, not the analytic potential.  No value from
  the reference enters the solve anywhere.  The truncation error is a reported,
  refinable quantity (:func:`em_grid_convergence`), not a hidden calibration.
* :func:`free_field_monopole_fdtd` marches the wave equation in time with an
  absorbing sponge and extracts the steady-state phasor.  The radiation
  condition is imposed by the physics of the march, not by analytic boundary
  data, and the source is a bare lattice delta whose strength is fixed
  *a priori* (:math:`Q=4\\pi A`) rather than fitted to the reference.

Both solvers are second order and both are shipped with convergence helpers so
that a verdict can be read as "converges to the reference", never as "happens to
agree at one resolution".

References
----------
Hämäläinen M et al. (1993) Rev Mod Phys 65:413-497 (quasi-static approximation).
Wolters CH et al. (2007) SIAM J Sci Comput 30:24-45 (dipole FE forward models).
Berenger JP (1994) J Comput Phys 114:185-200 (absorbing layers).
Treeby BE, Cox BT (2010) J Biomed Opt 15:021314 (k-Wave; unusable on aarch64,
see :func:`scwbd.intervene.tfus.acoustics.kwave_status`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .base import SIMULATION_ONLY_NOTICE

__all__ = [
    "FDPoissonResult",
    "quasistatic_dipole_potential_fd",
    "solve_dipole_potential",
    "em_grid_convergence",
    "FDTDResult",
    "free_field_monopole_fdtd",
    "run_free_field_monopole",
    "acoustic_grid_convergence",
]


# ---------------------------------------------------------------------------
# N3: quasi-static current dipole, second-order finite differences
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FDPoissonResult:
    """Potential at the requested points plus the discretisation that produced it."""

    potential: np.ndarray  # [N] volts
    n_per_axis: int
    spacing_m: float
    half_width_m: float
    boundary: str
    notice: str = SIMULATION_ONLY_NOTICE
    meta: dict[str, Any] = field(default_factory=dict)


def solve_dipole_potential(
    points: np.ndarray,
    dipole_pos: Sequence[float] = (0.0, 0.0, 0.0),
    dipole_moment: Sequence[float] = (0.0, 0.0, 1e-8),
    sigma: float = 0.33,
    *,
    n_per_axis: int = 256,
    half_width_m: float | None = None,
    margin: float = 1.9,
    interp_order: int = 3,
) -> FDPoissonResult:
    """Solve :math:`\\nabla\\cdot(\\sigma\\nabla\\phi) = \\nabla\\cdot(p\\,\\delta)` by
    second-order finite differences.

    The grid is a cube centred on ``dipole_pos`` with the dipole **exactly on a
    node** (``n_per_axis`` even), so no sub-cell source placement error is
    introduced.  The dipole is represented by a :math:`\\pm I` monopole pair on
    the two neighbouring nodes of each active component, with :math:`I = p/2h`;
    this reproduces an ideal dipole to :math:`O(h^2)`.

    The homogeneous-Dirichlet system is diagonalised exactly by the type-I
    discrete sine transform, so this is a *direct* solve -- no iteration, no
    tolerance, nothing to tune.  Boundary values are **zero**: the truncation
    error that costs is quantified by :func:`em_grid_convergence` and shrinks
    with ``margin``.  It is not repaired with analytic data, because a solver
    handed the answer on its boundary is not being tested.

    Parameters
    ----------
    points : ``[N,3]`` field points, metres.
    dipole_pos : dipole location, metres.
    dipole_moment : current-dipole moment :math:`p`, A m.
    sigma : conductivity, S/m.
    n_per_axis : cells per axis (even).  Cost is :math:`O(n^3\\log n)`.
    half_width_m : truncation half-width.  Defaults to ``margin`` times the
        farthest requested point, so the boundary is well outside the data.
    """
    from scipy.fft import dstn
    from scipy.ndimage import map_coordinates

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    r0 = np.asarray(dipole_pos, dtype=float).reshape(3)
    p = np.asarray(dipole_moment, dtype=float).reshape(3)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"conductivity must be positive, got {sigma}")
    N = int(n_per_axis)
    if N % 2:
        raise ValueError("n_per_axis must be even so the dipole sits on a node")

    rmax = float(np.linalg.norm(pts - r0, axis=1).max())
    L = float(half_width_m) if half_width_m is not None else margin * rmax
    if L <= rmax:
        raise ValueError(
            f"truncation half-width {L:.4f} m does not enclose the farthest "
            f"requested point at {rmax:.4f} m"
        )
    h = 2 * L / N
    n = N - 1  # interior nodes per axis

    # --- right-hand side: sigma * lap(phi) = -I delta(r - r_+) + I delta(r - r_-)
    rhs = np.zeros((n, n, n))
    i0 = np.full(3, N // 2)  # the dipole node, exact by construction
    for comp in range(3):
        if p[comp] == 0.0:
            continue
        current = p[comp] / (2 * h)  # I d = p with d = 2h
        for step, sign in ((+1, +1.0), (-1, -1.0)):
            idx = i0.copy()
            idx[comp] += step
            ii = idx - 1  # interior indexing
            rhs[ii[0], ii[1], ii[2]] += -(sign * current) / (sigma * h**3)

    # --- direct solve: DST-I diagonalises the Dirichlet 7-point Laplacian
    lam = (2 * np.cos(math.pi * np.arange(1, n + 1) / (n + 1)) - 2) / h**2
    F = dstn(rhs, type=1, norm="ortho")
    F /= lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    phi = dstn(F, type=1, norm="ortho")
    del F, rhs

    full = np.zeros((N + 1, N + 1, N + 1))
    full[1:-1, 1:-1, 1:-1] = phi
    del phi

    coords = ((pts - r0 + L) / h).T
    values = map_coordinates(full, coords, order=int(interp_order), mode="nearest")
    return FDPoissonResult(
        potential=np.asarray(values, dtype=float),
        n_per_axis=N,
        spacing_m=h,
        half_width_m=L,
        boundary="homogeneous_dirichlet_on_truncation_box",
        meta={
            "sigma_S_per_m": sigma,
            "dipole_moment_A_m": p.tolist(),
            "farthest_point_m": rmax,
            "interp_order": int(interp_order),
            "scheme": "7-point second-order FD, DST-I direct solve",
            "analytic_data_used": False,
        },
    )


def quasistatic_dipole_potential_fd(
    points: np.ndarray,
    dipole_pos: Sequence[float] = (0.0, 0.0, 0.0),
    dipole_moment: Sequence[float] = (0.0, 0.0, 1e-8),
    sigma: float = 0.33,
    **kwargs: Any,
) -> np.ndarray:
    """Bare-array adapter with the signature ``scwbd.bench.numerics`` calls."""
    return solve_dipole_potential(
        points, dipole_pos, dipole_moment, sigma, **kwargs
    ).potential


def em_grid_convergence(
    points: np.ndarray,
    *,
    n_list: Sequence[int] = (128, 192, 256),
    dipole_pos: Sequence[float] = (0.0, 0.0, 0.0),
    dipole_moment: Sequence[float] = (0.0, 0.0, 1e-8),
    sigma: float = 0.33,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Refinement study against the closed form, with observed orders.

    Returned rows carry ``spacing_m``, ``mean_relative_error`` (normalised by
    the mean |reference|, exactly as the gate does) and ``observed_order``
    relative to the previous row.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d = pts - np.asarray(dipole_pos, dtype=float)
    dist = np.linalg.norm(d, axis=-1)
    ref = (d @ np.asarray(dipole_moment, dtype=float)) / (
        4.0 * math.pi * float(sigma) * dist**3
    )
    scale = float(np.abs(ref).mean())

    rows: list[dict[str, float]] = []
    for N in n_list:
        res = solve_dipole_potential(
            pts, dipole_pos, dipole_moment, sigma, n_per_axis=int(N), **kwargs
        )
        err = float((np.abs(res.potential - ref) / scale).mean())
        row = {
            "n_per_axis": float(N),
            "spacing_m": res.spacing_m,
            "mean_relative_error": err,
            "max_relative_error": float((np.abs(res.potential - ref) / scale).max()),
            "observed_order": float("nan"),
        }
        if rows:
            prev = rows[-1]
            row["observed_order"] = math.log(
                prev["mean_relative_error"] / err
            ) / math.log(prev["spacing_m"] / res.spacing_m)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# N4: free-field monopole, second-order FDTD to steady state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FDTDResult:
    """Steady-state phasor at the requested points, plus a clean grid block."""

    pressure: np.ndarray  # [N] complex, Pa (arbitrary source normalisation)
    grid_block: np.ndarray  # [m,m,m] complex, for the Helmholtz residual
    spacing_m: float
    n_per_axis: int
    points_per_wavelength: float
    n_steps: int
    notice: str = SIMULATION_ONLY_NOTICE
    meta: dict[str, Any] = field(default_factory=dict)


def run_free_field_monopole(
    points: np.ndarray,
    source_pos: Sequence[float] = (0.0, 0.0, 0.0),
    k: float = 100.0,
    *,
    amplitude: float = 1.0,
    sound_speed_m_per_s: float = 1500.0,
    points_per_wavelength: int = 20,
    steps_per_period: int | None = None,
    sponge_wavelengths: float = 2.5,
    n_transient_periods: int = 20,
    n_measure_periods: int = 10,
    device: str | None = None,
    block_wavelengths: float = 1.0,
    block_offset_wavelengths: float = 1.5,
) -> FDTDResult:
    """Radiate a lattice-delta monopole and read off the steady-state phasor.

    Marches :math:`\\partial_t^2 p + 2\\gamma\\,\\partial_t p = c^2\\nabla^2 p +
    c^2 Q\\cos(\\omega t)\\,\\delta^3(r-r_0)` with a second-order leapfrog and a
    quadratically ramped damping sponge on the outer ``sponge_wavelengths`` of
    each axis.  Outgoing waves leave; nothing analytic is imposed at the
    boundary.

    The source strength is fixed *a priori*: the continuum solution of that
    equation is :math:`p = Q\\cos(\\omega(t-r/c))/(4\\pi r)`, so ``Q = 4 pi A``
    gives amplitude ``A/r``.  It is **not** rescaled to match the reference --
    the residual amplitude bias of the lattice delta is left visible and is
    reported by :func:`acoustic_grid_convergence`.

    ``steps_per_period`` is an integer so the phasor extraction integrates a
    whole number of cycles exactly.  ``grid_block`` is a source-free,
    sponge-free cube for the Helmholtz-residual subcheck.
    """
    import torch

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    r0 = np.asarray(source_pos, dtype=float).reshape(3)
    k = float(k)
    if k <= 0:
        raise ValueError(f"wavenumber must be positive, got {k}")
    c = float(sound_speed_m_per_s)
    lam = 2 * math.pi / k
    omega = k * c
    period = 2 * math.pi / omega
    dx = lam / float(points_per_wavelength)
    # 3 steps per point per wavelength holds the CFL number at 1/sqrt(3) for any
    # resolution, so refining h refines dt with it and the Helmholtz residual --
    # which is set by the TEMPORAL dispersion -- actually moves
    steps_per_period = int(steps_per_period or 3 * int(points_per_wavelength))
    dt = period / steps_per_period
    if dt >= dx / (c * math.sqrt(3.0)):
        raise ValueError(
            f"CFL violated: dt={dt:.3e} s exceeds dx/(c*sqrt(3))="
            f"{dx / (c * math.sqrt(3.0)):.3e} s; raise steps_per_period"
        )

    reach = float(np.abs(pts - r0).max())
    # the undamped region must hold the data and leave a source-free wavelength
    # for the Helmholtz-residual block; for realistic clouds the data dominates
    inner = max(reach + 2 * dx, 1.25 * lam)
    sponge = sponge_wavelengths * lam
    N = 2 * int(math.ceil((inner + sponge) / dx)) + 1  # odd -> source on a node
    L = (N - 1) / 2 * dx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float32

    ax = (torch.arange(N, device=dev, dtype=dtype) - (N - 1) // 2) * dx
    ramp = ((ax.abs() - inner).clamp_min(0.0) / sponge) ** 2
    gamma_max = 8.0 * c * 3.0 / (2.0 * sponge)
    gam = (gamma_max / 3.0) * (
        ramp[:, None, None] + ramp[None, :, None] + ramp[None, None, :]
    )
    denom = 1.0 / (1.0 + gam * dt)
    fac_old = 1.0 - gam * dt
    del gam, ramp

    p = torch.zeros((N, N, N), device=dev, dtype=dtype)
    pm = torch.zeros_like(p)
    acc_c = torch.zeros((N, N, N), device=dev, dtype=torch.float64)
    acc_s = torch.zeros_like(acc_c)

    i0 = (N - 1) // 2
    cdt2 = (c * dt) ** 2
    src_scale = cdt2 * (4.0 * math.pi * float(amplitude)) / dx**3
    n_pp = int(steps_per_period)
    n_steps = (int(n_transient_periods) + int(n_measure_periods)) * n_pp
    measure_from = int(n_transient_periods) * n_pp

    lap = torch.zeros_like(p)
    for it in range(n_steps):
        lap[1:-1, 1:-1, 1:-1] = (
            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
            - 6.0 * p[1:-1, 1:-1, 1:-1]
        ) / dx**2
        soft = min(1.0, it / (3.0 * n_pp))  # switch the source on smoothly
        newp = (2.0 * p - pm * fac_old + cdt2 * lap) * denom
        newp[i0, i0, i0] += src_scale * math.cos(omega * it * dt) * soft * float(
            denom[i0, i0, i0]
        )
        pm, p = p, newp
        if it >= measure_from:
            t_next = (it + 1) * dt
            acc_c += p.double() * math.cos(omega * t_next)
            acc_s += p.double() * math.sin(omega * t_next)

    norm = 2.0 / (int(n_measure_periods) * n_pp)
    a_c = (acc_c * norm).cpu().numpy()
    a_s = (acc_s * norm).cpu().numpy()
    del p, pm, lap, acc_c, acc_s, denom, fac_old
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    from scipy.ndimage import map_coordinates

    coords = ((pts - r0 + L) / dx).T
    re = map_coordinates(a_c, coords, order=3, mode="nearest")
    im = map_coordinates(a_s, coords, order=3, mode="nearest")

    # source-free, sponge-free cube for the Helmholtz residual
    inner_cells = int(inner / dx)
    # keep the block clear of the lattice source and inside the undamped region,
    # shrinking it to fit rather than failing on a small domain
    half = min(max(2, int(round(block_wavelengths * lam / dx / 2))),
               max(2, (inner_cells - 4) // 2))
    off = min(int(round(block_offset_wavelengths * lam / dx)), inner_cells - half - 2)
    if off <= half:
        raise ValueError(
            "no source-free, sponge-free block fits: the physical region is only "
            f"{inner_cells} cells across but the residual block needs "
            f"{2 * half + 1} plus clearance from the source. Widen the domain."
        )
    lo, hi = i0 + off - half, i0 + off + half + 1
    block = (a_c + 1j * a_s)[lo:hi, i0 - half : i0 + half + 1, i0 - half : i0 + half + 1]

    return FDTDResult(
        pressure=re + 1j * im,
        grid_block=np.ascontiguousarray(block),
        spacing_m=dx,
        n_per_axis=N,
        points_per_wavelength=float(points_per_wavelength),
        n_steps=n_steps,
        meta={
            "wavenumber_per_m": k,
            "sound_speed_m_per_s": c,
            "time_step_s": dt,
            "cfl": c * dt * math.sqrt(3.0) / dx,
            "half_width_m": L,
            "sponge_m": sponge,
            "device": str(dev),
            "scheme": "second-order leapfrog FDTD, quadratic damping sponge",
            "analytic_data_used": False,
            "amplitude_fitted": False,
        },
    )


def free_field_monopole_fdtd(
    points: np.ndarray,
    source_pos: Sequence[float] = (0.0, 0.0, 0.0),
    k: float = 100.0,
    **kwargs: Any,
) -> np.ndarray:
    """Bare-array adapter with the signature ``scwbd.bench.numerics`` calls."""
    return run_free_field_monopole(points, source_pos, k, **kwargs).pressure


def acoustic_grid_convergence(
    points: np.ndarray,
    *,
    ppw_list: Sequence[int] = (10, 14, 20),
    source_pos: Sequence[float] = (0.0, 0.0, 0.0),
    k: float = 100.0,
    amplitude: float = 1.0,
    steps_per_period_factor: int = 3,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Refinement study against free-field spreading.

    ``dt`` is refined **with** ``h`` (``steps_per_period =
    steps_per_period_factor * ppw``), which holds the CFL number fixed and is
    what makes the Helmholtz residual a refinement statement rather than a
    constant.  Holding ``dt`` fixed while refining ``h`` leaves the residual
    flat, because the discrete steady state satisfies the *discrete* Helmholtz
    equation exactly: measuring it with the scheme's own Laplacian cancels the
    spatial error and leaves only the temporal dispersion,
    :math:`|k^2-\\kappa^2|/k^2 = (\\omega\\,\\mathrm dt)^2/12 + O(\\mathrm dt^4)`
    where :math:`\\kappa = (2/c\\,\\mathrm dt)\\sin(\\omega\\,\\mathrm dt/2)`.

    Also reports ``mean_amplitude_ratio``: the FDTD amplitude divided by the
    reference, whose departure from 1 is the lattice-delta source bias.  It is
    reported rather than divided out.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    dist = np.linalg.norm(pts - np.asarray(source_pos, dtype=float), axis=-1)
    ref = float(amplitude) / dist
    scale = float(ref.mean())

    rows: list[dict[str, float]] = []
    for ppw in ppw_list:
        res = run_free_field_monopole(
            pts, source_pos, k, amplitude=amplitude,
            points_per_wavelength=int(ppw),
            steps_per_period=int(steps_per_period_factor) * int(ppw), **kwargs
        )
        got = np.abs(res.pressure)
        err = float((np.abs(got - ref) / scale).mean())
        row = {
            "points_per_wavelength": float(ppw),
            "spacing_m": res.spacing_m,
            "time_step_s": float(res.meta["time_step_s"]),
            "n_per_axis": float(res.n_per_axis),
            "mean_relative_error": err,
            "max_relative_error": float((np.abs(got - ref) / scale).max()),
            "mean_amplitude_ratio": float((got / ref).mean()),
            "helmholtz_relative_residual": _helmholtz_residual(
                res.grid_block, dx=res.spacing_m, k=k
            ),
            "observed_order": float("nan"),
        }
        if rows:
            prev = rows[-1]
            row["observed_order"] = math.log(
                prev["mean_relative_error"] / err
            ) / math.log(prev["spacing_m"] / res.spacing_m)
        rows.append(row)
    return rows


def _helmholtz_residual(field: np.ndarray, *, dx: float, k: float) -> float:
    """Local mirror of the gate's residual, so convergence can be tabulated here."""
    p = np.asarray(field)
    lap = np.zeros_like(p)
    for axis in range(p.ndim):
        lap = lap + (
            np.roll(p, 1, axis=axis) - 2.0 * p + np.roll(p, -1, axis=axis)
        ) / dx**2
    core = tuple(slice(1, -1) for _ in range(p.ndim))
    res = lap[core] + (k**2) * p[core]
    return float(np.linalg.norm(res) / (np.linalg.norm((k**2) * p[core]) + 1e-30))
