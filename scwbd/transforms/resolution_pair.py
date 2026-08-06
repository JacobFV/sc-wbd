"""The one declared fine/coarse resolution pair -- thesis §4.2, narrowing N-12.

    "Where the resolution poset admits the relationship, SC-WBD learns and
     validates paired restriction and prolongation operators [...] where R_i
     compresses fine state into the coarse sufficient statistics consumed by
     neighbors and P_i expands incoming coarse messages into a conditional
     distribution over fine states.  Cross-resolution consistency is assessed by
     whether a refined module preserves observable and perturbational
     predictions at its boundary, not merely whether its internal trajectory
     looks plausible."

Until now ``scwbd.transforms.sheaf`` implemented that machinery and nothing
declared an instance of it, so refusal **R02** had nothing to object to
(``compiler_bridge.py`` said so in as many words).  This module declares
exactly one pair and makes it measurable.

The pair
--------
``cortical_source_dipole`` (fine) ``<=`` ``parcel`` (coarse)

The coarse id is deliberately the same string ``scwbd.foundation``'s schema
already uses for the support every region's state lives at, so the pair binds
to the artifact rather than to a parallel vocabulary.

* **fine** -- the subject's own cortical source space: one normal-oriented
  current dipole per decimated white-surface vertex.  This is the support at
  which the EEG/MEG lead field is *defined*: every column of ``G`` is one of
  these dipoles.  It is not a raster and not nested inside the parcel grid in
  any dyadic sense; it is a mesh, which is precisely §2.6's point.
* **coarse** -- anatomical parcels.  This is the support at which
  ``scwbd.foundation`` keeps regional state, at which the connectome is
  defined, and at which every θ prior is tabulated.

Both supports are real: the fine one because the forward model is solved on it
from a real BEM of a real subject's MRI, the coarse one because the region
state and the coupling graph live there.  The boundary between them is where
§4.2's question actually bites.

The operators
-------------
``R`` -- area-weighted parcel mean of the normal current::

    (R x)_p = sum_{v in p} a_v x_v / sum_{v in p} a_v

``P`` -- indicator prolongation: a parcel's state describes its whole extent::

    (P c)_v = c_{p(v)}

These are *paired*, not two unrelated maps: ``R P = I`` exactly on the coarse
support (a proper right inverse, verified to machine precision), and ``P R`` is
the ``a``-weighted orthogonal projector onto piecewise-constant fields.  ``P``
is wrapped in :class:`~scwbd.transforms.sheaf.Prolongation`, so it returns a
:class:`~scwbd.transforms.sheaf.FineDistribution` -- the ``n_fine - n_coarse``
directions the parcel state cannot see carry prior variance, never
reconstructed structure.

Why the indicator and not the pseudoinverse: ``pinv(R)`` is also a right
inverse, but it has no content.  "Every source in this parcel takes the
parcel's value" is what a parcel-level state *means*, and it is the map the
foundation model implicitly applies every time it treats a region variable as
describing that region.  Declaring the map the artifact already uses is the
point of the exercise.

Boundary consistency, per §4.2
------------------------------
Not "does ``P R x`` look like ``x``" -- that question is answered ~1 by
construction and answers nothing.  The declared test is whether the coarse view
preserves what crosses the boundary:

* :func:`observable_error` -- ``|| G P R x - G x ||`` against ``|| G x ||``,
  whitened by a measured sensor noise covariance so the answer is in noise
  standard deviations.  ``G`` is a real BEM lead field; ``x`` are real source
  estimates from real evoked EEG.
* :func:`perturbational_error` -- the same quantity for a *focal* perturbation
  ``delta`` at a single fine dipole, i.e. the TMS query §4.2 names.  This is the
  operator norm question and needs no state prior at all.
* :func:`lead_field_energy_retained` -- the prior-free summary: what fraction of
  the whitened lead field's energy lies inside ``row(R)``.  A restriction whose
  row space misses the lead field cannot carry the observable no matter what
  state you put on it.

Authority policy
----------------
:data:`AUTHORITY_POLICY` is ``"fine_authoritative"``, the first of §4.2's three.
That choice is *forced by the measurement*, not assumed; see
``reports/transforms/resolution_pair.md``.

Measured artefact
-----------------
:func:`load_measurement` reads ``reports/transforms/resolution_pair.json``,
which ``benchmarks/transforms/resolution_pair.py`` writes.  Nothing here
hard-codes a residual or a coverage.  If the artefact is absent, stale, or
disagrees with the supports it claims to describe, :func:`load_measurement`
returns ``None`` and the caller must declare the pair *untested* -- at which
point R02 refuses it.  That is deliberate: the guard is armed by default and
only a real measurement disarms it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from .errors import SiteError
from .se3 import DTYPE, as_tensor
from .sheaf import (
    Prolongation,
    Restriction,
    ScalePair,
    measure_coverage,
)

__all__ = [
    "SCALE_FINE",
    "SCALE_COARSE",
    "AUTHORITY_POLICY",
    "MEASUREMENT_RELPATH",
    "SCHEMA_VERSION",
    "restriction_matrix",
    "prolongation_matrix",
    "membership_digest",
    "build_scale_pair",
    "observable_error",
    "perturbational_error",
    "lead_field_energy_retained",
    "BoundaryConsistency",
    "PairMeasurement",
    "load_measurement",
    "measurement_path",
]

#: The two poset elements.  ``SCALE_FINE <= SCALE_COARSE`` in the thesis
#: convention (``a <= b`` means ``a`` is the finer, more informative support).
SCALE_FINE = "cortical_source_dipole"
SCALE_COARSE = "parcel"

#: One of §4.2's three.  See the module docstring and the report.
AUTHORITY_POLICY = "fine_authoritative"

#: Where the measured artefact lives, relative to the repository root.
MEASUREMENT_RELPATH = "reports/transforms/resolution_pair.json"

#: Bumped whenever the meaning of a recorded field changes, so a stale artefact
#: is rejected rather than silently reinterpreted.
SCHEMA_VERSION = 1


# ==========================================================================
# the operators
# ==========================================================================
def _check_membership(
    assign: Any, areas: Any, n_coarse: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.as_tensor(assign, dtype=torch.int64).reshape(-1)
    w = torch.as_tensor(areas, dtype=DTYPE).reshape(-1)
    if a.numel() != w.numel():
        raise SiteError(
            f"membership has {a.numel()} entries but {w.numel()} areas",
            remedy="One area per fine element; areas are the restriction weights.",
            offending_object=(a.numel(), w.numel()),
        )
    if int(a.max()) >= int(n_coarse):
        raise SiteError(
            f"membership references coarse element {int(a.max())} but only "
            f"{n_coarse} are declared",
            remedy="Declare every coarse element the membership uses.",
            offending_object=int(a.max()),
        )
    if bool((w <= 0).any()):
        raise SiteError(
            "a fine element has non-positive area, so its restriction weight is "
            "undefined",
            remedy=(
                "An element with no support cannot contribute a weighted mean. "
                "Drop it from the fine support or supply its real area."
            ),
            offending_object=int(torch.argmin(w)),
        )
    empty = [p for p in range(int(n_coarse)) if not bool((a == p).any())]
    if empty:
        raise SiteError(
            f"{len(empty)} coarse element(s) own no fine element (e.g. {empty[:5]}); "
            "their restriction row would be all zeros",
            remedy=(
                "A parcel with no source is not a coarse state, it is a name. "
                "Drop it from the coarse support."
            ),
            offending_object=empty[:5],
        )
    return a, w


def restriction_matrix(assign: Any, areas: Any, n_coarse: int) -> torch.Tensor:
    """``R`` -- the area-weighted parcel mean, shape ``(n_coarse, n_fine)``.

    ``assign[v]`` is the coarse element owning fine element ``v``, or ``-1`` for
    a fine element no coarse element represents (the medial wall under a
    cortical parcellation).  Unassigned elements get weight zero in every row:
    they are *not* silently folded into a neighbour, and the fraction of the
    fine support they take is reported as the prolongation's coverage deficit.
    """
    a, w = _check_membership(assign, areas, n_coarse)
    R = torch.zeros((int(n_coarse), a.numel()), dtype=DTYPE)
    for p in range(int(n_coarse)):
        m = a == p
        R[p, m] = w[m] / w[m].sum()
    return R


def prolongation_matrix(assign: Any, n_coarse: int) -> torch.Tensor:
    """``P`` -- the indicator fill, shape ``(n_fine, n_coarse)``.

    Unassigned fine elements receive zero from every coarse element, which is
    the honest statement that the coarse state says nothing about them.
    """
    a = torch.as_tensor(assign, dtype=torch.int64).reshape(-1)
    P = torch.zeros((a.numel(), int(n_coarse)), dtype=DTYPE)
    for p in range(int(n_coarse)):
        P[a == p, p] = 1.0
    return P


def membership_digest(assign: Any, areas: Any, n_coarse: int) -> str:
    """Stable digest of the supports, so a stale measurement is detectable."""
    import hashlib

    a = torch.as_tensor(assign, dtype=torch.int64).reshape(-1).numpy()
    w = torch.as_tensor(areas, dtype=DTYPE).reshape(-1).numpy()
    h = hashlib.sha256()
    h.update(f"{SCHEMA_VERSION}|{int(n_coarse)}|{a.size}|".encode())
    h.update(a.tobytes())
    h.update(w.round(12).tobytes())
    return h.hexdigest()[:32]


def assigned_area_fraction(assign: Any, areas: Any) -> float:
    """Fraction of the fine support's measure that a coarse element represents.

    This is the ``landmark_coverage`` of the schema's
    :class:`~scwbd.schema.poset.ScaleMapPair`: the share of the prolongation's
    codomain the coarse state actually reaches.
    """
    a = torch.as_tensor(assign, dtype=torch.int64).reshape(-1)
    w = torch.as_tensor(areas, dtype=DTYPE).reshape(-1)
    return float(w[a >= 0].sum() / w.sum())


# ==========================================================================
# boundary consistency (§4.2)
# ==========================================================================
def _whiten(whitener: Any | None, M: torch.Tensor) -> torch.Tensor:
    if whitener is None:
        return M
    return as_tensor(whitener) @ M


def observable_error(
    G: Any, R: Any, P: Any, fine_states: Any, *, whitener: Any | None = None
) -> dict[str, float]:
    """``|| G P R x - G x ||`` vs ``|| G x ||`` on given fine states.

    ``fine_states`` is ``(n_fine, n_samples)``.  Returned in three currencies:
    the scale-free relative error, the whitened residual per channel (in noise
    standard deviations, if a whitener is supplied) and the whitened signal per
    channel, so the reader can see whether the coarsening error is small
    *relative to what the instrument can resolve* rather than merely small.
    """
    G_, R_, P_ = as_tensor(G), as_tensor(R), as_tensor(P)
    X = torch.as_tensor(fine_states, dtype=DTYPE)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    fine = G_ @ X
    coarse = (G_ @ P_) @ (R_ @ X)
    res = coarse - fine
    wr, wf = _whiten(whitener, res), _whiten(whitener, fine)
    n_ch = wr.shape[0]
    per_t_res = torch.linalg.norm(wr, dim=0) / math.sqrt(n_ch)
    per_t_sig = torch.linalg.norm(wf, dim=0) / math.sqrt(n_ch)
    return {
        "relative_error": float(torch.linalg.norm(wr) / torch.linalg.norm(wf)),
        "unwhitened_relative_error": float(
            torch.linalg.norm(res) / torch.linalg.norm(fine)
        ),
        "residual_sd_per_channel_mean": float(per_t_res.mean()),
        "residual_sd_per_channel_max": float(per_t_res.max()),
        "signal_sd_per_channel_mean": float(per_t_sig.mean()),
        "signal_sd_per_channel_max": float(per_t_sig.max()),
        "n_samples": int(X.shape[1]),
    }


def perturbational_error(
    G: Any, R: Any, P: Any, *, whitener: Any | None = None
) -> dict[str, float]:
    """Per-fine-element focal perturbation error -- §4.2's other half.

    For a unit perturbation at fine element ``v`` the model's predicted sensor
    change is ``G e_v``; the coarse view predicts ``G P R e_v``.  The relative
    topography error ``|| G (P R - I) e_v || / || G e_v ||`` needs no prior over
    states, so it cannot be flattered by choosing a convenient one.  ``1.0``
    means the coarse view predicts a response orthogonal in magnitude to the
    true one; ``0`` means it predicts it exactly.
    """
    G_, R_, P_ = as_tensor(G), as_tensor(R), as_tensor(P)
    WG = _whiten(whitener, G_)
    WD = _whiten(whitener, (G_ @ P_) @ R_) - WG
    num = torch.linalg.norm(WD, dim=0)
    den = torch.linalg.norm(WG, dim=0)
    rel = num / torch.clamp(den, min=1e-300)
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=DTYPE)
    p10, p25, med, p75, p90 = (float(x) for x in torch.quantile(rel, q))
    return {
        "median": med,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "fraction_below_half": float((rel < 0.5).to(DTYPE).mean()),
        "n_fine": int(rel.numel()),
    }


def lead_field_energy_retained(
    G: Any, R: Any, P: Any, areas: Any, *, whitener: Any | None = None
) -> float:
    """Share of the observable the coarse support can carry, over one canonical
    prior instead of one experiment.

    "Prior-free" is not available here and saying otherwise would be a cheat: a
    lead field row is a *functional* on fine states, and functionals have no
    norm until a prior on states supplies one.  The prior fixed here is the only
    one with no free parameters -- ``x ~ N(0, A^-1)`` with ``A = diag(areas)``,
    i.e. white per unit cortical area, no length scale, no orientation
    structure, nothing to tune.  Under it,

        E|| G P R x ||^2 / E|| G x ||^2  =  eta,
        E|| G (P R - I) x ||^2 / E|| G x ||^2  =  1 - eta,

    exactly (the cross term is ``||Pi u||^2_A`` because ``P R`` is the
    ``A``-orthogonal projector, so the decomposition is Pythagorean).  The two
    halves therefore add up, which is what makes ``eta`` comparable with the
    per-ensemble relative errors from :func:`observable_error` rather than a
    second unrelated number.

    ``1.0`` means the coarse support can represent every sensor-visible
    direction.  The complement is the fraction of the observable that *no* state
    on the coarse support can ever produce, whatever the dynamics do.
    """
    G_, R_, P_ = as_tensor(G), as_tensor(R), as_tensor(P)
    a = torch.as_tensor(areas, dtype=DTYPE).reshape(-1)
    U = _whiten(whitener, G_) / a  # rows: the functionals, in state coordinates
    PiU = (U @ R_.T) @ P_.T
    num = float((a * PiU**2).sum())
    den = float((a * U**2).sum())
    return num / den if den > 0 else 0.0


# ==========================================================================
# the pair, and the record that licenses it
# ==========================================================================
def build_scale_pair(
    assign: Any,
    areas: Any,
    n_coarse: int,
    landmark_fine_states: Any,
    *,
    prior_sd_unresolved: float | None = None,
    version: str = "1.0.0",
) -> ScalePair:
    """Assemble the :class:`~scwbd.transforms.sheaf.ScalePair`.

    ``landmark_fine_states`` are the *held-out* fine states the coverage test
    runs on.  ``prior_sd_unresolved`` is the standard deviation the prolongation
    declares for the ``n_fine - n_coarse`` directions ``R`` cannot see; the
    caller is expected to have estimated it on a disjoint split, so that
    ``max_heldout_error`` -- pinned to that same value -- is a genuine held-out
    test.  :class:`~scwbd.transforms.sheaf.Prolongation` then raises R02 when
    the coverage test finds a larger residual than the prolongation declares it
    has, i.e. when the map claims more precision than the data supports.

    Passing ``None`` estimates the sd from the same states it is then tested on.
    That is a self-test, not a held-out test, and callers who do it must say so.
    """
    R = restriction_matrix(assign, areas, n_coarse)
    P = prolongation_matrix(assign, n_coarse)
    n_fine = R.shape[1]
    rho = Restriction(
        source=SCALE_FINE,
        target=SCALE_COARSE,
        matrix=R,
        version=version,
        method="area_weighted_parcel_mean",
        notes=(
            "(R x)_p = sum_{v in p} a_v x_v / sum_{v in p} a_v; a_v is the "
            "white-surface patch area of source v. Fine elements owned by no "
            "coarse element carry weight zero in every row."
        ),
    )
    X = torch.as_tensor(landmark_fine_states, dtype=DTYPE)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != n_fine:
        X = X.T
    coverage = measure_coverage(
        rho,
        P,
        X,
        domain={
            "fine": SCALE_FINE,
            "coarse": SCALE_COARSE,
            "n_fine": n_fine,
            "n_coarse": int(n_coarse),
            "note": (
                "held-out landmarks are fine states from experimental "
                "conditions disjoint from those used to set prior_sd_unresolved"
            ),
        },
    )
    sd = (
        float(coverage.heldout_error)
        if prior_sd_unresolved is None
        else float(prior_sd_unresolved)
    )
    if not (sd > 0.0 and math.isfinite(sd)):
        raise SiteError(
            f"prior sd for the unresolved directions is {sd}, which cannot be "
            "declared",
            remedy="Measure P R x - x on real fine states.",
            offending_object=sd,
        )
    prolong = Prolongation(
        restriction=rho,
        matrix=P,
        coverage=coverage,
        prior_sd_unresolved=sd,
        prior_sd_resolved=0.0,
        version=version,
        method="indicator_fill",
        max_heldout_error=sd,
    )
    return ScalePair(restriction=rho, prolongation=prolong)


@dataclass(frozen=True)
class BoundaryConsistency:
    """The §4.2 verdict for one fine-state ensemble."""

    ensemble: str
    description: str
    observable: dict[str, float]
    #: The pre-registered bound the residual is compared against, and its basis.
    tolerance_sd_per_channel: float
    tolerance_basis: str

    @property
    def passes(self) -> bool:
        return (
            self.observable["residual_sd_per_channel_mean"]
            <= self.tolerance_sd_per_channel
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "ensemble": self.ensemble,
            "description": self.description,
            "observable": dict(self.observable),
            "tolerance_sd_per_channel": self.tolerance_sd_per_channel,
            "tolerance_basis": self.tolerance_basis,
            "passes": self.passes,
        }


@dataclass(frozen=True)
class PairMeasurement:
    """Everything a declaration of the pair is allowed to assert.

    Every field is produced by ``benchmarks/transforms/resolution_pair.py`` from
    the data on disk.  Nothing in the library may supply a default for any of
    the residuals: a missing measurement is an untested pair, and R02 exists to
    say so.
    """

    schema_version: int
    n_fine: int
    n_coarse: int
    membership_digest: str
    authority_policy: str
    #: ``max |R P - I|`` on the coarse support.  The pair is paired or it is not.
    coarse_roundtrip_residual: float
    coarse_roundtrip_tolerance: float
    #: RMS of ``P R x - x`` on held-out fine states, and the sd the prolongation
    #: declares for its unresolved directions.  Calibrated iff the first <= the
    #: second.
    heldout_fine_residual: float
    declared_prior_sd_unresolved: float
    #: Share of the fine support's measure represented by some coarse element.
    landmark_coverage: float
    required_coverage: float
    #: Prior-free: share of the whitened lead field inside ``row(R)``.
    lead_field_energy_retained: float
    #: sqrt(mean element area), metres -- the nominal size of one element of
    #: each support, so the poset node carries a measured scale rather than a
    #: guessed one.
    fine_characteristic_scale_m: float
    coarse_characteristic_scale_m: float
    #: The §4.2 boundary numbers.
    boundary: tuple[dict[str, Any], ...]
    perturbational: dict[str, float]
    #: The first estimator tried for ``declared_prior_sd_unresolved``: the plain
    #: RMS of ``P R x - x`` on the training split, used directly as the
    #: admissibility bound.  R02 refused it on the held-out split, by 0.7%.
    #: Recorded rather than deleted: a point estimate is the wrong *kind* of
    #: object to use as a bound, and the fix was to change the estimator, not
    #: the threshold.  See ``reports/transforms/resolution_pair.md`` §5.
    rejected_point_estimate_prior_sd: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- the four things a declaration may claim ---------------------------
    @property
    def roundtrip_ok(self) -> bool:
        return self.coarse_roundtrip_residual <= self.coarse_roundtrip_tolerance

    @property
    def prolongation_calibrated(self) -> bool:
        """Does P declare at least as much uncertainty as it actually has?"""
        return self.heldout_fine_residual <= self.declared_prior_sd_unresolved

    @property
    def coverage_ok(self) -> bool:
        return self.landmark_coverage >= self.required_coverage

    @property
    def boundary_sufficient(self) -> bool:
        """§4.2's question.  Deliberately *not* required for admissibility."""
        return bool(self.boundary) and all(b["passes"] for b in self.boundary)

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["boundary"] = [dict(b) for b in self.boundary]
        d["derived"] = {
            "roundtrip_ok": self.roundtrip_ok,
            "prolongation_calibrated": self.prolongation_calibrated,
            "coverage_ok": self.coverage_ok,
            "boundary_sufficient": self.boundary_sufficient,
        }
        return d

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> "PairMeasurement":
        kw = {k: rec[k] for k in cls.__dataclass_fields__ if k in rec}
        kw["boundary"] = tuple(dict(b) for b in kw.get("boundary", ()))
        kw["perturbational"] = dict(kw.get("perturbational", {}))
        kw["provenance"] = dict(kw.get("provenance", {}))
        return cls(**kw)  # type: ignore[arg-type]


def measurement_path(root: str | Path | None = None) -> Path:
    """Absolute path to the measured artefact."""
    if root is not None:
        return Path(root) / MEASUREMENT_RELPATH
    # scwbd/transforms/resolution_pair.py -> repo root
    return Path(__file__).resolve().parents[2] / MEASUREMENT_RELPATH


def load_measurement(
    path: str | Path | None = None,
    *,
    expect_digest: str | None = None,
    expect_n_fine: int | None = None,
    expect_n_coarse: int | None = None,
) -> PairMeasurement | None:
    """Read the measured artefact, or ``None`` if it is absent or stale.

    ``None`` is not an error and is not a default: it is the input that makes a
    caller declare the pair untested, which is what R02 refuses.  A silent
    fallback to nominal residuals here would convert the entire guard into
    decoration, which is the failure mode ``reports/decorative_guards.md``
    catalogues.
    """
    p = Path(path) if path is not None else measurement_path()
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, Mapping) or rec.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        m = PairMeasurement.from_record(rec)
    except TypeError:
        return None
    if expect_digest is not None and m.membership_digest != expect_digest:
        return None
    if expect_n_fine is not None and m.n_fine != int(expect_n_fine):
        return None
    if expect_n_coarse is not None and m.n_coarse != int(expect_n_coarse):
        return None
    if m.authority_policy != AUTHORITY_POLICY:
        return None
    return m


def save_measurement(m: PairMeasurement, path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else measurement_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m.as_record(), indent=2, sort_keys=False) + "\n")
    return p
