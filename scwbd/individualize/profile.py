"""What this patient's data can and cannot personalise -- computed *before* fitting.

The clinical question is not "did the fit converge?" but "is there anything in
this patient's data that could have moved this parameter?".  That is answered by
the expected Fisher information of the design the patient's modalities imply,
which needs no patient data at all -- only the declaration of what was measured.
So it can be reported to a clinician *before* anyone runs an optimiser, which is
the whole point.

Statistic
---------
For a parameter group ``g`` we report the minimum eigenvalue of the Schur
complement of the **likelihood-only** expected information on ``g``, all other
parameters profiled out, in the prior-standardised basis:

    I_g|rest = I_gg - I_g,rest I_rest,rest^+ I_rest,g ,   lambda_min(I_g|rest)

This is the same statistic agent Fisher's benchmark uses for ``theta``
(``theta_profile_min_eigenvalue_nonprior``), applied per group.  In the
prior-standardised basis ``I_prior`` is exactly the identity, so ``lambda_min``
is in units of *prior precision*: ``lambda_min = 1`` means the data carry as
much information about the group's worst-determined direction as the prior does,
and the posterior sd of that direction shrinks to ``1/sqrt(1+lambda) = 71%`` of
the prior sd.  ``lambda_min = 1e-3`` shrinks it to ``99.95%`` of the prior sd --
a "posterior" that is the prior with a rounding error.  The thresholds in
:class:`IdentifiabilityThresholds` are exactly that reading and nothing more.

Which statistic decides
-----------------------
Two variants are computed for every group and both are reported:

``lambda_min`` (**decides the status**)
    likelihood-only, other parameters profiled out under an improper flat
    prior.  Identical to the quantity in agent Fisher's committed benchmark,
    so the two are directly comparable, and it is the *conservative* one.
``lambda_min_with_nuisance_prior``
    the same Schur complement with the prior precision added on the
    profiled-out coordinates.  Algebraically this is
    ``Schur(I_like + I_prior, g) - I_prior``: the posterior precision of group
    ``g`` after marginalising the other parameters *under their priors*, minus
    the prior's own.  It is the estimator that actually runs, and it is never
    smaller than the likelihood-only variant.

The status is decided by the conservative one on purpose.  The two errors are
not symmetric: being too strict costs an individualization we might have
managed, being too lax hands a clinician a population number wearing a
patient's name.  Where the two disagree the report shows both, so the choice is
visible rather than buried.

Numerics
--------
Under precisely the single-modality configurations this module exists for, the
profiled-out block is **exactly singular**: EEG-only data give the three
haemodynamic parameters identically zero information, fMRI-only data do the same
to the two EEG lead-field parameters.
:func:`profiled_information` therefore takes the pseudo-inverse through a
symmetric eigendecomposition with an explicit relative cutoff, which is the
correct limit -- profiling out a direction the data say nothing about removes
nothing -- and it **records** the retained rank, the dropped count and the
conditioning, so a caller can see that a profile was taken over a rank-deficient
block rather than infer it.  ``scwbd.infer.fisher.schur_information`` was checked
against this on every design and group used here and agrees throughout
(``tests/individualize/test_numerics.py``); the local version is kept for the
diagnostics and for the nuisance-prior variant, not because the shipped one was
found wrong.

The hazard that *was* found is elsewhere and is guarded by
:func:`assert_delay_line_adequate` -- see its docstring for the measured
twenty-five-order-of-magnitude failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scwbd.infer.linear_gaussian import PARAM_INDEX, PARAM_NAMES, SystemConfig

from .availability import Modality, ModalityAvailability
from .groups import GROUPS, LIKELIHOOD_GROUPS, ParameterGroup

__all__ = [
    "IdentifiabilityProfile",
    "IdentifiabilityThresholds",
    "GroupIdentifiability",
    "InadequateDelayLine",
    "Status",
    "assert_delay_line_adequate",
    "benchmark_config",
    "profile_identifiability",
    "profiled_information",
]


class InadequateDelayLine(ValueError):
    """The configuration's delay line is too short to represent the delay."""


def assert_delay_line_adequate(cfg: SystemConfig, eta: np.ndarray) -> None:
    """Refuse a configuration whose delay line cannot represent ``tau``.

    The fractional-delay kernel is a Gaussian-windowed sinc over taps
    ``p = 0..D`` evaluated at ``x = tau/dt - p``, then **normalised by its own
    sum**.  If ``D`` is smaller than ``tau/dt`` the peak at ``x = 0`` is never
    reached, the raw weights are all in the far tail, and the normalisation
    divides by a number near zero.  Nothing raises.  What comes out is a
    finite, plausible-looking information matrix with entries inflated by
    twenty orders of magnitude: measured on a ``n_delay_taps=10`` configuration
    at ``tau = 12 ms``, the conduction-delay profile read ``1.9e+25`` -- a
    number that would have been reported as spectacular identifiability.

    Measured on ``epoch_seconds=1.5, n_epochs=2, dt=1 ms, tau=12 ms``, EEG-only,
    reference regime::

        n_delay_taps = 10   I[tau, tau] = 1.78932e+25   profile = 1.33199e+25
        n_delay_taps = 26   I[tau, tau] = 2.21145       profile = 2.00851

    Twenty-five orders of magnitude, no exception raised, and the inflated
    reading is the one that says "spectacularly identifiable".

    ``D = 0`` is permitted: that is the deliberate "no delay line at all"
    configuration of the naive-resampling control, where ``d mu / d tau == 0``
    exactly and the degeneracy is visible rather than disguised.
    """
    from scwbd.infer.linear_gaussian import natural_from_unconstrained

    D = int(cfg.n_delay_taps)
    if D == 0:
        return
    tau = float(natural_from_unconstrained(np.asarray(eta, dtype=float))["tau"])
    need = tau / cfg.dt + 3.0 * cfg.sinc_sigma
    if D < need:
        raise InadequateDelayLine(
            f"n_delay_taps={D} cannot represent tau={tau:.6g} s at dt={cfg.dt:g} s: "
            f"the windowed-sinc kernel needs at least tau/dt + 3*sinc_sigma = "
            f"{need:.1f} taps, otherwise its normalisation divides by the far "
            "tail and the Fisher information is inflated by many orders of "
            "magnitude without any error being raised."
        )


Status = str  # "identifiable" | "weakly_identifiable" | "not_identifiable"


def benchmark_config(**overrides: Any) -> SystemConfig:
    """The configuration of the **committed** identifiability benchmark.

    Read off ``reports/identifiability/manifest.json`` (``extra.command``):
    ``epoch_seconds=3.0``, ``n_epochs=30``, ``dtype="float64"``, all other
    fields at their ``SystemConfig`` defaults.  Device is forced to CPU here --
    a 12 h training run owns the GPU.  Reproducing the committed
    ``theta_profile_min_eigenvalue_nonprior`` numbers under this config is
    ``tests/individualize/test_matches_fisher_benchmark.py``.
    """
    base: dict[str, Any] = dict(
        device="cpu", dtype="float64", epoch_seconds=3.0, n_epochs=30
    )
    base.update(overrides)
    return SystemConfig(**base)


@dataclass(frozen=True)
class IdentifiabilityThresholds:
    """Where "identifiable" stops.  Declared once, in prior-precision units.

    ``lambda_min`` is likelihood information in units of prior precision, so the
    posterior sd of the worst-determined direction in the group is
    ``1/sqrt(1 + lambda_min)`` times the prior sd.  The two thresholds are:

    ``identifiable = 1.0``
        the data are worth at least as much as the prior; posterior sd <= 71% of
        prior sd.
    ``weak = 1e-3``
        posterior sd <= 99.95% of prior sd.  Below this the data have moved
        nothing a clinician could act on and the honest answer is "the prior".
    """

    identifiable: float = 1.0
    weak: float = 1.0e-3

    def __post_init__(self) -> None:
        if not (0.0 < self.weak < self.identifiable):
            raise ValueError("require 0 < weak < identifiable")

    def classify(self, lambda_min: float) -> Status:
        if not np.isfinite(lambda_min):
            return "not_identifiable"
        if lambda_min >= self.identifiable:
            return "identifiable"
        if lambda_min >= self.weak:
            return "weakly_identifiable"
        return "not_identifiable"

    def to_dict(self) -> dict[str, float]:
        return {"identifiable": self.identifiable, "weak": self.weak}


def sd_ratio(lambda_min: float) -> float:
    """Posterior/prior sd of the worst-determined direction, Gaussian/Laplace."""
    return float(1.0 / np.sqrt(1.0 + max(lambda_min, 0.0)))


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------
def profiled_information(
    I: np.ndarray,
    keep: Sequence[int],
    *,
    nuisance_prior: float = 0.0,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Schur complement on ``keep``, robust to an exactly singular nuisance block.

    ``nuisance_prior=0`` reproduces the likelihood-only statistic of the
    committed benchmark.  ``nuisance_prior=1`` adds the (identity) prior
    precision on the profiled-out coordinates, which is the practically relevant
    variant: we do have priors on lead-field gain and HRF shape, and profiling
    them out under an improper flat prior is more pessimistic than the situation
    warrants.  Both are reported.
    """
    I = 0.5 * (np.asarray(I, dtype=float) + np.asarray(I, dtype=float).T)
    keep = list(keep)
    rest = [i for i in range(I.shape[0]) if i not in keep]
    Ikk = I[np.ix_(keep, keep)]
    diag: dict[str, Any] = {
        "n_keep": len(keep),
        "n_profiled": len(rest),
        "nuisance_prior": nuisance_prior,
    }
    if not rest:
        diag.update(nuisance_rank=0, nuisance_dropped=0, nuisance_condition=None)
        return Ikk, diag
    Inn = I[np.ix_(rest, rest)] + nuisance_prior * np.eye(len(rest))
    Ikn = I[np.ix_(keep, rest)]
    w, V = np.linalg.eigh(0.5 * (Inn + Inn.T))
    wmax = float(w.max()) if w.size else 0.0
    cut = rcond * max(wmax, 0.0)
    good = w > cut
    diag["nuisance_rank"] = int(good.sum())
    diag["nuisance_dropped"] = int((~good).sum())
    diag["nuisance_condition"] = (
        float(wmax / w[good].min()) if good.any() else float("inf")
    )
    if good.any():
        Vg = V[:, good]
        Pinv = (Vg / w[good]) @ Vg.T
        S = Ikk - Ikn @ (Pinv @ Ikn.T)
    else:
        # the data say nothing about any nuisance direction: profiling removes
        # nothing, and the profile equals the block itself.
        S = Ikk
    S = 0.5 * (S + S.T)
    diag["leaked_negative_eigenvalue"] = float(
        min(0.0, float(np.linalg.eigvalsh(S).min()))
    )
    return S, diag


def _min_eig(M: np.ndarray) -> float:
    if M.size == 0:
        return float("nan")
    return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupIdentifiability:
    """Status of one parameter group under one modality availability."""

    group: str
    kind: str
    evidence_kind: str
    status: Status
    parameters: tuple[str, ...]
    #: ``None`` for presence-determined (anatomical) groups: there is no Fisher
    #: number for them, and inventing one would be the decorative-guard failure.
    lambda_min: float | None
    lambda_min_with_nuisance_prior: float | None
    per_parameter_information: Mapping[str, float]
    posterior_sd_ratio: float | None
    reason: str
    clinical_meaning: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_identifiable(self) -> bool:
        return self.status == "identifiable"

    @property
    def may_be_individualized(self) -> bool:
        """Weakly identifiable groups may be fitted, and are labelled weak."""
        return self.status in ("identifiable", "weakly_identifiable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "kind": self.kind,
            "evidence_kind": self.evidence_kind,
            "status": self.status,
            "parameters": list(self.parameters),
            "lambda_min_likelihood": self.lambda_min,
            "lambda_min_with_nuisance_prior": self.lambda_min_with_nuisance_prior,
            "per_parameter_information": dict(self.per_parameter_information),
            "posterior_sd_ratio": self.posterior_sd_ratio,
            "reason": self.reason,
            "clinical_meaning": self.clinical_meaning,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class IdentifiabilityProfile:
    """Per-group identifiability for one patient's modality availability."""

    patient_id: str
    availability_digest: str
    present: tuple[Modality, ...]
    design: str
    channels: tuple[str, ...]
    groups: Mapping[str, GroupIdentifiability]
    thresholds: IdentifiabilityThresholds
    basis: str
    regime: str
    config: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: ``modality added -> {group: status}``, measured by re-profiling.  Empty
    #: unless the caller asked for counterfactuals.
    counterfactuals: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __getitem__(self, group: str) -> GroupIdentifiability:
        if group not in self.groups:
            raise KeyError(
                f"group {group!r} not in profile; known: {sorted(self.groups)}"
            )
        return self.groups[group]

    def status(self, group: str) -> Status:
        return self[group].status

    @property
    def identifiable(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.groups.items() if v.status == "identifiable")

    @property
    def weakly_identifiable(self) -> tuple[str, ...]:
        return tuple(
            k for k, v in self.groups.items() if v.status == "weakly_identifiable"
        )

    @property
    def not_identifiable(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.groups.items() if v.status == "not_identifiable")

    def fittable_parameters(self) -> tuple[str, ...]:
        """Model parameters the data may move.  Everything else stays at prior."""
        out: list[str] = []
        for g in LIKELIHOOD_GROUPS:
            gi = self.groups.get(g.name)
            if gi is not None and gi.may_be_individualized:
                out.extend(g.parameters)
        return tuple(p for p in PARAM_NAMES if p in set(out))

    def fit_mask(self) -> np.ndarray:
        """Boolean mask over ``eta``: True where the data may move a coordinate."""
        m = np.zeros(len(PARAM_NAMES), dtype=bool)
        for p in self.fittable_parameters():
            m[PARAM_INDEX[p]] = True
        return m

    def remedies(self, group: str) -> tuple[dict[str, Any], ...]:
        """Measured counterfactuals that would make ``group`` fittable at all.

        A remedy is a modality whose addition moves the group from
        ``not_identifiable`` to ``identifiable`` **or** ``weakly_identifiable``.
        The status it would reach is carried through, so "weakly" is never
        quoted as "identifiable"; ranking is by the measured ``lambda_min``.
        """
        here = self.groups.get(group)
        if here is not None and here.status != "not_identifiable":
            return ()
        out = []
        for mod, res in self.counterfactuals.items():
            entry = res.get(group)
            if entry and entry.get("status") in ("identifiable", "weakly_identifiable"):
                out.append(
                    {
                        "add_modality": mod,
                        "status": entry["status"],
                        "lambda_min": entry.get("lambda_min"),
                    }
                )
        return tuple(sorted(out, key=lambda d: -(d.get("lambda_min") or 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "availability_digest": self.availability_digest,
            "present": list(self.present),
            "design": self.design,
            "channels": list(self.channels),
            "basis": self.basis,
            "regime": self.regime,
            "config": dict(self.config),
            "thresholds": self.thresholds.to_dict(),
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
            "identifiable": list(self.identifiable),
            "weakly_identifiable": list(self.weakly_identifiable),
            "not_identifiable": list(self.not_identifiable),
            "fittable_parameters": list(self.fittable_parameters()),
            "counterfactuals": {k: dict(v) for k, v in self.counterfactuals.items()},
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
        }

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True, **kw)


# --------------------------------------------------------------------------
# the computation
# --------------------------------------------------------------------------
#: Content-keyed cache of ``(design, regime, cfg, eta, seed) -> I_likelihood``.
#: Distinct modality combinations frequently resolve to the *same* design --
#: MRI-only, dMRI-only, behaviour-only and no-data-at-all all resolve to
#: ``prior``; MEG resolves to the EEG channel -- so a report over eight
#: availabilities runs four Fisher computations, not eight.  Keyed on content,
#: never on identity, so a changed configuration can never hit a stale entry.
_FISHER_CACHE: dict[tuple, tuple[np.ndarray, dict[str, Any]]] = {}


def _cache_key(design, cfg, regime, eta, seed) -> tuple:
    return (
        design,
        getattr(regime, "name", str(regime)),
        tuple(sorted((k, repr(v)) for k, v in vars(cfg).items())),
        None if eta is None else tuple(np.asarray(eta, dtype=float).ravel().tolist()),
        int(seed),
    )


def clear_fisher_cache() -> None:
    _FISHER_CACHE.clear()


def _fisher_for_design(
    design: str,
    *,
    cfg: SystemConfig,
    regime: Any,
    eta: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``I_likelihood`` in the prior-standardised basis for one design."""
    from scwbd.infer.fisher import expected_fisher
    from scwbd.infer.identifiability import DESIGNS, build_design

    key = _cache_key(design, cfg, regime, eta, seed)
    hit = _FISHER_CACHE.get(key)
    if hit is not None:
        I, prov = hit
        return I.copy(), {**prov, "cache": "hit"}

    if design == "prior":
        # No channel the reference likelihood can read.  The information is
        # *identically* the zero matrix -- there is no record to run a filter
        # over, and fabricating one (a zero-filled fMRI series, say) is exactly
        # the imputation availability.py refuses.  Nothing is computed because
        # there is nothing to compute, and that fact is recorded.
        n = len(PARAM_NAMES)
        out = (
            np.zeros((n, n)),
            {
                "computed": "structural_zero",
                "why": (
                    "design 'prior': the patient has no modality the reference "
                    "likelihood reads, so I_likelihood is identically zero. No "
                    "record was synthesised."
                ),
            },
        )
        _FISHER_CACHE[key] = out
        return out[0].copy(), dict(out[1])
    spec = next((d for d in DESIGNS if d.name == design), None)
    if spec is None:
        raise KeyError(f"unknown design {design!r}")
    assert_delay_line_adequate(cfg, eta if eta is not None else regime.eta_true())
    bd = build_design(spec, cfg, regime, seed=seed)
    rep = expected_fisher(
        eta if eta is not None else regime.eta_true(),
        bd.cfg,
        bd.proto,
        design=design,
        channels=bd.channels,
        include_impulse=bd.include_impulse,
        eeg_steps=bd.eeg_steps,
        method="analytic",
        standardised=True,
        device="cpu",
    )
    out = (
        np.asarray(rep.I_likelihood, dtype=float),
        {
            "computed": "expected_fisher(method='analytic', standardised=True)",
            "sigma_eeg": bd.cfg.sigma_eeg,
            "sigma_bold": bd.cfg.sigma_bold,
            "n_eeg_samples_used": rep.metrics.get("n_eeg_samples_used"),
            "n_bold_samples_used": rep.metrics.get("n_bold_samples_used"),
            "notes": list(bd.notes),
        },
    )
    _FISHER_CACHE[key] = out
    return out[0].copy(), dict(out[1])


def _anatomical_status(
    group: ParameterGroup, availability: ModalityAvailability
) -> GroupIdentifiability:
    have = [m for m in group.informed_by if m in availability]
    if have:
        status = "identifiable"
        reason = (
            f"presence-determined: the patient has {have}, from which this group "
            "is individualised directly. NOTE: this status is established by "
            "PRESENCE, not by a Fisher computation -- there is no lambda_min for "
            "it and none is reported."
        )
    else:
        status = "not_identifiable"
        reason = (
            f"the patient has none of {list(group.informed_by)}; this group "
            "stays at the population/template value."
        )
    return GroupIdentifiability(
        group=group.name,
        kind=group.kind,
        evidence_kind=group.evidence_kind,
        status=status,
        parameters=group.parameters,
        lambda_min=None,
        lambda_min_with_nuisance_prior=None,
        per_parameter_information={},
        posterior_sd_ratio=None,
        reason=reason,
        clinical_meaning=group.clinical_meaning,
        diagnostics={"modalities_present": have},
    )


def profile_identifiability(
    availability: ModalityAvailability,
    *,
    cfg: SystemConfig | None = None,
    regime: Any = None,
    eta: np.ndarray | None = None,
    thresholds: IdentifiabilityThresholds | None = None,
    groups: Sequence[ParameterGroup] = GROUPS,
    seed: int = 20260805,
    counterfactual_modalities: Iterable[Modality] = (),
) -> IdentifiabilityProfile:
    """Per-group identifiability for this patient, **before** any fitting.

    Requires no patient data -- only the declaration of what was measured.

    ``counterfactual_modalities`` re-runs the profile with each named modality
    added, so a refusal downstream can say *"an EEG would fix this, measured"*
    instead of asserting it.  Each entry costs one more Fisher computation.
    """
    from scwbd.infer.identifiability import REGIMES

    cfg = cfg or benchmark_config()
    regime = regime if regime is not None else REGIMES[0]
    thresholds = thresholds or IdentifiabilityThresholds()

    I, prov = _fisher_for_design(
        availability.design, cfg=cfg, regime=regime, eta=eta, seed=seed
    )

    out: dict[str, GroupIdentifiability] = {}
    for g in groups:
        if g.kind == "anatomical":
            out[g.name] = _anatomical_status(g, availability)
            continue
        idx = g.index
        S, diag = profiled_information(I, idx, nuisance_prior=0.0)
        Sp, _ = profiled_information(I, idx, nuisance_prior=1.0)
        lam = _min_eig(S)
        lam_p = _min_eig(Sp)
        status = thresholds.classify(lam)
        per_param = {p: float(I[PARAM_INDEX[p], PARAM_INDEX[p]]) for p in g.parameters}
        if status == "identifiable":
            reason = (
                f"lambda_min = {lam:.6g} prior-precision units >= "
                f"{thresholds.identifiable:g}: the patient's data determine every "
                "direction in this group at least as well as the prior does "
                f"(posterior sd <= {sd_ratio(lam):.3f} x prior sd)."
            )
        elif status == "weakly_identifiable":
            reason = (
                f"lambda_min = {lam:.6g} prior-precision units, between "
                f"{thresholds.weak:g} and {thresholds.identifiable:g}: the data "
                "move this group, but the posterior is still prior-dominated "
                f"(posterior sd = {sd_ratio(lam):.4f} x prior sd). Any fitted "
                "value must be reported as weakly identified."
            )
        else:
            if prov.get("computed") == "structural_zero":
                reason = (
                    "no modality the reference likelihood can read: the expected "
                    "information about this group is identically zero. Nothing in "
                    "this patient's data could move these parameters."
                )
            else:
                reason = (
                    f"lambda_min = {lam:.6g} prior-precision units < "
                    f"{thresholds.weak:g}: the available modalities carry "
                    "essentially no information about the worst-determined "
                    "direction of this group (posterior sd = "
                    f"{sd_ratio(lam):.6f} x prior sd -- the prior, to six "
                    "figures). Fitting it would return the prior wearing a "
                    "posterior's label."
                )
        out[g.name] = GroupIdentifiability(
            group=g.name,
            kind=g.kind,
            evidence_kind=g.evidence_kind,
            status=status,
            parameters=g.parameters,
            lambda_min=lam,
            lambda_min_with_nuisance_prior=lam_p,
            per_parameter_information=per_param,
            posterior_sd_ratio=sd_ratio(lam),
            reason=reason,
            clinical_meaning=g.clinical_meaning,
            diagnostics=diag,
        )

    cfs: dict[str, dict[str, Any]] = {}
    for mod in counterfactual_modalities:
        if mod in availability:
            continue
        extended = ModalityAvailability(
            patient_id=availability.patient_id,
            records=availability.records
            + ModalityAvailability.from_modalities(
                availability.patient_id, [mod], prefix="counterfactual"
            ).records,
            group=availability.group,
        )
        sub = profile_identifiability(
            extended,
            cfg=cfg,
            regime=regime,
            eta=eta,
            thresholds=thresholds,
            groups=groups,
            seed=seed,
        )
        cfs[mod] = {
            k: {"status": v.status, "lambda_min": v.lambda_min}
            for k, v in sub.groups.items()
        }

    notes = []
    if availability.uses_meg_as_eeg_proxy:
        notes.append(
            "MEG is profiled through the T2 electrophysiology channel as a "
            "PROXY: the reference slice has no magnetic lead field, so these "
            "numbers are EEG-shaped and must not be quoted as MEG-specific."
        )
    if any(
        not r.fully_objectified for r in availability.records
    ):
        notes.append(
            "at least one modality's source card / support / clock / calibration "
            "is a NAME, not an object: the R01 declarations exist but have not "
            "been resolved to records that could be checked."
        )
    return IdentifiabilityProfile(
        patient_id=availability.patient_id,
        availability_digest=availability.digest(),
        present=availability.present,
        design=availability.design,
        channels=availability.channels,
        groups=out,
        thresholds=thresholds,
        basis="prior_standardised",
        regime=getattr(regime, "name", str(regime)),
        config={
            "epoch_seconds": cfg.epoch_seconds,
            "n_epochs": cfg.n_epochs,
            "dt": cfg.dt,
            "dt_bold": cfg.dt_bold,
            "n_delay_taps": cfg.n_delay_taps,
            "hrf_stages": cfg.hrf_stages,
            "dtype": cfg.dtype,
            "device": cfg.device,
        },
        provenance={
            "statistic": (
                "min eigenvalue of the Schur complement of the LIKELIHOOD-only "
                "expected Fisher information on the group, all other parameters "
                "profiled out, prior-standardised basis"
            ),
            "seed": seed,
            **prov,
        },
        counterfactuals=cfs,
        notes=tuple(notes),
    )
