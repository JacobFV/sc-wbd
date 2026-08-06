"""Stable, thin bindings from ``scwbd.infer`` to the rest of SC-WBD.

Two consumers exist today.

``scwbd.bench.gates`` (agent J)
    Claim gate **G4** -- *perturbation reduces non-identifiability* -- needs a
    ``design -> information`` map rather than a call per design.  It compares
    designs *after removing the prior*, and its key statistic is the minimum
    eigenvalue of the ``theta`` block with the observation nuisances
    ``(ell, rho)`` **profiled out** (Schur complement), because that is how
    "the intervention only added field-model uncertainty" is detected instead
    of assumed.  :func:`fisher_design_map` returns exactly that, with the block
    structure exposed so no consumer has to re-derive the parameter ordering.

``scwbd.schema`` / ``scwbd.compiler`` (agent A)
    :func:`load_reference_schema` resolves ``build_three_region_schema()`` when
    it is importable and returns ``None`` otherwise, so nothing here is ever
    blocked on another module landing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .fisher import FisherReport, expected_fisher, prior_information, schur_information
from .linear_gaussian import PARAMS, PARAM_NAMES, SystemConfig, THETA_NAMES
from .types import as_builtin

__all__ = [
    "DesignInformation",
    "PARAMETER_BLOCKS",
    "design_information_map",
    "fisher_design_information",
    "theta_partition",
    "load_reference_schema",
    "reference_benchmark_config",
]


#: ``eta = (theta, ell, rho)`` -- index blocks, stable across releases.
PARAMETER_BLOCKS: dict[str, list[int]] = {
    "theta": [i for i, p in enumerate(PARAMS) if p.group == "theta"],
    "ell": [i for i, p in enumerate(PARAMS) if p.group == "ell"],
    "rho": [i for i, p in enumerate(PARAMS) if p.group == "rho"],
}
PARAMETER_BLOCK_NAMES: dict[str, list[str]] = {
    k: [PARAM_NAMES[i] for i in v] for k, v in PARAMETER_BLOCKS.items()
}


@dataclass
class DesignInformation:
    """Expected information for one design, with the prior kept separable.

    Attributes
    ----------
    information_likelihood:
        ``sum_m J_m^T R_m^{-1} J_m`` -- **the prior is not included**.  This is
        the matrix G4 must use for its rank/eigenvalue comparison.
    information_prior:
        ``I_prior``.  In the prior-standardised basis this is the identity.
    theta_profile_information:
        Schur complement ``I_tt - I_tn I_nn^{-1} I_nt`` of the *likelihood*
        information on the ``theta`` block, nuisances ``(ell, rho)`` profiled
        out.  ``min(eigvalsh(.))`` is G4's headline statistic.
    """

    name: str
    label: str
    basis: str
    parameter_names: list[str]
    parameter_blocks: dict[str, list[int]]
    information_likelihood: np.ndarray
    information_prior: np.ndarray
    information_by_modality: dict[str, np.ndarray]
    theta_profile_information: np.ndarray
    metrics: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    report: FisherReport | None = None

    # ---- accessors G4 uses directly -------------------------------------
    @property
    def information_total(self) -> np.ndarray:
        return self.information_likelihood + self.information_prior

    @property
    def theta_index(self) -> list[int]:
        return self.parameter_blocks["theta"]

    @property
    def nuisance_index(self) -> list[int]:
        return self.parameter_blocks["ell"] + self.parameter_blocks["rho"]

    def block(self, which: str, *, include_prior: bool = False) -> np.ndarray:
        I = self.information_total if include_prior else self.information_likelihood
        idx = self.parameter_blocks[which]
        return I[np.ix_(idx, idx)]

    def profile(self, which: str = "theta", *, include_prior: bool = False) -> np.ndarray:
        I = self.information_total if include_prior else self.information_likelihood
        return schur_information(I, self.parameter_blocks[which])

    def min_eigenvalue(self, *, profiled: bool = True, include_prior: bool = False) -> float:
        M = (
            self.profile("theta", include_prior=include_prior)
            if profiled
            else (self.information_total if include_prior else self.information_likelihood)
        )
        return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())

    def rank(self, *, include_prior: bool = False, rtol: float = 1e-10) -> int:
        M = self.information_total if include_prior else self.information_likelihood
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        mx = float(ev.max())
        return 0 if mx <= 0 else int((ev > rtol * mx).sum())

    def to_dict(self) -> dict[str, Any]:
        return as_builtin(
            {
                "name": self.name,
                "label": self.label,
                "basis": self.basis,
                "parameter_names": self.parameter_names,
                "parameter_blocks": self.parameter_blocks,
                "information_likelihood": self.information_likelihood,
                "information_prior": self.information_prior,
                "information_total": self.information_total,
                "information_by_modality": self.information_by_modality,
                "theta_profile_information": self.theta_profile_information,
                "theta_profile_min_eigenvalue": self.min_eigenvalue(),
                "rank_likelihood": self.rank(),
                "metrics": self.metrics,
                "notes": self.notes,
            }
        )


def reference_benchmark_config(**overrides: Any) -> SystemConfig:
    """The preregistered instrument/protocol configuration of the benchmark."""
    base = dict(
        dt=1e-3, dt_eeg=1e-3, dt_bold=1.0,
        epoch_seconds=6.0, n_epochs=32,
        n_delay_taps=26, hrf_stages=8,
        dtype="float64",
    )
    base.update(overrides)
    return SystemConfig(**base)


def design_information_map(
    *,
    cfg: SystemConfig | None = None,
    regime: Any = None,
    eta: np.ndarray | None = None,
    designs: Sequence[Any] | None = None,
    seed: int = 20260805,
    standardised: bool = True,
    theta_names: Sequence[str] = THETA_NAMES,
    joint_whitening: bool = False,
) -> dict[str, DesignInformation]:
    """``{design_name: DesignInformation}`` for the five-design benchmark.

    Parameters
    ----------
    cfg:
        Instrument/protocol configuration; defaults to
        :func:`reference_benchmark_config`.
    regime:
        An :class:`~scwbd.infer.identifiability.Regime`; defaults to the first
        preregistered regime.  ``eta`` overrides its truth if given.
    designs:
        Sequence of :class:`~scwbd.infer.identifiability.DesignSpec`; defaults
        to all seven (five primary + two controls).
    joint_whitening:
        ``False`` reproduces T4 literally (modality-block-diagonal); ``True``
        whitens the stacked multirate record and is the exact joint
        information.  G4 should use the default unless it states otherwise.
    """
    from .identifiability import DESIGNS, REGIMES, build_design

    cfg = cfg or reference_benchmark_config()
    regime = regime if regime is not None else REGIMES[0]
    designs = designs if designs is not None else DESIGNS
    u_true = np.asarray(eta if eta is not None else regime.eta_true(), dtype=float)

    out: dict[str, DesignInformation] = {}
    for spec in designs:
        bd = build_design(spec, cfg, regime, seed=seed)
        rep = expected_fisher(
            u_true, bd.cfg, bd.proto, design=spec.name, channels=bd.channels,
            include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
            method="analytic", joint_whitening=joint_whitening,
            standardised=standardised, theta_names=theta_names,
        )
        notes = list(bd.notes)
        I_like = rep.I_likelihood
        if spec.coarse_model:
            # For a naively resampled design the information *available in the
            # data* and the information *the naive estimator can use* differ.
            # G4 must see the second, which is where the delay disappears.
            repc = expected_fisher(
                u_true, bd.fit_cfg, bd.fit_proto, design=spec.name + "/coarse",
                channels=bd.channels, include_impulse=bd.include_impulse,
                method="analytic", joint_whitening=joint_whitening,
                standardised=standardised, theta_names=theta_names,
            )
            I_like = repc.I_likelihood
            rep = repc
            notes.append(
                "information reported for the 1 s (naively resampled) estimator, "
                "not for the fine model applied to decimated data; see design "
                "'joint_resampled_exactmodel' for the latter"
            )
        idx = PARAMETER_BLOCKS["theta"]
        out[spec.name] = DesignInformation(
            name=spec.name,
            label=spec.label,
            basis=rep.basis,
            parameter_names=list(PARAM_NAMES),
            parameter_blocks={k: list(v) for k, v in PARAMETER_BLOCKS.items()},
            information_likelihood=I_like,
            information_prior=prior_information(standardised),
            information_by_modality=dict(rep.I_by_modality),
            theta_profile_information=schur_information(I_like, idx),
            metrics=rep.metrics,
            notes=notes,
            report=rep,
        )
    return out


#: Backwards/forwards-compatible alias.  ``scwbd.bench.adapters`` owns the
#: consumer-side binding of the same name; this one returns richer objects.
fisher_design_information = design_information_map


def theta_partition() -> tuple[list[str], list[str]]:
    """``(theta_names, all_parameter_names)`` in the stable matrix ordering.

    ``scwbd.bench.adapters.theta_partition`` probes
    ``scwbd.infer.fisher.THETA_NAMES`` / ``PARAM_NAMES`` directly; this function
    is the same declaration with the index arrays attached, for callers that
    want them.
    """
    return list(THETA_NAMES), list(PARAM_NAMES)


def theta_index() -> list[int]:
    return list(PARAMETER_BLOCKS["theta"])


def nuisance_index() -> list[int]:
    return list(PARAMETER_BLOCKS["ell"] + PARAMETER_BLOCKS["rho"])


def load_reference_schema():
    """``build_three_region_schema()`` if agent A's module is importable.

    Returns ``None`` rather than raising so that ``scwbd.infer`` is never
    blocked on the schema package; the synthetic-slice artifact uses this to
    prefer the compiled schema and falls back to its own local declaration.
    """
    try:
        from scwbd.schema.examples.three_region import (  # type: ignore
            build_three_region_schema,
        )
    except Exception:
        return None
    try:
        return build_three_region_schema()
    except Exception:
        return None
