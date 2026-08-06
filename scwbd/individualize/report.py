"""The per-patient report a clinician reads.

Four things must be visible without scrolling and without arithmetic:

1. what modalities the patient has;
2. what was individualized;
3. what stayed at the population value;
4. **what cannot be individualized from this data, and why** -- with the
   measured number, not an adjective.

The uncertainty ledger for every group is printed whether or not the group was
fitted, because "we did not fit this" is exactly the case where a reader is most
likely to assume a number is the patient's.
"""

from __future__ import annotations

import json

from .fit import (
    INDIVIDUALIZED,
    POPULATION_PRIOR,
    WEAKLY_INDIVIDUALIZED,
    IndividualizationResult,
)
from .profile import IdentifiabilityProfile

__all__ = ["patient_report", "profile_report", "result_json"]

_STATUS_WORD = {
    INDIVIDUALIZED: "INDIVIDUALIZED",
    WEAKLY_INDIVIDUALIZED: "WEAKLY INDIVIDUALIZED",
    POPULATION_PRIOR: "POPULATION PRIOR -- NOT THIS PATIENT",
}


def _fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.{digits}g}"


def profile_report(profile: IdentifiabilityProfile) -> str:
    """The pre-fit answer: what could this patient's data personalise at all?"""
    L: list[str] = []
    L.append(f"# Identifiability profile -- patient `{profile.patient_id}`")
    L.append("")
    L.append(
        "Computed **before any fitting**, from the declaration of what was "
        "measured. No patient data were used to produce this table."
    )
    L.append("")
    L.append(f"- modalities present: `{list(profile.present)}`")
    L.append(f"- reference-slice design: `{profile.design}`  channels: `{list(profile.channels)}`")
    L.append(f"- regime: `{profile.regime}`  basis: `{profile.basis}`")
    L.append(
        f"- thresholds (prior-precision units): identifiable >= "
        f"{profile.thresholds.identifiable:g}, weak >= {profile.thresholds.weak:g}"
    )
    L.append("")
    L.append("| group | status | lambda_min (likelihood) | posterior sd / prior sd | evidence |")
    L.append("|---|---|---|---|---|")
    for name, g in profile.groups.items():
        L.append(
            f"| `{name}` | **{g.status}** | {_fmt(g.lambda_min)} | "
            f"{_fmt(g.posterior_sd_ratio, 4)} | {g.evidence_kind} |"
        )
    L.append("")
    L.append("## Why")
    for name, g in profile.groups.items():
        L.append(f"- **`{name}`** ({g.clinical_meaning}): {g.reason}")
    if profile.counterfactuals:
        L.append("")
        L.append("## Measured counterfactuals -- what acquiring more data would buy")
        L.append("")
        L.append("| add modality | group | status it would reach | lambda_min |")
        L.append("|---|---|---|---|")
        for mod, res in profile.counterfactuals.items():
            for gname, entry in res.items():
                L.append(
                    f"| `{mod}` | `{gname}` | {entry['status']} | "
                    f"{_fmt(entry.get('lambda_min'))} |"
                )
    if profile.notes:
        L.append("")
        L.append("## Notes")
        for n in profile.notes:
            L.append(f"- {n}")
    L.append("")
    L.append(
        f"_statistic: {profile.provenance.get('statistic')}_  \n"
        f"_computation: {profile.provenance.get('computed')}_  \n"
        f"_config: {json.dumps(dict(profile.config), sort_keys=True)}_"
    )
    return "\n".join(L)


def patient_report(result: IndividualizationResult) -> str:
    """The post-fit report: what was personalised, what was not, and why not."""
    L: list[str] = []
    av = result.availability
    L.append(f"# Individualization report -- patient `{result.patient_id}`")
    L.append("")

    # 1 -- modalities present
    L.append("## 1. Modalities present")
    L.append("")
    L.append("| modality | source card | support | clock | calibration | declarations |")
    L.append("|---|---|---|---|---|---|")
    for r in av.records:
        d = r.to_dict()
        kinds = ",".join(f"{k}={v}" for k, v in sorted(d["declaration_kind"].items()))
        L.append(
            f"| `{d['modality']}` | {d['source_card']} | {d['support']} | "
            f"{d['clock']} | {d['calibration']} | {kinds} |"
        )
    if not av.records:
        L.append("| _(none)_ | | | | | |")
    L.append("")
    L.append(f"- absent: `{list(av.absent)}` -- absent, not zero-imputed.")
    L.append(f"- reference-slice design: `{av.design}`, channels `{list(av.channels)}`")
    L.append(f"- sessions: `{list(result.decomposition.session_ids)}`")
    c = result.consistency
    if c is None:
        L.append(
            "- record/model consistency: **not run** (no optimiser ran, so there "
            "was no record to check against the model)"
        )
    else:
        L.append(
            f"- record/model consistency: **{'PASS' if c.passed else 'REJECTED'}** "
            f"-- whitened innovations mean square {_fmt(c.statistic, 4)} over "
            f"{c.n_samples} samples, tolerance {c.rel_tolerance:.0%} around 1"
        )
        if not c.passed:
            L.append(f"  - {c.reason}")
    L.append("")

    # 2 -- individualized
    L.append("## 2. What was individualized")
    L.append("")
    ind = [result.outcomes[g] for g in result.individualized_groups]
    if ind:
        L.append("| group | status | parameters | value (natural) | posterior sd (unconstrained) | source |")
        L.append("|---|---|---|---|---|---|")
        for o in ind:
            vals = ", ".join(
                f"{k}={_fmt(v, 4)}" for k, v in o.value_natural.items()
            ) or "(outside the dynamical parameter vector)"
            sds = ", ".join(
                f"{k}={_fmt(v, 3)}" for k, v in o.posterior_sd_unconstrained.items()
            ) or "n/a"
            L.append(
                f"| `{o.group}` | {_STATUS_WORD.get(o.status, o.status)} | "
                f"`{list(o.parameters)}` | {vals} | {sds} | {o.source} |"
            )
    else:
        L.append("**Nothing was individualized.** Every value below is the population's.")
    L.append("")

    # 3 -- population
    L.append("## 3. What remained at the population value")
    L.append("")
    pop = [result.outcomes[g] for g in result.population_prior_groups]
    if pop:
        L.append("| group | parameters | value (natural) | prior sd | label |")
        L.append("|---|---|---|---|---|")
        for o in pop:
            vals = ", ".join(
                f"{k}={_fmt(v, 4)}" for k, v in o.value_natural.items()
            ) or "(outside the dynamical parameter vector)"
            sds = ", ".join(
                f"{k}={_fmt(v, 3)}" for k, v in o.posterior_sd_unconstrained.items()
            ) or "n/a"
            L.append(
                f"| `{o.group}` | `{list(o.parameters)}` | {vals} | {sds} | "
                f"`{POPULATION_PRIOR}` |"
            )
        L.append("")
        L.append(
            "These values are **bit-identical** to the population values the fit "
            "started from; no optimiser touched them."
        )
    else:
        L.append("_(none)_")
    L.append("")

    # 4 -- what cannot be individualized, and why
    L.append("## 4. What CANNOT be individualized from this data, and why")
    L.append("")
    cannot = [
        result.outcomes[g.group]
        for g in result.profile.groups.values()
        if g.status == "not_identifiable"
    ]
    if cannot:
        for o in cannot:
            gi = result.profile.groups[o.group]
            L.append(f"### `{o.group}` -- {gi.clinical_meaning}")
            L.append("")
            L.append(f"- measured `lambda_min` = **{_fmt(gi.lambda_min)}** prior-precision units")
            L.append(
                f"- posterior sd would be **{_fmt(gi.posterior_sd_ratio, 6)} x** the "
                "prior sd: the prior, to six figures"
            )
            L.append(f"- {gi.reason}")
            rem = result.profile.remedies(o.group)
            if rem:
                for r in rem:
                    L.append(
                        f"- **measured remedy**: acquiring `{r['add_modality']}` "
                        f"would make this group identifiable "
                        f"(lambda_min = {_fmt(r['lambda_min'])})"
                    )
            L.append("")
    else:
        L.append("_(every declared group is at least weakly identifiable here)_")
    L.append("")

    # 5 -- hierarchical decomposition
    d = result.decomposition
    L.append("## 5. Hierarchical decomposition (body.tex sec. 6.5)")
    L.append("")
    L.append("`theta_{p,s} = mu + alpha_{g(p)} + delta_p + zeta_{p,s}`")
    L.append("")
    L.append(f"- population group: `{result.group}`")
    L.append(
        f"- group effects centered: `sum_g n_g alpha_g = "
        f"{_fmt(float(abs((result.population.group_counts.reshape(-1,1)/result.population.group_counts.sum()*result.population.alpha).sum(0)).max()))}`"
    )
    L.append(
        f"- session effects centered within patient: "
        f"`max_j |sum_s zeta_{{p,s}}[j]| = {_fmt(d.centering_residual)}`"
    )
    if not result.fit_mask.any():
        L.append(
            "- `delta_p` and `zeta_{p,s}` are **exactly zero** for every "
            "coordinate: no optimiser ran, so `theta_{p,s} = mu + alpha_g` "
            "identically. The trait/state question does not arise."
        )
    else:
        L.append(
            f"- delta/zeta separable: **{d.separable}** -- {d.separability_reason}"
        )
    shrink = [
        f"{n}={_fmt(float(s), 3)}"
        for n, s in zip(d.parameter_names, d.shrinkage_factor)
        if float(s) != 0.0
    ]
    where = (
        "applied inside the per-session MAP fit, via the individual prior "
        "`N(mu + alpha_g, Sigma_person + Sigma_session)`; the factors below are "
        "the implied normal-normal weights, reported as a diagnostic and NOT "
        "applied a second time"
        if d.shrinkage_applied_in_fit
        else (
            "not applicable -- nothing was fitted"
            if not result.fit_mask.any()
            else "applied here, to the raw per-session offsets"
        )
    )
    L.append(f"- shrinkage of `delta_p`: {where}")
    L.append(f"- implied shrinkage factors: {', '.join(shrink) or '(none fitted)'}")
    L.append("")

    # 6 -- ledger
    L.append("## 6. Uncertainty ledger")
    L.append("")
    L.append("| group | status | variance source | variance | prior fraction |")
    L.append("|---|---|---|---|---|")
    for name, o in result.outcomes.items():
        led = o.ledger
        var = ", ".join(f"{k}={_fmt(v,3)}" for k, v in led.get("variance", {}).items())
        pf = ", ".join(
            f"{k}={_fmt(v,3)}" for k, v in (led.get("prior_fraction") or {}).items()
        )
        L.append(
            f"| `{name}` | {_STATUS_WORD.get(o.status, o.status)} | "
            f"{led.get('variance_source', led.get('note', '-'))} | {var or '-'} | {pf or '-'} |"
        )
    L.append("")
    L.append(
        "`prior fraction` is `1/(1 + I_ii)`: the share of the posterior precision "
        "that came from the prior rather than from this patient. A group at 1.000 "
        "is the prior."
    )

    if result.notes:
        L.append("")
        L.append("## Notes")
        for n in result.notes:
            L.append(f"- {n}")
    L.append("")
    L.append("---")
    L.append("")
    L.append(profile_report(result.profile))
    return "\n".join(L)


def result_json(result: IndividualizationResult) -> str:
    return json.dumps(result.to_dict(), indent=1, sort_keys=True, default=str)
