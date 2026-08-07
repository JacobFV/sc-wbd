"""The validator that refuses an inverted curriculum.

Eight refusals.  Each names the offending stage, and each has a stated reading in
the world where the curriculum is *correct* --- otherwise it is a tripwire across
a corridor everyone walks down (``reports/decorative_guards.md``, recommendation
3).

===== ================================= ==================================================
code   refuses                            reads differently when...
===== ================================= ==================================================
X01    an inverted admission order        every lower-integrity tier is first admitted
                                          strictly after every higher-integrity tier
X02    an impure founding stage           the first stage that trains admits the founding
                                          tier and nothing else
X03    a parameter founded below tier 1   every glob a tier >= 2 may update was already
                                          reachable by a tier-1 source in this or an
                                          earlier stage, or carries a named exemption
X04    permissions that widen with tier   within a stage, each admitted tier's expanded
                                          mask is a subset of the tier above it
X05    an information-blind update        no source may update a parameter its modality
                                          carries no measured information about
X06    a config the trainer will not run  the trainer's own admission gates agree with
                                          what the config declares
X07    a card with undeclared provenance  every ``role: prior`` card says whether it is
                                          simulated
X08    an unrecorded absence              a stage admitting a tier with no live source
                                          says so in its ``absence`` block
===== ================================= ==================================================

Globs are resolved against the **real tensor names of the model that runs** ---
``SCWBD`` + ``AmortizedPosterior`` + ``Individualizer`` assembled through the
trainer's own ``_CombinedModule``, whose ``named_parameters`` strips the
``model.`` prefix.  Comparing glob *strings* would repeat defect 1 of the
decorative-guards register, where a permission set was compared in one name space
and enforced in another.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .information import BlindRule, derive_blind_rules, load_modality_information
from .spec import Curriculum, StageCurriculum, TierPolicy, load_tier_policy
from .tiers import TIER_NONE, RawCard, TierAssignment, load_mixture_cards, tier_of

__all__ = [
    "Refusal",
    "Verdict",
    "parameter_universe",
    "expand",
    "validate",
    "validate_config",
]


@dataclass
class Refusal:
    code: str
    stage: str | None
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f" [stage {self.stage}]" if self.stage else ""
        return f"{self.code}{where}: {self.message}"


@dataclass
class Verdict:
    config: str
    refusals: tuple[Refusal, ...]
    tiers: dict[str, TierAssignment]
    not_evaluable: tuple[dict[str, Any], ...]
    universe_size: int
    universe_provenance: str
    curriculum: dict[str, Any]
    blind_rules: tuple[dict[str, Any], ...]
    information_provenance: dict[str, Any]

    @property
    def ok(self) -> bool:
        """Accepted *and* fully evaluated. A check that could not run blocks this.

        This used to be ``not self.refusals``, which made "no refusal fired" and
        "the check could not run" indistinguishable in the headline. That is not
        hypothetical: when ``217b01f`` removed the stage-name gates from
        ``run_stage``, the X06 trainer-gate check stopped being evaluable, moved
        from ``refusals`` into ``not_evaluable``, and the corrected config's
        verdict silently changed from REFUSED to ACCEPTED. Nothing about the
        config improved. The check just stopped being able to fail, and a
        verdict that reports that as acceptance is the permissive default one
        level up from the ones in ``reports/decorative_guards.md``.
        """
        return not self.refusals and not self.not_evaluable

    @property
    def inconclusive(self) -> bool:
        """Nothing refused, but at least one check could not be run."""
        return not self.refusals and bool(self.not_evaluable)

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted({r.code for r in self.refusals}))

    def unevaluable_checks(self) -> tuple[str, ...]:
        return tuple(sorted({str(n.get("check", "?")) for n in self.not_evaluable}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            # Three states, not two: a run whose checks could not all execute is
            # not the same artifact as one that passed them.
            "verdict": (
                "ACCEPTED" if self.ok else "INCONCLUSIVE" if self.inconclusive else "REFUSED"
            ),
            "refusal_codes": list(self.codes()),
            "refusals": [
                {"code": r.code, "stage": r.stage, "message": r.message, "evidence": r.evidence}
                for r in self.refusals
            ],
            "not_evaluable": [dict(n) for n in self.not_evaluable],
            "tiers": {
                k: {"tier": v.tier, "reason": v.reason, "refusal": v.refusal}
                for k, v in self.tiers.items()
            },
            "parameter_universe": {
                "n_tensors": self.universe_size,
                "provenance": self.universe_provenance,
            },
            "information_rules": list(self.blind_rules),
            "information_provenance": self.information_provenance,
            "curriculum": self.curriculum,
        }

    def report(self) -> str:
        lines = [f"config: {self.config}", f"verdict: {'ACCEPTED' if self.ok else 'REFUSED'}"]
        if self.refusals:
            lines.append(f"{len(self.refusals)} refusal(s):")
            lines += [f"  {r}" for r in self.refusals]
        if self.not_evaluable:
            lines.append(f"{len(self.not_evaluable)} check(s) NOT EVALUABLE:")
            lines += [
                f"  {n['check']} [{n.get('source', '?')}]: {n['reason']}" for n in self.not_evaluable
            ]
        return "\n".join(lines)


# ======================================================================
# parameter universe
# ======================================================================
def parameter_universe(config_path: str | Path) -> tuple[tuple[str, ...], str]:
    """The trainable tensor names of the model this config builds.

    Assembled through ``scwbd.foundation.train._CombinedModule`` so the names are
    the ones ``GradientGate`` will actually see, including its ``model.`` prefix
    stripping and the ``posterior.``/``individualizer.`` prefixes it keeps.
    """
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.config import load_config
    from scwbd.foundation.individual import Individualizer
    from scwbd.foundation.model import SCWBD
    from scwbd.foundation.posterior import AmortizedPosterior
    from scwbd.foundation.simulate import THETA_NAMES, ThetaPrior
    from scwbd.foundation.train import _CombinedModule
    from scwbd.foundation.util import logical_param_name

    cfg = load_config(config_path)
    # Built with the same call the trainer makes (train.py:178-187), so the names
    # are the ones GradientGate will see rather than an equivalent construction.
    anat = load_anatomy(n_cortex=400)
    model = SCWBD(cfg.model, anat)
    post = AmortizedPosterior(
        cfg.posterior,
        len(THETA_NAMES),
        prior=ThetaPrior(),
        fs=cfg.data.fs_hz,
        nuisance_dim=cfg.posterior.nuisance_dim,
    )
    indiv = Individualizer(len(THETA_NAMES), n_groups=2, n_participants=8, n_sessions=8)
    combined = _CombinedModule({"model": model, "posterior": post, "individualizer": indiv})
    names = tuple(
        logical_param_name(n) for n, p in combined.named_parameters() if p.requires_grad
    )
    prov = (
        f"SCWBD({cfg.model.n_regions} regions, anatomy provenance={anat.provenance!r}, "
        f"is_biological={anat.is_biological()}) + AmortizedPosterior + Individualizer, "
        "named through scwbd.foundation.train._CombinedModule"
    )
    return names, prov


def expand(globs: Iterable[str], universe: Sequence[str], frozen: Iterable[str] = ()) -> set[str]:
    """Concrete parameter names a glob set reaches, minus anything frozen."""
    gl, fr = list(globs), list(frozen)
    out = {n for n in universe if any(fnmatch.fnmatch(n, g) for g in gl)}
    if fr:
        out -= {n for n in out if any(fnmatch.fnmatch(n, f) for f in fr)}
    return out


# ======================================================================
# the validator
# ======================================================================
def validate(
    curriculum: Curriculum,
    cards: Mapping[str, RawCard],
    policy: TierPolicy,
    *,
    universe: Sequence[str],
    universe_provenance: str = "",
    blind_rules: Sequence[BlindRule] = (),
    information_provenance: Mapping[str, Any] | None = None,
    legacy: Any | None = None,
) -> Verdict:
    refusals: list[Refusal] = []
    not_evaluable: list[dict[str, Any]] = []

    # ---- tier assignment, and X07 -------------------------------------
    tiers = {sid: tier_of(c) for sid, c in sorted(cards.items())}
    for sid, a in tiers.items():
        if a.refusal:
            refusals.append(
                Refusal(
                    code=a.refusal,
                    stage=None,
                    message=f"source {sid!r}: {a.reason}",
                    evidence={"card": str(cards[sid].path), **a.evidence},
                )
            )

    live_by_tier: dict[int, list[str]] = {}
    for sid, a in tiers.items():
        if a.tier in (None, TIER_NONE):
            continue
        if not cards[sid].spec.enabled:
            continue
        live_by_tier.setdefault(a.tier, []).append(sid)

    stages = [s for s in curriculum.stages if s.enabled and s.steps > 0]

    # ---- X01: inverted admission order --------------------------------
    first: dict[int, StageCurriculum] = {}
    for s in stages:
        for t in s.admits:
            first.setdefault(t, s)
    ordered = sorted(first)
    for lo in ordered:  # lo = a lower-integrity (larger) tier number
        for hi in ordered:
            if hi >= lo:
                continue
            if first[lo].order <= first[hi].order:
                refusals.append(
                    Refusal(
                        code="X01_inverted_admission",
                        stage=first[lo].name,
                        message=(
                            f"tier {lo} ({_tier_name(lo)}) is first admitted at stage "
                            f"{first[lo].name!r} (order {first[lo].order}), which is not after "
                            f"tier {hi} ({_tier_name(hi)}) at stage {first[hi].name!r} "
                            f"(order {first[hi].order}). A lower-integrity source may extend a "
                            "representation measured evidence founded; it may not arrive first "
                            "or alongside. Appendix B: the roles are 'deliberately "
                            "non-equivalent'."
                        ),
                        evidence={
                            "lower_integrity_tier": lo,
                            "first_admitted_stage": first[lo].name,
                            "first_admitted_order": first[lo].order,
                            "higher_integrity_tier": hi,
                            "higher_stage": first[hi].name,
                            "higher_order": first[hi].order,
                            "sources_in_lower_tier": live_by_tier.get(lo, []),
                            "sources_in_higher_tier": live_by_tier.get(hi, []),
                        },
                    )
                )

    # ---- X02: founding purity -----------------------------------------
    if stages:
        s0 = stages[0]
        impure = tuple(t for t in s0.admits if t != policy.founding_tier)
        if impure:
            refusals.append(
                Refusal(
                    code="X02_founding_stage_not_pure",
                    stage=s0.name,
                    message=(
                        f"the first stage that trains admits tier(s) {list(impure)} alongside "
                        f"tier {policy.founding_tier}. The curriculum must start from ground "
                        "truth alone: whatever founds the representation becomes the prior every "
                        "later source has to argue against."
                    ),
                    evidence={"admits": list(s0.admits), "founding_tier": policy.founding_tier},
                )
            )
        elif policy.founding_tier not in s0.admits:
            refusals.append(
                Refusal(
                    code="X02_founding_stage_not_pure",
                    stage=s0.name,
                    message=(
                        f"the first stage that trains does not admit the founding tier "
                        f"{policy.founding_tier} at all (admits {list(s0.admits)})."
                    ),
                    evidence={"admits": list(s0.admits)},
                )
            )

    # ---- X03 / X04: founding and narrowing of gradient permission -----
    reachable_by_tier_upto: dict[int, set[str]] = {}
    for s in stages:
        # what each admitted tier may touch in THIS stage
        per_tier: dict[int, set[str]] = {}
        for t in sorted(s.admits):
            declared = s.permits(t)
            reach: set[str] = set()
            for sid in live_by_tier.get(t, []):
                spec = cards[sid].spec
                # Each source's own frozen list applies to its OWN permissions.
                # Unioning frozen lists across a tier would let one source's
                # freeze silence another's permission and hide a violation.
                own = expand(spec.gradient_permission, universe, spec.frozen)
                if declared:
                    own &= expand(declared, universe)  # a stage restricts, never grants
                reach |= own
            if not live_by_tier.get(t) and declared:
                # tier admitted with no live source: the stage's own declaration
                # is all there is, and X08 will record the absence separately.
                reach = expand(declared, universe)
            per_tier[t] = reach

        for t, reach in per_tier.items():
            if t <= policy.founding_tier:
                reachable_by_tier_upto.setdefault(t, set()).update(reach)
                continue
            founded = set()
            for ht in range(1, t):
                founded |= reachable_by_tier_upto.get(ht, set())
            unfounded = sorted(reach - founded)
            unexempt = [
                n for n in unfounded if policy.exemption_for(_glob_key(n), t) is None
            ]
            if unexempt:
                refusals.append(
                    Refusal(
                        code="X03_unfounded_parameter",
                        stage=s.name,
                        message=(
                            f"tier {t} ({_tier_name(t)}) may update {len(unexempt)} tensor(s) that "
                            "no higher-integrity tier has been permitted to update in this or an "
                            f"earlier stage: {unexempt[:8]}"
                            + (" …" if len(unexempt) > 8 else "")
                            + ". A lower-integrity source that is the sole author of a parameter "
                            "has founded it. Declare a founding exemption in "
                            "configs/curriculum/tiers.yaml or withdraw the permission."
                        ),
                        evidence={
                            "tier": t,
                            "n_unfounded": len(unexempt),
                            "unfounded": unexempt,
                            "founded_by_higher_tiers": len(founded),
                        },
                    )
                )
            reachable_by_tier_upto.setdefault(t, set()).update(reach)

        # X04: narrowing must be visible WITHIN the stage.
        #
        # Distinct from X03, and strictly stronger, on purpose. X03 asks "did some
        # higher-integrity tier reach this tensor at any point up to now?" -- a
        # source may therefore extend a parameter a tier-1 stage founded three
        # stages ago. X04 asks the same question of *this stage alone*: when a
        # lower tier is training, a higher one must be training the same tensors
        # concurrently. That is what stops the measured signal from being founded
        # once and then drifted away from for the rest of the run.
        for t in sorted(per_tier):
            if t <= policy.founding_tier:
                continue
            in_stage_higher: set[str] = set()
            for ht in sorted(per_tier):
                if ht < t:
                    in_stage_higher |= per_tier[ht]
            widened = sorted(per_tier[t] - in_stage_higher)
            widened = [n for n in widened if policy.exemption_for(_glob_key(n), t) is None]
            if widened:
                refusals.append(
                    Refusal(
                        code="X04_permission_widens_with_tier",
                        stage=s.name,
                        message=(
                            f"in this stage tier {t} ({_tier_name(t)}) may update "
                            f"{len(widened)} tensor(s) that no higher-integrity tier admitted to "
                            f"the SAME stage may update: {widened[:8]}"
                            + (" …" if len(widened) > 8 else "")
                            + ". Each successive tier must enter with a mask contained in the "
                            "concurrently-admitted higher tiers, not one that widens beyond them."
                        ),
                        evidence={
                            "tier": t,
                            "higher_tiers_in_stage": [ht for ht in sorted(per_tier) if ht < t],
                            "n_widened": len(widened),
                            "widened": widened,
                            "n_reach_higher_in_stage": len(in_stage_higher),
                            "n_reach_this_tier": len(per_tier[t]),
                        },
                    )
                )

    # ---- X05: information-blind updates -------------------------------
    #
    # A source may update a parameter if ANY modality it observes carries
    # information about that parameter.  The refusal therefore needs *every*
    # observed modality to be blind -- the quantifier matters: requiring only one
    # blind modality would refuse an EEG source for BOLD's blindness, which is
    # the opposite of the rule.
    binding = [r for r in blind_rules if r.binds]
    blind_index: dict[tuple[str, str], BlindRule] = {(r.modality, r.param): r for r in binding}
    #: The modalities the laboratory actually has a Fisher block for.  A source
    #: observing anything else cannot be judged by these rules, and says so.
    modelled_modalities = {r.modality for r in blind_rules} or {"eeg", "bold"}
    checked_any = False
    for sid, c in sorted(cards.items()):
        if "observes" not in c.raw:
            not_evaluable.append(
                {
                    "check": "X05_information_blind_update",
                    "status": "not_evaluable",
                    "source": sid,
                    "reason": (
                        "the card declares no `observes:` field, so the modalities whose measured "
                        "information would license or refuse its gradients are unknown. Not a pass."
                    ),
                }
            )
            continue
        obs = list(c.raw["observes"] or [])
        if not obs:
            not_evaluable.append(
                {
                    "check": "X05_information_blind_update",
                    "status": "not_applicable",
                    "source": sid,
                    "reason": (
                        "declares `observes: []` -- a non-observational source (a prior or a "
                        "teacher). The per-modality Fisher rules describe observation models and "
                        "say nothing about it."
                    ),
                }
            )
            continue
        unmodelled = [m for m in obs if m not in modelled_modalities]
        if unmodelled:
            not_evaluable.append(
                {
                    "check": "X05_information_blind_update",
                    "status": "not_evaluable",
                    "source": sid,
                    "reason": (
                        f"observes {unmodelled}, for which the identifiability laboratory has no "
                        "Fisher block. No blindness can be established, so no permission is "
                        "refused on this ground and none is licensed either."
                    ),
                }
            )
            continue
        checked_any = True
        reach = expand(c.spec.gradient_permission, universe, c.spec.frozen)
        for param, (globs, _desc) in _params_with_globs(blind_rules):
            rules_hit = [blind_index.get((m, param)) for m in obs]
            if not all(rules_hit):
                continue  # at least one observed modality sees this parameter
            hit = sorted(expand(globs, universe) & reach)
            if not hit:
                continue
            r0 = rules_hit[0]
            assert r0 is not None
            refusals.append(
                Refusal(
                    code="X05_information_blind_update",
                    stage=None,
                    message=(
                        f"source {sid!r} observes {obs} and may update {hit[:6]}"
                        + (" …" if len(hit) > 6 else "")
                        + f", but every modality it observes carries "
                        f"{r0.kind.replace('_', ' ')} information about {param} in every regime "
                        f"({r0.ratios})."
                    ),
                    evidence={
                        "source": sid,
                        "observes": obs,
                        "lab_parameter": param,
                        "rules": [r.as_dict() for r in rules_hit if r is not None],
                        "tensors": hit,
                    },
                )
            )
    if binding and not checked_any:
        not_evaluable.append(
            {
                "check": "X05_information_blind_update",
                "status": "not_evaluable",
                "source": "*",
                "reason": (
                    f"{len(binding)} binding information rule(s) were derived and no card in this "
                    "mixture declares an observable modality the laboratory models, so none of "
                    "them could be applied."
                ),
            }
        )

    # ---- X06: the trainer will not honour this config ------------------
    if legacy is not None:
        for s in stages:
            if not s.declared:
                continue
            recon = legacy.for_stage(s.name)
            if tuple(sorted(recon.admits)) != tuple(sorted(s.admits)):
                refusals.append(
                    Refusal(
                        code="X06_trainer_gate_contradicts_config",
                        stage=s.name,
                        message=(
                            f"the config declares this stage admits tiers {list(sorted(s.admits))}, "
                            f"but scwbd.foundation.train.FoundationTrainer.run_stage would admit "
                            f"{list(recon.admits)} (sources {list(recon.source_ids)}). Source "
                            "admission is hard-coded there by stage NAME, so this ordering cannot "
                            "be enacted by editing the config alone."
                        ),
                        evidence={
                            "declared_admits": list(sorted(s.admits)),
                            "trainer_admits": list(recon.admits),
                            "trainer_source_ids": list(recon.source_ids),
                            "trainer_gates": {
                                "sim_excluded_stage": legacy.sim_excluded_stage,
                                "real_admitted_stages": list(legacy.real_admitted_stages),
                            },
                        },
                    )
                )

    # ---- X09: a declared provenance the runtime object contradicts ------
    #
    # A card is a claim about an object, and a tier is assigned from the claim.
    # Nothing so far has checked that the object agrees. This runs the *same call
    # the trainer makes* (``load_anatomy()``, train.py:178) and compares.
    for sid, c in sorted(cards.items()):
        check = c.raw.get("runtime_provenance_check")
        if not check:
            if tiers[sid].tier == 3:
                not_evaluable.append(
                    {
                        "check": "X09_declared_provenance_contradicted",
                        "source": sid,
                        "reason": (
                            "tier-3 card declares no `runtime_provenance_check:`, so nothing "
                            "verifies that the object delivered under this name is the measured "
                            "population prior the card claims."
                        ),
                    }
                )
            continue
        if check != "anatomy":
            not_evaluable.append(
                {
                    "check": "X09_declared_provenance_contradicted",
                    "source": sid,
                    "reason": f"no runtime probe is implemented for check {check!r}",
                }
            )
            continue
        from scwbd.foundation.anatomy import load_anatomy

        anat = load_anatomy(n_cortex=400)
        if c.spec.is_simulated is False and not anat.is_biological():
            refusals.append(
                Refusal(
                    code="X09_declared_provenance_contradicted",
                    stage=None,
                    message=(
                        f"source {sid!r} declares `is_simulated: false` and is therefore ranked "
                        f"tier {tiers[sid].tier}, but the object the trainer actually loads reports "
                        f"provenance {anat.provenance!r} with is_biological()=False. The population "
                        "prior this stage admits is a labelled synthetic stand-in, which belongs to "
                        "tier 4, not tier 3."
                    ),
                    evidence={
                        "source": sid,
                        "declared_is_simulated": False,
                        "runtime_provenance": anat.provenance,
                        "runtime_is_biological": bool(anat.is_biological()),
                        "probe": "scwbd.foundation.anatomy.load_anatomy(n_cortex=400)",
                    },
                )
            )

    # ---- X08: an admitted tier with no live source, unrecorded ---------
    for s in stages:
        recorded = {a.get("tier") for a in s.absence}
        for t in s.admits:
            if live_by_tier.get(t):
                continue
            if t in policy.silently_optional_tiers or t in recorded:
                continue
            refusals.append(
                Refusal(
                    code="X08_unrecorded_absence",
                    stage=s.name,
                    message=(
                        f"this stage admits tier {t} ({_tier_name(t)}) but no enabled source card "
                        "carries that tier, and the stage records no `absence:` entry saying so. "
                        "An unwired tier and a tier that contributed nothing must not look alike."
                    ),
                    evidence={"tier": t, "live_by_tier": {k: v for k, v in sorted(live_by_tier.items())}},
                )
            )

    return Verdict(
        config=str(curriculum.config_path),
        refusals=tuple(refusals),
        tiers=tiers,
        not_evaluable=tuple(not_evaluable),
        universe_size=len(universe),
        universe_provenance=universe_provenance,
        curriculum=curriculum.as_dict(),
        blind_rules=tuple(r.as_dict() for r in blind_rules),
        information_provenance=dict(information_provenance or {}),
    )


def _tier_name(t: int) -> str:
    from .tiers import TIER_BY_NUMBER

    tt = TIER_BY_NUMBER.get(t)
    return tt.name if tt else f"tier_{t}"


def _params_with_globs(rules: Sequence[BlindRule]) -> list[tuple[str, tuple[tuple[str, ...], str]]]:
    """Laboratory parameters that bind a trainable tensor, with their globs."""
    from .information import LAB_PARAM_TO_GLOBS

    seen = {r.param for r in rules}
    return [(p, LAB_PARAM_TO_GLOBS[p]) for p in sorted(seen) if LAB_PARAM_TO_GLOBS.get(p, ((),))[0]]


def _glob_key(name: str) -> str:
    """The glob an exemption would have been written as, for a concrete name."""
    head = name.split(".", 1)[0]
    return f"{head}.*"


def validate_config(
    config_path: str | Path,
    *,
    tiers_path: str | Path = "configs/curriculum/tiers.yaml",
    cards_dir: str | Path | None = None,
    results_path: str | Path = "reports/identifiability/results.json",
    manifest_path: str | Path = "reports/identifiability/manifest.json",
) -> Verdict:
    """Load everything a validation needs and run it."""
    cur = Curriculum.from_config(config_path)
    policy = load_tier_policy(tiers_path)
    cards = load_mixture_cards(cards_dir or cur.mixture_cards)
    universe, prov = parameter_universe(config_path)

    info = load_modality_information(results_path, manifest_path)
    rules = derive_blind_rules(info)

    # Always reconstruct the trainer's own gates -- especially for a *fully
    # declared* config, which is the only kind that can contradict them. Skipping
    # this when the config declares its curriculum would mean the better-specified
    # config got the weaker check.
    from .legacy import GateNotFound, reconstruct_stage_admission

    card_tiers = {sid: a.tier for sid, a in ((s, tier_of(c)) for s, c in cards.items()) if a.tier}
    legacy: Any | None
    try:
        legacy = reconstruct_stage_admission(card_tiers=card_tiers)
    except GateNotFound as exc:
        legacy = None
        gate_failure = str(exc)
    else:
        gate_failure = ""

    verdict = validate(
        cur,
        cards,
        policy,
        universe=universe,
        universe_provenance=prov,
        blind_rules=rules,
        information_provenance=info.provenance,
        legacy=legacy,
    )
    if gate_failure:
        verdict.not_evaluable += (
            {
                "check": "X06_trainer_gate_contradicts_config",
                "status": "not_evaluable",
                "source": "scwbd.foundation.train",
                "reason": gate_failure,
            },
        )
    return verdict
