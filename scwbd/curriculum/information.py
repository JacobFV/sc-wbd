"""Gradient permission grounded in measured information, not in judgement.

Item 3 of the integrity-ordering brief: *a source that carries no information
about a parameter should not be permitted to update it.*  This module derives
that permission from :file:`reports/identifiability/results.json` rather than
asserting it.

What the numbers are, stated before they are used
-------------------------------------------------
``reports/identifiability/manifest.json`` lists ``"no real datasets"`` among the
run's **non-goals**.  These are therefore *not* measurements of real fMRI or real
EEG.  They are the expected Fisher information of the T4 linear observation
model --- a 105-dimensional linear-Gaussian state space with a 26-tap delay line,
30 epochs of 3.0 s at ``dt = 1 ms``, float64, seed 20260805, ``--no-monte-carlo``
so the reported information is analytic rather than sampled --- evaluated in
three held-out regimes.

That provenance decides what the numbers may be used *for*.  They ground a
**refusal** (this source may not update that parameter) and never a **grant**
(this source may update that parameter because it is informative).  A refusal
derived from a simulator-conditioned geometry is conservative: if the linear
laboratory is wrong about how much a modality sees, the cost is a permission
withheld from a source that might have deserved it.  A *grant* derived from the
same evidence would import the simulator's idiosyncrasies into the gradient
mask, which is the error this whole curriculum exists to correct.

Two rules, and why the second one has to be a ratio
---------------------------------------------------
``I_by_modality`` gives a per-modality Fisher block in the prior-standardised
basis.  Its diagonal is "how much this modality sees of this parameter".

* ``structural_zero`` --- the diagonal is exactly ``0.0`` in every regime.  This
  is not a small number, it is the absence of a path: EEG does not observe the
  haemodynamic cascade at all, and BOLD does not observe the EEG lead field at
  all.
* ``negligible`` --- the modality's diagonal is below
  :data:`NEGLIGIBLE_RATIO` of the best modality's, in **every** regime.

The ratio bar is stated as a ratio, and required in every regime, because an
absolute information bar would encode an invisible assumption about scale
(``reports/decorative_guards.md``, "a preregistered threshold with no reference
class").  The best modality in the same regime is the reference class.

**This bar discriminates, and the file it reads proves it.**  At
``NEGLIGIBLE_RATIO = 1e-4``:

===================== ============================== ==========
parameter              measured BOLD/EEG ratio        verdict
===================== ============================== ==========
``tau`` (delay)        4.0e-07, 9.1e-07, 1.0e-06      negligible
``a21`` (coupling)     1.3e-03, 4.0e-03, 5.0e-03      **not** negligible
``a32`` (coupling)     1.8e-03, 6.1e-03, 8.0e-03      **not** negligible
``a13`` (coupling)     3.1e-03, 9.0e-03, 9.5e-03      **not** negligible
===================== ============================== ==========

So the rule fires on delay and does not fire on coupling, from the same file, in
the same call.  A bar that marked both would not be measuring anything.  Note
this also corrects a convenient over-reading: the frequently quoted
``2.9e-06 vs 16.008`` pair is the *joint* theta-profile minimum eigenvalue over
``{a21, a32, a13, tau}``, which is dominated by ``tau``.  Per parameter, BOLD
carries ~10^-3 of EEG's information about coupling --- small, and not nothing.

What binds in SC-WBD-001-beta, and what does not
-------------------------------------------------
:data:`LAB_PARAM_TO_GLOBS` maps the laboratory's parameters onto this model's
trainable tensors.  Two entries map to **nothing**, and say so:

* ``tau`` --- conduction delay is a *buffer* cut from tract length
  (``FOUNDATION_BINDING["operator:long_range:delay"] == ()``).  It is declared,
  frozen by construction and deliberately not trainable.  The sharpest
  information result in the file therefore constrains no gradient in this
  architecture.  Recorded rather than quietly dropped.
* ``gain_eeg`` / ``tilt_eeg`` map to ``eeg.*``, which every simulated and prior
  card already freezes.

The entry that does bind is ``bold.*``: ``I_eeg`` for ``beta_hrf``, ``c_under``
and ``gain_bold`` is exactly ``0.0`` in all three regimes.  Every measured source
live in this corpus is EEG-only, so *no measured evidence available to this
project carries any information whatsoever about the haemodynamic parameters* --
which converts ``eegmmidb``'s hand-written ``frozen: ["bold.*"]`` from an
author's judgement into a measured fact, and forces the question of who is
allowed to train ``bold.*`` at all.  See :mod:`scwbd.curriculum.validate`,
refusal ``X03``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NEGLIGIBLE_RATIO",
    "LAB_PARAM_TO_GLOBS",
    "ModalityInformation",
    "BlindRule",
    "load_modality_information",
    "derive_blind_rules",
]

#: A modality is *negligible* for a parameter when it carries less than this
#: fraction of the best modality's information, in every regime.  Reference
#: class: the best modality in the same regime.  Measured margins above.
NEGLIGIBLE_RATIO = 1e-4

#: Laboratory parameter -> the globs over ``SCWBD.named_parameters()`` it stands
#: for.  An empty tuple is a *claim*, not an omission: it says the quantity is
#: modelled and not trained.  Cross-checked against
#: ``scwbd.foundation.compiler_bridge.FOUNDATION_BINDING``.
LAB_PARAM_TO_GLOBS: dict[str, tuple[tuple[str, ...], str]] = {
    "a21": (("coupling.gain_*", "coupling.global_scale"), "directed coupling gain"),
    "a32": (("coupling.gain_*", "coupling.global_scale"), "directed coupling gain"),
    "a13": (("coupling.gain_*", "coupling.global_scale"), "directed coupling gain"),
    "tau": (
        (),
        "network conduction delay -- a frozen buffer cut from tract length "
        "(FOUNDATION_BINDING['operator:long_range:delay'] == ()); not trainable in this model",
    ),
    "gain_eeg": (("eeg.log_gain", "eeg.*"), "EEG lead-field global gain"),
    "tilt_eeg": (("eeg.*",), "EEG electrode-placement tilt"),
    "beta_hrf": (("bold.*",), "haemodynamic cascade time constant"),
    "c_under": (("bold.*",), "HRF undershoot weight"),
    "gain_bold": (("bold.*",), "BOLD global gain"),
}

#: Which observable each modality label in the results file corresponds to.
MODALITY_OBSERVABLE = {"eeg": "scalp electric potential", "bold": "haemodynamic signal"}


@dataclass
class ModalityInformation:
    """Per-(regime, modality, parameter) diagonal Fisher information."""

    #: ``{regime: {modality: {param: float}}}``
    diagonal: dict[str, dict[str, dict[str, float]]]
    params: tuple[str, ...]
    regimes: tuple[str, ...]
    modalities: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def ratio(self, modality: str, param: str, regime: str) -> float | None:
        """``I[modality] / max_m I[m]`` for one parameter in one regime.

        ``None`` when *every* modality is blind to the parameter, which is a
        different statement from a ratio of zero and must not be flattened into
        one.
        """
        row = self.diagonal[regime]
        best = max(row[m].get(param, 0.0) for m in row)
        if best <= 0.0:
            return None
        return row[modality].get(param, 0.0) / best


@dataclass
class BlindRule:
    """A measured refusal: ``modality`` may not update ``globs``."""

    modality: str
    param: str
    globs: tuple[str, ...]
    kind: str  # structural_zero | negligible
    ratios: dict[str, float]
    binds: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "lab_parameter": self.param,
            "globs": list(self.globs),
            "kind": self.kind,
            "ratio_by_regime": self.ratios,
            "binds_a_trainable_tensor": self.binds,
            "note": self.note,
        }


def load_modality_information(
    results_path: str | Path = "reports/identifiability/results.json",
    manifest_path: str | Path = "reports/identifiability/manifest.json",
    *,
    design: str = "joint_native",
) -> ModalityInformation:
    """Read the per-modality Fisher blocks out of the identifiability run.

    ``design`` defaults to ``joint_native`` because that design carries *both*
    modality blocks, evaluated on the same trajectories --- so the EEG and BOLD
    numbers compared here differ in the observation operator and in nothing else.
    Comparing ``eeg_only`` against ``fmri_only`` would additionally differ in
    which samples were drawn.
    """
    results_path, manifest_path = Path(results_path), Path(manifest_path)
    res = json.loads(results_path.read_text())
    man = json.loads(manifest_path.read_text())
    params = tuple(p["name"] for p in man["parameters"])

    diagonal: dict[str, dict[str, dict[str, float]]] = {}
    modalities: set[str] = set()
    for regime, rd in res["results"]["regimes"].items():
        if design not in rd["designs"]:
            raise KeyError(f"design {design!r} absent from regime {regime!r}")
        by_mod = rd["designs"][design]["fisher_T4"]["I_by_modality"]
        diagonal[regime] = {}
        for mod, matrix in by_mod.items():
            modalities.add(mod)
            if len(matrix) != len(params):
                raise ValueError(
                    f"{regime}/{design}/{mod}: Fisher block is {len(matrix)}x{len(matrix)} but the "
                    f"manifest declares {len(params)} parameters -- refusing to align them by "
                    "position on a guess"
                )
            diagonal[regime][mod] = {p: float(matrix[i][i]) for i, p in enumerate(params)}

    return ModalityInformation(
        diagonal=diagonal,
        params=params,
        regimes=tuple(sorted(diagonal)),
        modalities=tuple(sorted(modalities)),
        provenance={
            "results": str(results_path),
            "manifest": str(manifest_path),
            "design": design,
            "artifact": man.get("artifact"),
            "status": man.get("status"),
            "seed": man.get("seed"),
            "written_at": man.get("written_at"),
            "n_regimes": len(diagonal),
            "epoch_seconds": man.get("instrument", {}).get("epoch_seconds"),
            "n_epochs": man.get("instrument", {}).get("n_epochs"),
            "monte_carlo_disabled": bool(man.get("extra", {}).get("command", {}).get("no_monte_carlo")),
            "non_goals": man.get("non_goals"),
            "statistic": "diagonal of the per-modality expected Fisher block, prior-standardised basis",
        },
    )


def derive_blind_rules(
    info: ModalityInformation, *, negligible_ratio: float = NEGLIGIBLE_RATIO
) -> list[BlindRule]:
    """Every (modality, parameter) pair the measurement says carries no signal.

    A pair qualifies only if it qualifies in **every** regime.  One regime in
    which a modality sees a parameter is enough to refuse the refusal.
    """
    rules: list[BlindRule] = []
    for modality in info.modalities:
        for param in info.params:
            ratios: dict[str, float] = {}
            structural = True
            negligible = True
            for regime in info.regimes:
                v = info.diagonal[regime][modality].get(param, 0.0)
                r = info.ratio(modality, param, regime)
                if r is None:
                    # No modality sees this parameter anywhere: the pair says
                    # nothing about *this* modality's blindness.
                    structural = negligible = False
                    break
                ratios[regime] = r
                if v != 0.0:
                    structural = False
                if r >= negligible_ratio:
                    negligible = False
            if not (structural or negligible):
                continue
            globs,description = LAB_PARAM_TO_GLOBS.get(param, ((), "unmapped laboratory parameter"))
            rules.append(
                BlindRule(
                    modality=modality,
                    param=param,
                    globs=globs,
                    kind="structural_zero" if structural else "negligible",
                    ratios=ratios,
                    binds=bool(globs),
                    note=(
                        f"{description}. "
                        + (
                            "No trainable tensor corresponds to this parameter in "
                            "SC-WBD-001-beta, so the refusal is recorded and binds nothing."
                            if not globs
                            else f"{modality} may not update {list(globs)}."
                        )
                    ),
                )
            )
    return rules
