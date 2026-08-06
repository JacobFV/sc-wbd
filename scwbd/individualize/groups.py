"""Parameter groups: the granularity at which individualization is decided.

Individualization is **per parameter group, not all-or-nothing**.  The reason is
measured, not asserted: agent Fisher's committed benchmark
(``reports/identifiability/results.json``) shows that in the reference regime
the theta-profile likelihood information is ``16.008`` for EEG alone, ``16.008``
for joint EEG+fMRI and ``2.93e-06`` for fMRI alone -- five and a half orders of
magnitude apart for the *same* four parameters.  A single "can we individualize
this patient?" flag would have to answer for both, and would therefore be wrong
about one of them.

Two kinds of group live here.

``likelihood`` groups
    Blocks of ``eta = (theta, ell, rho)`` of ``scwbd.infer.linear_gaussian``.
    Their identifiability is *computed* from the expected Fisher information of
    the design the patient's modalities imply.

``anatomical`` groups
    Quantities individualised from structural data outside the dynamical
    likelihood -- head geometry, lead fields, tract-length priors.  Their status
    is **presence-determined**: it follows from whether the patient has the
    modality, not from a Fisher computation.  That distinction is carried in
    :attr:`ParameterGroup.evidence_kind` and printed in every report, so a
    presence-determined "identifiable" is never mistaken for a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from scwbd.infer.linear_gaussian import PARAM_INDEX, PARAM_NAMES

from .availability import Modality

__all__ = [
    "GROUPS",
    "ANATOMICAL_GROUPS",
    "LIKELIHOOD_GROUPS",
    "GroupKind",
    "ParameterGroup",
    "group_by_name",
    "groups_for_parameter",
]

GroupKind = Literal["likelihood", "anatomical"]


@dataclass(frozen=True)
class ParameterGroup:
    """A set of parameters whose individualization is decided together."""

    name: str
    parameters: tuple[str, ...]
    kind: GroupKind
    #: How the status of this group is established.
    evidence_kind: Literal["fisher_information", "modality_presence"]
    #: Modalities that, if present, could inform this group.  For likelihood
    #: groups this is *documentation*: the status comes from the Fisher
    #: computation, and this field is never allowed to override it.
    informed_by: tuple[Modality, ...]
    clinical_meaning: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind == "likelihood":
            unknown = [p for p in self.parameters if p not in PARAM_INDEX]
            if unknown:
                raise ValueError(
                    f"group {self.name!r} names parameters absent from the "
                    f"reference model: {unknown}; known: {list(PARAM_NAMES)}"
                )
            if self.evidence_kind != "fisher_information":
                raise ValueError(
                    f"group {self.name!r} is a likelihood group but claims "
                    f"evidence_kind={self.evidence_kind!r}; a likelihood group's "
                    "status must be measured, not asserted from presence"
                )
        elif self.evidence_kind != "modality_presence":
            raise ValueError(
                f"anatomical group {self.name!r} must be presence-determined"
            )

    @property
    def index(self) -> list[int]:
        """Indices into ``eta``.  Empty for anatomical groups."""
        return [PARAM_INDEX[p] for p in self.parameters if p in PARAM_INDEX]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "kind": self.kind,
            "evidence_kind": self.evidence_kind,
            "informed_by": list(self.informed_by),
            "clinical_meaning": self.clinical_meaning,
            "description": self.description,
        }


LIKELIHOOD_GROUPS: tuple[ParameterGroup, ...] = (
    ParameterGroup(
        name="coupling",
        parameters=("a21", "a32", "a13"),
        kind="likelihood",
        evidence_kind="fisher_information",
        informed_by=("eeg", "meg", "fmri"),
        clinical_meaning=(
            "directed effective coupling gains between regions -- what a "
            "stimulation target, a seizure-propagation prediction or a "
            "disconnection claim is actually about"
        ),
        description="theta block, coupling gains (1/s)",
    ),
    ParameterGroup(
        name="conduction_delay",
        parameters=("tau",),
        kind="likelihood",
        evidence_kind="fisher_information",
        informed_by=("eeg", "meg", "fmri"),
        clinical_meaning=(
            "network conduction delay (s) -- sets the phase relationships any "
            "oscillation-timed or closed-loop protocol depends on"
        ),
        description="theta block, conduction delay",
    ),
    ParameterGroup(
        name="eeg_lead_field",
        parameters=("gain_eeg", "tilt_eeg"),
        kind="likelihood",
        evidence_kind="fisher_information",
        informed_by=("eeg", "meg"),
        clinical_meaning=(
            "global gain and electrode-placement tilt of the patient's EEG "
            "lead field -- the sensor-to-source map"
        ),
        description="ell block, observation nuisances of the fast channel",
    ),
    ParameterGroup(
        name="hemodynamic",
        parameters=("beta_hrf", "c_under", "gain_bold"),
        kind="likelihood",
        evidence_kind="fisher_information",
        informed_by=("fmri",),
        clinical_meaning=(
            "the patient's haemodynamic response: cascade time constant, "
            "undershoot weight and BOLD gain -- vascular, not neural"
        ),
        description="rho block, observation nuisances of the slow channel",
    ),
)

ANATOMICAL_GROUPS: tuple[ParameterGroup, ...] = (
    ParameterGroup(
        name="head_geometry",
        parameters=(),
        kind="anatomical",
        evidence_kind="modality_presence",
        informed_by=("structural_mri",),
        clinical_meaning=(
            "the patient's own scalp/skull/brain surfaces and source space, "
            "instead of a template head"
        ),
        description=(
            "individualised by segmentation of the structural scan; the "
            "downstream lead field is then this patient's, which is a real "
            "personalisation even when no dynamical parameter can be fitted"
        ),
    ),
    ParameterGroup(
        name="structural_connectivity_prior",
        parameters=(),
        kind="anatomical",
        evidence_kind="modality_presence",
        informed_by=("dmri",),
        clinical_meaning=(
            "tract-derived prior on which connections exist and how long they "
            "are -- a PRIOR on coupling and delay, never a measurement of them"
        ),
        description=(
            "dMRI narrows the coupling/delay prior; it does not make the "
            "coupling likelihood informative, and the two must not be conflated"
        ),
    ),
)

GROUPS: tuple[ParameterGroup, ...] = LIKELIHOOD_GROUPS + ANATOMICAL_GROUPS

_BY_NAME = {g.name: g for g in GROUPS}


def group_by_name(name: str) -> ParameterGroup:
    if name not in _BY_NAME:
        raise KeyError(f"unknown parameter group {name!r}; known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def groups_for_parameter(parameter: str) -> tuple[ParameterGroup, ...]:
    return tuple(g for g in GROUPS if parameter in g.parameters)


def _coverage_check() -> None:
    """Every model parameter belongs to exactly one likelihood group.

    A parameter that belongs to no group would silently escape the profile --
    it would never be reported as unidentifiable, because nothing would ask.
    """
    seen: dict[str, list[str]] = {}
    for g in LIKELIHOOD_GROUPS:
        for p in g.parameters:
            seen.setdefault(p, []).append(g.name)
    missing = [p for p in PARAM_NAMES if p not in seen]
    doubled = {p: gs for p, gs in seen.items() if len(gs) > 1}
    if missing or doubled:
        raise AssertionError(
            f"parameter group partition is broken: unassigned={missing}, "
            f"multiply-assigned={doubled}"
        )


_coverage_check()
