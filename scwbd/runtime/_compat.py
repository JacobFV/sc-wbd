"""Adapters to the modules other agents own.

``scwbd.runtime`` is the boundary where a research model meets a robot aimed at
a human head, and it is built while agents A/C/D/E/F/G/I are still landing the
modules it depends on.  Every cross-module import therefore goes through this
file, in one of two modes:

* **required** -- the module is part of the binding contract in
  ``ARCHITECTURE.md`` and is imported directly from its deepest concrete module
  path (``scwbd.transforms.se3``, not ``scwbd.transforms``) so a package
  ``__init__`` being rewritten under us cannot break the runtime;
* **optional** -- looked up with :func:`optional`, returning ``None`` when the
  owning agent has not landed it.  A missing optional is never silently
  replaced by a plausible number: it changes the *provenance* of the answer
  (see :mod:`scwbd.runtime.provenance`) so that a consumer can tell an analytic
  fallback from a trained artifact.

Claim limit: nothing in this module computes anything.  It only decides which
implementation is available and records that choice.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

__all__ = [
    "optional",
    "Pose",
    "PoseUncertainty",
    "ChainUncertainty",
    "propagate_chain",
    "adjoint",
    "exp_se3",
    "TransformError",
    "FrameMismatchError",
    "Ledger",
    "InterventionRefusal",
    "MechanisticUncertainty",
    "NetworkEffect",
    "PhysicalDose",
    "TargetEngagement",
    "SIMULATION_ONLY_NOTICE",
    "CompilerRefusal",
    "Defer",
    "FeasibleSet",
    "NoRecommendation",
    "ProposedIntervention",
    "SafetyLimits",
    "UncertaintyLedger",
    "VARIANCE_COMPONENTS",
    "Unresolved",
    "ClaimManifest",
    "figure_eight_coil",
    "tms_pulse_biphasic",
    "analytic_sphere_efield",
    "coil_dipoles_in_head_frame",
    "efield_from_coil",
    "spherical_head_model",
    "charge_bem",
    "layered_sphere_bem",
    "graded_icosphere",
    "tri_mesh",
    "icosphere",
    "bem_error_envelope",
    "max_panel_to_standoff",
    "impossible_geometry",
    "is_resolution_refusal",
    "anatomy_prior",
]


def optional(module: str, name: str) -> Any:
    """Return ``module.name`` if the owning agent has landed it, else ``None``.

    A ``None`` here must always change what the runtime *reports*, never only
    what it computes.
    """
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:  # pragma: no cover - depends on which agents have landed
        return None


# -- required: agent D, transforms ------------------------------------------
from scwbd.transforms.errors import (  # noqa: E402
    FrameMismatchError,
    TransformError,
)
from scwbd.transforms.se3 import Pose, adjoint, exp_se3  # noqa: E402
from scwbd.transforms.uncertainty import (  # noqa: E402
    ChainUncertainty,
    PoseUncertainty,
    propagate_chain,
)

# -- required: agent G, intervention + A_safe --------------------------------
from scwbd.intervene.base import (  # noqa: E402
    SIMULATION_ONLY_NOTICE,
    InterventionRefusal,
    Ledger,
    MechanisticUncertainty,
    NetworkEffect,
    PhysicalDose,
    TargetEngagement,
)
from scwbd.intervene.safety import (  # noqa: E402
    CompilerRefusal,
    Defer,
    FeasibleSet,
    NoRecommendation,
    ProposedIntervention,
    SafetyLimits,
)

# -- required: agent A, ledger ----------------------------------------------
from scwbd.schema.ledger import (  # noqa: E402
    VARIANCE_COMPONENTS,
    UncertaintyLedger,
)

# -- optional: agent F's Unresolved (mirrored locally if absent) -------------
Unresolved = optional("scwbd.observe.base", "Unresolved")
if Unresolved is None:  # pragma: no cover - agent F has landed this
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Unresolved:  # type: ignore[no-redef]
        """A read that cannot be supported. Never a number, never a zero."""

        reason: str
        missing: tuple[str, ...] = ()

        def __bool__(self) -> bool:
            return False


# -- optional: agent A's claim manifest --------------------------------------
ClaimManifest = optional("scwbd.schema.claims", "ClaimManifest")

# -- optional: agent G's coil geometry and E-field solvers -------------------
figure_eight_coil = optional("scwbd.intervene.tms.coil", "FigureEightCoil")
tms_pulse_biphasic = optional("scwbd.intervene.tms.coil", "biphasic")

#: Agent G / Faraday own the TMS field physics.  These are the gated
#: implementations (``reports/intervene/N6_induced_efield.md``,
#: ``N8_induced_efield_contact.md``); the runtime prefers them and falls back to
#: the closed-form backend in :mod:`scwbd.runtime.backends` only when they are
#: absent, *saying so* in
#: :class:`~scwbd.runtime.provenance.ModelProvenance.efield_backend_class`.
analytic_sphere_efield = optional("scwbd.intervene.tms.efield", "analytic_sphere_efield")
coil_dipoles_in_head_frame = optional(
    "scwbd.intervene.tms.efield", "coil_dipoles_in_head_frame"
)
efield_from_coil = optional("scwbd.intervene.tms.efield", "efield_from_coil")
spherical_head_model = optional("scwbd.intervene.tms.efield", "SphericalHeadModel")
charge_bem = optional("scwbd.intervene.tms.efield", "ChargeBEM")
layered_sphere_bem = optional("scwbd.intervene.tms.efield", "LayeredSphereBEM")
graded_icosphere = optional("scwbd.intervene.tms.efield", "graded_icosphere")
tri_mesh = optional("scwbd.intervene.tms.efield", "TriMesh")
icosphere = optional("scwbd.intervene.tms.efield", "icosphere")
bem_error_envelope = optional("scwbd.intervene.tms.efield", "bem_error_envelope")
max_panel_to_standoff = optional("scwbd.intervene.tms.efield", "MAX_PANEL_TO_STANDOFF")
impossible_geometry = optional("scwbd.intervene.tms.efield", "ImpossibleGeometry")


def is_resolution_refusal(exc: BaseException) -> bool:
    """Distinguish "the mesh is too coarse" from "the coil is inside the head".

    Both arrive as ``ImpossibleGeometry`` (refusal ``R06``), and the runtime
    must answer differently: an unresolved near field is a *modelling* gap the
    caller can close, so it becomes ``Defer``; a source inside the conductor is
    not a placement at all, so it becomes ``Refuse``.  The discriminator is the
    refusal's own ``offending_object``, which
    ``ChargeBEM.assert_resolves_sources`` populates with the measured
    resolution dict.
    """
    obj = getattr(exc, "offending_object", None)
    return isinstance(obj, Mapping) and "panel_to_standoff" in obj

# -- optional: agent C/I anatomy prior ---------------------------------------
anatomy_prior = optional("scwbd.foundation.anatomy", "load_anatomy")
