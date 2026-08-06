"""``TargetingService`` -- the *neuro* half of TMS target selection.

``ARCHITECTURE.md`` Sec. 6 defines this module's entire job:

    given a subject head model and a candidate coil pose, return the induced
    E-field, predicted target engagement, predicted network response, and the
    full uncertainty ledger -- **and refuse when transform or model uncertainty
    dominates the benefit difference**.

and, just as bindingly, what it must never do:

    SC-WBD supplies **no joint commands, no trajectories, no actuation, and no
    stimulation authority**, and must not create a path to any of them.  It
    sits strictly upstream of the "registered external scalp target" node.

So this module has no notion of a robot, a joint, a trajectory, a controller,
or a device driver, and importing it cannot reach one.  Its outputs are
predictions and refusals.

The three refusal paths, and why each exists
--------------------------------------------
``UnderspecifiedPose`` (R01)
    ``thesis_contract.tex`` Sec. 0.5 step 2: *"A pose expressed only as '5 cm
    anterior' is rejected."*  Raised, not returned, because without a declared
    frame and a full 6-DoF transform there is no field to solve and therefore
    no evaluation to attach a decision to.

``Refuse(code="R11")``
    The pose leaves the independently declared feasible set
    :math:`\\mathcal A_{\\rm safe}`.  Returned rather than raised, because the
    consumer must be able to see *which axis* left the set, and the field and
    ledger are still informative.

``Defer(...)``
    Sec. 0.5 step 6: *"If model disagreement or transform uncertainty dominates
    the estimated benefit difference, the output is another calibration
    measurement, a reversible probe, or no recommendation."*  This is a real
    branch with a real threshold read from ``limits/a_safe.toml``, and the
    suggested next measurement is chosen by *which* term dominates: transform
    uncertainty asks for calibration, model disagreement asks for a reversible
    probe.

Claim limits (every public function in this module inherits these)
------------------------------------------------------------------
The predictions are simulations of an analytic phantom or of whatever head
model the caller supplies.  Nothing here has been validated against a measured
cortical field, a measured evoked response, or any clinical outcome.  A
``Recommend`` is a statement that two simulated quantities separated by more
than their simulated uncertainty.  It is not medical advice, not a dose, not
permission to move anything, and not evidence that any admitted operator is
neurally realized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ._compat import (
    Defer,
    FeasibleSet,
    Pose,
    PoseUncertainty,
    ProposedIntervention,
    SafetyLimits,
    UncertaintyLedger,
    Unresolved,
    adjoint,
    exp_se3,
)
from .backends import (
    DEFAULT_PROPAGATORS,
    DEFAULT_RESPONSE_OPERATORS,
    CoilSpec,
    EFieldBackend,
    FieldResolutionUnresolved,
    FieldSolve,
    ImpossiblePlacement,
    NetworkPropagator,
    ResponseOperator,
    resolve_efield_backend,
)
from .frames import ResolvedChain
from .head import HeadModel
from .provenance import ModelProvenance
from .types import (
    Decision,
    EFieldPrediction,
    EngagementDistribution,
    FieldAccuracy,
    LedgerIncomplete,
    NetworkResponse,
    PoseEvaluation,
    PoseRequest,
    Recommend,
    Refuse,
    UtilityStatus,
    full_ledger,
)

__all__ = ["TargetingConfig", "TargetingService", "SessionProtocol"]

_DT = torch.float64
#: Finite-difference step for the pose Jacobian: 0.1 mm and 1e-4 rad.
_TWIST_EPS = 1.0e-4


@dataclass(frozen=True)
class SessionProtocol:
    """The declared *simulated* protocol axes A_safe is checked against.

    These are inputs to a refusal, not a prescription.  Changing them changes
    which axes the feasible set can block; it does not authorise anything.
    """

    pulses_per_session: float = 600.0
    frequency_hz: float = 10.0
    intertrain_interval_s: float = 8.0
    session_duration_s: float = 900.0

    def exposure_axes(self) -> dict[str, float]:
        return {
            "tms.pulses_per_session": float(self.pulses_per_session),
            "tms.frequency_hz": float(self.frequency_hz),
            "tms.intertrain_interval_s": float(self.intertrain_interval_s),
            "protocol.session_duration_s": float(self.session_duration_s),
        }


@dataclass(frozen=True)
class TargetingConfig:
    """Everything the service needs declared before it will answer."""

    target_region: str = "left_dlpfc"
    readout: str = "parcel_state_change"
    readout_units: str = "dimensionless"
    horizon_s: float = 0.25
    comparator: str = "no_stimulation"
    n_draws: int = 128
    seed: int = 0
    protocol: SessionProtocol = field(default_factory=SessionProtocol)
    #: Refinement stride for the numerical-variance estimate.
    refinement_stride: int = 2


# ---------------------------------------------------------------------------
# internal bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FieldBundle:
    """The nominal field plus every perturbation the ledger needs.

    ``perturbed[i]`` is the pair ``(E_plus, E_minus)`` for twist basis
    direction ``i``.  Computing these once and reusing them for the field
    covariance, the engagement variance and the benefit variance is what makes
    the transform-uncertainty term *the same number* everywhere it appears.

    ``solver_variance`` and ``solver_relative_bound`` come from the solver's own
    ledger.  For the charge BEM that is
    ``bem_error_envelope(panel_to_standoff)`` evaluated on the mesh actually
    used -- a step function over gate N7/N8's measured refinement table.  There
    is no fixed percentage anywhere on this path.
    """

    nominal: Tensor
    perturbed: tuple[tuple[Tensor, Tensor], ...]
    coarse: Tensor
    eps: float
    solver_variance: float = 0.0
    solver_relative_bound: float | None = None
    solver_resolution: Mapping[str, float] | None = None
    solver_validity: Mapping[str, Any] = field(default_factory=dict)


class TargetingService:
    """Load once, evaluate many poses. Read-only, and refusal-first.

    Public surface, deliberately small:

    * :meth:`load` -- construct from a model designation and device.
    * :meth:`evaluate_pose` -- one candidate pose in, one
      :class:`~scwbd.runtime.types.PoseEvaluation` out.
    * :attr:`provenance` -- exactly what is being served.

    There is intentionally **no** method that emits a trajectory, a joint
    target, a device setting, a stimulation command, or a protocol for a
    person.  :mod:`tests.runtime.test_no_command_surface` asserts that this
    stays true.
    """

    def __init__(
        self,
        *,
        head_default: HeadModel | None = None,
        coil: CoilSpec | None = None,
        efield_backend: EFieldBackend | None = None,
        response_operators: Sequence[ResponseOperator] = DEFAULT_RESPONSE_OPERATORS,
        propagators: Sequence[NetworkPropagator] = DEFAULT_PROPAGATORS,
        feasible_set: FeasibleSet | None = None,
        config: TargetingConfig | None = None,
        provenance: ModelProvenance | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or TargetingConfig()
        self.coil = coil or CoilSpec.figure_eight()
        self.efield_backend = resolve_efield_backend(efield_backend)
        self.response_operators = tuple(response_operators)
        self.propagators = tuple(propagators)
        self.feasible_set = feasible_set or FeasibleSet()
        self.head_default = head_default
        self.device = device
        if not self.response_operators or not self.propagators:
            raise ValueError("at least one response operator and propagator are required")
        self.provenance = provenance or self._default_provenance()

    # -- construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        model: str = "scwbd-001-beta",
        device: str = "cuda",
        **kwargs: Any,
    ) -> "TargetingService":
        """Load the serving path for ``model``.

        Signature matches ``ARCHITECTURE.md`` Sec. 6.  Checkpoint discovery,
        the ``ClaimManifest``, and the provenance handshake live in
        :mod:`scwbd.runtime.serving`; this classmethod delegates so that
        ``TargetingService.load(...)`` and ``ServedModel.load(...)`` can never
        disagree about what was loaded.
        """
        from .serving import ServedModel

        return ServedModel.load(model, device=device, **kwargs).targeting

    def _default_provenance(self) -> ModelProvenance:
        limits: SafetyLimits = self.feasible_set.limits
        return ModelProvenance(
            weights_status="analytic_backend",
            claim_class="surrogate",
            posterior_class="pseudo",
            efield_backend=getattr(self.efield_backend, "name", "unknown"),
            efield_backend_class=getattr(self.efield_backend, "backend_class", "unknown"),
            efield_gate_evidence=tuple(
                getattr(self.efield_backend, "gate_evidence", ()) or ()
            ),
            response_models=tuple(r.name for r in self.response_operators),
            dynamics_model_classes=tuple(p.name for p in self.propagators),
            a_safe_source=str(limits.meta.get("source_path", "")),
            a_safe_citations=limits.citations(),
            device=self.device,
            torch_version=torch.__version__,
            notes={
                "coil": self.coil.device_id,
                "coil_dipoles": self.coil.n_dipoles,
                "efield_citation": getattr(self.efield_backend, "citation", ""),
            },
        )

    # -- the one public evaluation entry point ------------------------------
    def evaluate_pose(
        self,
        head_model: HeadModel | None = None,
        coil_pose: PoseRequest | Pose | Mapping[str, Any] | None = None,
        *,
        pose_uncertainty: PoseUncertainty | None = None,
        label: str | None = None,
    ) -> PoseEvaluation:
        """Evaluate one candidate coil pose against one head model.

        Parameters
        ----------
        head_model:
            The geometry, parcellation and *declared registration chain* the
            pose must reach.  Falls back to the service default if one was
            configured; there is no implicit average brain.
        coil_pose:
            A :class:`~scwbd.runtime.types.PoseRequest`, a
            :class:`~scwbd.transforms.se3.Pose`, or a mapping.  Anything
            underspecified raises
            :class:`~scwbd.runtime.types.UnderspecifiedPose` (R01).

        Returns
        -------
        PoseEvaluation
            with ``decision`` one of ``Recommend | Defer | Refuse``.

        Claim limits
        ------------
        A prediction about a simulation.  Not a dose, not a protocol, not a
        command, and not applicable to any person.
        """
        head = head_model if head_model is not None else self.head_default
        if head is None:
            raise ValueError(
                "no head model supplied and the service carries no default; "
                "SC-WBD does not substitute an average brain for a missing one"
            )
        req = PoseRequest.coerce(coil_pose if coil_pose is not None else {})
        pose_src = req.resolve()
        name = label or req.label

        pose_head, chain_labels, chain = self._to_head_frame(pose_src, head)
        sigma, bias, sigma_by_scope = self._twist_uncertainty(
            pose_src, chain, req, pose_uncertainty
        )

        try:
            bundle = self._field_bundle(head, pose_head)
        except (FieldResolutionUnresolved, ImpossiblePlacement) as exc:
            # The field solver refused. Two different refusals, two different
            # answers, and neither is a number (ARCHITECTURE.md Sec. 6).
            return self._unresolved_evaluation(
                name=name,
                pose_head=pose_head,
                pose_src=pose_src,
                chain_labels=chain_labels,
                chain=chain,
                sigma=sigma,
                sigma_by_scope=sigma_by_scope,
                head=head,
                exc=exc,
            )
        efield = self._efield_prediction(head, bundle, sigma)

        mask = head.region_mask(self.config.target_region)
        engagement = self._engagement(head, bundle, mask, sigma)
        network, benefit_per_class, class_weights = self._network(head, bundle, mask)

        benefit = float((torch.softmax(class_weights, dim=0) * benefit_per_class).sum())
        model_disagreement = float(
            (
                torch.softmax(class_weights, dim=0) * (benefit_per_class - benefit) ** 2
            ).sum().sqrt()
        )
        transform_sd, gain_sd, numerical_sd = self._benefit_uncertainty(
            head, bundle, mask, sigma
        )
        epistemic = math.hypot(model_disagreement, transform_sd)

        ledger = self._evaluation_ledger(
            head=head,
            chain=chain,
            benefit=benefit,
            model_disagreement=model_disagreement,
            gain_sd=gain_sd,
            numerical_sd=numerical_sd,
            sigma_by_scope=sigma_by_scope,
            bias=bias,
            bundle=bundle,
            mask=mask,
        )

        verdict, proposal = self._safety_verdict(head, pose_head, efield)
        decision = self._decide(
            name=name,
            verdict=verdict,
            benefit=benefit,
            model_disagreement=model_disagreement,
            transform_sd=transform_sd,
            epistemic=epistemic,
        )

        return PoseEvaluation(
            label=name,
            pose=pose_head,
            requested_frame=pose_src.parent,
            transform_chain=chain_labels,
            field_accuracy=self._field_accuracy(efield, bundle, sigma),
            target_engagement=engagement,
            network_response=network,
            utility=UtilityStatus(),
            efield=efield,
            ledger=ledger,
            decision=decision,
            provenance=self.provenance,
            safety_axes_checked=verdict.checked_axes,
            safety_axes_unchecked=verdict.unchecked_declared_axes,
        )

    # -- frames -------------------------------------------------------------
    def _to_head_frame(
        self, pose: Pose, head: HeadModel
    ) -> tuple[Pose, tuple[str, ...], ResolvedChain | None]:
        """Bring ``pose`` into ``head.frame`` through *declared* edges only."""
        if pose.parent == head.frame:
            return (
                pose,
                (
                    f"direct: coil pose declared in {head.frame!r} "
                    f"(method={pose.provenance.get('method', 'unstated')!r})",
                ),
                None,
            )
        transformed, chain = head.frames.transform_pose(pose, into=head.frame)
        return transformed, chain.labels, chain

    def _twist_uncertainty(
        self,
        pose_src: Pose,
        chain: ResolvedChain | None,
        req: PoseRequest,
        override: PoseUncertainty | None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Total right-perturbation twist covariance of the composed pose.

        With ``T = C P`` and right perturbations on each factor,
        ``xi_T = Ad(P^-1) xi_C + xi_P``, so the covariances add under the
        adjoint (body.tex Sec. 2.8).  Chain and pose-measurement errors are
        treated as independent, which is *declared* in the ledger's
        ``validity_domain`` rather than assumed silently.
        """
        declared = override if override is not None else req.uncertainty
        if declared is None and chain is None:
            raise LedgerIncomplete(
                "the coil pose carries no declared uncertainty and reached the "
                "head model through no registration chain, so there is nothing "
                "from which to build a variance decomposition; declare the "
                "tracker/registration error (body.tex Sec. 2.8) rather than "
                "letting it default to zero",
                offending_object=req.label,
            )
        sigma = torch.zeros(6, 6, dtype=_DT)
        bias = torch.zeros(6, dtype=_DT)
        by_scope = {
            "measurement": torch.zeros(6, 6, dtype=_DT),
            "within_session": torch.zeros(6, 6, dtype=_DT),
            "between_session": torch.zeros(6, 6, dtype=_DT),
        }
        if chain is not None:
            Ad = adjoint(torch.linalg.inv(pose_src.matrix))
            sigma = sigma + Ad @ chain.twist_covariance @ Ad.T
            bias = bias + Ad @ chain.twist_bias
            for scope, block in chain.covariance_by_scope().items():
                by_scope[scope] = by_scope[scope] + Ad @ block @ Ad.T
        if declared is not None:
            sigma = sigma + declared.cov
            bias = bias + declared.bias
            by_scope[req.uncertainty_scope] = (
                by_scope[req.uncertainty_scope] + declared.cov
            )
        if float(torch.trace(sigma)) <= 0.0:
            raise LedgerIncomplete(
                "the composed coil pose has zero declared twist covariance; a "
                "pose known exactly is a claim, not a default",
                offending_object=req.label,
            )
        return sigma, bias, by_scope

    # -- field --------------------------------------------------------------
    def _field_bundle(self, head: HeadModel, pose: Pose) -> _FieldBundle:
        solve = self._solve_field
        nominal = solve(head, pose, self.coil)
        eps = _TWIST_EPS
        perturbed: list[tuple[Tensor, Tensor]] = []
        for i in range(6):
            xi = torch.zeros(6, dtype=_DT)
            xi[i] = eps
            plus = Pose(
                pose.matrix @ exp_se3(xi),
                pose.parent,
                pose.child,
                units=pose.units,
                handedness=pose.handedness,
                epoch=pose.epoch,
            )
            minus = Pose(
                pose.matrix @ exp_se3(-xi),
                pose.parent,
                pose.child,
                units=pose.units,
                handedness=pose.handedness,
                epoch=pose.epoch,
            )
            # every perturbed pose is checked too: a Jacobian leg that leaves
            # the solver's validated envelope refuses like any other pose
            perturbed.append(
                (solve(head, plus, self.coil).e, solve(head, minus, self.coil).e)
            )
        coarse = solve(head, pose, self.coil.coarsened(self.config.refinement_stride))
        return _FieldBundle(
            nominal=nominal.e,
            perturbed=tuple(perturbed),
            coarse=coarse.e,
            eps=eps,
            solver_variance=float(nominal.numerical_variance),
            solver_relative_bound=nominal.relative_error_bound,
            solver_resolution=nominal.resolution,
            solver_validity=dict(nominal.validity_domain),
        )

    def _solve_field(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> FieldSolve:
        """One field solve, with the backend's accuracy report attached.

        Backends written before :class:`FieldSolve` existed expose only
        ``solve``; they are wrapped rather than required to change, and a
        backend that reports no numerical accuracy gets ``None`` -- which the
        ledger records as unknown, never as zero.
        """
        if hasattr(self.efield_backend, "solve_field"):
            return self.efield_backend.solve_field(head, pose, coil)
        return FieldSolve(  # pragma: no cover - third-party backend
            e=self.efield_backend.solve(head, pose, coil),
            relative_error_bound=None,
            validity_domain={"solver": getattr(self.efield_backend, "name", "unknown")},
        )

    def _field_jacobian(self, bundle: _FieldBundle) -> Tensor:
        """``[6, N, 3]`` derivative of the field w.r.t. the pose twist."""
        return torch.stack(
            [(p - m) / (2.0 * bundle.eps) for p, m in bundle.perturbed], dim=0
        )

    def _field_covariance(self, bundle: _FieldBundle, sigma: Tensor) -> Tensor:
        """``[N,3,3]`` = transform term + device-gain term.

        The gain term is exact rather than numerical: the induced field is
        linear in ``dI/dt``, so ``dE/dg = E`` for a multiplicative gain ``g``.
        """
        J = self._field_jacobian(bundle)  # [6,N,3]
        cov = torch.einsum("ina,ij,jnb->nab", J, sigma, J)
        gain = float(self.coil.didt_relative_sd)
        if gain > 0.0:
            e = bundle.nominal
            cov = cov + (gain**2) * torch.einsum("na,nb->nab", e, e)
        return cov

    def _field_transform_sd(self, bundle: _FieldBundle, sigma: Tensor) -> float:
        J = self._field_jacobian(bundle)
        mag = torch.linalg.norm(bundle.nominal, dim=-1)
        peak = int(torch.argmax(mag))
        u = bundle.nominal[peak] / mag[peak].clamp_min(1e-30)
        d = torch.stack([J[i, peak] @ u for i in range(6)])
        return float((d @ sigma @ d).clamp_min(0.0).sqrt())

    def _efield_prediction(
        self, head: HeadModel, bundle: _FieldBundle, sigma: Tensor
    ) -> EFieldPrediction:
        cov = self._field_covariance(bundle, sigma)
        mag = torch.linalg.norm(bundle.nominal, dim=-1)
        peak = float(mag.max())
        # Two distinct discretisations, measured separately and added:
        #   * the *coil*'s equivalent-sheet discretisation, measured here by an
        #     actual refinement of the sheet;
        #   * the *conductor* discretisation, measured by the solver and looked
        #     up in gate N7/N8's calibrated envelope. Never a constant.
        coil_numerical = float(
            (torch.linalg.norm(bundle.nominal - bundle.coarse, dim=-1)).max()
        )
        numerical_variance = coil_numerical**2 + float(bundle.solver_variance)
        lo, hi = getattr(self.efield_backend, "discrepancy_fraction", (-0.4, 0.4))
        ledger = full_ledger(
            units="V/m",
            measurement=0.0,
            within_session=float(torch.trace(cov[int(torch.argmax(mag))])) / 3.0,
            between_session=0.0,
            parameter=(float(self.coil.didt_relative_sd) * peak) ** 2,
            model_class=0.0,
            numerical=numerical_variance,
            bias_interval=(lo * peak, hi * peak),
            bias_status="prior_specified_sensitivity",
            model_discrepancy=abs(hi - lo) * peak / 2.0,
            validity_domain={
                **head.validity_domain(),
                **dict(bundle.solver_validity),
                "efield_backend": getattr(self.efield_backend, "name", "unknown"),
                "conductivity_enters_solution": bool(
                    bundle.solver_validity.get(
                        "conductivity_enters_solution",
                        getattr(self.efield_backend, "backend_class", "") != "analytic",
                    )
                ),
                "coil_discretisation_numerical_sd_v_per_m": coil_numerical,
                "solver_relative_error_bound": bundle.solver_relative_bound,
                "solver_numerical_variance": float(bundle.solver_variance),
                "note": (
                    "the numerical term is measured, not asserted: the coil "
                    "sheet by an explicit refinement here, and the conductor "
                    "discretisation by the solver's own calibrated envelope "
                    "(gate N7/N8). A spherical backend is "
                    "conductivity-independent and so reports no tissue-parameter "
                    "variance; its discrepancy against a subject-specific solve "
                    "is a separate, prior-specified range and a numerical gate "
                    "does not narrow it."
                ),
            },
            notes="E-field ledger",
        )
        return EFieldPrediction(
            frame=head.frame,
            support="cortical_surface_vertex",
            vector=bundle.nominal,
            covariance=cov,
            ledger=ledger,
            backend=getattr(self.efield_backend, "name", "unknown"),
            backend_class=getattr(self.efield_backend, "backend_class", "unknown"),
            is_trained_artifact=bool(
                getattr(self.efield_backend, "is_trained_artifact", False)
            ),
        )

    def _field_accuracy(
        self, efield: EFieldPrediction, bundle: _FieldBundle, sigma: Tensor
    ) -> FieldAccuracy:
        """Level 1 of the validation ladder, reporting what actually backs it.

        ``validation_status`` is raised to ``cross_solver`` only when the
        backend names a numerical gate that compared it against an
        *independent* reference.  Gate N8 does exactly that at contact geometry
        (0.73 % mean relative error against an axial spectral reference), which
        is why the contact regime is not a deferral.  It remains a statement
        about the *discretisation*, not about whether a sphere is a head.
        """
        gates = tuple(getattr(self.efield_backend, "gate_evidence", ()) or ())
        validated = ("coil_discretisation_refinement", *gates)
        return FieldAccuracy(
            peak_v_per_m=efield.peak(),
            peak_sd_v_per_m=efield.peak_sd(),
            transform_sd_v_per_m=self._field_transform_sd(bundle, sigma),
            validated_against=validated,
            validation_status="cross_solver" if gates else "solver_refinement_only",
            solver_relative_error_bound=bundle.solver_relative_bound,
            near_source_resolution=dict(bundle.solver_resolution or {}),
        )

    # -- when the solver refuses --------------------------------------------
    def _unresolved_evaluation(
        self,
        *,
        name: str,
        pose_head: Pose,
        pose_src: Pose,
        chain_labels: tuple[str, ...],
        chain: ResolvedChain | None,
        sigma: Tensor,
        sigma_by_scope: Mapping[str, Tensor],
        head: HeadModel,
        exc: Exception,
    ) -> PoseEvaluation:
        """Build the evaluation for a pose whose field could not be computed.

        Every downstream quantity comes back as ``Unresolved`` carrying the
        solver's own reason and remedy -- never a zero, never a number.  The
        decision is:

        * ``Defer`` when the refusal was
          ``ChargeBEM.assert_resolves_sources`` -- the geometry is physical and
          the mesh is the gap, so the informative next step is to refine the
          model, not to act;
        * ``Refuse(code="R06")`` when the coil is not outside the head, because
          that is not a placement at all.
        """
        resolved = isinstance(exc, FieldResolutionUnresolved)
        reason = str(exc)
        remedy = str(getattr(exc, "remedy", ""))
        detail = dict(
            getattr(exc, "resolution", None) or getattr(exc, "detail", None) or {}
        )
        unresolved = Unresolved(
            reason=reason + (f"  remedy: {remedy}" if remedy else ""),
            missing=("induced_efield",),
        )

        if resolved:
            decision: Decision = Defer(
                reason=(
                    "the induced-field solver refused: its discretisation does "
                    "not resolve the near-source field at this pose, and gate "
                    "N7/N8 measured non-monotonic refinement beyond that "
                    "envelope, so no error bound can be attached to a number "
                    f"computed here. {reason}"
                ),
                # Refining a mesh is a modelling step, not a measurement on a
                # subject and not a probe. Saying "reversible_probe" here would
                # propose an intervention to fix a discretisation.
                suggested_action="no_action",
                detail={k: float(v) for k, v in detail.items() if _is_number(v)},
            )
        else:
            decision = Refuse(
                code="R06",
                reason=reason,
                remedy=remedy,
                offending=name,
                violations=(reason,),
            )

        return PoseEvaluation(
            label=name,
            pose=pose_head,
            requested_frame=pose_src.parent,
            transform_chain=chain_labels,
            field_accuracy=unresolved,
            target_engagement=unresolved,
            network_response=unresolved,
            utility=UtilityStatus(),
            efield=unresolved,
            ledger=self._pose_only_ledger(
                head=head,
                chain=chain,
                sigma=sigma,
                sigma_by_scope=sigma_by_scope,
                reason=reason,
            ),
            decision=decision,
            provenance=self.provenance,
            safety_axes_checked=(),
            safety_axes_unchecked=tuple(
                sorted(
                    {s.key for s in self.feasible_set.limits.for_modality("tms")}
                    | {s.key for s in self.feasible_set.limits.for_modality("protocol")}
                )
            ),
        )

    def _pose_only_ledger(
        self,
        *,
        head: HeadModel,
        chain: ResolvedChain | None,
        sigma: Tensor,
        sigma_by_scope: Mapping[str, Tensor],
        reason: str,
    ) -> UncertaintyLedger:
        """The ledger of what is still known when the field is not: the pose.

        In metres, because there is no readout to express anything else in.
        The three terms that would have described the field are marked *not
        applicable* in ``validity_domain`` rather than being written as zero --
        a zero would assert those terms are absent, and they are simply not
        computed.
        """
        return full_ledger(
            units="m",
            measurement=float(torch.trace(sigma_by_scope["measurement"][:3, :3])),
            within_session=float(torch.trace(sigma_by_scope["within_session"][:3, :3])),
            between_session=float(
                torch.trace(sigma_by_scope["between_session"][:3, :3])
            ),
            parameter=0.0,
            model_class=0.0,
            numerical=0.0,
            bias_interval=(
                -float(torch.linalg.norm(sigma[:3, :3].diagonal().sqrt())) - 1e-9,
                float(torch.linalg.norm(sigma[:3, :3].diagonal().sqrt())) + 1e-9,
            ),
            bias_status="prior_specified_sensitivity",
            validity_domain={
                **head.validity_domain(),
                "registration_chain": list(chain.labels) if chain is not None else [],
                "field_unresolved_reason": reason,
                "terms_not_applicable": ["parameter", "model_class", "numerical"],
                "terms_not_applicable_reason": (
                    "the field solve refused, so nothing downstream of it was "
                    "computed; these are not zero, they are not applicable"
                ),
                "human_use_authorized": False,
            },
            notes=(
                "pose-only ledger for an evaluation whose induced field could "
                "not be computed"
            ),
        )

    # -- engagement ---------------------------------------------------------
    def _drive_means(self, head: HeadModel, efield: Tensor, mask: Tensor) -> Tensor:
        """``[M]`` mean drive over the target region, per response operator."""
        n = mask.sum().clamp_min(1)
        return torch.stack(
            [
                op.drive(efield, head.cortex_normals, mask).sum() / n
                for op in self.response_operators
            ]
        )

    def _engagement(
        self, head: HeadModel, bundle: _FieldBundle, mask: Tensor, sigma: Tensor
    ) -> EngagementDistribution:
        mean = self._drive_means(head, bundle.nominal, mask)  # [M]
        J = torch.stack(
            [
                (
                    self._drive_means(head, p, mask) - self._drive_means(head, m, mask)
                )
                / (2.0 * bundle.eps)
                for p, m in bundle.perturbed
            ],
            dim=0,
        )  # [6,M]
        var_transform = torch.einsum("im,ij,jm->m", J, sigma, J).clamp_min(0.0)
        gain = float(self.coil.didt_relative_sd)
        d_gain = (
            self._drive_means(head, bundle.nominal * (1.0 + gain), mask) - mean
        ) if gain > 0 else torch.zeros_like(mean)
        var = var_transform + d_gain**2

        g = torch.Generator().manual_seed(int(self.config.seed))
        draws = torch.randn(
            len(self.response_operators), int(self.config.n_draws),
            generator=g, dtype=_DT,
        )
        samples = mean.unsqueeze(-1) + var.sqrt().unsqueeze(-1) * draws

        weights = torch.zeros(len(self.response_operators), dtype=_DT)
        per_model_mean = mean
        between = float(per_model_mean.var(unbiased=False))
        ledger = full_ledger(
            units="dimensionless",
            measurement=0.0,
            within_session=float(var_transform.mean()),
            between_session=0.0,
            parameter=float((d_gain**2).mean()),
            model_class=between,
            numerical=float(
                (
                    self._drive_means(head, bundle.coarse, mask) - mean
                ).pow(2).mean()
            ),
            bias_interval=(-float(mean.max()) * 0.4, float(mean.max()) * 0.4),
            bias_status="prior_specified_sensitivity",
            model_discrepancy=float(between**0.5),
            validity_domain={
                **head.validity_domain(),
                "target_region": self.config.target_region,
                "mechanism": "unresolved; candidate operators retained, not selected",
            },
            notes="target-engagement ledger",
        )
        return EngagementDistribution(
            target=self.config.target_region,
            units="dimensionless",
            response_models=tuple(op.name for op in self.response_operators),
            mechanistic_status=tuple(
                str(op.mechanistic_status) for op in self.response_operators
            ),
            log_weights=weights,
            samples=samples,
            ledger=ledger,
        )

    # -- network ------------------------------------------------------------
    def _class_names(self) -> tuple[str, ...]:
        return tuple(
            f"{op.name}+{prop.name}"
            for op in self.response_operators
            for prop in self.propagators
        )

    def _network_profiles(
        self, head: HeadModel, efield: Tensor, mask: Tensor
    ) -> Tensor:
        """``[K, P]`` predicted parcel change under every model class."""
        pmat = head.parcel_matrix()
        rows: list[Tensor] = []
        for op in self.response_operators:
            drive = op.drive(efield, head.cortex_normals, mask)
            u = pmat @ drive
            for prop in self.propagators:
                rows.append(prop.propagate(u, head.connectivity, self.config.horizon_s))
        return torch.stack(rows, dim=0)

    def _target_parcels(self, head: HeadModel, mask: Tensor) -> Tensor:
        idx = torch.unique(head.vertex_parcel[mask])
        sel = torch.zeros(head.n_parcels, dtype=torch.bool)
        sel[idx] = True
        return sel

    def _benefit_from_profiles(self, profiles: Tensor, target: Tensor) -> Tensor:
        """``[K]`` benefit per model class: mean predicted change on target."""
        return profiles[:, target].mean(dim=1)

    def _network(
        self, head: HeadModel, bundle: _FieldBundle, mask: Tensor
    ) -> tuple[NetworkResponse, Tensor, Tensor]:
        profiles = self._network_profiles(head, bundle.nominal, mask)
        target = self._target_parcels(head, mask)
        benefit = self._benefit_from_profiles(profiles, target)
        weights = torch.zeros(profiles.shape[0], dtype=_DT)
        w = torch.softmax(weights, dim=0).unsqueeze(-1)
        mean = (w * profiles).sum(0)
        spread = float((((w * (profiles - mean) ** 2).sum(0)).sqrt()).max())
        coarse_profiles = self._network_profiles(head, bundle.coarse, mask)
        ledger = full_ledger(
            units=self.config.readout_units,
            measurement=0.0,
            within_session=0.0,
            between_session=0.0,
            parameter=0.0,
            model_class=spread**2,
            numerical=float((profiles - coarse_profiles).pow(2).mean()),
            bias_interval=(-abs(float(mean.max())), abs(float(mean.max()))),
            bias_status="prior_specified_sensitivity",
            model_discrepancy=spread,
            validity_domain={
                **head.validity_domain(),
                "horizon_s": self.config.horizon_s,
                "propagators": [p.name for p in self.propagators],
                "mechanistic_status": "surrogate",
                "note": (
                    "connectivity is a topology prior, not a measured effective "
                    "connectivity (thesis Sec. 2.5); the propagators are "
                    "surrogates retained under comparison, not selected"
                ),
            },
            notes="network-response ledger",
        )
        response = NetworkResponse(
            readout=self.config.readout,
            units=self.config.readout_units,
            horizon_s=self.config.horizon_s,
            model_classes=self._class_names(),
            log_weights=weights,
            per_model=profiles,
            elements=head.parcels,
            ledger=ledger,
        )
        return response, benefit, weights

    def _benefit_uncertainty(
        self, head: HeadModel, bundle: _FieldBundle, mask: Tensor, sigma: Tensor
    ) -> tuple[float, float, float]:
        """(transform sd, device-gain sd, numerical sd) of the benefit."""
        target = self._target_parcels(head, mask)
        w = torch.softmax(torch.zeros(len(self.response_operators) * len(self.propagators), dtype=_DT), dim=0)

        def benefit_of(e: Tensor) -> float:
            profiles = self._network_profiles(head, e, mask)
            return float((w * self._benefit_from_profiles(profiles, target)).sum())

        nominal = benefit_of(bundle.nominal)
        J = torch.tensor(
            [
                (benefit_of(p) - benefit_of(m)) / (2.0 * bundle.eps)
                for p, m in bundle.perturbed
            ],
            dtype=_DT,
        )
        transform_sd = float((J @ sigma @ J).clamp_min(0.0).sqrt())
        gain = float(self.coil.didt_relative_sd)
        gain_sd = (
            abs(benefit_of(bundle.nominal * (1.0 + gain)) - nominal) if gain > 0 else 0.0
        )
        numerical_sd = abs(benefit_of(bundle.coarse) - nominal)
        return transform_sd, gain_sd, numerical_sd

    # -- ledger -------------------------------------------------------------
    def _evaluation_ledger(
        self,
        *,
        head: HeadModel,
        chain: ResolvedChain | None,
        benefit: float,
        model_disagreement: float,
        gain_sd: float,
        numerical_sd: float,
        sigma_by_scope: Mapping[str, Tensor],
        bias: Tensor,
        bundle: _FieldBundle,
        mask: Tensor,
    ) -> UncertaintyLedger:
        """The evaluation ledger, in the units of the network readout.

        Every component of ``VARIANCE_COMPONENTS`` is populated, and each is
        traced to something declared: the per-scope twist covariances come from
        the declared registration edges, ``parameter`` from the declared device
        gain prior, ``model_class`` from the retained model classes, and
        ``numerical`` from an actual coil-discretisation refinement.
        """
        target = self._target_parcels(head, mask)
        w = torch.softmax(
            torch.zeros(len(self.response_operators) * len(self.propagators), dtype=_DT),
            dim=0,
        )

        def benefit_of(e: Tensor) -> float:
            return float(
                (w * self._benefit_from_profiles(self._network_profiles(head, e, mask), target)).sum()
            )

        J = torch.tensor(
            [
                (benefit_of(p) - benefit_of(m)) / (2.0 * bundle.eps)
                for p, m in bundle.perturbed
            ],
            dtype=_DT,
        )
        per_scope = {
            k: float((J @ v @ J).clamp_min(0.0)) for k, v in sigma_by_scope.items()
        }
        bias_shift = float(J @ bias)
        lo, hi = getattr(self.efield_backend, "discrepancy_fraction", (-0.4, 0.4))
        model_bias = (lo * abs(benefit), hi * abs(benefit))
        bias_interval = (
            min(model_bias) - abs(bias_shift),
            max(model_bias) + abs(bias_shift),
        )
        return full_ledger(
            units=self.config.readout_units,
            measurement=per_scope["measurement"],
            within_session=per_scope["within_session"],
            between_session=per_scope["between_session"],
            parameter=gain_sd**2,
            model_class=model_disagreement**2,
            numerical=numerical_sd**2,
            bias_interval=bias_interval,
            bias_status="prior_specified_sensitivity",
            model_discrepancy=model_disagreement,
            validity_domain={
                **head.validity_domain(),
                "readout": self.config.readout,
                "comparator": self.config.comparator,
                "horizon_s": self.config.horizon_s,
                "registration_chain": list(chain.labels) if chain is not None else [],
                "chain_depends_on_assumed": (
                    bool(chain.depends_on_assumed) if chain is not None else False
                ),
                "chain_and_pose_errors_assumed_independent": True,
                "efield_backend_class": getattr(
                    self.efield_backend, "backend_class", "unknown"
                ),
                "human_use_authorized": False,
            },
            notes=(
                "pose-evaluation ledger; bias is a prior-specified sensitivity "
                "range combining the field backend's declared discrepancy with "
                "the propagated systematic twist bias"
            ),
        )

    # -- A_safe -------------------------------------------------------------
    def _safety_verdict(
        self, head: HeadModel, pose: Pose, efield: EFieldPrediction
    ) -> tuple[Any, ProposedIntervention]:
        gap_mm = head.scalp_distance(pose.t) * 1000.0
        exposure = {
            "tms.peak_efield_v_per_m": efield.peak(),
            "tms.coil_scalp_distance_mm": gap_mm,
            **self.config.protocol.exposure_axes(),
        }
        proposal = ProposedIntervention(
            label=f"{head.subject_id}:{self.config.target_region}",
            modality="tms",
            exposure=exposure,
            # The pose reached here through declared, checked transforms and a
            # full 6-DoF specification; that is exactly what "certified" means
            # in scwbd.intervene.safety, and nothing more.
            pose_certified=True,
            reversible=True,
            provenance={"efield_backend": efield.backend, "frame": pose.parent},
        )
        return self.feasible_set.contains(proposal), proposal

    # -- the decision -------------------------------------------------------
    def _decide(
        self,
        *,
        name: str,
        verdict: Any,
        benefit: float,
        model_disagreement: float,
        transform_sd: float,
        epistemic: float,
    ) -> Decision:
        if not verdict.feasible:
            return Refuse(
                code="R11",
                reason=(
                    "candidate pose leaves the independently declared feasible "
                    "set A_safe: " + "; ".join(str(v) for v in verdict.violations)
                ),
                remedy=(
                    "restrict the search to A_safe; a proposal outside it is "
                    "not repaired by projection, because that would hide that "
                    "the objective wanted to leave the set"
                ),
                offending=name,
                violations=tuple(str(v) for v in verdict.violations),
            )

        limits: SafetyLimits = self.feasible_set.limits
        n_models = len(self.response_operators)
        if n_models < limits.min_candidate_models():
            return Defer(
                reason=(
                    f"only {n_models} candidate response operator(s) retained; "
                    "the coupling mechanism is unresolved and a comparison "
                    "under a single response model is not admissible"
                ),
                suggested_action="additional_calibration_measurement",
                detail={"n_response_models": float(n_models)},
            )

        ratio = limits.defer_ratio()
        if epistemic >= ratio * abs(benefit):
            transform_dominates = transform_sd >= model_disagreement
            return Defer(
                reason=(
                    f"epistemic uncertainty {epistemic:.4g} dominates the "
                    f"estimated benefit difference {benefit:.4g} against the "
                    f"{self.config.comparator} comparator (threshold ratio "
                    f"{ratio:g}); "
                    + (
                        "transform uncertainty is the larger term, so the "
                        "informative next step is a calibration measurement of "
                        "the registration chain, not a stimulation"
                        if transform_dominates
                        else "model-class disagreement is the larger term, so "
                        "the informative next step is a reversible probe that "
                        "discriminates the retained response operators"
                    )
                ),
                suggested_action=(
                    "additional_calibration_measurement"
                    if transform_dominates
                    else "reversible_probe"
                ),
                detail={
                    "benefit_margin": float(benefit),
                    "epistemic": float(epistemic),
                    "transform_sd": float(transform_sd),
                    "model_disagreement": float(model_disagreement),
                    "threshold_ratio": float(ratio),
                },
            )

        return Recommend(
            label=name,
            rationale=(
                f"predicted change on {self.config.target_region} separates "
                f"from {self.config.comparator} by {benefit:.4g} against an "
                f"epistemic uncertainty of {epistemic:.4g}"
            ),
            benefit_margin=float(benefit),
            epistemic_uncertainty=float(epistemic),
            comparator=self.config.comparator,
            basis=limits.citations(),
        )

    # -- misc ---------------------------------------------------------------
    def with_config(self, **kwargs: Any) -> "TargetingService":
        """A copy with a modified :class:`TargetingConfig`. Never mutates."""
        return TargetingService(
            head_default=self.head_default,
            coil=self.coil,
            efield_backend=self.efield_backend,
            response_operators=self.response_operators,
            propagators=self.propagators,
            feasible_set=self.feasible_set,
            config=replace(self.config, **kwargs),
            provenance=self.provenance,
            device=self.device,
        )

    def read(self, name: str) -> Unresolved:
        """Any read this service does not support returns ``Unresolved``.

        Included deliberately: a consumer that asks the targeting service for
        something it does not model (a joint angle, a dose, a clinical score)
        gets a reason, never a number.
        """
        return Unresolved(
            reason=(
                f"{name!r} is not a quantity the SC-WBD targeting service "
                "models; it returns E-field, target engagement, network "
                "response, an uncertainty ledger and a decision, and nothing "
                "downstream of the registered external scalp target"
            ),
            missing=(name,),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TargetingService({self.provenance.label})"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
