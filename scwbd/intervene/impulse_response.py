"""TMS impulse response: E-field -> latent drive -> predicted trajectory.

This is the join that did not exist.  Both ends were already built and neither
reached the other:

* ``scwbd.intervene.tms.efield`` computes an induced E-field and its solvers
  are the most independently verified code in this repository (gates ``N3``,
  ``N4``, ``N6``, ``N8`` all pass);
* ``SCWBD.rollout`` has accepted an additive latent drive ``u`` since it was
  written, and ``scwbd.foundation.train`` uses it for boundary-randomisation
  noise;

but nothing converted a field into a ``u``.  ``scwbd.intervene.base``'s
``InterventionOperator`` was designed for it and is unreachable: its
``DriftFn`` is ``(Tensor, float) -> Tensor`` over a flat state vector, with no
region axis, so it cannot accept either real rollout loop.  Consequently a
computed field has never produced a predicted response.

What this module does
---------------------
``body.tex`` §2.4 writes the controlled SDE as

.. math::

    \\mathrm dX = \\mathcal F(X,t)\\,\\mathrm dt
        + \\mathcal G_k(X,t)\\,u_k(t)\\,\\mathrm dt + Q^{1/2}\\mathrm dW_t,

so an intervention is a **term in the latent dynamics**, not an annotation
beside them.  Here :math:`\\mathcal G_k` is the field-to-parcel coupling built
by :func:`parcel_drive` and :math:`u_k(t)` is the pulse time course.

Three steps, and the first is where the physics lives:

1. **Project, do not take a magnitude.**  Induced current couples to the
   component of :math:`\\mathbf E` **along the cortical normal**.  A parcel's
   field magnitude is the wrong quantity: it is sign-blind, so it cannot tell
   a field driving into the cortex from one driving out of it, and those have
   opposite physiological effect.  On the model's own 400 cortical parcels
   orientation carries 2.6x what a scalar does, and more per degree of freedom
   than any subdivision measured; a magnitude discards all of it.

2. **Weight by coherence.**  A parcel spanning two banks of a sulcus has
   normals pointing opposite ways, and a uniform field drives them in
   opposite directions, so the parcel's *net* effect largely cancels.  Agent
   Cajal's ``normal_coherence`` is exactly that cancellation factor,
   :math:`|\\sum_f a_f \\hat n_f| / \\sum_f a_f`.  Using ``normal`` without it
   would treat a coherent gyral crown and a cancelling sulcal parcel as
   equally drivable, and a unit vector always looks equally informative --
   which is the trap the geometry module's own docstring warns about.

3. **Inject and roll forward**, so a pulse produces a *trajectory* that the
   observation heads can read, rather than a number sitting beside the model.

What this module is not
-----------------------
A **forward model**.  It computes what the physics implies and predicts what
the model says follows.  There is no optimiser over coil positions, no ranking
of candidate protocols, no recommendation, and no dose.  Those are absent by
construction, not gated: there is nothing here to switch off.

**The prediction is untrustworthy and says so.**  No checkpoint in this
repository has been trained on perturbational data, so the mapping from drive
to response is whatever the untrained or passively-trained dynamics happen to
produce.  :class:`ImpulseResponse.provenance` records that on every result.
It ships anyway, because a path that exists and is honestly labelled is how we
find out whether the model has anything to say, and one that waits for
validation never runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

__all__ = [
    "UNTRAINED_PREDICTION_NOTICE",
    "ParcelDrive",
    "ImpulseResponse",
    "parcel_drive",
    "pulse_time_course",
    "build_latent_drive",
    "predict_impulse_response",
]

_DT = torch.float32

#: Attached to every prediction.  The number this path produces is a statement
#: about the model, not about a person, and the model has seen no perturbation
#: data.  A reader who sees only the result must still be told that.
UNTRAINED_PREDICTION_NOTICE = (
    "PREDICTED RESPONSE FROM AN UNVALIDATED MODEL. The field is computed by a "
    "gated solver (N3/N4/N6/N8), but the mapping from that field to a neural "
    "response has never been fitted to perturbational data -- no checkpoint in "
    "this repository has seen a TMS-evoked response. The trajectory below is "
    "what this model implies, not what a brain would do. It is not a dose, not "
    "a protocol, and not a recommendation for any person."
)


# ---------------------------------------------------------------------------
# 1. field -> per-parcel drive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParcelDrive:
    """Signed per-parcel drive :math:`\\mathcal G_k`, and how it was obtained.

    ``values`` is **signed**: positive means the field drives *into* the
    cortex (anti-parallel to the outward normal), matching the convention in
    ``scwbd.runtime.backends.NormalComponentResponse``, which computes
    ``-(E . n)`` for the same reason.  Keeping the sign is the point -- a
    magnitude cannot distinguish the two directions and they are not
    physiologically equivalent.
    """

    #: ``[N]`` signed drive per parcel, V/m along the inward normal, scaled by
    #: coherence.  Exactly zero on parcels with no cortical normal.
    values: Tensor
    #: ``[N]`` the raw projection before coherence weighting, for diagnosis.
    projection_v_per_m: Tensor
    #: ``[N]`` the coherence factor applied, 0 where uncovered.
    coherence: Tensor
    #: ``[N]`` True where the parcel has a defined cortical normal.
    covered: Tensor
    units: str = "V/m (inward normal component, coherence-weighted)"
    notice: str = UNTRAINED_PREDICTION_NOTICE

    @property
    def n_parcels(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())

    def peak_parcel(self) -> int:
        """Index of the most strongly driven parcel, by absolute drive."""
        return int(torch.argmax(self.values.abs()))

    def summary(self) -> dict[str, Any]:
        v = self.values
        return {
            "n_parcels": self.n_parcels,
            "n_covered": self.n_covered,
            "peak_parcel": self.peak_parcel(),
            "peak_abs_drive": float(v.abs().max()),
            "mean_abs_drive": float(v.abs().mean()),
            "signed_range": [float(v.min()), float(v.max())],
            "units": self.units,
        }


def parcel_drive(
    efield: Tensor,
    normal: Tensor,
    *,
    coherence: Tensor | None = None,
    covered: Tensor | None = None,
) -> ParcelDrive:
    """Project a per-parcel E-field onto the cortical normal.

    ``efield`` is ``[N,3]`` in V/m, in the same frame as ``normal``.  ``normal``
    is ``[N,3]`` outward-positive (agent Cajal's convention: outward from the
    hemisphere centroid), ``nan`` on parcels with no cortical surface --
    subcortex and cerebellum.  ``coherence`` is ``[N]`` in ``[0,1]``.

    Uncovered parcels get **exactly zero**, never ``nan``.  That matters more
    than it looks: a ``nan`` drive propagates through the rollout and
    ``torch.nan_to_num`` in ``SCWBD.step`` would silently convert it to zero
    several operations later, so the model would appear to have been driven
    and would not have been.  Zeroing here makes the absence explicit and
    keeps ``covered`` alongside so a caller can see which parcels were skipped.
    """
    e = torch.as_tensor(efield, dtype=_DT)
    n = torch.as_tensor(normal, dtype=_DT)
    if e.shape != n.shape or e.ndim != 2 or e.shape[-1] != 3:
        raise ValueError(
            f"efield and normal must both be [N,3]; got {tuple(e.shape)} and "
            f"{tuple(n.shape)}"
        )

    if covered is None:
        covered_t = torch.isfinite(n).all(dim=-1)
    else:
        covered_t = torch.as_tensor(covered, dtype=torch.bool)
    covered_t = covered_t & torch.isfinite(e).all(dim=-1)

    n_safe = torch.where(covered_t.unsqueeze(-1), n, torch.zeros_like(n))
    e_safe = torch.where(covered_t.unsqueeze(-1), e, torch.zeros_like(e))

    # Inward-positive, matching runtime.backends.NormalComponentResponse.
    projection = -(e_safe * n_safe).sum(dim=-1)

    if coherence is None:
        coh = covered_t.to(_DT)
    else:
        coh = torch.as_tensor(coherence, dtype=_DT)
        coh = torch.where(covered_t & torch.isfinite(coh), coh, torch.zeros_like(coh))

    values = projection * coh
    return ParcelDrive(
        values=values,
        projection_v_per_m=projection,
        coherence=coh,
        covered=covered_t,
    )


# ---------------------------------------------------------------------------
# 2. the pulse time course
# ---------------------------------------------------------------------------


def pulse_time_course(
    n_steps: int,
    *,
    dt_s: float = 1e-3,
    onset_step: int = 1,
    duration_s: float = 3e-4,
) -> Tensor:
    """``[T]`` dimensionless envelope of the pulse, area-normalised to 1.

    A biphasic TMS pulse lasts ~100-300 us while the model's fast clock is
    ~1 ms, so the pulse is sub-timestep: it is an impulse at this resolution,
    and pretending otherwise would be false precision.  The envelope is
    normalised to unit area so the *delivered impulse* is invariant to
    ``dt_s`` -- halving the step doubles the height rather than halving the
    effect, which is what makes a result comparable across integration
    settings.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if not 0 <= onset_step < n_steps:
        raise ValueError(f"onset_step {onset_step} outside [0,{n_steps})")
    env = torch.zeros(n_steps, dtype=_DT)
    n_on = max(1, int(round(duration_s / max(dt_s, 1e-12))))
    n_on = min(n_on, n_steps - onset_step)
    env[onset_step : onset_step + n_on] = 1.0 / n_on
    return env


# ---------------------------------------------------------------------------
# 3. drive -> latent u
# ---------------------------------------------------------------------------


def build_latent_drive(
    model: Any,
    drive: ParcelDrive | Tensor,
    *,
    n_steps: int,
    batch: int = 1,
    component: str = "rate_e",
    gain: float = 1.0,
    time_course: Tensor | None = None,
    dt_s: float = 1e-3,
    device: "torch.device | str | None" = None,
) -> Tensor:
    """``(B,T,N,D)`` additive latent drive for ``SCWBD.rollout(u=...)``.

    Writes only into ``component`` (default ``rate_e``, the excitatory rate in
    the shared interface prefix every family declares at identical offsets).
    That is the channel a TMS pulse belongs in: in every mechanistic backend
    the ``u`` term enters the excitatory input -- Jansen-Rit's ``dy4``,
    Wilson-Cowan's ``x_e`` -- so driving ``rate_e`` is the family-agnostic
    equivalent.

    **The pad is never written.**  ``SCWBD.rollout`` calls
    ``family_layout.assert_clean`` on the whole trajectory, and ``u`` is added
    to ``dx`` raw, with no mask, so a full-width drive would trip
    ``SpanViolation`` -- correctly, because the pad channels are not state.
    The result is masked by ``in_span_mask()`` before it is returned, and
    :func:`predict_impulse_response` asserts the guard still holds afterwards.
    """
    vals = drive.values if isinstance(drive, ParcelDrive) else torch.as_tensor(drive, dtype=_DT)
    vals = vals.reshape(-1).to(_DT)
    if not torch.isfinite(vals).all():
        raise ValueError("parcel drive contains non-finite values; refusing to inject")
    # Build on the drive's own device by default. A CPU-only assembly silently
    # worked for every caller that passed a constant `drive`, and fails the
    # moment the drive is a LEARNED tensor living with the model on CUDA -- the
    # scatter mixes devices. Defaulting to `vals.device` keeps the autograd path
    # intact instead of forcing a copy that would strand the gradient.
    dev = torch.device(device) if device is not None else vals.device

    flayout = getattr(model, "family_layout", None)
    layout = getattr(model, "layout", None)
    if layout is None:
        raise AttributeError("model exposes no state layout")
    dim = int(layout.dim)
    n_regions = int(vals.shape[0])

    env = pulse_time_course(n_steps, dt_s=dt_s) if time_course is None else torch.as_tensor(
        time_course, dtype=_DT
    ).reshape(-1)
    if env.shape[0] != n_steps:
        raise ValueError(f"time_course has {env.shape[0]} steps, expected {n_steps}")
    env = env.to(dev)

    u = torch.zeros(batch, n_steps, n_regions, dim, dtype=_DT, device=dev)

    if flayout is not None:
        # Resolve the component's offset per family. The shared prefix puts it
        # at the same place for every family, but resolve rather than assume:
        # the assumption is exactly what a heterogeneous layout will break.
        for fam in flayout.partition.families:
            name = fam.name
            try:
                sl = flayout.component_slice(name, component)
            except Exception:  # family does not declare it -- skip, do not guess
                continue
            idx = flayout.index(name, device=dev).to(torch.long)
            u[:, :, idx, sl] = (
                gain * vals[idx].reshape(1, 1, -1, 1) * env.reshape(1, -1, 1, 1)
            )
        u = u * flayout.in_span_mask(dtype=_DT).to(dev).reshape(1, 1, n_regions, dim)
    else:
        sl = layout.slice(component)
        u[:, :, :, sl] = gain * vals.reshape(1, 1, -1, 1) * env.reshape(1, -1, 1, 1)

    return u


# ---------------------------------------------------------------------------
# 4. the prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpulseResponse:
    """A predicted trajectory and its EEG readout. Never a protocol."""

    #: ``(B,T,N,D)`` perturbed latent trajectory.
    state: Tensor
    #: ``(B,T,C)`` predicted sensor mean -- where a TEP would appear.
    eeg: Tensor
    #: ``(B,T,C)`` predicted sensor log-variance.
    eeg_logvar: Tensor
    #: The same rollout with ``u = 0``, so the *evoked* part is a difference
    #: rather than a level. Without it a caller cannot tell a response from
    #: the model's own ongoing activity.
    baseline_eeg: Tensor
    drive: ParcelDrive
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notice: str = UNTRAINED_PREDICTION_NOTICE

    @property
    def evoked(self) -> Tensor:
        """``(B,T,C)`` perturbed minus baseline: the predicted evoked response."""
        return self.eeg - self.baseline_eeg

    def peak_evoked_amplitude(self) -> float:
        return float(self.evoked.abs().max())

    def summary(self) -> dict[str, Any]:
        return {
            "n_steps": int(self.eeg.shape[1]),
            "n_channels": int(self.eeg.shape[-1]),
            "peak_evoked_amplitude": self.peak_evoked_amplitude(),
            "drive": self.drive.summary(),
            **dict(self.provenance),
            "notice": self.notice,
        }


def predict_impulse_response(
    model: Any,
    drive: ParcelDrive,
    *,
    y_context: Tensor,
    theta: Tensor,
    n_steps: int = 64,
    gain: float = 1.0,
    dt_s: float = 1e-3,
    component: str = "rate_e",
    time_course: Tensor | None = None,
    context_mask: Tensor | None = None,
    baseline_eeg: Tensor | None = None,
) -> ImpulseResponse:
    """Roll the model forward with and without the drive, and read the EEG head.

    Both rollouts use the same ``y_context``, ``theta`` and initial state, so
    the difference between them is the drive and nothing else.  That is what
    makes :attr:`ImpulseResponse.evoked` interpretable at all.

    ``baseline_eeg`` lets a caller supply an already-computed unperturbed
    readout.  This is an **exact** reuse, not an approximation: the baseline is
    the ``u=None`` rollout, so it is a function of ``model``, ``y_context``,
    ``theta`` and ``context_mask`` alone and does not depend on the drive.  A
    permutation null that varies only the drive would otherwise recompute the
    identical trajectory once per draw, which for the staged pose-contrast
    analysis is half the total cost.  Passing a baseline from a *different*
    model or context would silently corrupt every evoked response, so it is
    opt-in and the caller owns that correspondence.
    """
    batch = int(torch.as_tensor(y_context).shape[0])
    u = build_latent_drive(
        model,
        drive,
        n_steps=n_steps,
        batch=batch,
        component=component,
        gain=gain,
        time_course=time_course,
        dt_s=dt_s,
    )

    # Bind theta-conditioned ParamPacks before rolling out.  Sixth site to need
    # this: the trainer (3), predict(), and here.  Every one was a path declared
    # for both arms and exercised only on the control, which has no mechanistic
    # families.  Without it a family-state checkpoint raises SpanViolation.
    _bind = getattr(model, "set_mechanistic_theta", None)
    if _bind is not None and getattr(model, "family_layout", None) is not None:
        # `is None`, not `or`. The previous line was
        #     _anat = getattr(model, "anat", None) or getattr(model, "_anat", None)
        # which treats any FALSY anatomy as absent -- and an object defining
        # __len__ or __bool__ is falsy without being missing. The distinction
        # matters here because the consequence of "absent" is loading a
        # different brain.
        _anat = getattr(model, "anat", None)
        if _anat is None:
            _anat = getattr(model, "_anat", None)
        if _anat is None:
            # Refuse rather than substitute. Loading "the" anatomy binds whatever
            # prior happens to be on disk to a model built from some other one;
            # it raised out-of-bounds here only because the region counts
            # differed, and two anatomies of equal size with different family
            # membership would have bound silently and returned numbers.
            raise ValueError(
                "this model does not carry the anatomy it was built with, so the "
                "mechanistic theta cannot be bound. Loading the default prior "
                "here would silently attach a different brain: family membership "
                "is indexed by parcel, and nothing downstream checks that the "
                "indices belong to the same parcellation. Build the model via "
                "SCWBD(cfg, anat) (which records it) or pass the anatomy in."
            )
        _n = int(getattr(_anat, "n_regions", 0) or 0)
        _layout = getattr(model, "family_layout", None)
        _max = max((max(f.regions) for f in getattr(_layout, "families", ()) or ()), default=-1)
        if _n and _max >= _n:
            raise ValueError(
                f"the model's family layout indexes region {_max} but this anatomy has "
                f"{_n} regions. These are different parcellations, and binding one to "
                "the other would mis-assign every family."
            )
        _bind(theta, _anat)

    with torch.no_grad():
        perturbed = model.rollout(
            y_context=y_context, theta=theta, n_steps=n_steps,
            context_mask=context_mask, u=u,
        )
        mu, lv = model.eeg(perturbed.state)
        if baseline_eeg is None:
            baseline = model.rollout(
                y_context=y_context, theta=theta, n_steps=n_steps,
                context_mask=context_mask, u=None,
            )
            base_mu, _ = model.eeg(baseline.state)
        else:
            base_mu = torch.as_tensor(baseline_eeg, dtype=mu.dtype)
            if base_mu.shape != mu.shape:
                raise ValueError(
                    f"supplied baseline_eeg has shape {tuple(base_mu.shape)}, "
                    f"expected {tuple(mu.shape)}; a mismatched baseline would "
                    "corrupt every evoked response"
                )

    flayout = getattr(model, "family_layout", None)
    if flayout is not None:
        # The guard that justifies writing into a padded layout at all. If the
        # drive had reached a pad channel this raises rather than returning a
        # trajectory that quietly includes it.
        flayout.assert_clean(perturbed.state, where="impulse response rollout")

    return ImpulseResponse(
        state=perturbed.state,
        eeg=mu,
        eeg_logvar=lv,
        baseline_eeg=base_mu,
        drive=drive,
        provenance={
            "path": "efield -> normal projection -> coherence -> rate_e drive -> rollout -> EEG head",
            "component_driven": component,
            "gain": float(gain),
            "dt_s": float(dt_s),
            "n_steps": int(n_steps),
            "field_solver_gates": ["N3", "N4", "N6", "N8"],
            "response_mapping_validated": False,
            "trained_on_perturbation_data": False,
            "claim": (
                "a prediction about this model's dynamics under a computed "
                "field; not a measurement, not a dose, not a protocol"
            ),
        },
    )
