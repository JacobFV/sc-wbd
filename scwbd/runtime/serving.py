"""The stable serving surface: load, handshake, warm up, evaluate in batches.

This is the only module a downstream consumer needs to import to *get* a
service.  Everything it exposes is read-only:

* :meth:`ServedModel.load` -- discover ``checkpoints/<designation>/``, read the
  ``ClaimManifest`` beside it, and build a :class:`ModelProvenance` that states
  exactly what was found, **including when nothing was found**;
* :meth:`ServedModel.handshake` -- a consumer asserts what it expects and gets
  a hard :class:`~scwbd.runtime.provenance.ProvenanceMismatch` if it is wrong;
* :meth:`ServedModel.warm_up` -- deterministic, seeded, and it returns the
  evaluation it ran so a caller can see the fixture rather than trust it;
* :meth:`ServedModel.evaluate_batch` -- many poses, each with its own decision.

The load path deliberately does **not** fail when there is no trained
checkpoint.  It loads an analytic backend and records
``weights_status="analytic_backend"``.  Failing would push consumers toward
building their own uncontrolled fallback; recording it makes the fallback
visible and lets a consumer that requires a trained artifact refuse in one
line.

Claim limits: a handshake asserts identity, never validity.  A batch of
evaluations is a batch of simulations.  Nothing here authorises anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ._compat import ClaimManifest, Pose
from .backends import CoilSpec
from .head import HeadModel, spherical_phantom
from .provenance import (
    MODEL_DESIGNATION,
    RUNTIME_API_VERSION,
    SCHEMA_VERSION,
    ModelProvenance,
    ProvenanceExpectation,
    ProvenanceMismatch,
    WeightsStatus,
)
from .targeting import TargetingConfig, TargetingService
from .types import PoseEvaluation, PoseRequest

__all__ = [
    "CheckpointNotFound",
    "CheckpointRecord",
    "discover_checkpoint",
    "ServedModel",
    "ProvenanceExpectation",
    "ProvenanceMismatch",
    "DEFAULT_CHECKPOINT_ROOT",
    "MODEL_DESIGNATION",
    "SCHEMA_VERSION",
    "RUNTIME_API_VERSION",
]

#: ``ARCHITECTURE.md`` Sec. 5: "Checkpoints: ``checkpoints/scwbd-001-beta/``
#: with a ``ClaimManifest`` alongside."
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")

_WEIGHT_NAMES = ("weights.pt", "model.pt", "model.safetensors", "state_dict.pt")
_MANIFEST_NAMES = ("claim_manifest.json", "claim.json", "manifest.json")


class CheckpointNotFound(FileNotFoundError):
    """Raised only when a caller *demanded* a trained checkpoint."""


@dataclass(frozen=True)
class CheckpointRecord:
    """What was actually found on disk, including "nothing"."""

    designation: str
    root: Path | None
    weights_path: Path | None
    weights_sha256: str | None
    manifest_path: Path | None
    manifest: Any = None
    weights_status: WeightsStatus = "analytic_backend"

    @property
    def found(self) -> bool:
        return self.weights_path is not None

    def claim_fields(self) -> dict[str, Any]:
        """Claim identity from the manifest, or the honest default."""
        m = self.manifest
        if m is None:
            return {
                "claim_manifest_id": "absent",
                "claim_class": "surrogate",
                "posterior_class": "pseudo",
            }
        if ClaimManifest is not None and isinstance(m, ClaimManifest):
            return {
                "claim_manifest_id": m.id,
                "claim_class": m.effective_claim_class(),
                "posterior_class": m.posterior_class,
            }
        if isinstance(m, Mapping):
            return {
                "claim_manifest_id": str(m.get("id", "unnamed")),
                "claim_class": str(m.get("claim_class", "surrogate")),
                "posterior_class": str(m.get("posterior_class", "pseudo")),
            }
        return {  # pragma: no cover - defensive
            "claim_manifest_id": "unrecognised",
            "claim_class": "surrogate",
            "posterior_class": "pseudo",
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_checkpoint(
    designation: str = "scwbd-001-beta",
    *,
    root: Path | str = DEFAULT_CHECKPOINT_ROOT,
    require: bool = False,
) -> CheckpointRecord:
    """Look for ``<root>/<designation>/`` and report what is there.

    ``require=True`` turns an absent checkpoint into
    :class:`CheckpointNotFound`.  The default is ``False`` and yields a record
    with ``weights_status="analytic_backend"``, because a consumer that needs a
    trained artifact should say so once, in the handshake, rather than the
    library guessing on its behalf.
    """
    base = Path(root) / designation
    if not base.is_dir():
        if require:
            raise CheckpointNotFound(
                f"no checkpoint directory {base}; SC-WBD-001-beta has not been "
                "trained into this working tree. Load without require=True to "
                "get the analytic backend, which is labelled as such in "
                "ModelProvenance.weights_status."
            )
        return CheckpointRecord(
            designation=designation,
            root=None,
            weights_path=None,
            weights_sha256=None,
            manifest_path=None,
            weights_status="analytic_backend",
        )

    weights = next((base / n for n in _WEIGHT_NAMES if (base / n).is_file()), None)
    manifest_path = next(
        (base / n for n in _MANIFEST_NAMES if (base / n).is_file()), None
    )
    manifest: Any = None
    if manifest_path is not None:
        raw = json.loads(manifest_path.read_text())
        if ClaimManifest is not None:
            try:
                manifest = ClaimManifest(**raw)
            except Exception:  # pragma: no cover - a hand-edited manifest
                manifest = raw
        else:  # pragma: no cover - agent A has landed this
            manifest = raw

    if weights is None:
        if require:
            raise CheckpointNotFound(
                f"checkpoint directory {base} exists but contains none of "
                f"{list(_WEIGHT_NAMES)}"
            )
        status: WeightsStatus = "analytic_backend"
        sha = None
    else:
        status = "trained"
        sha = _sha256(weights)

    return CheckpointRecord(
        designation=designation,
        root=base,
        weights_path=weights,
        weights_sha256=sha,
        manifest_path=manifest_path,
        manifest=manifest,
        weights_status=status,
    )


@dataclass(frozen=True)
class ServedModel:
    """A loaded targeting service plus the provenance of what was loaded."""

    targeting: TargetingService
    provenance: ModelProvenance
    checkpoint: CheckpointRecord

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(
        cls,
        model: str = "scwbd-001-beta",
        *,
        device: str = "cuda",
        checkpoint_root: Path | str = DEFAULT_CHECKPOINT_ROOT,
        require_checkpoint: bool = False,
        head_default: HeadModel | None = None,
        coil: CoilSpec | None = None,
        config: TargetingConfig | None = None,
        **service_kwargs: Any,
    ) -> "ServedModel":
        """Build the serving path and record exactly what backs it.

        ``ModelProvenance.device`` records where the field solve **actually
        runs**, which is not always what was asked for: CUDA may be
        unavailable, and the analytic backend computes in CPU float64 by
        design (``ARCHITECTURE.md`` Sec. 3 keeps solvers and covariance
        propagation out of low precision).  The requested device is kept in
        ``notes["requested_device"]`` so the difference is visible rather than
        asserted away.
        """
        record = discover_checkpoint(model, root=checkpoint_root, require=require_checkpoint)
        resolved_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"

        service = TargetingService(
            head_default=head_default,
            coil=coil,
            config=config,
            device=resolved_device,
            **service_kwargs,
        )
        compute_device = str(
            getattr(service.efield_backend, "device", resolved_device)
        )
        prov = replace(
            service.provenance,
            model_designation=MODEL_DESIGNATION,
            checkpoint_id=model if record.found else "",
            checkpoint_sha256=record.weights_sha256,
            weights_status=record.weights_status,
            device=compute_device,
            notes={
                **dict(service.provenance.notes),
                "requested_device": device,
                "resolved_device": resolved_device,
                "compute_device": compute_device,
                "checkpoint_root": str(checkpoint_root),
                "checkpoint_found": record.found,
                "weights_path": str(record.weights_path) if record.weights_path else "",
                "manifest_path": (
                    str(record.manifest_path) if record.manifest_path else ""
                ),
                "untrained_warning": (
                    ""
                    if record.weights_status == "trained"
                    else (
                        "no trained SC-WBD-001-beta checkpoint was loaded; the "
                        "predictions come from a closed-form field model and "
                        "surrogate propagators and must not be reported as "
                        "outputs of a trained whole-brain model"
                    )
                ),
            },
            **record.claim_fields(),
        )
        service.provenance = prov
        return cls(targeting=service, provenance=prov, checkpoint=record)

    # -- handshake ---------------------------------------------------------
    def handshake(self, expectation: ProvenanceExpectation) -> ModelProvenance:
        """Assert what is being served, or raise :class:`ProvenanceMismatch`.

        A consumer calls this **once, before it uses any prediction**.  The
        returned provenance is the object it should record in its own evidence
        graph; its ``content_hash()`` binds the whole assertion.
        """
        return expectation.check(self.provenance)

    def describe(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.canonical(),
            "provenance_hash": self.provenance.content_hash(),
            "checkpoint_found": self.checkpoint.found,
            "is_trained_artifact": self.provenance.is_trained_artifact,
        }

    # -- warm-up and batching ----------------------------------------------
    def warm_up(
        self,
        head_model: HeadModel | None = None,
        coil_pose: PoseRequest | Pose | Mapping[str, Any] | None = None,
    ) -> PoseEvaluation:
        """Run one deterministic evaluation and return it.

        Warm-up returns its result rather than swallowing it so that a caller
        can inspect the fixture it just ran -- including its decision, which
        for the shipped phantom is frequently a ``Defer``.  That is the
        expected outcome, not a failure.
        """
        head = head_model or self.targeting.head_default or spherical_phantom()
        if coil_pose is None:
            coil_pose = _default_warmup_pose(head)
        return self.targeting.evaluate_pose(head, coil_pose)

    def evaluate_batch(
        self,
        head_model: HeadModel,
        coil_poses: Iterable[PoseRequest | Pose | Mapping[str, Any]],
        *,
        stop_on_refusal: bool = False,
    ) -> tuple[PoseEvaluation, ...]:
        """Evaluate many candidate poses against one head model.

        Batching is a throughput convenience only: every pose gets its own
        field solve, its own ledger and its own decision, and no pose's
        decision is influenced by another's.  Cross-candidate comparison is a
        different operation with different rules and lives in
        :mod:`scwbd.runtime.compare`.
        """
        out: list[PoseEvaluation] = []
        for i, p in enumerate(coil_poses):
            req = PoseRequest.coerce(p)
            ev = self.targeting.evaluate_pose(
                head_model, req, label=req.label if req.label != "candidate" else f"pose_{i:02d}"
            )
            out.append(ev)
            if stop_on_refusal and ev.refused:
                break
        return tuple(out)


def _default_warmup_pose(head: HeadModel) -> PoseRequest:
    """A pose over the declared target region, in the head model's own frame."""
    from ._compat import PoseUncertainty

    region = next(iter(head.target_regions), None)
    if region is None:  # pragma: no cover - phantom always declares one
        direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    else:
        mask = head.region_mask(region)
        direction = (head.cortex_vertices[mask] - head.centre).mean(dim=0)
        direction = direction / torch.linalg.norm(direction)

    z = -direction  # coil axis points into the head
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    if float(torch.abs(z @ up)) > 0.95:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    x = torch.linalg.cross(up, z)
    x = x / torch.linalg.norm(x)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=1)
    t = head.centre + direction * (head.scalp_radius + 0.004)
    pose = Pose.from_Rt(
        R, t, head.frame, "coil", provenance={"method": "warm_up_fixture"}
    )
    return PoseRequest(
        pose=pose,
        frame=head.frame,
        label="warm_up",
        uncertainty=PoseUncertainty.isotropic(0.002, 0.02),
        uncertainty_scope="within_session",
        notes={"purpose": "deterministic warm-up fixture, not a candidate"},
    )

