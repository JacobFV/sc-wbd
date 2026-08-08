"""The staged curriculum (body.tex §6.1-§6.5) for SC-WBD-001-beta.

Stage I    regional phenotype pretraining behind declared ports, with boundary
           randomization, delayed/corrupted inputs and missing channels
Stage II   interface and pathway calibration -- the long-range operators only,
           penalised for transmitting outside their declared type
Stage III  heterogeneous sliced-trajectory training (equation 4): an episode
           declares an observed subgraph ``S_e``, and unobserved interior state
           is marginalised, never imputed
Stage IV   connectome assembly and joint multimodal training on the full mixture
Stage V    individualization with centered population effects and hierarchical
           shrinkage (refusal R07)

Every stage is resumable, seeded, and wall-clock bounded.  Every source enters
through a :class:`~scwbd.foundation.mixture.SourceSpec` with a gradient mask;
the stage curriculum may only **restrict** what a source updates, never expand
it, so a stage cannot quietly grant a permission the source card withheld.

TRIBE v2 distillation is **off by default** and is never a subject likelihood.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .anatomy import AnatomyPrior, load_anatomy
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import FoundationConfig, StageConfig, load_config
from .heads import gaussian_nll
from .individual import Individualizer
from .manifest import Claim, ClaimManifest
from .mixture import MixtureTrainer, SourceSpec
from .model import SCWBD
from .posterior import AmortizedPosterior
from .curriculum_admission import StageAdmission, assert_region_count, stage_admission
from .simulate import THETA_NAMES, CorpusSpec, SimCorpus, ThetaPrior, generate_corpus
from .util import (
    JsonlLogger,
    Timer,
    cap_cuda_reserve,
    count_parameters,
    cuda_reserved_gb,
    env_fingerprint,
    git_sha,
    set_determinism,
)

__all__ = ["FoundationTrainer", "BindingDriftError", "STAGE_PERMISSIONS", "main"]


class BindingDriftError(RuntimeError):
    """The compiler compiled, but its groups no longer name tensors we have.

    Distinct from :class:`~scwbd.foundation.compiler_bridge.CompilerUnavailable`
    on purpose: an absent compiler is a degraded-but-honest mode the source cards
    can cover, whereas a *present* compiler whose bindings miss means the
    permission system is reporting enforcement it is not performing.  Only the
    first is recoverable, so only the first falls back.
    """

#: What each stage is allowed to touch.  Intersected with (never added to) the
#: source card's own ``A_k``.
STAGE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "I_regional": ("local.*", "log_dt_scale", "readout.*", "assimilate.*", "context.*", "residual.*"),
    "II_interface": ("coupling.*", "msg_proj.*", "msg_readin.*", "readout.*"),
    "III_sliced": ("*",),
    "IV_assembly": ("*",),
    "V_individual": ("individualizer.*", "eeg.log_gain", "eeg.offset", "eeg.log_noise", "eeg.nuisance*"),
}

#: Stages whose curriculum admits the measured (tier-1) sources. These are RUN-1
#: stage names. Run 2's "integrity-ordered" curriculum renamed every stage to
#: `T1_measured_founding … T5_distillation`, and nothing here was updated, so on
#: a run-2 config this set intersects the schedule in NOTHING and the measured
#: likelihood never reaches a loss. See `smoke()`, which refuses to launch on it.
REAL_DATA_STAGES: frozenset[str] = frozenset({"III_sliced", "IV_assembly", "V_individual"})

#: The loss key `real_losses` emits, and the source card it is scored against.
REAL_LOSS_KEY = "eegmmidb_real"


def _cycle(loader: Iterable) -> Iterator:
    while True:
        for b in loader:
            yield b


class SensorToParcel(nn.Module):
    """Minimum-norm projection of sensor data into parcel space, with its uncertainty.

    Real EEG gives 64 scalp channels; the model's assimilator consumes regional
    activity.  Rather than pretend the sensors *are* regions, we apply the
    explicit, calibrated pseudo-inverse of the lead field and carry the
    resulting resolution loss as a mask/uncertainty: the projection is an
    assimilation *input*, and the **likelihood is always evaluated back in
    sensor space**, where the measurement actually lives.  Nothing is upsampled
    into fictitious precision (§4.2).
    """

    def __init__(self, lead: Tensor, *, reg: float = 1e-2) -> None:
        super().__init__()
        L = lead.float()
        C = L.shape[0]
        scale = L.pow(2).mean().sqrt().clamp_min(1e-12)
        Ln = L / scale
        G = Ln @ Ln.transpose(0, 1)
        A = G + reg * torch.eye(C, device=L.device) * float(torch.diagonal(G).mean())
        Winv = Ln.transpose(0, 1) @ torch.linalg.inv(A)  # (N, C)
        self.register_buffer("Winv", Winv)
        res = Ln @ Winv  # (C, C) resolution matrix
        self.register_buffer("resolution_trace", torch.diagonal(res).mean())
        self.effective_rank = float(torch.linalg.matrix_rank(Ln.double()).item())

    def forward(self, y: Tensor) -> Tensor:
        """``(B,T,C) -> (B,T,N)``"""
        return torch.einsum("nc,btc->btn", self.Winv.to(y.dtype), y)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "minimum_norm_pseudoinverse",
            "effective_rank": self.effective_rank,
            "mean_resolution_diagonal": float(self.resolution_trace),
            "note": (
                "Assimilation input only. The likelihood is evaluated in sensor space; "
                "this projection supports no source-localisation claim."
            ),
        }


class FoundationTrainer:
    """Owns the model, the mixture, the data and the staged curriculum."""

    #: The posterior flow trains at this fraction of the stage LR.  See run_stage.
    #:
    #: 0.1 was not enough.  With the conditioning normalised, nuisance_dim at 0
    #: and the coupling translation bounded, -log q still drifted 8 -> 30 -> 67
    #: by step 1020 while the forecast head kept improving (nll 1.49 -> 0.52).
    #: Each earlier fix slowed the drift without stopping it; a density chasing
    #: a conditioning distribution that changes every batch -- sliced-trajectory
    #: training re-draws the observed subgraph each step -- needs a smaller step
    #: than a residual stack, not merely a smaller one than before.
    POSTERIOR_LR_SCALE: float = 0.02

    #: Weight decay for the posterior group.
    #:
    #: The drift is the coupling nets' weights growing, which grows the
    #: translation, which grows |z|.  Decay opposes that directly, where a
    #: smaller step only delays it.  Applied to the flow alone: decaying the
    #: dynamics operators would pull them toward zero drift, which is a
    #: statement about the brain rather than about optimisation.
    POSTERIOR_WEIGHT_DECAY: float = 1e-2

    def __init__(
        self,
        cfg: FoundationConfig,
        *,
        anat: AnatomyPrior | None = None,
        device: str | None = None,
        resume: bool | None = None,
        quick: bool = False,
    ) -> None:
        self.cfg = cfg
        self.quick = quick
        self.device = torch.device(device or cfg.train.device)
        cap_cuda_reserve(self.device, cfg.train.cuda_reserve_gb)
        set_determinism(cfg.train.seed)
        torch.backends.cuda.matmul.allow_tf32 = False  # solvers stay fp32 (ARCH §3)
        if cfg.model.compile:
            # torch.compile's donated-buffer optimisation frees backward
            # intermediates it believes are dead, which requires
            # retain_graph=False on every backward.  The mixture takes one
            # backward **per source** (`mixture.step` -> `gate.grads`, with
            # `retain_graph=(more sources) or measure_conflict`) so that each
            # source's gradient is restricted to the parameters its card permits,
            # and so that per-source gradient conflict can be measured at all
            # (Appendix D: "gradient conflict is measured by module and source").
            #
            # Those are requirements, so the allocator optimisation gives way --
            # not the other way round.  Summing the losses into a single backward
            # would fix the crash and silently delete both the per-source
            # permission enforcement and the conflict measurement.
            #
            # It surfaced only at Stage III because that is the first stage with
            # two sources live at once (simulated + measured); Stages I and II
            # take a single backward, so the incompatibility could not appear.
            # NB: import the submodule explicitly. ``torch._functorch.config``
            # is not reachable by attribute access from a bare ``import torch``,
            # and I verified the knob's existence through a different import
            # form than the one the code used -- so the check passed and the
            # code raised AttributeError on the next launch.
            import torch._functorch.config as _functorch_config

            _functorch_config.donated_buffer = False
        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(cfg.train.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.anat = (
            anat
            or load_anatomy(
                device=self.device,
                n_cortex=400,
                force_fallback=cfg.train.anatomy_force_fallback,
            )
        ).to(self.device)
        if not self.anat.is_biological():
            # Loud, every run, in the log the operator actually reads -- the
            # provenance field said this correctly all along and nobody looked.
            print(
                "[anatomy] WARNING: prior is SYNTHETIC "
                f"(provenance={self.anat.provenance!r}, n_regions={self.anat.n_regions}). "
                "This model carries NO biological content: G2 is void, and no claim "
                "about connectome-masked coupling, receptor-derived E/I or anatomical "
                "regional heterogeneity is supportable from it.",
                flush=True,
            )
        # The config's n_regions is otherwise inert -- SCWBD takes its shape from
        # the anatomy -- so without this the field can name a parcellation the
        # model never loaded, which is how 001-beta's "454 = Schaefer-400 + 32 +
        # 22" survived a whole night unchallenged.
        assert_region_count(cfg.model.n_regions, self.anat)
        self.theta_prior = ThetaPrior()
        self.model = SCWBD(cfg.model, self.anat).to(self.device)
        # Step-0 only: the pack/backend cover check below is a claim about the
        # model that the model can falsify, so it is worth making once and
        # pointless to repeat every batch.
        self._theta_bind_audited = False
        self.posterior = AmortizedPosterior(
            cfg.posterior,
            len(THETA_NAMES),
            prior=self.theta_prior,
            fs=cfg.data.fs_hz,
            nuisance_dim=cfg.posterior.nuisance_dim,
        ).to(self.device)
        self.sensor_to_parcel = SensorToParcel(self.model.eeg.L).to(self.device)
        if cfg.model.compile and self.device.type == "cuda":
            # Fuse the per-parcel elementwise chain. On the GB10's unified LPDDR5X
            # the regional operator is memory-bandwidth bound, not FLOP bound, so
            # fusing the FiLM/GELU/residual chain is worth ~20% wall clock.
            # ONLY what this arm actually built. `local`/`residual` are the
            # CONTROL arm's module names; on the family arm both are None, and
            # `torch.compile(None)` does not raise -- it returns a *function*,
            # which silently overwrote the None sentinel and turned a clean
            # `AttributeError: 'NoneType'` into `'function' object has no
            # attribute 'log_scale'`. Net effect: the treatment arm was never
            # compiled at all, so the ~20% fusion below applies to the control
            # arm only. Left that way deliberately rather than compiling the
            # family operators on the eve of a launch -- see the run manifest.
            for _name in ("local", "residual"):
                _mod = getattr(self.model, _name, None)
                if _mod is not None:
                    setattr(self.model, _name, torch.compile(_mod, dynamic=False))

        self.sources = self._load_sources()
        self.compiler_report: dict[str, Any] = self._bind_compiler_masks()
        self.individualizer: Individualizer | None = None
        self.logger = JsonlLogger(self.report_dir / f"{cfg.train.run_name}_train.jsonl", echo=True, echo_every=1)
        self.timer = Timer()
        self.history: list[dict[str, Any]] = []
        self.global_step = 0
        self.completed_stages: list[str] = []
        self._resume = cfg.train.resume if resume is None else resume
        self._data_ready = False
        self._mixture_reports: list[dict[str, Any]] = []
        #: per-stage admission, as decided from the config.  Written to the
        #: report so the artifact records which sources each stage admitted
        #: rather than leaving it to be inferred from the trainer's source.
        self._admissions: dict[str, Any] = {}
        #: Source ids that actually produced a loss term at least once. Derived
        #: from what ran, not from what a card claimed -- run 2 shipped with
        #: cards asserting a source trained modules it could not reach.
        self._contributed: set[str] = set()
        #: Per stage, the sources it admitted that produced no term at all.
        self._absent_admitted: dict[str, list[str]] = {}
        #: Admitted ids that are never expected to appear as their own loss key,
        #: because they constrain OTHER terms rather than adding one:
        #: `montage_calibration` narrows the nuisance parameters a stage may
        #: train, and `negative_control_shuffled` grants nothing by design.
        #:
        #: `anatomical_prior` and `sim_wholebrain` are deliberately NOT here.
        #: They arrive through `anat_losses` and `sim_losses` rather than through
        #: the measured loop, but both emit a term under their own source id, so
        #: exempting them would suppress the exact check this exists for -- an
        #: admitted source that silently contributes nothing.
        self._never_a_term: set[str] = {
            "montage_calibration",
            "negative_control_shuffled",
        }
        #: Built with the perturbation corpus; `None` when none is on disk.
        self.tms_drive = None
        #: sha256 of every parameter AT INITIALISATION, taken before a single
        #: step. This is the measurement half of the run-2 repair: a card
        #: pattern that matches a name proves the mechanism, and a tensor whose
        #: hash has changed proves the outcome. Run 2 shipped with the first
        #: broken and nothing checking the second, so the loss fell and 88.8% of
        #: the model sat at its initialisation for the whole run.
        self.init_fingerprint: dict[str, str] = self._fingerprint_parameters()
        self.flop_per_step = 0.0

    def _fingerprint_parameters(self) -> dict[str, str]:
        """``name -> sha256`` over every trainable tensor's current bytes."""
        import hashlib

        out: dict[str, str] = {}
        for prefix, mod in (
            ("model", self.model),
            ("posterior", self.posterior),
            ("tms_drive", getattr(self, "tms_drive", None)),
        ):
            if mod is None:
                continue
            for name, p in mod.named_parameters():
                key = name if prefix == "model" else f"{prefix}.{name}"
                out[key] = hashlib.sha256(
                    p.detach().to(torch.float32).cpu().numpy().tobytes()
                ).hexdigest()
        return out

    def _fingerprint_late_module(self, prefix: str, mod) -> None:
        """Record a module's initialisation when it is built after ``__init__``.

        `tms_drive` is constructed by `build_data`, because it only exists when
        the perturbation corpus is on disk. Without this, its parameters are
        absent from `init_fingerprint`, and `moved_since_init` compares a hash
        against `None` -- which is unequal, so the drive reports as MOVED on a
        checkpoint that has never taken a step. Measured exactly that on the
        architecture-only checkpoint: `tms_drive moved 4/4`, everything else
        frozen.

        That is a false pass in the one guard run 3 exists for, on the one
        module carrying its novel claim.
        """
        import hashlib

        for name, p in mod.named_parameters():
            self.init_fingerprint[f"{prefix}.{name}"] = hashlib.sha256(
                p.detach().to(torch.float32).cpu().numpy().tobytes()
            ).hexdigest()

    def moved_since_init(self) -> dict[str, Any]:
        """Which parameters differ from their initialisation, by module.

        Bit-comparison via hash, deliberately, not a tolerance: the question is
        "did this tensor receive a gradient at all", and any answer that
        involves a threshold invites one to be tuned until the answer is yes.

        A parameter with no recorded initialisation is reported separately
        rather than counted as moved: "I never saw this start" and "this
        changed" are different facts, and collapsing them is how a module that
        never trained reads as one that did.
        """
        now = self._fingerprint_parameters()
        unfingerprinted = sorted(k for k in now if k not in self.init_fingerprint)
        moved = {
            k
            for k, v in now.items()
            if k in self.init_fingerprint and self.init_fingerprint[k] != v
        }
        frozen = sorted(set(now) - moved)
        by_module: dict[str, dict[str, int]] = {}
        for k in now:
            top = k.split(".")[0]
            e = by_module.setdefault(top, {"moved": 0, "frozen": 0})
            e["moved" if k in moved else "frozen"] += 1
        return {
            "n_parameters": len(now),
            "n_moved": len(moved),
            "n_frozen": len(frozen),
            "by_module": by_module,
            "frozen_tensors": frozen[:200],
            # Should always be empty. Non-empty means a module was built after
            # the fingerprint was taken and nobody registered it, so its
            # moved/frozen status here is not evidence either way.
            "unfingerprinted": unfingerprinted,
            "note": (
                "Bit-identical to initialisation means the tensor received no "
                "gradient. A module entirely in `frozen` while on the forward "
                "path is the run-2 defect. Anything listed in `unfingerprinted` "
                "has no recorded initialisation and its status here is not "
                "evidence -- it is counted as frozen so the number cannot "
                "flatter the run."
            ),
        }

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------
    def _load_sources(self) -> dict[str, SourceSpec]:
        specs = SourceSpec.load_dir(self.cfg.mixture_cards)
        if not specs:
            raise FileNotFoundError(
                f"no source cards in {self.cfg.mixture_cards!r}. Training without source cards is "
                "exactly what ARCHITECTURE.md §7 rule 2 forbids: every source enters through a "
                "SourceCard with a gradient mask."
            )
        return specs

    def _bind_compiler_masks(self) -> dict[str, Any]:
        """Let agent A's compiler decide reachability for each source's ``A_k``.

        The compiler owns the authoritative parameter-group vocabulary and the
        refusals (R01-R11).  Its masks are translated into globs over this
        model's torch parameter names and **intersected** with the card's own
        declaration: the compiler may restrict a permission, never grant one the
        card withheld.  If the compiler or the bridge is unavailable the card's
        own globs are used and the fact is recorded -- never silently.
        """
        rep: dict[str, Any] = {"used": False}
        try:
            from . import compiler_bridge as cb

            if not cb.compiler_available():
                rep["reason"] = "scwbd.compiler unavailable"
                return rep
            probe = {
                sid: SourceSpec(**{**s.as_dict(), "gradient_permission": s.compiler_permission})
                for sid, s in self.sources.items()
            }
            compiled = cb.compile_foundation(self.anat.to("cpu"), list(probe.values()))
            binds = cb.bind_masks(self.model, compiled)
            audit = cb.audit_binding(self.model, compiled)
            if audit["problems"]:
                # Fail closed.  "The compiler is unavailable" and "the compiler
                # is available and says our binding table no longer describes
                # this model" are opposite situations, and only the first is a
                # reason to fall back to the cards' own globs.  Training through
                # the second produces a checkpoint whose gradient masks are
                # decorative -- which is what happened on 2026-08-05 -- so it is
                # raised past the fallback handler below rather than warned
                # about in a log nobody reads until the postmortem.
                raise BindingDriftError(
                    "compiler->torch binding is incomplete; refusing to train a model whose "
                    "gradient masks would not govern the tensors they name:\n  "
                    + "\n  ".join(audit["problems"])
                    + "\n\nFix scwbd/foundation/compiler_bridge.py:FOUNDATION_BINDING / "
                    "FOUNDATION_FROZEN_BINDING so every declared group names tensors that "
                    "exist.  tests/foundation/test_compiler_binding.py covers this."
                )
            # Modules the compiled schema does not model at all (the Stage-V
            # individualizer lives outside the operator graph) keep the card's
            # own declaration; the compiler cannot restrict what it never saw,
            # and pretending otherwise would silently disable Stage V.
            covered = {g.split(".")[0] for gs in binds.values() for g in gs}
            changed: dict[str, Any] = {}
            outside: dict[str, list[str]] = {}
            for sid, spec in list(self.sources.items()):
                globs = tuple(binds.get(sid, ()))
                extra = tuple(g for g in spec.gradient_permission if g.split(".")[0] not in covered and g != "*")
                if extra:
                    outside[sid] = list(extra)
                if not globs:
                    changed[sid] = {"compiler_globs": [], "action": "kept card globs (compiler granted nothing)"}
                    continue
                keep = tuple(g for g in spec.gradient_permission if g == "*" or any(_glob_overlap(g, c) for c in globs))
                keep = globs if spec.gradient_permission == ("*",) else (keep or globs)
                keep = tuple(dict.fromkeys(keep + extra))
                self.sources[sid] = SourceSpec(**{**spec.as_dict(), "gradient_permission": keep})
                changed[sid] = {"compiler_globs": list(globs), "effective": list(keep)}
            rep = {
                "used": True,
                "n_groups": len(compiled.gradient_masks.group_names),
                "schedule": cb.schedule_plan(compiled),
                "binding_audit": {
                    k: v for k, v in audit.items() if k in ("unclaimed_parameters", "empty_bindings", "unbound_groups", "declared_empty_groups", "frozen_groups", "problems")
                },
                "per_source": changed,
                "outside_compiler_schema": outside,
            }
        except BindingDriftError:
            raise
        except Exception as exc:  # noqa: BLE001 - the compiler is authoritative but not yet mandatory
            rep = {"used": False, "reason": f"{type(exc).__name__}: {exc}"}
            print(f"[warn] compiler bridge unavailable ({rep['reason']}); using the source cards' own "
                  "torch-level gradient permissions", flush=True)
        return rep

    def stage_uses_real(self, stage: StageConfig) -> bool:
        """Does this stage's curriculum admit the measured (tier-1) sources?

        ONE definition, called by both `run_stage`'s admission and the smoke
        precondition, so the guard tests the gate the trainer actually runs
        rather than a second copy that can agree with a wrong answer (RL-9).

        That shape is ported deliberately from `wt/curie`, which is the branch
        that had this check and whose trainer was dropped in the merge. Its
        MECHANISM is not ported. Curie's read

            return stage.name in REAL_DATA_STAGES

        which decides admission from the stage's NAME -- and that is the defect
        it was written to catch, not a fix for it. Run 2 renamed every stage,
        `REAL_DATA_STAGES` still listed run 1's, the intersection was empty, and
        the tier-1 measured likelihood in a stage called `T1_measured_founding`
        contributed nothing. Admission here comes from the config, through the
        same `stage_admission` object `run_stage` builds.
        """
        admission = stage_admission(stage, cards_dir=self.cfg.mixture_cards, strict=False)
        return bool(admission.admits_measured(self.sources))

    def stage_sources(
        self, stage: StageConfig, admission: StageAdmission | None = None
    ) -> dict[str, SourceSpec]:
        """Intersect each card's ``A_k`` with the stage allowlist (restrict only).

        With an ``admission`` both the allowlist and the admitted source set come
        from the config.  Without one the legacy behaviour is kept for the five
        stage names 001-beta used -- but note its default,
        ``STAGE_PERMISSIONS.get(name, ("*",))``, grants **everything** to any
        other name, so an unrecognised stage silently widened every mask.
        """
        if admission is not None:
            allow = admission.allow_globs()
            admitted: set[str] | None = set(admission.source_ids)
        else:
            allow = STAGE_PERMISSIONS.get(stage.name, ("*",))
            admitted = None
        out: dict[str, SourceSpec] = {}
        for sid, s in self.sources.items():
            if not s.enabled:
                continue
            if admitted is not None and sid not in admitted:
                continue
            if allow == ("*",):
                perm = s.gradient_permission
            else:
                # "Restrict only" means the RESULT must be no broader than either
                # side. Where a card pattern and an allowlist entry overlap, keep
                # the NARROWER of the two. Keeping the card's pattern -- the
                # previous behaviour -- silently widened the stage: `eeg.*`
                # survived against an allowlist naming only `eeg.log_gain`,
                # `eeg.offset`, `eeg.log_noise`, `eeg.nuisance*`, which let
                # Stage V train `eeg.source_proj.*` (1,281 params) undeclared.
                narrowed: list[str] = []
                for p in s.gradient_permission:
                    if p == "*":
                        continue  # handled by the "*" branch below
                    for a in allow:
                        if not _glob_overlap(p, a):
                            continue
                        if fnmatch.fnmatch(a, p):
                            narrowed.append(a)  # allowlist entry is the narrower
                        elif fnmatch.fnmatch(p, a):
                            narrowed.append(p)  # card pattern is the narrower
                        else:
                            narrowed.append(a)  # incomparable: prefer the stage
                perm = tuple(dict.fromkeys(narrowed))
                if "*" in s.gradient_permission:
                    perm = allow
            out[sid] = SourceSpec(**{**s.as_dict(), "gradient_permission": perm})
        return out

    # ------------------------------------------------------------------
    # leakage barrier
    # ------------------------------------------------------------------
    def _audit_real_split(self, split: Mapping[str, Any], dataset: Any) -> dict[str, Any]:
        """Refuse to train on measured data whose split has not been audited.

        Measured human recordings are the only source in this run that can
        support a claim about brains, and a participant appearing on both sides
        of the split turns memorisation of that person into a reported
        generalisation (refusal **R10**).  Unlike the corpus limitations, this
        one cannot be caveated afterwards -- it invalidates every held-out number
        the model could produce.

        So this is a **gate, not a report**: it runs before the first measured
        window reaches a loss, and raises rather than warning.  It also records
        the verdict on the source specs, which is what makes
        ``leakage_checked`` in the compiled schema mean something.
        """
        from .realdata import leakage_check

        audit = leakage_check(split, dataset)
        self.leakage_audit = audit
        backend = audit.get("split_backend", "unknown")

        if not audit["ok"]:
            raise RuntimeError(
                "participant-level leakage audit FAILED (R10); refusing to train on "
                "measured data. Violations: "
                + json.dumps(audit["violations"][:5], default=str)
            )
        if backend != "grouped_splitter":
            raise RuntimeError(
                f"leakage audit passed but the split backend is {backend!r}, not "
                "'grouped_splitter'. R10 requires grouping by immutable lineage before "
                "splitting; a split that is merely disjoint was not constructed to be. "
                f"reason={audit.get('split_fallback_reason', '')!r}"
            )

        # The audit ran and passed -> the schema may now say so.  Only measured
        # sources are covered: a simulated source has no participants, and
        # asserting a leakage check over one would be the same empty claim in a
        # different place.
        for sid, spec in list(self.sources.items()):
            if not spec.is_simulated:
                self.sources[sid] = SourceSpec(**{**spec.as_dict(), "leakage_audited": True})

        print(
            f"[leakage] R10 audit PASSED  backend={backend}  "
            f"participants train/val/test="
            f"{audit['n_subjects_per_fold'].get('train')}/"
            f"{audit['n_subjects_per_fold'].get('val')}/"
            f"{audit['n_subjects_per_fold'].get('test')}"
            f" of {audit['n_subjects_total']}",
            flush=True,
        )
        for w in audit.get("warnings", []):
            print(f"[leakage] warning: {w}", flush=True)
        gs = audit.get("grouped_splitter_audit") or {}
        if gs:
            print(f"[leakage] GroupedSplitter cross-check ok={gs.get('ok')}", flush=True)
            for w in gs.get("warnings", []):
                print(f"[leakage] cross-check warning: {w}", flush=True)
        return audit

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def build_data(self) -> None:
        if self._data_ready:
            return
        d = self.cfg.data
        idx = Path(d.sim_index_fast)
        if not idx.exists():
            raise FileNotFoundError(
                f"simulated corpus index {idx} not found. Run "
                f"`python -m scwbd.foundation.simulate --tier fast` first."
            )
        win = d.window + d.context
        self.sim_train = SimCorpus(idx, window=win, trajectory_subset="train", val_fraction=d.val_fraction, seed=d.seed)
        self.sim_val = SimCorpus(idx, window=win, trajectory_subset="val", val_fraction=d.val_fraction, seed=d.seed)
        self.sim_loader = _cycle(
            torch.utils.data.DataLoader(
                self.sim_train,
                batch_size=d.batch,
                shuffle=True,
                num_workers=d.num_workers,
                drop_last=True,
                persistent_workers=d.num_workers > 0,
                pin_memory=d.pin_memory,
            )
        )
        self.sim_val_loader = torch.utils.data.DataLoader(
            self.sim_val, batch_size=min(d.batch, 128), shuffle=False, num_workers=2
        )
        # real EEG (measured -- the only source that can support a claim about brains)
        self.real_train = self.real_val = self.real_test = None
        try:
            from .realdata import EEGMMIDBDataset, RealEEGConfig, participant_split

            rc = RealEEGConfig(
                window_s=win / d.fs_hz,
                fs_target=d.fs_hz,
                max_subjects=None if not self.quick else 6,
                max_runs_per_subject=None if not self.quick else 2,
                seed=d.seed,
            )
            ds = EEGMMIDBDataset(rc)
            if len(ds) > 0:
                split = participant_split(ds, test_fraction=d.real_test_fraction, val_fraction=0.1, seed=d.seed)
                # HARD GATE, before a single measured window can reach a loss.
                # The routine existed and simply was not called; Stage III was
                # gated by a coordinator remembering to ask, which worked once.
                self._audit_real_split(split, ds)
                self.real_dataset = ds
                self.real_split = split
                self.real_train = torch.utils.data.Subset(ds, split["train"])
                self.real_val = torch.utils.data.Subset(ds, split["val"])
                self.real_test = torch.utils.data.Subset(ds, split["test"])
                self.real_loader = _cycle(
                    torch.utils.data.DataLoader(
                        self.real_train,
                        batch_size=max(8, d.batch // 4),
                        shuffle=True,
                        num_workers=min(4, d.num_workers),
                        drop_last=True,
                    )
                )
                print(
                    f"real EEG: {len(ds)} windows, train/val/test = "
                    f"{len(self.real_train)}/{len(self.real_val)}/{len(self.real_test)}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - measured data optional at build time
            print(f"[warn] real EEG unavailable ({type(exc).__name__}: {exc}); "
                  "training will proceed on simulator-conditioned evidence only, and the claim "
                  "manifest will say so.", flush=True)

        # measured BOLD, already in parcel space with its coverage mask.  Built
        # only from the CACHE: parcellating a run costs ~160 s of registration,
        # which does not belong inside a training step, and a stage that silently
        # spent twenty minutes registering would look like a hang.
        self.bold_loader = None
        try:
            from .bolddata import ParcelBOLDConfig, ParcelBOLDDataset

            bcfg = ParcelBOLDConfig()
            bds = ParcelBOLDDataset(bcfg, build=False)
            if len(bds) > 0:
                self.bold_dataset = bds
                self.bold_loader = _cycle(
                    torch.utils.data.DataLoader(
                        bds,
                        batch_size=max(4, d.batch // 8),
                        shuffle=True,
                        num_workers=min(2, d.num_workers),
                        drop_last=True,
                    )
                )
                s = bds.summary()
                print(
                    f"measured BOLD: {s['windows']} windows over {s['participants']} "
                    f"participants, {s['runs_cached']}/{s['runs_discovered']} runs cached, "
                    f"{s['runs_dropped_low_coverage']} dropped for coverage",
                    flush=True,
                )
                if bds.dropped_runs:
                    print(f"  dropped: {bds.dropped_runs}", flush=True)
            else:
                print(
                    "[warn] parcel-space BOLD found no cached windows. Run "
                    "ParcelBOLDDataset(cfg).build_cache() first (~160 s per run); this "
                    "stage will contribute no BOLD term rather than a zero one.",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - BOLD optional at build time
            print(f"[warn] parcel BOLD unavailable ({type(exc).__name__}: {exc}); "
                  "no BOLD likelihood will be computed.", flush=True)

        self._build_run3_sources(win, d)
        self._data_ready = True

    # ------------------------------------------------------------------
    # run 3 sources
    # ------------------------------------------------------------------
    def _build_run3_sources(self, win: int, d) -> None:
        """The sources added for run 3, each keyed by the id its card uses.

        Every one of these is optional at build time and reports why it is
        absent. What must NOT happen is a source that a stage admits, whose
        loader is missing, contributing a zero term that reads in the mixture
        report as "trained and had no effect" -- so `run_stage` checks the
        loader dict and `contributed_sources` records what actually ran.
        """
        from .realdata import (
            DS000117BehaviourDataset,
            DS000117EEGDataset,
            DS004024RestDataset,
            RealEEGConfig,
            SleepEDFDataset,
            participant_split,
        )

        self.eeg_loaders: dict[str, Any] = {}
        self.eeg_datasets: dict[str, Any] = {}
        self.behaviour_loader = None
        self.behaviour_dataset = None
        self.perturb_loader = None
        self.perturb_dataset = None
        self.bold_loaders: dict[str, Any] = {}
        self.bold_datasets: dict[str, Any] = {}

        # The founding montage is already built above; register it under its
        # card id so one loop can serve every EEG source.
        if getattr(self, "real_loader", None) is not None:
            self.eeg_loaders["eegmmidb_real"] = self.real_loader
            self.eeg_datasets["eegmmidb_real"] = self.real_dataset

        rc = RealEEGConfig(
            eegmmidb_root=Path(d.real_eeg_root),
            sleep_edfx_root=Path(d.real_sleep_root) / "sleep-cassette"
            if not str(d.real_sleep_root).endswith("sleep-cassette")
            else Path(d.real_sleep_root),
            ds000117_root=Path(d.ds000117_root),
            ds004024_root=Path(d.ds004024_root),
            window_s=win / d.fs_hz,
            fs_target=d.fs_hz,
            max_subjects=None if not self.quick else 2,
            max_runs_per_subject=None if not self.quick else 1,
            seed=d.seed,
        )

        def _add(source_id: str, cls, batch_div: int = 4) -> None:
            try:
                ds = cls(rc)
            except Exception as exc:  # noqa: BLE001 - a source is optional, its absence is not
                print(f"[warn] {source_id} unavailable ({type(exc).__name__}: {exc})", flush=True)
                return
            if len(ds) == 0:
                print(f"[warn] {source_id}: 0 windows on disk; it will contribute "
                      "no term rather than a zero one", flush=True)
                return
            split = participant_split(ds, test_fraction=d.real_test_fraction, val_fraction=0.1, seed=d.seed)
            self._audit_real_split(split, ds)
            self.eeg_datasets[source_id] = ds
            self.eeg_loaders[source_id] = _cycle(
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(ds, split["train"]),
                    batch_size=max(4, d.batch // batch_div),
                    shuffle=True,
                    num_workers=min(2, d.num_workers),
                    drop_last=True,
                )
            )
            print(f"{source_id}: {len(ds)} windows over {len(ds.subjects)} participants, "
                  f"{len(ds.channel_names)} channels", flush=True)

        _add("sleepedf_real", SleepEDFDataset)
        _add("ds000117_real", DS000117EEGDataset)
        _add("ds004024_rest_real", DS004024RestDataset)

        # -- the boundary output ------------------------------------------
        try:
            bds = DS000117BehaviourDataset(rc)
            if len(bds) > 0:
                split = participant_split(bds, test_fraction=d.real_test_fraction, val_fraction=0.1, seed=d.seed)
                self.behaviour_dataset = bds
                self.behaviour_loader = _cycle(
                    torch.utils.data.DataLoader(
                        torch.utils.data.Subset(bds, split["train"]),
                        batch_size=max(4, d.batch // 8),
                        shuffle=True,
                        num_workers=0,
                        drop_last=True,
                    )
                )
                print(f"ds000117_behaviour: {len(bds)} stimulus-locked episodes over "
                      f"{len(bds.subjects)} participants -- the first boundary_output "
                      "in the mixture", flush=True)
            else:
                print("[warn] ds000117_behaviour: no episodes; `behaviour.*` will be "
                      "unreachable and the attachment report will say so", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] ds000117_behaviour unavailable ({type(exc).__name__}: {exc})", flush=True)

        # -- measured perturbation ----------------------------------------
        if d.enable_perturbation:
            try:
                from .perturb import TMSDrive, TMSEpochConfig, TMSEpochDataset

                pds = TMSEpochDataset(TMSEpochConfig(
                    root=d.ds004024_root,
                    fs_target=d.fs_hz,
                    max_subjects=None if not self.quick else 1,
                    max_runs_per_subject=None if not self.quick else 1,
                ))
                if len(pds) > 0:
                    self.perturb_dataset = pds
                    self.perturb_loader = _cycle(
                        torch.utils.data.DataLoader(
                            pds, batch_size=max(4, d.batch // 8), shuffle=True,
                            num_workers=0, drop_last=True,
                        )
                    )
                    if getattr(self, "tms_drive", None) is None:
                        self.tms_drive = TMSDrive(self.anat).to(self.device)
                        # Record its initialisation NOW. It is built here rather
                        # than in `__init__`, so without this its parameters are
                        # missing from `init_fingerprint` and every checkpoint
                        # would report the drive as moved -- including one that
                        # never took a step.
                        self._fingerprint_late_module("tms_drive", self.tms_drive)
                    s = pds.summary()
                    print(f"ds004024_perturb: {s['epochs']} TMS-evoked epochs over "
                          f"{s['participants']} participants, {s['by_hemisphere']}, "
                          f"{s['scored_steps_per_epoch']} scored steps each", flush=True)
                    if pds.skipped:
                        print(f"  skipped: {pds.skipped}", flush=True)
                else:
                    print("[warn] ds004024_perturb: no epochs; the measured-perturbation "
                          "term will be absent", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] ds004024_perturb unavailable ({type(exc).__name__}: {exc})", flush=True)

        # -- extra BOLD corpora --------------------------------------------
        if getattr(self, "bold_loader", None) is not None:
            self.bold_loaders["ds002336_real"] = self.bold_loader
            self.bold_datasets["ds002336_real"] = getattr(self, "bold_dataset", None)
        for sid, root in (d.bold_roots or {}).items():
            try:
                from .bolddata import ParcelBOLDConfig, ParcelBOLDDataset

                bd = ParcelBOLDDataset(ParcelBOLDConfig(root=root, source=sid), build=False)
                if len(bd) == 0:
                    print(f"[warn] {sid}: no cached parcellated windows under {root}", flush=True)
                    continue
                self.bold_datasets[sid] = bd
                self.bold_loaders[sid] = _cycle(
                    torch.utils.data.DataLoader(
                        bd, batch_size=max(4, d.batch // 8), shuffle=True,
                        num_workers=min(2, d.num_workers), drop_last=True,
                    )
                )
                s = bd.summary()
                print(f"{sid}: {s['windows']} BOLD windows over {s['participants']} "
                      f"participants, {s['runs_cached']}/{s['runs_discovered']} runs cached",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {sid} unavailable ({type(exc).__name__}: {exc})", flush=True)

    # ------------------------------------------------------------------
    # smoke
    # ------------------------------------------------------------------
    def smoke(self) -> dict[str, Any]:
        """One batch, forward **and backward**, through **both** loss paths.

        A launch precondition, not a habit. Every launch-blocking defect of this
        cycle -- binding drift, the capacity mismatch, and the `SpanViolation`
        that cost this run its first start -- was a constructor- or
        first-rollout-time failure, and each was found by launching into a log
        nobody was watching. All three surface here in under a minute.

        Build-only is not enough: the `SpanViolation` was raised *inside a
        rollout*, so the batch must actually roll and actually backpropagate.

        **Both arms must be smoked.** The control arm has no mechanistic
        families, so it passes the `SpanViolation` path no matter what is or is
        not bound -- only the treatment arm exercises that guard at all, and a
        fix verified against the control is not verified (RL-11's corollary).
        """
        t0 = time.time()
        print(f"\n=== SMOKE: {self.cfg.train.run_name} ===", flush=True)
        self.build_data()
        stages = [s for s in self.cfg.train.stages if s.enabled]
        if not stages:
            raise RuntimeError("smoke: config has no enabled stage to draw loss weights from")
        sim_stage = stages[0]
        real_stage = next((s for s in stages if s.name in ("III_sliced", "IV_assembly", "V_individual")), sim_stage)

        mech = sorted(self.model.family_local.mech) if self.model.family_local is not None else []
        print(
            f"[smoke] mechanistic families ({len(mech)}): {mech if mech else '(none -- control arm; '
            'this arm CANNOT verify the theta bind)'}",
            flush=True,
        )

        params = [p for p in list(self.model.parameters()) + list(self.posterior.parameters()) if p.requires_grad]
        self.model.train()
        self.posterior.train()
        report: dict[str, Any] = {
            "run_name": self.cfg.train.run_name,
            "mechanistic_families": mech,
            "paths": {},
        }
        if self.real_train is None:
            # build_data swallows real-EEG failures with a warning; for a launch
            # precondition that is a failure, not a warning.
            raise RuntimeError(
                "smoke: measured EEG is unavailable, so the real loss path cannot be "
                "exercised. build_data() only warns about this. Refusing to report a "
                "pass on half the paths."
            )
        # Will the SCHEDULE ever run the measured source? A loss path that works
        # when called directly (as below) says nothing about whether `run_stage`
        # ever calls it. Run 2 renamed every stage, `REAL_DATA_STAGES` still
        # names run 1's, and the intersection is empty -- so the tier-1 measured
        # likelihood that "FOUNDS the representation" would contribute exactly
        # nothing, in a run whose first stage is called `T1_measured_founding`.
        # Scoped to exactly the source `real_losses` emits (REAL_LOSS_KEY). Every
        # OTHER enabled card -- montage_calibration, negative_control_shuffled --
        # is a separate question, and sweeping them in here would make this fire
        # for four reasons when one is true, which is this register's own failure
        # mode. `anatomical_prior` in particular is NOT affected: it arrives via
        # `sim_losses` and reaches a loss in every stage.
        enabled_stages = [s.name for s in self.cfg.train.stages if s.enabled]
        admitting = [s.name for s in self.cfg.train.stages if s.enabled and self.stage_uses_real(s)]
        spec = self.sources.get(REAL_LOSS_KEY)
        report["stages_admitting_measured"] = admitting
        if spec is not None and spec.enabled and not admitting:
            raise RuntimeError(
                f"smoke: source card {REAL_LOSS_KEY!r} is enabled and is the tier-1 measured "
                f"likelihood, but NONE of the enabled stages {enabled_stages} admits it -- "
                f"`REAL_DATA_STAGES`={sorted(REAL_DATA_STAGES)} names run 1's stages and "
                "intersects this schedule in nothing. Every measured window would be loaded, "
                "split, leakage-audited and then never used: the run would complete on "
                "simulation alone, and no claim about brains would be supportable from it. "
                "Fix the curriculum, not this check."
            )
        print(f"[smoke] measured source {REAL_LOSS_KEY!r} admitted by stages {admitting}", flush=True)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for p in params:
            p.grad = None

        # BOTH forwards BEFORE any backward -- this is what `run_stage` does, and
        # `MixtureTrainer.step` then backwards each source with retain_graph=True,
        # so the real peak holds both activation graphs at once. Backwarding each
        # path separately would understate peak memory and let an OOM through.
        losses: dict[str, Tensor] = {}
        for path, fn, st, loader in (
            ("sim_losses", self.sim_losses, sim_stage, "sim"),
            ("real_losses", self.real_losses, real_stage, "real"),
        ):
            batch = next(self.sim_loader if loader == "sim" else self.real_loader)
            ls, _diag = fn(batch, st)
            for k, v in ls.items():
                if not torch.isfinite(v):
                    raise RuntimeError(f"smoke: {path} term {k!r} is non-finite ({float(v)!r})")
            report["paths"][path] = {"stage": st.name, "terms": sorted(ls),
                                     "values": {k: float(v.detach()) for k, v in ls.items()}}
            print(f"[smoke] {path:12s} stage={st.name:14s} "
                  f"{ {k: round(float(v.detach()), 4) for k, v in ls.items()} }", flush=True)
            losses.update(ls)

        # -- the paths run 3 adds -----------------------------------------
        # Without these the precondition passes while three loss paths are
        # unverified, which is a smoke test that measures the parts that already
        # worked. Each is REQUIRED when its loader exists: a path that silently
        # skips is the same failure as a card that grants nothing.
        params = params + [p for p in (self.tms_drive.parameters() if self.tms_drive else [])]
        extra_paths: list[tuple[str, Any, Any]] = []
        for sid, loader in sorted(getattr(self, "eeg_loaders", {}).items()):
            if sid == "eegmmidb_real":
                continue
            extra_paths.append((f"real_losses[{sid}]",
                                lambda b, s, _sid=sid: self.real_losses(b, s, source_id=_sid),
                                loader))
        for sid, loader in sorted(getattr(self, "bold_loaders", {}).items()):
            if sid == "ds002336_real":
                continue
            extra_paths.append((f"real_bold_losses[{sid}]",
                                lambda b, s, _sid=sid: self.real_bold_losses(b, s, source_id=_sid),
                                loader))
        if getattr(self, "behaviour_loader", None) is not None:
            extra_paths.append(("behaviour_losses", self.behaviour_losses, self.behaviour_loader))
        if getattr(self, "perturb_loader", None) is not None:
            extra_paths.append(("perturb_losses", self.perturb_losses, self.perturb_loader))

        for path, fn, loader in extra_paths:
            batch = next(loader)
            ls, diag = fn(batch, real_stage)
            for k, v in ls.items():
                if not torch.isfinite(v):
                    raise RuntimeError(f"smoke: {path} term {k!r} is non-finite ({float(v)!r})")
            report["paths"][path] = {
                "stage": real_stage.name,
                "terms": sorted(ls),
                "values": {k: float(v.detach()) for k, v in ls.items()},
                "diagnostics": {k: v for k, v in diag.items() if isinstance(v, (int, float))},
            }
            print(f"[smoke] {path:34s} "
                  f"{ {k: round(float(v.detach()), 4) for k, v in ls.items()} }", flush=True)
            losses.update(ls)

        # The attachment kinds this smoke actually exercised, named rather than
        # counted: "five loss paths ran" does not say a boundary output did.
        report["attachment_kinds_exercised"] = sorted(
            {
                "observation",
                *(["boundary_output"] if "behaviour_losses" in report["paths"] else []),
                *(["stimulus"] if "perturb_losses" in report["paths"] else []),
            }
        )
        print(f"[smoke] attachment kinds exercised: "
              f"{report['attachment_kinds_exercised']}", flush=True)

        total = sum(losses.values())
        total.backward()
        with torch.no_grad():
            grads = [p.grad for p in params if p.grad is not None]
            n_nonfinite = sum(1 for g in grads if not torch.isfinite(g).all())
            gnorm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
        if not grads:
            raise RuntimeError("smoke: backward produced no gradients at all")
        if n_nonfinite:
            raise RuntimeError(f"smoke: {n_nonfinite} parameter gradients are non-finite")
        report["total_loss"] = float(total.detach())
        report["params_with_grad"] = len(grads)
        report["grad_norm"] = gnorm
        print(f"[smoke] backward ok: grads={len(grads)}/{len(params)} |g|={gnorm:.3e} "
              f"total={float(total.detach()):.4f}", flush=True)

        # Which MODULES actually received a gradient, at step 0, before any
        # optimiser has run. This is the run-2 question asked at the only moment
        # it is cheap to answer: `test_regional_tensors_moved` needs a trained
        # stage, and by then the run has spent hours. A module here with
        # `n_nonzero == 0` while it sits on the forward path will still be at
        # its initialisation when the stage ends.
        #
        # Non-zero, not merely non-None: a `.grad` full of exact zeros is what a
        # module reached only through a detached path looks like, and it counts
        # as "has a gradient" to every check that tests for None.
        with torch.no_grad():
            named = list(self.model.named_parameters())
            named += [(f"posterior.{n}", p) for n, p in self.posterior.named_parameters()]
            if self.tms_drive is not None:
                named += [(f"tms_drive.{n}", p) for n, p in self.tms_drive.named_parameters()]
            by_module: dict[str, dict[str, int]] = {}
            for name, p in named:
                top = name.split(".")[0]
                e = by_module.setdefault(top, {"n": 0, "n_grad": 0, "n_nonzero": 0})
                e["n"] += 1
                if p.grad is not None:
                    e["n_grad"] += 1
                    if bool((p.grad != 0).any()):
                        e["n_nonzero"] += 1
        report["gradient_by_module"] = by_module
        dead = sorted(m for m, e in by_module.items() if e["n_nonzero"] == 0)
        report["modules_with_no_gradient"] = dead
        print("[smoke] gradient by module: "
              + ", ".join(f"{m}={e['n_nonzero']}/{e['n']}" for m, e in sorted(by_module.items())),
              flush=True)
        if dead:
            print(f"[smoke] MODULES WITH NO GRADIENT AT STEP 0: {dead}. Each will be "
                  "bit-identical to its initialisation when this stage ends unless a "
                  "later stage reaches it.", flush=True)
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            reserved = torch.cuda.max_memory_reserved(self.device) / 1e9
            cap = float(self.cfg.train.cuda_reserve_gb)
            report["peak_allocated_gb"] = round(peak, 2)
            report["peak_reserved_gb"] = round(reserved, 2)
            report["cuda_reserve_gb"] = cap
            print(f"[smoke] peak CUDA allocated={peak:.2f} GB reserved={reserved:.2f} GB "
                  f"against cap {cap:.1f} GB ({reserved / max(cap, 1e-9):.0%} of budget)", flush=True)
        for p in params:
            p.grad = None
        nf = self.model.eeg.noise_floor_report() if hasattr(self.model.eeg, "noise_floor_report") else {}
        report["noise_floor_report"] = nf
        print(f"[smoke] noise_floor_report(eeg) = {json.dumps(nf, default=str)}", flush=True)
        report["wall_seconds"] = round(time.time() - t0, 1)
        print(f"[smoke] PASS in {report['wall_seconds']}s -- both loss paths fwd+bwd\n", flush=True)
        return report

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def _split_window(self, act: Tensor) -> tuple[Tensor, Tensor]:
        c = self.cfg.data.context
        return act[:, :c], act[:, c:]

    def _slice_mask(self, B: int, N: int, *, p_observed: float = 0.65, generator=None) -> Tensor:
        """The observed subgraph ``S_e`` of an episode (equation 4).

        Regions outside ``S_e`` receive no reconstruction loss and their state is
        marginalised.  They are **not** trained toward an arbitrary default: "if
        an eye-tracking dataset says nothing about hypothalamus, the hypothalamic
        state is not trained to an arbitrary default" (§6.3).
        """
        u = torch.rand(B, N, device=self.device, generator=generator)
        keep = (u < p_observed).float()
        # every episode observes something
        empty = keep.sum(1) == 0
        if empty.any():
            keep[empty, 0] = 1.0
        return keep

    def sim_losses(
        self, batch: Mapping[str, Any], stage: StageConfig, admission: StageAdmission | None = None
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        act = batch["activity"].to(self.device, non_blocking=True)
        theta = batch["theta"].to(self.device, non_blocking=True)
        B, T, N = act.shape
        ctx_y, tgt_y = self._split_window(act)
        n_pred = tgt_y.shape[1]
        losses: dict[str, Tensor] = {}
        diag: dict[str, Any] = {}

        obs_mask = self._slice_mask(B, N)
        ctx_mask = obs_mask.unsqueeze(1).expand_as(ctx_y)

        # amortized posterior over theta from the *observed slice only*
        y_summary = ctx_y * ctx_mask
        nuis = torch.zeros(B, self.posterior.nuisance_dim, device=self.device)
        theta_full = torch.cat([theta, nuis], -1) if self.posterior.nuisance_dim else theta
        npe = self.posterior.loss(y_summary, theta_full)

        # boundary randomisation / corrupted inputs (Stage I) --------------
        u = None
        randomise = (
            admission.boundary_randomisation if admission is not None else stage.name == "I_regional"
        )
        if randomise:
            with torch.no_grad():
                amp = torch.rand(B, 1, 1, 1, device=self.device) * 0.4
            u = amp * torch.randn(B, n_pred, N, self.model.layout.dim, device=self.device)

        # Bind theta-conditioned ParamPacks for every mechanistic family BEFORE the
        # rollout, and OUTSIDE autocast: the packs are fp32 and the backends integrate
        # in fp32 per the ARCHITECTURE.md §3 numerical contract.  Per batch, never at
        # construction -- set_mechanistic_theta sizes each pack with batch=theta.shape[0]
        # and fills it with per-row values, so a construction-time bind would pin every
        # batch in the run to whichever theta was drawn first (silent; the loss still
        # falls).  One call: it fans out over family_local.mech internally, slicing each
        # pack to that family's own parcels.
        self.model.set_mechanistic_theta(theta, self.anat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(
                y_context=ctx_y,
                theta=theta,
                n_steps=n_pred,
                context_mask=ctx_mask,
                u=u,
                with_hemo=(
                    admission.with_hemo if admission is not None else stage.name in ("IV_assembly",)
                ),
                enforce_r05=False,
            )
        pred, lv = roll.activity.float(), roll.activity_logvar.float()
        tmask = obs_mask.unsqueeze(1).expand_as(tgt_y)
        forecast = gaussian_nll(tgt_y, pred, lv, mask=tmask)

        # KL[q(Z_e) || p(Z_e | U_e, C_e)] over the episode's latent theta
        with torch.no_grad():
            th_s = self.posterior.sample(y_summary, 1)[:, 0]
        kl = (self.posterior.log_prob(y_summary, th_s) - self.theta_prior.log_prob(th_s[:, : theta.shape[1]])).mean()

        reg = (
            stage.lambda_anat * self.model.coupling.topology_penalty()
            + stage.lambda_homeo * self.model.stability_penalty(roll.state)
            + stage.lambda_port * self.model.port_penalty(roll.state)
            + stage.lambda_residual * self.model.residual_penalty()
            + stage.lambda_cal * self.model.bold.prior_penalty()
        )
        losses["sim_wholebrain"] = (
            stage.lambda_forecast * forecast
            + stage.lambda_posterior * npe
            + stage.lambda_kl * kl.clamp(-50, 50)
            + reg
        )
        diag.update(
            sim_forecast_nll=float(forecast.detach()),
            npe_loss=float(npe.detach()),
            npe_rejected=int(type(self.posterior).npe_rejected),
            npe_seen_max=float(type(self.posterior).npe_seen_max),
            kl=float(kl.detach()),
            rho=float(roll.rho.detach()),
            observed_fraction=float(obs_mask.mean()),
            **{k: v for k, v in roll.diagnostics.items() if isinstance(v, (int, float))},
        )
        return losses, diag

    def anat_losses(self, admission: StageAdmission | None = None) -> dict[str, Tensor]:
        """The tier-3 population-prior term, independent of any data batch.

        This used to be composed inside :meth:`sim_losses`, so the anatomical
        prior only contributed on a step where the **simulated** loader ran.
        Under an integrity ordering there is a stage that admits tier 3 and not
        tier 4, and on the old code that stage would have emitted no tier-3 loss
        at all -- admitting the population prior in name only.
        """
        if "anatomical_prior" not in self.sources:
            return {}
        if admission is not None and "anatomical_prior" not in admission.source_ids:
            return {}
        return {
            "anatomical_prior": self.model.coupling.topology_penalty()
            + self.model.bold.prior_penalty()
        }

    def eeg_projector(self, source_id: str) -> "SensorToParcel":
        """The minimum-norm projector for one source's montage.

        One per montage, because the pseudo-inverse is a function of that
        montage's lead field. Sharing the 64-channel projector with a 2-channel
        source would be a shape error; sharing it with another 64-channel source
        on different electrodes would not be, and would quietly project through
        the wrong geometry. Cached: the inverse is a fixed cost per montage.
        """
        cache = getattr(self, "_projectors", None)
        if cache is None:
            cache = {}
            self._projectors = cache
        if source_id not in cache:
            head = self.model.eeg_head_for(source_id)
            cache[source_id] = (
                self.sensor_to_parcel
                if head is self.model.eeg
                else SensorToParcel(head.L).to(self.device)
            )
        return cache[source_id]

    def real_losses(
        self,
        batch: Mapping[str, Any],
        stage: StageConfig,
        *,
        source_id: str = "eegmmidb_real",
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Measured EEG: likelihood in **sensor space**, always.

        ``source_id`` selects the observation head and the projector. A source
        recorded on other electrodes is observed through its own operator; see
        ``SCWBD.eeg_head_for``.
        """
        eeg = batch["eeg"].to(self.device, non_blocking=True)  # (B,T,C)
        B, T, C = eeg.shape
        head = self.model.eeg_head_for(source_id)
        if C != head.L.shape[0]:
            raise ValueError(
                f"{source_id}: batch carries {C} channels and its observation head "
                f"has {head.L.shape[0]}. The channel axis is positional -- refusing "
                "to broadcast, which would score one electrode's data against "
                "another's forward row."
            )
        c = self.cfg.data.context
        ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
        n_pred = tgt_e.shape[1]
        with torch.no_grad():
            src_ctx = self.eeg_projector(source_id)(ctx_e)
            sd = src_ctx.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            src_ctx = src_ctx / sd
        th = self.posterior.sample(ctx_e, 1)[:, 0][:, : len(THETA_NAMES)].detach()
        if self.individualizer is not None:
            pid = self.participant_index(batch.get("subject", []))
            th = self.individualizer(participant=pid, base=th)
            self.individualizer.observe_session(pid)
        # Bind theta-conditioned ParamPacks for every mechanistic family BEFORE the
        # rollout, and OUTSIDE autocast: the packs are fp32 and the backends integrate
        # in fp32 per the ARCHITECTURE.md §3 numerical contract.  Per batch, never at
        # construction -- set_mechanistic_theta sizes each pack with batch=theta.shape[0]
        # and fills it with per-row values, so a construction-time bind would pin every
        # batch in the run to whichever theta was drawn first (silent; the loss still
        # falls).  One call: it fans out over family_local.mech internally, slicing each
        # pack to that family's own parcels.
        self.model.set_mechanistic_theta(th, self.anat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(y_context=src_ctx, theta=th, n_steps=n_pred, enforce_r05=False)
            mu, lv = head(roll.state)
        mu, lv = mu.float(), lv.float()
        scale = tgt_e.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        nll = gaussian_nll(tgt_e / scale, mu / scale.clamp_min(1e-8), lv - 2 * torch.log(scale))
        key = REAL_LOSS_KEY if source_id == "eegmmidb_real" else source_id
        losses = {key: nll}

        if self.individualizer is not None and source_id == "eegmmidb_real":
            losses[key] = losses[key] + 1e-3 * self.individualizer.prior_penalty()
        return losses, {f"{source_id}_eeg_nll": float(nll.detach())}

    def behaviour_losses(
        self,
        batch: Mapping[str, Any],
        stage: StageConfig,
        *,
        source_id: str = "ds000117_behaviour",
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """A boundary output: the button press the participant produced.

        The distinction this exercises is the one ``schema/attachment.py``
        exists for. An EEG channel is an ``observation`` -- the carrier seen
        through a lead field. A button press is a ``boundary_output``: produced
        *by* the subject and measured outside the skull, evidence about the
        carrier that reaches the world through the body rather than through a
        forward model of neural activity. It therefore attaches at
        ``BehaviourHead``, which reads pooled state directly and declares no
        operator, and **not** at an observation head.

        Two terms, kept separate because they are different quantities: a
        cross-entropy over which button, and a Gaussian NLL over log response
        time with the head's own predicted variance. Collapsing them into one
        scalar would let a confident-and-wrong RT be paid for by an easy choice.

        Scored at the LAST rolled step. The head pools over regions and time is
        not part of its input, so scoring every step would count one decision
        once per timestep and weight this source by its window length.
        """
        eeg = batch["eeg"].to(self.device, non_blocking=True)  # (B,T,C)
        choice = batch["choice"].to(self.device).long()
        log_rt = batch["log_rt"].to(self.device).float()
        B, T, C = eeg.shape
        head_id = "ds000117_real"  # same cap, same operator as the EEG source
        obs = self.model.eeg_head_for(head_id)
        if C != obs.L.shape[0]:
            raise ValueError(
                f"{source_id}: batch carries {C} channels, the {head_id} head has "
                f"{obs.L.shape[0]}"
            )
        c = self.cfg.data.context
        ctx_e = eeg[:, :c] if T > c else eeg
        n_pred = max(1, T - ctx_e.shape[1])
        with torch.no_grad():
            src_ctx = self.eeg_projector(head_id)(ctx_e)
            src_ctx = src_ctx / src_ctx.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        th = self.posterior.sample(ctx_e, 1)[:, 0][:, : len(THETA_NAMES)].detach()
        self.model.set_mechanistic_theta(th, self.anat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(y_context=src_ctx, theta=th, n_steps=n_pred, enforce_r05=False)
            out = self.model.behaviour(roll.state[:, -1:])
        logits = out["choice_logits"].float()[:, 0]  # (B, n_out)
        n_out = logits.shape[-1]
        if int(choice.max()) >= n_out:
            raise ValueError(
                f"{source_id}: a choice index {int(choice.max())} does not fit the "
                f"behaviour head's {n_out} outputs. Set cfg.model.n_behaviour to the "
                "number of distinct responses the source actually records."
            )
        ce = torch.nn.functional.cross_entropy(logits, choice)
        mean = out["log_rt_mean"].float()[:, 0]
        lv = out["log_rt_logvar"].float()[:, 0]
        rt_nll = gaussian_nll(log_rt, mean, lv)
        total = ce + rt_nll
        acc = float((logits.argmax(-1) == choice).float().mean().detach())
        return (
            {source_id: total},
            {
                "behaviour_choice_ce": float(ce.detach()),
                "behaviour_rt_nll": float(rt_nll.detach()),
                "behaviour_choice_acc": acc,
                # The majority-class rate is logged beside the accuracy on
                # purpose: this source's two buttons are far from balanced
                # (105/22 measured on sub-01), so an accuracy read without it
                # would look like learning when it is the prior.
                "behaviour_majority_rate": float(
                    torch.bincount(choice, minlength=n_out).max().float().div(len(choice)).detach()
                ),
            },
        )

    def perturb_losses(
        self,
        batch: Mapping[str, Any],
        stage: StageConfig,
        *,
        source_id: str = "ds004024_perturb",
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Measured perturbation: predict the response to a TMS pulse.

        The pre-pulse interval assimilates, the pulse enters as an exogenous
        latent drive (``SCWBD.rollout(u=...)``), and the likelihood is scored on
        post-pulse samples **outside the artefact window**. The rollout still
        integrates through the excluded interval -- the state evolves -- while
        those samples contribute no gradient, because deleting them would splice
        two segments together and ask the operator to cross a discontinuity it
        did not produce.

        What this does and does not license is stated on ``TMSDrive``: the drive
        is learned, anchored to the MEP-derived hemisphere's motor parcels, not
        computed from an E-field, because ds004024 distributes no coil pose.
        """
        drive = getattr(self, "tms_drive", None)
        if drive is None:
            raise RuntimeError(
                f"{source_id} was admitted but no TMSDrive was built; refusing to "
                "score a perturbation with no pulse in it"
            )
        eeg = batch["eeg"].to(self.device, non_blocking=True)  # (B,T,C)
        mask = batch["loss_mask"].to(self.device)  # (B,T) bool
        onsets = batch["onset_step"]
        onset = int(onsets[0])
        # One pulse index for the whole batch, so it must be the same for every
        # row. It is, because every run shares `tmin` and `fs_target` -- but a
        # future config that varied either per source would silently roll half
        # the batch from the wrong sample, and the response would still look
        # plausible.
        if int(onsets.min()) != onset or int(onsets.max()) != onset:
            raise ValueError(
                f"{source_id}: the batch mixes pulse onsets "
                f"{int(onsets.min())}..{int(onsets.max())}. The rollout starts at one "
                "index for the whole batch; refusing to apply it to rows whose pulse "
                "is elsewhere."
            )
        hemis = list(batch["hemisphere"])
        B, T, C = eeg.shape
        head = self.model.eeg_head_for("ds004024_rest_real")
        if C != head.L.shape[0]:
            raise ValueError(
                f"{source_id}: batch carries {C} channels, its head has {head.L.shape[0]}"
            )
        ctx_e = eeg[:, :onset]
        tgt = eeg[:, onset:]
        n_pred = tgt.shape[1]
        if n_pred < 1:
            raise ValueError(f"{source_id}: no post-pulse samples to score")
        with torch.no_grad():
            src_ctx = self.eeg_projector("ds004024_rest_real")(ctx_e)
            src_ctx = src_ctx / src_ctx.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        th = self.posterior.sample(ctx_e, 1)[:, 0][:, : len(THETA_NAMES)].detach()
        self.model.set_mechanistic_theta(th, self.anat)
        # The pulse lands on the first rolled step: the rollout begins at the
        # onset sample, so `onset_step=0` on the rollout's own axis.
        u = drive(
            self.model, hemis, n_steps=n_pred, onset_step=0, dt_s=self.cfg.model.dt_model
        ).to(device=self.device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(
                y_context=src_ctx, theta=th, n_steps=n_pred, u=u, enforce_r05=False
            )
            mu, lv = head(roll.state)
        mu, lv = mu.float(), lv.float()
        m = mask[:, onset:].unsqueeze(-1).expand_as(tgt).float()
        if float(m.sum()) == 0.0:
            raise ValueError(
                f"{source_id}: every post-pulse sample is masked out; the artefact "
                "window covers the whole scored interval"
            )
        scale = tgt.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        nll = gaussian_nll(
            tgt / scale, mu / scale.clamp_min(1e-8), lv - 2 * torch.log(scale), mask=m
        )
        gains = {h: float(drive.log_gain[h].detach().exp()) for h in ("left", "right")}
        return (
            {source_id: nll},
            {
                "perturb_nll": float(nll.detach()),
                "perturb_scored_frac": float(m.mean().detach()),
                "perturb_gain_left": gains["left"],
                "perturb_gain_right": gains["right"],
            },
        )

    def _whole_brain_hemo(self, state: "Tensor") -> "Tensor":
        """``(..., N, D)`` structured state -> ``(..., N, 4)`` Balloon state.

        This line used to read ``family_layout.component(state, "hemo")``, a
        method that does not exist on :class:`FamilyStateLayout`. It had never
        raised because the whole BOLD branch was unreachable until
        ``ds002336_real`` was admitted -- the fourth thing in this function to
        fail the first time it ran.

        The layout is per-family by construction: ``get()`` returns
        ``(..., n_f, dim)`` for one family, and there is no whole-brain read,
        because a component means different channels in different families. So
        the assembly is explicit -- gather each family's ``hemo`` and scatter it
        onto that family's region indices.

        A family that declares no ``hemo`` **refuses** rather than contributing
        zeros. Zero is a valid Balloon state (``v`` and ``q`` at rest), so a
        silent zero-fill would produce a plausible BOLD prediction for regions
        with no haemodynamic model at all, which is a fabricated observation on
        the model side of the likelihood.
        """
        layout = getattr(self.model, "family_layout", None)
        if layout is None:
            return state
        out = torch.zeros(
            (*state.shape[:-1], 4), dtype=state.dtype, device=state.device
        )
        for fam in layout.families:
            # `.name` on the layout's own RegionFamily, NOT `.family_id` -- the
            # anatomy prior's RegionFamily uses `family_id`, the state layout's
            # uses `name`, and they are different classes with overlapping
            # vocabulary. Reaching for the wrong one interpolated an entire
            # dataclass into a KeyError.
            name = str(getattr(fam, "name", fam))
            try:
                block = layout.get(state, name, "hemo")  # (..., n_f, 4)
            except Exception as exc:
                raise ValueError(
                    f"family {name!r} declares no 'hemo' component "
                    f"({type(exc).__name__}). The BOLD "
                    "likelihood needs a Balloon state for every region it scores. "
                    "Refusing to zero-fill: zero is a legal Balloon state, so the "
                    "missing regions would score as resting tissue rather than as "
                    "unmodelled ones."
                ) from exc
            idx = layout.index(name, device=state.device)
            out = out.index_copy(-2, idx, block.to(out.dtype))
        return out

    def real_bold_losses(
        self, batch: Mapping[str, Any], stage: StageConfig, *, source_id: str = "ds002336_real"
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Measured BOLD: likelihood in **parcel space**, over covered parcels only.

        The counterpart to :meth:`real_losses`, and deliberately not folded into
        it. EEG is compared in *sensor* space -- the model projects its parcel
        state forward through the lead field and the likelihood lives on
        channels. BOLD is compared in *parcel* space, because ``BOLDHead.signal``
        already emits per-region percent-signal-change and the data has been
        brought to parcels by ``scwbd.sources.parcellate_bold``. Two modalities,
        two supports, one carrier -- which is the whole point of O-1.

        **``mask`` is required and is not a convenience.** A parcel outside the
        acquisition's field of view has no observation, and the difference
        between excluding it and scoring it against 0.0 is the difference
        between a likelihood and an imputation (``ARCHITECTURE.md`` §7 rule 1).
        ``parcellate_run`` emits ``NaN`` for exactly those parcels, so a caller
        that forgets the mask gets ``NaN`` loss rather than a plausible number --
        the failure is loud by construction.
        """
        bold = batch["bold"].to(self.device, non_blocking=True)  # (B, T_slow, N)
        mask = batch.get("bold_mask")
        if mask is None:
            raise ValueError(
                "real_bold_losses requires batch['bold_mask']: parcels outside the "
                "acquisition are unobserved, and scoring them against any value at "
                "all is an imputation. parcellate_run() returns the coverage mask "
                "alongside the timeseries so this cannot be skipped by accident."
            )
        mask = mask.to(self.device, non_blocking=True).bool()  # (B, N) or (N,)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(bold.shape[0], -1)
        if not bool(mask.any()):
            raise ValueError(
                f"{source_id}: no parcel is covered in this batch. A run whose "
                "field of view intersects no parcel is a registration failure, not "
                "an empty likelihood."
            )

        # The atlas is cortex-only; the carrier is not.
        #
        # `Schaefer400x7` gives 400 parcels. The model's regional state is 414 --
        # 400 cortical, 14 subcortical (Tian) -- and the anatomy's label order is
        # cortex first, verified below rather than assumed. So parcel *i* is
        # region *i* for i < 400, and the 14 subcortical regions have no BOLD
        # measurement at all.
        #
        # They are padded in as UNCOVERED rather than dropped, because dropping
        # them would mean the likelihood scores a 400-vector against a 414-vector
        # by truncating the model -- silently deciding that subcortex does not
        # exist. Uncovered is the true statement: this acquisition and this atlas
        # say nothing about those regions, and `gaussian_nll(..., mask=)`
        # marginalises them out.
        n_regions = int(getattr(self.anat, "n_regions", bold.shape[-1]))
        if bold.shape[-1] != n_regions:
            n_atlas = int(bold.shape[-1])
            # Verified against the carrier's own per-region division, not
            # against a count. The property that makes index alignment valid is
            # "the first n_atlas regions are exactly the cortical ones", and a
            # count of 400 would also be satisfied by an ordering that
            # interleaves them.
            div = [str(x) for x in (getattr(self.anat, "division", ()) or ())]
            head_ok = n_atlas <= n_regions and len(div) == n_regions
            head_ok = head_ok and all(d == "cortex" for d in div[:n_atlas])
            head_ok = head_ok and not any(d == "cortex" for d in div[n_atlas:])
            if not head_ok:
                seen = sorted(set(div)) or ["<no division>"]
                raise ValueError(
                    f"{source_id}: BOLD carries {n_atlas} parcels and the carrier has "
                    f"{n_regions} regions (divisions {seen}). Index alignment is only "
                    "valid when the atlas is exactly the carrier's cortical prefix, and "
                    "that is not the case here. Refusing to align two parcellations by "
                    "index when nothing establishes they are the same parcellation -- "
                    "the namespace error this project keeps finding, applied to anatomy."
                )
            pad = torch.full(
                (*bold.shape[:-1], n_regions - n_atlas),
                float("nan"),
                dtype=bold.dtype,
                device=bold.device,
            )
            bold = torch.cat([bold, pad], dim=-1)
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], n_regions - n_atlas), dtype=torch.bool, device=mask.device)],
                dim=-1,
            )

        c = self.cfg.data.context
        ctx_b, tgt_b = bold[:, :c], bold[:, c:]
        n_pred = tgt_b.shape[1]

        # Context enters the carrier the same way EEG's does: normalised, and
        # over covered parcels only. Uncovered entries are NaN from the
        # parcellation, so they are zeroed AFTER normalisation statistics are
        # taken over the covered set -- zeroing first would drag the mean.
        with torch.no_grad():
            src_ctx = torch.nan_to_num(ctx_b, nan=0.0)
            m3 = mask.unsqueeze(1).expand_as(src_ctx)
            denom = m3.sum(dim=(1, 2), keepdim=True).clamp_min(1)
            mean = (src_ctx * m3).sum(dim=(1, 2), keepdim=True) / denom
            var = (((src_ctx - mean) * m3) ** 2).sum(dim=(1, 2), keepdim=True) / denom
            scale = var.sqrt().clamp_min(1e-6)
            src_ctx = torch.where(m3, (src_ctx - mean) / scale, torch.zeros_like(src_ctx))

        # theta from the PRIOR, not from the amortised posterior.
        #
        # This line used to read `self.posterior.sample(ctx_b.mean(-1), 1)`, and
        # it had never executed: `ds002336_real` was not admitted by any stage
        # until 2026-08-07, so the whole BOLD branch was unreachable. The first
        # time it ran it raised `not enough values to unpack (expected 3, got 2)`
        # -- `ctx_b.mean(-1)` is `(B, T)` and the summary encoder wants
        # `(B, T, C)`.
        #
        # The rank was the symptom. The posterior's summary encoder is built
        # around a fixed channel count -- the EEG montage's 64 -- and BOLD
        # context is 400 parcels. There is no shape-compatible way to pass it
        # without inventing a 400 -> 64 projection, and a projection invented
        # here would be an unvalidated forward operator sitting inside a
        # likelihood, which is precisely what the lead field exists to prevent
        # (ARCHITECTURE.md RL-05).
        #
        # So the BOLD likelihood gets prior theta and says so. That is weaker
        # than amortised inference and it is honest about which: the posterior
        # was fitted on EEG summaries and has no claim on parcel-space input.
        # The real fix is a per-modality posterior, which is a modelling change
        # and not something to smuggle in as a reshape.
        th = self.theta_prior.sample(
            bold.shape[0], seed=int(self.global_step), device=str(bold.device)
        )[:, : len(THETA_NAMES)].detach()

        self.model.set_mechanistic_theta(th, self.anat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(y_context=src_ctx, theta=th, n_steps=n_pred, enforce_r05=False)
            hemo = self._whole_brain_hemo(roll.state)
            mu, lv = self.model.bold.signal(hemo, roll.state)
        mu, lv = mu.float(), lv.float()

        # The mask is the whole point: gaussian_nll marginalises unobserved
        # elements out rather than scoring them, so uncovered parcels contribute
        # nothing and do not enter the denominator either.
        # The TARGET is normalised with the CONTEXT's statistics.
        #
        # It was not, and the first run of this path scored a normalised
        # prediction against a raw percent-signal-change target: real_bold_nll
        # came out at 1.7e+07, which would have dominated every other term and
        # destroyed the run without ever looking like a units bug -- it looks
        # like a model that cannot fit BOLD.
        #
        # This is the same defect already catalogued on the EEG side, where the
        # published comparison is offset by `mean(log s) = 0.5694` nats
        # (reports/RUN2.md §4). The lesson taken from it here is not just to
        # scale, but to REPORT the scale: `bold_log_scale` is the Jacobian term,
        # so `NLL_raw = NLL_scaled + bold_log_scale` and nobody has to rediscover
        # the offset from the source.
        #
        # Statistics come from the context window only. Taking them over the
        # target too would leak the thing being predicted into its own
        # normalisation.
        m3t = mask.unsqueeze(1).expand_as(tgt_b).to(mu.dtype)
        tgt_n = torch.where(
            m3t.bool(),
            (torch.nan_to_num(tgt_b, nan=0.0) - mean) / scale,
            torch.zeros_like(tgt_b),
        )
        nll = gaussian_nll(tgt_n, mu, lv, mask=m3t)
        log_scale = float(scale.log().mean().detach())

        return {source_id: nll}, {
            "real_bold_nll": float(nll.detach()),
            # Named in the metrics so a reader of the run does not have to infer
            # it from the source: this term is not amortised.
            "bold_theta_source": "prior",
            # NLL_raw = NLL_scaled + bold_log_scale. Reported so the BOLD term is
            # comparable against anything scored in percent signal change.
            "bold_log_scale": log_scale,
            "bold_parcels_covered": int(mask[0].sum()),
            "bold_parcels_total": int(mask.shape[1]),
        }

    # ------------------------------------------------------------------
    # stage loop
    # ------------------------------------------------------------------
    def run_stage(self, stage: StageConfig, *, deadline: float | None = None) -> dict[str, Any]:
        if not stage.enabled:
            return {"stage": stage.name, "skipped": True}
        self.build_data()
        # Source admission comes from the config, not from the stage's name.
        # ``strict=False`` keeps 001-beta's five stage names working unchanged;
        # any other name must declare `extra.curriculum` or this raises.
        admission = stage_admission(stage, cards_dir=self.cfg.mixture_cards, strict=False)
        self._admissions[stage.name] = admission.as_dict()
        print(
            f"[curriculum] {stage.name}: tiers {list(admission.admits)} "
            f"sources {list(admission.source_ids)} "
            f"absent_tiers {list(admission.absent_tiers)} "
            f"provenance={admission.provenance}",
            flush=True,
        )
        specs = self.stage_sources(stage, admission)
        params = list(self.model.parameters()) + list(self.posterior.parameters())
        modules: dict[str, nn.Module] = {"model": self.model, "posterior": self.posterior}
        # The pulse's own parameters. Registered under `tms_drive` so a card has
        # a name to grant and `test_card_patterns_reach_the_model` can see them:
        # a drive that cannot receive a gradient is a stimulus the model is not
        # allowed to learn the effect of, and the loss would still fall.
        if self.tms_drive is not None:
            params = params + list(self.tms_drive.parameters())
            modules["tms_drive"] = self.tms_drive
        if admission.individualize:
            if self.individualizer is None:
                n_p = max(len(self._participant_ids()), 1)
                self.individualizer = Individualizer(
                    len(THETA_NAMES), n_groups=2, n_participants=max(n_p, 1), n_sessions=max(n_p * 4, 1)
                ).to(self.device)
            params = list(self.individualizer.parameters()) + [
                p for n, p in self.model.named_parameters() if n.startswith("eeg.")
            ]
            modules["individualizer"] = self.individualizer

        gate_model = _CombinedModule(modules)
        mixture = MixtureTrainer(gate_model, specs)
        # The normalizing flow needs a lower LR than the rest of the model.
        #
        # Measured on run 2: with a single LR the posterior is stable at
        # -log q ~ 8.2 through step 80 and then climbs (110 at step 100, seen_max
        # 34 -> 848) at exactly the point OneCycle's ramp reaches max_lr.  The
        # forecast head is unaffected (sim_forecast_nll keeps falling), so this is
        # the flow's own conditioning-sensitivity, not a global instability: a
        # coupling layer's translation is linear in its input far from the origin,
        # so a step that is merely large for a residual block is destabilising for
        # a density.
        #
        # A separate group at POSTERIOR_LR_SCALE keeps one schedule shape while
        # letting the two parts move at their own rates.
        post_ids = {id(q) for q in self.posterior.parameters()}
        model_params = [q for q in params if id(q) not in post_ids]
        post_params = [q for q in params if id(q) in post_ids]
        groups = [{"params": model_params, "lr": stage.lr}]
        if post_params:
            groups.append(
                {
                    "params": post_params,
                    "lr": stage.lr * self.POSTERIOR_LR_SCALE,
                    "weight_decay": self.POSTERIOR_WEIGHT_DECAY,
                }
            )
        opt = torch.optim.AdamW(groups, lr=stage.lr, weight_decay=stage.weight_decay, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[g["lr"] for g in groups],
            total_steps=max(stage.steps, 2),
            pct_start=min(0.3, stage.warmup / max(stage.steps, 1)),
        )
        self.model.train()
        self.posterior.train()
        t0 = time.time()
        best = math.inf
        step = 0
        for step in range(1, stage.steps + 1):
            if deadline is not None and time.time() > deadline:
                print(f"[{stage.name}] wall-clock deadline reached at step {step}", flush=True)
                break
            losses: dict[str, Tensor] = {}
            diag: dict[str, Any] = {}
            if admission.admits_simulated(self.sources):
                sl, sd = self.sim_losses(next(self.sim_loader), stage, admission)
                losses.update(sl)
                diag.update(sd)
            losses.update(self.anat_losses(admission))
            if self.real_train is not None and admission.admits_measured(self.sources):
                try:
                    rl, rd = self.real_losses(next(self.real_loader), stage)
                    losses.update(rl)
                    diag.update(rd)
                except StopIteration:  # pragma: no cover
                    pass
            # A second measured modality, in its own support.  Gated on the same
            # admission the EEG term is, and on the loader existing -- a stage
            # may admit a BOLD source in the config before the corpus is cached,
            # and that must be a no-op rather than a crash or a silent zero.
            bold_loader = getattr(self, "bold_loader", None)
            if bold_loader is not None and admission.admits_measured(self.sources):
                bold_ids = [
                    sid
                    for sid in admission.source_ids
                    if getattr(self.sources.get(sid), "modality", "") == "bold"
                    or sid.startswith("ds002336")
                ]
                if bold_ids:
                    try:
                        bl, bd = self.real_bold_losses(
                            next(bold_loader), stage, source_id=bold_ids[0]
                        )
                        losses.update(bl)
                        diag.update(bd)
                    except StopIteration:  # pragma: no cover
                        pass
            # -- the sources added for run 3 -----------------------------
            # Each is gated on BOTH the stage admitting it and its loader
            # existing. A source admitted with no loader contributes nothing and
            # is recorded in `self._absent_admitted`, because the failure this
            # whole run exists to avoid is a source that reads as trained in the
            # report and never produced a gradient.
            if admission.admits_measured(self.sources):
                for sid, loader in getattr(self, "eeg_loaders", {}).items():
                    if sid == "eegmmidb_real" or sid not in admission.source_ids:
                        continue
                    try:
                        rl, rd = self.real_losses(next(loader), stage, source_id=sid)
                        losses.update(rl)
                        diag.update(rd)
                        self._contributed.add(sid)
                    except StopIteration:  # pragma: no cover
                        pass
                for sid, loader in getattr(self, "bold_loaders", {}).items():
                    if sid == "ds002336_real" or sid not in admission.source_ids:
                        continue
                    try:
                        bl, bd = self.real_bold_losses(next(loader), stage, source_id=sid)
                        losses.update(bl)
                        diag.update(bd)
                        self._contributed.add(sid)
                    except StopIteration:  # pragma: no cover
                        pass
                bl_ = getattr(self, "behaviour_loader", None)
                if bl_ is not None and "ds000117_behaviour" in admission.source_ids:
                    try:
                        hl, hd = self.behaviour_losses(next(bl_), stage)
                        losses.update(hl)
                        diag.update(hd)
                        self._contributed.add("ds000117_behaviour")
                    except StopIteration:  # pragma: no cover
                        pass
                pl_ = getattr(self, "perturb_loader", None)
                if pl_ is not None and "ds004024_perturb" in admission.source_ids:
                    try:
                        ql, qd = self.perturb_losses(next(pl_), stage)
                        losses.update(ql)
                        diag.update(qd)
                        self._contributed.add("ds004024_perturb")
                    except StopIteration:  # pragma: no cover
                        pass
            if step == 1:
                # An admitted source with no loss term is the run-2 failure in a
                # different costume: the card says the source trains the model
                # and nothing does. Named at step 1, once, and kept.
                produced = set(losses) | self._contributed
                if REAL_LOSS_KEY in losses:
                    produced.add("eegmmidb_real")
                missing = sorted(set(admission.source_ids) - produced - self._never_a_term)
                if missing:
                    print(
                        f"[curriculum] {stage.name}: admitted but produced no term: "
                        f"{missing}. Recorded, not silently dropped.",
                        flush=True,
                    )
                    self._absent_admitted[stage.name] = missing
            if not losses:
                raise RuntimeError(f"stage {stage.name} produced no admissible loss; check the source cards")
            mdiag = mixture.step(losses, measure_conflict=(step % 10 == 0))
            torch.nn.utils.clip_grad_norm_(params, stage.grad_clip)
            opt.step()
            sched.step()
            self.global_step += 1
            if self.individualizer is not None:
                self.individualizer.assert_centered()
            total = float(mdiag["mixture_total"])
            best = min(best, total)
            if step % stage.log_every == 0 or step == 1:
                rec = {
                    "stage": stage.name,
                    "step": step,
                    "global_step": self.global_step,
                    "loss": total,
                    "lr": sched.get_last_lr()[0],
                    "wall_s": round(time.time() - t0, 1),
                    "traj_s_per_s": round(
                        step * self.cfg.data.batch * self.cfg.data.window * self.cfg.model.dt_model
                        / max(time.time() - t0, 1e-9),
                        1,
                    ),
                    # Reserved, not allocated: the caching allocator's footprint
                    # is the machine's actual exposure, and it is the number the
                    # cgroup cannot see (reports/training/platform_memory_limits.md).
                    "gpu_reserved_gb": round(cuda_reserved_gb(self.device), 2),
                    **{k: v for k, v in diag.items() if isinstance(v, (int, float))},
                }
                self.logger.log(**rec)
                self.history.append(rec)
            if step % stage.ckpt_every == 0:
                self._save("last.pt", stage.name, step, metrics={"loss": total})
        wall = time.time() - t0
        # Record completion BEFORE writing the checkpoints, so the artifacts say
        # the stage is done.  Appending afterwards (as this did) means every
        # stage-end checkpoint records the stage as *incomplete*, and any resume
        # replays a stage that had already run to its final step -- with a fresh
        # OneCycle schedule, silently re-training it.  Observed: Stage II ran
        # 700/700, wrote stage_II_interface.pt, and the resume still reported
        # ``completed stages: ['I_regional']``.
        #
        # The mid-stage saves inside the loop above correctly exclude the current
        # stage: at that point it genuinely is incomplete.
        self.completed_stages.append(stage.name)
        self._save("last.pt", stage.name, step, metrics={"loss": best})
        self._save(f"stage_{stage.name}.pt", stage.name, step, metrics={"loss": best})
        rep = mixture.report()
        rep["stage"] = stage.name
        rep["run_name"] = self.cfg.train.run_name
        self._mixture_reports.append(rep)
        # Scoped by RUN as well as stage. Keyed on the stage name alone, run 3
        # silently overwrites run 2: both declare stages called
        # `T1_measured_founding` and `T3_population_prior`, and run 2's reports
        # are the published evidence for an artifact that is already cited.
        # Same class as `--out moves checkpoints, not logs`, one directory over.
        #
        # The legacy flat path is still written so nothing that reads it breaks,
        # but the run-scoped copy is the one that cannot collide.
        run_dir = self.report_dir / self.cfg.train.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(rep, indent=2, default=float)
        (run_dir / f"mixture_{stage.name}.json").write_text(payload)
        (self.report_dir / f"mixture_{stage.name}.json").write_text(payload)
        return {
            "stage": stage.name,
            "steps": step,
            "wall_seconds": wall,
            "best_loss": best,
            "steps_per_second": step / max(wall, 1e-9),
            "mixture": rep,
        }

    def _participant_ids(self) -> list[str]:
        """Stable, sorted participant ids -- the grouping unit for R10 and Stage V."""
        ds = getattr(self, "real_dataset", None)
        if ds is None:
            return []
        for attr in ("subjects", "participant_ids", "subject_ids"):
            v = getattr(ds, attr, None)
            if v:
                return sorted(set(map(str, v)))
        try:
            return sorted({str(ds.provenance(i).subject) for i in range(len(ds))})
        except Exception:  # noqa: BLE001 - fall back to sampling the items
            step = max(1, len(ds) // 500)
            return sorted({str(ds[i]["subject"]) for i in range(0, len(ds), step)})

    def real_split_fingerprint(self) -> dict[str, Any] | None:
        """Participant **ids** per fold plus a sha256, recorded in every checkpoint.

        Ids rather than indices: indices are meaningless if the corpus is rebuilt,
        so an index-based fingerprint would pass silently on a different dataset.
        Evaluation recomputes this and refuses to score on a mismatch.
        """
        ds = getattr(self, "real_dataset", None)
        split = getattr(self, "real_split", None)
        if ds is None or not split:
            return None
        import hashlib

        def _sub(i: int) -> str:
            rec_idx, _ = ds.window_index[int(i)]
            return str(ds.recordings[rec_idx]["subject"])

        folds = {k: sorted({_sub(i) for i in v}) for k, v in split.items()}
        blob = json.dumps(folds, sort_keys=True).encode()
        return {"participants_per_fold": folds, "sha256": hashlib.sha256(blob).hexdigest()}

    def participant_index(self, subjects: Sequence[str]) -> Tensor:
        """Map subject ids to Individualizer rows; unknown ids map to row 0.

        Unknown-at-fit-time participants are *not* silently given someone else's
        effect: row 0 is the population row (``delta = 0`` at initialisation) and
        the fact is recorded in ``self.unknown_participants``.
        """
        if not hasattr(self, "_pidx"):
            self._pidx = {s: i for i, s in enumerate(self._participant_ids())}
            self.unknown_participants: set[str] = set()
        idx = []
        for s in subjects:
            s = str(s)
            if s not in self._pidx:
                self.unknown_participants.add(s)
            idx.append(self._pidx.get(s, 0))
        return torch.tensor(idx or [0], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    def _save(self, name: str, stage: str, step: int, *, metrics: Mapping[str, Any] | None = None) -> Path:
        return save_checkpoint(
            self.out_dir / name,
            model=self.model,
            config=self.cfg,
            step=self.global_step,
            stage=stage,
            posterior=self.posterior,
            individualizer=self.individualizer,
            tms_drive=self.tms_drive,
            metrics={**(metrics or {}), "stage_step": step, "completed_stages": self.completed_stages},
            extra={
                "anatomy": self.anat.summary(),
                "lead_field": self.model.eeg.lead_field_meta,
                "sensor_to_parcel": self.sensor_to_parcel.summary(),
                "real_split": self.real_split_fingerprint(),
                "theta_names": list(THETA_NAMES),
                "theta_prior": self.theta_prior.as_dict(),
                "parameter_report": self.model.parameter_report(),
                "posterior_parameters": count_parameters(self.posterior),
                # Derived from the weights in this very file, not asserted. The
                # published card reads `contributed_sources` and `moved` off the
                # checkpoint rather than off the config that hoped for them.
                "moved_since_init": self.moved_since_init(),
                "contributed_sources": sorted(self._contributed),
                "admitted_but_no_term": {k: v for k, v in self._absent_admitted.items() if v},
                "tms_drive": self.tms_drive.summary() if self.tms_drive is not None else None,
            },
        )

    def maybe_resume(self) -> None:
        if not self._resume:
            return
        p = self.out_dir / "last.pt"
        if not p.exists():
            return
        # `build_data` constructs `tms_drive`, so resume must happen after it or
        # the drive is None here and its weights are silently not restored --
        # the training would resume with a re-initialised pulse and nothing
        # would say so. `load_checkpoint` records `tms_drive_absent` either way.
        self.build_data()
        payload = load_checkpoint(
            p,
            model=self.model,
            posterior=self.posterior,
            tms_drive=self.tms_drive,
            map_location=str(self.device),
            strict=False,
        )
        self.global_step = int(payload.get("step", 0))
        self.completed_stages = list(payload.get("metrics", {}).get("completed_stages", []))
        print(f"resumed from {p} at step {self.global_step}; completed stages: {self.completed_stages}", flush=True)

    def _corpus_sha(self) -> str | None:
        """The simulated corpus index's own sha -- provenance that discriminates.

        `git_sha()` is `-dirty` for every checkpoint this project has produced,
        so it cannot distinguish two runs. The corpus index records the sha of
        the code that generated it and changes when the corpus changes.
        """
        try:
            with open(self.cfg.data.sim_index_fast) as fh:
                return json.load(fh).get("git_sha")
        except Exception:  # noqa: BLE001 - provenance is recorded or absent, never fabricated
            return None

    @torch.no_grad()
    def calibrate_noise_floor(self, *, max_batches: int = 8, tag: str = "") -> dict[str, Any]:
        """Set ``eeg.log_noise`` to its closed form on held-out measured windows.

        **This is the repair for the whole of run 1's FAIL**
        (`reports/training/p0_variance_channel.md`). `eeg.log_noise` was
        trainable in stage V only; stage V ran 900 steps at lr 5.77e-5 for 134
        seconds, and the parameter travelled 20% of the way to an optimum that
        has a closed form. The model shipped asserting predictive variance 1.31
        against a realised 3.97 -- overconfident by 3.0x -- which cost +0.4467
        nats, 1.62x its entire deficit to persistence.

        The Gaussian NLL is stationary in ``log_noise`` at ``log(mean residual
        variance)``. It is computed here instead of searched for.

        Run on the **val** fold: the mean must not be fitted on the windows the
        variance is calibrated on, or the residuals are optimistic and the floor
        is set too low -- which is the direction that caused the original defect.
        """
        val = getattr(self, "real_val", None)
        if val is None or len(val) == 0 or not hasattr(self.model.eeg, "calibrate_noise_floor"):
            return {
                "applied": False,
                "reason": "no held-out measured val fold, or head predates the repair",
            }
        loader = torch.utils.data.DataLoader(
            val, batch_size=max(8, self.cfg.data.batch // 4), shuffle=False, num_workers=0
        )
        was_training = self.model.training
        self.model.eval()
        c = self.cfg.data.context
        resid: list[Tensor] = []
        states: list[Tensor] = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            eeg = batch["eeg"].to(self.device, non_blocking=True)
            ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
            src = self.sensor_to_parcel(ctx_e)
            src = src / src.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            th = self.posterior.sample(ctx_e, 1)[:, 0][:, : len(THETA_NAMES)]
            if self.individualizer is not None:
                th = self.individualizer(participant=self.participant_index(batch.get("subject", [])), base=th)
            self.model.set_mechanistic_theta(th, self.anat)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
                roll = self.model.rollout(y_context=src, theta=th, n_steps=tgt_e.shape[1], enforce_r05=False)
                mu, _ = self.model.eeg(roll.state)
            resid.append((tgt_e.float() - mu.float()).cpu())
            states.append(roll.state.float().cpu())
        if was_training:
            self.model.train()
        if not resid:
            return {"applied": False, "reason": "held-out loader yielded no batches"}
        r = torch.cat(resid)
        s = torch.cat(states)
        rep = self.model.eeg.calibrate_noise_floor(r.to(self.device), state=s.to(self.device))
        rep["applied"] = True
        rep["tag"] = tag
        rep["realised_residual_variance"] = float(r.pow(2).mean())
        print(
            f"[noise-floor {tag}] log_noise {rep['log_noise_before_mean']:+.4f} -> "
            f"{rep['log_noise_after_mean']:+.4f}  (variance {rep['implied_variance_before']:.4f} -> "
            f"{rep['implied_variance_after']:.4f}, realised {rep['realised_residual_variance']:.4f})",
            flush=True,
        )
        return rep

    def train(self) -> dict[str, Any]:
        # The init calibration needs the val fold. `main()` builds data first,
        # but `train()` is public and was reachable without it -- in which case
        # the init call silently returned `applied: False` and the run lost the
        # one calibration it most needed. Idempotent; `build_data` guards itself.
        if not getattr(self, "_data_ready", False):
            self.build_data()
        self.maybe_resume()
        deadline = time.time() + self.cfg.train.max_wall_seconds
        results = []
        # Closed form BEFORE the first step, so the head does not spend the run
        # climbing toward a number it could have been handed.
        noise_floor_calibrations = [self.calibrate_noise_floor(tag="init")]
        for stage in self.cfg.train.stages:
            if stage.name in self.completed_stages:
                print(f"skipping completed stage {stage.name}", flush=True)
                continue
            if time.time() > deadline:
                print("global wall-clock budget exhausted", flush=True)
                break
            print(f"\n=== Stage {stage.name}: {stage.steps} steps, lr={stage.lr} ===", flush=True)
            results.append(self.run_stage(stage, deadline=deadline))
            # Re-solve at every stage boundary. The optimum MOVES: it is
            # log(residual variance), and the residual shrinks as the mean path
            # improves, so a floor calibrated once at init is stale by the end.
            noise_floor_calibrations.append(self.calibrate_noise_floor(tag=f"after:{stage.name}"))
        summary = {
            "run_name": self.cfg.train.run_name,
            # `git_sha()` carries `-dirty` on every checkpoint this project has
            # produced, because the run writes to tracked files. It distinguishes
            # nothing (reports/decorative_guards.md #4) and is retained only so
            # its uselessness stays visible. Provenance that DOES discriminate:
            # the corpus index sha and the split fingerprint, below.
            "git_sha": git_sha(),
            "corpus_git_sha": self._corpus_sha(),
            "environment": env_fingerprint(),
            "stages": results,
            "noise_floor_calibrations": noise_floor_calibrations,
            # Whether the heads were ever fitted, stated rather than inferable.
            "noise_floor_report": {
                "eeg": self.model.eeg.noise_floor_report()
                if hasattr(self.model.eeg, "noise_floor_report")
                else {},
                "bold": self.model.bold.noise_floor_report()
                if hasattr(self.model.bold, "noise_floor_report")
                else {},
            },
            "total_wall_seconds": self.timer.elapsed,
            "global_steps": self.global_step,
            "model_parameters": self.model.parameter_report(),
            "posterior_parameters": count_parameters(self.posterior),
            "sources": {k: v.as_dict() for k, v in self.sources.items()},
            "compiler_bridge": self.compiler_report,
        }
        (self.report_dir / f"{self.cfg.train.run_name}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return summary


class _CombinedModule(nn.Module):
    """Presents several modules as one so ``GradientGate`` can name them all."""

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        super().__init__()
        for k, v in modules.items():
            self.add_module(k, v)

    def named_parameters(self, *a, **kw):  # type: ignore[override]
        for n, p in super().named_parameters(*a, **kw):
            # strip the container prefix so patterns match the model's own names
            yield (n.split(".", 1)[1] if n.startswith("model.") else n), p


def _glob_overlap(a: str, b: str) -> bool:
    """Cheap test for whether two glob patterns can match a common name."""
    import fnmatch

    return fnmatch.fnmatch(a.rstrip("*").rstrip("."), b.rstrip("*").rstrip(".")) or fnmatch.fnmatch(
        b.rstrip("*").rstrip("."), a.rstrip("*").rstrip(".")
    ) or a.split(".")[0] == b.split(".")[0]


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:  # pragma: no cover - entry point
    import argparse

    p = argparse.ArgumentParser(description="train SC-WBD-001-beta")
    p.add_argument("--config", default="configs/scwbd_001_beta.yaml")
    p.add_argument("--quick", action="store_true", help="CI-sized smoke run")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="build, run ONE batch fwd+bwd through both loss paths, report the noise floor, "
             "and exit non-zero on any raise. A launch precondition -- run it on BOTH arms.",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-wall", type=float, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.max_wall is not None:
        cfg.train.max_wall_seconds = a.max_wall
    if a.out:
        cfg.train.out_dir = a.out
    t = FoundationTrainer(cfg, resume=not a.no_resume, quick=a.quick)
    print(json.dumps({"parameters": t.model.parameter_report(), "posterior": count_parameters(t.posterior)}, indent=2))
    if a.smoke:
        import sys
        import traceback

        try:
            return t.smoke()
        except Exception:  # noqa: BLE001 - the whole point is a non-zero exit
            traceback.print_exc()
            print("\n=== SMOKE FAILED -- do not launch ===", flush=True)
            sys.exit(1)
    return t.train()


if __name__ == "__main__":  # pragma: no cover
    main()
