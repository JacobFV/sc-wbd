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
            self.model.local = torch.compile(self.model.local, dynamic=False)
            self.model.residual = torch.compile(self.model.residual, dynamic=False)

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
        self.flop_per_step = 0.0

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

        self._data_ready = True

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

    def real_losses(self, batch: Mapping[str, Any], stage: StageConfig) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Measured EEG: likelihood in **sensor space**, always."""
        eeg = batch["eeg"].to(self.device, non_blocking=True)  # (B,T,C)
        B, T, C = eeg.shape
        c = self.cfg.data.context
        ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
        n_pred = tgt_e.shape[1]
        with torch.no_grad():
            src_ctx = self.sensor_to_parcel(ctx_e)
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
            mu, lv = self.model.eeg(roll.state)
        mu, lv = mu.float(), lv.float()
        scale = tgt_e.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        nll = gaussian_nll(tgt_e / scale, mu / scale.clamp_min(1e-8), lv - 2 * torch.log(scale))
        losses = {"eegmmidb_real": nll}

        if self.individualizer is not None:
            losses["eegmmidb_real"] = losses["eegmmidb_real"] + 1e-3 * self.individualizer.prior_penalty()
        return losses, {"real_eeg_nll": float(nll.detach())}

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
            src_ctx = torch.where(m3, (src_ctx - mean) / var.sqrt().clamp_min(1e-6), torch.zeros_like(src_ctx))

        # theta from the same amortised posterior the EEG path uses. Its
        # conditioning was fitted on EEG context, so on BOLD it is out of
        # distribution -- recorded rather than hidden, because a posterior asked
        # for theta from an input it never saw is a real caveat and not a
        # detail. Ordering the posterior separately per modality is the fix and
        # is not attempted here.
        th = self.posterior.sample(ctx_b.mean(-1), 1)[:, 0][:, : len(THETA_NAMES)].detach()

        self.model.set_mechanistic_theta(th, self.anat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(y_context=src_ctx, theta=th, n_steps=n_pred, enforce_r05=False)
            hemo = (
                self.model.family_layout.component(roll.state, "hemo")
                if self.model.family_layout is not None
                else roll.state
            )
            mu, lv = self.model.bold.signal(hemo, roll.state)
        mu, lv = mu.float(), lv.float()

        # The mask is the whole point: gaussian_nll marginalises unobserved
        # elements out rather than scoring them, so uncovered parcels contribute
        # nothing and do not enter the denominator either.
        m3t = mask.unsqueeze(1).expand_as(tgt_b).to(mu.dtype)
        nll = gaussian_nll(torch.nan_to_num(tgt_b, nan=0.0), mu, lv, mask=m3t)

        return {source_id: nll}, {
            "real_bold_nll": float(nll.detach()),
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
        self._mixture_reports.append(rep)
        (self.report_dir / f"mixture_{stage.name}.json").write_text(json.dumps(rep, indent=2, default=float))
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
            },
        )

    def maybe_resume(self) -> None:
        if not self._resume:
            return
        p = self.out_dir / "last.pt"
        if not p.exists():
            return
        payload = load_checkpoint(
            p, model=self.model, posterior=self.posterior, map_location=str(self.device), strict=False
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
    return t.train()


if __name__ == "__main__":  # pragma: no cover
    main()
