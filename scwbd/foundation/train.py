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
from .simulate import THETA_NAMES, CorpusSpec, SimCorpus, ThetaPrior, generate_corpus
from .util import JsonlLogger, Timer, count_parameters, env_fingerprint, git_sha, set_determinism

__all__ = ["FoundationTrainer", "STAGE_PERMISSIONS", "main"]

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
        set_determinism(cfg.train.seed)
        torch.backends.cuda.matmul.allow_tf32 = False  # solvers stay fp32 (ARCH §3)
        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(cfg.train.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.anat = (anat or load_anatomy(device=self.device, n_cortex=400)).to(self.device)
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
                    k: v for k, v in audit.items() if k in ("unclaimed_parameters", "empty_bindings", "unbound_groups", "declared_empty_groups")
                },
                "per_source": changed,
                "outside_compiler_schema": outside,
            }
        except Exception as exc:  # noqa: BLE001 - the compiler is authoritative but not yet mandatory
            rep = {"used": False, "reason": f"{type(exc).__name__}: {exc}"}
            print(f"[warn] compiler bridge unavailable ({rep['reason']}); using the source cards' own "
                  "torch-level gradient permissions", flush=True)
        return rep

    def stage_sources(self, stage: StageConfig) -> dict[str, SourceSpec]:
        """Intersect each card's ``A_k`` with the stage allowlist (restrict only)."""
        allow = STAGE_PERMISSIONS.get(stage.name, ("*",))
        out: dict[str, SourceSpec] = {}
        for sid, s in self.sources.items():
            if not s.enabled:
                continue
            if allow == ("*",):
                perm = s.gradient_permission
            else:
                perm = tuple(p for p in s.gradient_permission if p == "*" or any(_glob_overlap(p, a) for a in allow))
                if "*" in s.gradient_permission:
                    perm = allow
            out[sid] = SourceSpec(**{**s.as_dict(), "gradient_permission": perm})
        return out

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
                pin_memory=True,
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

    def sim_losses(self, batch: Mapping[str, Any], stage: StageConfig) -> tuple[dict[str, Tensor], dict[str, Any]]:
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
        if stage.name == "I_regional":
            with torch.no_grad():
                amp = torch.rand(B, 1, 1, 1, device=self.device) * 0.4
            u = amp * torch.randn(B, n_pred, N, self.model.layout.dim, device=self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.use_bf16):
            roll = self.model.rollout(
                y_context=ctx_y,
                theta=theta,
                n_steps=n_pred,
                context_mask=ctx_mask,
                u=u,
                with_hemo=stage.name in ("IV_assembly",),
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
            kl=float(kl.detach()),
            rho=float(roll.rho.detach()),
            observed_fraction=float(obs_mask.mean()),
            **{k: v for k, v in roll.diagnostics.items() if isinstance(v, (int, float))},
        )
        if "anatomical_prior" in self.sources:
            losses["anatomical_prior"] = self.model.coupling.topology_penalty() + self.model.bold.prior_penalty()
        return losses, diag

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

    # ------------------------------------------------------------------
    # stage loop
    # ------------------------------------------------------------------
    def run_stage(self, stage: StageConfig, *, deadline: float | None = None) -> dict[str, Any]:
        if not stage.enabled:
            return {"stage": stage.name, "skipped": True}
        self.build_data()
        specs = self.stage_sources(stage)
        params = list(self.model.parameters()) + list(self.posterior.parameters())
        modules: dict[str, nn.Module] = {"model": self.model, "posterior": self.posterior}
        if stage.name == "V_individual":
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
        opt = torch.optim.AdamW(params, lr=stage.lr, weight_decay=stage.weight_decay, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=stage.lr, total_steps=max(stage.steps, 2), pct_start=min(0.3, stage.warmup / max(stage.steps, 1))
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
            if stage.name != "V_individual":
                sl, sd = self.sim_losses(next(self.sim_loader), stage)
                losses.update(sl)
                diag.update(sd)
            if self.real_train is not None and stage.name in ("III_sliced", "IV_assembly", "V_individual"):
                try:
                    rl, rd = self.real_losses(next(self.real_loader), stage)
                    losses.update(rl)
                    diag.update(rd)
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
                    **{k: v for k, v in diag.items() if isinstance(v, (int, float))},
                }
                self.logger.log(**rec)
                self.history.append(rec)
            if step % stage.ckpt_every == 0:
                self._save("last.pt", stage.name, step, metrics={"loss": total})
        wall = time.time() - t0
        self._save("last.pt", stage.name, step, metrics={"loss": best})
        self._save(f"stage_{stage.name}.pt", stage.name, step, metrics={"loss": best})
        rep = mixture.report()
        rep["stage"] = stage.name
        self._mixture_reports.append(rep)
        (self.report_dir / f"mixture_{stage.name}.json").write_text(json.dumps(rep, indent=2, default=float))
        self.completed_stages.append(stage.name)
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

    def train(self) -> dict[str, Any]:
        self.maybe_resume()
        deadline = time.time() + self.cfg.train.max_wall_seconds
        results = []
        for stage in self.cfg.train.stages:
            if stage.name in self.completed_stages:
                print(f"skipping completed stage {stage.name}", flush=True)
                continue
            if time.time() > deadline:
                print("global wall-clock budget exhausted", flush=True)
                break
            print(f"\n=== Stage {stage.name}: {stage.steps} steps, lr={stage.lr} ===", flush=True)
            results.append(self.run_stage(stage, deadline=deadline))
        summary = {
            "run_name": self.cfg.train.run_name,
            "git_sha": git_sha(),
            "environment": env_fingerprint(),
            "stages": results,
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
