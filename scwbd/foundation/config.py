"""Configuration objects for SC-WBD-001-beta.

Plain dataclasses + YAML.  A config is part of the artifact: the exact config
that produced a checkpoint is written next to the weights together with the git
SHA (ARCHITECTURE.md §5, deliverable 4).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ArmConfig",
    "ModelConfig",
    "PosteriorConfig",
    "DataConfig",
    "StageConfig",
    "TrainConfig",
    "FoundationConfig",
    "load_config",
]


@dataclass
class ArmConfig:
    """Which arm of a comparison this run is (refusal R12).

    ``role="model"`` is the default and the fail-closed direction: a run that
    says nothing claims to be the model and is held to the model's structure.
    A control run must say ``role="control"``, name the comparison, and write
    down why -- and its checkpoints are then emitted under a *control*
    designation instead of the model's.  See
    :mod:`scwbd.schema.designation` and ``reports/scope_gap.md``.

    This is *declared intent*, and it is deliberately separate from
    ``FoundationConfig.ablation_arm()`` / ``SCWBD.family_report()``, which are
    *derived structure*.  R12 is the rule that the two must agree, or the
    artifact does not get the model's name.  Run 1 had the structure and no
    declaration, so nothing could notice they disagreed.
    """

    role: str = "model"
    controls_for: str = ""
    justification: str = ""


@dataclass
class ModelConfig:
    n_regions: int = 454
    #: which operator supplies F_local: "learned" or a mechanistic backend name.
    #:
    #: **One string for the whole brain.**  Kept, and kept working, because it is
    #: the equal-capacity generic-operator arm of body.tex §11.4's first required
    #: ablation ("structured regional state versus one scalar or pooled vector
    #: per region").  It is a *control*, not the model: see ``family_state``.
    local_core: str = "learned"
    #: Region-indexed state (body.tex §2.1, ARCHITECTURE.md §5).  When True the
    #: model partitions regions into families derived from the anatomy prior,
    #: each with its own component list, dimension, backend and ports.
    #:
    #: ``False`` is the §11.4 **control** arm and must be declared as such: R12
    #: refuses to emit a checkpoint that claims heterogeneous regional state
    #: while this is off.
    family_state: bool = False
    #: family name -> backend name.  Families not listed take
    #: ``families.DEFAULT_FAMILY_CORES`` for their kind, and cortical families
    #: fall through to ``local_core``.  Assigning a backend to a family the
    #: anatomy prior does not produce raises rather than silently doing nothing.
    family_cores: dict[str, str] = field(default_factory=dict)
    #: Permit ``scwbd.foundation.families`` to INFER a family partition when the
    #: anatomy prior declares none.
    #:
    #: Default ``False``, and it must stay ``False`` for any training run. The
    #: inferred cortical split is Yeo-7, which agent C's Vasa spin null rejects
    #: as a partition (6 of 21 pairs separate); the evidence supports
    #: unimodal/association and nothing finer. Setting this ``True`` makes "this
    #: run's operator assignment rests on a split the evidence rejects" a
    #: declaration in the config rather than a silent default.
    family_allow_derived_partition: bool = False
    #: hippocampal H_t = {k,v,g,c,rho} widths (body.tex §5.1).  These set the
    #: padded dimension D for the whole state, so they are the price of `padded-family-state` --
    #: ``FamilyStateLayout.padding_fraction()`` reports it.
    d_key: int = 16
    d_value: int = 16
    d_grid: int = 12
    d_context: int = 4
    #: cerebellar forward-model / error / eligibility width
    d_prediction: int = 8
    #: Drive ``X_i^uncertainty`` (body.tex §2.1) with an integrated innovation /
    #: decay law and expose it to the observation heads as a **state-dependent**
    #: predictive log-variance.
    #:
    #: ``False`` restores the run-1 behaviour: ``SCWBD.observation is None`` and
    #: ``heads.py`` falls back to its broadcast ``log_noise`` parameter — the
    #: variance that is constant in state, time, horizon, window, participant and
    #: condition, and that cost run 1 its NLL.  Retained as a switch **only** so
    #: the repair itself can be ablated; it is not a supported configuration.
    #:
    #: Applies to BOTH §11.4 arms on purpose.  Giving the treatment arm a
    #: state-dependent variance and leaving the control arm on a broadcast
    #: constant would make A1 measure the variance path instead of the structured
    #: state -- the same class of error as an interface that silently narrows one
    #: arm.  See reports/dynamics/family_state.md §9.
    state_dependent_variance: bool = True

    #: Declared cross-scale prolongations, written ``"fine<=coarse"``.  Empty
    #: means this run declares no cross-scale prolongation -- the second of the
    #: two conditions R12 refuses on (body.tex §0.2's second differentiator).
    #: Whatever is named here must appear in the compiled resolution poset,
    #: where **R02** refuses a prolongation lacking a restriction partner and
    #: tested coverage: R12 asks whether one was declared, R02 asks whether it
    #: is any good.
    scale_prolongations: list[str] = field(default_factory=list)
    hidden: int = 320
    n_local_layers: int = 3
    region_embed: int = 96
    context_dim: int = 128
    message_dim: int = 12
    n_delay_bins: int = 8
    n_spectral_modes: int = 8
    n_adaptation: int = 2
    n_uncertainty: int = 4
    dt_model: float = 0.008  # s, fast (neural) clock == corpus sampling interval
    hemo_ratio: int = 25  # slow (hemodynamic) clock = 25 * dt_model = 200 ms
    #: BOLD frames the measured path predicts per window. ISSUE-008's fix rolls
    #: the neural clock for the duration a frame actually covers, so the cost is
    #: ``frames * (TR / dt_model)`` steps -- 250 per frame at TR = 2 s against
    #: run 3's 8 for the whole window. 2 frames (4 s, 500 steps) is what fits the
    #: step budget; it is a REDUCTION IN WHAT IS PREDICTED, not a free choice,
    #: and it belongs in the model card next to any fMRI number.
    bold_predict_frames: int = 2
    #: Duty cycle for the measured BOLD term, in optimiser steps. A slow modality
    #: may attach on a slow schedule -- that is multirate in the training loop as
    #: well as in the model -- but it means the term sees 1/N of the data a
    #: fast source does, which is a statement about evidence, not just cost.
    bold_every: int = 1
    #: R05 guard: max admissible ||R_theta|| / ||F_local + F_long||
    residual_rho_max: float = 0.35
    residual_init_scale: float = 0.05
    encoder_channels: int = 96
    encoder_layers: int = 3
    dropout: float = 0.0
    #: G2 control: which connectome the coupling is masked by
    control_graph: str = "none"
    #: ablation: collapse the structured state to one scalar per region
    scalar_state_ablation: bool = False
    #: ablation: remove the connectome mask (dense coupling at matched capacity)
    dense_coupling_ablation: bool = False
    n_eeg_channels: int = 64
    n_behaviour: int = 4
    #: Observation montages beyond the founding 64-channel one, keyed by source
    #: id.  Each value is ``{"kind": "monopolar"|"bipolar", "channels": [...]}``.
    #:
    #: A source whose electrodes are not the eegmmidb montage needs its own
    #: operator, not a padded row in someone else's.  ``sleepedf_real`` carries
    #: two *bipolar* derivations, and forcing them through the 64-channel head
    #: is what its card recorded as the block; ``build_bipolar_lead_field``
    #: derives the correct 2-row operator instead.  Declared here rather than
    #: inferred from the enabled cards, so the model a checkpoint describes is a
    #: function of the config alone.
    montages: dict[str, Any] = field(default_factory=dict)
    use_bf16: bool = True
    compile: bool = False


@dataclass
class PosteriorConfig:
    summary_channels: int = 128
    summary_layers: int = 4
    flow_layers: int = 6
    flow_hidden: int = 256
    n_bands: int = 7
    n_pcs: int = 16
    dropout: float = 0.0
    #: Observation nuisance parameters appended to theta (gain, sensor noise).
    #:
    #: 2 -> 0.  The trainer has never estimated these; it passes
    #: ``torch.zeros(B, nuisance_dim)``, so the flow was asked to put finite
    #: density on a point mass in two of its eight dimensions.  Run 2's pilot
    #: measured that (`configs/run2/pilot-families.yaml`: -log q 5.30 with 0
    #: rejections at 0, no convergence at 2) and set its own file to 0 -- but the
    #: DEFAULT was left at 2, and `configs/run3/scwbd-003.yaml` says "untouched
    #: from the default, deliberately", so run 3 inherited the defect a pilot had
    #: already retired.  Measured on run 3's checkpoint: those two coordinates
    #: collapsed to a sampled sd of 6.0e-4 and took 12.99 of the 15.9 nats by
    #: which npe_loss fell.  Raise this above 0 only together with a code path
    #: that estimates real nuisance values and passes them; `loss` refuses a
    #: constant target column, so a placeholder cannot silently return.
    #: ISSUE-012.
    nuisance_dim: int = 0


@dataclass
class DataConfig:
    sim_index_fast: str = "/data/scwbd/sim_corpus/index_fast.json"
    sim_index_slow: str = "/data/scwbd/sim_corpus/index_slow.json"
    real_eeg_root: str = "/data/scwbd/eegmmidb/1.0.0"
    real_sleep_root: str = "/data/scwbd/sleep-edfx/1.0.0"
    #: Roots for the sources added in run 3. A root that does not exist yields
    #: an empty dataset and a printed reason, never a silent zero-weight term.
    ds000117_root: str = "data/ds000117/1.1.0"
    ds004024_root: str = "data/ds004024/1.0.0"
    ds000113_root: str = "data/ds000113"
    #: Extra parcel-space BOLD corpora beyond ds002336, as ``source_id -> root``.
    bold_roots: dict[str, str] = field(default_factory=dict)
    #: Build the measured-perturbation (TMS-EEG) epochs.
    enable_perturbation: bool = True
    window: int = 64  # model steps
    context: int = 24  # assimilation window, model steps
    fs_hz: float = 125.0
    batch: int = 192
    num_workers: int = 4
    #: Pin the loader's host buffers.  Default **off**: the GB10 has one unified
    #: LPDDR5X pool, so there is no host->device copy for pinning to overlap --
    #: it just makes pages unevictable in the same budget CUDA allocates from.
    #: On a discrete GPU this is worth turning back on.
    pin_memory: bool = False
    val_fraction: float = 0.05
    real_test_fraction: float = 0.25
    seed: int = 20260805


@dataclass
class StageConfig:
    name: str = "stage"
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup: int = 100
    grad_clip: float = 1.0
    batch: int | None = None
    log_every: int = 20
    ckpt_every: int = 500
    enabled: bool = True
    #: loss weights (assembly objective, body.tex §6.4)
    lambda_slice: float = 1.0
    lambda_obs: float = 1.0
    lambda_forecast: float = 1.0
    lambda_perturb: float = 0.0
    lambda_anat: float = 0.05
    lambda_scale: float = 0.05
    lambda_homeo: float = 0.02
    lambda_cal: float = 0.1
    lambda_posterior: float = 1.0
    lambda_kl: float = 0.01
    lambda_residual: float = 0.02
    lambda_port: float = 0.02
    extra: dict[str, Any] = field(default_factory=dict)


def _stage(name: str, **kw: Any) -> StageConfig:
    return StageConfig(name=name, **kw)


@dataclass
class TrainConfig:
    run_name: str = "scwbd-001-beta"
    out_dir: str = "checkpoints/scwbd-001-beta"
    report_dir: str = "reports/training"
    seed: int = 20260805
    device: str = "cuda"
    #: Train on the **synthetic** anatomical prior on purpose.
    #:
    #: Default ``False``: ``load_anatomy`` now *raises* if ``scwbd.anatomy`` is
    #: importable but cannot be adapted, rather than substituting the synthetic
    #: prior silently.  Setting this to ``True`` is the only way to get the
    #: fallback, and it makes "this run carries no biological content" a
    #: **declaration in the config** rather than an accident in a swallowed
    #: exception.
    #:
    #: SC-WBD-001-beta sets it ``True`` because it was *already* trained that
    #: way -- see reports/training/anatomy_provenance.md.  The flag changes no
    #: behaviour for that run; it states what the run was always doing.
    anatomy_force_fallback: bool = False
    amp_dtype: str = "bfloat16"
    #: Hard ceiling, in GB, on what the CUDA caching allocator may reserve.
    #:
    #: This is **not** redundant with ``systemd-run -p MemoryMax``.  On the GB10
    #: the GPU allocates from the same physical pool as the host, but those
    #: allocations are not charged to the systemd cgroup: on 2026-08-06 a run
    #: held 97.9 GB of device memory while its cgroup reported
    #: ``memory.current = 8.17 GB`` against a 40 GB cap that never fired.  The
    #: cgroup bounds host-side allocation only.  Without this ceiling the
    #: caching allocator grows unopposed -- it reserves freed blocks rather than
    #: returning them -- until the machine dies.  ``0`` disables the cap.
    cuda_reserve_gb: float = 40.0
    max_wall_seconds: float = 6 * 3600.0
    resume: bool = True
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            _stage("I_regional", steps=1500, lr=6e-4, lambda_obs=0.0, lambda_posterior=0.0, lambda_anat=0.0),
            _stage("II_interface", steps=1200, lr=4e-4, lambda_posterior=0.0),
            _stage("III_sliced", steps=4000, lr=3e-4),
            _stage("IV_assembly", steps=6000, lr=2e-4),
            _stage("V_individual", steps=1500, lr=1e-4, lambda_posterior=0.3),
        ]
    )


@dataclass
class FoundationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: R12: which arm of §11.4's comparisons this run is.  Default "model".
    arm: ArmConfig = field(default_factory=ArmConfig)
    mixture_cards: str = "configs/source_cards"
    notes: str = ""

    # -- §11.4 arm --------------------------------------------------------
    def ablation_arm(self) -> str:
        """Which arm of body.tex §11.4's first ablation this config *is*.

        ``"control"``   one operator and one state dimension for every parcel —
                        the equal-capacity generic arm.
        ``"treatment"`` heterogeneous region-indexed state with per-family
                        operators.

        This is a property of the config, not of what anyone wrote in a report.
        ``manifest.refuse_r12`` reads it.
        """
        return "treatment" if self.model.family_state else "control"

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.as_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FoundationConfig":
        return _build(cls, d)


def _build(cls, d: Any):
    if not is_dataclass(cls) or not isinstance(d, dict):
        return d
    kw: dict[str, Any] = {}
    ftypes = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in ftypes:
            raise KeyError(f"unknown config key {k!r} for {cls.__name__}; keys: {sorted(ftypes)}")
        f = ftypes[k]
        t = f.type
        if k == "stages" and isinstance(v, list):
            kw[k] = [_build(StageConfig, s) for s in v]
        elif is_dataclass(t) and isinstance(v, dict):
            kw[k] = _build(t, v)
        elif isinstance(t, str) and t in _NAMES and isinstance(v, dict):
            kw[k] = _build(_NAMES[t], v)
        else:
            kw[k] = v
    return cls(**kw)


_NAMES = {
    "ArmConfig": ArmConfig,
    "ModelConfig": ModelConfig,
    "PosteriorConfig": PosteriorConfig,
    "DataConfig": DataConfig,
    "TrainConfig": TrainConfig,
    "StageConfig": StageConfig,
}


def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_base(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config and apply ``base:`` inheritance **transitively**.

    Resolution used to stop after one level: the parent was read raw, so if it
    declared its own ``base`` that key survived the merge and ``from_dict``
    rejected it with ``unknown config key 'base'``. Any two-level hierarchy --
    e.g. a pilot override on top of an arm config on top of a run config --
    was therefore unbuildable, and the failure named the symptom rather than
    the cause.

    Cycles raise instead of recursing until the stack ends.
    """
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ValueError(f"cyclic config inheritance: {chain}")
    payload = yaml.safe_load(path.read_text()) or {}
    base_ref = payload.pop("base", None)
    if base_ref is None:
        return payload
    base = _resolve_base(path.parent / base_ref, (*_seen, path))
    return _deep_update(base, payload)


def load_config(path: str | Path, **overrides: Any) -> FoundationConfig:
    """Load a YAML config, applying ``base:`` inheritance and CLI overrides."""
    p = Path(path)
    payload = _resolve_base(p)
    for dotted, val in overrides.items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return FoundationConfig.from_dict(payload)


def designation(cfg: Any) -> str:
    """The model's own designation, derived once, never a literal.

    This lives in ``config`` rather than in any one consumer because the naming
    class of defect is *several* consumers each spelling the name themselves.
    ``evaluate.py`` hardcoded ``"SC-WBD-001-beta"`` and stamped a run-1 name onto
    run-2 results; ``checkpoint.py`` hardcoded the same literal into the payload
    of every checkpoint the run-2 trainer wrote. Fixing the first by grepping did
    not find the second, because the second was in a different module and the
    same spelling -- and a literal you fix by grepping is a literal you fix only
    in the spelling you thought of.

    The fallback is deliberate and is never a real designation. An **unnamed**
    artifact is a visible defect; a **misnamed** one is not. If this lookup ever
    fails we would rather ship something obviously broken than something quietly
    wrong.
    """
    for obj in (getattr(cfg, "train", None), getattr(cfg, "model", None), cfg):
        for attr in ("model_id", "designation", "run_name"):
            v = getattr(obj, attr, None) if obj is not None else None
            if isinstance(v, str) and v:
                return v
    return "SC-WBD-unnamed"
