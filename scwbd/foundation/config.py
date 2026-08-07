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
    "ModelConfig",
    "PosteriorConfig",
    "DataConfig",
    "StageConfig",
    "TrainConfig",
    "FoundationConfig",
    "load_config",
]


@dataclass
class ModelConfig:
    n_regions: int = 454
    #: which operator supplies F_local: "learned" or a mechanistic backend name
    local_core: str = "learned"
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
    #: observation nuisance parameters appended to theta (gain, sensor noise)
    nuisance_dim: int = 2


@dataclass
class DataConfig:
    sim_index_fast: str = "/data/scwbd/sim_corpus/index_fast.json"
    sim_index_slow: str = "/data/scwbd/sim_corpus/index_slow.json"
    real_eeg_root: str = "/data/scwbd/eegmmidb/1.0.0"
    real_sleep_root: str = "/data/scwbd/sleep-edfx/1.0.0"
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
    mixture_cards: str = "configs/source_cards"
    notes: str = ""

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


def load_config(path: str | Path, **overrides: Any) -> FoundationConfig:
    """Load a YAML config, applying ``base:`` inheritance and CLI overrides."""
    p = Path(path)
    payload = yaml.safe_load(p.read_text()) or {}
    if "base" in payload:
        base_path = (p.parent / payload.pop("base")).resolve()
        base = yaml.safe_load(Path(base_path).read_text()) or {}
        payload = _deep_update(base, payload)
    for dotted, val in overrides.items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return FoundationConfig.from_dict(payload)
