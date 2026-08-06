"""Phase 2 -- the simulated whole-brain corpus generator.

Sweeps global coupling ``G``, conduction velocity ``v``, regional E/I balance,
noise level, external drive and **backend family**, integrating delayed
connectome-coupled SDEs on the GPU in fp32 and writing fp16 HDF5 shards.

Every trajectory written here is **simulator-conditioned evidence**
(``paper/body.tex`` §6.3): it exercises physics, missingness and rare regimes,
it can *never* count as participant data, and it cannot establish biological
validity.  Each shard carries its source-card id and a model-discrepancy term;
:mod:`scwbd.foundation.mixture` refuses to give a simulated source the
``likelihood`` role over measured observables.

Layout of a shard (``*.h5``)::

    activity        (n_traj, T, N)  float16   box-averaged regional activity
    theta           (n_traj, P)     float32   generative parameters (the NPE target)
    theta_names     attr, list[str]
    ei_regional     (n_traj, N)     float16   realised per-region E/I ratio
    attrs: backend, tier, dt, store_every, integration_window_s, fs_hz, seed,
           velocity_m_s, control_graph, anatomy_provenance, source_card,
           delay_quantisation_rmse_s, git_sha, ...
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from .anatomy import AnatomyPrior, load_anatomy
from .backends import DelayedCoupling, resolve_backend
from .util import git_sha, set_determinism

__all__ = [
    "CorpusSpec",
    "ThetaPrior",
    "generate_corpus",
    "simulate_batch",
    "CorpusIndex",
    "SimCorpus",
    "THETA_NAMES",
]

#: The parameters the amortized posterior must recover (ARCHITECTURE.md §5).
THETA_NAMES = (
    "log_G",  # global coupling scale
    "log_velocity",  # conduction velocity, m/s
    "ei_global",  # mean regional E/I balance
    "ei_gradient",  # loading of E/I on the principal gradient
    "log_sigma",  # process-noise amplitude
    "drive",  # external/tonic drive
)


@dataclass
class ThetaPrior:
    """Uniform-in-transformed-space prior over the generative parameters.

    Deliberately broad: the corpus must cover regimes the fitted model will be
    asked about, including ones no human brain occupies, so that the amortized
    posterior is calibrated rather than merely confident.
    """

    log_G: tuple[float, float] = (math.log(0.02), math.log(3.0))
    log_velocity: tuple[float, float] = (math.log(1.5), math.log(12.0))
    ei_global: tuple[float, float] = (0.6, 1.6)
    ei_gradient: tuple[float, float] = (-0.45, 0.45)
    log_sigma: tuple[float, float] = (math.log(2e-3), math.log(1.5e-1))
    drive: tuple[float, float] = (0.3, 1.9)

    def bounds(self) -> Tensor:
        return torch.tensor([getattr(self, n) for n in THETA_NAMES], dtype=torch.float32)

    def sample(self, n: int, *, seed: int, device="cpu") -> Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        b = self.bounds()
        u = torch.rand(n, len(THETA_NAMES), generator=g)
        return (b[:, 0] + u * (b[:, 1] - b[:, 0])).to(device)

    def normalise(self, theta: Tensor) -> Tensor:
        """Map to [-1,1] for the posterior's unconstrained space."""
        b = self.bounds().to(theta.device)
        return 2 * (theta - b[:, 0]) / (b[:, 1] - b[:, 0]) - 1

    def denormalise(self, z: Tensor) -> Tensor:
        b = self.bounds().to(z.device)
        return b[:, 0] + (z + 1) / 2 * (b[:, 1] - b[:, 0])

    def log_prob(self, theta: Tensor) -> Tensor:
        b = self.bounds().to(theta.device)
        inside = ((theta >= b[:, 0]) & (theta <= b[:, 1])).all(-1)
        lp = -torch.log(b[:, 1] - b[:, 0]).sum()
        return torch.where(inside, lp, torch.full_like(inside, -1e30, dtype=theta.dtype))

    def as_dict(self) -> dict[str, list[float]]:
        return {n: list(getattr(self, n)) for n in THETA_NAMES}


@dataclass
class CorpusSpec:
    """What to generate.  Serialised verbatim into the corpus index."""

    out_dir: str = "data/sim_corpus"
    tier: str = "fast"  # "fast" (EEG-scale) | "slow" (BOLD-scale)
    backends: tuple[str, ...] = (
        "wilson_cowan",
        "jansen_rit",
        "wong_wang",
        "stuart_landau",
        "linear_gaussian",
    )
    #: relative share of trajectories per backend (mechanistic families favoured)
    backend_weights: tuple[float, ...] = (0.30, 0.22, 0.22, 0.16, 0.10)
    control_graphs: tuple[str, ...] = ("none",)
    control_graph_fraction: float = 0.0
    batch: int = 1024
    dt: float = 1e-3
    duration_s: float = 12.0
    warmup_s: float = 3.0
    store_every: int = 8  # 125 Hz after box-averaging
    n_delay_bins: int = 16
    target_traj_seconds: float = 6.0e5
    seed: int = 20260805
    max_wall_seconds: float = 3600.0
    shard_trajectories: int = 1024
    device: str = "cuda"
    theta_prior: ThetaPrior = field(default_factory=ThetaPrior)

    @property
    def fs_hz(self) -> float:
        return 1.0 / (self.dt * self.store_every)

    @property
    def n_stored(self) -> int:
        return int(round(self.duration_s / self.dt)) // self.store_every

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["theta_prior"] = self.theta_prior.as_dict()
        d["fs_hz"] = self.fs_hz
        d["n_stored"] = self.n_stored
        return d


SLOW_TIER = dict(tier="slow", duration_s=240.0, warmup_s=20.0, store_every=100, batch=512)


# ----------------------------------------------------------------------
class ParameterMappingError(ValueError):
    """A theta component would have been written to a name the backend never reads.

    This is the silent failure that would ruin the amortized posterior: the
    corpus would carry a label the simulator ignored, and the posterior would
    happily learn to "recover" a parameter that did nothing.  We refuse instead.
    """


def _regional_theta(theta: Tensor, anat: AnatomyPrior, backend_name: str, *, defaults: dict | None = None) -> dict[str, Tensor]:
    """Expand the 6-D theta into per-region backend parameters.

    E/I balance is *regional* (ARCHITECTURE.md §5) but not 454 free knobs: it is
    a global level times the receptor-derived regional prior, plus a loading on
    the principal anatomical gradient.  That keeps the recovery problem posed
    and keeps the prior anatomically anchored.

    Each backend gets its **own** mapping, because the same physical statement
    ("more excitation relative to inhibition") is a different symbol in each
    formalism; pretending otherwise is exactly the semantic collapse the thesis
    forbids.  For the effective backends (Stuart-Landau, linear Gaussian) the
    E/I mapping is declared *effective*, not mechanistic, and the resulting
    posterior width over ``ei_*`` is an honest report of that.
    """
    dev = theta.device
    grad = anat.gradient.to(dev).reshape(1, -1)
    ei_prior = anat.ei_prior.to(dev).reshape(1, -1)
    tau_prior = anat.timescale_prior.to(dev).reshape(1, -1)
    ones = torch.ones_like(grad)
    G = theta[:, 0:1].exp()
    ei = (theta[:, 2:3] * ei_prior * (1.0 + theta[:, 3:4] * grad)).clamp(0.3, 2.4)
    sig = theta[:, 4:5].exp()
    drive = theta[:, 5:6]

    if backend_name == "wilson_cowan":
        out = {
            "g_coupling": (G * 1.0).clamp(0.0, 3.0) * ones,
            "ei_ratio": ei,
            "P": (0.4 + drive) * ones,
            "sigma": (sig * 0.5).clamp(1e-4, 0.06) * ones,
        }
    elif backend_name == "jansen_rit":
        out = {
            "g_coupling": (G * 6.0).clamp(0.0, 20.0) * ones,
            "p_mean": (60.0 + drive * 90.0) * ones,
            # inhibitory feedback factor: more E/I <-> weaker inhibitory loop
            "c4_f": (0.25 / ei).clamp(0.08, 0.7),
            "sigma": (sig * 250.0).clamp(0.5, 60.0) * ones,
        }
    elif backend_name in ("wong_wang", "reduced_wong_wang"):
        out = {
            "G": (G * 1.2).clamp(0.0, 3.6) * ones,
            "w_plus": (1.4 * ei).clamp(0.7, 2.2),
            "ei_ratio": ei,
            "sigma": (sig * 0.15).clamp(3e-4, 0.03) * ones,
        }
        if defaults is not None and "I_ext" in defaults:
            out["I_ext"] = ((drive - 1.0) * 0.04) * ones
    elif backend_name == "stuart_landau":
        out = {
            "G": (G * 1.0).clamp(0.0, 2.5) * ones,
            "a": (-0.16 + 0.15 * drive) * ones,  # bifurcation parameter
            # effective E/I -> excitability/frequency of the normal form
            "f": (10.0 * ei * (1.0 - 0.25 * grad)).clamp(2.0, 40.0),
            "sigma": (sig * 0.4).clamp(2e-3, 0.09) * ones,
        }
    elif backend_name == "linear_gaussian":
        out = {
            "G": (G * 0.45).clamp(0.0, 1.4) * ones,
            "tau": (tau_prior * (0.5 + drive)).clamp(0.004, 0.15),
            "self_gain": (0.4 * (ei - 1.0)).clamp(-0.35, 0.35),
            "sigma": (sig * 1.5).clamp(0.01, 0.35) * ones,
        }
    # -- per-family engineered backends (§5) -------------------------------
    # Each gets its OWN mapping for the same reason the cortical backends do:
    # "more excitation relative to inhibition" is a different symbol in each
    # formalism, and a shared mapping is exactly the semantic collapse §5 names.
    elif backend_name == "thalamic_relay":
        out = {
            "G": (G * 0.8).clamp(0.0, 2.5) * ones,
            "ei_ratio": ei,
            # a longer intrinsic timescale de-inactivates the T-current more
            # slowly, so tau_h tracks the regional timescale prior
            "tau_h": (tau_prior * 3.0).clamp(0.03, 0.30),
            "sigma": (sig * 0.4).clamp(1e-4, 0.08) * ones,
        }
    elif backend_name == "basal_ganglia_gate":
        out = {
            "G": (G * 1.0).clamp(0.0, 2.5) * ones,
            "ei_ratio": ei,
            # dopaminergic gain is a CONTROL PARAMETER on corticostriatal
            # transmission (§5), not a reward. It is driven by the theta drive
            # term and nothing here consumes an outcome.
            "dopamine": (drive - 1.0).clamp(-0.9, 0.9) * ones,
            "sigma": (sig * 0.2).clamp(1e-4, 0.04) * ones,
        }
    elif backend_name == "hippocampal_code":
        out = {
            "G": (G * 0.9).clamp(0.0, 2.0) * ones,
            # theta rhythm is a regional oscillatory prior, not free
            "theta_hz": (7.0 * (1.0 + 0.15 * grad)).clamp(4.0, 10.0),
            # retrieval sharpness rises with E/I (less inhibitory smoothing)
            "beta": (8.0 * ei).clamp(1.0, 24.0),
            "tau_c": (tau_prior * 6.0).clamp(0.05, 1.0),
            "sigma": (sig * 0.2).clamp(1e-4, 0.04) * ones,
        }
    elif backend_name == "cerebellar_forward_model":
        out = {
            "G": (G * 0.7).clamp(0.0, 2.0) * ones,
            "gain": ei.clamp(0.2, 2.0),
            "tau_elig": (tau_prior * 4.0).clamp(0.02, 0.30),
            "sigma": (sig * 0.2).clamp(1e-4, 0.04) * ones,
        }
    else:
        raise ParameterMappingError(f"no theta->parameter mapping declared for backend {backend_name!r}")

    if defaults is not None:
        unknown = sorted(set(out) - set(defaults))
        if unknown:
            raise ParameterMappingError(
                f"backend {backend_name!r} does not read {unknown}; the corpus would carry a "
                "label the simulator ignored. Update the mapping in simulate._regional_theta."
            )
    return out


@torch.no_grad()
def simulate_batch(
    *,
    anat: AnatomyPrior,
    backend_name: str,
    theta: Tensor,
    spec: CorpusSpec,
    seed: int,
    control_graph: str = "none",
    device: torch.device | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Integrate one homogeneous-velocity batch.  Returns ``(B, T, N)`` fp32 activity.

    ``theta`` is ``(B, P)`` in the :data:`THETA_NAMES` parameterisation.  All
    rows must share ``log_velocity`` (the delay matrix is per-batch, not
    per-row: a per-row ``(B,N,N)`` delay tensor would cost tens of GB).
    """
    dev = device or torch.device(spec.device)
    B = theta.shape[0]
    N = anat.n_regions
    velocity = float(theta[0, 1].exp().item())
    W = anat.weights if control_graph == "none" else anat.control_graph(control_graph, seed=seed)
    cpl = DelayedCoupling(W.to(dev), anat.delay_matrix(velocity).to(dev), dt=spec.dt, n_bins=spec.n_delay_bins, device=dev)
    backend = resolve_backend(backend_name).to(dev)
    params = _regional_theta(theta.to(dev), anat.to(dev), backend_name, defaults=dict(backend.defaults))
    pack = backend.make_theta(B, N, device=dev)
    for k, v in params.items():
        pack.set(k, v.to(torch.float32))

    n_steps = int(round(spec.duration_s / spec.dt))
    warm = int(round(spec.warmup_s / spec.dt))
    x = backend.init_state(B, N, seed=seed, device=dev, theta=pack)
    hist = cpl.new_history(B, backend.n_coupling_channels)
    L = hist.shape[1]
    hist[:, 0] = backend.coupling_variable(x, pack)
    gen = torch.Generator(device=dev).manual_seed(seed + 104729)
    sqdt = spec.dt**0.5
    se = spec.store_every
    n_store = n_steps // se
    out = torch.empty(B, n_store, N, device=dev, dtype=torch.float32)
    acc = torch.zeros(B, N, device=dev, dtype=torch.float32)
    si = 0
    nonfinite = 0
    for t in range(warm + n_steps):
        c = cpl(hist, t)
        f = backend.drift(x, c, pack, None, t * spec.dt)
        gd = backend.diffusion(x, pack)
        noise = torch.randn(x.shape, generator=gen, device=dev, dtype=torch.float32)
        x = x + spec.dt * f + gd * sqdt * noise
        if not torch.isfinite(x).all():
            nonfinite += 1
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-1e3, 1e3)
        hist[:, (t + 1) % L] = backend.coupling_variable(x, pack)
        if t >= warm:
            acc += backend.observables(x)["activity"]
            k = t - warm
            if (k + 1) % se == 0 and si < n_store:
                out[:, si] = acc / se  # box average == declared integration window
                acc.zero_()
                si += 1
    meta = {
        "backend": backend_name,
        "backend_origin": getattr(backend, "origin", "unknown"),
        "velocity_m_s": velocity,
        "control_graph": control_graph,
        "nonfinite_steps": nonfinite,
        **cpl.budget(),
    }
    return out, meta


# ----------------------------------------------------------------------
@dataclass
class CorpusIndex:
    """The manifest that makes a corpus reproducible and auditable."""

    spec: dict[str, Any]
    shards: list[dict[str, Any]] = field(default_factory=list)
    anatomy: dict[str, Any] = field(default_factory=dict)
    git_sha: str = ""
    created_utc: str = ""
    total_trajectories: int = 0
    total_trajectory_seconds: float = 0.0
    wall_seconds: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        return cls(**json.loads(Path(path).read_text()))


def generate_corpus(spec: CorpusSpec, *, anat: AnatomyPrior | None = None, verbose: bool = True) -> CorpusIndex:
    """Run the sweep and write shards.  Resumable: existing shards are skipped."""
    import h5py

    set_determinism(spec.seed)
    dev = torch.device(spec.device)
    anat = (anat or load_anatomy(device=dev)).to(dev)
    out_dir = Path(spec.out_dir) / spec.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(spec.out_dir) / f"index_{spec.tier}.json"
    idx = CorpusIndex(
        spec=spec.as_dict(),
        anatomy=anat.summary(),
        git_sha=git_sha(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    existing = {p.name for p in out_dir.glob("*.h5")}
    if index_path.exists():
        try:
            prev = CorpusIndex.load(index_path)
            idx.shards = [s for s in prev.shards if s["file"] in existing]
            idx.total_trajectories = sum(s["n_traj"] for s in idx.shards)
            idx.total_trajectory_seconds = sum(s["traj_seconds"] for s in idx.shards)
            idx.wall_seconds = prev.wall_seconds
        except Exception:  # noqa: BLE001
            pass

    weights = np.asarray(spec.backend_weights, dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(spec.seed)
    t_start = time.time()
    shard_no = len(idx.shards)
    per_traj_seconds = spec.duration_s

    while (
        idx.total_trajectory_seconds < spec.target_traj_seconds
        and (time.time() - t_start) + idx.wall_seconds < spec.max_wall_seconds
    ):
        b_i = int(rng.choice(len(spec.backends), p=weights))
        backend_name = spec.backends[b_i]
        cg = "none"
        if spec.control_graph_fraction > 0 and rng.random() < spec.control_graph_fraction:
            cg = str(rng.choice([c for c in spec.control_graphs if c != "none"]))
        seed = int(spec.seed + 1_000_003 * shard_no + b_i)
        theta = spec.theta_prior.sample(spec.batch, seed=seed, device=dev)
        theta[:, 1] = theta[0, 1]  # one conduction velocity per batch (see simulate_batch)
        t0 = time.time()
        act, meta = simulate_batch(
            anat=anat, backend_name=backend_name, theta=theta, spec=spec, seed=seed, control_graph=cg, device=dev
        )
        torch.cuda.synchronize() if dev.type == "cuda" else None
        sim_s = time.time() - t0

        # drop degenerate trajectories rather than silently training on NaNs
        finite = torch.isfinite(act).all(dim=(1, 2)) if act.ndim == 3 else torch.ones(act.shape[0], dtype=torch.bool)
        sd = act.std(dim=1).mean(dim=1)
        keep = finite & (sd > 1e-7) & (sd < 1e3)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            shard_no += 1
            continue
        act = act[keep]
        theta_k = theta[keep]
        ei = _regional_theta(theta_k, anat, backend_name).get("ei_ratio")
        if ei is None:
            ei = torch.ones(n_keep, anat.n_regions, device=dev)
        ei = ei.expand(n_keep, anat.n_regions)

        fname = f"{spec.tier}_{shard_no:05d}_{backend_name}.h5"
        fpath = out_dir / fname
        with h5py.File(fpath, "w") as f:
            f.create_dataset(
                "activity",
                data=act.to(torch.float16).cpu().numpy(),
                dtype="float16",
                chunks=(1, act.shape[1], act.shape[2]),
            )
            f.create_dataset("theta", data=theta_k.cpu().numpy().astype("float32"))
            f.create_dataset("ei_regional", data=ei.to(torch.float16).cpu().numpy())
            f.attrs.update(
                {
                    "theta_names": json.dumps(list(THETA_NAMES)),
                    "tier": spec.tier,
                    "dt": spec.dt,
                    "store_every": spec.store_every,
                    "fs_hz": spec.fs_hz,
                    "integration_window_s": spec.dt * spec.store_every,
                    "duration_s": spec.duration_s,
                    "warmup_s": spec.warmup_s,
                    "seed": seed,
                    "n_regions": anat.n_regions,
                    "anatomy_provenance": anat.provenance,
                    "source_card": f"sim_wholebrain_{backend_name}",
                    "evidence_status": "simulator_conditioned",
                    "not_participant_data": True,
                    "git_sha": git_sha(),
                    **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in meta.items()},
                }
            )
        rec = {
            "file": fname,
            "path": str(fpath),
            "backend": backend_name,
            "control_graph": cg,
            "n_traj": n_keep,
            "n_dropped": int(spec.batch - n_keep),
            "traj_seconds": n_keep * per_traj_seconds,
            "velocity_m_s": meta["velocity_m_s"],
            "delay_quantisation_rmse_s": meta["delay_quantisation_rmse_s"],
            "sim_wall_s": round(sim_s, 2),
            "bytes": fpath.stat().st_size,
            "seed": seed,
        }
        idx.shards.append(rec)
        idx.total_trajectories += n_keep
        idx.total_trajectory_seconds += rec["traj_seconds"]
        idx.wall_seconds = (time.time() - t_start) + idx.wall_seconds if shard_no == len(idx.shards) - 1 else idx.wall_seconds
        shard_no += 1
        if verbose:
            el = time.time() - t_start
            print(
                f"[{spec.tier} {shard_no:5d}] {backend_name:<18s} v={meta['velocity_m_s']:5.2f} "
                f"keep={n_keep:4d}/{spec.batch} {sim_s:6.1f}s "
                f"| {idx.total_trajectory_seconds/1e3:8.1f}k traj-s "
                f"| {idx.total_trajectory_seconds/max(el,1e-9):6.0f} traj-s/s "
                f"| {sum(s['bytes'] for s in idx.shards)/1e9:6.1f} GB",
                flush=True,
            )
        idx.wall_seconds = time.time() - t_start
        idx.save(index_path)
    idx.wall_seconds = time.time() - t_start
    idx.save(index_path)
    return idx


# ----------------------------------------------------------------------
#: A window's scale may never fall below this fraction of its most active
#: region's standard deviation.  See :func:`normalise_window`.
#:
#: Chosen from a sweep over the fast corpus (888 windows).  Non-pathological
#: windows are left *exactly* unchanged for every value in 0.00-0.40, and the
#: tail bound tightens monotonically, so 0.25 sits in the middle of a flat
#: region rather than on a knife edge:
#:
#:     q      p99 max|z|   worst max|z|
#:     0.00     740.60        2426.86     <- the old behaviour
#:     0.10      22.96          41.57
#:     0.25      10.27          21.12     <- chosen
#:     0.40       6.63          13.20
SCALE_PEAK_FLOOR = 0.25


def normalise_window(a: np.ndarray) -> np.ndarray:
    """Mean-remove and rescale one ``(T, N)`` activity window.

    The scale is the median per-region standard deviation, **floored at a fixed
    fraction of the most active region's**.  The floor is the whole point.

    Dividing by the bare median is unsafe on this corpus: in ~6 % of windows a
    majority of parcels are near-silent, so the median collapses while the peak
    does not.  Measured on the fast tier, pathological windows had a median
    regional sd of 6.74e-4 against 7.48e-2 for normal ones -- **111x smaller** --
    while their *maximum* regional sd was unchanged (1.68e-1 vs 1.48e-1).  These
    are not louder windows; their denominator fell out from under them.  The
    result was ``max|z|`` up to **2427** in real data and **3.4e6** in the
    synthetic worst case, concentrated in ``wilson_cowan`` (11.7 % of its
    windows), which supplies ~76 % of post-normalisation batch variance.

    Flooring the scale against the window's own peak activity says: **silence
    cannot inflate amplitude.**  A window in which most parcels are quiet is
    quiet -- it is not a window whose few active parcels are enormous.

    Bounded and verified: worst ``max|z|`` falls 2426.86 -> 21.12, p99
    740.60 -> 10.27, while windows that were never pathological are returned
    *bit-identical* (measured shift x1.000).
    """
    mu = a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True) + 1e-6
    scale = max(float(np.median(sd)), SCALE_PEAK_FLOOR * float(sd.max()), 1e-6)
    return (a - mu) / scale


class SimCorpus(torch.utils.data.Dataset):
    """Random-access windows over a generated corpus.

    Yields ``dict`` with ``activity (T, N) float32``, ``theta (P,)``,
    ``ei (N,)``, ``backend`` index and ``source`` id.  Windows are drawn from
    within a trajectory; the trajectory is the grouping unit for splits (there
    is no participant here -- a simulated replica is never an independent
    subject, Appendix D "Derived-data duplication").
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        window: int = 256,
        backends: Sequence[str] | None = None,
        trajectory_subset: str = "all",  # all | train | val
        val_fraction: float = 0.05,
        seed: int = 0,
        normalise: bool = True,
    ) -> None:
        import h5py  # noqa: F401

        self.index = CorpusIndex.load(index_path)
        self.window = int(window)
        self.normalise = normalise
        self.backend_names = sorted({s["backend"] for s in self.index.shards})
        keep = [s for s in self.index.shards if backends is None or s["backend"] in backends]
        keep = [s for s in keep if Path(s["path"]).exists()]
        rng = np.random.default_rng(seed)
        self.items: list[tuple[str, int, int]] = []  # (path, traj_idx, backend_idx)
        for s in keep:
            n = s["n_traj"]
            u = rng.random(n)
            for i in range(n):
                if trajectory_subset == "train" and u[i] < val_fraction:
                    continue
                if trajectory_subset == "val" and u[i] >= val_fraction:
                    continue
                self.items.append((s["path"], i, self.backend_names.index(s["backend"])))
        self._files: dict[str, Any] = {}
        self.n_regions = int(self.index.anatomy.get("n_regions", 0)) or 454
        self.fs_hz = float(self.index.spec.get("fs_hz", 125.0))

    def __len__(self) -> int:
        return len(self.items)

    def _f(self, path: str):
        import h5py

        if path not in self._files:
            self._files[path] = h5py.File(path, "r", swmr=True)
        return self._files[path]

    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> dict[str, Any]:
        path, ti, bi = self.items[i]
        f = self._f(path)
        T = f["activity"].shape[1]
        w = min(self.window, T)
        off = int(np.random.randint(0, T - w + 1)) if T > w else 0
        a = np.asarray(f["activity"][ti, off : off + w], dtype=np.float32)
        th = np.asarray(f["theta"][ti], dtype=np.float32)
        ei = np.asarray(f["ei_regional"][ti], dtype=np.float32)
        if self.normalise:
            a = normalise_window(a)
        return {
            "activity": torch.from_numpy(a),
            "theta": torch.from_numpy(th),
            "ei": torch.from_numpy(ei),
            "backend": bi,
            "source": "simulated",
        }

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def stats(self) -> dict[str, Any]:
        by_backend: dict[str, int] = {}
        for s in self.index.shards:
            by_backend[s["backend"]] = by_backend.get(s["backend"], 0) + s["n_traj"]
        return {
            "n_windows": len(self),
            "n_trajectories": self.index.total_trajectories,
            "trajectory_seconds": self.index.total_trajectory_seconds,
            "by_backend": by_backend,
            "bytes": sum(s["bytes"] for s in self.index.shards),
            "fs_hz": self.fs_hz,
            "n_regions": self.n_regions,
        }


def _cli() -> None:  # pragma: no cover - operational entry point
    import argparse

    p = argparse.ArgumentParser(description="generate the SC-WBD simulated corpus")
    p.add_argument("--tier", default="fast", choices=["fast", "slow"])
    p.add_argument("--out", default=os.environ.get("SCWBD_SIM_DIR", "data/sim_corpus"))
    p.add_argument("--target-seconds", type=float, default=6e5)
    p.add_argument("--max-wall", type=float, default=3600.0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--control-fraction", type=float, default=0.0)
    a = p.parse_args()
    kw: dict[str, Any] = dict(
        out_dir=a.out,
        target_traj_seconds=a.target_seconds,
        max_wall_seconds=a.max_wall,
        batch=a.batch,
        seed=a.seed,
        control_graph_fraction=a.control_fraction,
        control_graphs=("randomized", "distance_matched", "dense", "local_only"),
    )
    if a.tier == "slow":
        kw.update(SLOW_TIER)
    spec = CorpusSpec(**kw)
    idx = generate_corpus(spec)
    print(
        json.dumps(
            {
                "tier": spec.tier,
                "trajectories": idx.total_trajectories,
                "trajectory_seconds": idx.total_trajectory_seconds,
                "wall_seconds": idx.wall_seconds,
                "gigabytes": sum(s["bytes"] for s in idx.shards) / 1e9,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    _cli()
