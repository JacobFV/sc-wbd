"""Hippocampal high-dimensional codes (§5.1) with interchangeable backends.

The state is ``H_t = {k_t, v_t, g_t, c_t, rho_t}``: cue/index, bound content,
multiscale relational (grid-like) code, temporal/contextual state, retrieval
confidence.  Four backends share that interface:

``ModernHopfield``      softmax-similarity associative memory (Ramsauer et al. 2020)
``VectorHaSH``          factorized modular grid code + sparse expansion (Chandra et al. 2025)
``SparseDistributedMemory``  Kanerva SDM with random hard locations and counters
``SuccessorRepresentation``  TD-learned SR — a *relational* memory, not an episodic one

The thesis is explicit that "a bare softmax retrieval equation does not
distinguish hippocampal hypotheses".  So the module ships the **discriminating
benchmark**, not just the backends: capacity vs code dimension and sparsity,
interference as episodes accumulate, cue-degradation curves, pattern
separation/completion, and replay order.  A backend is selected by those
signatures.  :func:`compare_backends` runs them all and returns a table; nothing
in this module tells you which backend is "right".
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, default_device, make_generator

__all__ = [
    "HippocampalState",
    "RetrievalResult",
    "HippocampalBackend",
    "ModernHopfield",
    "VectorHaSH",
    "SparseDistributedMemory",
    "SuccessorRepresentation",
    "HIPPOCAMPAL_BACKENDS",
    "get_hippocampal_backend",
    "MemorySignature",
    "capacity_curve",
    "interference_curve",
    "cue_degradation_curve",
    "pattern_separation",
    "replay_order_score",
    "signature",
    "compare_backends",
]


@dataclass
class HippocampalState:
    """``H_t = {k_t, v_t, g_t, c_t, rho_t}`` (§5.1)."""

    k: Tensor | None = None  # (M, d_k) cue/index
    v: Tensor | None = None  # (M, d_v) bound content
    g: Tensor | None = None  # (M, d_g) multiscale relational / grid-like code
    c: Tensor | None = None  # (M, d_c) temporal & contextual state
    rho: Tensor | None = None  # (M,) retrieval confidence / trace strength

    @property
    def n_items(self) -> int:
        return 0 if self.k is None else int(self.k.shape[0])


@dataclass
class RetrievalResult:
    value: Tensor  # (Q, d_v)
    rho: Tensor  # (Q,) retrieval confidence in [0, 1]
    code: Tensor | None = None  # (Q, d_g)
    index: Tensor | None = None  # (Q,) argmax stored item, when defined


class HippocampalBackend(nn.Module, abc.ABC):
    """One interface, several mechanistic hypotheses."""

    name: ClassVar[str] = "abstract"
    #: what this backend claims; used when reporting a comparison
    hypothesis: ClassVar[str] = ""

    def __init__(self, d_key: int, d_value: int, *, device=None, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.d_key, self.d_value = int(d_key), int(d_value)
        self.device_ = default_device(device)
        self.dtype = dtype
        self.state = HippocampalState()

    @abc.abstractmethod
    def write(self, k: Tensor, v: Tensor, c: Tensor | None = None) -> None:
        """Bind ``(k, v)`` pairs.  ``k``: ``(M, d_key)``, ``v``: ``(M, d_value)``."""

    @abc.abstractmethod
    def read(self, cue: Tensor) -> RetrievalResult:
        """Retrieve from ``(Q, d_key)`` cues."""

    def encode(self, k: Tensor) -> Tensor:
        """Cue -> internal code ``g`` (the pattern-separation stage)."""
        return k

    def reset(self) -> None:
        self.state = HippocampalState()

    def replay(
        self, n: int, *, order: Literal["forward", "reverse", "random"] = "forward", seed: int = 0
    ) -> Tensor:
        """Return indices of replayed items.

        The default is encoding order / reverse; backends with genuine sequence
        structure (SR) override this and their replay order is *predicted*, not
        stipulated — which is one of the discriminating signatures.
        """
        M = self.state.n_items
        n = min(n, M)
        if order == "forward":
            return torch.arange(M - n, M, device=self.device_)
        if order == "reverse":
            return torch.arange(M - 1, M - n - 1, -1, device=self.device_)
        g = make_generator(seed, self.device_)
        return torch.randperm(M, generator=g, device=self.device_)[:n]

    def capacity_hint(self) -> float:
        """Backend's own theoretical capacity claim (items).  For reporting only."""
        return float("nan")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "d_key": self.d_key,
            "d_value": self.d_value,
            "n_items": self.state.n_items,
            "capacity_hint": self.capacity_hint(),
            "parameters": sum(p.numel() for p in self.parameters()),
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ModernHopfield(HippocampalBackend):
    """Softmax-similarity associative memory: ``v_hat = V^T softmax(beta K k)``.

    Exponential capacity in ``d_key`` for well-separated patterns, and one
    retrieval step is one attention operation.  The mechanistic commitment is
    weak on purpose — this is the backend a generic transformer already implies,
    so it is the *control* for the more committed hippocampal hypotheses.
    """

    name: ClassVar[str] = "modern_hopfield"
    hypothesis: ClassVar[str] = "attractor retrieval by similarity softmax; exponential capacity"

    def __init__(self, d_key: int, d_value: int, *, beta: float = 8.0, normalize: bool = True, **kw):
        super().__init__(d_key, d_value, **kw)
        self.beta = float(beta)
        self.normalize = bool(normalize)

    def write(self, k: Tensor, v: Tensor, c: Tensor | None = None) -> None:
        k = k.to(self.device_, self.dtype)
        v = v.to(self.device_, self.dtype)
        st = self.state
        st.k = k if st.k is None else torch.cat([st.k, k])
        st.v = v if st.v is None else torch.cat([st.v, v])
        st.rho = torch.ones(st.k.shape[0], device=self.device_, dtype=self.dtype)

    def read(self, cue: Tensor) -> RetrievalResult:
        if self.state.k is None:
            raise RuntimeError("memory is empty")
        q = cue.to(self.device_, self.dtype)
        K, V = self.state.k, self.state.v
        if self.normalize:
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            Kn = K
        logits = self.beta * (q @ Kn.T)
        p = torch.softmax(logits, dim=-1)
        v_hat = p @ V
        rho, idx = p.max(dim=-1)
        return RetrievalResult(v_hat, rho, code=q, index=idx)

    def capacity_hint(self) -> float:
        return float(math.exp(self.d_key / 4.0))


class VectorHaSH(HippocampalBackend):
    """Factorized modular grid code + sparse hippocampal expansion.

    ``g`` is the concatenation of one-hot phases in coprime modules (a CRT
    code), giving an exponentially large, *fixed*, non-interfering scaffold;
    ``h`` is a sparse random expansion of ``g``; sensory content binds to ``h``
    by a Hebbian outer product.  The discriminating prediction is that capacity
    is set by the **scaffold**, so it is nearly flat in the number of stored
    items until the scaffold is exhausted, and degrades gracefully rather than
    catastrophically (contrast Hopfield's cliff).
    """

    name: ClassVar[str] = "vector_hash"
    hypothesis: ClassVar[str] = "fixed factorized grid scaffold sets capacity; content binds by Hebbian outer product"

    def __init__(
        self,
        d_key: int,
        d_value: int,
        *,
        modules: Sequence[int] = (3, 4, 5),
        d_hidden: int = 256,
        sparsity: float = 0.05,
        seed: int = 0,
        **kw,
    ):
        super().__init__(d_key, d_value, **kw)
        self.modules = tuple(int(m) for m in modules)
        self.d_g = sum(self.modules)
        self.d_h = int(d_hidden)
        self.sparsity = float(sparsity)
        g = make_generator(seed, self.device_)
        # fixed random projections: cue -> grid phases, grid -> sparse expansion
        self.register_buffer(
            "P_kg", torch.randn(d_key, len(self.modules), generator=g, device=self.device_, dtype=self.dtype)
        )
        self.register_buffer(
            "W_gh", torch.randn(self.d_g, self.d_h, generator=g, device=self.device_, dtype=self.dtype)
            / math.sqrt(self.d_g)
        )
        self.register_buffer("M_hv", torch.zeros(self.d_h, d_value, device=self.device_, dtype=self.dtype))
        self.register_buffer("M_vh", torch.zeros(d_value, self.d_h, device=self.device_, dtype=self.dtype))
        self.n_written = 0

    def grid_code(self, k: Tensor) -> Tensor:
        """Cue -> concatenated one-hot module phases (the CRT scaffold).

        The projection is squashed into ``(-1, 1)`` before binning so that the
        module phases behave like grid phases over a bounded latent variable.
        Without the squash the bin width is arbitrary relative to the cue scale
        and the "grid code" degenerates into random hashing — which destroys the
        graded pattern-separation curve that is supposed to distinguish this
        backend from a random-projection memory.
        """
        proj = torch.tanh(k @ self.P_kg / math.sqrt(self.d_key))  # (Q, n_modules) in (-1, 1)
        u = 0.5 * (proj + 1.0)
        outs = []
        for i, m in enumerate(self.modules):
            phase = torch.remainder(torch.floor(u[:, i] * m), m).long()
            outs.append(torch.nn.functional.one_hot(phase, m).to(self.dtype))
        return torch.cat(outs, dim=-1)

    def encode(self, k: Tensor) -> Tensor:
        g = self.grid_code(k.to(self.device_, self.dtype))
        h = g @ self.W_gh
        kth = max(int(self.sparsity * self.d_h), 1)
        thresh = torch.topk(h, kth, dim=-1).values[:, -1:]
        return (h >= thresh).to(self.dtype)  # sparse binary expansion

    def write(self, k: Tensor, v: Tensor, c: Tensor | None = None) -> None:
        k = k.to(self.device_, self.dtype)
        v = v.to(self.device_, self.dtype)
        h = self.encode(k)
        self.M_hv += h.T @ v
        self.M_vh += v.T @ h
        st = self.state
        st.k = k if st.k is None else torch.cat([st.k, k])
        st.v = v if st.v is None else torch.cat([st.v, v])
        st.g = h if st.g is None else torch.cat([st.g, h])
        st.rho = torch.ones(st.k.shape[0], device=self.device_, dtype=self.dtype)
        self.n_written += int(k.shape[0])

    def read(self, cue: Tensor) -> RetrievalResult:
        h = self.encode(cue.to(self.device_, self.dtype))
        n_active = h.sum(dim=-1, keepdim=True).clamp_min(1.0)
        v_hat = (h @ self.M_hv) / n_active
        # confidence: agreement between the retrieved content's own code and the cue's
        h_back = self.encode_from_value(v_hat)
        rho = (h * h_back).sum(-1) / h.sum(-1).clamp_min(1.0)
        return RetrievalResult(v_hat, rho.clamp(0, 1), code=h)

    def encode_from_value(self, v: Tensor) -> Tensor:
        h = v @ self.M_vh
        kth = max(int(self.sparsity * self.d_h), 1)
        thresh = torch.topk(h, kth, dim=-1).values[:, -1:]
        return (h >= thresh).to(self.dtype)

    def capacity_hint(self) -> float:
        return float(math.prod(self.modules))


class SparseDistributedMemory(HippocampalBackend):
    """Kanerva SDM: random hard locations, Hamming-radius activation, counters.

    The mechanistic commitment is a *distributed* write: each item updates every
    location within a radius, so interference accumulates smoothly and
    cue-degradation tolerance is set by the activation radius rather than by
    pattern orthogonality.  That is a different curve shape from Hopfield's, and
    the benchmark can see it.
    """

    name: ClassVar[str] = "sdm"
    hypothesis: ClassVar[str] = "distributed writes over random address decoders; radius-limited generalisation"

    def __init__(
        self,
        d_key: int,
        d_value: int,
        *,
        n_locations: int = 1024,
        activation_frac: float = 0.02,
        seed: int = 0,
        **kw,
    ):
        super().__init__(d_key, d_value, **kw)
        g = make_generator(seed, self.device_)
        self.n_locations = int(n_locations)
        self.activation_frac = float(activation_frac)
        self.register_buffer(
            "addresses",
            torch.sign(torch.randn(n_locations, d_key, generator=g, device=self.device_, dtype=self.dtype)),
        )
        self.register_buffer("counters", torch.zeros(n_locations, d_value, device=self.device_, dtype=self.dtype))
        self.register_buffer("hits", torch.zeros(n_locations, device=self.device_, dtype=self.dtype))

    def _active(self, k: Tensor) -> Tensor:
        sim = torch.sign(k) @ self.addresses.T / self.d_key  # (Q, L) in [-1, 1]
        kth = max(int(self.activation_frac * self.n_locations), 1)
        thresh = torch.topk(sim, kth, dim=-1).values[:, -1:]
        return (sim >= thresh).to(self.dtype)

    def encode(self, k: Tensor) -> Tensor:
        return self._active(k.to(self.device_, self.dtype))

    def write(self, k: Tensor, v: Tensor, c: Tensor | None = None) -> None:
        k = k.to(self.device_, self.dtype)
        v = v.to(self.device_, self.dtype)
        a = self._active(k)  # (M, L)
        self.counters += a.T @ v
        self.hits += a.sum(dim=0)
        st = self.state
        st.k = k if st.k is None else torch.cat([st.k, k])
        st.v = v if st.v is None else torch.cat([st.v, v])
        st.rho = torch.ones(st.k.shape[0], device=self.device_, dtype=self.dtype)

    def read(self, cue: Tensor) -> RetrievalResult:
        a = self._active(cue.to(self.device_, self.dtype))
        n = a.sum(dim=-1, keepdim=True).clamp_min(1.0)
        v_hat = (a @ self.counters) / n
        used = (a * self.hits.unsqueeze(0)).sum(-1) / n.squeeze(-1).clamp_min(1.0)
        rho = 1.0 / (1.0 + used / max(self.n_locations * self.activation_frac, 1.0))
        return RetrievalResult(v_hat, rho.clamp(0, 1), code=a)

    def capacity_hint(self) -> float:
        return 0.15 * self.n_locations


class SuccessorRepresentation(HippocampalBackend):
    """TD-learned successor representation over discrete states.

    ``M[s, s'] = E[sum_t gamma^t 1(s_t = s') | s_0 = s]``.  This backend does not
    store episodes at all: it stores *predictive relations*.  Its discriminating
    signature is replay order — reverse replay after reward and forward
    sweeps during planning fall out of the SR, whereas an episodic store has to
    be told which order to replay in.
    """

    name: ClassVar[str] = "successor_representation"
    hypothesis: ClassVar[str] = "predictive map over states; replay order is derived, not stipulated"

    def __init__(self, d_key: int, d_value: int, *, n_states: int = 64, gamma: float = 0.9, lr: float = 0.2, seed: int = 0, **kw):
        super().__init__(d_key, d_value, **kw)
        self.n_states = int(n_states)
        self.gamma = float(gamma)
        self.lr = float(lr)
        g = make_generator(seed, self.device_)
        self.register_buffer(
            "codebook", torch.randn(n_states, d_key, generator=g, device=self.device_, dtype=self.dtype)
        )
        self.register_buffer("M", torch.eye(n_states, device=self.device_, dtype=self.dtype))
        self.register_buffer("V", torch.zeros(n_states, d_value, device=self.device_, dtype=self.dtype))
        self.register_buffer("counts", torch.zeros(n_states, device=self.device_, dtype=self.dtype))
        self._last_state: int | None = None

    def quantize(self, k: Tensor) -> Tensor:
        kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cb = self.codebook / self.codebook.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (kn @ cb.T).argmax(dim=-1)

    def encode(self, k: Tensor) -> Tensor:
        s = self.quantize(k.to(self.device_, self.dtype))
        return self.M[s]

    def write(self, k: Tensor, v: Tensor, c: Tensor | None = None) -> None:
        k = k.to(self.device_, self.dtype)
        v = v.to(self.device_, self.dtype)
        states = self.quantize(k)
        for i in range(states.shape[0]):
            s = int(states[i])
            onehot = torch.zeros(self.n_states, device=self.device_, dtype=self.dtype)
            onehot[s] = 1.0
            if self._last_state is not None:
                p = self._last_state
                td = onehot + self.gamma * self.M[s] - self.M[p]
                self.M[p] += self.lr * td
            self.counts[s] += 1
            self.V[s] += (v[i] - self.V[s]) / self.counts[s]
            self._last_state = s
        st = self.state
        st.k = k if st.k is None else torch.cat([st.k, k])
        st.v = v if st.v is None else torch.cat([st.v, v])
        st.rho = torch.ones(st.k.shape[0], device=self.device_, dtype=self.dtype)

    def read(self, cue: Tensor) -> RetrievalResult:
        s = self.quantize(cue.to(self.device_, self.dtype))
        m = self.M[s]  # (Q, S) predictive weights
        w = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        v_hat = w @ self.V
        rho = (self.counts[s] > 0).to(self.dtype) * w.max(dim=-1).values
        return RetrievalResult(v_hat, rho.clamp(0, 1), code=m, index=s)

    def replay(self, n: int, *, order: str = "forward", seed: int = 0) -> Tensor:
        """Replay order **derived** from the successor matrix.

        Forward: greedily follow the highest successor weight from the last
        state.  Reverse: follow the highest *predecessor* weight (column of M).
        """
        if self._last_state is None:
            return torch.zeros(0, dtype=torch.long, device=self.device_)
        seq = [self._last_state]
        M = self.M.clone()
        M.fill_diagonal_(0.0)
        for _ in range(n - 1):
            cur = seq[-1]
            nxt = int((M[cur] if order != "reverse" else M[:, cur]).argmax())
            if order == "random":
                g = make_generator(seed + len(seq), self.device_)
                nxt = int(torch.randint(0, self.n_states, (1,), generator=g, device=self.device_))
            seq.append(nxt)
        return torch.tensor(seq, device=self.device_, dtype=torch.long)

    def capacity_hint(self) -> float:
        return float(self.n_states)


HIPPOCAMPAL_BACKENDS: dict[str, type[HippocampalBackend]] = {
    ModernHopfield.name: ModernHopfield,
    VectorHaSH.name: VectorHaSH,
    SparseDistributedMemory.name: SparseDistributedMemory,
    SuccessorRepresentation.name: SuccessorRepresentation,
}


def get_hippocampal_backend(name: str) -> type[HippocampalBackend]:
    if name not in HIPPOCAMPAL_BACKENDS:
        raise KeyError(f"unknown hippocampal backend {name!r}; available: {sorted(HIPPOCAMPAL_BACKENDS)}")
    return HIPPOCAMPAL_BACKENDS[name]


# ---------------------------------------------------------------------------
# The discriminating benchmark
# ---------------------------------------------------------------------------


def _random_patterns(n: int, d: int, *, seed: int, device, sparsity: float | None = None) -> Tensor:
    g = make_generator(seed, device)
    x = torch.randn(n, d, generator=g, device=device, dtype=DTYPE)
    if sparsity is not None:
        kth = max(int(sparsity * d), 1)
        thresh = torch.topk(x, kth, dim=-1).values[:, -1:]
        x = (x >= thresh).to(DTYPE)
    return x


def _recall_accuracy(v_true: Tensor, v_hat: Tensor) -> float:
    """Cosine-nearest-neighbour recall: is the retrieved content closest to the true item?"""
    a = v_hat / v_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = v_true / v_true.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = a @ b.T
    pred = sim.argmax(dim=-1)
    target = torch.arange(v_true.shape[0], device=v_true.device)
    return float((pred == target).to(DTYPE).mean())


BackendFactory = Callable[[], HippocampalBackend]


def capacity_curve(
    factory: BackendFactory,
    *,
    loads: Sequence[int],
    d_key: int,
    d_value: int,
    criterion: float = 0.9,
    seed: int = 0,
    device=None,
) -> dict[str, Any]:
    """Recall accuracy vs number of stored episodes; capacity = last load at criterion."""
    dev = default_device(device)
    acc: list[float] = []
    for i, m in enumerate(loads):
        mem = factory()
        k = _random_patterns(m, d_key, seed=seed + i, device=dev)
        v = _random_patterns(m, d_value, seed=seed + 1000 + i, device=dev)
        mem.write(k, v)
        acc.append(_recall_accuracy(v, mem.read(k).value))
    cap = 0
    for m, a in zip(loads, acc):
        if a >= criterion:
            cap = m
    return {"loads": list(loads), "accuracy": acc, "capacity_at_criterion": cap, "criterion": criterion}


def interference_curve(
    factory: BackendFactory,
    *,
    n_episodes: int,
    chunk: int,
    d_key: int,
    d_value: int,
    seed: int = 0,
    device=None,
) -> dict[str, Any]:
    """Accuracy on the **first** chunk as later episodes accumulate."""
    dev = default_device(device)
    mem = factory()
    k0 = _random_patterns(chunk, d_key, seed=seed, device=dev)
    v0 = _random_patterns(chunk, d_value, seed=seed + 7, device=dev)
    mem.write(k0, v0)
    xs, ys = [chunk], [_recall_accuracy(v0, mem.read(k0).value)]
    written = chunk
    step = 0
    while written < n_episodes:
        step += 1
        k = _random_patterns(chunk, d_key, seed=seed + 100 * step, device=dev)
        v = _random_patterns(chunk, d_value, seed=seed + 100 * step + 7, device=dev)
        mem.write(k, v)
        written += chunk
        xs.append(written)
        ys.append(_recall_accuracy(v0, mem.read(k0).value))
    return {"n_stored": xs, "first_chunk_accuracy": ys, "retention_final": ys[-1]}


def cue_degradation_curve(
    factory: BackendFactory,
    *,
    n_items: int,
    fractions: Sequence[float],
    d_key: int,
    d_value: int,
    seed: int = 0,
    device=None,
) -> dict[str, Any]:
    """Accuracy vs fraction of the cue replaced by noise (completion from partial cues)."""
    dev = default_device(device)
    mem = factory()
    k = _random_patterns(n_items, d_key, seed=seed, device=dev)
    v = _random_patterns(n_items, d_value, seed=seed + 7, device=dev)
    mem.write(k, v)
    g = make_generator(seed + 99, dev)
    noise = torch.randn(k.shape, generator=g, device=dev, dtype=k.dtype)
    accs = []
    for f in fractions:
        mask = (torch.rand(k.shape, generator=g, device=dev) < f).to(DTYPE)
        cue = k * (1 - mask) + noise * mask
        accs.append(_recall_accuracy(v, mem.read(cue).value))
    return {"corruption": list(fractions), "accuracy": accs}


def pattern_separation(
    factory: BackendFactory,
    *,
    n_pairs: int,
    input_similarities: Sequence[float],
    d_key: int,
    d_value: int,
    seed: int = 0,
    device=None,
) -> dict[str, Any]:
    """Output-code similarity vs input similarity.

    Separation slope < 1 means similar inputs are decorrelated (separation);
    slope > 1 means they are pulled together (completion).  The *shape* of this
    curve — where it crosses — is a discriminating signature.
    """
    dev = default_device(device)
    mem = factory()
    a = _random_patterns(n_pairs, d_key, seed=seed, device=dev)
    out_sims: list[float] = []
    for s in input_similarities:
        noise = _random_patterns(n_pairs, d_key, seed=seed + 31, device=dev)
        b = s * a + math.sqrt(max(1 - s * s, 0.0)) * noise
        ca, cb = mem.encode(a), mem.encode(b)
        ca = ca / ca.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cb = cb / cb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out_sims.append(float((ca * cb).sum(-1).mean()))
    xs = torch.tensor(list(input_similarities), dtype=torch.float64)
    ys = torch.tensor(out_sims, dtype=torch.float64)
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum().clamp_min(1e-12))
    return {"input_similarity": list(input_similarities), "output_similarity": out_sims, "slope": slope}


def replay_order_score(
    factory: BackendFactory,
    *,
    n_items: int,
    d_key: int,
    d_value: int,
    seed: int = 0,
    device=None,
) -> dict[str, Any]:
    """Does replay recover the encoding order (forward) and its reverse?

    Reported as Spearman-style rank correlation between the replayed sequence
    and the encoding sequence.  A store with no sequence structure scores ~0 on
    a *derived* order and 1.0 only because it was told the order — which the
    report distinguishes via ``derived``.
    """
    dev = default_device(device)
    mem = factory()
    k = _random_patterns(n_items, d_key, seed=seed, device=dev)
    v = _random_patterns(n_items, d_value, seed=seed + 7, device=dev)
    for i in range(n_items):  # sequential encoding matters for SR
        mem.write(k[i : i + 1], v[i : i + 1])
    fwd = mem.replay(n_items, order="forward")
    rev = mem.replay(n_items, order="reverse")

    def rank_corr(seq: Tensor) -> float:
        if seq.numel() < 3:
            return float("nan")
        x = torch.arange(seq.numel(), dtype=torch.float64, device=seq.device)
        y = seq.to(torch.float64)
        x = x - x.mean()
        y = y - y.mean()
        den = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).clamp_min(1e-12)
        return float((x * y).sum() / den)

    return {
        "forward_corr": rank_corr(fwd),
        "reverse_corr": rank_corr(rev),
        "derived": isinstance(mem, SuccessorRepresentation),
    }


@dataclass
class MemorySignature:
    """The full discriminating signature of one hippocampal backend."""

    name: str
    hypothesis: str
    capacity: dict[str, Any]
    interference: dict[str, Any]
    cue_degradation: dict[str, Any]
    separation: dict[str, Any]
    replay: dict[str, Any]
    describe: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "capacity": self.capacity,
            "interference": self.interference,
            "cue_degradation": self.cue_degradation,
            "separation": self.separation,
            "replay": self.replay,
            "backend": self.describe,
        }

    def summary(self) -> str:
        return (
            f"{self.name:<26} capacity@0.9={self.capacity['capacity_at_criterion']:<5d} "
            f"retention={self.interference['retention_final']:.2f} "
            f"cue50={self.cue_degradation['accuracy'][len(self.cue_degradation['accuracy'])//2]:.2f} "
            f"sep_slope={self.separation['slope']:+.2f} "
            f"replay_fwd={self.replay['forward_corr']:+.2f}"
        )


def signature(
    factory: BackendFactory,
    *,
    d_key: int = 64,
    d_value: int = 64,
    loads: Sequence[int] = (8, 16, 32, 64, 128),
    n_episodes: int = 128,
    chunk: int = 16,
    corruptions: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75),
    similarities: Sequence[float] = (0.0, 0.3, 0.6, 0.9, 0.99),
    seed: int = 0,
    device=None,
) -> MemorySignature:
    mem = factory()
    return MemorySignature(
        name=mem.name,
        hypothesis=mem.hypothesis,
        capacity=capacity_curve(factory, loads=loads, d_key=d_key, d_value=d_value, seed=seed, device=device),
        interference=interference_curve(
            factory, n_episodes=n_episodes, chunk=chunk, d_key=d_key, d_value=d_value, seed=seed, device=device
        ),
        cue_degradation=cue_degradation_curve(
            factory, n_items=min(loads), fractions=corruptions, d_key=d_key, d_value=d_value, seed=seed, device=device
        ),
        separation=pattern_separation(
            factory, n_pairs=32, input_similarities=similarities, d_key=d_key, d_value=d_value, seed=seed, device=device
        ),
        replay=replay_order_score(factory, n_items=12, d_key=d_key, d_value=d_value, seed=seed, device=device),
        describe=mem.describe(),
    )


def compare_backends(
    factories: Mapping[str, BackendFactory] | None = None,
    *,
    d_key: int = 64,
    d_value: int = 64,
    seed: int = 0,
    device=None,
    **kw,
) -> dict[str, MemorySignature]:
    """Run the discriminating benchmark over backends.

    Returns the signatures.  It deliberately does **not** rank them: selection
    is by which signature matches the data you are modelling, and that decision
    belongs to the claim report, not to this function.
    """
    if factories is None:
        factories = {
            "modern_hopfield": lambda: ModernHopfield(d_key, d_value, device=device),
            "vector_hash": lambda: VectorHaSH(d_key, d_value, device=device),
            "sdm": lambda: SparseDistributedMemory(d_key, d_value, device=device),
            "successor_representation": lambda: SuccessorRepresentation(d_key, d_value, device=device),
        }
    return {
        name: signature(f, d_key=d_key, d_value=d_value, seed=seed, device=device, **kw)
        for name, f in factories.items()
    }
