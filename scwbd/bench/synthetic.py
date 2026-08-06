"""Synthetic fixtures used to *test the gates themselves* (agent J).

Nothing in this module is a brain model, a dataset, or a scientific result.
It exists for one reason: **a gate that cannot fail is worthless**, so every
gate needs a world in which its null hypothesis is literally true and a world
in which the effect it looks for is literally present.  These generators
build both, so that ``tests/bench`` can prove each gate reports ``FAIL`` on
the former and ``PASS`` on the latter.

The linear--Gaussian system follows ``thesis_contract.tex`` §0.3 (T1)--(T4) in
spirit: latent regional state, a fast instantaneously-mixing observer, a slow
convolutional observer, an unknown delay, and an optional impulse.  It is a
*fixture*, not agent E's dynamics core and not agent H's Fisher machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from .harness import Dataset, Prediction

__all__ = [
    "RidgeGaussian",
    "MeanModel",
    "SmoothedModel",
    "OverconfidentModel",
    "HallucinatingFineModel",
    "HybridMechResidual",
    "make_fusion_dataset",
    "make_graph_dataset",
    "make_multiscale_dataset",
    "make_individualization_dataset",
    "make_intervention_dataset",
    "SyntheticFisher",
]

_EPS = 1e-9


# ==========================================================================
# reference arms
# ==========================================================================
def _design(data: Dataset, blocks: Sequence[str] | None) -> np.ndarray:
    keys = list(blocks) if blocks is not None else sorted(data.inputs)
    missing = [k for k in keys if k not in data.inputs]
    if missing:
        # a withheld evidence block is *not* imputed as zero (rule 1); the
        # model simply does not see it.
        keys = [k for k in keys if k in data.inputs]
    if not keys:
        return np.ones((data.n, 1))
    X = np.concatenate([np.atleast_2d(data.inputs[k].reshape(data.n, -1)) for k in keys], axis=1)
    return np.concatenate([X, np.ones((data.n, 1))], axis=1)


@dataclass
class RidgeGaussian:
    """Ridge regression with a homoscedastic (or per-output) Gaussian head.

    Deliberately boring.  It is a *measuring stick*: gates need arms whose
    capacity is exactly known, so ``n_parameters`` is the number of free
    coefficients actually fit (plus one sd per output).
    """

    name: str = "ridge"
    blocks: Sequence[str] | None = None
    alpha: float = 1.0
    #: optional (n_outputs, n_features) boolean mask restricting which inputs
    #: may drive which outputs — this is how a topology prior enters.
    mask: np.ndarray | None = None
    #: inflate/deflate the fitted sd (used to build overconfident fixtures)
    sd_scale: float = 1.0
    extra_parameters: int = 0

    W: np.ndarray | None = field(default=None, init=False, repr=False)
    sd_: np.ndarray | None = field(default=None, init=False, repr=False)
    _n_free: int = field(default=0, init=False, repr=False)

    def fit(self, data: Dataset, *, seed: int = 0) -> "RidgeGaussian":
        X = _design(data, self.blocks)
        Y = data.targets.reshape(data.n, -1)
        n_out, n_feat = Y.shape[1], X.shape[1]
        W = np.zeros((n_out, n_feat))
        free = 0
        for j in range(n_out):
            cols = np.ones(n_feat, dtype=bool)
            if self.mask is not None:
                m = np.asarray(self.mask)
                mj = m[j] if m.ndim == 2 else m
                cols = np.zeros(n_feat, dtype=bool)
                k = min(len(mj), n_feat - 1)
                cols[:k] = np.asarray(mj, dtype=bool)[:k]
                cols[-1] = True  # intercept always free
            Xc = X[:, cols]
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            W[j, cols] = np.linalg.solve(A, Xc.T @ Y[:, j])
            free += int(cols.sum())
        self.W = W
        resid = Y - X @ W.T
        dof = max(data.n - free / max(n_out, 1), 1.0)
        self.sd_ = np.sqrt(np.maximum((resid**2).sum(axis=0) / dof, 1e-8)) * self.sd_scale
        self._n_free = free + n_out
        return self

    def predict(self, data: Dataset) -> Prediction:
        if self.W is None or self.sd_ is None:
            raise RuntimeError(f"{self.name}: predict before fit")
        X = _design(data, self.blocks)
        if X.shape[1] != self.W.shape[1]:
            # evidence was withheld at test time: pad with the fitted
            # intercept only for the missing columns is NOT allowed (rule 1),
            # so we refuse instead of silently zero-filling.
            raise ValueError(
                f"{self.name}: test design has {X.shape[1]} columns, fit used "
                f"{self.W.shape[1]}; withheld evidence must be withheld at fit time too"
            )
        mu = X @ self.W.T
        sd = np.broadcast_to(self.sd_, mu.shape).copy()
        if data.targets.ndim == 1:
            mu, sd = mu.ravel(), sd.ravel()
        return Prediction(mu, np.maximum(sd, 1e-6))

    def n_parameters(self) -> int:
        return int(self._n_free + self.extra_parameters)


@dataclass
class MeanModel:
    """Predict the training mean.  The floor every claim must clear."""

    name: str = "mean"
    mu_: np.ndarray | None = field(default=None, init=False, repr=False)
    sd_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, data: Dataset, *, seed: int = 0) -> "MeanModel":
        Y = data.targets.reshape(data.n, -1)
        self.mu_ = Y.mean(axis=0)
        self.sd_ = np.maximum(Y.std(axis=0), 1e-6)
        return self

    def predict(self, data: Dataset) -> Prediction:
        assert self.mu_ is not None and self.sd_ is not None
        mu = np.broadcast_to(self.mu_, (data.n, self.mu_.size)).copy()
        sd = np.broadcast_to(self.sd_, mu.shape).copy()
        if data.targets.ndim == 1:
            mu, sd = mu.ravel(), sd.ravel()
        return Prediction(mu, sd)

    def n_parameters(self) -> int:
        return 0 if self.mu_ is None else int(2 * self.mu_.size)


@dataclass
class SmoothedModel:
    """Shrink another arm's predictions toward the mean.

    This is the fixture for §11.4's warning: it is *more stable* than the arm
    it wraps and it *destroys the effect of interest*.  ``smoothing_check``
    must fire on it.
    """

    inner: Any
    shrink: float = 0.85
    name: str = "oversmoothed"
    sd_scale: float = 0.6

    def fit(self, data: Dataset, *, seed: int = 0) -> "SmoothedModel":
        self.inner.fit(data, seed=seed)
        self._mu0 = data.targets.reshape(data.n, -1).mean(axis=0)
        return self

    def predict(self, data: Dataset) -> Prediction:
        p = self.inner.predict(data)
        mu = p.mean.reshape(data.n, -1)
        mu = (1.0 - self.shrink) * mu + self.shrink * self._mu0
        sd = p.sd.reshape(data.n, -1) * self.sd_scale
        if data.targets.ndim == 1:
            mu, sd = mu.ravel(), sd.ravel()
        return Prediction(mu, np.maximum(sd, 1e-6))

    def n_parameters(self) -> int:
        return int(self.inner.n_parameters())


@dataclass
class OverconfidentModel:
    """Same mean, smaller sd.  Fixture for the overconfidence falsifier."""

    inner: Any
    factor: float = 0.35
    name: str = "overconfident"

    def fit(self, data: Dataset, *, seed: int = 0) -> "OverconfidentModel":
        self.inner.fit(data, seed=seed)
        return self

    def predict(self, data: Dataset) -> Prediction:
        p = self.inner.predict(data)
        return Prediction(p.mean, np.maximum(p.sd * self.factor, 1e-6))

    def n_parameters(self) -> int:
        return int(self.inner.n_parameters())


@dataclass
class HallucinatingFineModel:
    """Emits fine detail it has no evidence for, at unchanged confidence.

    The fixture for G3's mandatory hallucination control: when the fine
    evidence block is withheld it substitutes a *memorised* fine-scale pattern
    from training and does not widen its predictive interval.  This is exactly
    the behaviour Appendix D's "Scale hallucination" row exists to catch, and
    the gate must report FAIL on it.
    """

    name: str = "hallucinating_fine"
    blocks: Sequence[str] | None = None
    alpha: float = 1.0
    _inner: RidgeGaussian | None = field(default=None, init=False, repr=False)

    def fit(self, data: Dataset, *, seed: int = 0) -> "HallucinatingFineModel":
        self._inner = RidgeGaussian(name=self.name, blocks=self.blocks, alpha=self.alpha)
        self._inner.fit(data, seed=seed)
        Y = data.targets.reshape(data.n, -1)
        # memorise the fine-scale (within-parcel) pattern and its amplitude
        self._pattern = Y - Y.mean(axis=0, keepdims=True)
        self._sd_floor = float(np.mean(self._inner.sd_))
        return self

    def predict(self, data: Dataset) -> Prediction:
        assert self._inner is not None
        p = self._inner.predict(data)
        mu = p.mean.reshape(data.n, -1)
        pat = self._pattern
        reps = int(np.ceil(data.n / pat.shape[0]))
        mu = mu + np.tile(pat, (reps, 1))[: data.n, : mu.shape[1]]
        sd = np.full_like(mu, self._sd_floor)   # confidence never widens
        if data.targets.ndim == 1:
            mu, sd = mu.ravel(), sd.ravel()
        return Prediction(mu, np.maximum(sd, 1e-6))

    def n_parameters(self) -> int:
        return 0 if self._inner is None else int(self._inner.n_parameters())


@dataclass
class HybridMechResidual:
    """Masked "mechanistic" term + unmasked learned residual.

    Exposes ``residual_energy()`` returning ``(||R||, ||F_mech||)``.  A model
    whose residual can silently absorb a wrong topology is exactly what G2's
    absorption sub-check and compiler refusal ``R05`` are aimed at, so the
    gate needs a fixture that does it.
    """

    mask: np.ndarray
    name: str = "hybrid"
    alpha: float = 1.0
    residual_strength: float = 1.0
    residual_rank: int | None = None

    def fit(self, data: Dataset, *, seed: int = 0) -> "HybridMechResidual":
        X = _design(data, None)
        Y = data.targets.reshape(data.n, -1)
        n_out, n_feat = Y.shape[1], X.shape[1]
        Wm = np.zeros((n_out, n_feat))
        free = 0
        for j in range(n_out):
            cols = np.zeros(n_feat, dtype=bool)
            mj = np.asarray(self.mask)[j] if np.ndim(self.mask) == 2 else np.asarray(self.mask)
            k = min(len(mj), n_feat - 1)
            cols[:k] = np.asarray(mj, dtype=bool)[:k]
            cols[-1] = True
            Xc = X[:, cols]
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            Wm[j, cols] = np.linalg.solve(A, Xc.T @ Y[:, j])
            free += int(cols.sum())
        self.W_mech = Wm
        R = Y - X @ Wm.T
        A = X.T @ X + self.alpha * np.eye(n_feat)
        self.W_res = self.residual_strength * np.linalg.solve(A, X.T @ R).T
        resid = R - X @ self.W_res.T
        self.sd_ = np.sqrt(np.maximum((resid**2).mean(axis=0), 1e-8))
        self._n_free = free + n_out * n_feat + n_out
        self._mech_energy = float(np.linalg.norm(X @ Wm.T))
        self._res_energy = float(np.linalg.norm(X @ self.W_res.T))
        return self

    def predict(self, data: Dataset) -> Prediction:
        X = _design(data, None)
        mu = X @ (self.W_mech + self.W_res).T
        sd = np.broadcast_to(self.sd_, mu.shape).copy()
        if data.targets.ndim == 1:
            mu, sd = mu.ravel(), sd.ravel()
        return Prediction(mu, np.maximum(sd, 1e-6))

    def n_parameters(self) -> int:
        return int(self._n_free)

    def residual_energy(self) -> tuple[float, float]:
        """``(||R_theta||, ||F_mech||)`` on the fitted design (refusal R05)."""
        return float(self._res_energy), float(self._mech_energy)


# ==========================================================================
# data generators
# ==========================================================================
def _ar1(n: int, rho: float, rng: np.random.Generator, d: int = 1) -> np.ndarray:
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, 1, size=d)
    return x


def make_fusion_dataset(
    *,
    n_train: int = 400,
    n_test: int = 400,
    seed: int = 0,
    delay: int = 3,
    bold_informative: bool = True,
    n_groups: int = 8,
) -> dict[str, Any]:
    """Two source-native views of one latent process (T1)--(T3), fixture form.

    ``bold_informative=False`` builds the **null-true** world for G1: the slow
    modality carries no information about the target, so typed fusion has
    nothing to gain and the gate must report FAIL rather than a small win.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = _ar1(n + delay + 8, 0.85, rng, d=3)

    # fast observer: instantaneous linear mixing (EEG-like)
    L = rng.normal(0, 1, size=(6, 3))
    eeg = x[: n] @ L.T + rng.normal(0, 0.5, size=(n, 6))

    # slow observer: causal convolution over the last 8 samples (BOLD-like)
    h = np.exp(-np.arange(8) / 3.0)
    h /= h.sum()
    conv = np.stack([np.convolve(x[:, k], h, mode="full")[:n] for k in range(3)], axis=1)
    if bold_informative:
        bold = conv + rng.normal(0, 0.3, size=(n, 3))
    else:
        bold = rng.normal(0, 1.0, size=(n, 3))

    # target: the latent state `delay` steps ahead, which the slow view helps
    # disambiguate only when it is informative
    y = x[delay : delay + n, 0] + 0.5 * conv[:, 1] * (1.0 if bold_informative else 0.0)
    y = y + rng.normal(0, 0.2, size=n)

    groups = np.array([f"P{i % n_groups:02d}" for i in range(n)])
    sites = np.where(np.arange(n) % 2 == 0, "siteA", "siteB")
    # naive resampling: everything forced onto the slow grid and back, losing
    # the fast structure -- the baseline G1 must beat.
    step = 4
    naive = np.repeat(eeg[::step], step, axis=0)[:n]
    naive = np.concatenate([naive, np.repeat(bold[::step], step, axis=0)[:n]], axis=1)

    def _mk(sl: slice, nm: str) -> Dataset:
        return Dataset(
            name=nm,
            targets=y[sl],
            inputs={"eeg": eeg[sl], "bold": bold[sl], "naive_resampled": naive[sl]},
            strata={"site": sites[sl], "session": groups[sl]},
            groups=groups[sl],
            meta={"delay_true": delay, "bold_informative": bold_informative},
        )

    # group-disjoint split (participants never cross the holdout)
    tr_mask = np.array([int(g[1:]) < n_groups // 2 for g in groups])
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(~tr_mask)[0]
    full_tr, full_te = _mk(slice(None), "fusion"), _mk(slice(None), "fusion")
    return {
        "train": full_tr.subset(tr_idx, name="fusion.train"),
        "test": full_te.subset(te_idx, name="fusion.test"),
        "delay_true": delay,
        "bold_informative": bold_informative,
    }


def make_graph_dataset(
    *,
    n_regions: int = 12,
    n_train: int = 120,
    n_test: int = 400,
    seed: int = 0,
    density: float = 0.2,
    anatomy_is_true: bool = True,
    noise: float = 0.6,
) -> dict[str, Any]:
    """Regional targets driven through a sparse graph.

    ``anatomy_is_true=False`` is the **null-true** world for G2: the supplied
    "anatomical" topology is an unrelated random graph of the same density, so
    a topology prior carries no information and the gate must FAIL.
    """
    rng = np.random.default_rng(seed)
    N = n_regions
    A_true = (rng.random((N, N)) < density).astype(float)
    np.fill_diagonal(A_true, 1.0)
    W_true = A_true * rng.normal(0, 1.0, size=(N, N))

    if anatomy_is_true:
        A_anat = A_true.copy()
    else:
        A_anat = (rng.random((N, N)) < density).astype(float)
        np.fill_diagonal(A_anat, 1.0)

    # distance-matched control: same density, edges drawn by a distance kernel
    coords = rng.normal(0, 1, size=(N, 3))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    p = np.exp(-d)
    thresh = np.quantile(p, 1.0 - float(A_true.mean()))
    A_dist = (p >= thresh).astype(float)
    A_rand = A_true.copy().ravel()
    rng.shuffle(A_rand)
    A_rand = A_rand.reshape(N, N)
    A_dense = np.ones((N, N))

    def _gen(n: int, site: str, shift: float = 0.0) -> Dataset:
        X = rng.normal(0, 1, size=(n, N)) + shift
        Y = X @ W_true.T + rng.normal(0, noise, size=(n, N))
        return Dataset(
            name=f"graph.{site}",
            targets=Y,
            inputs={"x": X},
            strata={"site": np.array([site] * n), "device": np.array([site + "-dev"] * n)},
            groups=np.array([f"{site}-P{i:04d}" for i in range(n)]),
        )

    train = _gen(n_train, "train")
    test = _gen(n_test, "test")
    ood = _gen(n_test, "ood", shift=1.5)  # out-of-distribution mean shift
    return {
        "train": train,
        "test": test,
        "ood": ood,
        "A_true": A_true,
        "anatomy": A_anat,
        "controls": {"dense": A_dense, "randomized": A_rand, "distance_matched": A_dist},
        "anatomy_is_true": anatomy_is_true,
    }


def make_multiscale_dataset(
    *,
    n_train: int = 300,
    n_test: int = 300,
    n_coarse: int = 6,
    per_parcel: int = 4,
    seed: int = 0,
    fine_structure: bool = True,
) -> dict[str, Any]:
    """Coarse parcels with genuine (or absent) fine sub-parcel structure.

    ``fine_structure=False`` is the null-true world for G3: nothing exists
    below the parcel, so a fine model can only hallucinate high-frequency
    energy, and the gate must FAIL.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    N = n_coarse * per_parcel
    coarse_drive = rng.normal(0, 1, size=(n, n_coarse))
    fine_drive = rng.normal(0, 1, size=(n, N))
    Wc = rng.normal(0, 1, size=(N, n_coarse))
    coarse_part = coarse_drive @ Wc.T
    if fine_structure:
        Wf = rng.normal(0, 1.0, size=(N, N)) * (rng.random((N, N)) < 0.15)
        fine_part = fine_drive @ Wf.T
    else:
        fine_part = np.zeros((n, N))
    Y_fine = coarse_part + fine_part + rng.normal(0, 0.4, size=(n, N))
    # coarse observable = parcel average (the restriction map)
    R = np.zeros((n_coarse, N))
    for c in range(n_coarse):
        R[c, c * per_parcel : (c + 1) * per_parcel] = 1.0 / per_parcel
    Y_coarse = Y_fine @ R.T

    # contiguous group blocks so that the train/test cut is also a group cut:
    # the harness refuses to score a model on a group it was fit on.
    groups = np.array([f"S{(i * 20) // n:02d}" for i in range(n)])
    tr = np.arange(n_train)
    te = np.arange(n_train, n)

    def _mk(idx, nm, targets):
        return Dataset(
            name=nm,
            targets=targets[idx],
            inputs={"coarse_evidence": coarse_drive[idx], "fine_evidence": fine_drive[idx]},
            strata={"site": np.array(["s0"] * len(idx))},
            groups=groups[idx],
        )

    return {
        "fine_train": _mk(tr, "fine.train", Y_fine),
        "fine_test": _mk(te, "fine.test", Y_fine),
        "coarse_train": _mk(tr, "coarse.train", Y_coarse),
        "coarse_test": _mk(te, "coarse.test", Y_coarse),
        "restriction": R,
        "fine_structure": fine_structure,
        "per_parcel": per_parcel,
        "n_coarse": n_coarse,
    }


def make_individualization_dataset(
    *,
    n_subjects: int = 16,
    n_sessions: int = 3,
    n_per_session: int = 60,
    seed: int = 0,
    individual_effect: bool = True,
    anatomy_predicts: bool = False,
) -> dict[str, Any]:
    """Subjects x sessions x tasks, with or without a real individual effect.

    ``individual_effect=False`` is the null-true world for G5: subjects differ
    only by noise, so a personalized model cannot beat a population model and
    the gate must FAIL.  ``anatomy_predicts=True`` builds the trap named in
    the thesis — the person's *scan* is informative, which is **not**
    personalization; G5 must still fail unless the personalized model beats
    the anatomy-only baseline.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    sess_eye = np.eye(n_sessions)
    subj_effect = rng.normal(0, 1.0, size=n_subjects)
    anat = rng.normal(0, 1.0, size=(n_subjects, 4))
    anat_w = rng.normal(0, 1.0, size=4)
    for s in range(n_subjects):
        for sess in range(n_sessions):
            sess_effect = rng.normal(0, 0.4)
            for t in range(n_per_session):
                task = f"task{t % 3}"
                x = rng.normal(0, 1, size=3)
                y = float(x @ np.array([1.0, -0.5, 0.3]))
                y += sess_effect
                if individual_effect:
                    y += 1.5 * subj_effect[s] + 0.8 * subj_effect[s] * x[0]
                if anatomy_predicts:
                    y += 1.2 * float(anat[s] @ anat_w)
                y += rng.normal(0, 0.5)
                rows.append(
                    {
                        "subject": f"S{s:02d}",
                        "session": sess,
                        "task": task,
                        "x": x,
                        "anat": anat[s],
                        "subj_onehot": np.eye(n_subjects)[s],
                        "sess_onehot": sess_eye[sess],
                        "y": y,
                    }
                )

    def _mk(pred, nm: str) -> Dataset:
        sel = [r for r in rows if pred(r)]
        return Dataset(
            name=nm,
            targets=np.array([r["y"] for r in sel]),
            inputs={
                "x": np.array([r["x"] for r in sel]),
                "anat": np.array([r["anat"] for r in sel]),
                "subject_id": np.array([r["subj_onehot"] for r in sel]),
                "session_id": np.array([r["sess_onehot"] for r in sel]),
            },
            strata={
                "session": np.array([str(r["session"]) for r in sel]),
                "task": np.array([r["task"] for r in sel]),
                "site": np.array(["site0"] * len(sel)),
            },
            groups=np.array([r["subject"] for r in sel]),
        )

    return {
        # same subjects, earlier sessions -> later session (new session holdout)
        "train": _mk(lambda r: r["session"] < n_sessions - 1 and r["task"] != "task2", "indiv.train"),
        "new_session": _mk(lambda r: r["session"] == n_sessions - 1 and r["task"] != "task2",
                           "indiv.new_session"),
        "unseen_task": _mk(lambda r: r["task"] == "task2", "indiv.unseen_task"),
        "individual_effect": individual_effect,
        "anatomy_predicts": anatomy_predicts,
        "n_subjects": n_subjects,
    }


def make_intervention_dataset(
    *,
    n_train: int = 300,
    n_test: int = 200,
    seed: int = 0,
    intervention_informative: bool = True,
) -> dict[str, Any]:
    """Passive observations plus held-out interventions of varying dose/state.

    Used by G4's prospective-recovery sub-check and by the
    correlation-fitting-versus-perturbational-prediction ablation.  With
    ``intervention_informative=False`` the interventions carry no directional
    information and any "causal" claim must fail.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = rng.normal(0, 1, size=(n, 3))
    state = rng.integers(0, 2, size=n)          # state dependence
    dose = rng.uniform(0.0, 2.0, size=n)        # dose response
    gain_true, delay_true = 0.9, 0.25
    if intervention_informative:
        resp = gain_true * dose * (1.0 + 0.6 * state) + 0.3 * x[:, 0]
    else:
        resp = 0.3 * x[:, 0] + 0.0 * dose
    y = resp + rng.normal(0, 0.3, size=n)
    groups = np.array([f"P{i % 10:02d}" for i in range(n)])

    def _mk(idx, nm):
        return Dataset(
            name=nm,
            targets=y[idx],
            inputs={
                "x": x[idx],
                "dose": dose[idx][:, None],
                "state": state[idx][:, None].astype(float),
            },
            strata={"state": state[idx].astype(str), "site": np.array(["s0"] * len(idx))},
            groups=groups[idx],
        )

    tr = np.where(np.arange(n) < n_train)[0]
    te = np.where(np.arange(n) >= n_train)[0]
    return {
        "train": _mk(tr, "intervention.train"),
        "test": _mk(te, "intervention.heldout"),
        "truth": {"gain": gain_true, "delay": delay_true,
                  "direction": 1.0 if intervention_informative else 0.0},
        "intervention_informative": intervention_informative,
    }


# ==========================================================================
# Fisher fixture (G4 tests only -- agent H owns the real thing)
# ==========================================================================
@dataclass
class SyntheticFisher:
    """A stand-in Fisher-information map over designs, for gate tests only.

    Parameter vector is ``theta`` (the scientific parameters: coupling, delay,
    gain) followed by ``nuisance`` (lead field / hemodynamic / E-field model
    parameters).  ``nuisance_only_gain=True`` builds the falsifying world of
    G4: adding the intervention increases information **only in the field-model
    nuisance block**, which the thesis names explicitly as a failure.
    """

    n_theta: int = 3
    n_nuisance: int = 3
    nuisance_only_gain: bool = False
    seed: int = 0

    def __call__(self, design: str) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        p = self.n_theta + self.n_nuisance
        base = np.zeros((p, p))
        # EEG alone: informs theta[0], theta[1] and lead-field nuisance
        eeg = np.diag(np.array([1.0, 0.6, 0.0] + [0.8, 0.0, 0.0])[:p])
        # fMRI alone: informs theta[0] and hemodynamic nuisance
        fmri = np.diag(np.array([0.7, 0.0, 0.0] + [0.0, 0.9, 0.0])[:p])
        joint_native = eeg + fmri
        joint_resampled = 0.55 * joint_native  # information destroyed by resampling
        if self.nuisance_only_gain:
            impulse = np.diag(np.array([0.0, 0.0, 0.0] + [0.0, 0.0, 2.5])[:p])
        else:
            impulse = np.diag(np.array([0.4, 0.5, 1.4] + [0.0, 0.0, 0.6])[:p])
        table = {
            "prior": np.eye(p) * 1e-3,
            "eeg": eeg,
            "fmri": fmri,
            "joint_native": joint_native,
            "joint_resampled": joint_resampled,
            "joint_plus_impulse": joint_native + impulse,
        }
        if design not in table:
            raise KeyError(f"unknown design {design!r}; have {sorted(table)}")
        M = table[design] + base
        # keep it a valid information matrix
        return 0.5 * (M + M.T)

    @property
    def theta_index(self) -> np.ndarray:
        return np.arange(self.n_theta)

    @property
    def nuisance_index(self) -> np.ndarray:
        return np.arange(self.n_theta, self.n_theta + self.n_nuisance)
