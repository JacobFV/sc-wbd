"""Minimal model/data harness the gates measure through (agent J).

This module deliberately contains **no brain model**.  It defines the thin
protocol every arm of a gate or ablation must satisfy, so that agent E's
dynamics, agent I's foundation model, and a two-line ridge baseline are all
scored by exactly the same code path.

A candidate that cannot report its parameter count cannot enter a matched
comparison, and a candidate that cannot report predictive *uncertainty*
cannot be scored at all: :class:`Prediction` requires ``sd``.  Both refusals
are intentional — the thesis's headline metrics are calibrated log scores, and
a point prediction has no log score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .report import Interval, Metric
from .statistics import (
    CalibrationReport,
    bootstrap_ci,
    calibration,
    gaussian_log_score,
    stratified_bias,
)

__all__ = [
    "Dataset",
    "Prediction",
    "BenchModel",
    "Arm",
    "EvalResult",
    "evaluate",
    "as_factory",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    """A scored evaluation set.

    ``inputs`` is a dict of named blocks so that a *typed* model can consume
    ``{"eeg": ..., "bold": ...}`` on their native clocks while a naive
    resampling baseline consumes a single pre-resampled block.  Nothing here
    resamples anything: the blocks are whatever the source card produced.

    ``strata`` carries the §11.2 bias factors (session, device, site, anatomy,
    demographic stratum, task context).  ``groups`` carries the immutable
    participant/family group key per observation so that a scoring routine can
    check it never scores a group it trained on.
    """

    name: str
    targets: np.ndarray
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    strata: dict[str, np.ndarray] = field(default_factory=dict)
    groups: np.ndarray | None = None
    #: per-observation record id (links to scwbd.sources lineage records)
    record_ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targets = np.asarray(self.targets, dtype=float)
        n = self.n
        for k, v in list(self.inputs.items()):
            arr = np.asarray(v, dtype=float)
            if arr.shape[0] != n:
                raise ValueError(
                    f"input block {k!r} has {arr.shape[0]} rows, targets have {n}"
                )
            self.inputs[k] = arr
        for k, v in list(self.strata.items()):
            arr = np.asarray(list(v))
            if arr.shape[0] != n:
                raise ValueError(f"stratum {k!r} has {arr.shape[0]} labels, targets have {n}")
            self.strata[k] = arr
        if self.groups is not None:
            self.groups = np.asarray(list(self.groups))
            if self.groups.shape[0] != n:
                raise ValueError("groups length does not match targets")

    @property
    def n(self) -> int:
        return int(self.targets.shape[0])

    @property
    def target_dim(self) -> int:
        return 1 if self.targets.ndim == 1 else int(self.targets.shape[1])

    def subset(self, idx: np.ndarray | Sequence[int], *, name: str | None = None) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset(
            name=name or f"{self.name}[{len(idx)}]",
            targets=self.targets[idx],
            inputs={k: v[idx] for k, v in self.inputs.items()},
            strata={k: v[idx] for k, v in self.strata.items()},
            groups=None if self.groups is None else self.groups[idx],
            record_ids=None if self.record_ids is None else self.record_ids[idx],
            meta=dict(self.meta),
        )

    def without(self, *blocks: str, name: str | None = None) -> "Dataset":
        """Drop input blocks (evidence withholding: G3, leakage audits)."""
        return replace(
            self,
            name=name or f"{self.name}-without({','.join(blocks)})",
            inputs={k: v for k, v in self.inputs.items() if k not in blocks},
        )

    def group_ids(self) -> np.ndarray:
        if self.groups is None:
            return np.arange(self.n)
        return self.groups


# --------------------------------------------------------------------------
# predictions and models
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    """Predictive mean **and** sd.  A point estimate is not a prediction."""

    mean: np.ndarray
    sd: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.sd = np.asarray(self.sd, dtype=float)
        if self.mean.shape != self.sd.shape:
            raise ValueError(f"mean {self.mean.shape} and sd {self.sd.shape} disagree")
        if np.any(self.sd <= 0):
            raise ValueError("predictive sd must be strictly positive")


@runtime_checkable
class BenchModel(Protocol):
    """What every arm must provide.  Optional hooks are duck-typed."""

    name: str

    def fit(self, data: Dataset, *, seed: int = 0) -> "BenchModel": ...

    def predict(self, data: Dataset) -> Prediction: ...

    def n_parameters(self) -> int: ...


ModelFactory = Callable[[], Any]


def as_factory(model_or_factory: Any) -> ModelFactory:
    """Accept either an instance or a zero-arg factory; always return a factory.

    Instances are only acceptable when they can be re-fit; a gate that needs
    several fits (data-efficiency curves, seed repeats) should be given a
    factory so that state cannot leak between fits.
    """
    if callable(model_or_factory) and not hasattr(model_or_factory, "predict"):
        return model_or_factory  # type: ignore[return-value]
    return lambda: model_or_factory


@dataclass
class Arm:
    """One comparison arm: a named model factory plus its declared role."""

    name: str
    factory: ModelFactory
    role: str = "baseline"
    note: str = ""

    @classmethod
    def of(cls, name: str, model_or_factory: Any, role: str = "baseline", note: str = "") -> "Arm":
        return cls(name=name, factory=as_factory(model_or_factory), role=role, note=note)

    def build(self) -> Any:
        return self.factory()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    arm: str
    dataset: str
    n: int
    log_score: np.ndarray            # per-observation, higher is better
    prediction: Prediction
    targets: np.ndarray
    calibration: CalibrationReport
    rmse: float
    n_parameters: int | None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_log_score(self) -> float:
        return float(np.mean(self.log_score))

    def log_score_metric(self, *, threshold: float | None = None, seed: int = 0) -> Metric:
        pt, iv = bootstrap_ci(self.log_score, seed=seed)
        return Metric(
            name=f"{self.arm}.heldout_log_score",
            value=pt,
            units="nats/obs",
            kind="accuracy",
            interval=iv,
            threshold=threshold,
            direction="greater_is_better",
        )

    def metrics(self, *, seed: int = 0) -> list[Metric]:
        return [self.log_score_metric(seed=seed)] + self.calibration.metrics(prefix=self.arm)


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def evaluate(
    model_or_arm: Any,
    train: Dataset | None,
    test: Dataset,
    *,
    seed: int = 0,
    refuse_group_overlap: bool = True,
) -> EvalResult:
    """Fit (optionally) and score one arm on one held-out set.

    When both datasets carry ``groups``, an overlap between train and test
    groups raises: scoring a model on a participant it was fit on is not a
    held-out score, and this harness will not silently produce one.
    """
    arm_name = getattr(model_or_arm, "name", "model")
    if isinstance(model_or_arm, Arm):
        arm_name = model_or_arm.name
        model = model_or_arm.build()
    else:
        model = model_or_arm

    if (
        refuse_group_overlap
        and train is not None
        and train.groups is not None
        and test.groups is not None
    ):
        overlap = set(np.unique(train.groups).tolist()) & set(np.unique(test.groups).tolist())
        if overlap:
            raise ValueError(
                f"train/test group overlap for arm {arm_name!r}: {sorted(overlap)[:5]} — "
                "this is refusal R10 territory; group before you split"
            )

    if train is not None and hasattr(model, "fit"):
        model.fit(train, seed=seed)
    pred = model.predict(test)
    y = test.targets
    if pred.mean.shape != y.shape:
        if pred.mean.ravel().shape == y.ravel().shape:
            pred = Prediction(pred.mean.reshape(y.shape), pred.sd.reshape(y.shape), pred.extras)
        else:
            raise ValueError(
                f"arm {arm_name!r} predicted {pred.mean.shape} for targets {y.shape}"
            )
    ls = gaussian_log_score(_flat(y), _flat(pred.mean), _flat(pred.sd))
    # aggregate to per-observation scores when targets are multivariate
    if y.ndim > 1:
        ls = ls.reshape(y.shape[0], -1).sum(axis=1)
    cal = calibration(_flat(y), _flat(pred.mean), _flat(pred.sd), seed=seed)
    rmse = float(np.sqrt(np.mean((_flat(pred.mean) - _flat(y)) ** 2)))
    npar: int | None
    try:
        npar = int(model.n_parameters())
    except Exception:
        npar = None
    return EvalResult(
        arm=arm_name,
        dataset=test.name,
        n=test.n,
        log_score=ls,
        prediction=pred,
        targets=y,
        calibration=cal,
        rmse=rmse,
        n_parameters=npar,
        extras=dict(pred.extras),
    )


def bias_of(result: EvalResult, test: Dataset, *, seed: int = 0):
    """Convenience: §11.2 stratified bias analysis for an evaluation."""
    return stratified_bias(
        _flat(result.targets), _flat(result.prediction.mean), test.strata, seed=seed
    )
