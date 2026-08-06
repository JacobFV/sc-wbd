"""The path from a trained checkpoint to a typed prediction.

Why this module exists
----------------------
Until 2026-08-06 ``scwbd.runtime`` contained no ``torch.load``. It hashed
checkpoint files and reported their sha256; it never opened them. Every
consumer-facing number came from a closed-form field model and prior-specified
surrogates, and ``warm_up()`` returned byte-identical numbers whether the
service was backed by nothing, by the run-1 artifact, or by the declared
control arm. ``weights_status`` was a label attached to numbers it could not
affect -- the same defect as ``discover_checkpoint`` scanning for a filename
the trainer never writes, one level up: a runtime that was green because it
never loaded a model.

The test that closes it is the one that was failing implicitly the whole time:
**load two different checkpoints, get different numbers.**
``tests/runtime/test_prediction_path.py`` asserts exactly that against real
artifacts on disk.

What loading a checkpoint honestly requires
-------------------------------------------
Three things went wrong on the first attempt, each of which would have loaded
"successfully" and produced confident wrong numbers:

1. **The anatomy moved.** The live prior now returns 414 regions; the run-1
   checkpoint was trained at 454 on the synthetic fallback. Rebuilding with
   today's default anatomy raises on thirteen tensors -- which is lucky, because
   a prior that had merely *changed* rather than changed shape would have
   loaded silently. :func:`rebuild_anatomy` reconstructs the prior the
   checkpoint records, and refuses if it cannot reproduce the region count.

2. **``torch.compile`` renamed 29 parameters.** The run was configured
   ``compile: true``, so ``local`` and ``residual`` were wrapped and their state
   keys carry an ``_orig_mod.`` infix. Loaded naively with ``strict=False`` that
   is 29 silently-uninitialised tensors and a model that runs. We strip the
   prefix and *count* what we stripped.

3. **Config fields that did not exist when the checkpoint was written.**
   ``state_dependent_variance`` defaults ``True`` today; run 1 predates it.
   Taking today's default builds an architecture the checkpoint cannot fill, so
   the uncertainty modules stay at their random initialisation. We do **not**
   guess: absent boolean fields are *inferred from the state dict itself* by
   asking whether the parameters they gate are present, and every inference is
   recorded in :attr:`CheckpointLoadReport.inferred_config`.

``strict=False`` plus a discarded load report is a standing entry in
``reports/decorative_guards.md``. So :class:`CheckpointLoadReport` is returned,
carried in the prediction, and -- crucially -- any output whose parameters were
**not** restored comes back as :class:`~scwbd.runtime._compat.Unresolved`
rather than as a number. An uninitialised head does not produce a slightly
worse prediction; it produces a random one, and a random number with units on
it is the least honest thing this repository could emit.

Claim limits
------------
A prediction from this module is a simulation. The run-1 artifact is the
equal-capacity generic-operator control arm of ``body.tex`` Sec. 11.4's first
ablation, its anatomy is synthetic, and it is beaten on held-out real EEG by
all five baselines. Those facts are carried as labels by
:mod:`scwbd.runtime.admission`; nothing here revokes them. This module makes
the artifact *runnable*, not *right*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from ._compat import Unresolved
from .ports import PortContract, PortedState

__all__ = [
    "CheckpointLoadError",
    "CheckpointLoadReport",
    "Prediction",
    "LoadedModel",
    "rebuild_anatomy",
]

#: The ``torch.compile`` wrapper infix.  Present in any checkpoint saved from a
#: run configured ``compile: true``.
_COMPILE_INFIX = "._orig_mod."

#: Config fields whose presence is decidable from the state dict: field name ->
#: a parameter that exists **iff** the field is true.
_FIELD_EVIDENCE: Mapping[str, str] = {
    "state_dependent_variance": "observation.head.w",
    "family_state": "family_local.ports.proj.weight",
}

#: Which prediction each parameter prefix backs.  Used to turn "these tensors
#: were not restored" into "this output is Unresolved" rather than into a
#: warning nobody reads.
#:
#: The mean and the variance of an output are listed **separately**, because
#: they are separately parameterised in ``heads.py`` and an uninitialised
#: variance parameter says nothing about the mean. Conflating them would mark
#: a perfectly good EEG mean unresolved because a log-variance mixer was
#: absent -- a refusal that is wrong in the safe direction, which is still
#: wrong, and which would have made this whole module look broken on the only
#: checkpoint we have.
_OUTPUT_PARAMS: Mapping[str, tuple[str, ...]] = {
    "activity_logvar": ("observation.", "uncertainty_propagator."),
    "eeg": ("eeg.source_proj", "eeg.log_gain", "eeg.offset", "eeg.nuisance"),
    "eeg_logvar": ("observation.", "uncertainty_propagator.", "eeg.logvar_mix",
                   "eeg.log_noise"),
    "hemodynamic": ("bold.log_", "bold.alpha", "bold.rho", "bold.neural_gain"),
}

#: Parameters that exist on the module but are **never read** under a given
#: config.  ``heads.py`` allocates ``logvar_mix`` and ``logvar_gain``
#: unconditionally but only consults them when the observation module is
#: present, so under ``state_dependent_variance=False`` they are inert. An
#: inert tensor at its random initialisation cannot affect any number, and
#: reporting it as consequential would be a false alarm -- which is how a
#: warning channel stops being read.
_INERT_WITHOUT: Mapping[str, tuple[str, ...]] = {
    "state_dependent_variance": ("eeg.logvar_mix", "bold.logvar_gain"),
}


class CheckpointLoadError(RuntimeError):
    """A checkpoint that cannot be reconstructed into a runnable model."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    """What was actually restored, and what was not.

    Returned and carried rather than discarded.  ``clean`` is ``True`` only
    when every parameter of the reconstructed model came from the checkpoint.
    """

    restored: int
    #: Parameters the model has that the checkpoint did not supply.  These are
    #: at their random initialisation and any output depending on them is
    #: :class:`Unresolved`.
    uninitialised: tuple[str, ...] = ()
    #: Parameters the checkpoint carried that the model has no slot for.
    ignored: tuple[str, ...] = ()
    #: Keys whose ``torch.compile`` infix was stripped to match.
    renamed: tuple[str, ...] = ()
    #: Config fields absent from the checkpoint, and the value inferred for
    #: each from the state dict.  Never a silent default.
    inferred_config: Mapping[str, Any] = field(default_factory=dict)
    anatomy: Mapping[str, Any] = field(default_factory=dict)
    #: Uninitialised parameters that no output reads under this config.  Kept
    #: visible rather than dropped: "inert" is a claim about the config, and a
    #: reader should be able to check it.
    inert: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """Every parameter restored, ignoring provably-unread ones."""
        return not self.consequential and not self.ignored

    @property
    def consequential(self) -> tuple[str, ...]:
        """Uninitialised parameters that some output actually reads."""
        return tuple(k for k in self.uninitialised if k not in set(self.inert))

    def unresolved_outputs(self) -> tuple[str, ...]:
        """Outputs that cannot be produced because their parameters are random."""
        bad = self.consequential
        out = []
        for name, prefixes in _OUTPUT_PARAMS.items():
            if any(k.startswith(p) for k in bad for p in prefixes):
                out.append(name)
        return tuple(sorted(out))

    def summary(self) -> str:
        bits = [f"{self.restored} tensors restored"]
        if self.renamed:
            bits.append(f"{len(self.renamed)} torch.compile keys renamed")
        if self.consequential:
            bits.append(f"{len(self.consequential)} UNINITIALISED")
        if self.inert:
            bits.append(f"{len(self.inert)} inert under this config")
        if self.ignored:
            bits.append(f"{len(self.ignored)} ignored")
        if self.inferred_config:
            bits.append(f"inferred config {dict(self.inferred_config)}")
        return "; ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "restored": self.restored,
            "clean": self.clean,
            "uninitialised": list(self.uninitialised),
            "consequential": list(self.consequential),
            "inert": list(self.inert),
            "ignored": list(self.ignored),
            "renamed": list(self.renamed),
            "inferred_config": dict(self.inferred_config),
            "anatomy": dict(self.anatomy),
            "unresolved_outputs": list(self.unresolved_outputs()),
        }


def rebuild_anatomy(record: Mapping[str, Any], *, device: str = "cpu") -> Any:
    """Rebuild the anatomy prior a checkpoint was trained against.

    The prior on disk changes as real atlases land.  A checkpoint trained
    against the synthetic fallback at 454 regions cannot be reconstructed
    against today's 414-region biological prior, and the failure is only
    *loud* because the region count happens to differ -- a prior that changed
    content but not shape would load in silence.  So this refuses on any
    mismatch rather than trusting the shapes to catch it.
    """
    from scwbd.foundation.anatomy import load_anatomy

    want = int(record.get("n_regions", 0))
    anat = load_anatomy(
        device=device,
        n_cortex=int(record.get("n_cortex", 400)),
        n_subcortex=int(record.get("n_subcortex", 32)),
        n_cerebellum=int(record.get("n_cerebellum", 22)),
        density=float(record.get("density", 0.10)),
        # The checkpoint records which prior it had. Reproduce that one.
        force_fallback=str(record.get("provenance", "")) == "synthetic_fallback",
    )
    if want and anat.n_regions != want:
        raise CheckpointLoadError(
            f"checkpoint was trained against {want} regions "
            f"({record.get('provenance')!r}) but the prior in this checkout "
            f"builds {anat.n_regions}. The anatomy has moved underneath the "
            "artifact; reconstructing it against a different parcellation "
            "would produce numbers about a different brain"
        )
    return anat


def _strip_compile_infix(
    state: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    out: dict[str, Tensor] = {}
    renamed: list[str] = []
    for k, v in state.items():
        if _COMPILE_INFIX in k:
            nk = k.replace(_COMPILE_INFIX, ".")
            renamed.append(k)
            out[nk] = v
        else:
            out[k] = v
    return out, tuple(sorted(renamed))


@dataclass(frozen=True)
class Prediction:
    """Typed model outputs, with the load report that produced them.

    Every field is either a tensor or :class:`Unresolved`.  There is no third
    option and in particular no zero: an output whose parameters were not
    restored is a read this artifact cannot support, and saying so is the whole
    point of carrying the load report this far.
    """

    #: ``(B, T, N)`` predicted regional activity mean.
    activity: Tensor
    #: ``(B, T, N)`` predictive log-variance, or Unresolved.
    activity_logvar: Tensor | Any
    #: ``(B, T, C)`` predicted scalp potential, or Unresolved.
    eeg: Tensor | Any
    eeg_logvar: Tensor | Any
    #: ``(B, T_slow, N, 4)`` Balloon-Windkessel compartments, or Unresolved.
    hemodynamic: Tensor | Any
    #: The rolled state, readable only through declared ports.
    state: PortedState
    #: Mean ``||R|| / ||F_local + F_long||`` -- the R05 residual dominance ratio.
    residual_ratio: float
    load_report: CheckpointLoadReport
    notes: Mapping[str, Any] = field(default_factory=dict)

    def resolved(self) -> tuple[str, ...]:
        return tuple(
            n for n in ("activity", "activity_logvar", "eeg", "eeg_logvar",
                        "hemodynamic")
            if not isinstance(getattr(self, n), Unresolved)
        )

    def unresolved(self) -> dict[str, str]:
        return {
            n: getattr(self, n).reason
            for n in ("activity", "activity_logvar", "eeg", "eeg_logvar",
                      "hemodynamic")
            if isinstance(getattr(self, n), Unresolved)
        }


class LoadedModel:
    """A checkpoint reconstructed into a model that actually runs.

    Not a dataclass: it owns an ``nn.Module`` and must not be compared,
    hashed, or copied by value.
    """

    __slots__ = ("_model", "_cfg", "_anat", "_report", "_extra", "_path", "_contract")

    def __init__(self, model: Any, cfg: Any, anat: Any, report: CheckpointLoadReport,
                 extra: Mapping[str, Any], path: Path) -> None:
        self._model = model
        self._cfg = cfg
        self._anat = anat
        self._report = report
        self._extra = dict(extra)
        self._path = path
        self._contract: PortContract | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
        require_clean: bool = False,
    ) -> "LoadedModel":
        """Reconstruct and load.  ``require_clean`` refuses a partial restore.

        The default is **not** to refuse: a partially-restored model is usable
        for the outputs whose parameters *were* restored, and those outputs are
        the ones a consumer actually wants. What it may not do is emit the
        others as numbers, and :class:`Prediction` enforces that.
        """
        from scwbd.foundation.config import ModelConfig
        from scwbd.foundation.model import SCWBD

        p = Path(path)
        if not p.is_file():
            raise CheckpointLoadError(f"no checkpoint at {p}")
        payload = torch.load(str(p), map_location=device, weights_only=False)
        fmt = payload.get("format")
        if fmt != "scwbd-foundation-checkpoint/1":
            raise CheckpointLoadError(
                f"unrecognised checkpoint format {fmt!r} at {p}"
            )

        raw_state, renamed = _strip_compile_infix(payload["model"])

        stored_cfg = dict(payload.get("config", {}).get("model") or {})
        known = set(ModelConfig.__dataclass_fields__)
        kept = {k: v for k, v in stored_cfg.items() if k in known}

        # Fields that did not exist when this checkpoint was written. Infer
        # each from the state dict rather than taking today's default, and
        # record the inference.
        inferred: dict[str, Any] = {}
        for fname, evidence in _FIELD_EVIDENCE.items():
            if fname in stored_cfg or fname not in known:
                continue
            inferred[fname] = evidence in raw_state
            kept[fname] = inferred[fname]

        cfg = ModelConfig(**kept)
        anat_rec = dict((payload.get("extra") or {}).get("anatomy") or {})
        anat = rebuild_anatomy(anat_rec, device=device)

        model = SCWBD(cfg, anat).to(device)
        missing, unexpected = model.load_state_dict(raw_state, strict=False)
        inert = tuple(sorted(
            k for flag, names in _INERT_WITHOUT.items()
            if not getattr(cfg, flag, True)
            for k in names if k in set(missing)
        ))
        report = CheckpointLoadReport(
            restored=len(raw_state) - len(unexpected),
            uninitialised=tuple(sorted(missing)),
            ignored=tuple(sorted(unexpected)),
            renamed=renamed,
            inferred_config=inferred,
            inert=inert,
            anatomy={
                "n_regions": anat.n_regions,
                "provenance": anat_rec.get("provenance", "unstated"),
                "is_biological": bool(anat.is_biological()),
            },
        )
        if require_clean and not report.clean:
            raise CheckpointLoadError(
                f"checkpoint {p} did not restore cleanly: {report.summary()}. "
                f"Uninitialised: {list(report.uninitialised)[:8]}"
            )
        model.eval()
        return cls(model, cfg, anat, report, payload.get("extra") or {}, p)

    # -- description -------------------------------------------------------
    @property
    def load_report(self) -> CheckpointLoadReport:
        return self._report

    @property
    def n_regions(self) -> int:
        return int(self._cfg.n_regions)

    @property
    def theta_names(self) -> tuple[str, ...]:
        return tuple(self._extra.get("theta_names") or ())

    def port_contract(self) -> PortContract:
        """The declared ports of this model's *actual* state layout.

        Built from the live model rather than from the sidecar, so it cannot
        disagree with what ``predict`` returns.
        """
        if self._contract is None:
            lay = self._model.layout
            self._contract = PortContract.from_state_layout(
                lay.as_dict() if hasattr(lay, "as_dict") else lay,
                source=f"loaded_model:{self._path.name}",
            )
        return self._contract

    def default_theta(self) -> Tensor:
        """Midpoint of the checkpoint's own recorded prior over theta."""
        prior = self._extra.get("theta_prior") or {}
        names = self.theta_names
        if not names or not prior:
            raise CheckpointLoadError(
                "this checkpoint records no theta_prior, so there is no "
                "defensible default conditioning vector; pass theta= explicitly"
            )
        return torch.tensor(
            [[sum(prior[n]) / 2.0 for n in names]], dtype=torch.float32
        )

    # -- the forward path --------------------------------------------------
    def predict(
        self,
        context: Tensor,
        *,
        theta: Tensor | None = None,
        n_steps: int = 16,
        context_mask: Tensor | None = None,
        with_hemo: bool = True,
    ) -> Prediction:
        """Assimilate ``context`` and roll forward ``n_steps``.

        ``context`` is ``(B, T, N)`` **regional activity**, not sensor data:
        that is what ``Assimilator.forward`` consumes. Projecting sensors to
        parcels is a separate operation with its own claim limits and this
        module does not do it silently.
        """
        if context.dim() != 3:
            raise ValueError(
                f"context must be (B, T, N) regional activity, got "
                f"{tuple(context.shape)}"
            )
        if context.shape[-1] != self.n_regions:
            raise ValueError(
                f"context has {context.shape[-1]} regions but this checkpoint "
                f"models {self.n_regions}. This is the mismatch that must never "
                "be papered over: a differently-parcellated context is a "
                "context about a different brain"
            )
        th = self.default_theta() if theta is None else theta
        if th.shape[0] != context.shape[0]:
            th = th.expand(context.shape[0], -1)

        with torch.no_grad():
            out = self._model.rollout(
                y_context=context,
                theta=th,
                n_steps=int(n_steps),
                context_mask=context_mask,
                with_hemo=with_hemo,
            )
            # The heads run inside the same no_grad block. Outside it they
            # return graph-carrying tensors, and a consumer that keeps one
            # keeps the whole rollout graph alive -- a leak that shows up as
            # memory, not as a wrong number, which is why it is easy to miss.
            try:
                eeg, eeg_lv = self._model.eeg(out.state)
            except Exception as exc:  # pragma: no cover - head shape drift
                eeg = eeg_lv = Unresolved(reason=f"EEG head raised: {exc}")

        unresolved = set(self._report.unresolved_outputs())

        def gate(name: str, value: Any) -> Any:
            if name not in unresolved:
                return value
            bad = tuple(
                k for k in self._report.consequential
                if any(k.startswith(p) for p in _OUTPUT_PARAMS[name])
            )
            return Unresolved(
                reason=(
                    f"{name} depends on {len(bad)} parameter(s) this checkpoint "
                    "did not supply, which are therefore at their random "
                    "initialisation. A number from them would be noise with "
                    "units on it"
                ),
                missing=bad,
            )

        return Prediction(
            activity=out.activity,
            activity_logvar=gate("activity_logvar", out.activity_logvar),
            eeg=gate("eeg", eeg),
            eeg_logvar=gate("eeg_logvar", eeg_lv),
            hemodynamic=gate("hemodynamic", (
                out.hemo if out.hemo is not None
                else Unresolved(
                    reason=(
                        "no slow-clock sample was produced: the haemodynamic "
                        f"compartments advance every hemo_ratio="
                        f"{self._cfg.hemo_ratio} fast steps and this rollout ran "
                        f"{int(n_steps)}"
                        if with_hemo else
                        "rollout was run with with_hemo=False"
                    ),
                    missing=("bold",),
                )
            )),
            state=PortedState(self.port_contract(), out.state),
            residual_ratio=float(out.rho),
            load_report=self._report,
            notes={
                "checkpoint": str(self._path),
                "n_steps": int(n_steps),
                "theta": th[0].tolist(),
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LoadedModel({self._path.name}, N={self.n_regions}, "
            f"{self._report.summary()})"
        )
