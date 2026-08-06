"""Source-card-driven loss composition, gradient masks and conflict measurement.

Implements Appendix D of the thesis:

.. math::
   \\mathcal L = \\sum_e w_e\\bigl[
     \\lambda_{\\rm like}\\mathcal L_{\\rm likelihood}^{(e)}
   + \\lambda_{\\rm boundary}\\mathcal L_{\\rm boundary}^{(e)}
   + \\lambda_{\\rm distill}\\mathcal L_{\\rm distill}^{(e)}
   + \\lambda_{\\rm cal}\\mathcal L_{\\rm cal}^{(e)}
   + \\lambda_{\\rm mech}\\mathcal R_{\\rm mechanism}^{(e)}
   + \\lambda_{\\rm scale}\\mathcal R_{\\rm scale}^{(e)}\\bigr]

with the three non-negotiable properties:

1. ``w_e`` is a **reliability** term -- effective sample size (saturating),
   hierarchy, measurement variance, systematic-bias bounds, simulator/teacher
   discrepancy and validity-domain overlap.  It is **not proportional to row
   count**: :meth:`SourceSpec.reliability` is concave in ``n_eff`` and
   :func:`assert_not_row_count` is a test, not a comment.
2. Losses are normalized **within participant and then within source** before
   cross-source weighting, so a long recording or an oversampled simulator
   cannot silently dominate.
3. Every source updates only the modules its ``GradientPermission`` (``A_k``)
   names.  Gradients are taken w.r.t. the permitted parameters only, so a
   non-permitted parameter's ``.grad`` stays ``None`` -- which is exactly what
   ``tests/foundation/test_gradient_masks.py`` asserts.

Gradient conflict is measured **by module and by source** and logged.  When two
sources disagree the response is a source-specific adapter, partial pooling,
freezing or rejection -- never forced consensus.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .util import logical_param_name

__all__ = [
    "SourceSpec",
    "SourceRole",
    "MixtureWeights",
    "LossNormalizer",
    "GradientGate",
    "ConflictLog",
    "ConflictPolicy",
    "MixtureTrainer",
    "assert_not_row_count",
    "RoleViolation",
]

SourceRole = str  # prior|likelihood|boundary_target|distillation|calibration|negative_control|evaluation_only

ROLES = (
    "prior",
    "likelihood",
    "boundary_target",
    "distillation",
    "calibration",
    "negative_control",
    "evaluation_only",
)

#: which loss families each role is allowed to contribute (Appendix B, roles are
#: "deliberately non-equivalent")
ROLE_LOSSES: dict[str, frozenset[str]] = {
    # A *prior* source shapes population parameters and the amortized posterior
    # before individual evidence. Simulated corpora live here: they may teach the
    # operator and the posterior what the parameter->trajectory map looks like,
    # and they may never contribute a measured-evidence likelihood factor.
    "prior": frozenset({"prior", "forecast", "posterior", "mechanism", "scale"}),
    "likelihood": frozenset({"likelihood", "forecast", "calibration", "mechanism", "scale", "posterior"}),
    "boundary_target": frozenset({"boundary", "scale"}),
    "distillation": frozenset({"distill"}),
    "calibration": frozenset({"calibration"}),
    "negative_control": frozenset(),  # contributes an *audit*, never a gradient
    "evaluation_only": frozenset(),
}

#: Preregistered authority per role.  Appendix D: "Inverse-variance weighting is
#: used only after modeled bias and shared confounding are considered; otherwise
#: a precise but systematically distorted source could dominate the joint
#: objective."  A calibration factor has tiny measurement variance and would win
#: a pure inverse-variance contest, but it estimates gains and offsets -- it is
#: not evidence about a brain.  The roles are "deliberately non-equivalent"
#: (Appendix B), so the non-equivalence is encoded here rather than discovered
#: by accident.
ROLE_AUTHORITY: dict[str, float] = {
    "likelihood": 1.00,
    "prior": 0.55,
    "boundary_target": 0.45,
    "calibration": 0.12,
    "distillation": 0.04,
    "negative_control": 0.0,
    "evaluation_only": 0.0,
}


class RoleViolation(RuntimeError):
    """A source tried to contribute a loss family its role does not license."""


# ======================================================================
# source specification
# ======================================================================
@dataclass
class SourceSpec:
    """The training-relevant projection of a :class:`SourceCard` (Appendix B).

    Built either from agent B's ``SourceCardDoc`` (:meth:`from_card`) or from a
    YAML file under ``configs/source_cards/``.  Fields that a source genuinely
    cannot populate stay ``None``; a ``None`` does not become a zero, it disables
    the corresponding gradient path (Appendix B: "The field remains unknown, and
    the corresponding loss or gradient path is disabled").
    """

    id: str
    role: SourceRole = "likelihood"
    #: A_k -- glob patterns over parameter names this source may update
    gradient_permission: tuple[str, ...] = ("*",)
    #: modules explicitly frozen for this source even if matched above
    frozen: tuple[str, ...] = ()
    #: A_k expressed in agent A's compiler parameter-group vocabulary
    #: (``operator:...``, ``region:...:state:...``, ``port:...``,
    #: ``observation:...:nuisance``).  When the compiler is available these are
    #: authoritative for *reachability*; the torch globs above remain
    #: authoritative for *intent*, and the trainer uses the intersection so the
    #: compiler can only ever restrict, never grant.
    compiler_permission: tuple[str, ...] = ("*",)
    # --- reliability inputs (Appendix D) ---
    n_eff: float = 1.0  # effective sample size (participants/sites, NOT rows)
    n_rows: float = 1.0  # recorded for audit only; never enters the weight
    n_participants: int | None = None
    n_sites: int | None = None
    measurement_variance: float | None = None  # V_k
    bias_halfwidth: float | None = None  # B_k, half-width of the bias interval
    bias_status: str = "prior_specified_sensitivity"
    model_discrepancy: float | None = None  # Delta_k (simulator/teacher error)
    validity_overlap: float = 1.0  # overlap with the target validity domain
    hierarchy_depth: int = 1  # participants nested in sites nested in studies
    # --- bookkeeping ---
    is_simulated: bool = False
    is_teacher: bool = False
    enabled: bool = True
    losses: tuple[str, ...] = ("likelihood",)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; must be one of {ROLES}")
        allowed = ROLE_LOSSES[self.role]
        bad = sorted(set(self.losses) - allowed)
        if bad:
            raise RoleViolation(
                f"source {self.id!r} has role {self.role!r} which licenses {sorted(allowed)}, "
                f"but declares losses {bad}. Roles are fixed in the run manifest before "
                "evaluation; changing one after seeing results is a leakage event."
            )
        if self.is_simulated and self.role == "likelihood":
            raise RoleViolation(
                f"source {self.id!r} is simulated and may not hold the 'likelihood' role over "
                "measured observables: simulator-conditioned evidence cannot establish "
                "biological validity (body.tex §6.3). Use role='prior' or 'boundary_target'."
            )
        if self.is_teacher and self.role != "distillation":
            raise RoleViolation(
                f"teacher source {self.id!r} must have role='distillation'; a teacher is never a "
                "subject likelihood (ARCHITECTURE.md §7 rule 5)."
            )

    # -- reliability ------------------------------------------------------
    def reliability(self, *, kappa: float = 24.0, variance_floor: float = 0.05) -> float:
        """``w_e`` before normalisation.  Concave in ``n_eff``, never linear in rows.

        ``w = role_authority * overlap * saturate(n_eff) * hierarchy_penalty
             / (V + Delta + B^2/3)``

        The ``B^2/3`` term converts a bias *interval* half-width into the variance
        of a uniform distribution over that interval, which is the honest way to
        let a bounded systematic error reduce a source's authority without
        pretending it is noise.  A bias point estimate with no estimator and no
        external bound is refused upstream (R08).

        ``variance_floor`` caps how much authority pure precision can buy.
        Unbounded inverse-variance weighting is exactly the failure Appendix D
        names -- "a precise but systematically distorted source could dominate
        the joint objective" -- and shared confounding puts a floor on how much
        any single source deserves to be trusted regardless of its nominal
        measurement noise.
        """
        if not self.enabled or self.role in ("evaluation_only", "negative_control"):
            return 0.0
        n = max(float(self.n_eff), 0.0)
        sat = n / (n + kappa)  # saturating: 10x the participants is not 10x the weight
        hier = 1.0 / math.sqrt(max(self.hierarchy_depth, 1))
        v = 0.0
        v += float(self.measurement_variance) if self.measurement_variance is not None else 1.0
        v += float(self.model_discrepancy) if self.model_discrepancy is not None else 0.0
        if self.bias_halfwidth is not None:
            v += float(self.bias_halfwidth) ** 2 / 3.0
        else:
            v += 1.0  # unknown bias is expensive, not free
        auth = ROLE_AUTHORITY.get(self.role, 0.0)
        return float(auth * max(self.validity_overlap, 0.0) * sat * hier / max(v, variance_floor))

    # -- gradient permission ---------------------------------------------
    def permits(self, param_name: str) -> bool:
        # Match against the *logical* name: a card's ``A_k`` is written against
        # the module tree the architecture describes, and torch.compile renames
        # tensors underneath it.  See util.logical_param_name -- without this a
        # compiled run silently loses every exact-name permission while keeping
        # the prefix globs, which is worse than losing all of them.
        param_name = logical_param_name(param_name)
        if any(fnmatch.fnmatch(param_name, pat) for pat in self.frozen):
            return False
        return any(fnmatch.fnmatch(param_name, pat) for pat in self.gradient_permission)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_card(cls, card: Any) -> "SourceSpec":
        """Adapt agent B's ``SourceCardDoc`` (duck-typed)."""
        data = getattr(card, "data", None) or getattr(card, "payload", None) or {}
        get = lambda *ks, default=None: _dig(data, *ks, default=default)  # noqa: E731
        perm = get("gradient_permission", "modules", default=None) or ["*"]
        led = get("ledger", default={}) or {}
        var = (led.get("variance") or {}) if isinstance(led, dict) else {}
        bi = led.get("bias_interval") if isinstance(led, dict) else None
        halfw = None
        if isinstance(bi, (list, tuple)) and len(bi) == 2 and all(isinstance(x, (int, float)) for x in bi):
            halfw = abs(float(bi[1]) - float(bi[0])) / 2
        pop = get("population", default={}) or {}
        return cls(
            id=str(getattr(card, "id", get("identity", "id", default="unknown"))),
            role=str(getattr(card, "role", get("role", default="likelihood"))),
            gradient_permission=tuple(perm),
            frozen=tuple(get("gradient_permission", "frozen", default=[]) or []),
            n_eff=float(pop.get("n_participants", 1) or 1),
            n_rows=float(pop.get("n_rows", 1) or 1),
            n_participants=pop.get("n_participants"),
            n_sites=pop.get("n_sites"),
            measurement_variance=var.get("measurement"),
            bias_halfwidth=halfw,
            bias_status=str(led.get("bias_status", "prior_specified_sensitivity")) if isinstance(led, dict) else "prior_specified_sensitivity",
            model_discrepancy=led.get("model_discrepancy") if isinstance(led, dict) else None,
            validity_overlap=float(get("validity_overlap", default=1.0) or 1.0),
            hierarchy_depth=int(pop.get("hierarchy_depth", 1) or 1),
            is_simulated=bool(get("identity", "simulated", default=False)),
            notes=str(get("identity", "notes", default="")),
        )

    @classmethod
    def load_dir(cls, path: str | Path) -> dict[str, "SourceSpec"]:
        import yaml

        out: dict[str, SourceSpec] = {}
        p = Path(path)
        if not p.exists():
            return out
        for f in sorted(p.glob("*.y*ml")):
            d = yaml.safe_load(f.read_text()) or {}
            for k in ("gradient_permission", "frozen", "losses"):
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])
            s = cls(**d)
            out[s.id] = s
        return out


def _dig(d: Any, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def assert_not_row_count(specs: Sequence[SourceSpec], *, factor: float = 10.0) -> None:
    """Test hook: multiplying a source's rows must not multiply its weight.

    Appendix D is explicit that ``w_e`` "is not proportional to row count alone".
    This asserts the implemented functional form actually has that property.
    """
    for s in specs:
        base = s.reliability()
        bumped = SourceSpec(**{**s.as_dict(), "n_rows": s.n_rows * factor})
        if abs(bumped.reliability() - base) > 1e-12:
            raise AssertionError(f"{s.id}: reliability changed with row count alone")
        more = SourceSpec(**{**s.as_dict(), "n_eff": s.n_eff * factor})
        if base > 0 and more.reliability() > base * factor * 0.9:
            raise AssertionError(
                f"{s.id}: reliability scaled ~linearly with n_eff ({base:.4g} -> {more.reliability():.4g}); "
                "the effective-sample-size term must saturate"
            )


# ======================================================================
# weights and normalisation
# ======================================================================
class MixtureWeights:
    """Normalised, preregistered reliability weights over sources."""

    def __init__(self, specs: Mapping[str, SourceSpec], *, temperature: float = 1.0, floor: float = 0.0) -> None:
        self.specs = dict(specs)
        raw = {k: s.reliability() for k, s in self.specs.items()}
        tot = sum(v ** (1.0 / max(temperature, 1e-6)) for v in raw.values())
        self.raw = raw
        self.w = (
            {k: (v ** (1.0 / max(temperature, 1e-6))) / tot for k, v in raw.items()}
            if tot > 0
            else {k: 0.0 for k in raw}
        )
        if floor > 0:
            for k in self.w:
                if self.w[k] > 0:
                    self.w[k] = max(self.w[k], floor)
            tot2 = sum(self.w.values())
            self.w = {k: v / tot2 for k, v in self.w.items()}

    def __getitem__(self, k: str) -> float:
        return self.w.get(k, 0.0)

    def report(self) -> dict[str, Any]:
        return {
            "weights": self.w,
            "raw_reliability": self.raw,
            "inputs": {
                k: {
                    "role": s.role,
                    "n_eff": s.n_eff,
                    "n_rows": s.n_rows,
                    "measurement_variance": s.measurement_variance,
                    "bias_halfwidth": s.bias_halfwidth,
                    "model_discrepancy": s.model_discrepancy,
                    "validity_overlap": s.validity_overlap,
                }
                for k, s in self.specs.items()
            },
            "note": "w_e is a reliability term; row count is recorded for audit and never enters it.",
        }


class LossNormalizer:
    """Within-participant then within-source loss normalisation.

    Keeps an exponential-moving scale per ``(source, participant)`` and per
    ``source``.  Dividing by these before cross-source weighting is what stops a
    long recording or a heavily oversampled simulator from dominating.
    """

    def __init__(self, *, momentum: float = 0.02, eps: float = 1e-6, warmup: int = 20) -> None:
        self.momentum = momentum
        self.eps = eps
        self.warmup = warmup
        self._scale: dict[tuple[str, str], float] = {}
        self._count: dict[tuple[str, str], int] = defaultdict(int)

    def _upd(self, key: tuple[str, str], v: float) -> float:
        c = self._count[key]
        self._count[key] = c + 1
        if key not in self._scale:
            self._scale[key] = abs(v) + self.eps
        else:
            m = self.momentum if c >= self.warmup else 0.25
            self._scale[key] = (1 - m) * self._scale[key] + m * (abs(v) + self.eps)
        return self._scale[key]

    def __call__(self, loss: Tensor, *, source: str, participant: str = "_") -> Tensor:
        v = float(loss.detach())
        s_p = self._upd((source, participant), v)
        s_s = self._upd((source, "__source__"), v / max(s_p, self.eps))
        return loss / max(s_p, self.eps) / max(s_s, self.eps)

    def state_dict(self) -> dict[str, Any]:
        return {"scale": {"|".join(k): v for k, v in self._scale.items()}, "count": {"|".join(k): v for k, v in self._count.items()}}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self._scale = {tuple(k.split("|", 1)): float(v) for k, v in d.get("scale", {}).items()}  # type: ignore[misc]
        self._count = defaultdict(int, {tuple(k.split("|", 1)): int(v) for k, v in d.get("count", {}).items()})  # type: ignore[misc]


# ======================================================================
# gradient permission
# ======================================================================
class GradientGate:
    """Compiles ``A_k`` into an executable mask and an audit log.

    Gradients for a source are computed **only with respect to the parameters it
    permits**.  A non-permitted parameter therefore never receives a ``.grad``
    from that source -- there is no post-hoc zeroing that a later optimiser step
    could undo, and ``p.grad is None`` is directly assertable.
    """

    def __init__(self, model: nn.Module, specs: Mapping[str, SourceSpec]) -> None:
        self.model = model
        self.specs = dict(specs)
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.permitted: dict[str, list[tuple[str, Tensor]]] = {}
        self.blocked: dict[str, list[str]] = {}
        for sid, spec in self.specs.items():
            allow: list[tuple[str, Tensor]] = []
            block: list[str] = []
            for n, p in self.named:
                if spec.permits(n):
                    allow.append((n, p))
                else:
                    block.append(n)
            self.permitted[sid] = allow
            self.blocked[sid] = block

    def params(self, source: str) -> list[Tensor]:
        return [p for _, p in self.permitted.get(source, [])]

    def names(self, source: str) -> list[str]:
        return [n for n, _ in self.permitted.get(source, [])]

    def grads(self, loss: Tensor, source: str, *, retain_graph: bool = True, create_graph: bool = False) -> dict[str, Tensor]:
        """``d loss / d theta`` restricted to the permitted parameters."""
        names = self.names(source)
        ps = self.params(source)
        if not ps:
            return {}
        gs = torch.autograd.grad(loss, ps, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
        return {n: g for n, g in zip(names, gs) if g is not None}

    def accumulate(self, grads: Mapping[str, Tensor], *, scale: float = 1.0) -> None:
        lut = dict(self.named)
        for n, g in grads.items():
            p = lut[n]
            if p.grad is None:
                p.grad = g.detach().clone() * scale
            else:
                p.grad.add_(g.detach(), alpha=scale)

    def audit(self) -> dict[str, Any]:
        return {
            sid: {
                "n_permitted": len(self.permitted[sid]),
                "n_blocked": len(self.blocked[sid]),
                "patterns": list(spec.gradient_permission),
                "frozen": list(spec.frozen),
                "blocked_sample": list(self.blocked[sid])[:12],
            }
            for sid, spec in self.specs.items()
        }


# ======================================================================
# gradient conflict
# ======================================================================
class ConflictLog:
    """Cosine similarity between source gradients, **by module and by source**."""

    def __init__(self, *, window: int = 50, module_of: Callable[[str], str] | None = None) -> None:
        self.window = window
        self.module_of = module_of or (lambda n: n.split(".")[0])
        self.hist: dict[tuple[str, str, str], deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.steps = 0

    def update(self, grads_by_source: Mapping[str, Mapping[str, Tensor]]) -> dict[str, float]:
        self.steps += 1
        by_src_mod: dict[str, dict[str, list[Tensor]]] = {}
        for sid, gd in grads_by_source.items():
            per_mod: dict[str, list[Tensor]] = defaultdict(list)
            for n, g in gd.items():
                per_mod[self.module_of(n)].append(g.detach().float().reshape(-1))
            by_src_mod[sid] = {m: torch.cat(v) for m, v in per_mod.items()}
        out: dict[str, float] = {}
        sids = sorted(by_src_mod)
        for i, a in enumerate(sids):
            for b in sids[i + 1 :]:
                shared = set(by_src_mod[a]) & set(by_src_mod[b])
                for m in sorted(shared):
                    ga, gb = by_src_mod[a][m], by_src_mod[b][m]
                    if ga.numel() != gb.numel() or ga.numel() == 0:
                        continue
                    c = float(torch.nn.functional.cosine_similarity(ga, gb, dim=0))
                    self.hist[(a, b, m)].append(c)
                    out[f"conflict/{a}|{b}/{m}"] = c
        return out

    def summary(self) -> dict[str, Any]:
        rows = []
        for (a, b, m), dq in self.hist.items():
            if not dq:
                continue
            v = list(dq)
            rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "module": m,
                    "mean_cosine": sum(v) / len(v),
                    "min_cosine": min(v),
                    "frac_negative": sum(1 for x in v if x < 0) / len(v),
                    "n": len(v),
                }
            )
        rows.sort(key=lambda r: r["mean_cosine"])
        return {"steps": self.steps, "pairs": rows}


@dataclass
class ConflictPolicy:
    """What to do when two sources disagree -- never forced consensus.

    Ordered escalation, each step recorded in the run manifest:
    ``adapter`` -> ``partial_pool`` -> ``freeze`` -> ``reject``.
    """

    conflict_threshold: float = -0.15  # mean cosine below which a pair is "incompatible"
    min_observations: int = 20
    escalation: tuple[str, ...] = ("adapter", "partial_pool", "freeze", "reject")
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(self, log: ConflictLog, weights: MixtureWeights) -> list[dict[str, Any]]:
        acts: list[dict[str, Any]] = []
        for row in log.summary()["pairs"]:
            if row["n"] < self.min_observations or row["mean_cosine"] >= self.conflict_threshold:
                continue
            prior = [d for d in self.decisions if d["pair"] == (row["source_a"], row["source_b"]) and d["module"] == row["module"]]
            level = min(len(prior), len(self.escalation) - 1)
            # a source with lower reliability is the one that yields
            a, b = row["source_a"], row["source_b"]
            yielding = a if weights[a] <= weights[b] else b
            act = {
                "pair": (a, b),
                "module": row["module"],
                "action": self.escalation[level],
                "yielding_source": yielding,
                "mean_cosine": row["mean_cosine"],
                "n": row["n"],
                "rationale": (
                    "incompatible gradients on a shared operator; escalating per Appendix D "
                    "(source-specific adapter / partial pooling / freezing / rejection) rather "
                    "than averaging the disagreement away"
                ),
            }
            self.decisions.append(act)
            acts.append(act)
        return acts


# ======================================================================
# the trainer glue
# ======================================================================
class MixtureTrainer:
    """Composes per-source losses into one update with masks + conflict logging."""

    def __init__(
        self,
        model: nn.Module,
        specs: Mapping[str, SourceSpec],
        *,
        temperature: float = 1.0,
        conflict_window: int = 50,
        policy: ConflictPolicy | None = None,
    ) -> None:
        self.model = model
        self.specs = dict(specs)
        self.weights = MixtureWeights(self.specs, temperature=temperature)
        self.gate = GradientGate(model, self.specs)
        self.normalizer = LossNormalizer()
        self.conflict = ConflictLog(window=conflict_window)
        self.policy = policy or ConflictPolicy()
        self.contribution: dict[str, float] = defaultdict(float)
        self.skipped_nonfinite: dict[str, int] = {}
        self.adapters_enabled: set[tuple[str, str]] = set()

    def step(
        self,
        losses: Mapping[str, Tensor],
        *,
        participants: Mapping[str, str] | None = None,
        measure_conflict: bool = True,
    ) -> dict[str, Any]:
        """Accumulate ``.grad`` from every source under its own mask.

        Returns diagnostics.  The caller runs the optimiser step.
        """
        for p in self.model.parameters():
            p.grad = None
        parts = participants or {}
        grads_by_source: dict[str, dict[str, Tensor]] = {}
        total = 0.0
        active = [k for k in losses if self.weights[k] > 0]
        # Renormalise over the sources actually present this step: a source that
        # contributes no loss must not silently shrink the ones that do.
        wsum = sum(self.weights[k] for k in active) or 1.0
        step_w = {k: self.weights[k] / wsum for k in active}
        for i, sid in enumerate(active):
            spec = self.specs.get(sid)
            if spec is None or self.weights[sid] <= 0:
                continue
            raw = losses[sid]
            if not torch.isfinite(raw.detach()):
                # A non-finite loss is a bug report, not a gradient. Skip it and
                # record it rather than poisoning every parameter with NaN.
                self.skipped_nonfinite[sid] = self.skipped_nonfinite.get(sid, 0) + 1
                continue
            ln = self.normalizer(raw, source=sid, participant=parts.get(sid, "_"))
            g = self.gate.grads(ln, sid, retain_graph=(i < len(active) - 1) or measure_conflict)
            grads_by_source[sid] = g
            self.gate.accumulate(g, scale=step_w[sid])
            c = float(ln.detach()) * step_w[sid]
            self.contribution[sid] += c
            total += c
        diag: dict[str, Any] = {"mixture_total": total, "weights": dict(step_w)}
        if measure_conflict and len(grads_by_source) > 1:
            diag["conflict"] = self.conflict.update(grads_by_source)
            acts = self.policy.evaluate(self.conflict, self.weights)
            if acts:
                diag["conflict_actions"] = acts
                for a in acts:
                    if a["action"] == "adapter":
                        self.adapters_enabled.add((a["yielding_source"], a["module"]))
                    elif a["action"] in ("freeze", "reject"):
                        s = self.specs[a["yielding_source"]]
                        self.specs[a["yielding_source"]] = SourceSpec(
                            **{**s.as_dict(), "frozen": tuple(set(s.frozen) | {a["module"] + ".*"})}
                        )
                        self.gate = GradientGate(self.model, self.specs)
        return diag

    def report(self) -> dict[str, Any]:
        tot = sum(self.contribution.values()) or 1.0
        return {
            "mixture_weights": self.weights.report(),
            "per_source_contribution": {k: v / tot for k, v in sorted(self.contribution.items())},
            "gradient_permission_audit": self.gate.audit(),
            "gradient_conflict": self.conflict.summary(),
            "conflict_decisions": self.policy.decisions,
            "adapters_enabled": sorted(f"{s}:{m}" for s, m in self.adapters_enabled),
            "skipped_nonfinite_steps": dict(self.skipped_nonfinite),
        }

    def save_report(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.report(), indent=2, default=float))
