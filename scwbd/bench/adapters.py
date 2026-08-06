"""Thin adapters onto the other agents' modules (agent J).

Agent J is the falsification machinery; it is built while agents A–I are still
landing their modules.  Every dependency is therefore *probed*, never
imported at module scope.  When a dependency is absent the gate that needs it
emits ``COULD_NOT_RUN`` with the missing symbol named in the reason.

Rule (from the task, and from ARCHITECTURE.md §4): a gate that cannot run
reports ``COULD_NOT_RUN``.  It never reports a pass, and it never quietly
substitutes a stand-in for the thing it was supposed to measure.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Dependency",
    "probe",
    "probe_attr",
    "require",
    "AGENT_OF",
    "dependency_table",
    "fisher_backend",
    "fisher_design_map",
    "reference_compiled",
    "field_solvers",
    "theta_partition",
    "anatomy_controls",
]

#: which agent owns which module, so a COULD_NOT_RUN names the blocker
AGENT_OF: dict[str, str] = {
    "scwbd.schema": "A (schema)",
    "scwbd.compiler": "A (compiler)",
    "scwbd.sources": "B (sources)",
    "scwbd.anatomy": "C (anatomy)",
    "scwbd.transforms": "D (transforms)",
    "scwbd.dynamics": "E (dynamics)",
    "scwbd.observe": "F (observation physics)",
    "scwbd.intervene": "G (intervention)",
    "scwbd.infer": "H (inference)",
    "scwbd.foundation": "I (foundation model)",
    "scwbd.runtime": "K (runtime)",
}


@dataclass(frozen=True)
class Dependency:
    """Result of probing a module or symbol."""

    name: str
    available: bool
    obj: Any = None
    reason: str = ""

    @property
    def agent(self) -> str:
        root = ".".join(self.name.split(".")[:2])
        return AGENT_OF.get(root, "unknown owner")

    @property
    def blocker(self) -> str:
        return f"{self.name} unavailable (owner: agent {self.agent}) — {self.reason}"

    def __bool__(self) -> bool:
        return self.available


def probe(module: str) -> Dependency:
    """Import ``module`` if it exists; otherwise report why it does not."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # ImportError, and any import-time failure
        return Dependency(module, False, None, f"{type(exc).__name__}: {exc}")
    return Dependency(module, True, mod, "")


def probe_attr(module: str, attr: str) -> Dependency:
    """Probe ``module.attr``; a module without the symbol is still missing."""
    dep = probe(module)
    name = f"{module}.{attr}"
    if not dep.available:
        return Dependency(name, False, None, dep.reason)
    obj = getattr(dep.obj, attr, None)
    if obj is None:
        return Dependency(
            name, False, None, f"module {module} imported but defines no {attr!r}"
        )
    return Dependency(name, True, obj, "")


def require(*names: str) -> list[Dependency]:
    """Probe several dotted names (``module`` or ``module.attr``)."""
    out: list[Dependency] = []
    for n in names:
        parts = n.split(".")
        if len(parts) > 2 and parts[0] == "scwbd":
            # try as module first, then as module.attr
            dep = probe(n)
            if dep.available:
                out.append(dep)
                continue
            out.append(probe_attr(".".join(parts[:-1]), parts[-1]))
        else:
            out.append(probe(n))
    return out


def missing(deps: Iterable[Dependency]) -> list[Dependency]:
    return [d for d in deps if not d.available]


def blockers(deps: Iterable[Dependency]) -> str:
    return "; ".join(d.blocker for d in missing(deps))


def dependency_table() -> list[dict[str, Any]]:
    """Status of every sibling module, for reports/gates/SUMMARY.md."""
    rows: list[dict[str, Any]] = []
    for mod, agent in sorted(AGENT_OF.items()):
        dep = probe(mod)
        rows.append(
            {
                "module": mod,
                "agent": agent,
                "available": dep.available,
                "reason": dep.reason,
            }
        )
    return rows


# --------------------------------------------------------------------------
# named capabilities the gates need
# --------------------------------------------------------------------------
def fisher_backend() -> Dependency:
    """Agent H's Fisher-information machinery (G4).

    G4 *consumes* it; agent J does not implement Fisher information, because a
    gate that computes its own version of the quantity it is auditing is not
    an audit.
    """
    for cand in (
        ("scwbd.infer.fisher", "expected_fisher"),
        ("scwbd.infer.fisher", "expected_fisher_information"),
        ("scwbd.infer.fisher", "fisher_information"),
        ("scwbd.infer", "expected_fisher"),
        ("scwbd.infer", "fisher_information"),
    ):
        dep = probe_attr(*cand)
        if dep.available:
            return dep
    return Dependency(
        "scwbd.infer.fisher.expected_fisher",
        False,
        None,
        "agent H has not landed expected_fisher / fisher_information",
    )


def theta_partition() -> Dependency:
    """Agent H's split of the parameter vector into science and nuisance names.

    G4's second falsifier ("adds only field-model uncertainty") is only
    evaluable when someone has said which parameters are the *scientific* ones.
    That declaration belongs to the inference module, not to the benchmark.
    """
    names = probe_attr("scwbd.infer.fisher", "THETA_NAMES")
    allnames = probe_attr("scwbd.infer.fisher", "PARAM_NAMES")
    if names.available and allnames.available:
        return Dependency("scwbd.infer.fisher.THETA_NAMES", True,
                          (list(names.obj), list(allnames.obj)), "")
    return Dependency(
        "scwbd.infer.fisher.THETA_NAMES",
        False,
        None,
        "agent H has not declared THETA_NAMES/PARAM_NAMES; pass theta_index and "
        "nuisance_index explicitly",
    )


def reference_compiled() -> Dependency:
    """Compile agent A's worked three-region example, or report why not.

    This is the subject of the N1 compiler-correctness check.  It is a
    *reference example*, not the SC-WBD-001-beta schema: a pass licenses a
    statement about the compiler, never about a whole-brain model.
    """
    comp = probe_attr("scwbd.compiler", "compile")
    build_schema = probe_attr("scwbd.schema.examples.three_region",
                              "build_three_region_schema")
    build_claim = probe_attr("scwbd.schema.examples.three_region",
                             "build_three_region_claim")
    for dep in (comp, build_schema, build_claim):
        if not dep.available:
            return Dependency("scwbd.compiler.compile(three_region)", False, None, dep.reason)
    try:
        model = comp.obj(build_schema.obj(), claim=build_claim.obj())
    except Exception as exc:
        return Dependency(
            "scwbd.compiler.compile(three_region)", False, None,
            f"the reference example did not compile: {type(exc).__name__}: {exc}",
        )
    return Dependency("scwbd.compiler.compile(three_region)", True, model, "")


def fisher_design_map(u: Any, cfg: Any, proto: Any, **kw: Any) -> Dependency:
    """Bind agent H's ``expected_fisher`` into the ``design -> information`` map G4 wants.

    G4 deliberately does not build ``u``/``cfg``/``proto`` itself: choosing the
    system and protocol *is* the experiment, and it belongs to whoever is
    making the claim.  This helper only performs the binding::

        dep = fisher_design_map(u, cfg, proto)
        run_g4(fisher=dep.obj, ...)
    """
    dep = fisher_backend()
    if not dep.available:
        return dep
    fn = dep.obj

    def _map(design: str):
        return fn(u, cfg, proto, design=design, **kw)

    return Dependency(dep.name + "(bound)", True, _map, "")


def field_solvers() -> Dependency:
    """Agent Faraday's reference-problem field solvers for gates N3 and N4.

    These are *purpose-built discretisations of the reference problems*, not the
    production operators: N3's subject is a current dipole in an unbounded
    conductor (conduction), N4's a free-field monopole.  The production TMS
    induced-field operator is a different problem and has its own gate (N6).
    """
    em = probe_attr("scwbd.intervene.numerics", "quasistatic_dipole_potential_fd")
    ac = probe_attr("scwbd.intervene.numerics", "run_free_field_monopole")
    if em.available and ac.available:
        return Dependency("scwbd.intervene.numerics", True, (em.obj, ac.obj), "")
    reason = em.reason or ac.reason or "field solvers not exposed"
    return Dependency("scwbd.intervene.numerics", False, None, reason)


def anatomy_controls() -> Dependency:
    """Agent C's graph controls (dense / randomized / distance-matched) for G2."""
    for cand in (
        ("scwbd.anatomy.controls", "graph_controls"),
        ("scwbd.anatomy.controls", "make_controls"),
        ("scwbd.anatomy", "graph_controls"),
    ):
        dep = probe_attr(*cand)
        if dep.available:
            return dep
    return Dependency(
        "scwbd.anatomy.graph_controls",
        False,
        None,
        "agent C has not landed the randomized / distance-matched / dense graph controls; "
        "agent J will not fabricate them, because the control is the experiment",
    )
