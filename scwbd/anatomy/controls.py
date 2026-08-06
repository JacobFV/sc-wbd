"""The G2 null graphs, in the shape the benchmark gate asks for.

:func:`~scwbd.anatomy.connectome.StructuralPrior.controls` already builds the
five controls, but it is a *method on a loaded prior* and it returns
:class:`StructuralPrior` objects.  ``scwbd.bench.gates.run_g2`` wants a plain
``{name: adjacency}`` mapping and probes for ``scwbd.anatomy.controls
.graph_controls``.  Without that symbol G2 and Appendix-D row D07 report
``COULD_NOT_RUN`` even though the controls are implemented and tested -- so the
symbol is the deliverable, not a convenience.

The gate refuses to synthesise these itself, correctly: for G2 the control *is*
the experiment, and a null graph invented by the thing being tested is not a
null.  This module is the honest supply of them.

    from scwbd.anatomy.controls import anatomy_adjacency, graph_controls

    A     = anatomy_adjacency("Schaefer400x7")
    nulls = graph_controls("Schaefer400x7", seed=0)
    run_g2(anatomy=A, controls=nulls, ...)

Both are built from the same loaded prior, so the adjacency and its nulls always
describe the same parcellation in the same node order.  Mixing an adjacency from
one atlas with controls from another is the failure this pairing prevents.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .connectome import StructuralPrior, load_structural_prior

__all__ = ["CONTROL_NAMES", "anatomy_adjacency", "graph_controls", "control_report"]

#: The five nulls, in the order the report tabulates them.  ``run_g2`` requires
#: the first three; ``local_only`` and ``graph_only`` separate the long-range
#: operator and the weights, and are supplied because they are cheap and the
#: gate is free to ignore them.
CONTROL_NAMES: tuple[str, ...] = (
    "dense", "randomized", "distance_matched", "local_only", "graph_only",
)


def _prior(atlas: str, include_subcortex: bool, length: str) -> StructuralPrior:
    return load_structural_prior(atlas, include_subcortex=include_subcortex,
                                 length=length)


def anatomy_adjacency(atlas: str = "Schaefer400x7", *,
                      include_subcortex: bool = True,
                      length: str = "euclidean") -> np.ndarray:
    """The empirical weighted adjacency G2 tests *against* its nulls.

    Weights are log streamline density on an arbitrary monotone scale (see
    ``StructuralPrior.weights_units``); they are not in physical units and a
    model that depends on their absolute values depends on a pipeline artifact.
    """
    return _prior(atlas, include_subcortex, length).weights.copy()


def graph_controls(atlas: str = "Schaefer400x7", *,
                   seed: int = 0,
                   include_subcortex: bool = True,
                   length: str = "euclidean",
                   names: tuple[str, ...] = CONTROL_NAMES) -> dict[str, np.ndarray]:
    """``{control_name: weighted adjacency}`` for the G2 nulls.

    Parameters
    ----------
    seed
        Seeds the two stochastic controls.  ``randomized`` and
        ``distance_matched`` are deterministic given a seed, so a gate run is
        reproducible; the seed is recorded in each control's provenance.

    Notes
    -----
    Every control downgrades all of its edges to ``proposed`` evidence, because
    a null graph has no anatomical support by construction.  That downgrade is
    invisible in the bare adjacency returned here -- use
    :func:`control_report` if the evidence classes matter.
    """
    sc = _prior(atlas, include_subcortex, length)
    built = sc.controls(seed=seed)
    unknown = set(names) - set(built)
    if unknown:
        raise ValueError(f"unknown control(s) {sorted(unknown)}; have {sorted(built)}")
    return {k: built[k].weights.copy() for k in names}


def control_report(atlas: str = "Schaefer400x7", *, seed: int = 0,
                   include_subcortex: bool = True,
                   length: str = "euclidean") -> dict[str, Any]:
    """What each null preserved and destroyed, for the gate's manifest.

    G2's verdict is only interpretable if the reader can see that
    ``distance_matched`` really did keep the distance decay and that
    ``randomized`` really did destroy the topology.  These are the numbers
    tabulated in ``reports/anatomy_prior.md`` §4.
    """
    sc = _prior(atlas, include_subcortex, length)
    iu = np.triu_indices(sc.n_parcels, 1)
    we, dm = sc.weights[iu], sc.distance_mm[iu]
    deg0 = (sc.weights > 0).sum(0)
    me = we > 0

    def row(w2: np.ndarray) -> dict[str, Any]:
        m = w2[iu] > 0
        return {
            "n_edges": int(m.sum()),
            "total_weight": float(w2[iu].sum()),
            "degree_sequence_preserved": bool(((w2 > 0).sum(0) == deg0).all()),
            "r_weight_distance": (float(np.corrcoef(w2[iu][m], dm[m])[0, 1])
                                  if m.sum() > 2 and w2[iu][m].std() > 0 else None),
            "r_weight_empirical": (float(np.corrcoef(we, w2[iu])[0, 1])
                                   if w2[iu].std() > 0 else None),
        }

    out: dict[str, Any] = {
        "atlas": atlas,
        "n_parcels": int(sc.n_parcels),
        "seed": seed,
        "empirical": {
            "n_edges": int(me.sum()),
            "total_weight": float(we.sum()),
            "degree_sequence_preserved": True,
            "r_weight_distance": float(np.corrcoef(we[me], dm[me])[0, 1]),
            "r_weight_empirical": 1.0,
        },
        "weights_units": sc.weights_units,
    }
    for k, c in sc.controls(seed=seed).items():
        out[k] = row(c.weights)
        out[k]["control_kind"] = c.control_kind
        out[k]["note"] = c.provenance["control"]["note"]
    return out
