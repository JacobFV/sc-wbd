"""Matched-capacity accounting: an unmatched win is not a win."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scwbd.bench.matching import Budget, budget_of, check_matched, matched_subcheck
from scwbd.bench.synthetic import RidgeGaussian, make_graph_dataset


@dataclass
class _Sized:
    n: int
    name: str = "sized"

    def n_parameters(self) -> int:
        return self.n


class _Opaque:
    name = "opaque"


def test_matched_when_budgets_agree():
    v = check_matched(_Sized(100), {"a": _Sized(100), "b": _Sized(105)}, tol=0.10)
    assert v.matched
    assert matched_subcheck(v).status == "PASS"


def test_over_budget_candidate_cannot_pass():
    v = check_matched(_Sized(300), {"a": _Sized(100)}, tol=0.10)
    assert not v.matched
    sub = matched_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert "unmatched win is not a win" in sub.reason
    assert "3.00x" in sub.reason


def test_smaller_candidate_is_matched_and_flagged_favourable_to_the_null():
    v = check_matched(_Sized(50), {"a": _Sized(100)}, tol=0.10)
    assert v.matched and v.favourable_to_null
    assert matched_subcheck(v).status == "PASS"


def test_unknown_capacity_is_could_not_run_not_a_pass():
    v = check_matched(_Opaque(), {"a": _Sized(100)})
    sub = matched_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert "capacity accounting unavailable" in sub.reason


def test_budget_of_reads_a_torch_module_if_present():
    torch = __import__("torch")
    m = torch.nn.Linear(4, 3)
    b = budget_of(m)
    assert b.n_parameters == 4 * 3 + 3
    assert b.source in ("n_parameters()", "torch.parameters()")


def test_reference_arms_report_the_parameters_they_actually_fit():
    d = make_graph_dataset(seed=0, n_regions=6, n_train=80, n_test=80, density=0.5)
    dense = RidgeGaussian(mask=np.ones((6, 6))).fit(d["train"])
    sparse = RidgeGaussian(mask=d["anatomy"]).fit(d["train"])
    assert dense.n_parameters() > sparse.n_parameters()
    v = check_matched(sparse, {"dense": dense}, tol=0.10)
    # the sparse candidate is smaller: the comparison is valid and favourable
    # to the null, which is exactly the direction §11.4 permits
    assert v.matched and v.favourable_to_null
