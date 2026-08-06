"""Regional maps: coverage, ledgers, and the inferences they forbid."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy.maps import (
    RECEPTOR_GROUPS,
    SURFACE_MAPS,
    MapSet,
    available_maps,
    load_maps,
    receptor_matrix,
)

EXPECTED_SURFACE = [
    "fc_gradient1",
    "myelin_t1t2",
    "cortical_thickness",
    "sa_axis",
    "intrinsic_timescale_meg",
]


def test_the_headline_maps_are_present(maps_small):
    for k in EXPECTED_SURFACE:
        assert k in maps_small, f"{k} missing"


def test_the_full_receptor_panel_is_present(maps_small):
    assert len(maps_small.receptor_names) >= 18, maps_small.receptor_names
    for k in ("NMDA", "GABAa", "mGluR5", "D1", "D2", "5HTT", "MOR"):
        assert k in maps_small.receptor_names


def test_fdopa_is_not_called_a_receptor(maps_small):
    """FDOPA indexes synthesis capacity, not a receptor; it stays out of the panel."""
    assert "FDOPA" not in maps_small.receptor_names
    if "receptor_FDOPA" in maps_small:
        assert "not a receptor" in maps_small["receptor_FDOPA"].notes


@pytest.mark.parametrize("key", EXPECTED_SURFACE)
def test_values_are_finite_within_coverage_and_nan_outside(maps_small, key):
    m = maps_small[key]
    assert np.isfinite(m.values[m.coverage]).all()
    assert np.isnan(m.values[~m.coverage]).all(), "uncovered parcels are nan, never 0"


def test_no_map_is_all_nan_or_constant(maps_small):
    for k, m in maps_small.maps.items():
        v = m.values[m.coverage]
        assert v.size > 0, f"{k} has no coverage at all"
        assert np.nanstd(v) > 0, f"{k} is constant across parcels"


def test_every_map_carries_a_ledger_that_satisfies_r08(maps_small):
    for k, m in maps_small.maps.items():
        led = m.ledger
        assert led.has_estimator(), f"{k}: R08 would refuse this ledger"
        assert led.bias_status == "prior_specified_sensitivity", (
            f"{k}: a group average cannot claim a design-estimable or externally "
            "bounded bias"
        )
        assert led.bias_interval[0] < led.bias_interval[1], (
            f"{k}: prior_specified_sensitivity needs a swept range, not a point"
        )


def test_every_map_names_the_inference_it_forbids(maps_small):
    for k, m in maps_small.maps.items():
        fi = m.ledger.validity_domain.get("forbidden_inference")
        assert fi and len(fi) > 30, f"{k} does not say what it cannot support"


def test_receptor_maps_forbid_subject_level_inference(maps_small):
    """Thesis Appendix A is explicit about this one."""
    for r in maps_small.receptor_names:
        m = maps_small[f"receptor_{r}"]
        fi = m.ledger.validity_domain["forbidden_inference"]
        assert "NOT inferable" in fi
        assert "Appendix A" in fi
        assert m.ledger.validity_domain["license"].startswith("CC-BY-NC")


def test_receptor_maps_record_their_donor_count(maps_small):
    counts = [
        maps_small[f"receptor_{r}"].ledger.validity_domain.get("n_donors")
        for r in maps_small.receptor_names
    ]
    known = [c for c in counts if c]
    assert known, "donor counts should be recovered from the upstream filenames"
    assert min(known) < 60, "the point is that some of these cohorts are tiny"


def test_receptor_values_are_zscored(maps_small):
    for r in maps_small.receptor_names:
        m = maps_small[f"receptor_{r}"]
        v = m.values[m.coverage]
        assert abs(np.mean(v)) < 0.2
        assert m.units == "zscore"


def test_receptor_matrix_shape(parc_small):
    v, names, cov = receptor_matrix(parc_small)
    assert v.shape == (parc_small.n_parcels, len(names))
    assert cov.shape == v.shape
    assert len(names) >= 18


def test_ei_proxy_is_derived_and_marked_surrogate(maps_small):
    ei = maps_small["ei_proxy"]
    assert ei.mechanistic_status == "surrogate"
    fi = ei.ledger.validity_domain["forbidden_inference"]
    assert "Not a measurement" in fi
    assert set(ei.ledger.validity_domain["excitatory_markers"]) == set(
        RECEPTOR_GROUPS["excitatory"]
    )


def test_ei_proxy_has_the_sign_structure_of_its_definition(maps_small):
    """The contrast must run against GABA-A and with the glutamatergic markers."""
    from scipy import stats

    ei = maps_small["ei_proxy"]
    gaba = maps_small["receptor_GABAa"]
    mglur5 = maps_small["receptor_mGluR5"]
    m = ei.coverage & gaba.coverage & mglur5.coverage
    assert stats.pearsonr(ei.values[m], gaba.values[m]).statistic < -0.3
    assert stats.pearsonr(ei.values[m], mglur5.values[m]).statistic > 0.2


def test_ei_proxy_is_a_weak_second_order_contrast(maps_small):
    """The excitatory and inhibitory maps co-vary, so most variance cancels.

    This is a documented property of the proxy, not an accident, and it is the
    reason `ei_proxy` is `mechanistic_status="surrogate"` and carries a
    model-class variance as large as its measurement variance. If this test
    ever starts failing because the ingredients decorrelate, the docstring in
    `maps._ei_proxy` needs rewriting rather than the test.
    """
    from scipy import stats

    ei = maps_small["ei_proxy"]
    nmda = maps_small["receptor_NMDA"]
    gaba = maps_small["receptor_GABAa"]
    m = ei.coverage & nmda.coverage & gaba.coverage
    shared = stats.pearsonr(nmda.values[m], gaba.values[m]).statistic
    assert shared > 0.4, "excitatory and inhibitory markers should co-vary"
    assert ei.values[m].std() < nmda.values[m].std(), (
        "differencing co-varying maps must shrink the spread"
    )


def test_ei_proxy_still_tracks_the_cortical_hierarchy(maps_small):
    """Weak is not meaningless: the residual runs along the S-A axis."""
    from scipy import stats

    ei = maps_small["ei_proxy"]
    sa = maps_small["sa_axis"]
    m = ei.coverage & sa.coverage
    assert stats.pearsonr(ei.values[m], sa.values[m]).statistic > 0.2


def test_hierarchy_maps_agree_with_each_other(maps_small):
    """Myelin should run opposite to the principal gradient; a broken parcellation
    join would destroy this."""
    from scipy import stats

    a = maps_small["myelin_t1t2"]
    b = maps_small["fc_gradient1"]
    m = a.coverage & b.coverage
    rho = stats.spearmanr(a.values[m], b.values[m]).statistic
    assert abs(rho) > 0.35, f"myelin and the principal gradient are unrelated (rho={rho:.2f})"


def test_cortical_thickness_is_in_millimetres(maps_small):
    t = maps_small["cortical_thickness"]
    assert t.units == "mm"
    v = t.values[t.coverage]
    assert 1.5 < np.mean(v) < 4.0, f"mean cortical thickness {np.mean(v):.2f} mm"


def test_meg_timescale_is_in_milliseconds(maps_small):
    t = maps_small["intrinsic_timescale_meg"]
    assert t.units == "ms"
    v = t.values[t.coverage]
    assert 5.0 < np.mean(v) < 200.0


def test_resampled_maps_admit_they_were_resampled(maps_small):
    for k, spec in SURFACE_MAPS.items():
        if k not in maps_small or spec["den"] == "32k":
            continue
        vd = maps_small[k].ledger.validity_domain
        assert vd["native_density"] == spec["den"]
        assert vd["resampled_to"] == "32k"
        assert "resampled" in maps_small[k].ledger.notes


def test_rank_and_zscore_helpers(maps_small):
    m = maps_small["sa_axis"]
    r = m.rank()
    assert np.nanmin(r) == pytest.approx(0.0)
    assert np.nanmax(r) == pytest.approx(1.0)
    z = m.zscored()
    assert abs(np.nanmean(z)) < 1e-9
    assert np.isnan(r[~m.coverage]).all()


def test_available_maps_registry_is_populated():
    reg = available_maps()
    assert "ei_proxy" in reg
    assert any(k.startswith("receptor_") for k in reg)
    assert set(SURFACE_MAPS) <= set(reg)


def test_mapset_roundtrip(tmp_path, maps_small):
    p = tmp_path / "m.npz"
    maps_small.save(p)
    q = MapSet.load(p)
    assert set(q.maps) == set(maps_small.maps)
    assert q.receptor_names == maps_small.receptor_names
    for k in q.maps:
        np.testing.assert_allclose(q[k].values, maps_small[k].values, equal_nan=True)
        assert q[k].ledger.bias_status == maps_small[k].ledger.bias_status


def test_mapset_rejects_unknown_keys(maps_small):
    with pytest.raises(KeyError):
        maps_small["not_a_map"]
