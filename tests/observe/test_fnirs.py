"""fNIRS: photon path, partial volume, and extracerebral contamination.

The point of these tests is that the contamination is *modelled*.  A pipeline
that quietly assumes the whole optical-density change is cortical will pass a
shape test and fail every one of these.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import ObservationRefusal
from scwbd.observe.fnirs import FNIRSObservationOperator, PhotonPathModel

torch.set_default_dtype(torch.float64)


def test_short_separations_see_no_brain():
    pm = PhotonPathModel(extracerebral_thickness_mm=12.0)
    sep = torch.tensor([5.0, 8.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    cf = pm.cerebral_fraction(sep)
    assert float(cf[0]) < 0.01, "a 5 mm channel is reported as seeing cortex"
    assert float(cf[1]) < 0.02
    assert float(cf[2]) == 0.0, "a 10 mm channel is reported as seeing cortex"
    assert cf[-1] > cf[0]
    assert torch.all(torch.diff(cf) >= 0), "cerebral fraction must not fall with separation"
    assert torch.all(torch.diff(cf[3:]) > 0), "above onset it must grow strictly"


def test_cerebral_fraction_at_three_centimetres_is_a_minority():
    pm = PhotonPathModel(extracerebral_thickness_mm=12.0)
    cf = float(pm.cerebral_fraction(torch.tensor([30.0], dtype=torch.float64)))
    assert 0.05 < cf < 0.40, (
        f"cerebral path fraction at 3 cm is {cf:.3f}; the literature consensus is "
        "that a minority (roughly 15-25 %) of the fNIRS signal is cerebral"
    )


def test_partial_pathlengths_sum_to_the_total():
    pm = PhotonPathModel(dpf=6.0)
    sep = torch.tensor([8.0, 30.0, 45.0], dtype=torch.float64)
    ec, br = pm.partial_pathlengths_mm(sep)
    assert torch.allclose(ec + br, pm.total_pathlength_mm(sep), atol=1e-12)
    assert torch.all(ec > br), "the extracerebral path must dominate"


def test_photon_path_is_the_support_and_it_is_not_a_point():
    op = FNIRSObservationOperator(
        torch.tensor([8.0, 30.0], dtype=torch.float64), dtype=torch.float64
    )
    psf = op.support.psf
    assert psf is not None and psf.kind == "photon_path"
    assert psf.meta["shape"].startswith("banana")
    assert len(psf.meta["penetration_depth_mm"]) == 2


def test_extracerebral_signal_dominates_the_measurement(latent_temporal):
    """The dominant bias, quantified rather than assumed."""
    op = FNIRSObservationOperator(
        torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64), dt=0.1, dtype=torch.float64
    )
    n = 6000
    t = 1e-3 * torch.arange(n, dtype=torch.float64)
    task = (torch.sin(2 * math.pi * 0.05 * t) > 0).to(torch.float64)
    hb = {
        "HbO": 0.5 * task.unsqueeze(0).repeat(3, 1),
        "HbR": -0.15 * task.unsqueeze(0).repeat(3, 1),
    }
    r = op.observe(hb, latent_temporal, seed=0, task=task, include_noise=False)
    cereb = r.components["cerebral"]
    extra = r.components["extracerebral"]
    ratio = float(extra.abs().mean() / cereb.abs().mean().clamp_min(1e-30))
    assert ratio > 1.0, (
        f"extracerebral/cerebral amplitude ratio is {ratio:.2f}; the model does "
        "not reproduce the dominant systematic term"
    )
    assert r.ledger.bias_by_name("extracerebral_contamination") is not None


def test_without_short_channels_the_contamination_is_only_swept(latent_temporal):
    op = FNIRSObservationOperator(
        torch.tensor([30.0, 35.0], dtype=torch.float64), dt=0.1, dtype=torch.float64
    )
    hb = {
        "HbO": torch.randn((2, 4000), dtype=torch.float64),
        "HbR": torch.randn((2, 4000), dtype=torch.float64),
    }
    r = op.observe(hb, latent_temporal, seed=1)
    b = r.ledger.bias_by_name("extracerebral_contamination")
    assert b.status == "prior_specified_sensitivity"
    assert b.sensitivity_grid is not None and len(b.sensitivity_grid) >= 3


def test_with_short_channels_the_contamination_becomes_design_estimable(latent_temporal):
    op = FNIRSObservationOperator(
        torch.tensor([8.0, 30.0, 35.0], dtype=torch.float64), dt=0.1, dtype=torch.float64
    )
    hb = {
        "HbO": torch.randn((3, 4000), dtype=torch.float64),
        "HbR": torch.randn((3, 4000), dtype=torch.float64),
    }
    r = op.observe(hb, latent_temporal, seed=1)
    b = r.ledger.bias_by_name("extracerebral_contamination")
    assert b.status == "design_estimable"
    assert b.estimator and "short-separation" in b.estimator
    assert b.half_width < 1.0


def test_short_separation_regression_reduces_the_systemic_component(latent_temporal):
    op = FNIRSObservationOperator(
        torch.tensor([8.0, 30.0, 30.0, 35.0], dtype=torch.float64),
        dt=0.1,
        dtype=torch.float64,
    )
    n = 8000
    hb = {
        "HbO": torch.zeros((4, n), dtype=torch.float64),
        "HbR": torch.zeros((4, n), dtype=torch.float64),
    }
    r = op.observe(hb, latent_temporal, seed=2, include_noise=False)
    od = r.prediction.to(torch.float64)

    naive = op.recover_hb(od)
    cleaned = op.recover_hb(od, short_separation_regression=True)
    long_idx = (~op.short_channels).nonzero().flatten()
    a = float(naive["HbO"][long_idx].std())
    b = float(cleaned["HbO"][long_idx].std())
    assert b < a, "short-separation regression did not reduce the systemic variance"


def test_mbll_inversion_recovers_a_known_concentration_with_the_right_pathlength():
    op = FNIRSObservationOperator(
        torch.tensor([30.0], dtype=torch.float64), dtype=torch.float64
    )
    eps = op.extinction_matrix()
    L = torch.tensor([10.0], dtype=torch.float64)  # mm
    true = torch.tensor([[1.5], [-0.4]], dtype=torch.float64)  # uM HbO/HbR
    od = (eps @ true).unsqueeze(0) * L.reshape(-1, 1, 1)
    got = op.recover_hb(od, assumed_pathlength_mm=L)
    assert float(got["HbO"][0, 0]) == pytest.approx(1.5, rel=1e-9)
    assert float(got["HbR"][0, 0]) == pytest.approx(-0.4, rel=1e-9)


def test_wrong_pathlength_assumption_scales_the_answer(latent_temporal):
    """The DPF bias is multiplicative and the ledger says it is only swept."""
    op = FNIRSObservationOperator(
        torch.tensor([30.0], dtype=torch.float64), dtype=torch.float64
    )
    eps = op.extinction_matrix()
    true = torch.tensor([[1.0], [-0.3]], dtype=torch.float64)
    od = (eps @ true).unsqueeze(0) * 10.0
    a = op.recover_hb(od, assumed_pathlength_mm=torch.tensor([10.0], dtype=torch.float64))
    b = op.recover_hb(od, assumed_pathlength_mm=torch.tensor([20.0], dtype=torch.float64))
    assert float(a["HbO"][0, 0]) == pytest.approx(2.0 * float(b["HbO"][0, 0]), rel=1e-9)

    hb = {"HbO": torch.randn((1, 4000), dtype=torch.float64),
          "HbR": torch.randn((1, 4000), dtype=torch.float64)}
    r = op.observe(hb, latent_temporal, seed=0)
    assert r.ledger.bias_by_name("partial_pathlength_assumption").status == (
        "prior_specified_sensitivity"
    )


def test_wavelength_pair_conditioning_is_reported():
    good = FNIRSObservationOperator(
        torch.tensor([30.0], dtype=torch.float64), wavelengths_nm=(690.0, 830.0),
        dtype=torch.float64,
    )
    poor = FNIRSObservationOperator(
        torch.tensor([30.0], dtype=torch.float64), wavelengths_nm=(760.0, 850.0),
        dtype=torch.float64,
    )
    assert good.separation_condition_number() > 1.0
    assert poor.separation_condition_number() > 1.0
    assert good.separation_condition_number() != poor.separation_condition_number()


def test_untabulated_wavelength_is_refused():
    with pytest.raises(ObservationRefusal) as exc:
        FNIRSObservationOperator(
            torch.tensor([30.0], dtype=torch.float64), wavelengths_nm=(700.0, 800.0)
        )
    assert exc.value.code == "R01"
