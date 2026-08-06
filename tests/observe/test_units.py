"""Unit round trips: volts, tesla, and dimensionless BOLD percent.

ARCHITECTURE.md Sec. 2 makes units required everywhere; refusal R01 rejects an
unknown unit.  These tests check that the units a read *declares* are the units
its numbers are actually in, by reconstructing the physical quantity from first
principles rather than by trusting the string.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.observe.base import ObservationRefusal, validate_unit
from scwbd.observe.bold import (
    BOLDObservationOperator,
    CanonicalHRF,
    fraction_to_percent,
    percent_to_fraction,
)
from scwbd.observe.eeg import EEGNoiseModel, EEGObservationOperator
from scwbd.observe.leadfield import meg_lead_field

torch.set_default_dtype(torch.float64)


def test_unknown_unit_is_refused():
    with pytest.raises(ObservationRefusal) as exc:
        validate_unit("microvolts_probably")
    assert exc.value.code == "R01"


def test_eeg_read_is_in_volts_of_the_right_magnitude(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    """A 10 nA*m cortical dipole must produce microvolt-scale scalp potentials."""
    normals = source_positions / source_positions.norm(dim=-1, keepdim=True)
    lf = four_layer_head.lead_field(source_positions, sensor_positions).project(normals)
    op = EEGObservationOperator(lf, dt=1e-3, noise=EEGNoiseModel(line_v=0.0), dtype=torch.float64)

    q = torch.zeros((source_positions.shape[0], 200), dtype=torch.float64)
    q[0] = 10e-9  # 10 nA*m, a canonical single-dipole ERP source
    r = op.observe(q, latent_temporal, seed=0, include_noise=False, include_artifacts=False)

    assert r.units == "V"
    peak = float(r.prediction.abs().max())
    assert 1e-7 < peak < 5e-5, (
        f"scalp potential from a 10 nA*m dipole is {peak * 1e6:.3f} uV, outside "
        "the physiological 0.1-50 uV range -- the lead field is not in V/(A*m)"
    )

    # explicit dimensional round trip: V = (V/(A*m)) * (A*m)
    L = lf.as_matrix()
    assert torch.allclose(r.prediction, L.to(torch.float64) @ q, rtol=1e-12, atol=1e-20)
    assert lf.as_psf().units == "V/(A*m)"


def test_meg_read_is_in_tesla_of_the_right_magnitude(source_positions):
    g = torch.Generator().manual_seed(3)
    pos = torch.randn(20, 3, generator=g, dtype=torch.float64)
    pos = pos / pos.norm(dim=1, keepdim=True) * 0.115
    lf = meg_lead_field(source_positions, pos, pos / pos.norm(dim=1, keepdim=True),
                        dtype=torch.float64)
    assert lf.sensor_units == "T"

    tangential = torch.zeros_like(source_positions)
    radial = source_positions / source_positions.norm(dim=-1, keepdim=True)
    tangential[:, 0] = 1.0
    tangential = tangential - (tangential * radial).sum(-1, keepdim=True) * radial
    tangential = tangential / tangential.norm(dim=-1, keepdim=True)

    q = 10e-9 * tangential
    b = torch.einsum("esk,sk->e", lf.matrix.to(torch.float64), q)
    peak = float(b.abs().max())
    assert 1e-15 < peak < 1e-11, (
        f"field from a 10 nA*m tangential dipole is {peak * 1e15:.2f} fT, outside "
        "the physiological 10-1000 fT range"
    )


def test_bold_percent_round_trip_is_exact(latent_temporal):
    g = torch.Generator().manual_seed(9)
    latent = torch.randn((6, 5000), generator=g, dtype=torch.float64)
    op = BOLDObservationOperator(
        n_elements=6, tr=1.0, hrf=CanonicalHRF(), thermal_noise_sd=0.0, dtype=torch.float64
    )
    frac = op.observe(latent, latent_temporal, seed=1, include_noise=False,
                      units="dimensionless")
    pct = op.observe(latent, latent_temporal, seed=1, include_noise=False, units="%")

    assert frac.units == "dimensionless"
    assert pct.units == "%"
    assert torch.allclose(pct.prediction, 100.0 * frac.prediction, rtol=0, atol=1e-14)
    assert torch.allclose(
        percent_to_fraction(fraction_to_percent(frac.prediction)),
        frac.prediction,
        rtol=0,
        atol=1e-15,
    )
    assert torch.allclose(
        percent_to_fraction(pct.prediction), frac.prediction, rtol=0, atol=1e-15
    )
    assert frac.ledger.validity_domain["units"] == "dimensionless"
    assert pct.ledger.validity_domain["units"] == "%"


def test_balloon_windkessel_readout_is_dimensionless_percent_scale(latent_temporal):
    from scwbd.observe.bold import (
        BalloonWindkesselReadout,
        reference_balloon_windkessel,
    )

    n = 20000
    drive = torch.zeros((3, n), dtype=torch.float64)
    drive[:, 2000:4000] = 1.0
    state = reference_balloon_windkessel(drive, dt=1e-3)

    op = BOLDObservationOperator(
        n_elements=3,
        tr=1.0,
        hrf=None,
        balloon=BalloonWindkesselReadout(),
        thermal_noise_sd=0.0,
        dtype=torch.float64,
    )
    r = op.observe_hemodynamic_state(state, latent_temporal, seed=0, include_noise=False)
    assert r.units == "dimensionless"
    peak = float(r.prediction.abs().max())
    assert 1e-3 < peak < 0.20, (
        f"Balloon-Windkessel BOLD peak is {peak * 100:.3f} %, outside the "
        "0.1-20 % range that a fractional signal change can plausibly take"
    )
    pct = op.observe_hemodynamic_state(
        state, latent_temporal, seed=0, include_noise=False, units="%"
    )
    assert torch.allclose(pct.prediction, 100.0 * r.prediction, rtol=0, atol=1e-12)


def test_fnirs_optical_density_is_dimensionless_and_extinction_units_convert():
    from scwbd.observe.fnirs import EXTINCTION_COEFF, FNIRSObservationOperator

    op = FNIRSObservationOperator(
        torch.tensor([8.0, 30.0, 30.0, 40.0], dtype=torch.float64), dtype=torch.float64
    )
    eps = op.extinction_matrix()
    # table is cm^-1 / M ; matrix must be mm^-1 / uM
    assert eps[0, 0] == pytest.approx(EXTINCTION_COEFF[760.0]["HbO"] / 1e7)
    # a 1 uM HbO change over a 60 mm path at 850 nm gives a plausible dOD
    dod = float(eps[1, 0]) * 1.0 * 60.0
    assert 1e-4 < dod < 1e-1, f"dOD {dod} is not in the measurable range"
    assert op.units == "dimensionless"


def test_variance_units_are_squared_signal_units(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    normals = source_positions / source_positions.norm(dim=-1, keepdim=True)
    lf = four_layer_head.lead_field(source_positions, sensor_positions).project(normals)
    op = EEGObservationOperator(lf, dt=1e-3, dtype=torch.float64)
    q = 1e-9 * torch.randn((source_positions.shape[0], 500), dtype=torch.float64)
    r = op.observe(q, latent_temporal, seed=2)
    assert r.ledger.variance.units == "V^2"
    assert r.sd() > 0.0
