"""The EEG head: T2 exactness, reference operators, impedance, artifacts,
and the propagation of electrode-position covariance into sensor variance.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import ObservationRefusal, TemporalSupport
from scwbd.observe.eeg import (
    ArtifactModel,
    EEGNoiseModel,
    EEGObservationOperator,
    pink_noise,
)
from scwbd.observe.leadfield import (
    ElectrodeImpedance,
    ElectrodePositionUncertainty,
    ReferenceOperator,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture
def fixed_lf(four_layer_head, sensor_positions, source_positions):
    normals = source_positions / source_positions.norm(dim=-1, keepdim=True)
    return four_layer_head.lead_field(source_positions, sensor_positions).project(normals)


def test_t2_linear_gaussian_is_the_exact_special_case(fixed_lf, latent_temporal):
    """``y_E[k] = L x(k dt_E) + eps_E[k]`` with ``eps ~ N(0, R)``."""
    op = EEGObservationOperator(
        fixed_lf, dt=1e-3, noise=EEGNoiseModel(line_v=0.0), dtype=torch.float64
    )
    L, R = op.linear_gaussian()
    assert L.shape == (fixed_lf.n_sensors, fixed_lf.n_sources)
    assert R.shape == (fixed_lf.n_sensors, fixed_lf.n_sensors)
    assert torch.allclose(R, R.T, atol=1e-30)
    assert float(torch.linalg.eigvalsh(R).min()) > 0.0

    x = 1e-9 * torch.randn((fixed_lf.n_sources, 500), dtype=torch.float64)
    r = op.observe(x, latent_temporal, seed=0, include_noise=False, include_artifacts=False)
    assert torch.allclose(r.prediction, L @ x, rtol=1e-12, atol=1e-20)


def test_t2_noise_covariance_matches_the_generated_noise(fixed_lf, latent_temporal):
    """The declared ``R`` must be the covariance actually simulated."""
    op = EEGObservationOperator(
        fixed_lf,
        dt=1e-3,
        noise=EEGNoiseModel(line_v=0.0, background_exponent=0.0, background_source_am=3e-9),
        dtype=torch.float64,
    )
    _, R = op.linear_gaussian(include_background=True)
    n = 200000
    x = torch.zeros((fixed_lf.n_sources, n), dtype=torch.float64)
    r = op.observe(x, latent_temporal, seed=7, include_noise=True, include_artifacts=False)
    emp = torch.cov(r.prediction)
    rel = float((emp - R).norm() / R.norm())
    assert rel < 0.15, f"empirical noise covariance differs from declared R by {rel:.3f}"


def test_pink_noise_has_a_one_over_f_spectrum():
    x = pink_noise((4, 2**16), dt=1e-3, exponent=1.0, seed=1, f_knee=0.1)
    S = (torch.fft.rfft(x, dim=-1).abs() ** 2).mean(0)
    f = torch.fft.rfftfreq(x.shape[-1], d=1e-3)
    m = (f > 1.0) & (f < 100.0)
    slope = float(
        torch.linalg.lstsq(
            torch.stack([torch.log(f[m]), torch.ones(int(m.sum()), dtype=torch.float64)], -1),
            torch.log(S[m]).unsqueeze(-1),
        ).solution[0, 0]
    )
    assert -2.3 < slope < -1.7, f"1/f noise slope is {slope:.2f}, expected about -2"


def test_reference_operators_are_explicit_matrices(fixed_lf):
    n = fixed_lf.n_sensors
    avg = ReferenceOperator.average(n, dtype=torch.float64)
    x = torch.randn((n, 50), dtype=torch.float64)
    y = avg.apply(x)
    assert float(y.mean(0).abs().max()) < 1e-12
    assert avg.bias_term().status == "prior_specified_sensitivity"

    lm = ReferenceOperator.linked_mastoid(n, [0, 1], dtype=torch.float64)
    y2 = lm.apply(x)
    assert torch.allclose(y2[0], x[0] - 0.5 * (x[0] + x[1]), atol=1e-14)

    rest = ReferenceOperator.rest(fixed_lf, dtype=torch.float64)
    assert rest.matrix.shape == (n, n)
    assert rest.bias_term().status == "externally_bounded"


def test_rest_recovers_infinity_reference_for_a_source_in_the_model(fixed_lf):
    """REST is only as good as the lead field -- and that far, it works."""
    L = fixed_lf.as_matrix().to(torch.float64)
    x = torch.zeros((L.shape[1], 20), dtype=torch.float64)
    x[4] = 1e-8
    v_inf = L @ x
    v_avg = v_inf - v_inf.mean(0, keepdim=True)
    rest = ReferenceOperator.rest(fixed_lf, dtype=torch.float64)
    recovered = rest.apply(v_avg)
    rel = float((recovered - v_inf).norm() / v_inf.norm())
    assert rel < 1e-6, f"REST failed to invert its own lead field: {rel:.3e}"


def test_impedance_changes_gain_and_adds_johnson_noise(fixed_lf):
    n = fixed_lf.n_sensors
    z = torch.linspace(1e3, 1e5, n, dtype=torch.float64)
    imp = ElectrodeImpedance(z_electrode=z, z_input=1e6)
    assert float(imp.gain.max()) < 1.0
    assert float(imp.gain.min()) > 0.0
    assert imp.gain[0] > imp.gain[-1]
    v = imp.thermal_noise_variance
    assert float(v.min()) > 0.0
    # Johnson noise of a 100 kOhm electrode over 100 Hz is ~0.4 uV rms
    assert 1e-8 < float(v[-1].sqrt()) < 1e-5
    assert imp.imbalance_bias().status == "design_estimable"


def test_artifacts_are_generated_with_realistic_topography(fixed_lf, latent_temporal):
    op = EEGObservationOperator(
        fixed_lf, dt=1e-3, artifacts=ArtifactModel(), noise=EEGNoiseModel(line_v=0.0),
        dtype=torch.float64,
    )
    x = torch.zeros((fixed_lf.n_sources, 20000), dtype=torch.float64)
    r = op.observe(x, latent_temporal, seed=3, include_noise=False, include_artifacts=True)
    oc = r.components["ocular"]
    mu = r.components["muscle"]
    assert float(oc.abs().max()) > 1e-6, "no ocular artifact was generated"
    assert float(mu.abs().max()) > 1e-7, "no muscle artifact was generated"
    # muscle is high frequency, ocular is low frequency
    def centroid(sig):
        S = (torch.fft.rfft(sig, dim=-1).abs() ** 2).sum(0)
        f = torch.fft.rfftfreq(sig.shape[-1], d=1e-3)
        return float((S * f).sum() / S.sum())

    assert centroid(mu) > 5 * centroid(oc)
    assert r.ledger.bias_by_name("artifact_cleaning_residual") is not None


def test_line_noise_appears_at_the_declared_frequency(fixed_lf, latent_temporal):
    op = EEGObservationOperator(
        fixed_lf,
        dt=1e-3,
        noise=EEGNoiseModel(
            line_v=5e-6, line_hz=50.0, line_harmonics=2, sensor_white_v_per_rthz=0.0,
            background_source_am=0.0,
        ),
        dtype=torch.float64,
    )
    x = torch.zeros((fixed_lf.n_sources, 10000), dtype=torch.float64)
    r = op.observe(x, latent_temporal, seed=4, include_noise=True, include_artifacts=False)
    S = (torch.fft.rfft(r.prediction, dim=-1).abs() ** 2).sum(0)
    f = torch.fft.rfftfreq(10000, d=1e-3)
    peak = float(f[int(S.argmax())])
    assert abs(peak - 50.0) < 0.2, f"line-noise peak at {peak} Hz, expected 50 Hz"


def test_electrode_position_covariance_propagates_with_cross_terms(
    four_layer_head, sensor_positions, source_positions
):
    lf = four_layer_head.lead_field(source_positions, sensor_positions)
    sub_sens = sensor_positions[:8]
    lf8 = four_layer_head.lead_field(source_positions[:3], sub_sens)

    indep = ElectrodePositionUncertainty(
        ElectrodePositionUncertainty.isotropic(8, 3e-3)
    )
    shared = ElectrodePositionUncertainty(
        ElectrodePositionUncertainty.with_common_mode(
            sub_sens, independent_sd_m=3e-3, rotation_sd_rad=0.05
        )
    )
    q = 1e-8 * torch.randn((3, 3), dtype=torch.float64)
    op = EEGObservationOperator(
        lf8, dt=1e-3, position_uncertainty=indep, head_model=four_layer_head,
        dtype=torch.float64,
    )
    op_shared = EEGObservationOperator(
        lf8, dt=1e-3, position_uncertainty=shared, head_model=four_layer_head,
        dtype=torch.float64,
    )
    S1 = op.sensor_variance_from_electrode_positions(q)
    S2 = op_shared.sensor_variance_from_electrode_positions(q)

    assert S1.shape == (8, 8)
    assert torch.allclose(S1, S1.T, atol=1e-30)
    assert float(torch.linalg.eigvalsh(S1).min()) > -1e-24

    off1 = float((S1 - torch.diag(torch.diagonal(S1))).abs().sum())
    off2 = float((S2 - torch.diag(torch.diagonal(S2))).abs().sum())
    assert off2 > off1, (
        "a shared cap-pose error must induce cross-electrode covariance; dropping "
        "those cross terms is a bug, not an optimisation (ARCHITECTURE.md Sec. 3)"
    )
    # ... and it does not average away with more channels
    assert float(torch.diagonal(S2).mean()) > float(torch.diagonal(S1).mean())


def test_position_uncertainty_without_a_head_model_refuses(fixed_lf):
    op = EEGObservationOperator(fixed_lf, dt=1e-3, dtype=torch.float64)
    with pytest.raises(ObservationRefusal):
        op.sensor_variance_from_electrode_positions(torch.zeros((12, 3), dtype=torch.float64))


def test_observe_is_deterministic_given_a_seed(fixed_lf, latent_temporal):
    op = EEGObservationOperator(fixed_lf, dt=1e-3, artifacts=ArtifactModel(),
                                dtype=torch.float64)
    x = 1e-9 * torch.randn((fixed_lf.n_sources, 3000), dtype=torch.float64)
    a = op.observe(x, latent_temporal, seed=99)
    b = op.observe(x, latent_temporal, seed=99)
    c = op.observe(x, latent_temporal, seed=100)
    assert torch.equal(a.prediction, b.prediction)
    assert not torch.equal(a.prediction, c.prediction)


def test_mismatched_source_space_returns_unresolved_not_a_number(fixed_lf, latent_temporal):
    from scwbd.observe.base import Unresolved

    op = EEGObservationOperator(fixed_lf, dt=1e-3, dtype=torch.float64)
    x = torch.zeros((fixed_lf.n_sources + 5, 100), dtype=torch.float64)
    out = op.observe(x, latent_temporal, seed=0)
    assert isinstance(out, Unresolved)
    assert not out
    assert "source" in out.reason


def test_joint_t5_propagation_uses_agent_d_and_keeps_the_cross_term(
    four_layer_head, sensor_positions, source_positions
):
    """Electrode position and conductivity are *both* shared session calibration.

    Their cross term is the one thesis T5 and ARCHITECTURE.md Sec. 3 forbid
    dropping.  The propagation algebra is agent D's; only the physics Jacobians
    are computed here.  ``Sigma_xc`` is *built from* a shared calibration factor
    rather than asserted as a correlation, which is what keeps the joint
    covariance PSD (agent D refuses an inconsistent one, correctly).
    """
    pytest.importorskip("scwbd.transforms.uncertainty")
    from scwbd.observe.leadfield import ITIS_CONDUCTIVITY, SphericalHeadModel

    # the shared fixture pins conductivity to delta priors so it can be compared
    # against MNE; here we need the real IT'IS spreads, because a session-shared
    # calibration term with zero variance has no cross term to drop.
    head = SphericalHeadModel(radii=four_layer_head.radii, conductivity=ITIS_CONDUCTIVITY)
    sub_sens = sensor_positions[:6]
    q = 1e-8 * torch.randn((3, 3), dtype=torch.float64)

    # one shared session calibration factor: a head-size / digitiser scale error
    # that dilates every electrode radius AND biases every fitted conductivity.
    n_p = 3 * sub_sens.shape[0]
    sd_c = torch.tensor([p.sd for p in head.conductivity.priors], dtype=torch.float64)
    assert float(sd_c.min()) > 0.0
    n_c = sd_c.numel()

    A = (sub_sens / sub_sens.norm(dim=-1, keepdim=True)).reshape(-1, 1)  # (n_p, 1)
    s_cal = 1.5e-3  # metres of radial scale error, 1 sigma
    # a 1-sigma scale error also shifts every fitted conductivity by 10 %:
    # head-size mis-estimation and conductivity fitting are not independent.
    means = torch.tensor(head.conductivity.means(), dtype=torch.float64)
    B = (0.10 * means / s_cal).reshape(-1, 1)  # (n_c, 1), S/m per metre

    Sx = ElectrodePositionUncertainty.isotropic(sub_sens.shape[0], 3e-4) + (
        s_cal**2
    ) * (A @ A.T)
    Sc = torch.diag(sd_c**2) + (s_cal**2) * (B @ B.T)
    Sxc = (s_cal**2) * (A @ B.T)
    assert Sxc.shape == (n_p, n_c)
    assert float(Sxc.abs().max()) > 0.0

    pu = ElectrodePositionUncertainty(Sx)
    with_cross = pu.propagate(
        head, source_positions[:3], sub_sens, q, conductivity_cov=Sc, cross_cov=Sxc
    )
    without = pu.propagate(
        head,
        source_positions[:3],
        sub_sens,
        q,
        conductivity_cov=Sc,
        cross_cov=Sxc,
        include_cross=False,
    )
    assert with_cross.cov.shape == (6, 6)
    assert with_cross.method.startswith("first_order")
    assert float(with_cross.cross_term.abs().max()) > 0.0, (
        "J_x Sigma_xc J_c^T is identically zero, so the cross term is not being "
        "computed at all"
    )
    d = float((with_cross.cov - without.cov).abs().max())
    assert d > 0.0, (
        "dropping J_x Sigma_xc J_c^T changed nothing, so the cross term is not "
        "actually being applied"
    )

    # and agent D's own diagnostic must agree that assuming independence
    # misstates the aggregate uncertainty
    from scwbd.transforms.uncertainty import independence_understatement

    Jp = pu.jacobian(head, source_positions[:3], sub_sens, q)
    Jc = pu.conductivity_jacobian(head, source_positions[:3], sub_sens, q)
    p0 = sub_sens.reshape(-1).to(torch.float64)
    c0 = torch.tensor(head.conductivity.means(), dtype=torch.float64)
    y0 = torch.einsum(
        "esk,sk->e", head.potential(source_positions[:3], sub_sens), q
    )
    stats = independence_understatement(
        lambda pp, cc: y0 + Jp @ (pp - p0) + Jc @ (cc - c0),
        p0,
        c0,
        Sx=Sx,
        Sc=Sc,
        Sxc=Sxc,
    )
    assert stats["trace_ratio"] != 1.0, (
        "the independence assumption costs exactly nothing here, which would "
        f"mean the cross term is inert: {stats}"
    )

    # conductivity alone must contribute: MEG would not, EEG does
    pos_only = pu.sensor_covariance(head, source_positions[:3], sub_sens, q)
    assert float(torch.trace(with_cross.cov)) > float(torch.trace(pos_only))


def test_conductivity_jacobian_is_nonzero_and_skull_dominates(
    four_layer_head, sensor_positions, source_positions
):
    pu = ElectrodePositionUncertainty(
        ElectrodePositionUncertainty.isotropic(6, 1e-3)
    )
    from scwbd.observe.leadfield import ITIS_CONDUCTIVITY, SphericalHeadModel

    head = SphericalHeadModel(radii=four_layer_head.radii, conductivity=ITIS_CONDUCTIVITY)
    q = 1e-8 * torch.randn((3, 3), dtype=torch.float64)
    J = pu.conductivity_jacobian(head, source_positions[:3], sensor_positions[:6], q)
    assert J.shape == (6, 4)
    assert float(J.abs().max()) > 0.0
    # scaled by each layer's prior spread, the skull is the dominant contributor
    sd = torch.tensor([p.sd for p in head.conductivity.priors], dtype=torch.float64)
    if float(sd.sum()) > 0:
        contrib = (J.abs() * sd).sum(0)
        # brain conductivity enters the potential as an explicit 1/sigma_1 factor
        # and carries a wide prior, so it or the skull dominates; CSF has a tight
        # prior (Baumann 1997) and must not be the largest contributor.
        assert int(contrib.argmax()) in (0, 2), (
            f"conductivity sensitivity ranking {contrib.tolist()} puts CSF or "
            "scalp first, which contradicts published EEG head-model sensitivity "
            "analyses"
        )
        assert int(contrib.argmax()) != 1
