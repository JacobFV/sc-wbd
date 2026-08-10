"""E-field -> latent drive -> predicted response. The coupling must not be decorative.

The test that decides whether this path is worth anything is
:class:`TestTwoPosesPredictDifferentResponses`: **two different coil positions
must produce different predicted responses, differing in the direction the
physics implies.**  If they did not, the coupling would be decorative in
exactly the sense agent Asimov found in the runtime, where three different
checkpoints produced byte-identical numbers -- a path that runs, returns
plausible arrays, and carries no information from its input.

So every claim here is paired with the control that makes it discriminating:

* different poses differ **and** an identical pose does not (so the difference
  is the pose, not the RNG);
* the peak-driven parcel is the one nearest the coil **and** moving the coil
  moves it (so the geometry is being read, not a fixed index);
* a purely tangential field produces ~no drive **and** a purely normal field
  produces a large one (so the *projection* is load-bearing, not the
  magnitude -- on the model's own 400 cortical parcels orientation carries 2.6x
  what a scalar does, and a magnitude discards all of it);
* halving coherence halves the drive (so agent Cajal's cancellation factor is
  read rather than declared);
* a zero field leaves the trajectory exactly equal to baseline (so a "response"
  cannot be the model's own ongoing activity misread as evoked).

This is a forward model. Nothing here optimises a coil position, ranks a
protocol, or recommends anything, and there is no dose: those are absent, not
disabled.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import ModelConfig, load_config
from scwbd.foundation.model import SCWBD
from scwbd.foundation.simulate import ThetaPrior
from scwbd.foundation.util import set_determinism
from scwbd.intervene.impulse_response import (
    UNTRAINED_PREDICTION_NOTICE,
    ImpulseResponse,
    ParcelDrive,
    build_latent_drive,
    parcel_drive,
    predict_impulse_response,
    pulse_time_course,
)

REPO = Path(__file__).resolve().parents[2]
_DT = torch.float32


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anat():
    """The real biological prior -- this is where Cajal's normals live."""
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def model(anat):
    set_determinism(0)
    cfg = load_config(REPO / "configs" / "scwbd_ci_smoke.yaml")
    cfg.model.n_regions = anat.n_regions
    return SCWBD(cfg.model, anat)


@pytest.fixture(scope="module")
def positions(anat):
    """A parcel position per region, on a 70 mm shell. Deterministic."""
    pos = getattr(anat, "position", None)
    if pos is not None and torch.isfinite(torch.as_tensor(pos)).all():
        p = torch.as_tensor(pos, dtype=_DT).reshape(anat.n_regions, 3)
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9) * 0.07
    g = torch.Generator().manual_seed(11)
    p = torch.randn(anat.n_regions, 3, generator=g)
    return p / p.norm(dim=-1, keepdim=True) * 0.07


def _efield_from_coil_centre(centre: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """A dipole-like field that falls off with distance. V/m.

    Not the gated solver -- these tests are about the *coupling*, and a
    closed-form falloff is what lets them assert a direction the physics
    implies without a 20-second BEM solve per case.
    """
    d = pos - centre.reshape(1, 3)
    r = d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    return 1e4 * d / r.pow(3) * 1e-6


COIL_A = torch.tensor([0.0, 0.0, 0.10])
COIL_B = torch.tensor([0.0, 0.10, 0.0])


def _drive(anat, pos, centre):
    return parcel_drive(
        _efield_from_coil_centre(centre, pos),
        anat.normal,
        coherence=anat.normal_coherence,
    )


def _predict(model, anat, drive, *, n_steps=8, gain=50.0, seed=1):
    B, C = 2, 4
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(B, C, anat.n_regions, generator=g)
    th = ThetaPrior().sample(B, seed=seed)
    return predict_impulse_response(
        model, drive, y_context=y, theta=th, n_steps=n_steps, gain=gain
    )


# ---------------------------------------------------------------------------
# 1. THE TEST THAT MATTERS
# ---------------------------------------------------------------------------


class TestTwoPosesPredictDifferentResponses:
    def test_two_coil_positions_give_different_predicted_eeg(
        self, model, anat, positions
    ):
        """The anti-decorative check. If this fails the path carries no signal."""
        a = _predict(model, anat, _drive(anat, positions, COIL_A))
        b = _predict(model, anat, _drive(anat, positions, COIL_B))
        assert not torch.allclose(a.eeg, b.eeg), (
            "two different coil positions produced the same predicted EEG; the "
            "field is not reaching the dynamics"
        )
        assert (a.eeg - b.eeg).abs().max() > 1e-6

    def test_the_same_pose_twice_gives_the_identical_response(
        self, model, anat, positions
    ):
        """The control. Without it, the difference above could be RNG."""
        d = _drive(anat, positions, COIL_A)
        a = _predict(model, anat, d, seed=1)
        b = _predict(model, anat, d, seed=1)
        assert torch.allclose(a.eeg, b.eeg)

    def test_the_evoked_response_is_a_difference_not_a_level(
        self, model, anat, positions
    ):
        """`evoked` is perturbed minus unperturbed, both from the same context."""
        r = _predict(model, anat, _drive(anat, positions, COIL_A))
        assert torch.allclose(r.evoked, r.eeg - r.baseline_eeg)
        assert r.peak_evoked_amplitude() > 0.0

    def test_a_zero_field_leaves_the_trajectory_exactly_at_baseline(
        self, model, anat, positions
    ):
        """The null. A 'response' must not be ongoing activity misread as evoked."""
        zero = parcel_drive(
            torch.zeros(anat.n_regions, 3),
            anat.normal,
            coherence=anat.normal_coherence,
        )
        r = _predict(model, anat, zero)
        assert float(r.drive.values.abs().max()) == 0.0
        assert torch.allclose(r.eeg, r.baseline_eeg), (
            "a zero field changed the trajectory; something other than the "
            "drive is differing between the two rollouts"
        )
        assert r.peak_evoked_amplitude() == 0.0


# ---------------------------------------------------------------------------
# 2. the direction the physics implies
# ---------------------------------------------------------------------------


class TestTheDifferenceIsInThePhysicalDirection:
    def test_the_peak_driven_parcel_is_near_the_coil(self, anat, positions):
        """Field falls off with distance, so the strongest drive is nearby.

        Asserted as a rank statistic rather than 'the single nearest parcel':
        the projection also depends on orientation, so the nearest parcel can
        legitimately lose to a slightly further one that is better aligned.
        Requiring the peak to be in the nearest decile tests the falloff
        without pretending orientation does not matter.
        """
        for centre in (COIL_A, COIL_B):
            d = _drive(anat, positions, centre)
            dist = (positions - centre.reshape(1, 3)).norm(dim=-1)
            covered = d.covered
            rank = (dist[covered] < dist[d.peak_parcel()]).sum().item()
            frac = rank / int(covered.sum())
            assert frac < 0.10, (
                f"peak parcel is at distance rank {frac:.2%} from the coil; "
                "the drive is not following the field falloff"
            )

    def test_moving_the_coil_moves_the_peak(self, anat, positions):
        a = _drive(anat, positions, COIL_A)
        b = _drive(anat, positions, COIL_B)
        assert a.peak_parcel() != b.peak_parcel()

    def test_a_more_distant_coil_drives_less(self, anat, positions):
        near = _drive(anat, positions, COIL_A)
        far = _drive(anat, positions, COIL_A * 3.0)
        assert float(far.values.abs().max()) < float(near.values.abs().max())

    def test_a_stronger_drive_gives_a_larger_predicted_response(
        self, model, anat, positions
    ):
        d = _drive(anat, positions, COIL_A)
        weak = _predict(model, anat, d, gain=10.0)
        strong = _predict(model, anat, d, gain=100.0)
        assert strong.peak_evoked_amplitude() > weak.peak_evoked_amplitude()


# ---------------------------------------------------------------------------
# 3. orientation is load-bearing, not decorative
# ---------------------------------------------------------------------------


class TestTheNormalProjectionIsWhatCouples:
    """A magnitude would pass every test above. These are what separate them."""

    def test_a_purely_tangential_field_barely_drives(self, anat):
        """Same magnitude, orthogonal to the normal -> near-zero drive.

        This is the test a field-magnitude implementation fails.
        """
        n = torch.as_tensor(anat.normal, dtype=_DT)
        cov = torch.isfinite(n).all(dim=-1)
        ref = torch.tensor([1.0, 0.0, 0.0]).expand_as(n)
        tang = torch.cross(n.nan_to_num(), ref, dim=-1)
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        normal_field = n.nan_to_num() * 50.0
        tangential_field = tang * 50.0

        d_norm = parcel_drive(normal_field, anat.normal)
        d_tang = parcel_drive(tangential_field, anat.normal)

        assert float(d_norm.projection_v_per_m[cov].abs().mean()) > 40.0
        assert float(d_tang.projection_v_per_m[cov].abs().mean()) < 1.0

    def test_reversing_the_field_reverses_the_sign(self, anat, positions):
        e = _efield_from_coil_centre(COIL_A, positions)
        fwd = parcel_drive(e, anat.normal, coherence=anat.normal_coherence)
        rev = parcel_drive(-e, anat.normal, coherence=anat.normal_coherence)
        assert torch.allclose(fwd.values, -rev.values, atol=1e-6)
        assert float(fwd.values.min()) < 0.0 < float(fwd.values.max()), (
            "drive is single-signed; a signed projection should produce both "
            "inward and outward driven parcels"
        )

    def test_inward_field_is_positive_drive(self, anat):
        """Sign convention, pinned: matches runtime NormalComponentResponse."""
        n = torch.as_tensor(anat.normal, dtype=_DT)
        cov = torch.isfinite(n).all(dim=-1)
        inward = -n.nan_to_num() * 10.0
        d = parcel_drive(inward, anat.normal)
        assert float(d.projection_v_per_m[cov].min()) > 0.0


class TestCoherenceIsRead:
    def test_halving_coherence_halves_the_drive(self, anat, positions):
        e = _efield_from_coil_centre(COIL_A, positions)
        coh = torch.as_tensor(anat.normal_coherence, dtype=_DT)
        full = parcel_drive(e, anat.normal, coherence=coh)
        half = parcel_drive(e, anat.normal, coherence=coh * 0.5)
        assert torch.allclose(half.values, full.values * 0.5, atol=1e-6)

    def test_zero_coherence_kills_the_drive(self, anat, positions):
        e = _efield_from_coil_centre(COIL_A, positions)
        d = parcel_drive(e, anat.normal, coherence=torch.zeros(anat.n_regions))
        assert float(d.values.abs().max()) == 0.0
        # but the raw projection is retained, so the cancellation is visible
        assert float(d.projection_v_per_m.abs().max()) > 0.0

    def test_coherence_is_actually_non_degenerate_on_the_real_prior(self, anat):
        """If coherence were ~1 everywhere, weighting by it would be a no-op."""
        coh = torch.as_tensor(anat.normal_coherence, dtype=_DT)
        cov = torch.isfinite(coh)
        assert float(coh[cov].min()) < 0.9
        assert float(coh[cov].std()) > 0.01


# ---------------------------------------------------------------------------
# 4. uncovered parcels, and the pad
# ---------------------------------------------------------------------------


class TestUncoveredParcelsAreZeroNotNaN:
    def test_subcortex_gets_exactly_zero(self, anat, positions):
        d = _drive(anat, positions, COIL_A)
        assert torch.isfinite(d.values).all(), "a NaN drive would be silently zeroed later"
        assert float(d.values[~d.covered].abs().max()) == 0.0
        assert d.n_covered < d.n_parcels, "expected some uncovered parcels"

    def test_a_nan_drive_is_refused_rather_than_injected(self, model, anat):
        bad = ParcelDrive(
            values=torch.full((anat.n_regions,), float("nan")),
            projection_v_per_m=torch.zeros(anat.n_regions),
            coherence=torch.zeros(anat.n_regions),
            covered=torch.zeros(anat.n_regions, dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="non-finite"):
            build_latent_drive(model, bad, n_steps=4)


class TestTheDriveNeverWritesThePad:
    """Family layout: ``u`` is added to ``dx`` unmasked, so this is a real risk.

    Uses the **synthetic** prior, as ``tests/foundation/test_family_state.py``
    does, because ``derive_families`` currently refuses the real biological
    prior: ``AnatomyPrior.families`` there is a 9-element list of family
    *names*, while ``derive_families`` reads any ``families`` attribute as
    per-parcel labels and raises on the length mismatch. That is a defect in
    the families adapter, not in this path, and it is reported rather than
    worked around -- ``family_state=True`` cannot presently load real anatomy.
    The pad guard is a property of the layout, so the synthetic prior tests it
    exactly as well.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def fam_model(cls):
        set_determinism(0)
        a = load_anatomy(device="cpu", force_fallback=True)
        cfg = ModelConfig(
            family_state=True, hidden=64, n_local_layers=2, region_embed=16,
            context_dim=32, encoder_channels=16, encoder_layers=2,
            # The synthetic fallback anatomy declares no family partition, so the
            # model has to DERIVE one -- and deriving is refused by default now
            # that `allow_derived` is enforced rather than decorative. This test
            # is about pad-cleanliness under a family layout, not about which
            # partition is used, so opting in is the honest declaration: it says
            # a rejected partition is acceptable here, which it is.
            family_allow_derived_partition=True,
        )
        m = SCWBD(cfg, a)
        assert m.family_layout is not None, "expected a family-state model"
        return m, a

    @staticmethod
    def _synthetic_drive(a):
        """Deterministic normals; the synthetic prior carries none."""
        g = torch.Generator().manual_seed(5)
        n = torch.randn(a.n_regions, 3, generator=g)
        n = n / n.norm(dim=-1, keepdim=True)
        n[int(a.n_regions * 0.8):] = float("nan")     # some uncovered, as in reality
        e = torch.randn(a.n_regions, 3, generator=g) * 40.0
        return parcel_drive(e, n)

    def test_family_rollout_pad_stays_clean(self, fam_model):
        m, a = fam_model
        theta = torch.randn(2, 6) * 0.2
        m.set_mechanistic_theta(theta, a)
        r = predict_impulse_response(
            m, self._synthetic_drive(a),
            y_context=torch.randn(2, 8, a.n_regions) * 0.1,
            theta=theta, n_steps=6, gain=50.0,
        )
        m.family_layout.assert_clean(r.state, where="test")
        assert torch.isfinite(r.state).all()

    def test_the_built_u_is_zero_on_every_pad_channel(self, fam_model):
        m, a = fam_model
        u = build_latent_drive(m, self._synthetic_drive(a), n_steps=5, batch=2)
        pad = m.family_layout.pad_mask(dtype=torch.bool)
        assert float(u[..., pad].abs().max()) == 0.0
        assert float(u.abs().max()) > 0.0, "u is entirely zero; nothing was injected"

    def test_the_pad_guard_is_not_vacuous(self, fam_model):
        """A deliberately full-width drive DOES trip the guard.

        Without this the two tests above pass for a `u` that is all zeros, or
        for a layout with no pad at all.
        """
        from scwbd.foundation.families import SpanViolation

        m, a = fam_model
        pad = m.family_layout.pad_mask(dtype=torch.bool)
        if not bool(pad.any()):
            pytest.skip("this partition has no pad channels")
        bad = torch.ones(1, 1, a.n_regions, m.layout.dim)
        with pytest.raises(SpanViolation):
            m.family_layout.assert_clean(bad, where="deliberate pad write")


# ---------------------------------------------------------------------------
# 5. the pulse envelope
# ---------------------------------------------------------------------------


class TestThePulseEnvelope:
    def test_it_is_area_normalised(self):
        for dt in (1e-3, 5e-4, 1e-4):
            env = pulse_time_course(64, dt_s=dt)
            assert math.isclose(float(env.sum()), 1.0, rel_tol=1e-5)

    def test_the_delivered_impulse_is_invariant_to_timestep(self):
        """Halving dt must not halve the effect -- otherwise a result is not
        comparable across integration settings."""
        a = pulse_time_course(64, dt_s=1e-3).sum()
        b = pulse_time_course(64, dt_s=5e-4).sum()
        assert math.isclose(float(a), float(b), rel_tol=1e-5)

    def test_it_is_zero_before_onset(self):
        env = pulse_time_course(16, onset_step=4)
        assert float(env[:4].abs().max()) == 0.0
        assert float(env[4:].sum()) > 0.0

    def test_an_onset_outside_the_window_is_refused(self):
        with pytest.raises(ValueError, match="outside"):
            pulse_time_course(8, onset_step=8)


# ---------------------------------------------------------------------------
# 6. what the result claims
# ---------------------------------------------------------------------------


class TestTheResultSaysWhatItIs:
    def test_every_prediction_carries_the_untrained_notice(self, model, anat, positions):
        r = _predict(model, anat, _drive(anat, positions, COIL_A))
        assert r.notice == UNTRAINED_PREDICTION_NOTICE
        assert "not a dose" in r.notice
        assert "never been fitted to perturbational data" in r.notice

    def test_provenance_records_that_the_mapping_is_unvalidated(
        self, model, anat, positions
    ):
        p = _predict(model, anat, _drive(anat, positions, COIL_A)).provenance
        assert p["response_mapping_validated"] is False
        assert p["trained_on_perturbation_data"] is False
        assert p["field_solver_gates"] == ["N3", "N4", "N6", "N8"]

    def test_it_offers_no_recommendation_surface(self):
        """Forward model. The absence is structural, not a disabled feature."""
        for banned in (
            "recommend", "rank", "optimise", "optimize", "best_pose",
            "choose", "select_target", "dose",
        ):
            assert not hasattr(ImpulseResponse, banned)
            assert not hasattr(ParcelDrive, banned)

    def test_the_summary_is_json_shaped(self, model, anat, positions):
        import json

        s = _predict(model, anat, _drive(anat, positions, COIL_A)).summary()
        json.dumps(s)
        assert s["drive"]["n_covered"] > 0
