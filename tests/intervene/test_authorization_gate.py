"""The governance gate in front of :math:`\\mathcal A_{\\rm safe}`. SIMULATION ONLY.

Two properties, and the second is the one that matters:

1. a validated ``AuthorizationRecord`` admits a request that is inside
   :math:`\\mathcal A_{\\rm safe}`, and the admitted proposal carries the
   protocol it was performed under in its provenance;
2. **the gate still refuses.**  An authorized request outside
   :math:`\\mathcal A_{\\rm safe}` refuses ``R11`` exactly as an unauthorized
   one does, the feasible set is bit-identical with and without a record, and
   the limits remain unlearnable and unwritable.  A gate nobody has seen refuse
   is indistinguishable from one that cannot.

Nothing here authorises stimulating anybody.  There is no controller, no
device path and no dose in this module or its subject; a record is an input to
a refusal.
"""

from __future__ import annotations

import pytest

from scwbd.intervene.safety import (
    AuthorizationGate,
    AuthorizedRequest,
    CompilerRefusal,
    FeasibleSet,
    ProposedIntervention,
    SafetyLimits,
)
from scwbd.schema.authorization import epoch_seconds

WITHIN_WINDOW_S = epoch_seconds("2026-08-05")
AFTER_WINDOW_S = epoch_seconds("2027-06-01")

#: Axes the proposals below are checked on; the fixture protocol declares all
#: of them, which is what makes its A_safe attributable to the request.
_EXPOSURE = {
    "tms.peak_efield_v_per_m": 95.0,
    "tms.pulses_per_session": 600.0,
    "tms.coil_scalp_distance_mm": 4.0,
}


def _proposal(label: str = "pose_a", modality: str = "tms", **exposure):
    base = dict(_EXPOSURE)
    base.update(exposure)
    return ProposedIntervention(
        label=label,
        modality=modality,
        exposure=base,
        pose_certified=True,
        reversible=True,
    )


@pytest.fixture
def gate() -> AuthorizationGate:
    return AuthorizationGate(
        FeasibleSet(), a_safe_id="EX-TMS-DLPFC-01-asafe"
    )


def _request(record, cls: str = "tms", at: float = WITHIN_WINDOW_S) -> AuthorizedRequest:
    return AuthorizedRequest(record=record, intervention_class=cls, at_time_s=at)


# ---------------------------------------------------------------------------
# 1. what admits
# ---------------------------------------------------------------------------


class TestAValidatedRecordAdmitsInsideASafe:
    def test_an_authorized_in_bounds_proposal_is_admitted(self, gate, authorization):
        admitted = gate.admit(_proposal(), _request(authorization))
        assert admitted.verdict.feasible
        assert admitted.authorization.admitted
        assert admitted.claim_scope == "protocol:EX-TMS-DLPFC-01@3.2"

    def test_the_admitted_proposal_carries_the_protocol_in_provenance(
        self, gate, authorization
    ):
        prov = gate.admit(_proposal(), _request(authorization)).provenance()
        assert prov["claim_scope"] == "protocol:EX-TMS-DLPFC-01@3.2"
        assert prov["authorization"]["record_hash"] == authorization.content_hash()
        assert prov["authorization"]["approval_identifier"] == "IRB-2026-0417"
        assert set(prov["a_safe_axes_checked"]) == set(_EXPOSURE)
        assert "SIMULATION" in prov["notice"].upper()
        assert "DECLARATION, NOT VERIFICATION" in prov["authorization_notice"]


# ---------------------------------------------------------------------------
# 2. the negative controls: the gate can still refuse
# ---------------------------------------------------------------------------


class TestTheGateStillRefuses:
    def test_an_authorized_proposal_outside_a_safe_still_refuses_r11(
        self, gate, authorization
    ):
        """The negative control. Authorization is not a bypass.

        The coil sits 90 mm off the scalp -- far outside the declared bound --
        under a fully valid approval.  It refuses, and the refusal names the
        axis, not the governance.
        """
        far = _proposal("pose_far", **{"tms.coil_scalp_distance_mm": 90.0})
        with pytest.raises(CompilerRefusal) as excinfo:
            gate.admit(far, _request(authorization))
        exc = excinfo.value
        assert exc.code == "R11"
        assert "tms.coil_scalp_distance_mm" in str(exc)
        assert "authorization_failures" not in exc.evidence

    @pytest.mark.parametrize(
        "axis,value",
        [
            ("tms.peak_efield_v_per_m", 1.0e4),
            ("tms.pulses_per_session", 1.0e5),
            ("tms.coil_scalp_distance_mm", 90.0),
        ],
    )
    def test_every_axis_still_blocks_under_authorization(
        self, gate, authorization, axis, value
    ):
        with pytest.raises(CompilerRefusal):
            gate.admit(_proposal("bad", **{axis: value}), _request(authorization))

    def test_an_expired_approval_refuses_before_a_safe_is_even_reached(
        self, gate, authorization
    ):
        with pytest.raises(CompilerRefusal) as excinfo:
            gate.admit(_proposal(), _request(authorization, at=AFTER_WINDOW_S))
        codes = [f["code"] for f in excinfo.value.evidence["authorization_failures"]]
        assert codes == ["AUTH_EXPIRED"]

    def test_a_tms_approval_does_not_admit_a_tfus_proposal(self, gate, authorization):
        with pytest.raises(CompilerRefusal) as excinfo:
            gate.admit(_proposal(modality="tfus"), _request(authorization, cls="tfus"))
        codes = [f["code"] for f in excinfo.value.evidence["authorization_failures"]]
        assert "AUTH_CLASS_NOT_AUTHORIZED" in codes

    def test_a_proposal_of_a_class_the_request_did_not_name_refuses(
        self, gate, authorization
    ):
        with pytest.raises(CompilerRefusal, match="does not match"):
            gate.admit(_proposal(modality="tfus"), _request(authorization, cls="tms"))

    def test_no_record_refuses_with_absent_not_with_a_belief(self, gate):
        with pytest.raises(CompilerRefusal) as excinfo:
            gate.admit(_proposal(), _request(None))
        codes = [f["code"] for f in excinfo.value.evidence["authorization_failures"]]
        assert codes == ["AUTH_ABSENT"]

    def test_an_uncertified_pose_is_still_outside_a_safe_under_authorization(
        self, gate, authorization
    ):
        loose = ProposedIntervention(
            label="scalp_label_only",
            modality="tms",
            exposure=dict(_EXPOSURE),
            pose_certified=False,
        )
        with pytest.raises(CompilerRefusal, match="pose_certified"):
            gate.admit(loose, _request(authorization))


# ---------------------------------------------------------------------------
# 3. authorization never widens A_safe
# ---------------------------------------------------------------------------


class TestAuthorizationNeverWidensASafe:
    def test_the_verdict_is_identical_with_and_without_a_record(
        self, gate, authorization
    ):
        """``FeasibleSet.contains`` is not passed the record and cannot use it."""
        far = _proposal("pose_far", **{"tms.coil_scalp_distance_mm": 90.0})
        plain = FeasibleSet().contains(far)
        gated = gate.feasible_set.contains(far)
        assert plain.feasible == gated.feasible is False
        assert [str(v) for v in plain.violations] == [str(v) for v in gated.violations]

    def test_contains_takes_no_authorization_argument(self):
        import inspect

        params = set(inspect.signature(FeasibleSet.contains).parameters)
        assert params == {"self", "proposal"}

    def test_the_gate_holds_no_writable_limits(self, gate):
        with pytest.raises(CompilerRefusal, match="never learned"):
            gate.feasible_set.limits.foo = 1.0  # type: ignore[attr-defined]
        with pytest.raises(CompilerRefusal, match="no learnable parameters"):
            gate.feasible_set.limits.parameters()

    def test_a_limits_file_cannot_authorise_itself(self, tmp_path):
        from scwbd.intervene.safety import DEFAULT_LIMITS_PATH

        bad = tmp_path / "bad.toml"
        bad.write_text(
            DEFAULT_LIMITS_PATH.read_text().replace(
                "human_use_authorized = false", "human_use_authorized = true"
            )
        )
        with pytest.raises(CompilerRefusal, match="never widened by an authorization"):
            SafetyLimits.load(bad)

    def test_the_record_does_not_reach_the_limits_object(self, gate, authorization):
        before = SafetyLimits.load().keys()
        gate.admit(_proposal(), _request(authorization))
        assert gate.feasible_set.limits.keys() == before


# ---------------------------------------------------------------------------
# 4. what a validated record is, and is not
# ---------------------------------------------------------------------------


def test_admission_is_not_permission_to_stimulate(gate, authorization):
    """An admitted proposal is an offline comparison, labelled. Nothing more."""
    admitted = gate.admit(_proposal(), _request(authorization))
    assert "SIMULATION ONLY" in admitted.notice.upper()
    assert not hasattr(admitted, "dose")
    assert not hasattr(admitted, "command")
    assert not any(
        token in name.lower()
        for name in dir(admitted)
        for token in ("command", "actuate", "trajectory", "execute", "deliver")
    )
