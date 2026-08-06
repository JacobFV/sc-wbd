"""Offline ranking of a preregistered set, with the ambiguity structure intact.

Defects foreclosed:

* a ranking that always produces a strict order, thereby hiding its own
  non-identifiability.  ``UnresolvedCausalAmbiguity`` is returned instead of a
  tie-break whenever two candidates are observationally indistinguishable under
  the posterior.
* a post-hoc winner passed off as preregistered.  The candidate set is hashed at
  construction and the hash travels with the ranking.
* the four validation levels being fused into one comparison score.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import (
    Defer,
    PreregisteredCandidate,
    PreregisteredPoseSet,
    Recommend,
    Refuse,
    TargetingService,
    UnresolvedCausalAmbiguity,
    compare_poses,
)
from scwbd.runtime.backends import MagnitudeThresholdResponse
from scwbd.transforms.se3 import Pose, exp_se3
from scwbd.transforms.uncertainty import PoseUncertainty

_DT = torch.float64

QUESTION = (
    "For one simulated adult phantom in one session, which of a small "
    "preregistered set of left-DLPFC coil poses is predicted to produce the "
    "largest short-horizon change in the declared network readout while "
    "remaining inside A_safe?"
)


def _candidate(base, head, twist, label, hypothesis="declared offset hypothesis"):
    xi = torch.tensor(twist, dtype=_DT)
    pose = Pose(
        base.pose.matrix @ exp_se3(xi),
        head.frame,
        "coil",
        provenance={"method": "preregistered_offset"},
    )
    return PreregisteredCandidate(
        label=label,
        hypothesis=hypothesis,
        registered_at="2026-01-01T00:00:00Z",
        request=replace(base, pose=pose, label=label),
    )


@pytest.fixture()
def base_pose(head, nominal_pose):
    return replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(0.0015, 0.015))


@pytest.fixture()
def agreeing_service():
    """Two nearly-agreeing response operators: a separable comparison exists."""
    return TargetingService(
        response_operators=(
            MagnitudeThresholdResponse(threshold_v_per_m=40.0),
            MagnitudeThresholdResponse(
                threshold_v_per_m=42.0, name="efield_magnitude_threshold_b"
            ),
        )
    )


class TestPreregistrationIsEnforced:
    def test_a_set_needs_at_least_two_candidates(self, head, base_pose):
        with pytest.raises(ValueError):
            PreregisteredPoseSet(
                study_id="s",
                question=QUESTION,
                candidates=(_candidate(base_pose, head, [0] * 6, "A"),),
            )

    def test_a_set_needs_a_written_question(self, head, base_pose):
        with pytest.raises(ValueError):
            PreregisteredPoseSet(
                study_id="s",
                question="   ",
                candidates=(
                    _candidate(base_pose, head, [0] * 6, "A"),
                    _candidate(base_pose, head, [0.01, 0, 0, 0, 0, 0], "B"),
                ),
            )

    def test_a_candidate_needs_a_hypothesis(self, head, base_pose):
        with pytest.raises(ValueError):
            _candidate(base_pose, head, [0] * 6, "A", hypothesis="  ")

    def test_adding_a_candidate_changes_the_hash(self, head, base_pose):
        a = _candidate(base_pose, head, [0] * 6, "A")
        b = _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B")
        c = _candidate(base_pose, head, [0, 0.012, 0, 0, 0, 0], "C")
        two = PreregisteredPoseSet(study_id="s", question=QUESTION, candidates=(a, b))
        three = PreregisteredPoseSet(
            study_id="s", question=QUESTION, candidates=(a, b, c)
        )
        assert two.preregistration_hash() != three.preregistration_hash()

    def test_the_ranking_carries_the_hash(self, service, head, base_pose):
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_v1",
            question=QUESTION,
            candidates=(
                _candidate(base_pose, head, [0] * 6, "A"),
                _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B"),
            ),
        )
        ranking = compare_poses(service, head, pose_set)
        assert ranking.preregistration_hash == pose_set.preregistration_hash()


class TestAmbiguityIsPreservedNotBrokenArbitrarily:
    def test_indistinguishable_candidates_are_reported_not_ordered(
        self, service, head, base_pose
    ):
        """Three poses 12 mm apart under three disagreeing response operators."""
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_ambiguous",
            question=QUESTION,
            candidates=(
                _candidate(base_pose, head, [0] * 6, "A"),
                _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B"),
                _candidate(base_pose, head, [0, 0.012, 0, 0, 0, 0], "C"),
            ),
        )
        ranking = compare_poses(service, head, pose_set)
        assert ranking.ambiguities, "the tie was broken instead of being reported"
        ambiguity = ranking.ambiguities[0]
        assert isinstance(ambiguity, UnresolvedCausalAmbiguity)
        assert set(ambiguity.labels) == {"A", "B", "C"}
        assert ambiguity.separation < ambiguity.combined_epistemic
        assert ambiguity.discriminating_measurement
        assert ambiguity.dominant_term in (
            "transform_uncertainty",
            "model_disagreement",
        )
        assert isinstance(ranking.decision, Defer)
        assert not ranking.has_strict_winner

    def test_an_ambiguity_is_falsey_so_a_careless_branch_fails_closed(
        self, service, head, base_pose
    ):
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_ambiguous",
            question=QUESTION,
            candidates=(
                _candidate(base_pose, head, [0] * 6, "A"),
                _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B"),
            ),
        )
        ranking = compare_poses(service, head, pose_set)
        if ranking.ambiguities:
            assert not ranking.ambiguities[0]

    def test_tied_candidates_share_a_group_index(self, service, head, base_pose):
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_ambiguous",
            question=QUESTION,
            candidates=(
                _candidate(base_pose, head, [0] * 6, "A"),
                _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B"),
                _candidate(base_pose, head, [0, 0.012, 0, 0, 0, 0], "C"),
            ),
        )
        ranking = compare_poses(service, head, pose_set)
        assert {c.group for c in ranking.ordered} == {0}

    def test_a_genuinely_separable_set_does_produce_a_winner(
        self, agreeing_service, head, base_pose
    ):
        tight = replace(base_pose, uncertainty=PoseUncertainty.isotropic(5e-4, 5e-3))
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_separable",
            question=QUESTION,
            candidates=(
                _candidate(tight, head, [0] * 6, "on_target"),
                _candidate(tight, head, [0.045, 0, 0, 0, 0, 0], "far_lateral"),
            ),
        )
        ranking = compare_poses(agreeing_service, head, pose_set)
        assert isinstance(ranking.decision, Recommend)
        assert ranking.decision.label == "on_target"
        assert ranking.ambiguities == ()
        assert [c.group for c in ranking.ordered] == [0, 1]


class TestUnsafeCandidatesNeverEnterTheOrdering:
    def test_a_candidate_outside_a_safe_is_refused_and_excluded(
        self, agreeing_service, head, base_pose
    ):
        direction = base_pose.pose.t - head.centre
        direction = direction / torch.linalg.norm(direction)
        far_pose = Pose.from_Rt(
            base_pose.pose.R,
            head.centre + direction * (head.scalp_radius + 0.09),
            head.frame,
            "coil",
            provenance={"method": "out_of_envelope"},
        )
        far = PreregisteredCandidate(
            label="out_of_envelope",
            hypothesis="deliberately outside A_safe",
            request=replace(base_pose, pose=far_pose, label="out_of_envelope"),
        )
        tight = replace(base_pose, uncertainty=PoseUncertainty.isotropic(5e-4, 5e-3))
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_with_unsafe",
            question=QUESTION,
            candidates=(
                _candidate(tight, head, [0] * 6, "on_target"),
                _candidate(tight, head, [0.045, 0, 0, 0, 0, 0], "far_lateral"),
                far,
            ),
        )
        ranking = compare_poses(agreeing_service, head, pose_set)
        assert [label for label, _ in ranking.refused] == ["out_of_envelope"]
        assert "out_of_envelope" not in ranking.labels
        assert all(code.code == "R11" for _, code in ranking.refused)

    def test_a_set_where_everything_is_unsafe_refuses_the_whole_study(
        self, service, head, base_pose
    ):
        direction = base_pose.pose.t - head.centre
        direction = direction / torch.linalg.norm(direction)

        def unsafe(label, extra):
            pose = Pose.from_Rt(
                base_pose.pose.R,
                head.centre + direction * (head.scalp_radius + 0.09 + extra),
                head.frame,
                "coil",
                provenance={"method": "out_of_envelope"},
            )
            return PreregisteredCandidate(
                label=label,
                hypothesis="outside A_safe",
                request=replace(base_pose, pose=pose, label=label),
            )

        pose_set = PreregisteredPoseSet(
            study_id="all_unsafe",
            question=QUESTION,
            candidates=(unsafe("a", 0.0), unsafe("b", 0.01)),
        )
        ranking = compare_poses(service, head, pose_set)
        assert isinstance(ranking.decision, Refuse)
        assert ranking.decision.code == "R11"
        assert ranking.ordered == ()


class TestTheFourQuantitiesSurviveTheRanking:
    def test_each_ranked_candidate_reports_four_separate_objects(
        self, agreeing_service, head, base_pose
    ):
        tight = replace(base_pose, uncertainty=PoseUncertainty.isotropic(5e-4, 5e-3))
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_separable",
            question=QUESTION,
            candidates=(
                _candidate(tight, head, [0] * 6, "on_target"),
                _candidate(tight, head, [0.045, 0, 0, 0, 0, 0], "far_lateral"),
            ),
        )
        ranking = compare_poses(agreeing_service, head, pose_set)
        quantities = ranking.four_quantities("on_target")
        assert set(quantities) == {
            "field_accuracy",
            "target_engagement",
            "network_effect",
            "utility",
        }
        assert len({type(v) for v in quantities.values()}) == 4
        assert quantities["utility"].estimable is False

    def test_a_ranked_candidate_has_no_scalar_value(
        self, agreeing_service, head, base_pose
    ):
        tight = replace(base_pose, uncertainty=PoseUncertainty.isotropic(5e-4, 5e-3))
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_separable",
            question=QUESTION,
            candidates=(
                _candidate(tight, head, [0] * 6, "on_target"),
                _candidate(tight, head, [0.045, 0, 0, 0, 0, 0], "far_lateral"),
            ),
        )
        ranking = compare_poses(agreeing_service, head, pose_set)
        with pytest.raises(TypeError):
            float(ranking.ordered[0])

    def test_the_ordering_basis_is_declared_and_disclaims_fusion(
        self, service, head, base_pose
    ):
        pose_set = PreregisteredPoseSet(
            study_id="dlpfc_v1",
            question=QUESTION,
            candidates=(
                _candidate(base_pose, head, [0] * 6, "A"),
                _candidate(base_pose, head, [0.012, 0, 0, 0, 0, 0], "B"),
            ),
        )
        ranking = compare_poses(service, head, pose_set)
        assert "NOT a summary" in ranking.ordering_basis
        assert ranking.limits_citations
        assert "SIMULATION ONLY" in ranking.notice
