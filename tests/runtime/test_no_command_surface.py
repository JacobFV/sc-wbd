"""``scwbd.runtime`` must expose no actuation or command entry point, ever.

Defect foreclosed: ``ARCHITECTURE.md`` Sec. 6 -- *"SC-WBD supplies no joint
commands, no trajectories, no actuation, and no stimulation authority, and must
not create a path to any of them."*  The consumer repository's three standing
invariants (``sim2real_ready=false``, ``promotion_eligible=false``,
``robot_command_authority=false``) are preserved **by construction**, and the
construction is what this module checks.

These tests are reflective on purpose.  A future contributor who adds
``TargetingService.send_command`` or imports a transport does not have to
remember this rule; the suite fails.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import scwbd.runtime as runtime
from scwbd.runtime.admission import CONSUMER_STANDING_INVARIANTS

#: Verb stems that would indicate an actuation or command seam.
FORBIDDEN_SYMBOL_TOKENS = (
    "command",
    "actuat",
    "servo",
    "torque",
    "joint_target",
    "trajector",
    "waypoint",
    "publish",
    "execute",
    "move_to",
    "goto",
    "trigger_pulse",
    "fire_pulse",
    "energis",
    "energiz",
    "stimulate",
    "dose_protocol",
)

#: Modules that would drag a device, a bus, or a robot stack into the process.
FORBIDDEN_IMPORT_PREFIXES = (
    "rclpy",
    "ros",
    "ros2",
    "libfranka",
    "franka",
    "panda",
    "curobo",
    "isaacsim",
    "isaaclab",
    "omni",
    "xarm",
    "pyserial",
    "serial",
    "socket",
    "http",
    "urllib",
    "requests",
    "grpc",
    "zmq",
    "paho",
    "can",
    "usb",
    "robotic_tms",
)

#: Names that are legitimate here even though they contain a forbidden token.
ALLOWED_EXCEPTIONS = {
    # nothing so far; keep this empty and make it hard to add to
}

#: Attribute names that *name* a command authority in order to assert it is
#: absent.  ``robot_command_authority`` is a field of
#: :class:`~scwbd.runtime.admission.ConsumerInvariants` whose only permitted
#: value is ``False`` -- constructing it any other way raises.  Naming a
#: capability in order to refuse it is the opposite of exposing it, and the
#: test below proves the refusal rather than trusting this list: an entry here
#: is only honoured if the name is a standing invariant pinned to ``False``.
NEGATED_INVARIANT_ATTRS = frozenset(CONSUMER_STANDING_INVARIANTS)


def _runtime_modules() -> list[str]:
    root = Path(runtime.__file__).parent
    names = [f"scwbd.runtime.{m.name}" for m in pkgutil.iter_modules([str(root)])]
    return ["scwbd.runtime", *sorted(names)]


class TestThePublicSurfaceHasNoCommandEntryPoint:
    @pytest.mark.parametrize("module_name", _runtime_modules())
    def test_no_public_symbol_names_an_actuation(self, module_name):
        module = importlib.import_module(module_name)
        offenders = []
        for name, obj in vars(module).items():
            if name.startswith("_") or name in ALLOWED_EXCEPTIONS:
                continue
            lowered = name.lower()
            if any(token in lowered for token in FORBIDDEN_SYMBOL_TOKENS):
                offenders.append(f"{module_name}.{name}")
            if inspect.isclass(obj) and obj.__module__.startswith("scwbd.runtime"):
                for attr in vars(obj):
                    if attr.startswith("_"):
                        continue
                    if attr in NEGATED_INVARIANT_ATTRS:
                        # Named in order to be refused; proven below.
                        continue
                    if any(token in attr.lower() for token in FORBIDDEN_SYMBOL_TOKENS):
                        offenders.append(f"{module_name}.{name}.{attr}")
        assert not offenders, f"actuation-shaped symbols found: {offenders}"

    @pytest.mark.parametrize("invariant", sorted(CONSUMER_STANDING_INVARIANTS))
    def test_each_excused_invariant_name_is_actually_pinned_false(self, invariant):
        """The exemption above is only sound if the flag cannot be set true.

        Without this, ``NEGATED_INVARIANT_ATTRS`` would be a way to smuggle a
        real command-authority flag past the surface test by naming it after an
        invariant.  So the exemption is not taken on trust: every excused name
        must refuse construction in the widened state.
        """
        from scwbd.runtime import ConsumerInvariants, ConsumerInvariantViolation

        assert CONSUMER_STANDING_INVARIANTS[invariant] is False
        assert getattr(ConsumerInvariants(), invariant) is False
        with pytest.raises(ConsumerInvariantViolation) as exc:
            ConsumerInvariants(**{invariant: True})
        assert invariant in str(exc.value)

    def test_the_targeting_service_surface_is_exactly_reads_and_refusals(self):
        from scwbd.runtime import TargetingService

        public = {n for n in vars(TargetingService) if not n.startswith("_")}
        assert public == {
            "load",
            "evaluate_pose",
            "with_config",
            "read",
        }

    def test_the_served_model_surface_is_exactly_load_handshake_and_evaluate(self):
        from scwbd.runtime import ServedModel

        methods = {n for n in vars(ServedModel) if not n.startswith("_")}
        # ``port_contract`` is a read: it returns the model's own declaration of
        # what it exports, so a consumer can read state by name instead of by
        # index.  ``admission`` is the export-gate verdict this service was
        # loaded under, carried so a consumer can record it.  Neither reaches
        # anything downstream of the registered external scalp target.
        assert methods == {
            "load",
            "handshake",
            "describe",
            "warm_up",
            "evaluate_batch",
            "port_contract",
            # a dataclass field with a default, hence also a class attribute
            "admission",
        }
        assert set(ServedModel.__dataclass_fields__) == {
            "targeting",
            "provenance",
            "checkpoint",
            "admission",
        }

    def test_no_constructor_accepts_a_command_transport(self):
        from scwbd.runtime import BrainRuntime, ServedModel, TargetingService

        for cls in (TargetingService, ServedModel, BrainRuntime):
            names = set(inspect.signature(cls.__init__).parameters)
            assert not any(
                token in n.lower()
                for n in names
                for token in ("command", "transport", "actuator", "device_handle", "socket")
            ), f"{cls.__name__} takes a command-shaped parameter: {names}"

    def test_evaluate_pose_returns_predictions_and_a_decision_only(
        self, service, head, nominal_pose
    ):
        from scwbd.runtime import PoseEvaluation

        evaluation = service.evaluate_pose(head, nominal_pose)
        assert isinstance(evaluation, PoseEvaluation)
        fields = {f for f in evaluation.__dataclass_fields__}
        assert not any(
            token in f.lower() for f in fields for token in FORBIDDEN_SYMBOL_TOKENS
        )


class TestNoImportPathReachesADeviceOrARobot:
    @pytest.mark.parametrize("module_name", _runtime_modules())
    def test_the_source_imports_nothing_that_can_command_hardware(self, module_name):
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.append(node.module)
        offenders = [
            name
            for name in imported
            if any(
                name == p or name.startswith(p + ".")
                for p in FORBIDDEN_IMPORT_PREFIXES
            )
        ]
        assert not offenders, f"{module_name} imports {offenders}"

    def test_the_loaded_process_has_not_pulled_in_a_robot_stack(self):
        import sys

        loaded = {
            m
            for m in sys.modules
            for p in ("rclpy", "libfranka", "curobo", "isaacsim", "isaaclab", "robotic_tms")
            if m == p or m.startswith(p + ".")
        }
        assert not loaded, f"importing scwbd.runtime pulled in {sorted(loaded)}"


class TestTheSimulationOnlyNoticeTravels:
    def test_every_returned_object_carries_the_notice(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(head, nominal_pose)
        for obj in (
            evaluation,
            evaluation.efield,
            evaluation.field_accuracy,
            evaluation.target_engagement,
            evaluation.network_response,
            evaluation.utility,
            evaluation.provenance,
            evaluation.decision,
        ):
            notice = getattr(obj, "notice", "")
            assert "SIMULATION ONLY" in notice

    def test_the_notice_claims_only_what_the_code_enforces(self):
        """The notice must describe *this software*, never anyone's paperwork.

        It once asserted "no consent", "no participants", "no device" -- claims
        about the world that nothing here checked. A test that pins a false
        claim in place is worse than no test, so it is inverted: those phrases
        are banned, and the enforced properties are asserted instead.
        """
        from scwbd.runtime import SIMULATION_ONLY_NOTICE

        for banned in ("no consent", "no participants", "no ethics approval"):
            assert banned not in SIMULATION_ONLY_NOTICE, (
                f"the notice asserts {banned!r}, which is a claim about the "
                "world that no code in this repository checks"
            )
        # What it must say instead, each clause enforced somewhere:
        assert "not a device driver" in SIMULATION_ONLY_NOTICE
        assert "not a dosing" in SIMULATION_ONLY_NOTICE
        assert "no device command" in SIMULATION_ONLY_NOTICE

    def test_no_evaluation_exposes_a_command_or_a_dose(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(head, nominal_pose)
        for banned in ("command", "dose_mt", "amplitude", "trigger", "fire"):
            assert not hasattr(evaluation, banned)
