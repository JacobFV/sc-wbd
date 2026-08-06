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
                    if any(token in attr.lower() for token in FORBIDDEN_SYMBOL_TOKENS):
                        offenders.append(f"{module_name}.{name}.{attr}")
        assert not offenders, f"actuation-shaped symbols found: {offenders}"

    def test_the_targeting_service_surface_is_exactly_reads_and_refusals(self):
        from scwbd.runtime import TargetingService

        public = {n for n in vars(TargetingService) if not n.startswith("_")}
        assert public == {"load", "evaluate_pose", "with_config", "read"}

    def test_the_served_model_surface_is_exactly_load_handshake_and_evaluate(self):
        from scwbd.runtime import ServedModel

        methods = {n for n in vars(ServedModel) if not n.startswith("_")}
        assert methods == {"load", "handshake", "describe", "warm_up", "evaluate_batch"}
        assert set(ServedModel.__dataclass_fields__) == {
            "targeting",
            "provenance",
            "checkpoint",
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

    def test_the_notice_names_the_missing_approvals(self):
        from scwbd.runtime import SIMULATION_ONLY_NOTICE

        for phrase in ("no consent", "no participants", "no device"):
            assert phrase in SIMULATION_ONLY_NOTICE

    def test_nothing_can_be_flagged_as_authorized_for_human_use(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(head, nominal_pose)
        assert evaluation.provenance.human_use_authorized is False
        assert evaluation.provenance.prospective_human is False
        assert evaluation.ledger.validity_domain["human_use_authorized"] is False
