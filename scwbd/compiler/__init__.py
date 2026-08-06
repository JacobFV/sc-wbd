"""``scwbd.compiler`` - schema -> CompiledModel.

``compile(schema, *, claim)`` emits the memory map, block-sparse masks, typed
dispatch, multirate plan, gradient permissions, frame/clock indices and
uncertainty ledger described in ARCHITECTURE.md sec. 2 - or it refuses, with
the exact remedy from Table tab:compiler-refusals.
"""

from __future__ import annotations

from .adjacency import EVIDENCE_CLASSES, Adjacency, build_adjacency
from .checks import (
    check_r01,
    check_r02,
    check_r03,
    check_r04,
    check_r05,
    check_r06,
    check_r07,
    check_r08,
    check_r09,
    check_r10,
    check_r11,
)
from .compile import compile, first_refusal, run_checks
from .dispatch import Dispatch, OperatorDescriptor, build_dispatch, resolve_operator_clock
from .graphs import (
    CompiledClockGraph,
    CompiledFrameGraph,
    build_clock_graph,
    build_frame_graph,
)
from .layout import LayoutEntry, StateLayout, build_state_layout
from .ledger import CompiledLedger, build_ledger
from .masks import GradientMask, GradientMasks, build_gradient_masks, parameter_groups_of
from .model import COMPILER_VERSION, CompiledModel, Provenance
from .schedule import (
    NS,
    FieldPolicy,
    InterpolationContract,
    MultirateSchedule,
    ScheduleEvent,
    build_schedule,
)

__all__ = [
    "compile",
    "run_checks",
    "first_refusal",
    "CompiledModel",
    "Provenance",
    "COMPILER_VERSION",
    "StateLayout",
    "LayoutEntry",
    "build_state_layout",
    "Adjacency",
    "build_adjacency",
    "EVIDENCE_CLASSES",
    "Dispatch",
    "OperatorDescriptor",
    "build_dispatch",
    "resolve_operator_clock",
    "MultirateSchedule",
    "FieldPolicy",
    "InterpolationContract",
    "ScheduleEvent",
    "build_schedule",
    "NS",
    "GradientMasks",
    "GradientMask",
    "build_gradient_masks",
    "parameter_groups_of",
    "CompiledFrameGraph",
    "CompiledClockGraph",
    "build_frame_graph",
    "build_clock_graph",
    "CompiledLedger",
    "build_ledger",
    "check_r01",
    "check_r02",
    "check_r03",
    "check_r04",
    "check_r05",
    "check_r06",
    "check_r07",
    "check_r08",
    "check_r09",
    "check_r10",
    "check_r11",
]
