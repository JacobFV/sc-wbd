"""Interchangeable dynamical backends.

Every backend implements the same interface (:class:`scwbd.dynamics.base.DynamicsBackend`)
so that model comparison over backends is a first-class output rather than a
modelling assumption (ARCHITECTURE.md §5).  ``LearnedNeuralOperator`` is the
equal-capacity control for every mechanistic claim made by the others.
"""

from ..base import BackendInfo, DynamicsBackend, get_backend, list_backends, register_backend
from .jansen_rit import JansenRit
from .linear_gaussian import LinearGaussian, ou_propagate_moments, ou_stationary_covariance
from .neural_operator import LearnedNeuralOperator, assert_equal_capacity, match_capacity
from .oscillator import Kuramoto, StuartLandau, kuramoto_order_parameter, metastability
from .wilson_cowan import WilsonCowan
from .wong_wang import FICResult, ReducedWongWang, ReducedWongWangSingle, tune_fic

__all__ = [
    "BackendInfo",
    "DynamicsBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    "WilsonCowan",
    "JansenRit",
    "ReducedWongWang",
    "ReducedWongWangSingle",
    "tune_fic",
    "FICResult",
    "StuartLandau",
    "Kuramoto",
    "kuramoto_order_parameter",
    "metastability",
    "LinearGaussian",
    "ou_stationary_covariance",
    "ou_propagate_moments",
    "LearnedNeuralOperator",
    "match_capacity",
    "assert_equal_capacity",
]
