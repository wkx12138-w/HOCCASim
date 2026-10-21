"""HOCCA reference simulator for communication-constrained multi-UAV search."""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "HOCCA authors"

from .config import (  # noqa: F401
    BeliefConfig,
    CommunicationConfig,
    EpisodeConfig,
    GridConfig,
    ObservationConfig,
    REFERENCE_CORRIDORS,
    REFERENCE_MOVING_TARGETS,
    REFERENCE_STATIC_TARGETS,
    REFERENCE_UAV_POSES,
    RewardConfig,
    SensingConfig,
    SimConfig,
    TargetConfig,
    UAVConfig,
)
from .env import HoCCAEnv, Observation, StepResult, make_env, register_with_gymnasium  # noqa: F401
from .metrics import EpisodeMetrics, aggregate_metrics  # noqa: F401
from .policies import (  # noqa: F401
    Policy,
    PolicyRegistry,
    RandomPolicy,
    RuleBasedPolicy,
    available_policies,
    get_policy,
    register_policy,
)

__all__ = [
    "__version__", "SimConfig", "GridConfig", "SensingConfig", "UAVConfig",
    "TargetConfig", "CommunicationConfig", "BeliefConfig", "RewardConfig",
    "EpisodeConfig", "ObservationConfig", "HoCCAEnv", "Observation",
    "StepResult", "make_env", "register_with_gymnasium", "EpisodeMetrics",
    "aggregate_metrics", "Policy", "PolicyRegistry", "RandomPolicy",
    "RuleBasedPolicy", "available_policies", "get_policy", "register_policy",
    "REFERENCE_STATIC_TARGETS", "REFERENCE_MOVING_TARGETS",
    "REFERENCE_UAV_POSES", "REFERENCE_CORRIDORS",
]
