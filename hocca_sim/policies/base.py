"""Policy protocol and registry."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "ActionOutput", "Policy", "RLPolicy", "PolicyRegistry", "register_policy",
    "get_policy", "available_policies",
]


@dataclass
class ActionOutput:
    """Per-agent primitive actions and optional option indices."""

    actions: np.ndarray
    options: Optional[np.ndarray] = None
    annotations: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.int64).reshape(-1)
        if self.options is not None:
            self.options = np.asarray(self.options, dtype=np.int64).reshape(-1)


class Policy(abc.ABC):
    """Base class for every controller."""

    name: str = "policy"
    uses_options: bool = False

    def reset(self, env: Any) -> None:
        """Called once per episode."""

    def observe(
        self,
        env: Any,
        observation: Any,
        actions: Any,
        rewards: Any,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Optional transition callback for on-policy learners."""

    @abc.abstractmethod
    def act(self, env: Any, observation: Any) -> ActionOutput:
        """Return the actions (and options) for the current step."""

    def act_array(self, env: Any, observation: Any) -> np.ndarray:
        return self.act(env, observation).actions


class RLPolicy(Policy):
    """Adapter for an external learning agent exposing ``act(obs) -> int``."""

    name = "rl"

    def __init__(
        self,
        agent: Any = None,
        act_fn: Optional[Callable[[np.ndarray], int]] = None,
        obs_transform: Optional[Callable[[Any, int], np.ndarray]] = None,
        use_state_for_actions: bool = False,
        action_mask_fn: Optional[Callable[[Any, int], np.ndarray]] = None,
    ) -> None:
        if agent is None and act_fn is None:
            raise ValueError("provide either `agent` or `act_fn`")
        self.agent = agent
        self.act_fn = act_fn
        self.obs_transform = obs_transform or (lambda obs, u: obs["agents"][u])
        self.use_state_for_actions = use_state_for_actions
        self.action_mask_fn = action_mask_fn

    def reset(self, env: Any) -> None:
        if self.agent is not None and hasattr(self.agent, "reset"):
            self.agent.reset()

    def _single(self, observation: Any, u: int) -> int:
        features = self.obs_transform(observation, u)
        if self.act_fn is not None:
            return int(self.act_fn(features))
        mask = None
        if self.action_mask_fn is not None:
            mask = self.action_mask_fn(observation, u)
        try:
            return int(self.agent.act(features, mask=mask))
        except TypeError:
            return int(self.agent.act(features))

    def act(self, env: Any, observation: Any) -> ActionOutput:
        actions = np.asarray(
            [self._single(observation, u) for u in range(env.n_agents)], dtype=np.int64
        )
        return ActionOutput(actions=actions)


class PolicyRegistry:
    """Name to factory mapping."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Policy]] = {}

    def register(
        self, name: str
    ) -> Callable[[Callable[..., Policy]], Callable[..., Policy]]:
        def decorator(cls: Callable[..., Policy]) -> Callable[..., Policy]:
            self._factories[name] = cls
            return cls

        return decorator

    def add(self, name: str, factory: Callable[..., Policy]) -> None:
        self._factories[name] = factory

    def create(self, name: str, *args: Any, **kwargs: Any) -> Policy:
        if name not in self._factories:
            raise KeyError(
                "unknown policy %r; available: %s"
                % (name, ", ".join(sorted(self._factories)))
            )
        return self._factories[name](*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


REGISTRY = PolicyRegistry()


def register_policy(
    name: str,
) -> Callable[[Callable[..., Policy]], Callable[..., Policy]]:
    return REGISTRY.register(name)


def get_policy(name: str, *args: Any, **kwargs: Any) -> Policy:
    return REGISTRY.create(name, *args, **kwargs)


def available_policies() -> List[str]:
    return REGISTRY.names()
