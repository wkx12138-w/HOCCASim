"""Uniformly random controller."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import ActionOutput, Policy, register_policy

__all__ = ["RandomPolicy"]


class RandomPolicy(Policy):
    """Independent uniform sampling over the turn commands."""

    name = "random"
    uses_options = False

    def __init__(self, seed: Optional[int] = None, with_options: bool = False) -> None:
        self._rng = np.random.default_rng(seed)
        self.with_options = with_options

    def reset(self, env: Any) -> None:
        self._n_actions = int(env.uav_cfg.action_count)
        self._n_options = int(env.option_count)

    def act(self, env: Any, observation: Any) -> ActionOutput:
        n = env.n_agents
        actions = self._rng.integers(0, int(env.uav_cfg.action_count), size=n)
        options = None
        if self.with_options:
            options = self._rng.integers(0, int(env.option_count), size=n)
        return ActionOutput(actions=actions.astype(np.int64), options=options)


register_policy("random")(RandomPolicy)
