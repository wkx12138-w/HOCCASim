"""Adapters for Gymnasium, PettingZoo and single-agent usage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .env import HoCCAEnv
from .spaces import Box, Discrete, GYMNASIUM_AVAILABLE, spaces_module

if GYMNASIUM_AVAILABLE:
    import gymnasium as _gym

    _WrapperBase = _gym.Wrapper
else:
    _WrapperBase = object

__all__ = [
    "GymnasiumMultiAgentWrapper", "PettingZooParallelWrapper",
    "SingleAgentVectorWrapper", "EpisodeStatsWrapper",
]


class GymnasiumMultiAgentWrapper(_WrapperBase):
    """Parallel multi-agent view with string agent identifiers."""

    def __init__(self, env: HoCCAEnv, agent_prefix: str = "agent_") -> None:
        if GYMNASIUM_AVAILABLE:
            super().__init__(env)
        self._env = env
        self.agent_prefix = agent_prefix
        self.agents = [f"{agent_prefix}{u}" for u in range(env.n_agents)]
        self.possible_agents = list(self.agents)
        self.num_agents = env.n_agents
        self.action_space = Discrete(env.uav_cfg.action_count)
        self.observation_space = env.single_observation_space
        self.share_observation_space = env.state_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._pack(obs), info

    def step(self, actions: Any):
        action_vec = np.asarray([actions[name] for name in self.agents], dtype=np.int64)
        result = self._env.step(action_vec)
        rewards = {n: float(result.rewards[u]) for u, n in enumerate(self.agents)}
        terms = {n: bool(result.terminations[u]) for u, n in enumerate(self.agents)}
        truncs = {n: bool(result.truncations[u]) for u, n in enumerate(self.agents)}
        return self._pack(result.observation), rewards, terms, truncs, result.info

    def _pack(self, obs) -> Dict[str, Any]:
        packed: Dict[str, Any] = {
            name: obs["agents"][u] for u, name in enumerate(self.agents)
        }
        packed["state"] = obs["state"]
        return packed

    def render(self, *args: Any, **kwargs: Any):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()


class PettingZooParallelWrapper:
    """Minimal PettingZoo parallel-API adapter."""

    metadata = {"render_modes": ["rgb_array"], "name": "hocca_parallel"}

    def __init__(self, env: HoCCAEnv) -> None:
        self.env = env
        self.agents: List[str] = [f"uav_{u}" for u in range(env.n_agents)]
        self.possible_agents = list(self.agents)
        self.action_spaces = {n: Discrete(env.uav_cfg.action_count) for n in self.agents}
        self.observation_spaces = {n: env.single_observation_space for n in self.agents}
        self._last: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.agents = list(self.possible_agents)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last = {
            name: dict(observation=obs["agents"][u], reward=0.0, termination=False,
                       truncation=False, info=info)
            for u, name in enumerate(self.agents)
        }
        return self.observe(), {name: dict(info) for name in self.agents}

    def observe(self) -> Dict[str, np.ndarray]:
        return {name: self._last[name]["observation"] for name in self.agents}

    def step(self, actions: Dict[str, int]):
        action_vec = np.asarray([actions[n] for n in self.agents], dtype=np.int64)
        result = self.env.step(action_vec)
        observations = {n: result.observation["agents"][u] for u, n in enumerate(self.agents)}
        rewards = {n: float(result.rewards[u]) for u, n in enumerate(self.agents)}
        terminations = {n: bool(result.terminations[u]) for u, n in enumerate(self.agents)}
        truncations = {n: bool(result.truncations[u]) for u, n in enumerate(self.agents)}
        infos = {n: result.info for n in self.agents}
        self._last = {
            name: dict(observation=observations[name], reward=rewards[name],
                       termination=terminations[name], truncation=truncations[name],
                       info=infos[name])
            for name in self.agents
        }
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def agent_selection(self) -> str:
        return self.agents[0] if self.agents else ""

    def close(self) -> None:
        self.env.close()


class SingleAgentVectorWrapper:
    """Joint-action view: one policy drives the whole fleet."""

    def __init__(self, env: HoCCAEnv) -> None:
        self.env = env
        self.n_agents = env.n_agents
        self.n_actions = int(env.uav_cfg.action_count)
        self.observation_dim = int(env.obs_dim * self.n_agents + env.state_dim)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        try:
            from gymnasium import spaces as gym_spaces

            self.action_space = gym_spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        except Exception:
            self.action_space = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action: Any):
        result = self.env.step(np.asarray(action, dtype=np.int64).reshape(-1))
        return (
            self._flatten(result.observation),
            float(np.mean(result.rewards)),
            bool(result.terminations[0]),
            bool(result.truncations[0]),
            result.info,
        )

    def _flatten(self, obs) -> np.ndarray:
        return np.concatenate([
            obs["agents"].reshape(-1), np.asarray(obs["state"]).reshape(-1)
        ]).astype(np.float32)

    def close(self) -> None:
        self.env.close()


class EpisodeStatsWrapper:
    """Accumulate per-episode return and metrics."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_metrics: List[Dict[str, float]] = []
        self._return = 0.0
        self._length = 0

    def reset(self, *args: Any, **kwargs: Any):
        self._return = 0.0
        self._length = 0
        return self.env.reset(*args, **kwargs)

    def step(self, actions: Any):
        result = self.env.step(actions)
        self._return += float(result.info.get("team_reward", 0.0))
        self._length += 1
        if bool(result.terminations[0]) or bool(result.truncations[0]):
            self.episode_returns.append(self._return)
            self.episode_lengths.append(self._length)
            if hasattr(self.env, "episode_metrics"):
                self.episode_metrics.append(self.env.episode_metrics().as_row())
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)
