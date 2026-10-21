#!/usr/bin/env python
"""Plug-in skeleton for an external MARL algorithm."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import make_env
from hocca_sim.policies import ActionOutput, Policy


class PlaceholderLearner:
    """Stands in for MAPPO / QMIX / MADDPG / HOCCA; only the interface matters."""

    def __init__(self, obs_dim: int, n_actions: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(obs_dim, n_actions))
        self.n_actions = n_actions
        self.trajectory: List[Dict[str, np.ndarray]] = []

    def act(self, observation: np.ndarray, mask=None, epsilon: float = 0.1) -> int:
        logits = observation.astype(np.float64) @ self.weights
        if mask is not None:
            logits = np.where(mask, logits, -np.inf)
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.random.choice(self.n_actions, p=probs))

    def observe(self, agent_obs, state, actions, rewards, next_agent_obs,
                next_state, done) -> None:
        self.trajectory.append({
            "obs": agent_obs, "state": state, "actions": actions,
            "rewards": rewards, "next_obs": next_agent_obs,
            "next_state": next_state, "done": np.array([float(done)]),
        })

    def update(self, gamma: float = 0.99) -> Dict[str, float]:
        if not self.trajectory:
            return {"loss": 0.0, "transitions": 0.0}
        returns = np.zeros(len(self.trajectory))
        running = 0.0
        for index in range(len(self.trajectory) - 1, -1, -1):
            transition = self.trajectory[index]
            running = (float(transition["rewards"].mean()) + gamma * running
                       * (1.0 - float(transition["done"][0])))
            returns[index] = running
        loss = float(np.var(returns))
        self.trajectory.clear()
        return {"loss": loss, "returns_mean": float(returns.mean())}


class LearnerPolicy(Policy):
    """Decentralised execution: one shared learner, one observation per UAV."""

    name = "learner"
    uses_options = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.learner = None
        self.options = None

    def reset(self, env) -> None:
        if self.learner is None:
            self.learner = PlaceholderLearner(env.obs_dim, env.uav_cfg.action_count,
                                              self.seed)
        self.options = np.zeros(env.n_agents, dtype=np.int64)

    def _select_options(self, env, observation) -> np.ndarray:
        expired = env.option_expired_mask()
        if not expired.any():
            return self.options
        rng = np.random.default_rng(self.seed + env.step_index)
        self.options = np.where(expired, rng.integers(0, env.option_count),
                                self.options)
        return self.options

    def act(self, env, observation) -> ActionOutput:
        options = self._select_options(env, observation)
        actions = np.array([self.learner.act(observation["agents"][u])
                            for u in range(env.n_agents)], dtype=np.int64)
        return ActionOutput(actions=actions, options=options)

    def observe(self, env, observation, actions, rewards, next_observation, done) -> None:
        self.learner.observe(observation["agents"], observation["state"],
                             np.asarray(actions), np.asarray(rewards),
                             next_observation["agents"], next_observation["state"],
                             done)


def train(episodes: int = 5, seed: int = 0) -> None:
    env = make_env(seed=seed)
    policy = LearnerPolicy(seed)
    history: List[Tuple[int, float, float, int]] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        policy.reset(env)
        done = False
        episode_return = 0.0
        while not done:
            output = policy.act(env, observation)
            result = env.step(output.actions, options=output.options)
            policy.observe(env, observation, output.actions, result.rewards,
                           result.observation,
                           bool(result.terminations[0] or result.truncations[0]))
            episode_return += float(result.info["team_reward"])
            observation = result.observation
            done = bool(result.terminations[0] or result.truncations[0])
        stats = policy.learner.update(env.reward_cfg.gamma)
        metrics = env.episode_metrics()
        history.append((episode, episode_return, metrics.coverage, metrics.confirmed))
        print("episode %3d  return %8.1f  coverage %.3f  confirmed %2d/%d  loss %8.3f"
              % (episode, episode_return, metrics.coverage, metrics.confirmed,
                 metrics.target_count, stats.get("loss", 0.0)))
    env.close()


if __name__ == "__main__":
    train()
