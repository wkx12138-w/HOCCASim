"""Episode recorder used by every script and figure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .env import HoCCAEnv
from .metrics import EpisodeMetrics

__all__ = ["Rollout", "rollout"]


@dataclass
class Rollout:
    """Full spatio-temporal trace of one episode."""

    positions: np.ndarray
    headings: np.ndarray
    visited: np.ndarray
    target_positions: np.ndarray
    confirmed: np.ndarray
    coverage: np.ndarray
    recall: np.ndarray
    entropy: np.ndarray
    confirmed_cells: np.ndarray
    trace: List[Any]
    metrics: EpisodeMetrics
    config: Any = None
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0))
    options: Optional[np.ndarray] = None
    annotations: List[List[Dict[str, Any]]] = field(default_factory=list)
    confirmed_sets: Optional[List[Any]] = None

    def __len__(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def steps(self) -> int:
        return int(self.coverage.shape[0] - 1)

    @property
    def n_agents(self) -> int:
        return int(self.positions.shape[1])

    def frame(self, step: int) -> Dict[str, Any]:
        step = int(np.clip(step, 0, self.steps))
        return {
            "step": step,
            "positions": self.positions[step],
            "headings": self.headings[step],
            "visited": self.visited[step],
            "targets": self.target_positions[step],
            "coverage": float(self.coverage[step]),
            "recall": float(self.recall[step]),
            "confirmed": int(self.confirmed[step]),
            "confirmed_cells": int(self.confirmed_cells[step]),
        }


def rollout(
    env: HoCCAEnv,
    policy: Any,
    seed: Optional[int] = None,
    reset_options: Optional[Dict[str, Any]] = None,
    record_frames: bool = True,
) -> Rollout:
    """Run one episode and return the recorded trace."""

    observation, _ = env.reset(seed=seed, options=reset_options)
    if hasattr(policy, "reset"):
        policy.reset(env)

    n = env.n_agents
    g = env.grid_size
    horizon = env.horizon

    positions = [np.asarray(env.positions, dtype=np.float64).copy()]
    headings = [np.asarray(env.headings, dtype=np.float64).copy()]
    options_hist = [np.asarray(env.options, dtype=np.int64).copy()]
    visited_hist = [env.visited.copy()] if record_frames else None
    targets_hist = [env.evaluator.target_positions.copy()] if record_frames else None
    coverage_hist = [env.coverage_history[-1] if env.coverage_history else 0.0]
    recall_hist = [env.recall_history[-1] if env.recall_history else 0.0]
    confirmed_hist = [env.confirmed_history[-1] if env.confirmed_history else 0]
    entropy_hist = [env.entropy_history[-1] if env.entropy_history else 0.0]
    confirmed_cells_hist = [len(env.confirmed_cells)]
    confirmed_sets = [set(env.confirmed_cells)]
    rewards: List[float] = []
    annotations: List[List[Dict[str, Any]]] = []

    terminated = truncated = False
    while not (terminated or truncated):
        output = policy.act(env, observation)
        actions = getattr(output, "actions", output)
        opts = getattr(output, "options", None)
        notes = getattr(output, "annotations", None)
        result = env.step(actions, options=opts)
        if hasattr(policy, "observe"):
            policy.observe(
                env, observation, actions, result.rewards, result.observation,
                bool(result.terminations[0] or result.truncations[0]),
            )
        observation = result.observation
        terminated = bool(result.terminations[0])
        truncated = bool(result.truncations[0])

        rewards.append(float(result.info["team_reward"]))
        annotations.append(notes or [{} for _ in range(n)])
        positions.append(np.asarray(env.positions, dtype=np.float64).copy())
        headings.append(np.asarray(env.headings, dtype=np.float64).copy())
        options_hist.append(np.asarray(env.options, dtype=np.int64).copy())
        if record_frames:
            visited_hist.append(env.visited.copy())
            targets_hist.append(env.evaluator.target_positions.copy())
        coverage_hist.append(env.coverage_history[-1])
        recall_hist.append(env.recall_history[-1])
        confirmed_hist.append(env.confirmed_history[-1])
        entropy_hist.append(env.entropy_history[-1])
        confirmed_cells_hist.append(len(env.confirmed_cells))
        confirmed_sets.append(set(env.confirmed_cells))
        if len(rewards) >= horizon + 1:
            break

    return Rollout(
        positions=np.stack(positions, axis=0),
        headings=np.stack(headings, axis=0),
        visited=(
            np.stack(visited_hist, axis=0) if record_frames
            else np.zeros((0, g, g), dtype=bool)
        ),
        target_positions=(
            np.stack(targets_hist, axis=0) if record_frames
            else np.zeros((0, 0, 2), dtype=np.int64)
        ),
        confirmed=np.asarray(confirmed_hist, dtype=np.int64),
        coverage=np.asarray(coverage_hist, dtype=np.float64),
        recall=np.asarray(recall_hist, dtype=np.float64),
        entropy=np.asarray(entropy_hist, dtype=np.float64),
        confirmed_cells=np.asarray(confirmed_cells_hist, dtype=np.int64),
        trace=list(env.trace),
        metrics=env.episode_metrics(),
        config=env.config,
        rewards=np.asarray(rewards, dtype=np.float64),
        options=np.stack(options_hist, axis=0),
        annotations=annotations,
        confirmed_sets=confirmed_sets,
    )
