"""Task reward, potential shaping and differential information credit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .beliefs import binary_entropy, entropy_map, sigmoid
from .config import RewardConfig

__all__ = [
    "RewardBreakdown", "compute_task_reward", "potential", "information_credits",
    "distribute_credit", "summarise_reward_components",
]


@dataclass
class RewardBreakdown:
    """Per-step decomposition of the task reward."""

    confirmed: float = 0.0
    coverage: float = 0.0
    revisit: float = 0.0
    communication: float = 0.0
    energy: float = 0.0
    balance: float = 0.0
    boundary: float = 0.0
    collision: float = 0.0
    shaping: float = 0.0

    @property
    def task(self) -> float:
        return (self.confirmed + self.coverage + self.revisit + self.communication
                + self.energy + self.balance + self.boundary + self.collision)

    @property
    def team(self) -> float:
        return self.task + self.shaping

    def as_dict(self) -> Dict[str, float]:
        return {
            "confirmed": self.confirmed, "coverage": self.coverage,
            "revisit": self.revisit, "communication": self.communication,
            "energy": self.energy, "balance": self.balance,
            "boundary": self.boundary, "collision": self.collision,
            "shaping": self.shaping, "task": self.task, "team": self.team,
        }

    def __iadd__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        for name in ("confirmed", "coverage", "revisit", "communication", "energy",
                     "balance", "boundary", "collision", "shaping"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self


def potential(belief_probabilities: np.ndarray, lambda_H: float) -> float:
    """``Phi = -lambda_H * sum_c h(b_c)``."""

    return float(-lambda_H * np.sum(binary_entropy(belief_probabilities)))


def _balance_penalty(per_agent_covered: np.ndarray) -> float:
    if len(per_agent_covered) < 2:
        return 0.0
    mean = float(np.mean(per_agent_covered))
    if mean <= 1e-9:
        return 0.0
    return float(np.var(per_agent_covered)) / (mean * mean)


def compute_task_reward(
    cfg: RewardConfig,
    *,
    new_confirmations: int,
    new_cells: int,
    repeats: int,
    looks: int,
    items_attempted: int,
    item_budget_total: int,
    energy_spent_fraction: float,
    per_agent_new_cells: np.ndarray,
    boundary_blocks: int,
    collisions: int,
    agent_count: int,
) -> RewardBreakdown:
    out = RewardBreakdown()
    out.confirmed = cfg.lambda_D * float(new_confirmations)
    out.coverage = cfg.coverage_coef * float(new_cells)

    if cfg.revisit_normalisation == "count":
        out.revisit = -cfg.lambda_R * float(repeats)
    else:
        fraction = 0.0 if looks <= 0 else float(repeats) / float(looks)
        out.revisit = -cfg.lambda_R * fraction

    if cfg.communication_normalisation == "count":
        out.communication = -cfg.lambda_C * float(items_attempted)
    else:
        out.communication = -cfg.lambda_C * float(items_attempted) / float(
            max(item_budget_total, 1)
        )

    out.energy = -cfg.lambda_E * float(energy_spent_fraction)

    if cfg.spatial_balance_mode != "none":
        out.balance = -cfg.lambda_F * _balance_penalty(
            np.asarray(per_agent_new_cells, dtype=np.float64)
        )

    if boundary_blocks > 0 and cfg.boundary_penalty:
        out.boundary = -cfg.boundary_penalty * float(boundary_blocks) / max(
            agent_count, 1
        )
    if collisions > 0 and cfg.collision_penalty:
        out.collision = -cfg.collision_penalty * float(collisions)
    return out


def information_credits(
    pre_log_odds: np.ndarray,
    evidence: Sequence[tuple],
    order: Sequence[int],
    clip: float = 1e-9,
) -> np.ndarray:
    """Per-agent entropy reduction under one assimilation order."""

    work = np.array(pre_log_odds, dtype=np.float64, copy=True)
    credits = np.zeros(len(order), dtype=np.float64)
    before = entropy_map(sigmoid(work))
    for agent in order:
        cells, deltas = evidence[agent]
        if cells is None or len(cells) == 0:
            continue
        cells = np.asarray(cells, dtype=np.int64)
        deltas = np.asarray(deltas, dtype=np.float64)
        flat = cells[:, 1] * work.shape[0] + cells[:, 0]
        work += np.bincount(flat, weights=deltas, minlength=work.size).reshape(
            work.shape
        )
        after = entropy_map(sigmoid(work))
        credits[agent] = before - after
        before = after
    return credits


def distribute_credit(
    team_reward: float, credits: np.ndarray, lambda_A: float
) -> np.ndarray:
    n = len(credits)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    share = float(team_reward) / n
    if lambda_A == 0.0 or float(np.sum(np.abs(credits))) == 0.0:
        return np.full(n, share, dtype=np.float64)
    return share + float(lambda_A) * (credits - credits.mean())


def summarise_reward_components(
    breakdowns: Sequence[RewardBreakdown],
) -> Dict[str, float]:
    keys = ("confirmed", "coverage", "revisit", "communication", "energy",
            "balance", "boundary", "collision", "shaping", "task", "team")
    totals = {k: 0.0 for k in keys}
    for b in breakdowns:
        data = b.as_dict()
        for k in keys:
            totals[k] += data[k]
    return totals
