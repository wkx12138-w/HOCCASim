#!/usr/bin/env python
"""Write your own controller in thirty lines.

    python examples/custom_policy.py
    python scripts/run_episode.py --policy wallfollower
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import make_env
from hocca_sim.policies import ActionOutput, Policy, get_policy, register_policy
from hocca_sim.rollout import rollout


@register_policy("wallfollower")
class WallFollowerPolicy(Policy):
    """Fly straight until close to the border, then turn toward the centre."""

    name = "wallfollower"
    uses_options = False

    def __init__(self, margin: float = 4.0) -> None:
        self.margin = margin

    def _turn_choice(self, env, u: int, heading: float) -> int:
        centre = np.array([env.grid_size / 2.0, env.grid_size / 2.0])
        inward = centre - np.asarray(env.positions[u], dtype=float)
        desired = np.arctan2(inward[1], inward[0])
        turns = np.deg2rad(np.asarray(env.uav_cfg.turn_actions_deg, dtype=float))
        residuals = [abs(np.angle(np.exp(1j * (desired - (heading + t))))) for t in turns]
        return int(np.argmin(residuals))

    def act(self, env, observation) -> ActionOutput:
        actions = np.zeros(env.n_agents, dtype=np.int64)
        straight = int(np.argmin(np.abs(env.uav_cfg.turn_actions_deg)))
        limit = self.margin
        for u in range(env.n_agents):
            x, y = env.positions[u]
            near_edge = (x < limit or x > env.grid_size - 1 - limit
                         or y < limit or y > env.grid_size - 1 - limit)
            actions[u] = (self._turn_choice(env, u, float(env.headings[u]))
                          if near_edge else straight)
        return ActionOutput(actions=actions)


def main(seed: int = 0) -> None:
    env = make_env(seed=seed)
    roll = rollout(env, get_policy("wallfollower", margin=5.0), seed=seed,
                   record_frames=False)
    m = roll.metrics
    print("wall-follower controller")
    print("  coverage      %.3f" % m.coverage)
    print("  recall        %.3f (%d/%d)" % (m.recall, m.confirmed, m.target_count))
    print("  T80 coverage  %s" % (m.t80_coverage if m.t80_coverage is not None else "never"))
    print("  return        %.1f" % m.return_sum)


if __name__ == "__main__":
    main()
