#!/usr/bin/env python
"""Minimal example: uniformly random turn commands."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import make_env


def main(seed: int = 0) -> None:
    env = make_env(seed=seed)
    observation, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    total_return = 0.0
    done = False
    while not done:
        actions = rng.integers(0, env.uav_cfg.action_count, size=env.n_agents)
        observation, rewards, terminated, truncated, info = env.step(actions)
        total_return += float(info["team_reward"])
        done = bool(terminated[0] or truncated[0])

    m = env.episode_metrics()
    print("random controller")
    print("  steps         %d" % m.steps)
    print("  return        %.1f" % total_return)
    print("  coverage      %.3f" % m.coverage)
    print("  recall        %.3f (%d/%d)" % (m.recall, m.confirmed, m.target_count))
    print("  T80 coverage  %s" % (m.t80_coverage if m.t80_coverage is not None else "never"))
    print("  T80 detection %s" % (m.t80_detection if m.t80_detection is not None else "never"))


if __name__ == "__main__":
    main()
