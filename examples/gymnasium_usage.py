#!/usr/bin/env python
"""Using the environment through Gymnasium, PettingZoo and the raw API."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import make_env
from hocca_sim.spaces import GYMNASIUM_AVAILABLE
from hocca_sim.wrappers import GymnasiumMultiAgentWrapper, PettingZooParallelWrapper


def raw_environment() -> None:
    print("--- raw HoCCAEnv -------------------------------------------")
    env = make_env(seed=0)
    obs, info = env.reset(seed=0)
    print("  per-agent observation :", obs["agents"].shape)
    print("  centralised state     :", obs["state"].shape)
    print("  action space          :", env.action_space)
    print("  option space          :", env.option_space)
    for _ in range(5):
        result = env.step(np.zeros(env.n_agents, dtype=int))
    print("  after 5 steps         : coverage %.3f, confirmed %d"
          % (result.info["coverage"], result.info["confirmed"]))


def gymnasium_api() -> None:
    print("--- GymnasiumMultiAgentWrapper -----------------------------")
    env = GymnasiumMultiAgentWrapper(make_env(seed=1))
    obs, info = env.reset(seed=1)
    print("  observation keys      :", sorted(obs)[:4], "...")
    print("  obs['agent_0'].shape  :", obs["agent_0"].shape)
    print("  obs['state'].shape    :", obs["state"].shape)
    obs, rewards, terms, truncs, info = env.step({name: 1 for name in env.agents})
    print("  rewards               :",
          {k: round(v, 1) for k, v in list(rewards.items())[:2]})
    env.close()


def pettingzoo_api() -> None:
    print("--- PettingZooParallelWrapper ------------------------------")
    env = PettingZooParallelWrapper(make_env(seed=2))
    observations, infos = env.reset(seed=2)
    print("  agents                :", env.agents)
    observations, rewards, terminations, truncations, infos = env.step(
        {name: 0 for name in env.agents}
    )
    print("  next observations     :",
          {k: v.shape for k, v in list(observations.items())[:2]})
    print("  terminations          :", terminations)
    env.close()


def main() -> None:
    raw_environment()
    if GYMNASIUM_AVAILABLE:
        gymnasium_api()
    else:
        print("--- GymnasiumMultiAgentWrapper -----------------------------")
        print("  skipped: pip install gymnasium to enable it")
    pettingzoo_api()


if __name__ == "__main__":
    main()
