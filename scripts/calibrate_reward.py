#!/usr/bin/env python
"""Print the reward-component breakdown of an episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import SimConfig, make_env
from hocca_sim.policies import RuleBasedPolicy
from hocca_sim.rollout import rollout

RETURN_TARGET = 700.0
"""Order of magnitude the default ``coverage_coef`` aims for (illustrative)."""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reward-component report.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--coverage-coef", type=float, default=None)
    args = parser.parse_args(argv)

    totals = []
    for offset in range(args.seeds):
        cfg = SimConfig()
        if args.coverage_coef is not None:
            cfg.reward.coverage_coef = args.coverage_coef
        env = make_env(cfg, seed=offset)
        roll = rollout(env, RuleBasedPolicy(), seed=offset, record_frames=False)
        totals.append(roll.metrics.extra)

    print("%-14s %14s" % ("component", "mean"))
    print("-" * 30)
    for key in totals[0]:
        values = np.array([t[key] for t in totals], dtype=float)
        print("%-14s %10.2f  (sd %.1f)" % (key, values.mean(), values.std()))
    print("-" * 30)
    team = np.array([t["team"] for t in totals], dtype=float)
    print("%-14s %10.2f  (sd %.1f)" % ("episode return", team.mean(), team.std()))
    print("\norder-of-magnitude target for the episode return: %.0f" % RETURN_TARGET)
    print("coverage_coef in use: %.4f"
          % (args.coverage_coef if args.coverage_coef is not None
             else SimConfig().reward.coverage_coef))
    print("\nnote: the rule-based controller is an example, not a trained agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
