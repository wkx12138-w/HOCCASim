#!/usr/bin/env python
"""Run the built-in rule controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hocca_sim import make_env
from hocca_sim.policies import OPTION_NAMES, RuleBasedPolicy, RulePolicyConfig
from hocca_sim.rollout import rollout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the built-in rule controller.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fleet", type=int, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = None
    if args.fleet:
        from hocca_sim import SimConfig

        cfg = SimConfig().with_fleet(args.fleet)
    env = make_env(cfg, seed=args.seed)
    roll = rollout(env, RuleBasedPolicy(RulePolicyConfig()), seed=args.seed)
    m = roll.metrics

    print("rule controller, seed %d, fleet %d" % (args.seed, env.n_agents))
    print("  T80 coverage      %s" % (m.t80_coverage if m.t80_coverage is not None else "never"))
    print("  T80 detection     %s" % (m.t80_detection if m.t80_detection is not None else "never"))
    print("  coverage          %.3f" % m.coverage)
    print("  recall            %.3f (%d/%d)" % (m.recall, m.confirmed, m.target_count))
    print("  precision         %.3f" % m.precision)
    print("  messages/UAV-step %.2f" % m.messages_per_uav_step)
    print("  moves             %d" % m.moves)
    print("  return            %.1f" % m.return_sum)

    if roll.options is not None:
        counts = [int((roll.options[: roll.steps] == i).sum())
                  for i in range(env.option_count)]
        print("  option occupancy  " + "  ".join(
            "%s=%d" % (OPTION_NAMES[i], counts[i]) for i in range(env.option_count)))

    if args.save:
        from hocca_sim.render import SceneStyle, save_scene

        style = SceneStyle(grid_size=env.grid_size,
                           sensor_radius=env.sensing_cfg.radius, cell_offset=1.0)
        print("  saved             %s" % save_scene(roll, args.save, step=args.step,
                                                    style=style))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
