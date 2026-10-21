#!/usr/bin/env python
"""Run one HOCCA reference episode and print the evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import available_policies, get_policy, make_env
from hocca_sim.rollout import rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one episode of the HOCCA reference environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fleet", type=int, default=None)
    parser.add_argument("--targets", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--p-detect", type=float, default=None)
    parser.add_argument("--p-fa", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--packets", type=int, default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--no-comm", action="store_true")
    parser.add_argument("--reference-layout", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--trace", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = None
    if args.config is not None:
        from hocca_sim import SimConfig

        cfg = SimConfig.from_file(args.config)
    if args.reference_layout:
        from hocca_sim import SimConfig

        cfg = cfg or SimConfig()
        cfg.targets.reference_override = True
    if args.grid is not None:
        from hocca_sim import SimConfig

        cfg = cfg or SimConfig()
        cfg.grid.size = args.grid
    if args.fleet:
        from hocca_sim import SimConfig

        cfg = (cfg or SimConfig()).with_fleet(args.fleet)

    env = make_env(cfg, seed=args.seed)
    if args.targets is not None:
        env.target_cfg.count = args.targets
    if args.horizon is not None:
        env.episode_cfg.horizon = args.horizon
        env.horizon = args.horizon
    if args.p_detect is not None:
        env.sensor.p_detect = args.p_detect
        env.sensing_cfg.p_detect = args.p_detect
    if args.p_fa is not None:
        env.sensor.p_false_alarm = args.p_fa
        env.sensing_cfg.p_false_alarm = args.p_fa
    if args.p_detect is not None or args.p_fa is not None:
        env.sensor.log_odds_hit = float(
            np.log(env.sensor.p_detect / env.sensor.p_false_alarm)
        )
        env.sensor.log_odds_miss = float(
            np.log((1 - env.sensor.p_detect) / (1 - env.sensor.p_false_alarm))
        )
    if args.radius is not None:
        from hocca_sim.sensors import SensingModel

        env.sensor = SensingModel(env.grid_size, args.radius,
                                  env.sensor.p_detect, env.sensor.p_false_alarm)
        env.sensing_cfg.radius = args.radius
    if args.packets is not None:
        env.comm_cfg.packets_per_step = args.packets
    if args.items is not None:
        env.comm_cfg.items_per_packet = args.items
    if args.no_comm:
        env.comm_cfg.packets_per_step = 0

    policy = get_policy(args.policy)
    env.reset(seed=args.seed)
    roll = rollout(env, policy, seed=args.seed)
    metrics = roll.metrics

    def fmt(value):
        return "n/a" if value is None else "%d" % value

    print("=" * 66)
    print("HOCCA reference environment  --  policy '%s'  seed %d"
          % (args.policy, args.seed))
    print("-" * 66)
    print("grid / targets / fleet     : %dx%d / %d (%d static, %d moving) / %d"
          % (env.grid_size, env.grid_size, env.n_targets,
             env.target_cfg.static_count, env.target_cfg.moving_count, env.n_agents))
    print("sensor                     : R_s=%.1f  p_D=%.2f  p_FA=%.3f"
          % (env.sensor.radius, env.sensor.p_detect, env.sensor.p_false_alarm))
    print("communication              : B=%d  m=%d  Q=%d items/UAV-step"
          % (env.comm_cfg.packets_per_step, env.comm_cfg.items_per_packet,
             env.comm_cfg.item_budget))
    print("-" * 66)
    print("T80 coverage   (steps)     : %s" % fmt(metrics.t80_coverage))
    print("T80 detection  (steps)     : %s" % fmt(metrics.t80_detection))
    print("T_all (steps)              : %s" % fmt(metrics.t_all))
    print("final coverage             : %.3f" % metrics.coverage)
    print("final recall               : %.3f  (%d/%d)"
          % (metrics.recall, metrics.confirmed, metrics.target_count))
    print("precision                  : %.3f" % metrics.precision)
    print("messages per UAV-step      : %.2f  (%d packets)"
          % (metrics.messages_per_uav_step, metrics.packets))
    print("total fleet moves          : %d" % metrics.moves)
    print("episode return             : %.1f" % metrics.return_sum)
    print("application bytes          : %d" % env.comm.total_bytes)
    print("-" * 66)
    print("reward decomposition:")
    for key, value in metrics.extra.items():
        print("    %-14s %10.2f" % (key, value))
    print("=" * 66)

    if args.trace:
        print("\n step   coverage  recall  confirmed  packets  moves")
        for record in roll.trace:
            if record.step % 20 == 0 or record.step == roll.steps:
                print("%5d   %8.3f  %6.3f  %9d  %7d  %5d"
                      % (record.step, record.coverage, record.recall,
                         record.confirmed, record.packets_sent, record.moves))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        exists = args.csv.exists()
        with args.csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(["policy", "seed", "step", "coverage", "recall",
                                 "confirmed", "confirmed_cells", "false_cells",
                                 "entropy", "packets", "items", "moves", "reward"])
            for record in roll.trace:
                writer.writerow([args.policy, args.seed, record.step,
                                 "%.6f" % record.coverage, "%.6f" % record.recall,
                                 record.confirmed, record.confirmed_cells,
                                 record.false_cells, "%.4f" % record.entropy,
                                 record.packets_sent, record.items_attempted,
                                 record.moves, "%.4f" % record.reward_team])
        print("trace written to", args.csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
