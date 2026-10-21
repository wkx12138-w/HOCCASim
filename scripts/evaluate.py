#!/usr/bin/env python
"""Evaluate a controller over several seeds and write metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import SimConfig, available_policies, get_policy, make_env
from hocca_sim.metrics import aggregate_metrics
from hocca_sim.rollout import rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a HOCCA controller over multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--fleet", default="4")
    parser.add_argument("--targets", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    template = SimConfig.from_file(args.config) if args.config else SimConfig()
    if args.grid:
        template.grid.size = args.grid
    if args.targets:
        template.targets.count = args.targets
    if args.horizon:
        template.episode.horizon = args.horizon

    fleets = [int(x) for x in str(args.fleet).split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {}
    for fleet in fleets:
        episodes = []
        for offset in range(args.seeds):
            seed = args.seed0 + offset
            cfg = template.with_fleet(fleet) if fleet != template.uavs.count else template
            env = make_env(cfg, seed=seed)
            metrics = rollout(env, get_policy(args.policy), seed=seed,
                              record_frames=False).metrics
            episodes.append(metrics)
            rows.append(dict(method=args.policy, fleet=fleet, seed=seed,
                             **metrics.as_row()))
            if not args.quiet:
                print("  N=%d seed=%2d  cov %.3f  rec %.3f  prec %.3f  "
                      "T80cov %-4s T80det %-4s  moves %4d  msgs/UAV-step %.2f  ret %7.1f"
                      % (fleet, seed, metrics.coverage, metrics.recall,
                         metrics.precision,
                         metrics.t80_coverage if metrics.t80_coverage is not None else "-",
                         metrics.t80_detection if metrics.t80_detection is not None else "-",
                         metrics.moves, metrics.messages_per_uav_step,
                         metrics.return_sum))
        summary["fleet_%d" % fleet] = aggregate_metrics(episodes)

    with (args.out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n%-6s %-12s %-12s %-8s %-8s %-8s %s"
          % ("N", "T80cov", "T80det", "cov", "recall", "prec", "msgs/UAV-step"))
    for fleet in fleets:
        block = summary["fleet_%d" % fleet]
        print("%-6d %-12s %-12s %-8.3f %-8.3f %-8.3f %.2f"
              % (fleet,
                 _fmt(block.get("t80_coverage_mean"), block.get("t80_coverage_std")),
                 _fmt(block.get("t80_detection_mean"), block.get("t80_detection_std")),
                 block["coverage_mean"], block["recall_mean"], block["precision_mean"],
                 block["messages_per_uav_step_mean"]))
    print("\nwritten: %s" % (args.out / "episodes.csv"))
    print("written: %s" % (args.out / "summary.json"))
    return 0


def _fmt(mean, std) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "never"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return "%.0f" % mean
    return "%.0f+-%.0f" % (mean, std)


if __name__ == "__main__":
    raise SystemExit(main())
