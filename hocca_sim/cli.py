"""Console entry points.

    hocca-sim run --policy rule --seed 0
    hocca-sim eval --policy rule --seeds 3 --fleet 4,6,8
    hocca-sim figures --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import SimConfig
from .env import make_env
from .policies import available_policies, get_policy
from .rollout import rollout

__all__ = ["main", "run_episode_main", "figures_main", "eval_main"]


def _print_metrics(roll, env) -> None:
    m = roll.metrics
    print("=" * 62)
    print("HOCCA reference environment  --  N=%d  K=%d  steps=%d"
          % (env.n_agents, env.n_targets, m.steps))
    print("-" * 62)
    print("T80 coverage   (steps) : %s" % (m.t80_coverage if m.t80_coverage is not None else "never"))
    print("T80 detection  (steps) : %s" % (m.t80_detection if m.t80_detection is not None else "never"))
    print("final coverage         : %.3f" % m.coverage)
    print("final recall           : %.3f (%d/%d)" % (m.recall, m.confirmed, m.target_count))
    print("precision              : %.3f" % m.precision)
    print("messages per UAV-step  : %.2f" % m.messages_per_uav_step)
    print("total fleet moves      : %d" % m.moves)
    print("episode return         : %.1f" % m.return_sum)
    print("=" * 62)


def run_episode_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hocca-sim run")
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fleet", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = SimConfig.from_file(args.config) if args.config else SimConfig()
    if args.fleet:
        cfg = cfg.with_fleet(args.fleet)
    env = make_env(cfg, seed=args.seed)
    _print_metrics(rollout(env, get_policy(args.policy), seed=args.seed,
                           record_frames=False), env)
    return 0


def eval_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hocca-sim eval")
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--fleet", default="4")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    from .metrics import aggregate_metrics

    fleets = [int(x) for x in str(args.fleet).split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    summary = {}
    for fleet in fleets:
        episodes = []
        for offset in range(args.seeds):
            seed = args.seed0 + offset
            env = make_env(SimConfig().with_fleet(fleet), seed=seed)
            roll = rollout(env, get_policy(args.policy), seed=seed, record_frames=False)
            episodes.append(roll.metrics)
            rows.append(dict(fleet=fleet, seed=seed, **roll.metrics.as_row()))
        summary["fleet_%d" % fleet] = aggregate_metrics(episodes)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("written:", args.out / "summary.json")
    return 0


def figures_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hocca-sim figures")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)

    from .render import SceneStyle, make_case_study, make_setup_figure, save_scene

    args.out.mkdir(parents=True, exist_ok=True)
    produced = [make_setup_figure(args.out / "fig_setup.png", dpi=args.dpi)]

    env = make_env(seed=args.seed)
    roll = rollout(env, get_policy(args.policy), seed=args.seed)
    style = SceneStyle(grid_size=env.grid_size,
                       sensor_radius=env.sensing_cfg.radius, cell_offset=1.0)
    produced.append(save_scene(roll, args.out / "scene.png", step=roll.steps,
                               style=style, dpi=args.dpi))

    probe = make_env(seed=args.seed)
    probe.reset(seed=args.seed)
    layout = (
        probe.targets.positions[~probe.targets.is_moving].copy(),
        probe.targets.positions[probe.targets.is_moving].copy(),
    )
    rollouts = []
    for fleet in (4, 6, 8):
        sub_env = make_env(SimConfig().with_fleet(fleet), seed=args.seed)
        sub_env.reset(seed=args.seed, options={"targets": layout})
        rollouts.append(rollout(sub_env, get_policy(args.policy), seed=args.seed))
    produced.append(make_case_study(rollouts, (80, 180, 260),
                                    args.out / "case_study.png", dpi=args.dpi,
                                    style=style))

    for path in produced:
        print("written:", path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hocca-sim")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "eval", "figures", "version"])
    known, rest = parser.parse_known_args(argv)
    if known.command == "version":
        from . import __version__

        print(__version__)
        return 0
    if known.command == "eval":
        return eval_main(rest)
    if known.command == "figures":
        return figures_main(rest)
    return run_episode_main(rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
