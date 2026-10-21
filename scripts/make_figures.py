#!/usr/bin/env python
"""Generate the reference figures from the simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import SimConfig, make_env
from hocca_sim.policies import RandomPolicy, RuleBasedPolicy
from hocca_sim.render import (
    METHOD_COLORS, SceneStyle, make_case_study, make_setup_figure,
    plot_coverage_recall, plot_reward_curve, save_scene,
)
from hocca_sim.rollout import rollout

POLICIES = {"rule": RuleBasedPolicy, "random": RandomPolicy}

GUIDE_VALUES = {"T80_cov": 137, "T80_det": 176}
"""Optional reference guides, drawn only when ``--guides`` is passed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the HOCCA reference figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="rule", choices=sorted(POLICIES))
    parser.add_argument("--case-steps", default="80,180,260")
    parser.add_argument("--fleets", default="4,6,8")
    parser.add_argument("--only", default=None,
                        choices=["setup", "scene", "case", "curves", "reward"])
    parser.add_argument("--reward-json", type=Path, default=None)
    parser.add_argument("--guides", action="store_true",
                        help="draw optional reference guides on the curve panels")
    return parser


def figure_setup(out: Path, dpi: int) -> Path:
    return make_setup_figure(out / "fig_setup.png", dpi=dpi)


def figure_scene(out: Path, dpi: int, seed: int, policy_name: str,
                 step: int = 137) -> Path:
    env = make_env(seed=seed)
    roll = rollout(env, POLICIES[policy_name](), seed=seed)
    style = SceneStyle(grid_size=env.grid_size,
                       sensor_radius=env.sensing_cfg.radius,
                       cell_offset=1.0, background_mode="texture")
    return save_scene(roll, out / "scene.png", step=step, style=style, dpi=dpi)


def figure_case_study(out: Path, dpi: int, seed: int, policy_name: str,
                      steps: Sequence[int], fleets: Sequence[int]) -> Path:
    probe = make_env(seed=seed)
    probe.reset(seed=seed)
    layout = (
        probe.targets.positions[~probe.targets.is_moving].copy(),
        probe.targets.positions[probe.targets.is_moving].copy(),
    )
    rollouts = []
    for fleet in fleets:
        env = make_env(SimConfig().with_fleet(fleet), seed=seed)
        env.reset(seed=seed, options={"targets": layout})
        rollouts.append(rollout(env, POLICIES[policy_name](), seed=seed))

    style = SceneStyle(grid_size=rollouts[0].visited.shape[-1],
                       sensor_radius=rollouts[0].config.sensing.radius,
                       cell_offset=1.0)
    return make_case_study(rollouts, list(steps), out / "case_study.png",
                           dpi=dpi, style=style)


def figure_curves(out: Path, dpi: int, seed: int, names: Sequence[str],
                  repeats: int = 3, guides: bool = False) -> Path:
    curves: Dict[str, Dict[str, object]] = {}
    for name in names:
        cov_runs, rec_runs = [], []
        for offset in range(repeats):
            env = make_env(seed=seed + offset)
            roll = rollout(env, POLICIES[name](), seed=seed + offset)
            cov_runs.append(roll.coverage)
            rec_runs.append(roll.recall)
        length = min(len(c) for c in cov_runs)
        cov = np.stack([c[:length] for c in cov_runs])
        rec = np.stack([r[:length] for r in rec_runs])
        label = "HOCCA (rule-based replica)" if name == "rule" else name.upper()
        curves[label] = {
            "coverage": (cov.mean(axis=0), cov.std(axis=0)),
            "recall": (rec.mean(axis=0), rec.std(axis=0)),
        }
    METHOD_COLORS["HOCCA (rule-based replica)"] = "#8C564B"
    METHOD_COLORS["RANDOM"] = "#999999"

    anchors = None
    if guides:
        anchors = {
            "coverage": {"guide T80 =": GUIDE_VALUES["T80_cov"]},
            "recall": {"guide T80 =": GUIDE_VALUES["T80_det"]},
        }
    return plot_coverage_recall(curves, out / "coverage_recall.png", dpi=dpi,
                                anchors=anchors)


def figure_reward(out: Path, dpi: int, payload: Optional[Path]) -> Optional[Path]:
    if payload is None or not payload.exists():
        print("  (skipped: pass --reward-json with your own training logs)")
        return None
    curves = json.loads(payload.read_text(encoding="utf-8"))
    return plot_reward_curve(curves, out / "training_curves.png", dpi=dpi)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in args.case_steps.split(",") if s.strip()]
    fleets = [int(s) for s in args.fleets.split(",") if s.strip()]
    which = args.only or "all"
    produced: List[Path] = []

    if which in ("all", "setup"):
        print("[1/5] reference environment schematic ...")
        produced.append(figure_setup(args.out, args.dpi))
    if which in ("all", "scene"):
        print("[2/5] single rollout frame ...")
        produced.append(figure_scene(args.out, args.dpi, args.seed, args.policy))
    if which in ("all", "case"):
        print("[3/5] case study (fleets %s, steps %s) ..." % (fleets, steps))
        produced.append(figure_case_study(args.out, args.dpi, args.seed,
                                          args.policy, steps, fleets))
    if which in ("all", "curves"):
        print("[4/5] coverage / recall trajectories ...")
        produced.append(figure_curves(args.out, args.dpi, args.seed,
                                      ["rule", "random"], guides=args.guides))
    if which in ("all", "reward"):
        print("[5/5] training-dynamics template ...")
        path = figure_reward(args.out, args.dpi, args.reward_json)
        if path is not None:
            produced.append(path)

    print("\ngenerated:")
    for path in produced:
        print("  ", path)
    print("\nfigures produced by the demo controller; values are illustrative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
