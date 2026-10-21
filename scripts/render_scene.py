#!/usr/bin/env python
"""Render the HOCCA scene in the paper's style."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hocca_sim import available_policies, get_policy, make_env
from hocca_sim.render import (
    SceneStyle, animate_rollout, save_scene, set_matplotlib_style,
)
from hocca_sim.rollout import rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the HOCCA search scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--animate", action="store_true")
    source.add_argument("--save", metavar="PNG", type=Path)
    source.add_argument("--gif", metavar="GIF", type=Path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--policy", default="rule", choices=available_policies())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fleet", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-footprints", action="store_true")
    parser.add_argument("--paper-cells", action="store_true")
    parser.add_argument("--plain-ground", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = None
    if args.fleet:
        from hocca_sim import SimConfig

        cfg = SimConfig().with_fleet(args.fleet)
    env = make_env(cfg, seed=args.seed)
    roll = rollout(env, get_policy(args.policy), seed=args.seed)

    step = roll.steps if args.step is None else int(args.step)
    style = SceneStyle(
        grid_size=env.grid_size,
        sensor_radius=env.sensing_cfg.radius,
        cell_offset=0.0 if args.paper_cells else 1.0,
        figure_size=(8.0, 8.2),
        background_mode="plain" if args.plain_ground else "texture",
        show_footprints=not args.no_footprints,
    )

    if args.save:
        path = save_scene(roll, args.save, step=step, style=style, dpi=args.dpi)
        print("Saved:", path.resolve())
        index = min(step, roll.steps)
        print("Coverage %.1f%%  |  Confirmed %d/%d  |  t = %d"
              % (100.0 * roll.coverage[index], int(roll.confirmed[index]),
                 env.n_targets, step))
        return 0

    if args.gif:
        path = animate_rollout(roll, args.gif, style=style,
                              dpi=max(90, args.dpi // 2))
        print("Saved:", path.resolve())
        return 0

    import matplotlib.pyplot as plt
    import numpy as np

    set_matplotlib_style()
    from hocca_sim.render import (
        _confirmed_cells, _draw_masked_ground, _moving_flags,
        _update_agent_artists, build_scene,
    )

    fig, ax = plt.subplots(figsize=style.figure_size, facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.90)
    frame0 = roll.frame(0)
    handles = build_scene(
        positions=frame0["positions"],
        headings=frame0["headings"],
        paths=[np.asarray([frame0["positions"][u]]) for u in range(roll.n_agents)],
        targets=frame0["targets"],
        moving=_moving_flags(roll, roll.target_positions.shape[1]),
        style=style,
        coverage=frame0["visited"].astype(float),
        confirmed_cells=_confirmed_cells(roll, 0, frame0),
        visited=frame0["visited"],
        figure=fig,
        axes=ax,
        step_label="0",
    )
    title = ax.set_title("", loc="left", fontsize=style.title_size,
                         fontweight="bold", color=style.target_color, pad=15)
    footer = fig.text(0.55, 0.035, "", ha="center", fontsize=style.footer_size,
                      fontweight="bold", color=style.target_color)

    def update(index: int):
        frame = roll.frame(index)
        for image in list(ax.images):
            image.remove()
        _draw_masked_ground(ax, style, frame["visited"].astype(float))
        _update_agent_artists(
            handles.artists, frame["positions"], frame["headings"],
            [roll.positions[: index + 1, u, :] for u in range(roll.n_agents)],
            index, style,
        )
        title.set_text(style.title_template.format(n=roll.n_agents, step=index))
        footer.set_text(style.footer_template.format(
            coverage=100.0 * frame["coverage"], confirmed=frame["confirmed"],
            total=int(frame["targets"].shape[0]),
        ))
        return tuple(handles.artists)

    if args.animate:
        from matplotlib.animation import FuncAnimation

        roll.animation = FuncAnimation(fig, update, frames=range(roll.steps + 1),
                                       interval=60, repeat=True, blit=False)
    else:
        update(roll.steps)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
