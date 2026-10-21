"""Figure style of the HOCCA paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SceneStyle", "PALETTE_8", "PALETTE_REFERENCE_4", "METHOD_COLORS",
    "make_background", "plane_outline", "build_scene", "render_env",
    "save_scene", "animate_rollout", "make_case_study", "make_setup_figure",
    "plot_coverage_recall", "plot_reward_curve", "set_matplotlib_style",
]

PALETTE_8: Tuple[str, ...] = (
    "#c74247", "#d38923", "#3167bf", "#39865a",
    "#8754ab", "#159ba5", "#9d6b3f", "#c55594",
)
PALETTE_REFERENCE_4: Tuple[str, ...] = ("#c34450", "#d29427", "#3976b5", "#408760")

METHOD_COLORS: Dict[str, str] = {
    "IQL": "#777777",
    "QMIX": "#56B4E9",
    "MADDPG": "#E69F00",
    "MAPPO": "#009E73",
    "DTH-MADDPG": "#8C7BC9",
    "HOCCA": "#0067B1",
}

BACKGROUND = "#edf0f2"
TARGET_COLOR = "#263743"
AMBER = "#c98a1a"
UNOBSERVED = "#ffffff"


def set_matplotlib_style() -> None:
    """Apply the paper's matplotlib settings."""

    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "font.size": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


@dataclass
class SceneStyle:
    """Visual constants of a scene panel."""

    grid_size: int = 48
    cell_offset: float = 1.0
    colors: Tuple[str, ...] = PALETTE_8
    background: str = BACKGROUND
    target_color: str = TARGET_COLOR
    amber: str = AMBER
    figure_size: Tuple[float, float] = (8.0, 8.2)
    sensor_radius: float = 3.0
    plane_scale: float = 0.80
    marker_size: float = 105.0
    trail_lw: float = 1.25
    tail_lw: float = 2.6
    grid_color: str = "#bfc8cf"
    grid_lw: float = 0.55
    grid_alpha: float = 0.45
    spine_color: str = "#7e8b95"
    spine_lw: float = 0.9
    tick_color: str = "#52616c"
    tick_size: float = 12.0
    title_size: float = 19.0
    footer_size: float = 13.0
    label_size: float = 12.0
    background_mode: str = "texture"
    show_trails: bool = True
    show_footprints: bool = True
    show_planes: bool = True
    show_labels: bool = True
    show_title: bool = True
    show_footer: bool = True
    xlabel: str = "Grid x"
    ylabel: str = "Grid y"
    title_template: str = "UAV = {n}  |  t = {step}"
    footer_template: str = "Coverage {coverage:.1f}%  |  Confirmed {confirmed}/{total}"
    note: str = "Illustrative scene"
    texture_seed: int = 7
    coverage_alpha: float = 1.0

    @property
    def low(self) -> float:
        return float(self.cell_offset) - 0.5

    @property
    def high(self) -> float:
        return float(self.cell_offset) + self.grid_size - 0.5

    @property
    def ticks(self) -> List[float]:
        values = [float(self.cell_offset), float(self.cell_offset) + 7.0]
        value = float(self.cell_offset) + 15.0
        while value <= self.cell_offset + self.grid_size:
            values.append(value)
            value += 8.0
        return values

    def colour(self, index: int) -> str:
        return self.colors[index % len(self.colors)]

    def to_display(self, x: Any, y: Any = None):
        if y is None:
            return np.asarray(x, dtype=np.float64) + float(self.cell_offset)
        return (
            np.asarray(x, dtype=np.float64) + float(self.cell_offset),
            np.asarray(y, dtype=np.float64) + float(self.cell_offset),
        )


def make_background(seed: int = 7, coarse: int = 24, cell: int = 8,
                    speckles: int = 90) -> np.ndarray:
    """Blocky light-grey ground texture, identical on every run."""

    rng = np.random.default_rng(seed)
    side = coarse * cell
    field = rng.uniform(0.92, 0.96, (coarse, coarse))
    texture = np.repeat(np.repeat(field, cell, axis=0), cell, axis=1)
    texture = np.repeat(texture[:, :, None], 3, axis=2)
    texture[:, :, 2] = np.clip(texture[:, :, 2] + 0.012, 0.0, 1.0)
    for _ in range(speckles):
        y, x = rng.integers(0, side - 3, size=2)
        texture[y:y + 3, x:x + 3] = (0.975, 0.980, 0.985)
    return texture


def plane_outline(scale: float = 0.80) -> np.ndarray:
    """Fixed-wing top view; ``+x`` is the nose."""

    outline = np.array([
        (1.12, 0.00), (0.78, 0.13), (0.14, 0.14),
        (-0.13, 1.02), (-0.37, 1.05), (-0.29, 0.13),
        (-0.72, 0.10), (-0.91, 0.43), (-1.08, 0.43),
        (-0.97, 0.00), (-1.08, -0.43), (-0.91, -0.43),
        (-0.72, -0.10), (-0.29, -0.13), (-0.37, -1.05),
        (-0.13, -1.02), (0.14, -0.14), (0.78, -0.13),
    ], dtype=np.float64)
    return outline * float(scale)


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _draw_ground(ax, style: SceneStyle, coverage: Optional[np.ndarray] = None) -> None:
    from matplotlib.patches import Polygon

    low, high = style.low, style.high
    if style.background_mode == "coverage" and coverage is not None:
        _draw_masked_ground(ax, style, np.asarray(coverage, dtype=float))
        return
    if style.background_mode == "plain":
        ax.add_patch(Polygon(
            [(low, low), (high, low), (high, high), (low, high)],
            closed=True, facecolor=style.background, edgecolor="none", zorder=0,
        ))
        return
    ax.imshow(
        make_background(seed=style.texture_seed),
        extent=(low, high, low, high), origin="lower",
        interpolation="bilinear", zorder=0, aspect="auto",
    )


def _draw_masked_ground(ax, style: SceneStyle, mask: np.ndarray) -> None:
    """White page, textured ground only inside ``mask``."""

    texture = make_background(seed=style.texture_seed)
    factor = max(texture.shape[0] // mask.shape[0], 1)
    upsampled = np.kron((mask > 0.5).astype(np.float64), np.ones((factor, factor)))
    rgb = np.where(upsampled[:, :, None] > 0.5, texture, np.array([1.0, 1.0, 1.0]))
    ax.imshow(
        rgb, extent=(style.low, style.high, style.low, style.high),
        origin="lower", interpolation="bilinear", zorder=0, aspect="auto",
    )


def _draw_frame(ax, style: SceneStyle) -> None:
    from matplotlib.patches import Polygon

    low, high = style.low, style.high
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_xticks(style.ticks)
    ax.set_yticks(style.ticks)
    ax.grid(color=style.grid_color, linewidth=style.grid_lw, alpha=style.grid_alpha)
    ax.tick_params(colors=style.tick_color, labelsize=style.tick_size)
    for spine in ax.spines.values():
        spine.set_color(style.spine_color)
        spine.set_linewidth(style.spine_lw)
    ax.add_patch(Polygon(
        [(low, low), (high, low), (high, high), (low, high)],
        closed=True, fill=False, edgecolor="white", linewidth=2.0, zorder=1,
    ))


def _draw_targets(ax, targets: np.ndarray, moving: np.ndarray, style: SceneStyle,
                  confirmed_cells: Optional[set] = None,
                  visited: Optional[np.ndarray] = None) -> None:
    if len(targets) == 0:
        return
    display = np.asarray(targets, dtype=np.float64) + float(style.cell_offset)
    stateful = confirmed_cells is not None or visited is not None
    for index, is_moving in enumerate(np.asarray(moving).astype(bool)):
        if not stateful:
            face, edge = style.target_color, "white"
        else:
            cell = (int(targets[index, 0]), int(targets[index, 1]))
            if confirmed_cells is not None and cell in confirmed_cells:
                face, edge = style.target_color, "white"
            elif visited is not None and visited[cell[1], cell[0]]:
                face, edge = style.amber, "white"
            else:
                face, edge = "white", style.target_color
        ax.scatter(
            [display[index, 0]], [display[index, 1]], s=style.marker_size,
            marker="s" if is_moving else "o", color=face, edgecolors=edge,
            linewidths=1.7, zorder=5,
        )


def _draw_agent_artists(ax, style: SceneStyle, count: int, start: np.ndarray):
    from matplotlib import patheffects
    from matplotlib.patches import Circle, Polygon

    outline = plane_outline(style.plane_scale)
    artists: List[Dict[str, Any]] = []
    for index in range(count):
        color = style.colour(index)
        trail, = ax.plot([], [], color=color, lw=style.trail_lw, alpha=0.93, zorder=3)
        tail, = ax.plot([], [], color=color, lw=style.tail_lw, alpha=0.95, zorder=4)
        ax.plot(
            start[index, 0] + style.cell_offset, start[index, 1] + style.cell_offset,
            marker="o", ms=6.5, mfc="white", mec=color, mew=1.6, zorder=6,
        )
        footprint = Circle(
            (0.0, 0.0), style.sensor_radius, fill=False, edgecolor=color,
            linestyle=(0, (3, 2)), linewidth=0.85, alpha=0.75,
            clip_on=False, zorder=4,
        )
        plane = Polygon(
            outline, closed=True, facecolor="#fbfbf6", edgecolor="#50616b",
            linewidth=0.65, zorder=8,
        )
        cockpit, = ax.plot([], [], color=color, linewidth=2.1, zorder=9)
        label = ax.text(
            0.0, 0.0, "U%d" % (index + 1), color=color, fontsize=style.label_size,
            fontweight="bold", zorder=10, ha="center", va="center",
        )
        label.set_path_effects([patheffects.withStroke(linewidth=4, foreground="white")])
        if style.show_footprints:
            ax.add_patch(footprint)
        if style.show_planes:
            ax.add_patch(plane)
        artists.append({
            "color": color, "outline": outline, "trail": trail, "tail": tail,
            "footprint": footprint, "plane": plane, "cockpit": cockpit, "label": label,
        })
    return artists


def _update_agent_artists(artists, positions, headings, paths, step, style) -> None:
    for index, art in enumerate(artists):
        position = np.asarray(positions[index], dtype=np.float64) + style.cell_offset
        rotation = _rotation(float(headings[index]))
        if style.show_planes:
            art["plane"].set_xy(art["outline"] @ rotation.T + position)
            cockpit = np.array([[0.20, 0.0], [0.58, 0.0]]) @ rotation.T + position
            art["cockpit"].set_data(cockpit[:, 0], cockpit[:, 1])
        path = paths[index]
        if len(path) == 0:
            continue
        display = path[: step + 1] + style.cell_offset
        if style.show_trails:
            art["trail"].set_data(display[:, 0], display[:, 1])
            recent = display[max(0, len(display) - 10):]
            art["tail"].set_data(recent[:, 0], recent[:, 1])
        if style.show_footprints:
            art["footprint"].center = (position[0], position[1])
        if style.show_labels:
            offset = (0.0, 2.5) if index == 0 else (0.0, -2.2)
            art["label"].set_position((
                np.clip(position[0] + offset[0], style.low + 1.5, style.high - 1.5),
                np.clip(position[1] + offset[1], style.low + 1.5, style.high - 1.5),
            ))


@dataclass
class SceneHandles:
    """Figure, axes and artists of one scene panel."""

    fig: Any
    ax: Any
    artists: List[Dict[str, Any]]
    style: SceneStyle
    title: Any = None
    footer: Any = None


def build_scene(
    positions: np.ndarray,
    headings: np.ndarray,
    paths: Sequence[np.ndarray],
    targets: np.ndarray,
    moving: np.ndarray,
    style: Optional[SceneStyle] = None,
    coverage: Optional[np.ndarray] = None,
    confirmed_cells: Optional[set] = None,
    visited: Optional[np.ndarray] = None,
    figure: Any = None,
    axes: Any = None,
    step_label: str = "",
    start_positions: Optional[np.ndarray] = None,
) -> SceneHandles:
    """Draw one scene panel."""

    import matplotlib.pyplot as plt

    style = style or SceneStyle()
    set_matplotlib_style()

    if axes is None:
        fig, ax = plt.subplots(figsize=style.figure_size, facecolor="white")
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.90)
    else:
        fig, ax = figure, axes
    ax.set_facecolor(
        style.background if style.background_mode != "coverage" else "white"
    )

    mask = coverage if coverage is not None else visited
    if style.background_mode == "coverage" and mask is not None:
        _draw_masked_ground(ax, style, np.asarray(mask, dtype=float))
    else:
        _draw_ground(ax, style, None)

    _draw_frame(ax, style)
    _draw_targets(ax, targets, moving, style, confirmed_cells, visited if visited is not None else coverage)

    if start_positions is None:
        start = np.asarray([
            np.asarray(p, dtype=np.float64)[0] if len(p) else positions[u]
            for u, p in enumerate(paths)
        ], dtype=np.float64)
    else:
        start = np.asarray(start_positions, dtype=np.float64)

    artists = _draw_agent_artists(ax, style, len(positions), start)
    _update_agent_artists(artists, positions, headings, paths, len(paths[0]) - 1, style)

    handles = SceneHandles(fig=fig, ax=ax, artists=artists, style=style)
    if style.show_title:
        handles.title = ax.set_title(
            style.title_template.format(n=len(positions), step=step_label or 0),
            loc="left", fontsize=style.title_size, fontweight="bold",
            color=style.target_color, pad=15,
        )
    return handles


def render_env(env: Any, mode: str = "rgb_array") -> Optional[np.ndarray]:
    """Render the live environment."""

    import matplotlib.pyplot as plt

    style = SceneStyle(
        grid_size=env.grid_size, sensor_radius=env.sensing_cfg.radius, cell_offset=0.0
    )
    handles = build_scene(
        positions=np.asarray(env.positions),
        headings=np.asarray(env.headings),
        paths=[np.asarray([p], dtype=np.float64) for p in env.positions],
        targets=env.evaluator.target_positions,
        moving=env.evaluator.target_is_moving,
        style=style,
        coverage=env.visited.astype(float),
        confirmed_cells=env.confirmed_cells,
        visited=env.visited,
        step_label=str(env.step_index),
    )
    handles.fig.text(
        0.55, 0.035,
        style.footer_template.format(
            coverage=100.0 * float(env.visited.mean()),
            confirmed=int(env.confirmed_count), total=int(env.n_targets),
        ),
        ha="center", fontsize=style.footer_size, fontweight="bold",
        color=style.target_color,
    )
    handles.fig.text(0.96, 0.008, style.note, ha="right", color="#8a959e", fontsize=8)

    if mode in ("rgb_array", None):
        handles.fig.canvas.draw()
        image = np.asarray(handles.fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(handles.fig)
        return image
    plt.show()
    return None


def save_scene(roll: Any, path: str | Path, step: Optional[int] = None,
               style: Optional[SceneStyle] = None, dpi: int = 200) -> Path:
    """Save one frame of a rollout."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    step = roll.steps if step is None else int(step)
    style = style or SceneStyle(
        grid_size=roll.visited.shape[-1],
        sensor_radius=roll.config.sensing.radius if roll.config is not None else 3.0,
    )
    frame = roll.frame(step)
    handles = build_scene(
        positions=frame["positions"],
        headings=frame["headings"],
        paths=[roll.positions[: step + 1, u, :] for u in range(roll.n_agents)],
        targets=frame["targets"],
        moving=_moving_flags(roll, frame["targets"].shape[0]),
        style=style,
        coverage=frame["visited"].astype(float),
        confirmed_cells=_confirmed_cells(roll, step, frame),
        visited=frame["visited"],
        step_label=str(step),
    )
    handles.fig.text(
        0.55, 0.035,
        style.footer_template.format(
            coverage=100.0 * frame["coverage"], confirmed=frame["confirmed"],
            total=int(frame["targets"].shape[0]),
        ),
        ha="center", fontsize=style.footer_size, fontweight="bold",
        color=style.target_color,
    )
    handles.fig.text(0.96, 0.008, style.note, ha="right", color="#8a959e", fontsize=8)

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handles.fig.savefig(str(destination), dpi=dpi, facecolor="white")
    plt.close(handles.fig)
    return destination


def _moving_flags(roll: Any, count: int) -> np.ndarray:
    flags = np.zeros(count, dtype=bool)
    cfg = roll.config
    if cfg is not None and count:
        n_moving = int(cfg.targets.moving_count)
        if n_moving:
            flags[count - n_moving:] = True
    return flags


def _confirmed_cells(roll: Any, step: int, frame: Dict[str, Any]) -> set:
    stored = getattr(roll, "confirmed_sets", None)
    if stored:
        return stored[min(step, len(stored) - 1)]
    targets, visited = frame["targets"], frame["visited"]
    keep, budget = set(), int(frame["confirmed"])
    for index in range(len(targets)):
        if budget <= 0:
            break
        cell = (int(targets[index, 0]), int(targets[index, 1]))
        if visited[cell[1], cell[0]]:
            keep.add(cell)
            budget -= 1
    return keep


def make_case_study(
    rollouts: Sequence[Any],
    steps: Sequence[int],
    path: str | Path,
    dpi: int = 320,
    figure_size: Tuple[float, float] = (11.0, 10.55),
    style: Optional[SceneStyle] = None,
) -> Path:
    """Grid of fleet sizes against timestamps."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    base = style or SceneStyle()
    style = SceneStyle(
        grid_size=base.grid_size,
        sensor_radius=base.sensor_radius,
        cell_offset=base.cell_offset,
        colors=PALETTE_8,
        background_mode="coverage",
        figure_size=figure_size,
        show_title=True,
        note="Evaluator-side ground-truth overlay",
    )
    set_matplotlib_style()

    rows, cols = len(rollouts), len(steps)
    fig, axes = plt.subplots(
        rows, cols, figsize=figure_size, facecolor="white",
        gridspec_kw={"wspace": 0.10, "hspace": 0.16},
    )
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.945)

    for row, roll in enumerate(rollouts):
        moving = _moving_flags(roll, roll.target_positions.shape[1])
        for col, value in enumerate(steps):
            step = int(np.clip(value, 0, roll.steps))
            ax = axes[row, col]
            frame = roll.frame(step)
            build_scene(
                positions=frame["positions"],
                headings=frame["headings"],
                paths=[roll.positions[: step + 1, u, :] for u in range(roll.n_agents)],
                targets=frame["targets"],
                moving=moving,
                style=style,
                coverage=frame["visited"].astype(float),
                confirmed_cells=_confirmed_cells(roll, step, frame),
                visited=frame["visited"],
                figure=fig,
                axes=ax,
                step_label=str(step),
            )
            ax.set_title(
                "N = %d   |   t = %d" % (roll.n_agents, step), loc="left",
                fontsize=13.5, fontweight="bold", color=style.target_color, pad=8,
            )
            ax.text(
                0.5, -0.115,
                "C = %.1f%%   |   confirmed %d/%d"
                % (100.0 * frame["coverage"], frame["confirmed"],
                   int(frame["targets"].shape[0])),
                transform=ax.transAxes, ha="center", va="top", fontsize=11.0,
                fontweight="bold", color=style.target_color,
            )
            if col > 0:
                ax.set_ylabel("")
            if row < rows - 1:
                ax.set_xlabel("")

    _agent_legend(fig, style, max(r.n_agents for r in rollouts))
    fig.text(
        0.995, 0.006,
        "Sensed area: grey.  Unobserved: white.  Circle = static target, "
        "square = moving target.",
        ha="right", color="#8a959e", fontsize=9,
    )

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), dpi=dpi, facecolor="white")
    plt.close(fig)
    return destination


def _agent_legend(fig, style: SceneStyle, count: int, y: float = 0.988) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=style.colour(i), lw=2.2, label="UAV-%d" % (i + 1))
        for i in range(count)
    ]
    legend = fig.legend(
        handles=handles, loc="upper center", ncol=min(count, 8), frameon=False,
        fontsize=10.0, bbox_to_anchor=(0.5, y), handlelength=2.6,
        columnspacing=1.6, handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(style.target_color)


def make_setup_figure(
    path: str | Path,
    static: Optional[np.ndarray] = None,
    moving: Optional[np.ndarray] = None,
    uav_positions: Optional[np.ndarray] = None,
    corridors: Optional[Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    style: Optional[SceneStyle] = None,
    dpi: int = 300,
) -> Path:
    """Reference environment schematic."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .config import (
        REFERENCE_CORRIDORS, REFERENCE_MOVING_TARGETS, REFERENCE_STATIC_TARGETS,
        REFERENCE_UAV_POSES,
    )

    static = (np.asarray(REFERENCE_STATIC_TARGETS, dtype=np.float64)
              if static is None else np.asarray(static, dtype=np.float64))
    moving = (np.asarray(REFERENCE_MOVING_TARGETS, dtype=np.float64)
              if moving is None else np.asarray(moving, dtype=np.float64))
    positions = (np.asarray(REFERENCE_UAV_POSES, dtype=np.float64)
                 if uav_positions is None else np.asarray(uav_positions, dtype=np.float64))
    corridors = REFERENCE_CORRIDORS if corridors is None else corridors

    style = style or SceneStyle(
        grid_size=48, cell_offset=1.0, figure_size=(6.4, 6.6),
        background_mode="texture",
    )
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=style.figure_size, facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.88)

    _draw_ground(ax, style, None)
    _draw_frame(ax, style)

    for (x1, y1), (x2, y2) in corridors:
        ax.plot(
            [x1 + style.cell_offset, x2 + style.cell_offset],
            [y1 + style.cell_offset, y2 + style.cell_offset],
            color="#d55e00", linestyle=(0, (3, 2)), linewidth=1.1, zorder=2,
        )

    _draw_targets(ax, static, np.zeros(len(static), dtype=bool), style)
    _draw_targets(ax, moving, np.ones(len(moving), dtype=bool), style)

    headings = _centripetal_headings(positions, style.grid_size)
    artists = _draw_agent_artists(ax, style, len(positions), positions)
    _update_agent_artists(
        artists, positions, headings,
        [np.asarray([p]) for p in positions], 0, style,
    )

    legend = fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=style.target_color, markeredgewidth=1.4,
                   markersize=8, label="Static target (12)"),
            Line2D([], [], marker="s", linestyle="none", markerfacecolor="white",
                   markeredgecolor="#d55e00", markeredgewidth=1.4,
                   markersize=8, label="Moving target (4)"),
            Line2D([], [], color="#d55e00", linestyle=(0, (3, 2)), linewidth=1.1,
                   label="Example motion corridor"),
        ],
        loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    for text in legend.get_texts():
        text.set_color(style.target_color)

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), dpi=dpi, facecolor="white")
    plt.close(fig)
    return destination


def _centripetal_headings(positions: np.ndarray, grid_size: int) -> np.ndarray:
    centre = np.array([grid_size / 2.0, grid_size / 2.0])
    angles = []
    for p in positions:
        delta = centre - p
        angles.append(
            float(np.arctan2(delta[1], delta[0]))
            if np.linalg.norm(delta) > 1e-9 else 0.0
        )
    return np.asarray(angles)


def plot_coverage_recall(
    curves: Dict[str, Dict[str, Any]],
    path: str | Path,
    steps: Optional[Sequence[int]] = None,
    threshold: float = 0.8,
    dpi: int = 300,
    figure_size: Tuple[float, float] = (10.0, 3.9),
    anchors: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """Two-panel coverage and recall plot."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    set_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=figure_size, facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.93, wspace=0.22)

    panels = (
        ("coverage", "(a) Footprint coverage", "Coverage ratio"),
        ("recall", "(b) Target recall", "Detection recall"),
    )
    x = np.asarray(steps if steps is not None else [])
    for ax, (key, title, ylabel) in zip(axes, panels):
        for method, payload in curves.items():
            mean = np.asarray(payload[key][0], dtype=float)
            std = payload[key][1]
            axis = np.asarray(steps) if steps is not None else np.arange(len(mean))
            colour = METHOD_COLORS.get(method, "#444444")
            if std is not None and len(std) == len(mean):
                std = np.asarray(std, dtype=float)
                ax.fill_between(
                    axis, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                    color=colour, alpha=0.16, linewidth=0,
                )
            ax.plot(
                axis, mean, color=colour,
                lw=1.9 if method.upper().startswith("HOCCA") else 1.15,
                label=method,
            )
            x = axis
        ax.axhline(threshold, color="#555555", linestyle=(0, (3, 2)), lw=0.8, zorder=1)
        if anchors and key in anchors:
            for label, position in anchors[key].items():
                ax.axvline(float(position), color="#0067B1",
                           linestyle=(0, (1, 2)), lw=0.9, alpha=0.85, zorder=2)
                ax.annotate(
                    "%s %.0f" % (label, float(position)),
                    xy=(float(position), threshold), xytext=(4, 6),
                    textcoords="offset points", fontsize=8.5, color="#0067B1",
                )
        ax.set_xlim(0, float(x[-1]) if len(x) else 260)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Environment step")
        ax.set_ylabel(ylabel)
        ax.grid(color="#dedede", lw=0.35)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.65)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold",
                     color=TARGET_COLOR)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", ncol=min(len(labels), 6),
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005),
    )
    for text in legend.get_texts():
        text.set_color(TARGET_COLOR)

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), dpi=dpi, facecolor="white")
    plt.close(fig)
    return destination


def plot_reward_curve(
    curves: Dict[str, Dict[str, Any]],
    path: str | Path,
    dpi: int = 300,
    figure_size: Tuple[float, float] = (7.2, 4.2),
    ylabel: str = "Episode return",
    xlabel: str = "Episode",
) -> Path:
    """Training-dynamics plot."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.14, top=0.955)

    for method, payload in curves.items():
        mean = np.asarray(payload["mean"], dtype=float)
        axis = np.asarray(payload.get("x", np.arange(len(mean))))
        colour = METHOD_COLORS.get(method, "#444444")
        std = payload.get("std")
        if std is not None:
            std = np.asarray(std, dtype=float)
            ax.fill_between(axis, mean - std, mean + std, color=colour,
                            alpha=0.18, linewidth=0)
        ax.plot(axis, mean, color=colour,
                lw=2.1 if method.upper().startswith("HOCCA") else 1.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#dedede", lw=0.35)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.65)
    legend = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for text in legend.get_texts():
        text.set_color(TARGET_COLOR)

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), dpi=dpi, facecolor="white")
    plt.close(fig)
    return destination


def animate_rollout(
    roll: Any,
    path: str | Path,
    style: Optional[SceneStyle] = None,
    interval: int = 60,
    dpi: int = 130,
    fps: int = 16,
    max_frames: Optional[int] = None,
) -> Path:
    """Animate a rollout to GIF or MP4."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    style = style or SceneStyle(
        grid_size=roll.visited.shape[-1],
        sensor_radius=roll.config.sensing.radius if roll.config is not None else 3.0,
    )
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=style.figure_size, facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.90)

    moving = _moving_flags(roll, roll.target_positions.shape[1])
    frame0 = roll.frame(0)
    handles = build_scene(
        positions=frame0["positions"],
        headings=frame0["headings"],
        paths=[np.asarray([frame0["positions"][u]]) for u in range(roll.n_agents)],
        targets=frame0["targets"],
        moving=moving,
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

    total = roll.steps + 1
    frames = (range(total) if max_frames is None
              else range(0, total, max(1, total // max_frames)))

    def update(step: int):
        frame = roll.frame(step)
        for image in list(ax.images):
            image.remove()
        _draw_masked_ground(ax, style, frame["visited"].astype(float))
        _update_agent_artists(
            handles.artists, frame["positions"], frame["headings"],
            [roll.positions[: step + 1, u, :] for u in range(roll.n_agents)],
            step, style,
        )
        title.set_text(style.title_template.format(n=roll.n_agents, step=step))
        footer.set_text(style.footer_template.format(
            coverage=100.0 * frame["coverage"], confirmed=frame["confirmed"],
            total=int(frame["targets"].shape[0]),
        ))
        return tuple(handles.artists)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval,
                                   repeat=True, blit=False)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".gif":
        anim.save(str(destination), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(str(destination), dpi=dpi)
    plt.close(fig)
    return destination
