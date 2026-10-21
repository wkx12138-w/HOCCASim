"""Built-in rule-based controller: serpentine sweep plus belief-driven confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..beliefs import binary_entropy
from ..kinematics import angle_difference, wrap_angle
from ..sensors import disk_offsets
from .base import ActionOutput, Policy, register_policy

__all__ = ["RuleBasedPolicy", "RulePolicyConfig", "OPTION_NAMES"]

OPTION_NAMES = ("SWEEP", "INSPECT", "CONFIRM", "RELOCATE")


@dataclass
class RulePolicyConfig:
    """Controller weights and thresholds."""

    lane_spacing: float = 5.0
    sweep_lookahead: int = 6
    sweep_track_radius: float = 4.0
    sector_inner_margin: int = 1
    sweep_passes: int = 2

    weight_frontier: float = 1.0
    weight_uncertainty: float = 0.55
    weight_confirm: float = 6.0
    weight_recency: float = 0.85
    weight_travel: float = 0.045
    weight_separation: float = 2.6
    separation_sigma: float = 7.0
    weight_keep_away: float = 1.6
    keep_away_sigma: float = 3.5
    goal_reach_horizon: float = 24.0

    confirm_trigger: float = 1.5
    divert_radius: float = 20.0
    confirm_dwell: int = 6
    inspect_trigger: float = 2.0

    goal_switch_margin: float = 0.90
    goal_reached_radius: float = 1.5
    max_option_age: int = 8
    interior_margin: int = 2

    def __post_init__(self) -> None:
        if self.max_option_age < 1:
            raise ValueError("max_option_age must be >= 1")
        if self.lane_spacing < 1.0:
            raise ValueError("lane_spacing must be >= 1")


class RuleBasedPolicy(Policy):
    """Serpentine sweep plus belief-driven confirmation."""

    name = "rule"
    uses_options = True

    def __init__(self, config: Optional[RulePolicyConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = RulePolicyConfig(**kwargs)
        elif kwargs:
            for key, value in kwargs.items():
                setattr(config, key, value)
        self.cfg = config
        self._goals: Optional[np.ndarray] = None
        self._options: Optional[np.ndarray] = None
        self._option_age: Optional[np.ndarray] = None
        self._polylines: List[np.ndarray] = []
        self._cursors: Optional[np.ndarray] = None
        self._passes: Optional[np.ndarray] = None
        self._kernel: Optional[np.ndarray] = None
        self._kernel_fft: Dict[int, np.ndarray] = {}
        self._offsets: Optional[np.ndarray] = None
        self._grid_size: Optional[int] = None

    def reset(self, env: Any) -> None:
        n = env.n_agents
        self._goals = np.full((n, 2), -1, dtype=np.int64)
        self._options = np.zeros(n, dtype=np.int64)
        self._option_age = np.zeros(n, dtype=np.int64)
        self._cursors = np.zeros(n, dtype=np.int64)
        self._passes = np.zeros(n, dtype=np.int64)
        self._grid_size = env.grid_size
        self._offsets = disk_offsets(env.sensing_cfg.radius, env.grid_size)
        self._kernel = self._disk_kernel(env)
        self._polylines = [self._build_polyline(env, u) for u in range(n)]
        self._kernel_fft = {}

    def _disk_kernel(self, env: Any) -> np.ndarray:
        span = int(np.floor(env.sensing_cfg.radius))
        kernel = np.zeros((2 * span + 1, 2 * span + 1), dtype=np.float64)
        kernel[self._offsets[:, 1] + span, self._offsets[:, 0] + span] = 1.0
        return kernel

    def _kernel_transfer(self, size: int) -> np.ndarray:
        cached = self._kernel_fft.get(size)
        if cached is not None:
            return cached
        grid = np.zeros((size, size), dtype=np.float64)
        for dx, dy in self._offsets:
            grid[int(dy) % size, int(dx) % size] = 1.0
        transfer = np.fft.rfft2(grid)
        self._kernel_fft[size] = transfer
        return transfer

    def _convolve(self, field: np.ndarray) -> np.ndarray:
        """Disk sum of ``field`` over the footprint centred on each cell."""

        g = field.shape[0]
        pad = int(self._kernel.shape[0] // 2) + 2
        size = g + 2 * pad
        padded = np.zeros((size, size), dtype=np.float64)
        padded[pad:pad + g, pad:pad + g] = field
        spectrum = np.fft.rfft2(padded) * self._kernel_transfer(size)
        return np.fft.irfft2(spectrum, s=(size, size))[pad:pad + g, pad:pad + g]

    def _build_polyline(
        self, env: Any, u: int, lane_offset: float = 0.0
    ) -> np.ndarray:
        """Boustrophedon over UAV ``u``'s vertical sector."""

        g = env.grid_size
        margin = int(self.cfg.sector_inner_margin)
        width = g / float(env.n_agents)
        left = int(np.floor(u * width))
        right = min(int(np.floor((u + 1) * width)) - 1, g - 1 - margin)
        radius = env.sensing_cfg.radius

        lanes: List[int] = []
        x = float(left + max(int(np.ceil(radius)) - 1, 0) + lane_offset)
        limit = min(right + radius - 1, g - 1 - margin)
        while x <= limit + 1e-9:
            lanes.append(int(round(min(max(x, margin), g - 1 - margin))))
            x += self.cfg.lane_spacing
        if not lanes:
            lanes = [int(np.clip(left + radius, margin, g - 1 - margin))]
        if lanes[-1] < min(right, g - 1 - margin):
            lanes.append(int(np.clip(right, margin, g - 1 - margin)))

        low, high = float(margin), float(g - 1 - margin)
        corners: List[Tuple[float, float]] = []
        for index, lane in enumerate(dict.fromkeys(lanes)):
            if index % 2 == 0:
                corners.extend([(float(lane), low), (float(lane), high)])
            else:
                corners.extend([(float(lane), high), (float(lane), low)])

        dense: List[Tuple[float, float]] = []
        for a, b in zip(corners[:-1], corners[1:]):
            steps = max(int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]))), 1)
            for k in range(steps):
                t = k / steps
                dense.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        dense.append(corners[-1])
        return np.asarray(dense, dtype=np.float64)

    def act(self, env: Any, observation: Any) -> ActionOutput:
        if self._goals is None or self._grid_size != env.grid_size:
            self.reset(env)

        n = env.n_agents
        actions = np.zeros(n, dtype=np.int64)
        options = np.zeros(n, dtype=np.int64)
        annotations: List[Dict[str, Any]] = []
        claimed: List[np.ndarray] = []

        for u in range(n):
            maps = self._agent_maps(env, u)
            option, goal, score = self._select_goal(env, u, maps, claimed)
            options[u] = option
            self._goals[u] = goal
            claimed.append(np.asarray(goal, dtype=np.float64))
            actions[u] = self._steer(env, u, goal)
            annotations.append({
                "goal": tuple(int(v) for v in goal),
                "option": int(option),
                "option_name": OPTION_NAMES[int(option) % len(OPTION_NAMES)],
                "goal_score": float(score),
            })

        self._option_age += 1
        return ActionOutput(actions=actions, options=options, annotations=annotations)

    def _agent_maps(self, env: Any, u: int) -> Dict[str, np.ndarray]:
        g = env.grid_size
        visited = env.agent_visited[u]
        unvisited = (~visited).astype(np.float64)
        prob = np.clip(env.belief(u), 1e-9, 1 - 1e-9)

        band = (prob > 0.35) & (prob < 0.995) & ~env._confirmed_mask
        need = np.where(band, 0.4 + prob, 0.0)
        last = env.agent_visit_step[u].astype(np.float64)
        age = np.maximum(float(env.step_index) - last, 0.0)

        margin = int(self.cfg.interior_margin)
        valid = np.zeros((g, g), dtype=bool)
        valid[margin:g - margin, margin:g - margin] = True

        return {
            "frontier": self._convolve(unvisited),
            "uncertainty": self._convolve(binary_entropy(prob)),
            "confirm": self._convolve(need),
            "recency": self._convolve(np.where(visited, 1.0 / (1.0 + age), 0.0)),
            "valid": valid,
            "prob": prob,
        }

    def _sweep_goal(self, env: Any, u: int) -> Tuple[np.ndarray, bool]:
        polyline = self._polylines[u]
        position = np.asarray(env.positions[u], dtype=np.float64)
        cursor = int(self._cursors[u])

        distances = np.hypot(polyline[:, 0] - position[0], polyline[:, 1] - position[1])
        start = max(cursor - 4, 0)
        index = int(np.argmin(distances[start:])) + start
        if distances[index] <= self.cfg.sweep_track_radius:
            cursor = max(cursor, index + 1)

        if cursor >= len(polyline) - 1 and self._passes[u] + 1 < self.cfg.sweep_passes:
            self._passes[u] += 1
            self._polylines[u] = self._build_polyline(
                env, u, lane_offset=0.5 * self.cfg.lane_spacing * self._passes[u]
            )
            polyline = self._polylines[u]
            cursor = 0

        target = min(cursor + self.cfg.sweep_lookahead, len(polyline) - 1)
        self._cursors[u] = cursor
        done = (
            cursor >= len(polyline) - 1
            and self._passes[u] + 1 >= self.cfg.sweep_passes
        )
        goal = np.clip(
            np.rint(polyline[target]).astype(np.int64), 0, env.grid_size - 1
        )
        return goal, done

    def _best_goal(
        self,
        env: Any,
        u: int,
        maps: Dict[str, np.ndarray],
        claimed: List[np.ndarray],
        terms: Dict[str, float],
    ) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        g = env.grid_size
        position = np.asarray(env.positions[u], dtype=np.float64)
        yy, xx = np.mgrid[0:g, 0:g]
        distance = np.hypot(xx - position[0], yy - position[1])

        score = np.zeros((g, g), dtype=np.float64)
        for name, weight in terms.items():
            score += weight * maps[name]
        score -= cfg.weight_recency * maps["recency"]
        score -= cfg.weight_travel * np.maximum(
            distance - cfg.goal_reach_horizon, 0.0
        )
        for other in claimed:
            separation = np.hypot(xx - other[0], yy - other[1])
            score -= cfg.weight_separation * np.exp(
                -(separation ** 2) / (2.0 * cfg.separation_sigma ** 2)
            )
        for other_index in range(env.n_agents):
            if other_index == u:
                continue
            other_pos = np.asarray(env.positions[other_index], dtype=np.float64)
            separation = np.hypot(xx - other_pos[0], yy - other_pos[1])
            score -= cfg.weight_keep_away * np.exp(
                -(separation ** 2) / (2.0 * cfg.keep_away_sigma ** 2)
            )
        score[~maps["valid"]] = -np.inf

        best = np.unravel_index(int(np.argmax(score)), score.shape)
        return np.array([best[1], best[0]], dtype=np.int64), float(score[best])

    def _select_goal(
        self,
        env: Any,
        u: int,
        maps: Dict[str, np.ndarray],
        claimed: List[np.ndarray],
    ) -> Tuple[int, np.ndarray, float]:
        cfg = self.cfg
        confirm_max = float(maps["confirm"].max())
        sweep_goal, sweep_done = self._sweep_goal(env, u)
        distance = float(np.linalg.norm(sweep_goal - np.asarray(env.positions[u])))
        previous = int(self._options[u])
        dwell = int(self._option_age[u])

        if confirm_max >= cfg.confirm_trigger and (
            (previous == 2 and dwell < cfg.confirm_dwell)
            or distance <= cfg.divert_radius
            or sweep_done
        ):
            goal, score = self._best_goal(
                env, u, maps, claimed,
                {"confirm": cfg.weight_confirm, "frontier": 0.12 * cfg.weight_frontier},
            )
            return 2, goal, score

        if not sweep_done:
            return 0, sweep_goal, float(
                maps["frontier"][sweep_goal[1], sweep_goal[0]]
            )

        if float(maps["uncertainty"].max()) >= cfg.inspect_trigger:
            goal, score = self._best_goal(
                env, u, maps, claimed,
                {
                    "uncertainty": cfg.weight_uncertainty,
                    "confirm": 0.5 * cfg.weight_confirm,
                    "frontier": 0.30 * cfg.weight_frontier,
                },
            )
            return 1, goal, score

        goal, score = self._best_goal(
            env, u, maps, claimed,
            {"frontier": cfg.weight_frontier, "uncertainty": cfg.weight_uncertainty},
        )
        return 3, goal, score

    def _steer(self, env: Any, u: int, goal: np.ndarray) -> int:
        position = np.asarray(env.positions[u], dtype=np.float64)
        delta = np.asarray(goal, dtype=np.float64) - position
        turns = np.asarray(env.uav_cfg.turn_actions_deg, dtype=np.float64)
        straight = int(np.argmin(np.abs(turns)))
        current = float(env.headings[u])

        if np.linalg.norm(delta) < 1e-6:
            return straight

        desired = float(np.arctan2(delta[1], delta[0]))
        residuals = np.array([
            abs(angle_difference(desired, wrap_angle(current + np.deg2rad(t))))
            for t in turns
        ])
        choice = straight if residuals[straight] <= np.deg2rad(12.0) else int(
            np.argmin(residuals)
        )

        next_heading = wrap_angle(current + np.deg2rad(turns[choice]))
        ahead = position + np.array([np.cos(next_heading), np.sin(next_heading)])
        limit = env.sensing_cfg.radius - 1.0
        if (ahead[0] < limit or ahead[0] > env.grid_size - 1 - limit
                or ahead[1] < limit or ahead[1] > env.grid_size - 1 - limit):
            inward = np.array([env.grid_size / 2.0, env.grid_size / 2.0]) - position
            if np.linalg.norm(inward) > 1e-9:
                desired = float(np.arctan2(inward[1], inward[0]))
                residuals = np.array([
                    abs(angle_difference(desired, wrap_angle(current + np.deg2rad(t))))
                    for t in turns
                ])
                choice = int(np.argmin(residuals))
        return choice


register_policy("rule")(RuleBasedPolicy)
