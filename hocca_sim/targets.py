"""Static and mobile target process."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .beliefs import EIGHT_NEIGHBOURS
from .config import (
    REFERENCE_MOVING_TARGETS, REFERENCE_STATIC_TARGETS, GridConfig, TargetConfig,
)

__all__ = ["TargetField", "sample_target_layout", "reference_target_layout"]


def reference_target_layout() -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(REFERENCE_STATIC_TARGETS, dtype=np.int64),
        np.asarray(REFERENCE_MOVING_TARGETS, dtype=np.int64),
    )


def sample_target_layout(
    grid: GridConfig, cfg: TargetConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform-without-replacement placement of the targets."""

    if cfg.layout == "reference" or cfg.reference_override:
        static, moving = reference_target_layout()
        if cfg.count <= len(static):
            return static[: cfg.count], np.zeros((0, 2), dtype=np.int64)
        n_moving = min(cfg.moving_count, len(moving))
        return static[: max(cfg.count - n_moving, 0)], moving[:n_moving]

    if cfg.count > grid.cells:
        raise ValueError("more targets than cells")
    flat = rng.choice(grid.cells, size=cfg.count, replace=False)
    cells = np.column_stack([flat % grid.size, flat // grid.size]).astype(np.int64)
    return cells[cfg.moving_count:], cells[: cfg.moving_count]


class TargetField:
    """Target state with the hop clock ``T_m`` and mobility ``p_m``."""

    def __init__(
        self,
        grid: GridConfig,
        cfg: TargetConfig,
        static_cells: np.ndarray,
        moving_cells: np.ndarray,
    ) -> None:
        self.grid = grid
        self.cfg = cfg
        static_cells = np.asarray(static_cells, dtype=np.int64).reshape(-1, 2)
        moving_cells = np.asarray(moving_cells, dtype=np.int64).reshape(-1, 2)
        self.positions = np.vstack([static_cells, moving_cells]).astype(np.int64)
        self.is_moving = np.concatenate([
            np.zeros(len(static_cells), dtype=bool),
            np.ones(len(moving_cells), dtype=bool),
        ])
        self.last_hop_step = np.zeros(len(self.positions), dtype=np.int64)
        self.hop_count = 0

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def moving_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_moving)

    def occupancy(self) -> np.ndarray:
        occ = np.zeros(self.grid.shape, dtype=bool)
        if len(self.positions):
            occ[self.positions[:, 1], self.positions[:, 0]] = True
        return occ

    def multiplicity(self) -> np.ndarray:
        counts = np.zeros(self.grid.shape, dtype=np.int16)
        for x, y in self.positions:
            counts[y, x] += 1
        return counts

    def cell_set(self) -> set:
        return {(int(x), int(y)) for x, y in self.positions}

    def copy(self) -> "TargetField":
        other = TargetField.__new__(TargetField)
        other.grid = self.grid
        other.cfg = self.cfg
        other.positions = self.positions.copy()
        other.is_moving = self.is_moving.copy()
        other.last_hop_step = self.last_hop_step.copy()
        other.hop_count = self.hop_count
        return other

    def step(self, step_index: int, rng: np.random.Generator) -> bool:
        """Advance the hop process by one step; returns ``True`` on any hop."""

        if self.cfg.hop_period <= 0 or step_index <= 0:
            return False
        if step_index % self.cfg.hop_period != 0:
            return False
        moving = self.moving_indices
        if len(moving) == 0:
            return False
        hopped = False
        for k in moving:
            if rng.random() >= self.cfg.move_prob:
                continue
            x, y = self.positions[k]
            candidates = [
                (int(x) + dx, int(y) + dy)
                for dy, dx in EIGHT_NEIGHBOURS
                if 0 <= int(x) + dx < self.grid.size and 0 <= int(y) + dy < self.grid.size
            ]
            if not candidates:
                continue
            self.positions[k] = candidates[int(rng.integers(len(candidates)))]
            hopped = True
        if hopped:
            self.last_hop_step[moving] = step_index
            self.hop_count += 1
        return hopped

    def hop_due(self, step_index: int) -> bool:
        period = self.cfg.hop_period
        return period > 0 and step_index > 0 and step_index % period == 0
