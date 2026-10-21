"""Unit-speed motion on a quantised heading lattice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import GridConfig, UAVConfig

__all__ = [
    "wrap_angle", "quantise_angle", "angle_difference", "turn_to",
    "integrate_motion", "boundary_violation", "EnergyModel", "FleetState",
]


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quantise_angle(angle: float, bins: int = 12) -> float:
    step = 2.0 * np.pi / bins
    return float(np.round(angle / step) * step)


def angle_difference(target: float, current: float) -> float:
    return wrap_angle(target - current)


def turn_to(current: float, delta_deg: float, bins: int = 12) -> float:
    return quantise_angle(current + np.deg2rad(delta_deg), bins)


def boundary_violation(position: np.ndarray, size: int) -> bool:
    return bool(
        position[0] < 0.0 or position[1] < 0.0
        or position[0] > size - 1 or position[1] > size - 1
    )


def integrate_motion(
    position: np.ndarray, heading: float, speed: float, grid: GridConfig
) -> Tuple[np.ndarray, bool]:
    """Advance one step and apply the boundary rule."""

    size = grid.size
    proposed = position + speed * np.array([np.cos(heading), np.sin(heading)])
    if not boundary_violation(proposed, size):
        return proposed, False

    if grid.boundary == "wrap":
        return np.mod(proposed, size), True

    if grid.boundary == "reflect":
        reflected = proposed.copy()
        if reflected[0] < 0.0:
            reflected[0] = -reflected[0]
        elif reflected[0] > size - 1:
            reflected[0] = 2.0 * (size - 1) - reflected[0]
        if reflected[1] < 0.0:
            reflected[1] = -reflected[1]
        elif reflected[1] > size - 1:
            reflected[1] = 2.0 * (size - 1) - reflected[1]
        return np.clip(reflected, 0.0, size - 1), True

    return np.clip(proposed, 0.0, float(size - 1)), True


class EnergyModel:
    """Normalised remaining-energy bookkeeping."""

    def __init__(self, cfg: UAVConfig, count: int) -> None:
        self.cfg = cfg
        self.initial = np.full(count, float(cfg.energy_initial), dtype=np.float64)
        self.remaining = self.initial.copy()

    @property
    def enabled(self) -> bool:
        return self.cfg.energy_per_step > 0.0

    def reset(self, count: int) -> None:
        self.initial = np.full(count, float(self.cfg.energy_initial), dtype=np.float64)
        self.remaining = self.initial.copy()

    def consume(self, indices: np.ndarray, distances: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros(len(indices), dtype=np.float64)
        spent = distances * float(self.cfg.energy_per_step)
        self.remaining[indices] = np.maximum(self.remaining[indices] - spent, 0.0)
        return spent

    def active_mask(self) -> np.ndarray:
        if not self.enabled:
            return np.ones_like(self.remaining, dtype=bool)
        return self.remaining > 0.0

    @property
    def normalised(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = self.remaining / np.maximum(self.initial, 1e-12)
        return np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)


@dataclass
class FleetState:
    positions: np.ndarray
    headings: np.ndarray
    energy: np.ndarray
    active: np.ndarray

    def __len__(self) -> int:
        return len(self.positions)

    def copy(self) -> "FleetState":
        return FleetState(
            self.positions.copy(), self.headings.copy(),
            self.energy.copy(), self.active.copy(),
        )

    @property
    def headings_deg(self) -> np.ndarray:
        return np.rad2deg(self.headings)
