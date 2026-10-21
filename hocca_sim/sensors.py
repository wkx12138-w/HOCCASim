"""Disk footprint and Bernoulli measurement model."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "disk_offsets", "covered_cells", "footprint_mask", "sample_measurement",
    "innovation_of", "SensingModel",
]

_CACHE: Dict[Tuple[float, int], np.ndarray] = {}


def disk_offsets(radius: float, grid_size: int = 0) -> np.ndarray:
    """Integer offsets whose Euclidean norm is within ``radius``."""

    key = (float(radius), int(grid_size))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    reach = int(np.floor(radius))
    dy, dx = np.mgrid[-reach: reach + 1, -reach: reach + 1]
    keep = (dx * dx + dy * dy) <= radius * radius + 1e-12
    offsets = np.column_stack([dx[keep], dy[keep]]).astype(np.int64)
    offsets = offsets[np.lexsort((offsets[:, 0], offsets[:, 1]))]
    _CACHE[key] = offsets
    return offsets


def covered_cells(pose: np.ndarray, radius: float, grid_size: int) -> np.ndarray:
    offsets = disk_offsets(radius, grid_size)
    cells = np.rint(np.asarray(pose, dtype=np.float64)[None, :] + offsets).astype(np.int64)
    inside = (
        (cells[:, 0] >= 0) & (cells[:, 0] < grid_size)
        & (cells[:, 1] >= 0) & (cells[:, 1] < grid_size)
    )
    return cells[inside]


def footprint_mask(pose: np.ndarray, radius: float, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    cells = covered_cells(pose, radius, grid_size)
    mask[cells[:, 1], cells[:, 0]] = True
    return mask


def innovation_of(z, p_detect: float, p_false_alarm: float):
    z = np.asarray(z)
    return np.where(
        z > 0,
        np.log(p_detect / p_false_alarm),
        np.log((1.0 - p_detect) / (1.0 - p_false_alarm)),
    )


def sample_measurement(
    rng: np.random.Generator,
    cells: np.ndarray,
    occupancy: np.ndarray,
    p_detect: float,
    p_false_alarm: float,
) -> np.ndarray:
    if len(cells) == 0:
        return np.zeros(0, dtype=np.int8)
    occupied = occupancy[cells[:, 1], cells[:, 0]]
    p = np.where(occupied, p_detect, p_false_alarm)
    return (rng.random(len(cells)) < p).astype(np.int8)


class SensingModel:
    """Cached footprint geometry plus a vectorised sampler."""

    def __init__(
        self, grid_size: int, radius: float, p_detect: float, p_false_alarm: float
    ) -> None:
        self.grid_size = int(grid_size)
        self.radius = float(radius)
        self.p_detect = float(p_detect)
        self.p_false_alarm = float(p_false_alarm)
        self.offsets = disk_offsets(radius, self.grid_size)
        self.log_odds_hit = float(np.log(self.p_detect / self.p_false_alarm))
        self.log_odds_miss = float(
            np.log((1.0 - self.p_detect) / (1.0 - self.p_false_alarm))
        )

    def footprint_cells(self, pose: np.ndarray) -> np.ndarray:
        return covered_cells(pose, self.radius, self.grid_size)

    def footprint_mask(self, pose: np.ndarray) -> np.ndarray:
        return footprint_mask(pose, self.radius, self.grid_size)

    def footprint_size(self, pose: np.ndarray) -> int:
        return int(len(self.footprint_cells(pose)))

    def sample(
        self, rng: np.random.Generator, cells: np.ndarray, occupancy: np.ndarray
    ) -> np.ndarray:
        return sample_measurement(
            rng, cells, occupancy, self.p_detect, self.p_false_alarm
        )

    def innovations(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, self.log_odds_hit, self.log_odds_miss).astype(np.float64)

    def expected_looks_to_confirm(self, threshold: float, prior_log_odds: float) -> float:
        kl = self.p_detect * np.log(self.p_detect / self.p_false_alarm) + (
            1.0 - self.p_detect
        ) * np.log((1.0 - self.p_detect) / (1.0 - self.p_false_alarm))
        return float((threshold - prior_log_odds) / kl)
