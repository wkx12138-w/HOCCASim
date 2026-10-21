"""Evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "StepRecord", "EpisodeMetrics", "coverage_ratio", "threshold_crossing",
    "confusion_counts", "aggregate_metrics", "mean_std",
]


@dataclass
class StepRecord:
    """One row of a rollout trace."""

    step: int
    coverage: float
    recall: float
    confirmed: int
    confirmed_cells: int
    false_cells: int
    entropy: float
    packets_sent: int
    items_attempted: int
    moves: int
    reward_team: float


@dataclass
class EpisodeMetrics:
    """Episode-level summary."""

    coverage: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    confirmed: int = 0
    target_count: int = 0
    t80_coverage: Optional[int] = None
    t80_detection: Optional[int] = None
    t_all: Optional[int] = None
    success: bool = False
    steps: int = 0
    moves: int = 0
    packets: int = 0
    items: int = 0
    messages_per_uav_step: float = 0.0
    return_sum: float = 0.0
    trace: List[StepRecord] = field(default_factory=list)
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, float]:
        return {
            "coverage": self.coverage,
            "recall": self.recall,
            "precision": self.precision,
            "confirmed": float(self.confirmed),
            "t80_cov": float(self.t80_coverage or np.nan),
            "t80_det": float(self.t80_detection or np.nan),
            "t_all": float(self.t_all or np.nan),
            "success": float(self.success),
            "steps": float(self.steps),
            "moves": float(self.moves),
            "packets": float(self.packets),
            "items": float(self.items),
            "messages_per_uav_step": self.messages_per_uav_step,
            "return": self.return_sum,
        }


def coverage_ratio(visited: np.ndarray) -> float:
    return float(visited.mean())


def threshold_crossing(trace: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(trace):
        if value >= threshold:
            return int(index)
    return None


def confusion_counts(confirmed_cells: set, target_cells: set) -> tuple:
    tp = len(confirmed_cells & target_cells)
    fp = len(confirmed_cells - target_cells)
    fn = len(target_cells - confirmed_cells)
    return tp, fp, fn


def aggregate_metrics(
    episodes: Sequence[EpisodeMetrics], threshold: float = 0.8
) -> Dict[str, float]:
    if not episodes:
        return {}

    def stack(attr: str) -> np.ndarray:
        values = []
        for ep in episodes:
            value = getattr(ep, attr)
            values.append(np.nan if value is None else float(value))
        return np.asarray(values, dtype=np.float64)

    out: Dict[str, float] = {}
    for name in ("coverage", "recall", "precision", "moves", "packets", "items",
                 "messages_per_uav_step", "return_sum", "steps"):
        values = stack(name)
        out["%s_mean" % name] = float(np.nanmean(values))
        out["%s_std" % name] = float(np.nanstd(values))

    for name in ("t80_coverage", "t80_detection", "t_all"):
        values = stack(name)
        out["%s_mean" % name] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        out["%s_std" % name] = float(np.nanstd(values)) if np.isfinite(values).any() else float("nan")

    out["success_rate"] = float(np.mean([1.0 if ep.success else 0.0 for ep in episodes]))
    out["episodes"] = float(len(episodes))
    return out


def mean_std(values: Sequence[float]) -> tuple:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(arr.mean()), float(arr.std()))
