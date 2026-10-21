"""Log-odds occupancy belief and information measures."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "binary_entropy", "entropy_map", "logit", "sigmoid", "mutual_information",
    "expected_looks_to_confirm", "kl_bernoulli_after_item", "clip_probability",
    "EIGHT_NEIGHBOURS", "OccupancyBelief", "IndexedBelief",
]

EIGHT_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def clip_probability(p, eps: float = 1e-9):
    return np.clip(p, eps, 1.0 - eps)


def logit(p, eps: float = 1e-9):
    p = clip_probability(p, eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out if out.ndim else float(out)


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p) + q * np.log(q))
    return np.nan_to_num(h, nan=0.0, posinf=0.0)


def entropy_map(belief) -> float:
    return float(np.sum(binary_entropy(belief)))


def mutual_information(b, p_detect: float, p_false_alarm: float):
    b = np.asarray(b, dtype=np.float64)
    p_d, p_fa = float(p_detect), float(p_false_alarm)
    p1 = p_fa + (p_d - p_fa) * b
    mi = binary_entropy(p1) - b * binary_entropy(p_d) - (1.0 - b) * binary_entropy(p_fa)
    return np.maximum(mi, 0.0)


def expected_looks_to_confirm(
    threshold: float, prior_log_odds: float, p_detect: float, p_false_alarm: float
) -> float:
    p, q = float(p_detect), float(p_false_alarm)
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float((threshold - prior_log_odds) / kl)


def kl_bernoulli_after_item(receiver_prob, delta_log_odds, eps: float = 1e-9):
    q = clip_probability(np.asarray(receiver_prob, dtype=np.float64), eps)
    after = clip_probability(sigmoid(logit(q, eps) + delta_log_odds), eps)
    kl = after * (np.log(after) - np.log(q)) + (1.0 - after) * (
        np.log1p(-after) - np.log1p(-q)
    )
    return np.maximum(kl, 0.0)


def _valid_neighbour_counts(grid_size: int) -> np.ndarray:
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for dy, dx in EIGHT_NEIGHBOURS:
        counts[
            slice(max(0, -dy), grid_size - max(0, dy)),
            slice(max(0, -dx), grid_size - max(0, dx)),
        ] += 1.0
    return counts


def _scatter_hop(marginal: np.ndarray, move_probability: float) -> np.ndarray:
    grid_size = marginal.shape[0]
    weights = marginal / np.maximum(_valid_neighbour_counts(grid_size), 1.0)
    out = (1.0 - move_probability) * marginal
    for dy, dx in EIGHT_NEIGHBOURS:
        ys_src = slice(max(0, -dy), min(grid_size, grid_size - dy))
        xs_src = slice(max(0, -dx), min(grid_size, grid_size - dx))
        ys_dst = slice(max(0, dy), min(grid_size, grid_size + dy))
        xs_dst = slice(max(0, dx), min(grid_size, grid_size + dx))
        out[ys_dst, xs_dst] += move_probability * weights[ys_src, xs_src]
    return np.clip(out, 0.0, 1.0)


class OccupancyBelief:
    """Factorised cell-occupancy belief stored in log-odds."""

    def __init__(
        self,
        grid_size: int,
        prior_probability: float,
        clip: float = 1e-9,
        log_odds_clip: float = 1e6,
    ) -> None:
        self.grid_size = int(grid_size)
        self.prior_probability = float(prior_probability)
        self.clip = float(clip)
        self.log_odds_clip = float(log_odds_clip)
        self.ell = np.full(
            (self.grid_size, self.grid_size),
            float(logit(self.prior_probability, self.clip)),
            dtype=np.float64,
        )
        self.hits = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.looks = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)

    def copy(self) -> "OccupancyBelief":
        other = OccupancyBelief(
            self.grid_size, self.prior_probability, self.clip, self.log_odds_clip
        )
        other.ell = self.ell.copy()
        other.hits = self.hits.copy()
        other.looks = self.looks.copy()
        return other

    def probabilities(self) -> np.ndarray:
        return sigmoid(np.clip(self.ell, -self.log_odds_clip, self.log_odds_clip))

    def prob_at(self, x: int, y: int) -> float:
        return float(sigmoid(self.ell[y, x]))

    def entropy(self) -> float:
        return entropy_map(self.probabilities())

    def mass(self) -> float:
        return float(self.probabilities().sum())

    def active_mask(self, threshold: float) -> np.ndarray:
        return self.ell >= threshold

    def reset(self, prior_probability: Optional[float] = None) -> None:
        if prior_probability is not None:
            self.prior_probability = float(prior_probability)
        self.ell.fill(float(logit(self.prior_probability, self.clip)))
        self.hits.fill(0)
        self.looks.fill(0)

    def apply_innovations(self, cells: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        if len(cells) == 0:
            return np.zeros(0, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.int64)
        deltas = np.asarray(deltas, dtype=np.float64)
        size = self.grid_size ** 2
        flat = cells[:, 1] * self.grid_size + cells[:, 0]
        add = np.bincount(flat, weights=deltas, minlength=size).reshape(self.ell.shape)
        self.ell += add
        np.clip(self.ell, -self.log_odds_clip, self.log_odds_clip, out=self.ell)
        self.looks += np.bincount(flat, minlength=size).reshape(self.ell.shape).astype(
            np.int16
        )
        self.hits += np.bincount(
            flat, weights=(deltas > 0).astype(np.float64), minlength=size
        ).reshape(self.ell.shape).astype(np.int16)
        return add

    def apply_mask(self, mask: np.ndarray, delta: float) -> None:
        self.ell[mask] += float(delta)
        np.clip(self.ell, -self.log_odds_clip, self.log_odds_clip, out=self.ell)

    def predict_hop(self, move_probability: float) -> None:
        if float(move_probability) <= 0.0:
            return
        self.ell = logit(_scatter_hop(self.probabilities(), float(move_probability)),
                         self.clip)

    def normalise_mass(self, target_mass: float) -> None:
        b = self.probabilities()
        total = float(b.sum())
        if total <= 0.0:
            return
        scale = float(target_mass) / total
        self.ell = logit(np.clip(b * scale, 0.0, 1.0), self.clip)


class IndexedBelief:
    """Per-target occupancy marginals combined by inclusion-exclusion."""

    def __init__(
        self,
        grid_size: int,
        target_count: int,
        prior_probability: float,
        clip: float = 1e-9,
    ) -> None:
        self.grid_size = int(grid_size)
        self.target_count = int(target_count)
        self.prior_probability = float(prior_probability)
        self.clip = float(clip)
        self.rho = np.full(
            (self.target_count, self.grid_size, self.grid_size),
            self.prior_probability,
            dtype=np.float64,
        )

    def probabilities(self) -> np.ndarray:
        return 1.0 - np.prod(1.0 - np.clip(self.rho, 0.0, 1.0), axis=0)

    def entropy(self) -> float:
        return entropy_map(self.probabilities())

    def predict_hop(self, move_probability: float) -> None:
        if float(move_probability) <= 0.0:
            return
        for k in range(self.target_count):
            self.rho[k] = _scatter_hop(self.rho[k], float(move_probability))
