"""Configuration for the HOCCA reference simulator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GridConfig", "SensingConfig", "UAVConfig", "TargetConfig",
    "CommunicationConfig", "BeliefConfig", "RewardConfig", "EpisodeConfig",
    "ObservationConfig", "SimConfig", "REFERENCE_STATIC_TARGETS",
    "REFERENCE_MOVING_TARGETS", "REFERENCE_UAV_POSES", "REFERENCE_CORRIDORS",
    "default_launch_positions",
]

REFERENCE_STATIC_TARGETS: Tuple[Tuple[int, int], ...] = (
    (7, 9), (11, 21), (13, 42), (18, 13), (20, 35), (24, 29),
    (29, 17), (31, 24), (34, 37), (39, 11), (41, 30), (45, 45),
)
REFERENCE_MOVING_TARGETS: Tuple[Tuple[int, int], ...] = (
    (6, 24), (22, 44), (40, 20), (44, 38),
)
REFERENCE_UAV_POSES: Tuple[Tuple[float, float], ...] = (
    (4.0, 4.0), (44.0, 4.0), (4.0, 44.0), (44.0, 44.0),
)
REFERENCE_CORRIDORS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
    ((6, 24), (40, 20)),
    ((22, 44), (44, 38)),
)


def _as_tuple(obj: Any) -> None:
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, list):
            setattr(obj, f.name, tuple(value))


@dataclass
class GridConfig:
    """Search area."""

    size: int = 48
    boundary: str = "clamp"
    allow_target_overlap: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("grid size must be positive")
        if self.boundary not in ("clamp", "reflect", "wrap"):
            raise ValueError("boundary must be 'clamp', 'reflect' or 'wrap'")

    @property
    def cells(self) -> int:
        return self.size * self.size

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.size, self.size)


@dataclass
class SensingConfig:
    """Euclidean disk footprint with a binary measurement per cell."""

    radius: float = 3.0
    p_detect: float = 0.90
    p_false_alarm: float = 0.02

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("sensor radius must be positive")
        if not 0.0 < self.p_detect <= 1.0:
            raise ValueError("p_detect must lie in (0, 1]")
        if not 0.0 <= self.p_false_alarm < 1.0:
            raise ValueError("p_false_alarm must lie in [0, 1)")
        if self.p_detect == self.p_false_alarm:
            raise ValueError("p_D == p_FA gives no information")

    @property
    def kl_detect_vs_alarm(self) -> float:
        p, q = self.p_detect, self.p_false_alarm
        return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

    @property
    def log_odds_hit(self) -> float:
        return float(np.log(self.p_detect / self.p_false_alarm))

    @property
    def log_odds_miss(self) -> float:
        return float(np.log((1 - self.p_detect) / (1 - self.p_false_alarm)))


@dataclass
class UAVConfig:
    """Unit-speed agents on a quantised heading lattice."""

    count: int = 4
    speed: float = 1.0
    turn_actions_deg: Tuple[float, ...] = (-30.0, 0.0, 30.0)
    heading_bins: int = 12
    initial_positions: Optional[Sequence[Sequence[float]]] = None
    initial_headings_deg: Optional[Sequence[float]] = None
    heading_strategy: str = "centripetal"
    energy_initial: float = 1.0
    energy_per_step: float = 0.0
    min_separation: float = 0.0
    allow_hover: bool = False

    def __post_init__(self) -> None:
        _as_tuple(self)
        if self.count < 1:
            raise ValueError("fleet size must be >= 1")
        if self.heading_bins < 1:
            raise ValueError("heading_bins must be >= 1")
        if not self.turn_actions_deg:
            raise ValueError("turn_actions_deg must not be empty")
        step = 360.0 / self.heading_bins
        for angle in self.turn_actions_deg:
            if abs(round(angle / step) * step - angle) > 1e-9:
                raise ValueError(
                    "turn action %r is not a multiple of %.3f degrees" % (angle, step)
                )
        if self.initial_positions is not None:
            if len(self.initial_positions) != self.count:
                raise ValueError("initial_positions must contain %d poses" % self.count)
            self.initial_positions = tuple(
                (float(p[0]), float(p[1])) for p in self.initial_positions
            )
        if self.initial_headings_deg is not None:
            if len(self.initial_headings_deg) != self.count:
                raise ValueError(
                    "initial_headings_deg must contain %d values" % self.count
                )
            self.initial_headings_deg = tuple(
                float(h) for h in self.initial_headings_deg
            )

    @property
    def action_count(self) -> int:
        return len(self.turn_actions_deg)

    @property
    def heading_step_deg(self) -> float:
        return 360.0 / self.heading_bins


@dataclass
class TargetConfig:
    """Static and mobile targets."""

    count: int = 16
    moving_count: int = 4
    hop_period: int = 20
    move_prob: float = 0.25
    layout: Optional[str] = "random"
    reference_override: bool = False

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("target count must be >= 1")
        if not 0 <= self.moving_count <= self.count:
            raise ValueError("moving_count must lie in [0, count]")
        if self.hop_period < 1:
            raise ValueError("hop_period must be >= 1")
        if not 0.0 <= self.move_prob <= 1.0:
            raise ValueError("move_prob must lie in [0, 1]")
        if self.layout not in ("random", "reference") and not self.reference_override:
            raise ValueError("layout must be 'random' or 'reference'")

    @property
    def static_count(self) -> int:
        return self.count - self.moving_count


@dataclass
class CommunicationConfig:
    """Budgeted evidence exchange."""

    packets_per_step: int = 2
    items_per_packet: int = 8
    positive_net_score_only: bool = True
    item_cost: float = 1.0
    bit_cost_lambda: float = 0.05
    replay_window: int = 8
    allow_self_loop: bool = False
    packet_loss: float = 0.0
    delay_steps: int = 0
    outage_period: int = 0
    outage_length: int = 0
    bandwidth_schedule: Optional[Sequence[int]] = None
    header_bytes: int = 8
    item_bytes: int = 16
    ack_id_bytes: int = 8
    ack_header_bytes: int = 8

    def __post_init__(self) -> None:
        _as_tuple(self)
        if self.packets_per_step < 0:
            raise ValueError("packets_per_step must be >= 0")
        if self.items_per_packet < 1:
            raise ValueError("items_per_packet must be >= 1")
        if not 0.0 <= self.packet_loss <= 1.0:
            raise ValueError("packet_loss must lie in [0, 1]")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        if self.replay_window < 0:
            raise ValueError("replay_window must be >= 0")
        if self.bandwidth_schedule is not None:
            self.bandwidth_schedule = tuple(int(b) for b in self.bandwidth_schedule)

    @property
    def item_budget(self) -> int:
        return self.packets_per_step * self.items_per_packet

    @property
    def packet_bytes(self) -> int:
        return self.header_bytes + self.items_per_packet * self.item_bytes

    def acknowledgement_bytes(self, item_count: int) -> int:
        return self.ack_header_bytes + int(item_count) * self.ack_id_bytes

    def budget_at(self, step: int) -> int:
        if self.bandwidth_schedule:
            return int(self.bandwidth_schedule[step % len(self.bandwidth_schedule)])
        return self.packets_per_step

    def link_available(self, step: int) -> bool:
        if self.outage_period <= 0 or self.outage_length <= 0:
            return True
        return (step % self.outage_period) >= self.outage_length


@dataclass
class BeliefConfig:
    """Occupancy belief."""

    mode: str = "map"
    confirm_log_odds: float = 5.0
    clip: float = 1e-9
    log_odds_clip: float = 1e6
    predict_every_hop: bool = True
    mass_normalisation: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("map", "indexed"):
            raise ValueError("belief mode must be 'map' or 'indexed'")

    @staticmethod
    def prior_occupancy(target_count: int, cell_count: int) -> float:
        return float(target_count) / float(cell_count)


@dataclass
class RewardConfig:
    """Reward coefficients and credit assignment."""

    lambda_H: float = 1.0
    lambda_D: float = 25.0
    lambda_R: float = 1.5
    lambda_C: float = 0.3
    lambda_E: float = 0.02
    lambda_F: float = 0.2
    lambda_A: float = 0.5
    coverage_coef: float = 0.09
    revisit_normalisation: str = "fraction"
    communication_normalisation: str = "budget_fraction"
    revisit_window: int = 12
    boundary_penalty: float = 0.05
    collision_penalty: float = 0.5
    spatial_balance_mode: str = "quadratic"
    gamma: float = 0.99
    reward_clip: Optional[float] = None
    differential_credit: bool = True
    credit_order: str = "random"
    potential_belief: str = "team"

    def __post_init__(self) -> None:
        if self.spatial_balance_mode not in ("quadratic", "none"):
            raise ValueError("spatial_balance_mode must be 'quadratic' or 'none'")
        if self.credit_order not in ("random", "identity"):
            raise ValueError("credit_order must be 'random' or 'identity'")
        if self.revisit_normalisation not in ("fraction", "count"):
            raise ValueError("revisit_normalisation must be 'fraction' or 'count'")
        if self.communication_normalisation not in ("budget_fraction", "count"):
            raise ValueError(
                "communication_normalisation must be 'budget_fraction' or 'count'"
            )
        if self.potential_belief not in ("team", "mean"):
            raise ValueError("potential_belief must be 'team' or 'mean'")


@dataclass
class EpisodeConfig:
    """Episode length and termination."""

    horizon: int = 260
    terminate_on_all_confirmed: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


@dataclass
class ObservationConfig:
    """Observation layout."""

    mode: str = "local"
    window: int = 15
    include_belief: bool = True
    include_visited: bool = True
    include_evidence: bool = True
    include_neighbour_mask: bool = True
    include_option_state: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("local", "global"):
            raise ValueError("observation mode must be 'local' or 'global'")
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3")


@dataclass
class SimConfig:
    """Full simulator configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    uavs: UAVConfig = field(default_factory=UAVConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    comm: CommunicationConfig = field(default_factory=CommunicationConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def reference(cls) -> "SimConfig":
        cfg = cls()
        cfg.targets.layout = "reference"
        cfg.uavs.initial_positions = REFERENCE_UAV_POSES
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimConfig":
        return _from_mapping(cls, data)

    @classmethod
    def from_file(cls, path: str | Path) -> "SimConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML configs") from exc
            data = yaml.safe_load(text)
        elif suffix == ".json":
            import json

            data = json.loads(text)
        else:
            raise ValueError("config must be .yaml, .yml or .json")
        return cls.from_dict(data or {})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kwargs: Any) -> "SimConfig":
        return dataclasses.replace(self, **kwargs)

    def with_fleet(self, count: int) -> "SimConfig":
        import copy

        if count == self.uavs.count:
            return self
        cfg = copy.deepcopy(self)
        cfg.uavs.count = count
        cfg.uavs.initial_positions = default_launch_positions(cfg.grid.size, count)
        cfg.uavs.initial_headings_deg = None
        return cfg

    def validate(self) -> None:
        if self.belief.mode == "indexed" and self.targets.count < 1:
            raise ValueError("indexed belief requires at least one target")
        if self.comm.packets_per_step > 0 and self.uavs.count < 2:
            raise ValueError("communication requires at least two UAVs")
        if self.comm.delay_steps > self.comm.replay_window:
            raise ValueError("replay_window must be >= delay_steps")


def default_launch_positions(
    grid_size: int, count: int
) -> Tuple[Tuple[float, float], ...]:
    inset = 4.0
    hi = grid_size - inset
    corners = [(inset, inset), (hi, inset), (inset, hi), (hi, hi)]
    edges = [
        (grid_size / 2.0, inset), (hi, grid_size / 2.0),
        (grid_size / 2.0, hi), (inset, grid_size / 2.0),
    ]
    points = []
    for pool in (corners, edges):
        for p in pool:
            if len(points) < count:
                points.append(p)
    radius = grid_size / 2.0 - inset
    centre = (grid_size / 2.0, grid_size / 2.0)
    ring = 0
    while len(points) < count:
        theta = 2.0 * np.pi * ring / max(count - len(points), 1)
        points.append((centre[0] + radius * np.cos(theta),
                       centre[1] + radius * np.sin(theta)))
        ring += 1
    return tuple(points[:count])


def _from_mapping(cls: Any, data: Dict[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    names = {f.name for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in names:
            if "metadata" in names:
                extras[key] = value
                continue
            raise KeyError("unknown configuration key %r" % key)
        current = getattr(cls(), key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = _from_mapping(type(current), value)
        else:
            kwargs[key] = value
    if extras:
        kwargs["metadata"] = dict(kwargs.get("metadata") or {}, **extras)
    return cls(**kwargs)
