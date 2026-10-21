"""The HOCCA reference environment."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import sensors as _sensors
from .beliefs import OccupancyBelief, logit
from .communication import CommunicationSystem, DeliveryReport
from .config import SimConfig, default_launch_positions
from .kinematics import EnergyModel, integrate_motion, quantise_angle, turn_to
from .metrics import EpisodeMetrics, StepRecord, coverage_ratio, threshold_crossing
from .rewards import (
    RewardBreakdown, compute_task_reward, distribute_credit, information_credits,
    potential,
)
from .spaces import Box, Discrete, GYMNASIUM_AVAILABLE
from .targets import TargetField, sample_target_layout

__all__ = [
    "HoCCAEnv", "StepResult", "Observation", "Evaluator", "make_env",
    "register_with_gymnasium",
]


class Observation(dict):
    """Observation container with attribute access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


@dataclass
class StepResult:
    """Return value of :meth:`HoCCAEnv.step`."""

    observation: Observation
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    info: Dict[str, Any]

    def __iter__(self):
        yield self.observation
        yield self.rewards
        yield self.terminations
        yield self.truncations
        yield self.info

    def as_tuple(self) -> tuple:
        return tuple(self)


@dataclass
class Evaluator:
    """Ground truth for metrics and figures; never visible to a policy."""

    target_positions: np.ndarray
    target_is_moving: np.ndarray
    target_cells: set
    occupied: np.ndarray

    def confirmed_target_count(self, confirmed_cells: set) -> int:
        return sum(1 for cell in self.target_cells if cell in confirmed_cells)

    def confusion(self, confirmed_cells: set) -> Tuple[int, int, int]:
        tp = len(confirmed_cells & self.target_cells)
        fp = len(confirmed_cells - self.target_cells)
        fn = len(self.target_cells - confirmed_cells)
        return tp, fp, fn


class HoCCAEnv:
    """Parallel multi-agent environment for communication-constrained search."""

    metadata: Dict[str, Any] = {
        "name": "HOCCA-CooperativeSearch-v0",
        "render_modes": ["rgb_array", "human"],
        "is_parallel": True,
    }

    option_count: int = 4
    max_option_age: int = 8

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.config = config if config is not None else SimConfig()
        self.config.validate()
        self.render_mode = render_mode
        cfg = self.config

        self.grid = cfg.grid
        self.sensing_cfg = cfg.sensing
        self.uav_cfg = cfg.uavs
        self.target_cfg = cfg.targets
        self.comm_cfg = cfg.comm
        self.belief_cfg = cfg.belief
        self.reward_cfg = cfg.reward
        self.episode_cfg = cfg.episode
        self.obs_cfg = cfg.observation

        self.n_agents = int(self.uav_cfg.count)
        self.grid_size = int(self.grid.size)
        self.horizon = int(self.episode_cfg.horizon)

        self.sensor = _sensors.SensingModel(
            self.grid_size, self.sensing_cfg.radius,
            self.sensing_cfg.p_detect, self.sensing_cfg.p_false_alarm,
        )
        self.prior_probability = float(self.target_cfg.count) / float(self.grid.cells)
        self.prior_log_odds = float(logit(self.prior_probability, self.belief_cfg.clip))

        self._seed = seed
        self._rng_streams: Dict[str, np.random.Generator] = {}
        self._init_rng(seed)

        self.action_space = Discrete(self.uav_cfg.action_count)
        self.option_space = Discrete(self.option_count)
        self.single_observation_space = self._build_observation_space()
        self.observation_space = self.single_observation_space
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.share_observation_space = self.state_space

        self.step_index = 0
        self.rng = self._rng_streams["reset"]
        self._allocate_episode_state()
        self.reset(seed=seed)

    def _init_rng(self, seed: Optional[int]) -> None:
        streams = ("reset", "targets", "sensing", "channel", "credit", "layout")
        base = np.random.SeedSequence() if seed is None else np.random.SeedSequence(int(seed))
        self._rng_streams = {
            name: np.random.default_rng(child)
            for name, child in zip(streams, base.spawn(len(streams)))
        }
        self.rng = self._rng_streams["reset"]

    def _allocate_episode_state(self) -> None:
        g, n = self.grid_size, self.n_agents
        self.team_belief = OccupancyBelief(
            g, self.prior_probability, self.belief_cfg.clip,
            self.belief_cfg.log_odds_clip,
        )
        self.agent_beliefs = [
            OccupancyBelief(g, self.prior_probability, self.belief_cfg.clip,
                            self.belief_cfg.log_odds_clip)
            for _ in range(n)
        ]
        self._belief_cache = np.zeros((n, g, g), dtype=np.float64)
        self.comm = CommunicationSystem(
            self.comm_cfg, g, n, self._rng_streams["channel"]
        )
        self.energy = EnergyModel(self.uav_cfg, n)
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.headings = np.zeros(n, dtype=np.float64)
        self.options = np.full(n, -1, dtype=np.int64)
        self.option_age = np.zeros(n, dtype=np.int64)
        self.visited = np.zeros((g, g), dtype=bool)
        self.visit_step = np.full((g, g), -1, dtype=np.int64)
        self.agent_visited = np.zeros((n, g, g), dtype=bool)
        self.agent_visit_step = np.full((n, g, g), -1, dtype=np.int16)
        self.confirmed_cells: set = set()
        self._confirmed_mask = np.zeros((g, g), dtype=bool)
        self.footprint_masks = np.zeros((n, g, g), dtype=bool)
        self.received_mask = np.zeros((n, g, g), dtype=np.float64)
        self.received_count = np.zeros((n, n), dtype=np.int64)
        self.sensed_this_step = np.zeros((g, g), dtype=bool)

        self.coverage_history: List[float] = []
        self.recall_history: List[float] = []
        self.entropy_history: List[float] = []
        self.confirmed_history: List[int] = []
        self.trace: List[StepRecord] = []
        self.total_moves = 0
        self.total_packets = 0
        self.total_items = 0
        self.return_sum = 0.0
        self.breakdown_total = RewardBreakdown()
        self._last_report = DeliveryReport()

    def _build_observation_space(self) -> Box:
        cfg = self.obs_cfg
        side = self.grid_size if cfg.mode == "global" else int(cfg.window)
        channels = (
            int(cfg.include_belief) + int(cfg.include_visited)
            + int(cfg.include_evidence)
        )
        dim = side * side * max(channels, 1) + 12 + self.n_agents
        return Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.single_observation_space.shape))

    @property
    def state_dim(self) -> int:
        return self.n_agents * 9 + 12 * 12 * 2 + 8

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    def _to_cell(self, position: np.ndarray) -> Tuple[int, int]:
        return (
            int(np.clip(np.rint(position[0]), 0, self.grid_size - 1)),
            int(np.clip(np.rint(position[1]), 0, self.grid_size - 1)),
        )

    def _default_headings(self, positions: np.ndarray) -> np.ndarray:
        if self.uav_cfg.heading_strategy == "random":
            return quantise_angle(
                self._rng_streams["reset"].uniform(-np.pi, np.pi, len(positions)),
                self.uav_cfg.heading_bins,
            )
        centre = np.array([self.grid_size / 2.0, self.grid_size / 2.0])
        angles = []
        for p in positions:
            direction = centre - np.asarray(p, dtype=np.float64)
            angle = 0.0 if np.linalg.norm(direction) < 1e-9 else float(
                np.arctan2(direction[1], direction[0])
            )
            angles.append(quantise_angle(angle, self.uav_cfg.heading_bins))
        return np.asarray(angles, dtype=np.float64)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Tuple[Observation, Dict[str, Any]]:
        options = dict(options or {})
        if options.get("config") is not None:
            self.config = options["config"]
            self._reconfigure()
        if seed is not None:
            self._init_rng(seed)
            self.comm.rng = self._rng_streams["channel"]
        self.rng = self._rng_streams["reset"]

        self._allocate_episode_state()
        self.step_index = 0

        if options.get("targets") is not None:
            static_cells, moving_cells = options["targets"]
            static_cells = np.asarray(static_cells, dtype=np.int64).reshape(-1, 2)
            moving_cells = np.asarray(moving_cells, dtype=np.int64).reshape(-1, 2)
        else:
            static_cells, moving_cells = sample_target_layout(
                self.grid, self.target_cfg, self._rng_streams["layout"]
            )
        self.targets = TargetField(self.grid, self.target_cfg, static_cells, moving_cells)

        n = self.n_agents
        if options.get("positions") is not None:
            positions = np.asarray(options["positions"], dtype=np.float64).reshape(n, 2)
        elif self.uav_cfg.initial_positions is not None:
            positions = np.asarray(
                self.uav_cfg.initial_positions, dtype=np.float64
            ).reshape(n, 2).copy()
        else:
            positions = np.asarray(
                default_launch_positions(self.grid_size, n), dtype=np.float64
            )
        positions = np.clip(positions, 0.0, float(self.grid_size - 1))

        if options.get("headings_deg") is not None:
            headings = np.deg2rad(np.asarray(options["headings_deg"], dtype=np.float64))
            headings = np.asarray(
                [quantise_angle(h, self.uav_cfg.heading_bins) for h in headings]
            )
        elif self.uav_cfg.initial_headings_deg is not None:
            headings = np.deg2rad(
                np.asarray(self.uav_cfg.initial_headings_deg, dtype=np.float64)
            )
        else:
            headings = self._default_headings(positions)

        self.positions = positions.copy()
        self.headings = np.asarray(headings, dtype=np.float64).copy()
        self.energy.reset(n)

        for belief in self.agent_beliefs:
            belief.reset(self.prior_probability)
        self.team_belief.reset(self.prior_probability)
        self.comm.reset(self.prior_log_odds)

        self.evaluator = self._build_evaluator()
        self._pre_step_log_odds = self.team_belief.ell.copy()
        self._step_evidence = self._sense_and_update(step_index=0)
        self._update_confirmation()
        self._refresh_belief_cache()
        self._update_metrics(step_index=0, reward=0.0)
        self._phi_prev = potential(
            self._potential_belief_probabilities(), self.reward_cfg.lambda_H
        )

        info = self._build_info(
            reward=0.0, breakdown=RewardBreakdown(),
            terminated=False, truncated=False,
        )
        return self._build_observation(), info

    def _reconfigure(self) -> None:
        cfg = self.config
        cfg.validate()
        self.grid = cfg.grid
        self.sensing_cfg = cfg.sensing
        self.uav_cfg = cfg.uavs
        self.target_cfg = cfg.targets
        self.comm_cfg = cfg.comm
        self.belief_cfg = cfg.belief
        self.reward_cfg = cfg.reward
        self.episode_cfg = cfg.episode
        self.obs_cfg = cfg.observation
        self.n_agents = int(self.uav_cfg.count)
        self.grid_size = int(self.grid.size)
        self.horizon = int(self.episode_cfg.horizon)
        self.sensor = _sensors.SensingModel(
            self.grid_size, self.sensing_cfg.radius,
            self.sensing_cfg.p_detect, self.sensing_cfg.p_false_alarm,
        )
        self.prior_probability = float(self.target_cfg.count) / float(self.grid.cells)
        self.prior_log_odds = float(logit(self.prior_probability, self.belief_cfg.clip))
        self.action_space = Discrete(self.uav_cfg.action_count)
        self.single_observation_space = self._build_observation_space()
        self.observation_space = self.single_observation_space
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.share_observation_space = self.state_space

    def step(
        self,
        actions: Union[np.ndarray, Sequence[int], Mapping[int, int]],
        options: Optional[Union[np.ndarray, Sequence[int], Mapping[int, int]]] = None,
    ) -> StepResult:
        """Advance the environment by one step."""

        if self.step_index >= self.horizon:
            raise RuntimeError("episode finished; call reset()")

        action_vec = self._coerce_actions(actions)
        option_vec = self._coerce_options(options)

        self.step_index += 1
        step = self.step_index

        report = self.comm.transmit(step, packets=[])
        self._fuse_deliveries(report)
        self._last_report = report

        self._update_options(option_vec)

        blocked = 0
        moves = 0
        distances = np.zeros(self.n_agents, dtype=np.float64)
        active = self.energy.active_mask()
        turns = np.asarray(self.uav_cfg.turn_actions_deg, dtype=np.float64)
        for u in range(self.n_agents):
            if not active[u]:
                continue
            self.headings[u] = turn_to(
                self.headings[u], float(turns[action_vec[u]]), self.uav_cfg.heading_bins
            )
            before = self.positions[u].copy()
            new_position, was_blocked = integrate_motion(
                before, self.headings[u], self.uav_cfg.speed, self.grid
            )
            self.positions[u] = new_position
            distance = float(np.linalg.norm(new_position - before))
            distances[u] = distance
            if distance > 1e-9:
                moves += 1
            if was_blocked:
                blocked += 1
        self.total_moves += moves
        spent = self.energy.consume(np.arange(self.n_agents), distances)
        energy_fraction = (
            0.0 if not self.energy.enabled
            else float(spent.sum() / max(self.energy.initial.sum(), 1e-12))
        )
        collisions = self._count_collisions()

        self._pre_step_log_odds = self.team_belief.ell.copy()
        self._step_evidence = self._sense_and_update(step_index=step)
        self._predict_beliefs(step)
        self.evaluator = self._build_evaluator()

        self.comm.expire(step)
        packets: List[Any] = []
        for u in range(self.n_agents):
            if active[u]:
                packets.extend(
                    self.comm.schedule(step, u, self.agent_beliefs[u].ell, active)
                )
        report = self.comm.transmit(step, packets)
        self.total_packets += report.packets_sent
        self.total_items += report.items_attempted

        new_confirmations = self._update_confirmation()
        breakdown = compute_task_reward(
            self.reward_cfg,
            new_confirmations=new_confirmations,
            new_cells=int(self._new_cells_this_step),
            repeats=int(self._repeats_this_step),
            looks=int(self._looks_this_step),
            items_attempted=report.items_attempted,
            item_budget_total=max(
                self.comm_cfg.packets_per_step * self.comm_cfg.items_per_packet
                * self.n_agents, 1,
            ),
            energy_spent_fraction=energy_fraction,
            per_agent_new_cells=np.asarray(self._per_agent_new_cells),
            boundary_blocks=blocked,
            collisions=collisions,
            agent_count=self.n_agents,
        )

        phi_now = potential(
            self._potential_belief_probabilities(), self.reward_cfg.lambda_H
        )
        breakdown.shaping = self.reward_cfg.gamma * phi_now - self._phi_prev
        self._phi_prev = phi_now

        team_reward = breakdown.team
        credits = self._information_credits()
        rewards = distribute_credit(
            team_reward, credits,
            self.reward_cfg.lambda_A if self.reward_cfg.differential_credit else 0.0,
        )
        if self.reward_cfg.reward_clip is not None:
            rewards = np.clip(
                rewards, -self.reward_cfg.reward_clip, self.reward_cfg.reward_clip
            )
        self.return_sum += float(team_reward)
        self.breakdown_total += breakdown

        self._refresh_belief_cache()
        self._update_metrics(step_index=step, reward=team_reward)

        terminated = bool(
            self.episode_cfg.terminate_on_all_confirmed
            and self.confirmed_count >= self.n_targets
        )
        truncated = bool(step >= self.horizon)
        if terminated:
            self._phi_prev = 0.0

        info = self._build_info(
            reward=team_reward, breakdown=breakdown,
            terminated=terminated, truncated=truncated,
            report=report, credits=credits,
        )
        info["agents"] = [dict(info["agent_metrics"][u]) for u in range(self.n_agents)]
        return StepResult(
            observation=self._build_observation(),
            rewards=rewards,
            terminations=np.full(self.n_agents, terminated, dtype=bool),
            truncations=np.full(self.n_agents, truncated, dtype=bool),
            info=info,
        )

    def _coerce_actions(self, actions) -> np.ndarray:
        n = self.n_agents
        if isinstance(actions, Mapping):
            out = np.zeros(n, dtype=np.int64)
            for u in range(n):
                if u not in actions:
                    raise KeyError("missing action for agent %d" % u)
                out[u] = int(actions[u])
        else:
            out = np.asarray(actions, dtype=np.int64).reshape(-1)
            if out.size == 1:
                out = np.repeat(out, n)
        if out.size != n:
            raise ValueError("expected %d actions, received %d" % (n, out.size))
        if np.any(out < 0) or np.any(out >= self.uav_cfg.action_count):
            raise ValueError("actions must lie in [0, %d)" % self.uav_cfg.action_count)
        return out

    def _coerce_options(self, options) -> Optional[np.ndarray]:
        if options is None:
            return None
        n = self.n_agents
        if isinstance(options, Mapping):
            out = np.full(n, -1, dtype=np.int64)
            for u, value in options.items():
                out[int(u)] = int(value)
        else:
            out = np.asarray(options, dtype=np.int64).reshape(-1)
            if out.size == 1:
                out = np.repeat(out, n)
        if out.size != n:
            raise ValueError("expected %d options, received %d" % (n, out.size))
        if np.any(out >= self.option_count):
            raise ValueError("options must lie in [-1, %d)" % self.option_count)
        return out

    def _update_options(self, option_vec: Optional[np.ndarray]) -> None:
        if option_vec is None:
            self.options[...] = -1
            self.option_age += 1
            return
        for u, value in enumerate(option_vec):
            if value < 0:
                self.option_age[u] += 1
                continue
            if value != self.options[u]:
                self.options[u] = value
                self.option_age[u] = 0
            else:
                self.option_age[u] += 1
        expired = self.option_age >= self.max_option_age
        if np.any(expired):
            self.options[expired] = -1
            self.option_age[expired] = 0

    def _sense_and_update(self, step_index: int):
        g = self.grid_size
        occupied = self.targets.occupancy()
        rng = self._rng_streams["sensing"]
        active = self.energy.active_mask()

        self.sensed_this_step[:] = False
        per_agent_new = np.zeros(self.n_agents, dtype=np.int64)
        evidence: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros((0, 2), dtype=np.int64), np.zeros(0)) for _ in range(self.n_agents)
        ]

        for u in range(self.n_agents):
            self.footprint_masks[u][:] = False
            if not active[u]:
                continue
            cells = self.sensor.footprint_cells(self.positions[u])
            if len(cells) == 0:
                evidence[u] = (cells, np.zeros(0))
                continue
            self.footprint_masks[u][cells[:, 1], cells[:, 0]] = True
            self.agent_visited[u][cells[:, 1], cells[:, 0]] = True
            self.agent_visit_step[u][cells[:, 1], cells[:, 0]] = np.int16(
                min(step_index, 32000)
            )
            z = self.sensor.sample(rng, cells, occupied)
            deltas = self.sensor.innovations(z)
            self.agent_beliefs[u].apply_innovations(cells, deltas)
            self.team_belief.apply_innovations(cells, deltas)
            evidence[u] = (cells, deltas)
            self.comm.record_observations(u, step_index, cells, deltas)
            fresh = ~self.sensed_this_step[cells[:, 1], cells[:, 0]]
            per_agent_new[u] = int(fresh.sum())

        union = np.zeros((g, g), dtype=bool)
        for u in range(self.n_agents):
            union |= self.footprint_masks[u]
        brand_new = union & ~self.visited
        self._new_cells_this_step = int(brand_new.sum())
        self._looks_this_step = int(union.sum())
        window = max(int(self.reward_cfg.revisit_window), 0)
        stale = union & self.visited & (
            (self.visit_step < 0) | (step_index - self.visit_step > window)
        )
        self._repeats_this_step = int(stale.sum())
        self.visited |= union
        self.visit_step[union] = step_index
        self.sensed_this_step = union

        shares = np.zeros(self.n_agents, dtype=np.float64)
        for u in range(self.n_agents):
            if active[u]:
                shares[u] = float((self.footprint_masks[u] & brand_new).sum())
        self._per_agent_new_cells = shares
        return evidence

    def _predict_beliefs(self, step: int) -> None:
        if not self.targets.hop_due(step):
            return
        p_m = self.target_cfg.move_prob
        if p_m <= 0.0:
            return
        self.team_belief.predict_hop(p_m)
        for belief in self.agent_beliefs:
            belief.predict_hop(p_m)
        if self.belief_cfg.mass_normalisation:
            self.team_belief.normalise_mass(float(self.target_cfg.count))
            for belief in self.agent_beliefs:
                belief.normalise_mass(float(self.target_cfg.count))

    def _fuse_deliveries(self, report: DeliveryReport) -> None:
        self.received_mask[:] = 0.0
        self.received_count[:] = 0
        for receiver, (cells, deltas) in report.fused.items():
            if len(cells) == 0:
                continue
            self.agent_beliefs[receiver].apply_innovations(cells, deltas)
            self.received_mask[receiver][cells[:, 1], cells[:, 0]] += deltas
        for receiver, counts in report.sender_counts.items():
            for sender, count in counts.items():
                self.received_count[receiver, sender] += count

    def _update_confirmation(self) -> int:
        threshold = self.belief_cfg.confirm_log_odds
        newly = 0
        for u in range(self.n_agents):
            mask = self.agent_beliefs[u].ell >= threshold
            new_mask = mask & ~self._confirmed_mask
            if not new_mask.any():
                continue
            self._confirmed_mask |= mask
            for y, x in zip(*np.nonzero(new_mask)):
                cell = (int(x), int(y))
                if cell not in self.confirmed_cells:
                    self.confirmed_cells.add(cell)
                    newly += 1
        return newly

    @property
    def confirmed_count(self) -> int:
        return self.evaluator.confirmed_target_count(self.confirmed_cells)

    def _potential_belief_probabilities(self) -> np.ndarray:
        if self.reward_cfg.potential_belief == "mean":
            if self.n_agents == 1:
                return self.agent_beliefs[0].probabilities()
            return np.mean([b.probabilities() for b in self.agent_beliefs], axis=0)
        return self.team_belief.probabilities()

    def _information_credits(self) -> np.ndarray:
        cfg = self.reward_cfg
        if not cfg.differential_credit:
            return np.zeros(self.n_agents, dtype=np.float64)
        order = (
            list(range(self.n_agents)) if cfg.credit_order == "identity"
            else list(self._rng_streams["credit"].permutation(self.n_agents))
        )
        return information_credits(
            self._pre_step_log_odds, self._step_evidence, order, self.belief_cfg.clip
        )

    def _refresh_belief_cache(self) -> None:
        self._belief_cache = np.stack([b.probabilities() for b in self.agent_beliefs])

    def _update_metrics(self, step_index: int, reward: float = 0.0) -> None:
        coverage = coverage_ratio(self.visited)
        confirmed = self.confirmed_count
        recall = float(confirmed) / float(max(self.n_targets, 1))
        _, fp, _ = self.evaluator.confusion(self.confirmed_cells)
        entropy = float(self.team_belief.entropy())

        self.coverage_history.append(coverage)
        self.recall_history.append(recall)
        self.entropy_history.append(entropy)
        self.confirmed_history.append(int(confirmed))
        self.trace.append(
            StepRecord(
                step=step_index, coverage=coverage, recall=recall,
                confirmed=int(confirmed), confirmed_cells=len(self.confirmed_cells),
                false_cells=int(fp), entropy=entropy,
                packets_sent=int(self._last_report.packets_sent),
                items_attempted=int(self._last_report.items_attempted),
                moves=int(self.total_moves), reward_team=float(reward),
            )
        )

    def _agent_observation(self, u: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.obs_cfg
        g = self.grid_size
        if cfg.mode == "global":
            side, x0, y0 = g, 0, 0
        else:
            side = int(cfg.window)
            half = side // 2
            cx, cy = self._to_cell(self.positions[u])
            x0 = int(np.clip(cx - half, 0, max(g - side, 0)))
            y0 = int(np.clip(cy - half, 0, max(g - side, 0)))

        belief_p = self._belief_cache[u]
        window = np.s_[y0:y0 + side, x0:x0 + side]
        channels: List[np.ndarray] = []
        if cfg.include_belief:
            channels.append(belief_p[window])
        if cfg.include_visited:
            channels.append(self.agent_visited[u][window].astype(np.float64))
        if cfg.include_evidence:
            channels.append(self.received_mask[u][window])
        if not channels:
            channels = [np.zeros((side, side))]

        x, y = self._to_cell(self.positions[u])
        heading = self.headings[u]
        coverage = self.coverage_history[-1] if self.coverage_history else 0.0
        confirmed = self.confirmed_history[-1] if self.confirmed_history else 0
        scalars = np.array([
            x / max(g - 1, 1),
            y / max(g - 1, 1),
            float(np.cos(heading)),
            float(np.sin(heading)),
            float(self.energy.normalised[u]),
            float(self.options[u]) / max(self.option_count - 1, 1)
            if self.options[u] >= 0 else -1.0,
            float(self.option_age[u]) / max(self.max_option_age, 1),
            float(self.step_index) / max(self.horizon, 1),
            float(coverage),
            float(confirmed) / max(self.n_targets, 1),
            float(belief_p[y, x]),
            float(self.received_count[u].sum())
            / max(self.comm_cfg.items_per_packet * max(self.n_agents - 1, 1), 1),
        ], dtype=np.float64)
        neighbour = (
            self.received_count[u].astype(np.float64)
            / max(self.comm_cfg.items_per_packet, 1)
        )
        flat = np.concatenate([np.stack(channels).reshape(-1), scalars, neighbour])
        named = {
            "local_belief": belief_p[window][None, ...].astype(np.float32),
            "visited": self.agent_visited[u][window].astype(np.float32),
            "received": self.received_mask[u][window].astype(np.float32),
            "scalars": scalars.astype(np.float32),
            "neighbour_items": neighbour.astype(np.float32),
            "window_origin": np.array([x0, y0], dtype=np.int32),
        }
        return flat.astype(np.float32), named

    def _global_state(self) -> np.ndarray:
        side, g = 12, self.grid_size
        parts: List[np.ndarray] = []
        for u in range(self.n_agents):
            x, y = self._to_cell(self.positions[u])
            parts.append(np.array([
                x / max(g - 1, 1),
                y / max(g - 1, 1),
                float(np.cos(self.headings[u])),
                float(np.sin(self.headings[u])),
                float(self.energy.normalised[u]),
                float(self.options[u]) / max(self.option_count - 1, 1),
                float(self.option_age[u]) / max(self.max_option_age, 1),
                float(self.footprint_masks[u].sum()),
                1.0 if self.energy.active_mask()[u] else 0.0,
            ], dtype=np.float64))
        parts.append(_downsample(self.team_belief.probabilities(), side).reshape(-1))
        parts.append(_downsample(self.visited.astype(np.float64), side).reshape(-1))
        parts.append(np.array([
            float(self.coverage_history[-1]) if self.coverage_history else 0.0,
            float(self.confirmed_count) / max(self.n_targets, 1),
            float(self.step_index) / max(self.horizon, 1),
            float(self.team_belief.entropy()) / max(g * g, 1),
            float(len(self.confirmed_cells)),
            float(self._confirmed_mask.sum()),
            float(self.total_packets),
            float(self.total_items),
        ], dtype=np.float64))
        return np.concatenate(parts).astype(np.float32)

    def _build_observation(self) -> Observation:
        flats, named = [], []
        for u in range(self.n_agents):
            flat, channels = self._agent_observation(u)
            flats.append(flat)
            named.append(channels)
        return Observation(
            agents=np.stack(flats, axis=0),
            agent_dicts=named,
            state=self._global_state(),
            step=int(self.step_index),
        )

    def _build_info(
        self,
        reward: float,
        breakdown: RewardBreakdown,
        terminated: bool,
        truncated: bool,
        report: Optional[DeliveryReport] = None,
        credits: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        tp, fp, _ = self.evaluator.confusion(self.confirmed_cells)
        precision = 0.0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
        steps = max(self.step_index, 1)
        info: Dict[str, Any] = {
            "step": int(self.step_index),
            "team_reward": float(reward),
            "reward_breakdown": breakdown.as_dict(),
            "coverage": float(self.coverage_history[-1]) if self.coverage_history else 0.0,
            "recall": float(self.confirmed_count) / max(self.n_targets, 1),
            "precision": precision,
            "confirmed": int(self.confirmed_count),
            "confirmed_cells": int(len(self.confirmed_cells)),
            "false_cells": int(fp),
            "entropy": float(self.team_belief.entropy()),
            "moves": int(self.total_moves),
            "packets_sent": int(report.packets_sent) if report is not None else 0,
            "items_attempted": int(report.items_attempted) if report is not None else 0,
            "messages_per_uav_step": float(self.total_packets) / (self.n_agents * steps),
            "option_expired": (self.options < 0).astype(bool),
            "credits": np.zeros(self.n_agents) if credits is None else credits.copy(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "agent_metrics": [],
        }
        for u in range(self.n_agents):
            info["agent_metrics"].append({
                "position": self.positions[u].copy(),
                "heading": float(self.headings[u]),
                "energy": float(self.energy.normalised[u]),
                "option": int(self.options[u]),
                "option_age": int(self.option_age[u]),
                "credit": float(info["credits"][u]),
                "footprint": int(self.footprint_masks[u].sum()),
            })
        return info

    def _count_collisions(self) -> int:
        if self.n_agents < 2:
            return 0
        threshold = max(self.uav_cfg.min_separation, 1.0)
        count = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < threshold:
                    count += 1
        return count

    def episode_metrics(self) -> EpisodeMetrics:
        coverage = self.coverage_history[-1] if self.coverage_history else 0.0
        recall = self.recall_history[-1] if self.recall_history else 0.0
        tp, fp, _ = self.evaluator.confusion(self.confirmed_cells)
        precision = 0.0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
        steps = max(self.step_index, 1)
        return EpisodeMetrics(
            coverage=float(coverage),
            recall=float(recall),
            precision=precision,
            confirmed=int(self.confirmed_count),
            target_count=int(self.n_targets),
            t80_coverage=threshold_crossing(self.coverage_history, 0.8),
            t80_detection=threshold_crossing(self.recall_history, 0.8),
            t_all=threshold_crossing(self.confirmed_history, self.n_targets),
            success=bool(self.confirmed_count >= self.n_targets),
            steps=int(self.step_index),
            moves=int(self.total_moves),
            packets=int(self.total_packets),
            items=int(self.total_items),
            messages_per_uav_step=float(self.total_packets) / (self.n_agents * steps),
            return_sum=float(self.return_sum),
            trace=list(self.trace),
            extra=self.breakdown_total.as_dict(),
        )

    def render(self):
        from .render import render_env

        return render_env(self, mode=self.render_mode or "rgb_array")

    def close(self) -> None:
        return None

    def action_space_sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self._rng_streams["reset"]
        return rng.integers(0, self.uav_cfg.action_count, size=self.n_agents)

    def option_expired_mask(self) -> np.ndarray:
        """True for agents whose option must be reselected."""

        return (self.options < 0) | (self.option_age >= self.max_option_age)

    def belief(self, agent: Optional[int] = None) -> np.ndarray:
        return self.team_belief.probabilities() if agent is None else self._belief_cache[agent]

    def _build_evaluator(self) -> Evaluator:
        return Evaluator(
            target_positions=self.targets.positions.copy(),
            target_is_moving=self.targets.is_moving.copy(),
            target_cells=self.targets.cell_set(),
            occupied=self.targets.occupancy(),
        )

    def evaluator_state(self) -> Evaluator:
        return self.evaluator

    def __repr__(self) -> str:
        return "HoCCAEnv(grid=%dx%d, K=%d, N=%d, horizon=%d, comm=%dx%d)" % (
            self.grid_size, self.grid_size, self.target_cfg.count, self.n_agents,
            self.horizon, self.comm_cfg.packets_per_step,
            self.comm_cfg.items_per_packet,
        )


def _downsample(arr: np.ndarray, side: int) -> np.ndarray:
    g = arr.shape[0]
    if g == side:
        return np.asarray(arr, dtype=np.float64)
    edges = np.linspace(0, g, side + 1).astype(int)
    out = np.zeros((side, side), dtype=np.float64)
    for i in range(side):
        for j in range(side):
            block = arr[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def make_env(
    config: Optional[SimConfig] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    **overrides: Any,
) -> HoCCAEnv:
    """Create a :class:`HoCCAEnv`; overrides use ``group__field`` keys."""

    cfg = copy.deepcopy(config) if config is not None else SimConfig()
    for key, value in overrides.items():
        target = cfg
        parts = key.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return HoCCAEnv(cfg, seed=seed, render_mode=render_mode)


def register_with_gymnasium() -> bool:
    """Register the environment with Gymnasium when it is installed."""

    if not GYMNASIUM_AVAILABLE:
        return False
    import gymnasium as gym

    if "HOCCA-CooperativeSearch-v0" in gym.registry:
        return True
    gym.register(
        id="HOCCA-CooperativeSearch-v0",
        entry_point="hocca_sim.env:HoCCAEnv",
        max_episode_steps=None,
    )
    return True
