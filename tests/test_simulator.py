"""Unit tests for the HOCCA simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hocca_sim import SimConfig, make_env
from hocca_sim.beliefs import (
    OccupancyBelief, binary_entropy, expected_looks_to_confirm,
    kl_bernoulli_after_item, logit, mutual_information, sigmoid,
)
from hocca_sim.config import (
    REFERENCE_MOVING_TARGETS, REFERENCE_STATIC_TARGETS, REFERENCE_UAV_POSES,
)
from hocca_sim.kinematics import integrate_motion, quantise_angle, turn_to
from hocca_sim.policies import RandomPolicy, RuleBasedPolicy
from hocca_sim.rollout import rollout


class TestPaperParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimConfig()

    def test_grid_and_targets(self) -> None:
        self.assertEqual(self.cfg.grid.size, 48)
        self.assertEqual(self.cfg.grid.cells, 2304)
        self.assertEqual(self.cfg.targets.count, 16)
        self.assertEqual(self.cfg.targets.moving_count, 4)
        self.assertEqual(self.cfg.targets.static_count, 12)

    def test_prior_occupancy(self) -> None:
        prior = self.cfg.belief.prior_occupancy(16, 2304)
        self.assertAlmostEqual(prior, 16.0 / 2304.0, places=12)
        self.assertLess(prior, 0.01)

    def test_sensor_and_actions(self) -> None:
        self.assertEqual(self.cfg.sensing.radius, 3.0)
        self.assertEqual(self.cfg.sensing.p_detect, 0.90)
        self.assertEqual(self.cfg.sensing.p_false_alarm, 0.02)
        self.assertEqual(self.cfg.uavs.turn_actions_deg, (-30.0, 0.0, 30.0))
        self.assertEqual(self.cfg.uavs.heading_bins, 12)
        self.assertAlmostEqual(self.cfg.uavs.heading_step_deg, 30.0)

    def test_motion_and_budget(self) -> None:
        self.assertEqual(self.cfg.targets.hop_period, 20)
        self.assertEqual(self.cfg.targets.move_prob, 0.25)
        self.assertEqual(self.cfg.comm.packets_per_step, 2)
        self.assertEqual(self.cfg.comm.items_per_packet, 8)
        self.assertEqual(self.cfg.comm.item_budget, 16)
        self.assertEqual(self.cfg.episode.horizon, 260)

    def test_reward_coefficients(self) -> None:
        reward = self.cfg.reward
        self.assertEqual(
            (reward.lambda_H, reward.lambda_D, reward.lambda_R,
             reward.lambda_C, reward.lambda_E, reward.lambda_F),
            (1.0, 25.0, 1.5, 0.3, 0.02, 0.2),
        )
        self.assertEqual(reward.gamma, 0.99)

    def test_options(self) -> None:
        env = make_env(seed=0)
        self.assertEqual(env.option_count, 4)
        self.assertEqual(env.max_option_age, 8)

    def test_reference_geometry(self) -> None:
        self.assertEqual(len(REFERENCE_STATIC_TARGETS), 12)
        self.assertEqual(len(REFERENCE_MOVING_TARGETS), 4)
        self.assertEqual(REFERENCE_UAV_POSES,
                         ((4.0, 4.0), (44.0, 4.0), (4.0, 44.0), (44.0, 44.0)))


class TestBeliefAlgebra(unittest.TestCase):
    def test_sigmoid_logit_roundtrip(self) -> None:
        p = np.array([0.001, 0.01, 0.5, 0.9, 0.999])
        np.testing.assert_allclose(sigmoid(logit(p)), p, rtol=1e-9, atol=1e-12)

    def test_entropy(self) -> None:
        self.assertAlmostEqual(float(binary_entropy(0.0)), 0.0, places=12)
        self.assertAlmostEqual(float(binary_entropy(1.0)), 0.0, places=12)
        self.assertAlmostEqual(float(binary_entropy(0.5)), np.log(2.0), places=12)

    def test_log_odds_innovations(self) -> None:
        cfg = SimConfig()
        self.assertAlmostEqual(cfg.sensing.log_odds_hit, np.log(0.90 / 0.02), places=12)
        self.assertAlmostEqual(cfg.sensing.log_odds_miss, np.log(0.10 / 0.98), places=12)

    def test_three_hits_confirm(self) -> None:
        cfg = SimConfig()
        prior = logit(cfg.belief.prior_occupancy(16, 2304))
        self.assertGreater(prior + 3 * cfg.sensing.log_odds_hit,
                           cfg.belief.confirm_log_odds)
        self.assertLess(prior + 2 * cfg.sensing.log_odds_hit,
                        cfg.belief.confirm_log_odds)

    def test_expected_looks(self) -> None:
        cfg = SimConfig()
        prior = logit(cfg.belief.prior_occupancy(16, 2304))
        looks = expected_looks_to_confirm(cfg.belief.confirm_log_odds, prior, 0.9, 0.02)
        self.assertGreater(looks, 2.0)
        self.assertLess(looks, 4.0)

    def test_mutual_information(self) -> None:
        mi = mutual_information(np.array([0.0, 0.2, 0.5, 0.9, 1.0]), 0.9, 0.02)
        self.assertAlmostEqual(float(mi[0]), 0.0, places=12)
        self.assertAlmostEqual(float(mi[-1]), 0.0, places=12)
        self.assertGreater(float(mi[2]), float(mi[1]))
        self.assertTrue(np.all(mi >= 0.0))

    def test_voi_decreases_with_redundancy(self) -> None:
        self.assertLess(float(kl_bernoulli_after_item(0.999, 3.8066)),
                        float(kl_bernoulli_after_item(0.0069, 3.8066)))

    def test_belief_update(self) -> None:
        belief = OccupancyBelief(8, 0.25)
        belief.apply_innovations(np.array([[1, 1], [2, 3]]),
                                 np.array([3.8066, -2.2823]))
        self.assertAlmostEqual(belief.ell[1, 1], logit(0.25) + 3.8066, places=6)

    def test_prediction_conserves_mass(self) -> None:
        belief = OccupancyBelief(12, 0.1)
        belief.ell += np.random.default_rng(0).normal(0, 1, belief.ell.shape)
        before = belief.mass()
        belief.predict_hop(0.25)
        self.assertAlmostEqual(before, belief.mass(), places=6)


class TestEnvironmentProtocol(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(seed=0)

    def test_reset_shapes(self) -> None:
        obs, info = self.env.reset(seed=0)
        self.assertEqual(obs["agents"].shape, (4, self.env.obs_dim))
        self.assertEqual(obs["state"].shape, (self.env.state_dim,))
        self.assertEqual(obs["agents"].dtype, np.float32)
        self.assertIn("coverage", info)

    def test_action_validation(self) -> None:
        self.env.reset(seed=0)
        with self.assertRaises(ValueError):
            self.env.step(np.full(4, 7))
        with self.assertRaises(ValueError):
            self.env.step(np.zeros(3, dtype=int))

    def test_step_contract(self) -> None:
        self.env.reset(seed=0)
        result = self.env.step(np.zeros(4, dtype=int))
        self.assertEqual(result.rewards.shape, (4,))
        self.assertEqual(result.terminations.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.rewards)))
        self.assertIn("team_reward", result.info)

    def test_coverage_monotone(self) -> None:
        self.env.reset(seed=0)
        previous = self.env.coverage_history[-1]
        for _ in range(12):
            self.env.step(np.zeros(4, dtype=int))
            current = self.env.coverage_history[-1]
            self.assertGreaterEqual(current + 1e-12, previous)
            previous = current

    def test_footprint_bound(self) -> None:
        self.env.reset(seed=0)
        area = len(self.env.sensor.footprint_cells(self.env.positions[0]))
        before = int(self.env.visited.sum())
        self.env.step(np.zeros(4, dtype=int))
        after = int(self.env.visited.sum())
        self.assertLessEqual(after - before, 4 * area)
        self.assertGreater(after, before)

    def test_horizon_truncates(self) -> None:
        env = make_env(seed=1)
        env.reset(seed=1)
        result = None
        for _ in range(env.horizon):
            result = env.step(np.zeros(4, dtype=int))
            if result.terminations[0]:
                break
        self.assertTrue(result.truncations[0] or result.terminations[0])

    def test_no_ground_truth_in_observation(self) -> None:
        obs, _ = self.env.reset(seed=0)
        for key in obs["agent_dicts"][0]:
            self.assertNotIn("target", key.lower())
        self.assertNotIn("evaluator", obs)


class TestCommunication(unittest.TestCase):
    def test_budget_enforced(self) -> None:
        env = make_env(seed=2)
        env.reset(seed=2)
        for _ in range(30):
            result = env.step(np.zeros(4, dtype=int))
            self.assertLessEqual(
                result.info["packets_sent"],
                env.comm_cfg.packets_per_step * env.n_agents,
            )
            self.assertLessEqual(
                result.info["items_attempted"],
                env.comm_cfg.item_budget * env.n_agents,
            )

    def test_provenance_fused_once(self) -> None:
        env = make_env(seed=3)
        env.reset(seed=3)
        for _ in range(25):
            env.step(np.zeros(4, dtype=int))
        for applied in env.comm.applied_ids:
            self.assertEqual(len(applied), len(set(applied)))

    def test_silent_fleet(self) -> None:
        cfg = SimConfig()
        cfg.comm.packets_per_step = 0
        env = make_env(cfg, seed=4)
        env.reset(seed=4)
        for _ in range(10):
            env.step(np.zeros(4, dtype=int))
        self.assertEqual(env.comm.total_packets_attempted, 0)
        self.assertTrue(np.all(env.received_count == 0))

    def test_packet_loss(self) -> None:
        cfg = SimConfig()
        cfg.comm.packet_loss = 0.9
        env = make_env(cfg, seed=5)
        env.reset(seed=5)
        for _ in range(20):
            env.step(np.zeros(4, dtype=int))
        self.assertGreater(env.comm.total_packets_lost, 0)


class TestKinematics(unittest.TestCase):
    def test_heading_lattice(self) -> None:
        self.assertAlmostEqual(quantise_angle(np.deg2rad(31.0), 12),
                               np.deg2rad(30.0), places=12)
        self.assertAlmostEqual(turn_to(0.0, 30.0, 12), np.deg2rad(30.0), places=12)
        self.assertAlmostEqual(turn_to(0.0, -30.0, 12), np.deg2rad(-30.0), places=12)

    def test_unit_speed(self) -> None:
        grid = SimConfig().grid
        position = np.array([24.0, 24.0])
        new, blocked = integrate_motion(position, 0.0, 1.0, grid)
        self.assertFalse(blocked)
        self.assertAlmostEqual(float(np.linalg.norm(new - position)), 1.0, places=12)

    def test_boundary_clamp(self) -> None:
        grid = SimConfig().grid
        new, blocked = integrate_motion(np.array([0.0, 10.0]), np.pi, 1.0, grid)
        self.assertTrue(blocked)
        self.assertGreaterEqual(new[0], 0.0)


class TestRolloutAndPolicy(unittest.TestCase):
    def test_rule_policy_episode(self) -> None:
        env = make_env(seed=0)
        roll = rollout(env, RuleBasedPolicy(), seed=0, record_frames=False)
        self.assertGreater(roll.steps, 0)
        self.assertLessEqual(roll.steps, env.horizon)
        self.assertGreaterEqual(roll.metrics.coverage, 0.0)
        self.assertLessEqual(roll.metrics.coverage, 1.0)
        self.assertGreater(roll.metrics.moves, 0)

    def test_rule_beats_random(self) -> None:
        def t80(policy) -> float:
            roll = rollout(make_env(seed=0), policy, seed=0, record_frames=False)
            value = roll.metrics.t80_coverage
            return float("inf") if value is None else float(value)

        self.assertLess(t80(RuleBasedPolicy()), t80(RandomPolicy(seed=0)))

    def test_reference_layout(self) -> None:
        env = make_env(SimConfig.reference(), seed=0)
        env.reset(seed=0)
        for cell in REFERENCE_STATIC_TARGETS + REFERENCE_MOVING_TARGETS:
            self.assertIn(cell, env.evaluator.target_cells)
        np.testing.assert_allclose(env.positions, np.asarray(REFERENCE_UAV_POSES))

    def test_determinism(self) -> None:
        a = rollout(make_env(seed=7), RuleBasedPolicy(), seed=7, record_frames=False)
        b = rollout(make_env(seed=7), RuleBasedPolicy(), seed=7, record_frames=False)
        np.testing.assert_allclose(a.coverage, b.coverage)
        self.assertEqual(a.metrics.confirmed, b.metrics.confirmed)


class TestConfigFile(unittest.TestCase):
    def test_yaml_roundtrip(self) -> None:
        path = ROOT / "configs" / "reference_48x48.yaml"
        if not path.exists():
            self.skipTest("reference config not present")
        try:
            cfg = SimConfig.from_file(path)
        except RuntimeError:
            self.skipTest("PyYAML not installed")
        self.assertEqual(cfg.grid.size, 48)
        self.assertEqual(cfg.targets.count, 16)
        self.assertEqual(cfg.uavs.count, 4)
        self.assertEqual(cfg.comm.packets_per_step, 2)
        self.assertEqual(cfg.episode.horizon, 260)


if __name__ == "__main__":
    unittest.main(verbosity=2)
