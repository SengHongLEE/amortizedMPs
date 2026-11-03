"""Contracts for the two-policy MP_full_a1 training task."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from test_interoceptive_rewards import load_class
from test_equilibrium_dynamics import module


Config = load_class('mp_full_config.py', 'MotorPrimitivesCfg')
source = (Path(__file__).resolve().parents[1] / 'legged_gym/envs/a1/mp_full_config.py').read_text()
tree = ast.parse(source)
train_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotorPrimitivesCfgPPO')
train_node.bases = []
namespace = {'np': np, 'MotorPrimitivesCfg': Config,
             'MODULATION_PARAM_ACTOR_TYPE': 'snn',
             'PRIMITIVE_SELECTION_ACTOR_TYPE': 'lif_reservoir_snn_single_output'}
exec(compile(ast.Module(body=[train_node], type_ignores=[]), 'mp_full_config.py', 'exec'), namespace)
TrainConfig = namespace['MotorPrimitivesCfgPPO']
RewardEnv = load_class('mp_full.py', 'MotorPrimitives', {
    '_reward_termination', '_reward_locomotion_distance',
    '_passed_obstacle_count', '_reward_obstacle_clearance',
    '_reward_lateral_deviation',
    '_reward_orientation_yaw',
    '_clamp_hazard_curriculum_levels',
    '_get_env_origins',
    '_policy_command_observations',
    '_lane_heading_target',
    '_heading_rate_target',
    '_post_physics_step_callback', '_resample_commands',
    'check_termination', 'compute_reward'})


class MPFullTrainingContractTests(unittest.TestCase):
    def test_controller_receives_exactly_two_policy_outputs(self):
        self.assertEqual(Config.env.num_ee_equilibriums, 0)
        self.assertEqual(Config.env.num_actions, 1 + Config.env.num_modulation_params)
        self.assertEqual(Config.env.num_modulation_params, 18)
        self.assertEqual(Config.env.action_space, 'cpg_equilibrium_v3')
        self.assertEqual(module().AdaptiveEquilibriumDynamics.NUM_ACTIONS, Config.env.num_actions)
        self.assertEqual(module().AdaptiveEquilibriumDynamics.NUM_OBSERVATIONS,
                         Config.env.num_controller_observations)
        self.assertEqual(TrainConfig.runner_class_name, 'MultiInteroceptivePolicyRunner')

    def test_gap_variants_have_separate_easy_and_advanced_columns(self):
        gap_columns = [(i, name) for i, name in enumerate(Config.terrain.terrain_columns)
                       if name.startswith('gap_')]
        self.assertEqual([name for _, name in gap_columns],
                         ['gap_continuous', 'gap_single', 'gap_single', 'gap_continuous'])
        self.assertEqual(Config.terrain.curriculum_gap_medium_columns, 2)
        self.assertEqual(Config.terrain.curriculum_gap_hard_columns, 1)
        self.assertEqual(gap_columns[-1][1], 'gap_continuous')

    def test_hurdle_course_uses_uniform_band_spacing(self):
        self.assertEqual(Config.terrain.terrain_params['hurdle']['pattern'], 'uniform')

    def test_observation_keeps_all_effective_actions_and_looks_ahead(self):
        self.assertEqual(Config.env.num_observations,
                         37 + Config.env.num_actions + 2 * 16 * 5 + Config.env.num_controller_observations)
        self.assertGreaterEqual(max(Config.terrain.measured_points_x), 1.2)
        self.assertLessEqual(Config.control.primitive_selection_update_cycle, 20)

    def test_actor_observes_lateral_offset_in_unused_y_command_channel(self):
        self.assertFalse(Config.env.observe_lane_offset)
        env = RewardEnv()
        env.cfg = SimpleNamespace(env=SimpleNamespace(
            observe_lane_offset=True, lane_offset_observation_scale=1.5))
        env.commands = torch.tensor([[1., 0., .2], [1., 0., -.2]])
        env.commands_scale = torch.tensor([2., 2., 1.])
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 1] = torch.tensor([5.5, 2.5])
        env.env_origins = torch.tensor([[4., 4., 0.], [4., 4., 0.]])
        observed = env._policy_command_observations()
        torch.testing.assert_close(observed, torch.tensor([[2., 1., .2], [2., -1., -.2]]))
        env.cfg.env.observe_lane_offset = False
        torch.testing.assert_close(
            env._policy_command_observations(),
            torch.tensor([[2., 0., .2], [2., 0., -.2]]))
        self.assertEqual(Config.commands.ranges.lin_vel_y, [-0., 0.0])

    def test_lane_heading_feedback_points_toward_column_center(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(commands=SimpleNamespace(lane_heading_gain=1.))
        env.commands = torch.zeros(2, 4)
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 1] = torch.tensor([5., 3.])
        env.env_origins = torch.tensor([[4., 4., 0.], [4., 4., 0.]])
        torch.testing.assert_close(
            env._lane_heading_target(),
            torch.tensor([-torch.pi / 4, torch.pi / 4]))

    def test_yaw_reward_penalizes_heading_error_for_both_policies(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(commands=SimpleNamespace(lane_heading_gain=0.))
        env.commands = torch.zeros(2, 4)
        env.root_states = torch.zeros(2, 13)
        env.env_origins = torch.zeros(2, 3)
        half_yaw = torch.tensor([0., .25])
        env.base_quat = torch.stack((torch.zeros(2), torch.zeros(2),
                                     half_yaw.sin(), half_yaw.cos()), dim=1)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
        torch.testing.assert_close(env._reward_orientation_yaw(), torch.tensor([0., .25]))
        env.cfg.commands.lane_heading_gain = 1.
        env.root_states[0, 1] = 1.
        self.assertAlmostEqual(env._reward_orientation_yaw()[0].item(),
                               torch.pi ** 2 / 16, places=5)
        self.assertIn('orientation_yaw', Config.rewards.primitive_selection_rewards)
        self.assertIn('orientation_yaw', Config.rewards.modulation_param_rewards)

    def test_heading_rate_gain_can_recover_large_yaw_errors(self):
        self.assertEqual(Config.commands.heading_kp, 2.)
        env = RewardEnv()
        env.cfg = SimpleNamespace(commands=SimpleNamespace(
            lane_heading_gain=0., heading_kp=2.))
        env.commands = torch.zeros(2, 4)
        env.root_states = torch.zeros(2, 13)
        env.env_origins = torch.zeros(2, 3)
        globals_ = RewardEnv._heading_rate_target.__globals__
        original = globals_.get('wrap_to_pi')
        globals_['wrap_to_pi'] = lambda value: value
        try:
            rate = env._heading_rate_target(torch.tensor([.5, -.5]))
        finally:
            if original is None:
                del globals_['wrap_to_pi']
            else:
                globals_['wrap_to_pi'] = original
        torch.testing.assert_close(rate, torch.tensor([-1., 1.]))

    def test_fall_penalty_excludes_timeouts_and_reaches_both_policies(self):
        env = RewardEnv()
        env.reset_buf = torch.tensor([True, True, False])
        env.time_out_buf = torch.tensor([False, True, False])
        torch.testing.assert_close(env._reward_termination(), torch.tensor([1., 0., 0.]))
        self.assertIn('termination', Config.rewards.modulation_param_rewards)
        self.assertIn('termination', Config.rewards.primitive_selection_rewards)
        self.assertGreater(-Config.rewards.scales.termination,
                           Config.rewards.scales.obstacle_clearance)

    def test_fall_penalty_is_applied_to_both_policy_rewards_once(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(
            primitive_selection_rewards=['termination'],
            modulation_param_rewards=['termination'], ee_equilibrium_rewards=[],
            task_rewards=[], only_positive_rewards=False),
            env=SimpleNamespace(play=False))
        env.reset_buf = torch.tensor([True, True])
        env.time_out_buf = torch.tensor([False, True])
        env.rew_buf = torch.zeros(2)
        env.primitive_selection_rew_buf = torch.zeros(2)
        env.modulation_param_rew_buf = torch.zeros(2)
        env.ee_equilibrium_rew_buf = torch.zeros(2)
        env.reward_names = []
        env.reward_functions = []
        env.reward_scales = {'termination': -1.}
        env.primitive_selection_reward_scales = {}
        env.episode_sums = {'termination': torch.zeros(2), 'pi_g': torch.zeros(2)}
        rewards = env.compute_reward()
        for item in rewards[:3]:
            torch.testing.assert_close(item, torch.tensor([-1., 0.]))

    def test_forward_progress_does_not_reward_overspeed(self):
        env = RewardEnv()
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 0] = .03  # 1.5 m/s over a 20 ms control step
        env.last_base_pos = torch.zeros(2)
        env.commands = torch.tensor([[.5, 0., 0.], [1.2, 0., 0.]])
        env.dt = .02
        env.cfg = Config
        torch.testing.assert_close(env._reward_locomotion_distance(), torch.tensor([.5, 1.2]))

    def test_lateral_deviation_is_penalized_for_both_policies(self):
        env = RewardEnv()
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 1] = torch.tensor([4., 6.])
        env.env_origins = torch.tensor([[4., 4., 0.], [4., 4., 0.]])
        torch.testing.assert_close(env._reward_lateral_deviation(), torch.tensor([0., 4.]))
        self.assertLess(Config.rewards.scales.lateral_deviation, 0.)
        self.assertIn('lateral_deviation', Config.rewards.modulation_param_rewards)
        self.assertIn('lateral_deviation', Config.rewards.primitive_selection_rewards)

    def test_obstacle_curriculum_reserves_medium_and_hard_columns(self):
        env = RewardEnv()
        env.cfg = Config
        env.terrain_types = torch.arange(8)
        env.terrain_levels = torch.zeros(8, dtype=torch.long)
        env._gap_columns = torch.tensor([True, True, True, True, False, False, False, False])
        env._hurdle_columns = ~env._gap_columns
        env._clamp_hazard_curriculum_levels(torch.arange(8))
        self.assertEqual(env.terrain_levels.tolist(), [1, 1, 4, 7, 0, 0, 4, 7])

    def test_playback_skips_hazard_curriculum_floors(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=Config.terrain, env=SimpleNamespace(play=True))
        env.terrain_types = torch.tensor([0])
        env.terrain_levels = torch.tensor([0])
        env._gap_columns = torch.tensor([False])
        env._hurdle_columns = torch.tensor([True])
        env._clamp_hazard_curriculum_levels(torch.tensor([0]))
        self.assertEqual(env.terrain_levels.tolist(), [0])

    def test_hurdle_floor_is_applied_before_environment_origins_are_assigned(self):
        env = RewardEnv()
        env.device, env.num_envs = 'cpu', 8
        terrain_cfg = SimpleNamespace(
            mesh_type='trimesh', num_cols=8, num_rows=10, curriculum=True,
            curriculum_gap_min_level=1, curriculum_gap_medium_columns=2,
            curriculum_gap_medium_min_level=4, curriculum_gap_hard_columns=1,
            curriculum_gap_hard_min_level=7, curriculum_hurdle_min_level=0,
            curriculum_hurdle_medium_columns=2, curriculum_hurdle_medium_min_level=4,
            curriculum_hurdle_hard_columns=1, curriculum_hurdle_hard_min_level=7)
        env.cfg = SimpleNamespace(terrain=terrain_cfg, env=SimpleNamespace(play=False))
        env.terrain = SimpleNamespace(
            difficulties=np.zeros((10, 8)), env_origins=np.zeros((10, 8, 3)),
            terrain_names=np.array([['gap'] * 4 + ['hurdle'] * 4]))
        globals_ = RewardEnv._get_env_origins.__globals__
        original = globals_.get('TerrainCurriculum')
        globals_['TerrainCurriculum'] = lambda *args, **kwargs: SimpleNamespace(
            levels=torch.zeros(8, dtype=torch.long))
        try:
            env._get_env_origins()
        finally:
            if original is None:
                del globals_['TerrainCurriculum']
            else:
                globals_['TerrainCurriculum'] = original
        self.assertEqual(env.terrain_levels.tolist(), [1, 1, 4, 7, 0, 0, 4, 7])

    def test_obstacle_commands_cover_evaluation_speeds(self):
        self.assertLessEqual(Config.commands.obstacle_lin_vel_x[0], .8)
        self.assertGreaterEqual(Config.commands.obstacle_lin_vel_x[1], 1.2)
        self.assertGreaterEqual(Config.commands.obstacle_resampling_time,
                                Config.env.episode_length_s)

    def test_obstacle_speed_sampling_and_hold_period(self):
        env = RewardEnv()
        env.cfg = Config
        env.device = 'cpu'
        env.dt = .02
        env.commands = torch.zeros(2, 4)
        env.command_ranges = {'lin_vel_x': [.3, 1.5], 'lin_vel_y': [0., 0.],
                              'heading': [0., 0.]}
        env.terrain_types = torch.tensor([0, 1])
        env._gap_columns = torch.tensor([True, False])
        env._hurdle_columns = torch.tensor([False, False])
        globals_ = RewardEnv._resample_commands.__globals__
        original = globals_.get('torch_rand_float')
        globals_['torch_rand_float'] = lambda lo, hi, size, device: torch.full(
            size, (lo + hi) / 2, device=device)
        try:
            env._resample_commands(torch.arange(2))
        finally:
            if original is None:
                del globals_['torch_rand_float']
            else:
                globals_['torch_rand_float'] = original
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([1., .9]))

        env.cfg = SimpleNamespace(
            commands=SimpleNamespace(resampling_time=5, obstacle_resampling_time=10,
                                     heading_command=False),
            terrain=SimpleNamespace(measure_heights=False),
            domain_rand=SimpleNamespace(push_robots=False))
        called = []
        env._resample_commands = lambda ids: called.append(ids.tolist())
        env.episode_length_buf = torch.tensor([250, 250])
        env._post_physics_step_callback()
        self.assertEqual(called.pop(), [1])
        env.episode_length_buf[:] = 500
        env._post_physics_step_callback()
        self.assertEqual(called.pop(), [0, 1])

    def test_obstacle_lane_matches_evaluation_corridor(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=2, num_cols=2, boundary_margin=.6,
            border_size=4., min_base_clearance=.15, hurdle_min_base_clearance=.10,
            hazard_lane_half_width=1.5))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, :3] = torch.tensor([[4., 5.6, .32], [4., 5.6, .32]])
        env.env_origins = torch.tensor([[4., 4., 0.]] * 2)
        env.terrain_types = torch.tensor([0, 1])
        env._gap_columns = torch.tensor([True, False])
        env._hurdle_columns = torch.tensor([False, True])
        env.contact_forces = torch.zeros(2, 1, 3)
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.ones(2, dtype=torch.long)
        env.max_episode_length = 500
        env._get_terrain_heights = lambda xy: torch.zeros(len(xy))
        env.check_termination()
        self.assertEqual(env.terrain_failure_buf.tolist(), [True, True])
        self.assertEqual(env.terrain_lane_exit_buf.tolist(), [True, True])
        self.assertEqual(env.terrain_low_clearance_buf.tolist(), [False, False])

    def test_body_tilt_penalty_reaches_both_policy_optimizers(self):
        self.assertLess(Config.rewards.scales.orientation, 0.)
        self.assertIn('orientation', Config.rewards.modulation_param_rewards)
        self.assertIn('orientation', Config.rewards.primitive_selection_rewards)

    def test_obstacle_clearance_rewards_each_band_once_and_skips_falls(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(mesh_type='trimesh'))
        env.root_states = torch.zeros(3, 13)
        env.root_states[:, 0] = 6.1
        env.terrain_types = torch.tensor([0, 1, 0])
        env._obstacle_clear_targets = torch.tensor([
            [.85, 2.85, 6.05, 8.05, 8.85, 10.85, 14.05, 16.05],
            [float('inf')] * 8])
        env._obstacle_cleared_count = torch.tensor([2, 0, 2])
        env.terrain_failure_buf = torch.tensor([False, False, True])
        torch.testing.assert_close(env._reward_obstacle_clearance(), torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(env._reward_obstacle_clearance(), torch.zeros(3))
        env.root_states[0, 0] = 8.1
        torch.testing.assert_close(env._reward_obstacle_clearance(), torch.tensor([1., 0., 0.]))
        env.root_states[0, 0] = 8.9
        torch.testing.assert_close(env._reward_obstacle_clearance(), torch.tensor([1., 0., 0.]))
        env.root_states[0, 0] = 14.1
        torch.testing.assert_close(env._reward_obstacle_clearance(), torch.tensor([2., 0., 0.]))

    def test_one_time_clearance_reward_reaches_both_policies(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(
            primitive_selection_rewards=['obstacle_clearance'],
            modulation_param_rewards=['obstacle_clearance'], ee_equilibrium_rewards=[],
            task_rewards=[], only_positive_rewards=False),
            env=SimpleNamespace(play=False))
        env.rew_buf = torch.zeros(1)
        env.primitive_selection_rew_buf = torch.zeros(1)
        env.modulation_param_rew_buf = torch.zeros(1)
        env.ee_equilibrium_rew_buf = torch.zeros(1)
        env.reward_names = ['obstacle_clearance']
        calls = []

        def one_time_reward():
            calls.append(1)
            return torch.tensor([1.]) if len(calls) == 1 else torch.tensor([0.])

        env.reward_functions = [one_time_reward]
        env.reward_scales = {'obstacle_clearance': 3.}
        env.primitive_selection_reward_scales = {'obstacle_clearance': 3.}
        env.episode_sums = {name: torch.zeros(1) for name in (
            'obstacle_clearance', 'selection_obstacle_clearance', 'pi_g')}
        rewards = env.compute_reward()
        self.assertEqual(len(calls), 1)
        for reward in rewards[:3]:
            torch.testing.assert_close(reward, torch.tensor([3.]))

    def test_hurdle_clearance_threshold_avoids_false_fall(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=2, num_cols=2, boundary_margin=.6,
            min_base_clearance=.15, hurdle_min_base_clearance=.10))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 0] = 4.
        env.root_states[:, 1] = 4.
        env.root_states[:, 2] = .27
        env.env_origins = torch.tensor([[4., 4., 0.]] * 2)
        env.terrain_types = torch.tensor([0, 1])
        env._hurdle_columns = torch.tensor([True, False])
        env.contact_forces = torch.zeros(2, 1, 3)
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.ones(2, dtype=torch.long)
        env.max_episode_length = 500
        env._get_terrain_heights = lambda xy: torch.full((len(xy),), .125)
        env.check_termination()
        self.assertEqual(env.terrain_failure_buf.tolist(), [False, True])

    def test_last_tile_can_clear_its_second_band_in_the_map_border(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=2, num_cols=1, boundary_margin=.6,
            border_size=4., min_base_clearance=.15, hurdle_min_base_clearance=.10))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(1, 13)
        env.root_states[0, :3] = torch.tensor([16.2, 4., .32])
        env.env_origins = torch.tensor([[12., 4., 0.]])
        env.terrain_types = torch.tensor([0])
        env._hurdle_columns = torch.tensor([True])
        env.contact_forces = torch.zeros(1, 1, 3)
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.ones(1, dtype=torch.long)
        env.max_episode_length = 500
        env._get_terrain_heights = lambda xy: torch.zeros(len(xy))
        env.check_termination()
        self.assertFalse(env.time_out_buf.item())

    def test_crossing_into_neighboring_terrain_column_is_a_failure(self):
        env = RewardEnv()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=2, num_cols=2, boundary_margin=.6,
            border_size=4., min_base_clearance=.15, hurdle_min_base_clearance=.10))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(1, 13)
        env.root_states[0, :3] = torch.tensor([4., 7.5, .32])
        env.env_origins = torch.tensor([[4., 4., 0.]])
        env.terrain_types = torch.tensor([0])
        env._hurdle_columns = torch.tensor([True])
        env.contact_forces = torch.zeros(1, 1, 3)
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.ones(1, dtype=torch.long)
        env.max_episode_length = 500
        env._get_terrain_heights = lambda xy: torch.zeros(len(xy))
        env.check_termination()
        self.assertTrue(env.terrain_failure_buf.item())
        self.assertFalse(env.time_out_buf.item())


if __name__ == '__main__':
    unittest.main()
