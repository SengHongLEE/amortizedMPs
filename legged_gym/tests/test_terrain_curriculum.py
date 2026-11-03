"""Curriculum decisions independent of simulator stepping."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'legged_gym/utils/terrains/terrain_curriculum.py'


def options(**overrides):
    values = dict(curriculum=True, num_rows=10, difficulty_mode='linear',
                  row_difficulties=None, curriculum_min_level=0,
                  curriculum_max_level=None, curriculum_initial_level=1,
                  curriculum_promote_distance=2.5, curriculum_promote_ratio=.8,
                  curriculum_demote_ratio=.5, fixed_terrain_level=None,
                  max_init_terrain_level=None, terrain_length=8., terrain_width=8.,
                  boundary_margin=.6, spawn_xy_jitter=.1, border_size=4.)
    values.update(overrides)
    return SimpleNamespace(**values)


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('terrain_curriculum', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.curriculum_class = module.TerrainCurriculum

    def make(self, **kwargs):
        cfg = options(**kwargs)
        difficulties = np.tile(np.linspace(0, 1, cfg.num_rows)[:, None], (1, 2))
        c = self.curriculum_class(cfg, difficulties, 6, 'cpu')
        c.levels[:] = 4 if c.enabled else c.levels
        c.reset(torch.arange(6), torch.zeros(6, 2))
        return c

    def test_success_failure_and_hold(self):
        c = self.make()
        c.command_distance[:] = 3.
        c.command_displacement[:, 0] = 3.
        # success, slow, fall despite good progress, sideways, backwards, intermediate
        positions = torch.tensor([[2.7, 0], [.5, 0], [3., 0], [0., 3.], [-3., 0], [2., 0]])
        c.update(torch.arange(6), positions,
                 torch.tensor([False, False, True, False, False, False]), torch.ones(6))
        self.assertEqual(c.levels.tolist(), [5, 3, 3, 3, 3, 4])

    def test_stationary_command_and_empty_episodes_do_not_promote(self):
        c = self.make()
        c.update(torch.arange(6), torch.ones(6, 2) * 3, torch.zeros(6, dtype=torch.bool),
                 torch.ones(6))
        self.assertEqual(c.levels.tolist(), [4] * 6)
        c.update(torch.arange(6), torch.zeros(6, 2), torch.ones(6, dtype=torch.bool),
                 torch.zeros(6))
        self.assertEqual(c.levels.tolist(), [4] * 6)

    def test_spawn_offset_and_short_distance_do_not_count_as_success(self):
        c = self.make()
        c.reset(torch.arange(6), torch.tensor([[100., 20.]]).repeat(6, 1))
        c.command_distance[:] = 2.
        c.command_displacement[:, 0] = 2.
        c.update(torch.arange(6), c.start_xy + torch.tensor([2., 0.]),
                 torch.zeros(6, dtype=torch.bool), torch.ones(6))
        self.assertEqual(c.levels.tolist(), [4] * 6)

    def test_reaching_distance_without_command_tracking_does_not_promote(self):
        c = self.make()
        c.command_distance[:] = 5.
        c.command_displacement[:, 0] = 5.
        c.update(torch.arange(6), torch.tensor([[2.7, 0.]]).repeat(6, 1),
                 torch.zeros(6, dtype=torch.bool), torch.ones(6))
        self.assertEqual(c.levels.tolist(), [4] * 6)

    def test_safe_tile_crossing_is_credited_when_a_later_tile_causes_a_fall(self):
        c = self.make()
        c.command_distance[:] = 3.
        c.command_displacement[:, 0] = 3.
        c.observe_position(torch.tensor([[2.8, 0.]] * 6),
                           torch.tensor([True, False, False, False, False, False]))
        c.command_distance[:] = 5.
        c.command_displacement[:, 0] = 5.
        c.update(torch.arange(6), torch.tensor([[3., 0.]] * 6),
                 torch.ones(6, dtype=torch.bool), torch.ones(6))
        self.assertEqual(c.levels.tolist(), [5, 3, 3, 3, 3, 3])
        c.reset(torch.tensor([0]), torch.zeros(1, 2))
        self.assertEqual(c.max_progress[0].item(), 0.)

    def test_second_band_can_be_required_before_advanced_curriculum_promotion(self):
        c = self.make(curriculum_promote_distance=4.25)
        c.command_distance[:] = 5.
        c.command_displacement[:, 0] = 5.
        c.observe_position(torch.tensor([[4.3, 0.]] * 6),
                           torch.tensor([True, False, False, False, False, False]))
        c.update(torch.arange(6), torch.tensor([[4.5, 0.]] * 6),
                 torch.ones(6, dtype=torch.bool), torch.ones(6))
        self.assertEqual(c.levels.tolist(), [5, 3, 3, 3, 3, 3])

    def test_levels_clamp_and_partial_reset(self):
        c = self.make(curriculum_min_level=1, curriculum_max_level=5)
        c.levels[:] = torch.tensor([5, 1, 3, 3, 3, 3])
        c.command_distance[:] = 3.
        c.command_displacement[:, 0] = 3.
        c.update(torch.tensor([0, 1]), torch.tensor([[3., 0], [0., 0]]),
                 torch.tensor([False, True]), torch.ones(2))
        self.assertEqual(c.levels.tolist(), [5, 1, 3, 3, 3, 3])
        c.reset(torch.tensor([0, 1]), torch.ones(2, 2))
        self.assertEqual(c.command_distance.tolist(), [0, 0, 3, 3, 3, 3])
        self.assertTrue(torch.equal(c.start_xy[:2], torch.ones(2, 2)))

    def test_command_resampling_integrates_actual_distance_and_heading(self):
        c = self.make()
        commands = torch.zeros(6, 4)
        commands[:, 0] = .3
        quat = torch.tensor([0., 0., 2**-.5, 2**-.5]).repeat(6, 1)
        c.accumulate(commands, quat, 5.)
        commands[:, 0] = .8
        c.accumulate(commands, quat, 5.)
        torch.testing.assert_close(c.command_distance, torch.full((6,), 5.5))
        torch.testing.assert_close(c.command_displacement[:, 1], torch.full((6,), 5.5))

    def test_disabled_fixed_or_random_levels_never_adapt(self):
        for fixed in (None, 7):
            c = self.make(curriculum=False, fixed_terrain_level=fixed)
            before = c.levels.clone()
            if fixed is not None:
                self.assertEqual(before.tolist(), [fixed] * 6)
            c.update(torch.arange(6), torch.ones(6, 2) * 3,
                     torch.ones(6, dtype=torch.bool), torch.ones(6))
            self.assertTrue(torch.equal(before, c.levels))

    def test_initial_levels_and_playback(self):
        cfg = options()
        difficulty = np.linspace(0, 1, 10)[:, None]
        c = self.curriculum_class(cfg, difficulty, 128, 'cpu')
        self.assertEqual(set(c.levels.tolist()), {0, 1})
        cfg.fixed_terrain_level = 9
        c = self.curriculum_class(cfg, difficulty, 8, 'cpu', play=True)
        self.assertFalse(c.enabled)
        self.assertEqual(c.levels.tolist(), [9] * 8)

    def test_invalid_settings_and_unsorted_rows_fail_early(self):
        for kwargs in (dict(curriculum='false'), dict(curriculum_initial_level=10),
                       dict(curriculum_min_level=2), dict(fixed_terrain_level=10),
                       dict(curriculum_promote_distance=8.),
                       dict(curriculum_promote_ratio=float('nan')),
                       dict(curriculum_demote_ratio=.9),
                       dict(difficulty_mode='random')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            self.curriculum_class(options(), np.arange(10)[::-1, None], 2, 'cpu')

    def test_fall_at_boundary_is_failure_not_timeout(self):
        from test_interoceptive_rewards import load_class
        Environment = load_class('mp_proprio_terrain.py', 'MotorPrimitives', {'check_termination'})
        env = Environment()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(mesh_type='heightfield', boundary_margin=.6))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(4, 13)
        env.root_states[:, 2] = .32
        env.root_states[:2, 0] = 3.5
        env.root_states[2, 2] = .1
        env.env_origins = torch.zeros(4, 3)
        env.contact_forces = torch.zeros(4, 1, 3)
        env.contact_forces[0, 0, 2] = 2.
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.tensor([1, 1, 101, 101])
        env.max_episode_length = 100
        env._get_terrain_heights = lambda xy: torch.zeros(len(xy))
        env.check_termination()
        self.assertEqual(env.terrain_failure_buf.tolist(), [True, False, True, False])
        self.assertEqual(env.time_out_buf.tolist(), [False, True, False, True])
        self.assertTrue(env.reset_buf.all())

    def test_mp_full_crosses_tile_before_episode_ends(self):
        from test_interoceptive_rewards import load_class
        Environment = load_class('mp_full.py', 'MotorPrimitives', {'check_termination'})
        env = Environment()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=10, num_cols=2, boundary_margin=.6))
        env.terrain = SimpleNamespace(env_length=8., env_width=8.)
        env.root_states = torch.zeros(4, 13)
        env.root_states[:, 2] = .32
        env.root_states[:, 0] = torch.tensor([9., 79.5, 9., 9.])
        env.root_states[:, 1] = 4.
        env.env_origins = torch.tensor([[4., 4., 0.]] * 4)
        env.contact_forces = torch.zeros(4, 1, 3)
        env.contact_forces[2, 0, 2] = 2.
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.tensor([10, 10, 10, 101])
        env.max_episode_length = 100
        env._get_terrain_heights = lambda xy: torch.zeros(len(xy))

        env.check_termination()

        self.assertEqual(env.reset_buf.tolist(), [False, True, True, True])
        self.assertEqual(env.time_out_buf.tolist(), [False, True, False, True])
        self.assertEqual(env.terrain_failure_buf.tolist(), [False, False, True, False])

    def test_mp_full_episode_terrain_metrics_use_completed_envs_and_end_position(self):
        from test_interoceptive_rewards import load_class
        Environment = load_class('mp_full.py', 'MotorPrimitives', {'_terrain_episode_metrics'})
        env = Environment()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(num_rows=3))
        env.terrain = SimpleNamespace(env_length=8., terrain_names=np.array([['gap', 'hurdle']]))
        env.root_states = torch.zeros(3, 13)
        env.root_states[:, 0] = torch.tensor([9., 19., 4.])
        env.terrain_levels = torch.tensor([1, 1, 0])
        env.terrain_types = torch.tensor([0, 1, 0])
        env.device = 'cpu'
        env.terrain_difficulties = torch.tensor([[.1, .2], [.5, .6], [.9, 1.]])
        env.terrain_failure_buf = torch.tensor([False, True, False])
        ids = torch.tensor([0, 1])

        result = env._terrain_episode_metrics(ids, torch.tensor([0, 2]))

        self.assertAlmostEqual(result['terrain_start_level'].item(), 1.)
        self.assertAlmostEqual(result['terrain_start_difficulty'].item(), .55)
        self.assertAlmostEqual(result['terrain_end_row'].item(), 1.5)
        self.assertAlmostEqual(result['terrain_promoted_rate'].item(), .5)
        self.assertAlmostEqual(result['terrain_demoted_rate'].item(), .5)
        self.assertAlmostEqual(result['terrain_gap_promoted_rate'].item(), 1.)
        self.assertAlmostEqual(result['terrain_hurdle_demoted_rate'].item(), 1.)
        self.assertAlmostEqual(result['terrain_gap_failure_rate'].item(), 0.)
        self.assertAlmostEqual(result['terrain_hurdle_failure_rate'].item(), 1.)
        self.assertAlmostEqual(result['terrain_promoted_after_failure_rate'].item(), 0.)
        env.terrain_failure_buf[0] = True
        result = env._terrain_episode_metrics(ids, torch.tensor([0, 2]))
        self.assertAlmostEqual(result['terrain_promoted_after_failure_rate'].item(), .5)
        self.assertAlmostEqual(result['terrain_gap_promoted_after_failure_rate'].item(), 1.)

    def test_mp_full_gap_curriculum_keeps_nonzero_exposure(self):
        from test_interoceptive_rewards import load_class
        Environment = load_class('mp_full.py', 'MotorPrimitives', {
            '_clamp_hazard_curriculum_levels'})
        env = Environment()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            curriculum=True, curriculum_gap_min_level=1))
        env._gap_columns = torch.tensor([False, True, False])
        env.terrain_types = torch.tensor([0, 1, 1, 2])
        env.terrain_levels = torch.tensor([0, 0, 1, 0])
        env._clamp_hazard_curriculum_levels(torch.arange(4))
        self.assertEqual(env.terrain_levels.tolist(), [0, 1, 1, 0])
        env.cfg.terrain.curriculum = False
        env.terrain_levels[1] = 0
        env._clamp_hazard_curriculum_levels(torch.arange(4))
        self.assertEqual(env.terrain_levels[1].item(), 0)

    def test_mp_full_reserves_half_gap_columns_for_medium_difficulty(self):
        from test_interoceptive_rewards import load_class
        Environment = load_class('mp_full.py', 'MotorPrimitives', {
            '_clamp_hazard_curriculum_levels'})
        env = Environment()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            curriculum=True, curriculum_gap_min_level=1,
            curriculum_gap_medium_min_level=4,
            curriculum_gap_medium_columns=2))
        env._gap_columns = torch.tensor([False, True, True, False, True, True])
        env.terrain_types = torch.tensor([1, 2, 4, 5])
        env.terrain_levels = torch.zeros(4, dtype=torch.long)
        env._clamp_hazard_curriculum_levels(torch.arange(4))
        self.assertEqual(env.terrain_levels.tolist(), [1, 1, 4, 4])


if __name__ == '__main__':
    unittest.main()
