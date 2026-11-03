"""Reward and command sampling checks for condition-dependent gait selection."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_interoceptive_rewards import load_class


CONFIG = load_class('mp_full_config.py', 'MotorPrimitivesCfg')
REWARDS = load_class('mp_full.py', 'MotorPrimitives', {
    '_pi_g_psi', '_reward_pi_g_cot', '_reward_pi_g_context_guidance',
    '_primitive_context_targets',
    '_reward_tracking_lin_vel',
})


def load_speed_sampler():
    path = Path(__file__).resolve().parents[1] / 'legged_gym/envs/a1/mp_full.py'
    node = next((item for item in ast.parse(path.read_text()).body
                 if isinstance(item, ast.FunctionDef) and item.name == 'sample_speed_bands'), None)
    if node is None:
        return None
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sample_speed_bands']


class GaitSelectionTrainingTests(unittest.TestCase):
    def test_speed_bands_cover_low_flat_and_fast_hazard_commands(self):
        sampler = load_speed_sampler()
        self.assertIsNotNone(sampler)
        torch.manual_seed(17)
        ordinary = sampler(CONFIG.commands.lin_vel_x_bands, 20000, 'cpu')
        hazard = sampler(CONFIG.commands.obstacle_lin_vel_x_bands, 20000, 'cpu')
        self.assertTrue(((ordinary >= .3) & (ordinary <= 2.5)).all())
        self.assertTrue(((hazard >= .8) & (hazard <= 2.2)).all())
        self.assertAlmostEqual((ordinary < .8).float().mean().item(), .50, delta=.02)
        self.assertAlmostEqual((ordinary >= 1.5).float().mean().item(), .20, delta=.02)
        self.assertAlmostEqual((hazard >= 1.5).float().mean().item(), .40, delta=.02)
        self.assertEqual(len(CONFIG.terrain.terrain_columns), CONFIG.terrain.num_cols)
        self.assertIn('flat', CONFIG.terrain.terrain_columns)
        self.assertIn('gap_continuous', CONFIG.terrain.terrain_columns)
        self.assertIn('gap_single', CONFIG.terrain.terrain_columns)
        self.assertEqual(CONFIG.terrain.terrain_params['hurdle']['spacing'], 4.)

    def test_efficiency_reward_has_range_and_requires_speed_tracking(self):
        env = REWARDS()
        env.cfg = SimpleNamespace(rewards=CONFIG.rewards)
        env.commands = torch.tensor([[1.5, 0.], [1.5, 0.], [1.5, 0.]])
        env.base_lin_vel = torch.tensor([[1.5, 0.], [0., 0.], [1.5, 0.]])
        env._pi_g_cost_of_transport = lambda: torch.tensor([2.8, 2.8, 10.])
        reward = env._reward_pi_g_cot()
        self.assertGreater(reward[0].item(), .3)
        self.assertLess(reward[1].item(), .01)
        self.assertLess(reward[2].item(), .01)

    def test_context_guidance_is_narrow_and_requires_tracking(self):
        self.assertIn('pi_g_context_guidance', CONFIG.rewards.primitive_selection_rewards)
        self.assertGreaterEqual(CONFIG.rewards.primitive_selection_scales.pi_g_context_guidance, 20.)
        env = REWARDS()
        env.cfg = SimpleNamespace(rewards=CONFIG.rewards)
        env.terrain_types = torch.tensor([0, 0, 1, 2, 0, 0, 0, 1, 3, 1, 1, 4, 5, 6])
        env._flat_columns = torch.tensor([True, False, False, False, False, False, False])
        env._hurdle_columns = torch.tensor([False, True, False, False, False, False, False])
        env._gap_continuous_columns = torch.tensor([False, False, True, False, False, False, False])
        env._gap_single_columns = torch.tensor([False, False, False, True, False, False, False])
        env._pit_columns = torch.tensor([False, False, False, False, True, False, False])
        env._stairs_columns = torch.tensor([False, False, False, False, False, True, False])
        env.commands = torch.tensor([[.4, 0.], [1.5, 0.], [.8, 0.],
                                     [2.2, 0.], [.8, 0.], [1.5, 0.],
                                     [.4, 0.], [.8, 0.], [2.2, 0.],
                                     [1.5, 0.], [2.2, 0.], [.8, 0.], [1.5, 0.],
                                     [1.5, 0.]])
        env.actions = torch.tensor([[0.], [1.], [3.], [2.], [3.], [1.],
                                    [2.], [2.], [2.], [2.], [1.], [2.], [2.], [2.]])
        env.base_lin_vel = env.commands.clone()
        env.base_lin_vel[5] = 0.
        self.assertEqual(env._primitive_context_targets().tolist(),
                         [0, 1, -1, 2, -1, 1, 0, -1, 2, -1, -1, -1, -1, -1])
        reward = env._reward_pi_g_context_guidance()
        self.assertTrue((reward[:2] > .95).all())
        self.assertEqual(reward[2].item(), 0.)
        self.assertGreater(reward[3].item(), .95)
        self.assertEqual(reward[4].item(), 0.)
        self.assertLess(reward[5].item(), .01)
        self.assertLess(reward[6].item(), -.49)
        self.assertEqual(reward[7].item(), 0.)
        self.assertGreater(reward[8].item(), .95)
        self.assertTrue((reward[9:11] == 0).all())
        self.assertTrue((reward[11:] == 0).all())
        for gait in (0, 1, 3):
            env.actions[12, 0] = gait
            self.assertEqual(env._reward_pi_g_context_guidance()[12].item(), 0.)


if __name__ == '__main__':
    unittest.main()
