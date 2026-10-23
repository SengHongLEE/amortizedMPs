"""Shared high/low task rewards must not double-count episode diagnostics."""
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
from test_interoceptive_rewards import load_class

Env = load_class('mp_proprio_terrain.py', 'MotorPrimitives',
                 {'_prepare_reward_function', 'compute_reward'})


class AlignmentTests(unittest.TestCase):
    def test_saved_task_aligned_scales_survive_run_reload(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('config_loader', root / 'legged_gym/utils/load_config.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        cfg = load_class('mp_proprio_terrain_config.py', 'MotorPrimitivesCfg')()
        names = list(cfg.rewards.modulation_param_rewards)
        values = {name: getattr(cfg.rewards.scales, name) * .75 for name in names}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({
                'seed': 7, 'rewards': {'primitive_selection_rewards': names,
                                      'primitive_selection_scales': values}}))
            restored, _ = loader.load_run_config(cfg, SimpleNamespace(seed=1), directory)
        for name, value in values.items():
            self.assertEqual(getattr(restored.rewards.primitive_selection_scales, name, None), value)

    def test_task_rewards_match_and_each_log_has_one_contribution(self):
        env = Env()
        env.device, env.num_envs, env.dt = 'cpu', 2, .02
        env.reward_scales = dict(tracking_lin_vel=3., orientation=-2.)
        env.rew_buf = torch.zeros(2)
        env.primitive_selection_rew_buf = torch.zeros(2)
        env.modulation_param_rew_buf = torch.zeros(2)
        env._reward_tracking_lin_vel = lambda: torch.tensor([1., .2])
        env._reward_orientation = lambda: torch.tensor([.1, .5])
        env.cfg = SimpleNamespace(
            env=SimpleNamespace(play=False),
            rewards=SimpleNamespace(
                primitive_selection_rewards=['tracking_lin_vel', 'orientation'],
                primitive_selection_scales=SimpleNamespace(tracking_lin_vel=3., orientation=-2.),
                modulation_param_rewards=['tracking_lin_vel', 'orientation'],
                task_rewards=['tracking_lin_vel'], only_positive_rewards=False))
        env._prepare_reward_function()
        total, high, low = env.compute_reward()
        torch.testing.assert_close(high, low)
        torch.testing.assert_close(total, low)
        expected_tracking = torch.tensor([1., .2]) * 3. * .02
        torch.testing.assert_close(env.episode_sums['tracking_lin_vel'], expected_tracking)
        torch.testing.assert_close(env.episode_sums['selection_tracking_lin_vel'], expected_tracking)
        torch.testing.assert_close(high, torch.tensor([2.8, -.4]) * .02)


if __name__ == '__main__':
    unittest.main()
