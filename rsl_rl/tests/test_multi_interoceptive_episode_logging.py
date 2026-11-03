"""Optional terrain metrics use the episodes that produced them."""
import unittest
from types import SimpleNamespace

import torch

from rsl_rl.runners.multi_interoceptive_policy_runner import (
    MultiInteroceptivePolicyRunner, aggregate_episode_metrics)


class EpisodeLoggingTests(unittest.TestCase):
    def test_timestep_log_counts_physics_steps_between_event_updates(self):
        runner = MultiInteroceptivePolicyRunner.__new__(MultiInteroceptivePolicyRunner)
        runner.env = SimpleNamespace(num_envs=16)
        runner.tot_timesteps = 0
        runner.current_step = 120
        self.assertEqual(runner._record_collected_env_steps(), 120)
        self.assertEqual(runner.tot_timesteps, 1920)
        runner.current_step = 192
        self.assertEqual(runner._record_collected_env_steps(), 72)
        self.assertEqual(runner.tot_timesteps, 3072)

    def test_missing_terrain_keys_are_weighted_by_matching_counts(self):
        infos = [
            {'terrain_start_level': 2., 'terrain_gap_completed_count': 2.,
             'terrain_gap_start_level': 2., 'terrain_gap_failure_rate': .5,
             'terrain_hurdle_completed_count': 0.},
            {'terrain_start_level': 6., 'terrain_gap_completed_count': 1.,
             'terrain_gap_start_level': 6., 'terrain_gap_failure_rate': 1.,
             'terrain_hurdle_completed_count': 3.,
             'terrain_hurdle_start_level': 4.},
        ]
        result = aggregate_episode_metrics(infos, [5, 5], 'cpu')
        torch.testing.assert_close(result['terrain_start_level'], torch.tensor(4.))
        torch.testing.assert_close(result['terrain_gap_start_level'], torch.tensor(10. / 3))
        torch.testing.assert_close(result['terrain_hurdle_start_level'], torch.tensor(4.))
        torch.testing.assert_close(result['terrain_gap_failure_rate'], torch.tensor(2. / 3))
        torch.testing.assert_close(result['terrain_gap_completed_count'], torch.tensor(3.))
        torch.testing.assert_close(result['terrain_hurdle_completed_count'], torch.tensor(3.))


if __name__ == '__main__':
    unittest.main()
