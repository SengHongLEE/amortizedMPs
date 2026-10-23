"""Checkpoint update counts must advance even when logging is disabled."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


class IterationTests(unittest.TestCase):
    def make(self, logging):
        r = MultiInteroceptivePolicyRunner.__new__(MultiInteroceptivePolicyRunner)
        r.device, r.log_dir, r.writer = 'cpu', ('unused' if logging else None), object()
        r.current_learning_iteration, r.current_step, r.save_interval = 7, 0, 1
        r.primitive_selection_reservoir_states = r.modulation_param_reservoir_states = None
        r.primitive_selection_update_cycle = r.modulation_param_update_cycle = torch.ones(2, dtype=torch.long)
        obs, critic = torch.zeros(2, 1, 3), torch.zeros(2, 4)
        r.env = SimpleNamespace(num_envs=2, episode_length_buf=torch.ones(2, dtype=torch.long),
                                get_observations=lambda: obs, get_privileged_observations=lambda: critic)
        r.training_rollout = lambda *args: (obs, critic, [torch.zeros(2)] * 3,
                                            torch.zeros(2, dtype=torch.bool), {})
        steps = []
        def update():
            steps.append(1)
            if len(steps) > 3:
                raise RuntimeError('learn did not stop after requested updates')
            return 0., 0.
        def alg(ready):
            return SimpleNamespace(storage=SimpleNamespace(pending=torch.zeros(2, dtype=torch.bool),
                                                            ready=ready),
                                   actor_critic=SimpleNamespace(train=lambda: None),
                                   compute_returns=lambda *args: None, update=update)
        r.primitive_selection_alg, r.modulation_param_alg = alg(False), alg(True)
        r.log = r.final_log = lambda *args: None
        saved = []
        r.save_model = lambda path, iteration: saved.append((iteration, r.current_learning_iteration))
        return r, steps, saved

    def test_continued_update_count_is_saved(self):
        runner, steps, saved = self.make(True)
        runner.learn(2)
        self.assertEqual(len(steps), 2)
        self.assertEqual(runner.current_learning_iteration, 9)
        self.assertEqual(saved, [(8, 8), (9, 9), (9, 9)])

    def test_no_logging_still_finishes_without_saving(self):
        runner, steps, saved = self.make(False)
        runner.learn(2)
        self.assertEqual(len(steps), 2)
        self.assertEqual(runner.current_learning_iteration, 9)
        self.assertFalse(saved)


if __name__ == '__main__':
    unittest.main()
