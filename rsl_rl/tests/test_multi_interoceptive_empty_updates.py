"""A scheduling tick without decisions must hold actions, not call an actor."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


class EmptyDecisionTests(unittest.TestCase):
    def make_runner(self):
        obj = MultiInteroceptivePolicyRunner.__new__(MultiInteroceptivePolicyRunner)
        obj.device, obj.num = 'cpu', 0
        obj.actions = torch.ones(2, 17)
        obj.actions[:, 0] = 2
        obj.last_actions = obj.actions.clone()
        obj.primitive_selections = torch.full((2,), 2, dtype=torch.long)
        obj.last_primitive_selections = obj.primitive_selections.clone()
        obj.modulation_params = obj.actions[:, 1:].clone()
        obj.last_modulation_params = obj.modulation_params.clone()
        obj.primitive_selection_update_cycle = torch.full((2,), 25)
        obj.modulation_param_update_cycle = torch.full((2,), 2)
        obj.primitive_selection_reservoir_states = None
        obj.modulation_param_reservoir_states = None
        obs, critic = torch.zeros(2, 5, 124), torch.zeros(2, 131)

        def no_decision(*args):
            self.fail('actor/algorithm must not run when no environment needs a decision')

        obj.env = SimpleNamespace(
            episode_length_buf=torch.ones(2, dtype=torch.long),
            primitive_diff=torch.zeros(2), param_actions_diff=torch.zeros(2),
            step=lambda actions: (
                obs, critic, [torch.zeros(2)] * 3, torch.zeros(2, dtype=torch.bool), {}))
        algo = SimpleNamespace(
            actor_critic=SimpleNamespace(is_reservoir=False),
            act=no_decision, clear_start_transition=lambda: None)
        obj.primitive_selection_alg = obj.modulation_param_alg = algo
        return obj, obs, critic, no_decision

    def test_training_holds_previous_actions_on_empty_tick(self):
        obj, obs, critic, _ = self.make_runner()
        previous = obj.actions.clone()
        obj.training_rollout(obs, critic, 1)
        torch.testing.assert_close(obj.actions, previous)

    def test_inference_holds_previous_actions_on_empty_tick(self):
        obj, obs, _, actor = self.make_runner()
        previous = obj.actions.clone()
        obj.inference_rollout(obs, actor, actor)
        torch.testing.assert_close(obj.actions, previous)


if __name__ == '__main__':
    unittest.main()
