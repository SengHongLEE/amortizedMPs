import os
import sys
import unittest

import torch
import torch.nn as nn
from torch.distributions import Normal


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.algorithms.ppo_twin import PPOtwin


class TinyActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.std = nn.Parameter(torch.ones(1))
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, **kwargs):
        mean = observations[:, :1] * 0.0 + self.bias
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations, **kwargs):
        return observations[:, :1] + self.bias * 0.0

    def reset(self, dones=None):
        pass


class PPOtwinTest(unittest.TestCase):
    def make_algorithm(self, num_envs=3, num_steps=1):
        algorithm = PPOtwin(
            TinyActorCritic(),
            num_learning_epochs=1,
            num_mini_batches=1,
            schedule="fixed",
            device="cpu",
        )
        algorithm.init_storage(
            num_envs,
            num_steps,
            [2],
            [None],
            [1],
        )
        return algorithm

    def test_process_env_step_uses_pointer_duration(self):
        algorithm = self.make_algorithm()
        observations = torch.zeros(3, 2)
        critic_observations = observations.clone()
        env_ids = torch.tensor([2, 0])
        start_steps = torch.tensor([4, 8])
        algorithm.act(
            env_ids,
            start_steps,
            observations,
            critic_observations,
        )

        accumulated_rewards = torch.tensor([4.0, 0.0, 12.0])
        endpoint_observations = torch.zeros_like(observations)
        algorithm.process_env_step(
            10,
            env_ids,
            accumulated_rewards,
            torch.zeros(3, dtype=torch.bool),
            endpoint_observations,
            {},
        )

        self.assertTrue(torch.allclose(
            algorithm.storage.rewards[:2, 0],
            torch.tensor([2.0, 2.0]),
        ))

    def test_process_env_step_ignores_untracked_overflow_events(self):
        algorithm = self.make_algorithm(num_envs=3, num_steps=1)
        observations = torch.zeros(3, 2)
        first_env_ids = torch.tensor([0, 1])
        algorithm.act(
            first_env_ids,
            torch.zeros(2, dtype=torch.long),
            observations,
            observations,
        )
        algorithm.process_env_step(
            1,
            first_env_ids,
            torch.ones(3),
            torch.zeros(3, dtype=torch.bool),
            observations,
            {},
        )

        overflow_env_ids = torch.tensor([2, 0])
        algorithm.act(
            overflow_env_ids,
            torch.ones(2, dtype=torch.long),
            observations,
            observations,
        )
        algorithm.process_env_step(
            2,
            overflow_env_ids,
            torch.ones(3),
            torch.zeros(3, dtype=torch.bool),
            observations,
            {},
        )

        self.assertEqual(algorithm.storage.completed_count, 3)
        self.assertTrue(algorithm.storage.ready)

    def test_compute_returns_bootstraps_from_transition_endpoint(self):
        algorithm = self.make_algorithm(num_envs=1, num_steps=1)
        observations = torch.zeros(1, 2)
        env_ids = torch.tensor([0])
        algorithm.act(
            env_ids,
            torch.tensor([0]),
            observations,
            observations,
        )

        endpoint_observations = torch.tensor([[5.0, 0.0]])
        algorithm.process_env_step(
            1,
            env_ids,
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            endpoint_observations,
            {},
        )
        algorithm.compute_returns(torch.tensor([[999.0, 0.0]]))

        self.assertTrue(torch.allclose(
            algorithm.storage.returns[0],
            torch.tensor([1.0 + algorithm.gamma * 5.0]),
        ))
        self.assertTrue(torch.isfinite(
            algorithm.storage.advantages[0]
        ).all())

    def test_non_reservoir_act_returns_action_tensor(self):
        algorithm = self.make_algorithm(num_envs=2, num_steps=1)
        observations = torch.zeros(2, 2)

        actions = algorithm.act(
            torch.tensor([0, 1]),
            torch.zeros(2, dtype=torch.long),
            observations,
            observations,
        )

        self.assertIsInstance(actions, torch.Tensor)
        self.assertEqual(actions.shape, (2, 1))

    def test_non_reservoir_update_returns_losses(self):
        algorithm = self.make_algorithm(num_envs=2, num_steps=1)
        observations = torch.zeros(2, 2)
        env_ids = torch.tensor([0, 1])
        algorithm.act(
            env_ids,
            torch.zeros(2, dtype=torch.long),
            observations,
            observations,
        )
        algorithm.process_env_step(
            1,
            env_ids,
            torch.tensor([1.0, 2.0]),
            torch.zeros(2, dtype=torch.bool),
            observations,
            {},
        )
        algorithm.compute_returns(observations)

        losses = algorithm.update()

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(
            isinstance(loss, float)
            for loss in losses
        ))


if __name__ == "__main__":
    unittest.main()
