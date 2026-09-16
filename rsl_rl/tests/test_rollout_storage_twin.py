import os
import sys
import unittest

import torch


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.storage.rollout_storage_event import RolloutStorage_twin


def make_start(env_ids, step, obs_dim=2, action_dim=1):
    env_ids = torch.tensor(env_ids, dtype=torch.long)
    batch_size = env_ids.numel()
    transition = RolloutStorage_twin.StartTransition()
    transition.observations = torch.stack((
        env_ids.float(),
        torch.full((batch_size,), float(step)),
    ), dim=-1)
    transition.critic_observations = transition.observations.clone()
    transition.actions = torch.full((batch_size, action_dim), float(step))
    transition.values = torch.zeros(batch_size, 1)
    transition.actions_log_prob = torch.zeros(batch_size)
    transition.action_mean = torch.zeros(batch_size, action_dim)
    transition.action_sigma = torch.ones(batch_size, action_dim)
    transition.env_ids = env_ids
    transition.step_ids = torch.full(
        (batch_size,),
        step,
        dtype=torch.long,
    )
    return transition


def make_end(rewards, dones=None, obs_dim=2):
    batch_size = len(rewards)
    transition = RolloutStorage_twin.EndTransition()
    transition.rewards = torch.tensor(rewards, dtype=torch.float)
    if dones is None:
        dones = [False] * batch_size
    transition.dones = torch.tensor(dones, dtype=torch.bool)
    transition.critic_observations = torch.full(
        (batch_size, obs_dim),
        7.0,
    )
    return transition


class RolloutStorageTwinTest(unittest.TestCase):
    def make_storage(self, num_envs=3, num_steps=2):
        return RolloutStorage_twin(
            num_envs,
            num_steps,
            [2],
            [None],
            [1],
            "cpu",
        )

    def add_completed(self, storage, env_ids, step, rewards, dones=None):
        storage.add_start_transitions(make_start(env_ids, step))
        storage.add_end_transitions(
            torch.tensor(env_ids, dtype=torch.long),
            make_end(rewards, dones),
        )

    def test_accepts_irregular_event_counts_and_vectorizes_gae(self):
        storage = self.make_storage()
        self.add_completed(storage, [0, 1, 2], 0, [1.0, 10.0, 100.0])
        self.add_completed(storage, [0, 1], 1, [2.0, 20.0])

        storage.add_start_transitions(make_start([0], 2))
        self.assertFalse(storage.ready)
        storage.add_end_transitions(
            torch.tensor([0]),
            make_end([3.0]),
        )

        self.assertTrue(storage.ready)
        storage.compute_returns(torch.zeros(3, 1), gamma=1.0, lam=1.0)
        expected = torch.tensor([6.0, 30.0, 100.0, 5.0, 20.0, 3.0])
        self.assertTrue(torch.allclose(storage.returns[:, 0], expected))

    def test_capacity_overflow_keeps_only_available_events(self):
        storage = self.make_storage(num_envs=2, num_steps=2)
        self.add_completed(storage, [0, 1], 0, [1.0, 1.0])

        accepted = storage.add_start_transitions(make_start([0, 1, 0], 1))

        self.assertEqual(storage.counter, 4)
        self.assertTrue(torch.equal(accepted, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(
            storage.env_ids,
            torch.tensor([0, 1, 0, 1]),
        ))

    def test_clear_resets_lifecycle_without_per_env_history(self):
        storage = self.make_storage(num_envs=2, num_steps=1)
        self.add_completed(storage, [0, 1], 0, [1.0, 1.0])
        self.assertTrue(storage.ready)

        storage.clear()

        self.assertEqual(storage.counter, 0)
        self.assertEqual(storage.completed_count, 0)
        self.assertFalse(storage.ready)
        self.assertTrue(torch.equal(
            storage.env_ptrs,
            torch.full((2,), -1, dtype=torch.long),
        ))
        self.assertFalse(hasattr(storage, "env_indices"))

    def test_endpoints_update_bootstrap_observations(self):
        storage = self.make_storage(num_envs=3, num_steps=1)
        storage.add_start_transitions(make_start([2], 0))
        endpoint = make_end([1.0])
        endpoint.critic_observations[:] = torch.tensor([4.0, 5.0])

        storage.add_end_transitions(torch.tensor([2]), endpoint)

        self.assertTrue(torch.equal(
            storage.bootstrap_observations[2],
            torch.tensor([4.0, 5.0]),
        ))


if __name__ == "__main__":
    unittest.main()
