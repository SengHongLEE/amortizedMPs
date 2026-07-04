import os
import sys
import unittest

import torch
import torch.nn as nn


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.modules.actor_critic_twin import ActorCriticTwin


class ActorCriticTwinTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = ActorCriticTwin(
            5,
            5,
            2,
            actor_hidden_dims=[8, 4],
            critic_hidden_dims=[8, 4],
        )

    def test_uses_checkpoint_compatible_parameter_names(self):
        state_keys = set(self.model.state_dict().keys())

        self.assertIsInstance(self.model.actor, nn.Sequential)
        self.assertIn("std", state_keys)
        self.assertIn("actor.0.weight", state_keys)
        self.assertIn("critic.0.weight", state_keys)
        self.assertNotIn("actor.network.0.weight", state_keys)

    def test_checkpoint_strict_roundtrip_preserves_legacy_keys(self):
        checkpoint = self.model.state_dict()
        restored = ActorCriticTwin(
            5,
            5,
            2,
            actor_hidden_dims=[8, 4],
            critic_hidden_dims=[8, 4],
        )

        incompatible = restored.load_state_dict(checkpoint, strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            set(restored.state_dict()),
            set(checkpoint),
        )
        for key, expected in checkpoint.items():
            self.assertTrue(torch.equal(restored.state_dict()[key], expected))

    def test_legacy_inference_helpers_delegate_to_single_actor(self):
        observations = torch.randn(3, 5)
        expected = self.model.act_inference(observations)

        self.assertTrue(torch.equal(
            self.model.act_mu_inference(observations),
            expected,
        ))
        self.assertTrue(torch.equal(
            self.model.act_omega_inference(observations),
            expected,
        ))

    def test_legacy_distribution_helpers_use_single_distribution(self):
        observations = torch.randn(3, 5)
        actions = self.model.act_mu(observations)

        self.assertEqual(actions.shape, (3, 2))
        self.assertEqual(self.model.mu_actions_mean.shape, (3, 2))
        self.assertEqual(self.model.get_mu_actions_log_prob(actions).shape, (3,))


if __name__ == "__main__":
    unittest.main()
