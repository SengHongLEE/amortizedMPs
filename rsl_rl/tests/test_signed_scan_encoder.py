import unittest

import torch

from rsl_rl.modules.actor_critic import ActorCritic, ActorCriticDiscrete


class SignedScanEncoderTest(unittest.TestCase):
    def test_both_policies_accept_compact_signed_scan_and_train_encoders(self):
        shape = (2, 16, 5)
        high = ActorCriticDiscrete(
            262, 269, 4, 4,
            high_actor_type='lif_reservoir_snn_single_output',
            reservoir_dim=8, readout_hidden_dims=(8,), num_snn_steps=2,
            terrain_scan_shape=shape, terrain_encoder_dim=16)
        low = ActorCritic(
            262, 269, 18, actor_type='snn',
            actor_hidden_dims=(8,), num_snn_steps=2,
            terrain_scan_shape=shape, terrain_encoder_dim=16)
        history = torch.zeros(2, 5, 262)
        history[1, :, -160:-80] = -0.5
        high_logits, state = high.actor(history, torch.zeros(2, high.actor.reservoir_state_dim))
        low_actions = low.actor(history[:, -1])
        self.assertEqual(tuple(high_logits.shape), (2, 4))
        self.assertEqual(tuple(low_actions.shape), (2, 18))
        self.assertEqual(tuple(state.shape), (2, high.actor.reservoir_state_dim))
        (high_logits.sum() + low_actions.sum()).backward()
        for actor in (high.actor, low.actor):
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for parameter in actor.terrain_encoder.parameters()))

    def test_selector_ppo_replay_uses_the_same_terrain_head(self):
        high = ActorCriticDiscrete(
            262, 269, 4, 4,
            high_actor_type='lif_reservoir_snn_single_output',
            reservoir_dim=8, readout_hidden_dims=(8,), num_snn_steps=2,
            terrain_scan_shape=(2, 16, 5), terrain_encoder_dim=16)
        with torch.no_grad():
            high.actor.terrain_readout_head[-1].weight.normal_(std=0.1)
        obs = torch.randn(3, 5, 262)
        logits, updated = high.actor(obs, torch.zeros(3, high.actor.reservoir_state_dim))
        high.act_for_ppo_update(obs[:, -1], updated.detach())
        torch.testing.assert_close(high.distribution_1.probs, logits.softmax(-1))
        loss = -high.distribution_1.log_prob(torch.tensor([0, 1, 2])).sum()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in high.actor.terrain_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
