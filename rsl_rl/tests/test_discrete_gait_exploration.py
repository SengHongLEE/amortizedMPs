"""PPO gait exploration must retain rare gait support without changing argmax inference."""
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from rsl_rl.modules.actor_critic import ActorCriticDiscrete
from rsl_rl.algorithms.ppo_event import context_guidance_loss
from rsl_rl.storage.rollout_storage_event import RolloutStorage_event_discrete


class FixedLogits(nn.Module):
    def forward(self, observations):
        return observations.new_tensor([20., -20., -20., -20.]).expand(len(observations), -1)

    def readout_process(self, observations, reservoir_states):
        return self.forward(observations)


class DiscreteGaitExplorationTests(unittest.TestCase):
    def test_auxiliary_loss_only_teaches_validated_contexts(self):
        logits = torch.tensor([[0., 3., 1., 2.], [1., 2., 3., 0.]], requires_grad=True)
        loss = context_guidance_loss(logits, torch.tensor([0, -1]))
        loss.backward()
        self.assertGreater(loss.item(), 3.)
        self.assertLess(logits.grad[0, 0].item(), 0.)
        torch.testing.assert_close(logits.grad[1], torch.zeros(4))

    def test_auxiliary_loss_balances_rare_context_classes(self):
        logits = torch.tensor([[4., 0., 0., 0.]] * 10 + [[4., 0., 0., 0.]])
        targets = torch.tensor([0] * 10 + [1])
        expected = (F.cross_entropy(logits[:10], targets[:10])
                    + F.cross_entropy(logits[10:], targets[10:])) / 2
        torch.testing.assert_close(context_guidance_loss(logits, targets), expected)

    def test_context_targets_survive_discrete_rollout_batching(self):
        storage = RolloutStorage_event_discrete(2, 2, [8], [8], [1], [4], 'cpu')
        transition = storage.StartTransition()
        transition.env_ids = torch.tensor([0, 1])
        transition.step_ids = torch.zeros(2, dtype=torch.long)
        transition.observations = torch.zeros(2, 8)
        transition.critic_observations = torch.zeros(2, 8)
        transition.actions = torch.zeros(2, 1)
        transition.logits = torch.zeros(2, 4)
        transition.values = torch.zeros(2, 1)
        transition.actions_log_prob = torch.zeros(2)
        transition.action_mean = torch.zeros(2, 1)
        transition.action_sigma = torch.zeros(2, 1)
        transition.context_targets = torch.tensor([2, -1])
        storage.add_start_transitions(transition)
        batch = next(storage.mini_batch_generator(1, 1))
        self.assertEqual(sorted(batch[-1].tolist()), [-1, 2])

    def test_sampling_and_inference_use_different_exploration_rules(self):
        model = ActorCriticDiscrete(
            8, 8, 4, 4, high_actor_type='snn_single_output_head',
            actor_hidden_dims=(8,), critic_hidden_dims=(8,), num_snn_steps=1,
            primitive_uniform_mix=.2)
        model.actor = FixedLogits()
        obs = torch.zeros(3, 8)
        model.update_distribution(obs)
        self.assertTrue((model.distribution_1.probs[:, 1:] >= .0499).all())
        self.assertTrue(torch.isfinite(model.get_actions_log_prob(torch.ones(3, 1, dtype=torch.long))).all())
        self.assertTrue((model.act_inference(obs) == 0).all())

    def test_ppo_recomputes_the_same_mixed_action_probability(self):
        model = ActorCriticDiscrete(
            8, 8, 4, 4, high_actor_type='lif_reservoir_snn_single_output',
            actor_hidden_dims=(8,), critic_hidden_dims=(8,),
            reservoir_dim=8, readout_hidden_dims=(8,), num_snn_steps=1,
            primitive_uniform_mix=.2)
        model.actor = FixedLogits()
        model.act_for_ppo_update(torch.zeros(2, 8), torch.zeros(2, 8))
        prob = model.get_actions_log_prob(torch.ones(2, 1, dtype=torch.long)).exp()
        torch.testing.assert_close(prob, torch.full((2,), .05), atol=1e-4, rtol=0)

    def test_temperature_keeps_gradient_for_a_nearly_absent_gait(self):
        model = ActorCriticDiscrete(
            8, 8, 4, 4, high_actor_type='snn_single_output_head',
            actor_hidden_dims=(8,), critic_hidden_dims=(8,), num_snn_steps=1,
            primitive_uniform_mix=.2, primitive_temperature=3.)
        logits = torch.tensor([[0., 0., 10., 0.]], requires_grad=True)
        distribution = model._gait_distribution(logits)
        self.assertGreater(distribution.probs[0, 0].item(), .07)
        self.assertEqual(distribution.probs.argmax(-1).item(), 2)
        (-distribution.log_prob(torch.tensor([0]))).backward()
        self.assertGreater(abs(logits.grad[0, 0].item()), .05)

    def test_command_head_matches_ppo_replay_and_receives_gradient(self):
        model = ActorCriticDiscrete(
            262, 269, 4, 4, high_actor_type='lif_reservoir_snn_single_output',
            reservoir_dim=8, readout_hidden_dims=(8,), num_snn_steps=1,
            terrain_scan_shape=(2, 16, 5), terrain_encoder_dim=16,
            command_obs_start=53, primitive_uniform_mix=.2,
            primitive_temperature=3.)
        self.assertTrue(hasattr(model.actor, 'command_readout_head'))
        obs = torch.zeros(2, 5, 262)
        obs[0, :, 53], obs[1, :, 53] = .8, 3.
        state = torch.zeros(2, model.actor.reservoir_state_dim)
        logits, updated = model.actor(obs, state)
        model.act_for_ppo_update(obs[:, -1], updated.detach())
        torch.testing.assert_close(model.distribution_1.probs,
                                   model._gait_distribution(logits).probs)
        (-model.distribution_1.log_prob(torch.tensor([0, 1])).sum()).backward()
        self.assertGreater(model.actor.command_readout_head.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
