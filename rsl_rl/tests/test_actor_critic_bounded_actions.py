"""The modulation policy must optimize the actions the controller receives."""

import unittest

import torch

from rsl_rl.modules.actor_critic import ActorCritic


class BoundedActionTests(unittest.TestCase):
    def test_squashed_gaussian_actions_and_log_prob_stay_finite_after_legacy_std_load(self):
        policy = ActorCritic(
        4, 4, 2, actor_type="mlp", actor_hidden_dims=(8,),
        critic_hidden_dims=(8,), init_noise_std=0.35,
        squash_actions=True, min_action_std=0.05, max_action_std=0.6,
    )
        with torch.no_grad():
            policy.std.fill_(14.0)  # observed in the old MP_full_a1 checkpoint
        observations = torch.randn(1024, 4)
        actions = policy.act(observations)
        self.assertTrue(torch.all(actions.abs() < 1.0).item())
        self.assertTrue(torch.allclose(policy.action_std, torch.full_like(policy.action_std, 0.6)))
        self.assertTrue(torch.isfinite(policy.get_actions_log_prob(actions)).all().item())
        self.assertTrue(torch.isfinite(policy.entropy).all().item())
        inference = policy.act_inference(observations)
        self.assertTrue(torch.all(inference.abs() < 1.0).item())


    def test_squashed_log_prob_includes_tanh_jacobian(self):
        policy = ActorCritic(
        3, 3, 1, actor_type="mlp", actor_hidden_dims=(8,),
        critic_hidden_dims=(8,), init_noise_std=0.2,
        squash_actions=True, min_action_std=0.05, max_action_std=0.6,
    )
        observations = torch.zeros(2, 3)
        policy.update_distribution(observations)
        actions = torch.tensor([[0.0], [0.8]])
        latent = torch.atanh(actions)
        expected = policy.distribution.log_prob(latent).sum(-1) - torch.log1p(-actions.square()).sum(-1)
        self.assertTrue(torch.allclose(policy.get_actions_log_prob(actions), expected, atol=1e-5))

    def test_squashed_ppo_terms_have_finite_gradients_near_action_limits(self):
        policy = ActorCritic(
            3, 3, 2, actor_type="mlp", actor_hidden_dims=(8,),
            critic_hidden_dims=(8,), init_noise_std=0.35,
            squash_actions=True, min_action_std=0.05, max_action_std=0.6,
        )
        policy.update_distribution(torch.ones(4, 3))
        actions = torch.tensor([[.99, -.99]]).expand(4, -1)
        loss = -policy.get_actions_log_prob(actions).mean() - .01 * policy.entropy.mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.actor.parameters()
                            if p.grad is not None))
        self.assertTrue(torch.isfinite(policy.std.grad).all())


if __name__ == "__main__":
    unittest.main()
