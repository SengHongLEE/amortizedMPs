"""Explicit v1 -> v2 migration must preserve old actors and critics exactly."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rsl_rl'))
from rsl_rl.modules.actor_critic import ActorCritic, ActorCriticDiscrete

PATH = ROOT / 'legged_gym/legged_gym/utils/equilibrium_migration.py'


class MigrationTests(unittest.TestCase):
    def test_explicit_timing_prior_preserves_spatial_heads(self):
        spec = importlib.util.spec_from_file_location('migration', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'with_timing_prior'))
        from test_adaptive_equilibrium import timing_config
        source = {
            'actor.output_layer.weight': torch.randn(22, 32),
            'actor.output_layer.bias': torch.randn(22),
            'std': torch.ones(22)}
        result = module.with_timing_prior(source, timing_config(), .15, [.55, .1, .12], .2)
        for key in source:
            torch.testing.assert_close(result[key][:16], source[key][:16])
        means = torch.randn(8, 32) @ result['actor.output_layer.weight'].T + result['actor.output_layer.bias']
        expected = torch.tensor([-1/3] * 4 + [(.55 - 1) / .65, .4])
        torch.testing.assert_close(means[:, 16:], expected[None].expand(8, -1))
        torch.testing.assert_close(result['std'][16:], torch.full((6,), .2))
        self.assertTrue(source['actor.output_layer.weight'][16:].abs().sum() > 0)
        with self.assertRaises(ValueError):
            module.with_timing_prior(source, timing_config(), .15, [.1, .1, .12], .2)

    def test_modal_network_conversion_preserves_physical_means(self):
        spec = importlib.util.spec_from_file_location('migration', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'migrate_modal_state_dict'), 'Missing v2 to modal migration')
        options = dict(actor_type='snn', high_actor_type='lif_reservoir_snn_single_output',
                       actor_hidden_dims=[64, 32], critic_hidden_dims=[64, 32, 16],
                       readout_hidden_dims=[64, 32], num_snn_steps=8, reservoir_dim=16)
        with contextlib.redirect_stdout(io.StringIO()):
            old_low, new_low = ActorCritic(132, 139, 22, **options), ActorCritic(132, 139, 22, **options)
            old_high = ActorCriticDiscrete(132, 139, 6, 6, **options)
            new_high = ActorCriticDiscrete(132, 139, 6, 6, **options)
        h = torch.tensor([[1., 1., 1., 1.], [1., 1., -1., -1.],
                          [1., -1., 1., -1.], [1., -1., -1., 1.]])
        obs, critic = torch.randn(16, 132), torch.randn(16, 139)
        modal_obs, modal_critic = obs.clone(), critic.clone()
        modal_obs[:, 39:51] = (obs[:, 39:51].reshape(-1, 3, 4) @ h / 4).flatten(1)
        modal_critic[:, 46:58] = (critic[:, 46:58].reshape(-1, 3, 4) @ h / 4).flatten(1)
        for old, new, high in [(old_low, new_low, False), (old_high, new_high, True)]:
            new.load_state_dict(module.migrate_modal_state_dict(old.state_dict(), high=high), strict=True)
            torch.testing.assert_close(old.evaluate(critic), new.evaluate(modal_critic), atol=1e-6, rtol=1e-5)
        actions = new_low.act_inference(modal_obs)
        actions[:, 4:16] = (actions[:, 4:16].reshape(-1, 3, 4) @ h).flatten(1)
        torch.testing.assert_close(old_low.act_inference(obs), actions, atol=1e-6, rtol=1e-5)
        state = torch.randn(16, old_high.actor.reservoir_state_dim)
        logits_a, state_a = old_high.actor(obs[:, None], state)
        logits_b, state_b = new_high.actor(modal_obs[:, None], state)
        torch.testing.assert_close(logits_a, logits_b)
        torch.testing.assert_close(state_a, state_b, atol=1e-6, rtol=1e-5)

    def test_network_outputs_and_reservoir_states_preserved(self):
        self.assertTrue(PATH.exists(), 'Missing explicit checkpoint migration')
        spec = importlib.util.spec_from_file_location('migration', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        options = dict(actor_type='snn', high_actor_type='lif_reservoir_snn_single_output',
                       actor_hidden_dims=[64, 32], critic_hidden_dims=[64, 32, 16],
                       readout_hidden_dims=[64, 32], num_snn_steps=8, reservoir_dim=16)
        with contextlib.redirect_stdout(io.StringIO()):
            old_low, new_low = ActorCritic(124, 131, 16, **options), ActorCritic(132, 139, 22, **options)
            old_high = ActorCriticDiscrete(124, 131, 6, 6, **options)
            new_high = ActorCriticDiscrete(132, 139, 6, 6, **options)
        obs, critic = torch.randn(8, 124), torch.randn(8, 131)
        expanded, expanded_critic = torch.zeros(8, 132), torch.zeros(8, 139)
        expanded[:, :51], expanded[:, 57:130] = obs[:, :51], obs[:, 51:]
        expanded_critic[:, :58], expanded_critic[:, 64:137] = critic[:, :58], critic[:, 58:]
        for old, new, high in [(old_low, new_low, False), (old_high, new_high, True)]:
            migrated = module.migrate_state_dict(old.state_dict(), new.state_dict(), high=high, new_std=.2)
            new.load_state_dict(migrated, strict=True)
            torch.testing.assert_close(old.evaluate(critic), new.evaluate(expanded_critic))
        torch.testing.assert_close(old_low.act_inference(obs), new_low.act_inference(expanded)[:, :16])
        self.assertFalse(new_low.act_inference(expanded)[:, 16:].any())
        state = torch.randn(8, old_high.actor.reservoir_state_dim)
        logits_old, state_old = old_high.actor(obs[:, None], state)
        logits_new, state_new = new_high.actor(expanded[:, None], state)
        torch.testing.assert_close(logits_old, logits_new)
        torch.testing.assert_close(state_old, state_new)
        self.assertTrue((new_low.std[16:] == .2).all())
        damaged = old_low.state_dict()
        damaged['actor.linear_layers.0.weight'] = torch.zeros(64, 125)
        with self.assertRaises(ValueError):
            module.migrate_state_dict(damaged, new_low.state_dict(), high=False)


if __name__ == '__main__':
    unittest.main()
