import contextlib
import io
import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.algorithms import PPOevent_discrete
from rsl_rl.modules import ActorCriticDiscrete
from rsl_rl.runners.mp_policy_runner import MPPolicyRunner


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, float(value)))


class RecordingAlgorithm:
    def __init__(self, update_period):
        self.update_period = update_period
        self.storage = SimpleNamespace(ready=False)
        self.actor_critic = SimpleNamespace(train=lambda: None, std=torch.ones(3))
        self.learning_rate = 0.001
        self.mean_entropy = 0.7
        self.rewards = []

    def process_env_step(self, step, env_ids, rewards, dones, critic_obs, infos):
        self.rewards.append(rewards.clone())
        self.storage.ready = step % self.update_period == 0

    def compute_returns(self, critic_obs):
        pass

    def update(self):
        return 0., 0.


def run_logging_rollout(high_period=4, num_steps=4):
    runner = MPPolicyRunner.__new__(MPPolicyRunner)
    runner.device = 'cpu'
    runner.log_dir = 'unused'
    runner.writer = RecordingWriter()
    runner.num_steps_per_env = 1
    runner.save_interval = 5
    runner.current_learning_iteration = 0
    runner.current_high_step = 0
    runner.current_low_step = 0
    runner.high_tot_timesteps = runner.low_tot_timesteps = 0
    runner.high_tot_time = runner.low_tot_time = 0
    runner.high_reservoir_states = runner.low_reservoir_states = None
    runner.low_update_cycle = torch.tensor([10, 25, 50])
    runner.high_alg = RecordingAlgorithm(high_period)
    runner.low_alg = RecordingAlgorithm(2)
    runner.save_model = lambda *args: None
    runner.final_log = lambda: None
    runner.env = SimpleNamespace(
        num_envs=3,
        episode_length_buf=torch.zeros(3, dtype=torch.long),
        max_episode_length_s=10.,
        cfg=SimpleNamespace(
            control=SimpleNamespace(high_cycle=50, low_freq_idx=[10, 25, 50]),
            rewards=SimpleNamespace(q_bad=0.9, q_good=0.1),
        ),
        reward_scales={
            'tracking_lin_vel': 0.06, 'tracking_ang_vel': 0.03,
            'disturbance_penalty': -0.081, 'decision': -0.09,
        },
        episode_sums={name: torch.zeros(3) for name in ['decision', 'disturbance_penalty']},
        get_observations=lambda: torch.zeros(3, 5, 42),
        get_privileged_observations=lambda: None,
    )
    infos = {'episode': {'rew_tracking_lin_vel': torch.tensor(999.)}}

    def rollout(obs, critic_obs, high_step, low_step):
        step = high_step + 1
        dones = torch.tensor([step % 4 == 2, step % 4 == 2, step % 4 == 0])
        runner.env.episode_length_buf += 1
        runner.env.episode_length_buf[dones] = 0
        # Like the real environment, reset clears its sums before returning,
        # while steps without resets retain the previous episode dictionary.
        if dones.any():
            infos['episode'] = {'rew_tracking_lin_vel': torch.tensor(1. if step % 4 == 2 else 5.)}
            for name, values in runner.env.episode_sums.items():
                infos['episode']['rew_' + name] = values[dones].mean() / 10.
                values[dones] = 0.
        rewards = [torch.ones(3) * 99., torch.tensor([0.045, 0.063, 0.081]),
                   torch.tensor([0.02, 0.04, 0.06])]
        return obs, critic_obs, rewards, dones, infos

    runner.training_rollout = rollout
    with contextlib.redirect_stdout(io.StringIO()):
        runner.learn(num_steps // high_period)
    return runner


class MPPolicyLoggingTest(unittest.TestCase):
    def test_logs_categorical_entropy_instead_of_unused_std(self):
        scalars = run_logging_rollout().writer.scalars
        self.assertIn('Policy/high/entropy', scalars)
        self.assertAlmostEqual(scalars['Policy/high/entropy'][0][1], 0.7)
        self.assertNotIn('Policy/high/mean_noise_std', scalars)
        self.assertIn('Policy/low/mean_noise_std', scalars)

    def test_high_and_low_log_their_own_episode_rewards(self):
        scalars = run_logging_rollout().writer.scalars
        q = [value / (0.09 + 1e-6) for value in [0.045, 0.063, 0.081]]
        high_rewards = [max(0., 0.09 * raw_q
                           - 0.081 * (max(0., 0.9 - quality) / 0.9)**2 * cycle
                           - 0.09 * (max(0., quality - 0.1) / 0.9)**2 * (1 - cycle))
                        for raw_q, quality, cycle in zip([0.5, 0.7, 0.9], q, [0., 0.375, 1.])]
        expected_high = (2 * high_rewards[0] + 2 * high_rewards[1] + 4 * high_rewards[2]) / 3
        expected_low = (0.04 + 0.08 + 0.24) / 3
        self.assertAlmostEqual(scalars['Train/high/mean_reward'][-1][1], expected_high, places=6)
        self.assertAlmostEqual(scalars['Train/low/mean_reward'][-1][1], expected_low, places=6)
        self.assertAlmostEqual(scalars['Train/high/mean_reward/time'][-1][1], expected_high, places=6)

    def test_penalties_include_all_steps_and_weight_completed_episodes(self):
        scalars = run_logging_rollout().writer.scalars
        q = [value / (0.09 + 1e-6) for value in [0.045, 0.063, 0.081]]
        for name, values in [
            ('decision', [-0.09 * (max(0., quality - 0.1) / 0.9)**2 * (1 - cycle)
                          for quality, cycle in zip(q, [0., 0.375, 1.])]),
            ('disturbance_penalty', [-0.081 * (max(0., 0.9 - quality) / 0.9)**2 * cycle
                                     for quality, cycle in zip(q, [0., 0.375, 1.])]),
        ]:
            expected = (2 * values[0] + 2 * values[1] + 4 * values[2]) / 30.
            self.assertAlmostEqual(scalars['Episode/high/rew_' + name][-1][1], expected, places=7)
            self.assertAlmostEqual(scalars['Episode/low/rew_' + name][0][1], sum(values[:2]) / 10., places=7)
            self.assertAlmostEqual(scalars['Episode/low/rew_' + name][1][1], 4 * values[2] / 10., places=7)
        self.assertAlmostEqual(scalars['Episode/high/rew_tracking_lin_vel'][0][1], 7. / 3., places=6)

    def test_steps_without_completed_episodes_do_not_log_stale_data_or_nan(self):
        scalars = run_logging_rollout(high_period=1).writer.scalars
        episodes = scalars['Episode/high/rew_tracking_lin_vel']
        self.assertEqual([step for step, value in episodes], [1, 3])
        for entries in scalars.values():
            self.assertTrue(all(math.isfinite(value) for step, value in entries))

    def test_logging_does_not_change_rewards_sent_to_ppo(self):
        runner = run_logging_rollout()
        q = torch.tensor([0.045, 0.063, 0.081]) / (0.09 + 1e-6)
        cycles = torch.tensor([0., 0.375, 1.])
        expected = (torch.tensor([0.045, 0.063, 0.081])
                    - 0.081 * ((0.9 - q).relu() / 0.9).square() * cycles
                    - 0.09 * ((q - 0.1).relu() / 0.9).square() * (1 - cycles)).clamp_min(0.)
        torch.testing.assert_close(runner.high_alg.rewards[1], 2 * expected)
        torch.testing.assert_close(runner.high_alg.rewards[3], expected * torch.tensor([2., 2., 4.]))
        torch.testing.assert_close(runner.low_alg.rewards[3], torch.tensor([0.04, 0.08, 0.24]))

    def test_episode_accumulators_reset_across_updates_and_episodes(self):
        runner = run_logging_rollout(num_steps=8)
        scalars = runner.writer.scalars
        # At step 8, six episodes have ended: lengths 2, 2, 4, 4, 4, 4.
        high_rewards = runner.high_alg.rewards[0]
        expected_high = (6 * high_rewards[0] + 6 * high_rewards[1] + 8 * high_rewards[2]) / 6
        self.assertAlmostEqual(scalars['Train/high/mean_reward'][-1][1], float(expected_high), places=7)
        self.assertAlmostEqual(scalars['Train/low/mean_reward'][-1][1], (6 * 0.02 + 6 * 0.04 + 8 * 0.06) / 6, places=7)
        for name in ['decision', 'disturbance_penalty']:
            low = [value for step, value in scalars['Episode/low/rew_' + name]]
            self.assertAlmostEqual(low[2], 2 * low[0], places=7)
            self.assertAlmostEqual(low[3], low[1], places=7)
            expected = (2 * low[2] + low[3]) / 3
            self.assertAlmostEqual(scalars['Episode/high/rew_' + name][-1][1], expected, places=7)
        self.assertAlmostEqual(scalars['Episode/high/rew_tracking_lin_vel'][-1][1], 7. / 3., places=6)


class DiscreteEntropyLoggingTest(unittest.TestCase):
    def test_update_reports_entropy_over_all_batches_and_resets_each_update(self):
        torch.manual_seed(4)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCriticDiscrete(3, 3, 3, 3,
                high_actor_type='lif_reservoir_snn_single_output', reservoir_dim=4,
                readout_hidden_dims=[4], critic_hidden_dims=[4], num_snn_steps=2)
        alg = PPOevent_discrete(actor, num_learning_epochs=2, num_mini_batches=2,
                                learning_rate=0., entropy_coef=0.01)
        alg.init_storage(4, 1, [3], [None], [1], [3])
        for logits in [torch.zeros(3), torch.tensor([5., 0., -1.])]:
            with torch.no_grad():
                actor.actor.readout.output_head.weight.zero_()
                actor.actor.readout.output_head.bias.copy_(logits)
                obs = torch.zeros(4, 2, 3)
                alg.act(torch.arange(4), torch.zeros(4, dtype=torch.long), obs,
                        obs[:, -1], torch.zeros(4, 8))
                alg.process_env_step(1, torch.arange(4), torch.ones(4),
                                     torch.ones(4, dtype=torch.bool), obs[:, -1], {})
                alg.compute_returns(obs[:, -1])
            losses = alg.update()
            self.assertEqual(len(losses), 2)
            self.assertIsNotNone(getattr(alg, 'mean_entropy', None))
            self.assertAlmostEqual(alg.mean_entropy,
                torch.distributions.Categorical(logits=logits).entropy().item(), places=6)

    def test_entropy_averages_changing_distributions_not_just_last_batch(self):
        torch.manual_seed(4)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCriticDiscrete(3, 3, 3, 3,
                high_actor_type='lif_reservoir_snn_single_output', reservoir_dim=4,
                readout_hidden_dims=[4], critic_hidden_dims=[4], num_snn_steps=2)
        alg = PPOevent_discrete(actor, num_learning_epochs=3, num_mini_batches=2,
                                learning_rate=0.02, entropy_coef=0.01)
        alg.init_storage(16, 1, [3], [None], [1], [3])
        with torch.no_grad():
            obs = torch.randn(16, 2, 3)
            actions, _ = alg.act(torch.arange(16), torch.zeros(16, dtype=torch.long),
                                 obs, obs[:, -1], torch.zeros(16, 8))
            alg.process_env_step(1, torch.arange(16), (actions[:, 0] == 2).float(),
                                 torch.ones(16, dtype=torch.bool), obs[:, -1], {})
            alg.compute_returns(obs[:, -1])
        observed_entropies = []
        update_distribution = actor.act_for_ppo_update

        def record_distribution(observations, states):
            update_distribution(observations, states)
            observed_entropies.append(actor.entropy.detach().mean().item())

        actor.act_for_ppo_update = record_distribution
        alg.update()
        self.assertEqual(len(observed_entropies), 6)
        self.assertAlmostEqual(alg.mean_entropy, sum(observed_entropies) / 6, places=7)
        self.assertGreater(abs(alg.mean_entropy - observed_entropies[-1]), 1e-5)


if __name__ == '__main__':
    unittest.main()
