"""CPU checks for the paper's gait-selection reward, without Isaac Gym."""
import math
from types import SimpleNamespace
import unittest

import torch
from test_interoceptive_rewards import load_class

Env = load_class('mp_multi_primitive.py', 'MotorPrimitives', {
    '_init_pi_g_buffers',
    '_pi_g_cost_of_transport',
    '_pi_g_energy',
    '_pi_g_motion_costs',
    '_pi_g_psi',
    '_pi_g_torque_fraction',
    '_prepare_reward_function',
    '_reset_pi_g_buffers',
    '_reward_pi_g',
    '_reward_pi_g_contact',
    '_reward_pi_g_cot',
    '_reward_pi_g_torque',
    '_reward_pi_g_utility',
    '_reward_pi_g_work',
    '_update_pi_g_metrics',
    'compute_reward',
})

Config = load_class('mp_multi_primitive_config.py', 'MotorPrimitivesCfg')


def make_env():
    e = Env()
    e.num_envs, e.device, e.dt = 2, 'cpu', .02
    e.sim_params = SimpleNamespace(dt=.01, gravity=SimpleNamespace(z=-10.))
    e.cfg = SimpleNamespace(rewards=Config.rewards, env=SimpleNamespace(play=False))
    e.robot_mass = torch.full((2,), 10.)
    e.commands = torch.zeros(2, 4)
    e.base_lin_vel = torch.zeros(2, 3)
    e.base_ang_vel = torch.zeros(2, 3)
    e.actions, e.last_actions = torch.zeros(2, 10), torch.zeros(2, 10)
    e.episode_length_buf = torch.full((2,), 2)
    e.substep_torques = torch.zeros(2, 2, 12)
    e.substep_dof_vel = torch.zeros_like(e.substep_torques)
    e.torque_limits = torch.full((12,), 10.)
    e.ID = SimpleNamespace(phi=torch.zeros(2, 1), gait_phase_delta=torch.zeros(2, 4),
                           _primitive_duty_factors=torch.tensor([.75, .5, .45, .4]))
    e.contact_forces = torch.zeros(2, 4, 3)
    e.feet_indices = torch.arange(4)
    e._pi_g_energy = lambda: torch.zeros(2)
    e._init_pi_g_buffers()
    return e


class PiGTests(unittest.TestCase):
    def test_perfect_reward_and_fourth_power_tracking(self):
        e = make_env()
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), 4.4))
        e.base_lin_vel[:, 0] = .5
        expected = 4 + .4 * (1 - math.tanh(.5 ** 4))
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), expected))

    def test_cot_zero_command_and_torque_normalization(self):
        e = make_env()
        e.substep_torques[:] = 1.
        # At zero command CoT is inapplicable; no division by zero.
        expected = 3.4 + 1 - math.tanh(.1 ** 2)
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), expected))
        e.commands[:, 0] = e.base_lin_vel[:, 0] = 1.
        cot = 12 * .3 / 100
        expected = 2.4 + 1 - math.tanh(.1 ** 2) + 1 - math.tanh(cot ** 2)
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), expected))
        self.assertTrue(torch.isfinite(e._reward_pi_g()).all())

    def test_switch_penalty_is_boolean_and_not_repeated(self):
        e = make_env()
        e.actions[:, 0] = torch.tensor([1., 3.])
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), 3.6))
        e.last_actions.copy_(e.actions)
        torch.testing.assert_close(e._reward_pi_g(), torch.full((2,), 4.4))

    def test_switch_cost_is_modulated_by_motion_quality(self):
        e = make_env()
        e.commands[:, 0] = 1.
        e.substep_torques[1] = 5.
        e.substep_dof_vel[1] = 3.
        e._pi_g_contact_integral[1] = e.dt
        e._pi_g_work[:] = torch.tensor([-3., 8.])
        baseline = e._reward_pi_g().clone()
        e.actions[:, 0] = torch.tensor([1., 3.])
        costs = sum(e._pi_g_motion_costs())
        expected = .4 * 2. * (1 - torch.tanh(costs.square()))
        torch.testing.assert_close(baseline - e._reward_pi_g(), expected)

    def test_switch_cost_is_configurable_and_ignored_at_initialization(self):
        e = make_env()
        e.cfg.rewards = SimpleNamespace(pi_g_utility_weight=.4, pi_g_switch_cost=2.)
        baseline = e._reward_pi_g().clone()
        e.actions[:, 0] = 1.
        torch.testing.assert_close(baseline - e._reward_pi_g(), torch.full((2,), .8))
        e.episode_length_buf[:] = 1
        torch.testing.assert_close(e._reward_pi_g(), baseline)
        e.episode_length_buf[:] = 2
        e.cfg.rewards.pi_g_switch_cost = 0.
        torch.testing.assert_close(e._reward_pi_g(), baseline)

    def test_contact_reference_uses_nominal_phase_and_duty_factor(self):
        e = make_env()
        e.ID.phi[:] = math.pi
        e.contact_forces[0, :, 2] = 10.
        e._update_pi_g_metrics()
        e._update_pi_g_metrics()
        torch.testing.assert_close(e._pi_g_contact_integral / e.dt, torch.tensor([0., 1.]))

    def test_cycle_work_switch_and_partial_reset(self):
        e = make_env()
        energy = torch.zeros(2)
        e._pi_g_energy = lambda: energy
        e.ID.phi[:] = 6.
        e._update_pi_g_metrics()  # initialize; first partial cycle is discarded
        energy[:] = 1.
        e.ID.phi[:] = .1
        e._update_pi_g_metrics()
        torch.testing.assert_close(e._pi_g_work, torch.zeros(2))
        energy[:] = 3.
        e.ID.phi[:] = 6.
        e._update_pi_g_metrics()
        energy[:] = 4.
        e.ID.phi[:] = .1
        e._update_pi_g_metrics()
        torch.testing.assert_close(e._pi_g_work, torch.full((2,), 3.))
        e.actions[0, 0] = 2
        e._update_pi_g_metrics()
        torch.testing.assert_close(e._pi_g_work, torch.tensor([0., 3.]))
        e._reset_pi_g_buffers(torch.tensor([0]))
        self.assertEqual(e._pi_g_work[1].item(), 3.)
        self.assertFalse(e._pi_g_initialized[0])

    def test_reward_routing_preserves_modulation_and_total(self):
        e = make_env()
        names = set(Config.rewards.modulation_param_rewards + Config.rewards.task_rewards)
        e.reward_names = list(names)
        e.reward_functions = [lambda: torch.ones(2) for _ in names]
        e.reward_scales = {n: getattr(Config.rewards.scales, n) * e.dt for n in names}
        selection_names = Config.rewards.primitive_selection_rewards
        e.primitive_selection_reward_scales = {
            n: getattr(Config.rewards.primitive_selection_scales, n) * e.dt
            for n in selection_names
        }
        e.episode_sums = {n: torch.zeros(2) for n in names | set(selection_names) | {'pi_g'}}
        e.rew_buf = torch.zeros(2)
        e.primitive_selection_rew_buf = torch.zeros(2)
        e.modulation_param_rew_buf = torch.zeros(2)
        total, selection, modulation = e.compute_reward()
        torch.testing.assert_close(selection, torch.full((2,), 4.4 * e.dt))
        torch.testing.assert_close(total, torch.full((2,), sum(e.reward_scales.values())))
        expected = sum(e.reward_scales[n] for n in Config.rewards.modulation_param_rewards)
        torch.testing.assert_close(modulation, torch.full((2,), expected))

    def test_split_terms_match_original_formula(self):
        torch.manual_seed(7)
        e = make_env()
        for _ in range(20):
            e.commands.normal_()
            e.base_lin_vel.normal_()
            e.base_ang_vel.normal_()
            e.substep_torques.normal_()
            e.substep_dof_vel.normal_()
            e._pi_g_contact_integral.uniform_(0, e.dt)
            e._pi_g_work.normal_()
            e.actions[:, 0] = torch.tensor([0., 3.])
            psi = lambda x: 1 - torch.tanh(x.square())
            speed = e.commands[:, :3].norm(dim=1)
            power = (e.substep_torques * e.substep_dof_vel + .3 * e.substep_torques.square()).clamp_min(0).sum(-1).mean(1)
            cot = power / (e.robot_mass * 10 * speed)
            torque = (e.substep_torques.abs() / e.torque_limits).mean((1, 2))
            contact, work = e._pi_g_contact_integral / e.dt, e._pi_g_work
            error = (e.base_lin_vel[:, :2] - e.commands[:, :2]).square().sum(1)
            error += (e.base_ang_vel[:, 2] - e.commands[:, 2]).square()
            switch = torch.tensor([0., 1.])
            expected = .4 * (psi(error) - 2 * switch * psi(cot + torque + contact + work))
            expected += psi(cot) + psi(torque) + psi(contact) + psi(work)
            torch.testing.assert_close(e._reward_pi_g(), expected)

    def test_routing_evaluates_shared_terms_once_and_supports_zero_scale(self):
        e = make_env()
        e.rew_buf = torch.zeros(2)
        e.primitive_selection_rew_buf = torch.zeros(2)
        e.modulation_param_rew_buf = torch.zeros(2)
        e.reward_scales = {'tracking_lin_vel': 3., 'tracking_ang_vel': 0.}
        calls = []
        def tracking():
            calls.append(1)
            return torch.ones(2)
        e._reward_tracking_lin_vel = tracking
        e._prepare_reward_function()
        total, selection, modulation = e.compute_reward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(total, torch.full((2,), 3 * e.dt))
        torch.testing.assert_close(modulation, total)
        torch.testing.assert_close(selection, torch.full((2,), 4.4 * e.dt))
        torch.testing.assert_close(e.episode_sums['pi_g'], selection)


if __name__ == '__main__':
    unittest.main()
