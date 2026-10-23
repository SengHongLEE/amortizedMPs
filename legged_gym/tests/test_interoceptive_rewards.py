"""CPU reward tests; load the real methods without importing Isaac Gym."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ENV_DIR = Path(__file__).resolve().parents[1] / 'legged_gym/envs/a1'


def load_class(filename, class_name, methods=None):
    tree = ast.parse((ENV_DIR / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []
    if methods is not None:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        if not cls.body:
            cls.body = [ast.copy_location(ast.Pass(), cls)]
    ns = {'torch': torch, 'np': __import__('numpy')}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, 'exec'), ns)
    return ns[class_name]


METHODS = {
    '_init_homeostasis_buffers', '_reset_homeostasis_buffers',
    '_cycle_state', '_update_cycle_consistency', '_update_contact_rewards',
    '_reward_cycle_consistency', '_reward_stance_slip', '_reward_touchdown_impact',
}
Environment = load_class('mp_interoceptive.py', 'MotorPrimitives', METHODS)
Config = load_class('mp_interoceptive_config.py', 'MotorPrimitivesCfg')


def make_env():
    env = Environment()
    env.device, env.num_envs, env.num_dof, env.num_bodies = 'cpu', 2, 12, 4
    env.dt = .02
    env.sim_params = SimpleNamespace(dt=.001)
    env.cfg = SimpleNamespace(rewards=Config.rewards)
    env.feet_indices = torch.arange(4)
    env.root_states = torch.zeros(2, 13)
    env.env_origins = torch.zeros(2, 3)
    env.base_lin_vel = torch.zeros(2, 3)
    env.base_ang_vel = torch.zeros(2, 3)
    env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.dof_pos, env.dof_vel = torch.zeros(2, 12), torch.zeros(2, 12)
    env.default_dof_pos = torch.zeros(1, 12)
    env.contact_forces = torch.zeros(2, 4, 3)
    env.rigid_body_state = torch.zeros(8, 13)
    env.ID = SimpleNamespace(X=torch.zeros(2, 3, 4), phases=torch.ones(2, 4)*4.)
    env._init_homeostasis_buffers()
    return env


class RewardTests(unittest.TestCase):
    def test_repeatable_motion_and_cycle_drift(self):
        env = make_env()
        for step in range(800):
            phase = step * 2 * math.pi * 4 * .001
            env.ID.X[:, 2, 0] = phase % (2 * math.pi)
            env.dof_pos[:] = math.sin(phase)
            if step >= 500:
                env.dof_pos[1] += .5
            env._update_cycle_consistency()
        self.assertLess(env._reward_cycle_consistency()[0].item(), 1e-6)
        self.assertGreater(env._reward_cycle_consistency()[1].item(), 0.)

    def test_frequency_change_and_partial_reset_invalidate_history(self):
        env = make_env()
        env._update_cycle_consistency()
        env._cycle_sample_turn[:] = 3
        env.ID.phases[0, 0] = 8.
        env._update_cycle_consistency()
        self.assertTrue((env._cycle_sample_turn[0] == -1).all())
        self.assertTrue((env._cycle_sample_turn[1] == 3).all())
        env._reset_homeostasis_buffers(torch.tensor([0]))
        self.assertTrue((env._cycle_sample_turn[1] == 3).all())
        self.assertFalse(env._contact_initialized[0])

    def test_slip_uses_world_horizontal_velocity_and_contact(self):
        env = make_env()
        feet = env.rigid_body_state.view(2, 4, 13)
        feet[:, :, 7] = 2.
        feet[:, :, 9] = 99.
        env.contact_forces[0, :, 2] = 20.
        for _ in range(20):
            env._update_contact_rewards()
        torch.testing.assert_close(env._reward_stance_slip(), torch.tensor([16., 0.]))
        before = env._reward_stance_slip().clone()
        torch.testing.assert_close(before, env._reward_stance_slip())

    def test_touchdown_preimpact_velocity_peak_force_and_no_retrigger(self):
        env = make_env()
        feet = env.rigid_body_state.view(2, 4, 13)
        feet[:, 0, 9] = -2.
        for _ in range(30):
            env._update_contact_rewards()
        feet[:, 0, 9] = 0.
        env.contact_forces[:, 0, 2] = 20.
        env._update_contact_rewards()
        first = env._reward_touchdown_impact().clone() * env.dt
        torch.testing.assert_close(first, torch.full((2,), 4.))
        env.contact_forces[:, 0, 2] = 2 * env.cfg.rewards.max_contact_force
        env._update_contact_rewards()
        peak = env._reward_touchdown_impact().clone()
        self.assertTrue((peak > first / env.dt).all())
        for _ in range(5):
            env._update_contact_rewards()
        torch.testing.assert_close(env._reward_touchdown_impact(), peak)
        # A one-substep contact dropout must not create another landing event.
        env.contact_forces.zero_()
        env._update_contact_rewards()
        env.contact_forces[:, 0, 2] = 20.
        env._update_contact_rewards()
        torch.testing.assert_close(env._reward_touchdown_impact(), peak)

    def test_reset_does_not_create_touchdown_from_stale_velocity(self):
        env = make_env()
        env._previous_foot_velocity[:] = -10.
        env.contact_forces[:, :, 2] = 200.
        env._update_contact_rewards()
        torch.testing.assert_close(env._reward_touchdown_impact(), torch.zeros(2))

    def test_cycle_state_ignores_world_translation(self):
        env = make_env()
        original = env._cycle_state().clone()
        env.root_states[:, :2] += 10.
        torch.testing.assert_close(env._cycle_state(), original)

    def test_noninteger_periods_are_phase_aligned(self):
        for frequency in (4., 7., 10.):
            env = make_env()
            env.ID.phases[:, 0] = frequency
            for step in range(650):
                phase = step * 2 * math.pi * frequency * .001
                env.ID.X[:, 2, 0] = phase % (2 * math.pi)
                env.dof_pos[:] = math.sin(phase)
                env._update_cycle_consistency()
            self.assertLess(env._reward_cycle_consistency().max().item(), 1.e-5)

    def test_contact_hysteresis(self):
        env = make_env()
        env.contact_forces[:, :, 2] = 6.
        env._update_contact_rewards()
        env.contact_forces[:, :, 2] = 3.
        env._update_contact_rewards()
        self.assertTrue(env._stance_contact.all())
        env.contact_forces[:, :, 2] = .5
        env._update_contact_rewards()
        self.assertFalse(env._stance_contact.any())

    def test_impact_event_cost_is_independent_of_control_dt(self):
        totals = []
        for control_dt in (.01, .02):
            env = make_env()
            env.dt = control_dt
            env.rigid_body_state.view(2, 4, 13)[:, 0, 9] = -1.
            for _ in range(30):
                env._update_contact_rewards()
            env.contact_forces[:, 0, 2] = 20.
            env._update_contact_rewards()
            totals.append(env._reward_touchdown_impact() * env.dt)
        torch.testing.assert_close(totals[0], totals[1])


if __name__ == '__main__':
    unittest.main()
