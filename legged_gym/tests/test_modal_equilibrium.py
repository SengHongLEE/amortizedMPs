"""Coordinated equilibrium modes must preserve the Cartesian action span."""
import importlib.util
from pathlib import Path
import unittest
import torch
from types import SimpleNamespace
from test_equilibrium_dynamics import module, Config
from test_adaptive_equilibrium import timing_config
from test_interoceptive_rewards import load_class


class ModalTests(unittest.TestCase):
    def test_environment_clips_in_leg_coordinates_and_stores_consistent_history(self):
        mod = module()
        cls = load_class('mp_proprio_terrain.py', 'MotorPrimitives', {'step'})
        obj = mod.ModalEquilibriumDynamics(
            timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
            num_envs=2, device='cpu', nominal_clearance=.15, max_clearance=.2)
        # Stop the actual step method immediately after its action preprocessing.
        class Captured(Exception):
            pass
        def capture(*args):
            raise Captured()
        env = cls()
        env.ID, env.device = obj, 'cpu'
        env.actions = torch.zeros(2, 23)
        env.cfg = SimpleNamespace(normalization=SimpleNamespace(clip_actions=1.))
        env.terrain_curriculum = SimpleNamespace(accumulate=capture)
        env.commands, env.root_states, env.dt = torch.zeros(2, 3), torch.zeros(2, 13), .02
        original = torch.zeros(2, 23)
        original[:, 0] = torch.tensor([2., 5.])
        original[0, 5:9] = torch.tensor([3., -2., .4, .1])
        original[1, 13:17] = torch.tensor([4., 4., -4., -4.])
        original[:, 1] = 2.
        modal = original.clone()
        modal[:, 5:17] = (original[:, 5:17].reshape(-1, 3, 4) @ obj.mode_to_leg / 4).flatten(1)
        saved = modal.clone()
        with self.assertRaises(Captured):
            env.step(modal)
        bounded = original.clone()
        bounded[:, 1:].clamp_(-1., 1.)
        expected = bounded.clone()
        expected[:, 5:17] = (bounded[:, 5:17].reshape(-1, 3, 4) @ obj.mode_to_leg / 4).flatten(1)
        torch.testing.assert_close(env.actions, expected)
        torch.testing.assert_close(modal, saved)

    def test_action_rate_cost_is_measured_in_original_leg_coordinates(self):
        mod = module()
        obj = mod.ModalEquilibriumDynamics(
            timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
            num_envs=6, device='cpu', nominal_clearance=.15, max_clearance=.2)
        self.assertTrue(hasattr(obj, 'continuous_action_change_cost'), 'Missing coordinate-invariant action-rate cost')
        original = torch.randn(6, 22)
        modal = original.clone()
        modal[:, 4:16] = (original[:, 4:16].reshape(-1, 3, 4) @ obj.mode_to_leg / 4).flatten(1)
        torch.testing.assert_close(obj.continuous_action_change_cost(torch.zeros_like(modal), modal),
                                   original.square().sum(-1))

    def test_equivalent_unbounded_actions_preserve_dynamics(self):
        mod = module()
        self.assertTrue(hasattr(mod, 'ModalEquilibriumDynamics'), 'Missing coordinated equilibrium modes')
        kwargs = dict(timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
                      num_envs=6, device='cpu', nominal_clearance=.15, max_clearance=.20)
        old, new = mod.AdaptiveEquilibriumDynamics(**kwargs), mod.ModalEquilibriumDynamics(**kwargs)
        h = torch.tensor([[1., 1., 1., 1.], [1., 1., -1., -1.],
                          [1., -1., 1., -1.], [1., -1., -1., 1.]])
        torch.manual_seed(23)
        command = torch.tensor([[.5, 0., .2]]).expand(6, -1)
        for _ in range(80):
            original = torch.randn(6, 23) * 1.4
            original[:, 0] = torch.arange(6)
            modal = original.clone()
            modal[:, 5:17] = (original[:, 5:17].reshape(-1, 3, 4) @ h / 4).flatten(1)
            force = torch.rand(6, 4) * 3
            a = torch.stack(old.intrinsic_dynamics(original, command, contact_forces=force), -1)
            b = torch.stack(new.intrinsic_dynamics(modal, command, contact_forces=force), -1)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(old.controller_observations(), new.controller_observations(),
                                       atol=1e-6, rtol=1e-5)

    def test_front_back_mode_and_cartesian_limits(self):
        mod = module()
        self.assertTrue(hasattr(mod, 'ModalEquilibriumDynamics'), 'Missing coordinated equilibrium modes')
        obj = mod.ModalEquilibriumDynamics(
            timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
            num_envs=2, device='cpu', nominal_clearance=.15, max_clearance=.2)
        actions = torch.zeros(2, 23)
        actions[0, 14] = .5
        actions[1, 13:17] = 10.
        obj.intrinsic_dynamics(actions, torch.zeros(2, 3))
        torch.testing.assert_close(obj.offset_target[0, :, 2], torch.tensor([.02, .02, -.02, -.02]))
        self.assertTrue((obj.offset_target.abs() <= obj.offset_limits).all())
        torch.testing.assert_close(obj.offset_target[1, :, 2], torch.tensor([.04, 0., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
