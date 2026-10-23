"""Frequency-library tests for the currently enabled six-primitive controller."""
import importlib.util
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'legged_gym/utils/dynamics/a1/rhythmic_dynamics.py'
spec = importlib.util.spec_from_file_location('frequency_dynamics', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrimitiveFrequencyTests(unittest.TestCase):
    def test_library_selection_switch_and_partial_reset(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            obj = module.Multi_Primitive_Dynamics(num_envs=6, device=device)
            self.assertEqual(obj.primitive_frequencies,
                             dict(walk=6., pace=6., trot=8., canter=8., bound=10., gallop=10.))
            torch.testing.assert_close(obj.omega_g, torch.full((6, 4), 12 * torch.pi, device=device))
            actions = torch.zeros(6, 6, device=device)
            actions[:, 0] = torch.arange(6, device=device)
            expected = torch.tensor([[6.], [6.], [8.], [8.], [10.], [10.]], device=device)
            for _ in range(2):
                obj.intrinsic_dynamics(actions, torch.zeros(6, 3, device=device))
                torch.testing.assert_close(obj.f_gait, expected)
                torch.testing.assert_close(obj.omega_g, (2 * torch.pi * expected).expand(-1, 4))
                actions[:, 0] = actions[:, 0].roll(1)
                expected = expected.roll(1, 0)
            before = obj.f_gait.clone()
            before_omega = obj.omega_g.clone()
            obj.reset([])
            torch.testing.assert_close(obj.f_gait, before)
            obj.reset([1])
            before[1] = 6.
            before_omega[1] = 12 * torch.pi
            torch.testing.assert_close(obj.f_gait, before)
            torch.testing.assert_close(obj.omega_g, before_omega)
            torch.testing.assert_close(obj.omega[1], torch.full((4,), 6., device=device))

    def test_parent_retains_scalar_frequency(self):
        obj = module.Interoceptive_Reflex_Dynamics(num_envs=3, device='cpu', f_gait=7.)
        obj.reset([1])
        self.assertEqual(obj.f_gait, 7.)
        torch.testing.assert_close(obj.omega[1], torch.full((4,), 7.))


if __name__ == '__main__':
    unittest.main()
