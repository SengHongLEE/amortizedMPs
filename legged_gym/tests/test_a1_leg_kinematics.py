"""Verify analytic foot Jacobians against finite differences of physical FK."""
import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/dynamics/a1/leg_kinematics.py'


class KinematicsTests(unittest.TestCase):
    def test_jacobian_and_virtual_work_cpu_cuda(self):
        self.assertTrue(PATH.exists(), 'Missing A1 foot kinematics')
        spec = importlib.util.spec_from_file_location('a1_kinematics', PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(12)
            q = torch.randn(32, 4, 3, device=device, dtype=torch.float64) * .3
            q[..., 1] += .8
            q[..., 2] -= 1.5
            p, jac = mod.foot_kinematics(q)
            self.assertEqual(p.shape, (32, 4, 3))
            self.assertEqual(jac.shape, (32, 4, 3, 3))
            eps = 1e-6
            for joint in range(3):
                delta = torch.zeros_like(q)
                delta[..., joint] = eps
                plus, _ = mod.foot_kinematics(q + delta)
                minus, _ = mod.foot_kinematics(q - delta)
                torch.testing.assert_close(jac[..., joint], (plus - minus) / (2 * eps),
                                           atol=1e-9, rtol=1e-7)
            velocity, force = torch.randn_like(q), torch.randn_like(p)
            foot_velocity = (jac @ velocity[..., None]).squeeze(-1)
            torque = (jac.transpose(-1, -2) @ force[..., None]).squeeze(-1)
            torch.testing.assert_close((torque * velocity).sum(-1), (force * foot_velocity).sum(-1))


if __name__ == '__main__':
    unittest.main()
