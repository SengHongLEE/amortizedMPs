"""Test the real torque method and IK without starting an Isaac Gym simulation."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / 'legged_gym'
source = ROOT / 'envs/a1/mp_multi_primitive.py'
tree = ast.parse(source.read_text())
method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
              and cls.name == 'MotorPrimitives' for node in cls.body
              if isinstance(node, ast.FunctionDef) and node.name == '_compute_torques')
namespace = {'torch': torch, 'np': np}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
spec = importlib.util.spec_from_file_location('mapping_dynamics', ROOT / 'utils/dynamics/a1/rhythmic_dynamics.py')
dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamics)


def make_env(device):
    robot = SimpleNamespace(hip_link_length_a1=.0838, thigh_link_length_a1=.213, calf_link_length_a1=.213)
    env = SimpleNamespace(device=device, num_envs=1,
                          cfg=SimpleNamespace(asset=robot, control=SimpleNamespace(action_scale=1.),
                                              env=SimpleNamespace(play=True)))
    env.ID = dynamics.Multi_Primitive_Dynamics(device=device)
    env.ID.oscillator = lambda: None
    env.ID.X[:, 0] = torch.tensor([.2, .4, .6, .8], device=device)
    env.ID.X[:, 1] = .05
    env.ID.X[:, 2] = torch.tensor([.2, 1.5, 3.8, 5.2], device=device)
    env.ID.X_dot[:, 2] = torch.tensor([11., 12., 13., 14.], device=device)
    env.commands = torch.tensor([[1., 0., .3]], device=device)
    env.feet_indices = torch.arange(4, device=device)
    env.contact_forces = torch.zeros(1, 4, 3, device=device)
    env.torques = torch.zeros(1, 12, device=device)
    env.dof_pos = torch.zeros_like(env.torques)
    env.dof_vel = torch.zeros_like(env.torques)
    env.p_gains, env.d_gains, env.torque_limits = 1., 0., 100.
    for name in ('phase', 'amplitude', 'amplitude_dot', 'mu', 'omega', 'x', 'z'):
        setattr(env, name + '_history', [])
    return env


class LegMappingTests(unittest.TestCase):
    def test_physical_legs_receive_their_own_trajectory(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            env = make_env(device)
            actual = namespace['_compute_torques'](env, torch.zeros(1, 6, device=device))
            expected = []
            # Physical joint blocks are FL, FR, RL, RR; legacy IK IDs differ.
            for physical_leg, ik_id, side in ((0, 1, 1), (1, 0, -1), (2, 3, 1), (3, 2, -1)):
                expected.append(env.ID.compute_inverse_kinematics(
                    env.cfg.asset, ik_id, env.ID.x[:, physical_leg],
                    side * env.cfg.asset.hip_link_length_a1 + env.ID.y[:, physical_leg],
                    env.ID.z[:, physical_leg]))
            torch.testing.assert_close(actual, torch.cat(expected, dim=1))
            self.assertTrue(torch.isfinite(actual).all())

    def test_phase_and_angular_velocity_logs_use_phase_channel(self):
        env = make_env('cpu')
        expected_phase = env.ID.X[:, 2].clone().numpy()
        expected_omega = env.ID.X_dot[:, 2].clone().numpy()
        namespace['_compute_torques'](env, torch.zeros(1, 6))
        np.testing.assert_allclose(env.phase_history[0], expected_phase)
        np.testing.assert_allclose(env.omega_history[0], expected_omega)
        # CPU logging must snapshot, not retain a view of mutable state.
        env.ID.X[:, 2].zero_()
        env.ID.X_dot[:, 2].zero_()
        np.testing.assert_allclose(env.phase_history[0], expected_phase)
        np.testing.assert_allclose(env.omega_history[0], expected_omega)


if __name__ == '__main__':
    unittest.main()
