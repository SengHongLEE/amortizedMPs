"""Behavioral contracts for the proprio task's dynamical action space."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch

from test_interoceptive_rewards import load_class
from test_multi_primitive_dynamics import Multi

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'legged_gym/legged_gym/utils/dynamics/a1/equilibrium_dynamics.py'
Config = load_class('mp_proprio_terrain_config.py', 'MotorPrimitivesCfg')


def module():
    # A private package allows the production relative import without utils/__init__.
    import types
    package = types.ModuleType('_equilibrium_test')
    package.__path__ = [str(MODULE.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('_equilibrium_test.equilibrium_dynamics', MODULE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def limits():
    urdf = ET.parse(Config.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', str(ROOT / 'legged_gym')))
    return torch.tensor([
        [float(urdf.find(f".//joint[@name='{leg}_{joint}_joint']/limit").attrib[key])
         for key in ('lower', 'upper')]
        for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')
    ])


class EquilibriumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE.exists():
            cls.mod = module()

    def make(self, n=6, device='cpu'):
        self.assertTrue(MODULE.exists(), 'Missing equilibrium dynamics implementation')
        return self.mod.EquilibriumPrimitiveDynamics(
            equilibrium_cfg=Config.control.equilibrium, num_envs=n, device=device,
            nominal_clearance=Config.asset.ground_clearance,
            max_clearance=Config.asset.max_ground_clearance)

    def test_zero_equilibrium_preserves_all_six_nominal_gaits(self):
        obj = self.make()
        baseline = Multi(num_envs=6, device='cpu',
                         nominal_clearance=Config.asset.ground_clearance,
                         max_clearance=Config.asset.max_ground_clearance)
        a = torch.zeros(6, 17)
        a[:, 0] = torch.arange(6)
        cmd = torch.tensor([[.5, 0., .2]]).expand(6, -1)
        for _ in range(30):
            expected = torch.stack(baseline.intrinsic_dynamics(a[:, :5], cmd), -1)
            actual = torch.stack(obj.intrinsic_dynamics(a, cmd), -1)
            torch.testing.assert_close(actual, expected)

    def test_axis_major_leg_mapping_convergence_and_stance_height(self):
        obj = self.make(n=12)
        obj.oscillator = lambda: None
        obj.X[:, 2] = 1.5 * torch.pi  # stance, so swing lift cannot mask z offsets
        a = torch.zeros(12, 17)
        a[:, 5:] = torch.eye(12)
        cmd = torch.zeros(12, 3)
        initial = torch.stack(obj.intrinsic_dynamics(torch.zeros_like(a), cmd), -1)
        for _ in range(2400):
            feet = torch.stack(obj.intrinsic_dynamics(a, cmd), -1)
        expected = torch.eye(12).reshape(12, 3, 4).transpose(1, 2)
        expected *= torch.tensor(Config.control.equilibrium.offset_limits)
        torch.testing.assert_close(feet - initial, expected, atol=2e-5, rtol=0)
        torch.testing.assert_close(obj.offset, expected, atol=2e-5, rtol=0)

    def test_step_is_filtered_bounded_and_contact_slows_motion(self):
        obj = self.make(n=2)
        obj.oscillator = lambda: None
        obj.X[:, 2] = .5  # both nominally swinging
        a = torch.ones(2, 17)
        a[:, 0] = 0
        contact = torch.tensor([[0., 0., 0., 0.], [20., 20., 20., 20.]])
        old = obj.offset.clone()
        for _ in range(150):
            obj.intrinsic_dynamics(a, torch.zeros(2, 3), contact_forces=contact)
            speed = (obj.offset - old).abs() / obj._dt
            cfg = Config.control.equilibrium
            self.assertTrue((speed[0] <= torch.tensor(cfg.swing_velocity_limits) + 1e-5).all())
            self.assertTrue((speed[1] <= torch.tensor(cfg.stance_velocity_limits) + 1e-5).all())
            self.assertTrue((obj.offset.abs() <= torch.tensor(cfg.offset_limits) + 1e-6).all())
            old = obj.offset.clone()
        self.assertTrue((obj.offset[0] > obj.offset[1]).all())
        before, velocity = obj.offset.clone(), obj.offset_velocity.clone()
        obj.reset(torch.tensor([0]))
        self.assertFalse(obj.offset[0].any())
        self.assertFalse(obj.offset_velocity[0].any())
        torch.testing.assert_close(obj.offset[1], before[1])
        torch.testing.assert_close(obj.offset_velocity[1], velocity[1])

    def test_internal_state_is_finite_and_matches_config_dimensions(self):
        obj = self.make()
        state = obj.controller_observations()
        self.assertEqual(state.shape, (6, Config.env.num_controller_observations))
        self.assertTrue(torch.isfinite(state).all())
        before = state.clone()
        obj.offset_velocity[:, 0, 0] = .1
        self.assertFalse(torch.equal(before, obj.controller_observations()))

    def test_workspace_extremes_produce_finite_urdf_bounded_targets(self):
        obj = self.make()
        torch.manual_seed(19)
        target = torch.randn(256, 4, 3) * .5
        target[0] = 0.
        q, achieved = self.mod.bounded_inverse_kinematics(
            target, Config.asset, limits(), Config.control.equilibrium.joint_limit_margin)
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.isfinite(achieved).all())
        self.assertTrue((q >= limits()[:, 0] + .01999).all())
        self.assertTrue((q <= limits()[:, 1] - .01999).all())
        # Verify physical FK independently, including left/right hip offsets.
        joint = q.reshape(-1, 4, 3)
        h, t, k = joint.unbind(-1)
        side = torch.tensor([1., -1., 1., -1.])
        l1, l2, l3 = .0838, .2, .2
        z = -l2 * torch.cos(t) - l3 * torch.cos(t + k)
        expected = torch.stack((
            -l2 * torch.sin(t) - l3 * torch.sin(t + k),
            side * l1 * torch.cos(h) - z * torch.sin(h),
            side * l1 * torch.sin(h) + z * torch.cos(h)), -1)
        torch.testing.assert_close(achieved, expected)

    def test_nominal_reachable_target_is_not_distorted(self):
        obj = self.make()
        a, cmd = torch.zeros(6, 17), torch.zeros(6, 3)
        xs, ys, zs = obj.intrinsic_dynamics(a, cmd)
        ys += torch.tensor([1., -1., 1., -1.]) * Config.asset.hip_link_length_a1
        target = torch.stack((xs, ys, zs), -1)
        q, achieved = self.mod.bounded_inverse_kinematics(
            target, Config.asset, limits(), Config.control.equilibrium.joint_limit_margin)
        torch.testing.assert_close(achieved, target)

    def test_folded_leg_projection_does_not_amplify_small_foot_changes(self):
        obj = self.make()
        # Inboard + upward equilibrium at maximum swing lift approaches the
        # hip-axis singularity. Millimetres must not create half-radian jumps.
        feet = torch.tensor([
            [-.0063, .0588, -.0610], [-.0017, .0588, -.0604],
            [.0028, .0588, -.0601], [.0071, .0588, -.0600],
            [.0114, .0588, -.0603]])[:, None, :].repeat(1, 4, 1)
        feet[:, [1, 3], 1] *= -1
        q, _ = self.mod.bounded_inverse_kinematics(feet, Config.asset, limits(), .02)
        self.assertLess(torch.diff(q, dim=0).abs().max().item(), .15)

    def test_old_action_contract_is_rejected(self):
        obj = self.make()
        with self.assertRaisesRegex(ValueError, '17'):
            obj.intrinsic_dynamics(torch.zeros(6, 5), torch.zeros(6, 3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cpu_cuda_agree_through_switches_contacts_and_extreme_actions(self):
        cpu, gpu = self.make(), self.make(device='cuda')
        torch.manual_seed(23)
        for step in range(40):
            a = torch.rand(6, 17) * 4 - 2
            a[:, 0] = (torch.arange(6) + step // 10) % 6
            cmd = torch.tensor([[.5, 0., .3]]).expand(6, -1)
            forces = (torch.rand(6, 4) > .5).float() * 30
            expected = torch.stack(cpu.intrinsic_dynamics(a, cmd, contact_forces=forces), -1)
            actual = torch.stack(gpu.intrinsic_dynamics(
                a.cuda(), cmd.cuda(), contact_forces=forces.cuda()), -1)
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gpu.controller_observations().cpu(),
                                   cpu.controller_observations(), atol=1e-4, rtol=1e-4)
        target = actual.clone()
        target[..., 1] += target.new_tensor([1., -1., 1., -1.]) * .0838
        q_cpu, p_cpu = self.mod.bounded_inverse_kinematics(target.cpu(), Config.asset, limits(), .02)
        q_gpu, p_gpu = self.mod.bounded_inverse_kinematics(target, Config.asset, limits().cuda(), .02)
        torch.testing.assert_close(q_gpu.cpu(), q_cpu)
        torch.testing.assert_close(p_gpu.cpu(), p_cpu)

    def test_bad_filter_config_is_rejected(self):
        from copy import copy
        values = {name: getattr(Config.control.equilibrium, name)
                  for name in dir(Config.control.equilibrium) if not name.startswith('_')}
        for field, bad in [('offset_limits', [0., .01, .01]),
                           ('swing_velocity_limits', [1., 2.]),
                           ('stance_velocity_limits', [1., 1., 1.]),
                           ('frequency_hz', float('nan'))]:
            cfg = SimpleNamespace(**copy(values))
            setattr(cfg, field, bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.mod.EquilibriumPrimitiveDynamics(equilibrium_cfg=cfg, device='cpu')


class EnvironmentContracts(unittest.TestCase):
    def make_env(self):
        mod = module()
        cls = load_class('mp_proprio_terrain.py', 'MotorPrimitives', {
            '_compute_torques', 'compute_observations', '_get_noise_scale_vec',
            '_validate_action_config'})
        cls._compute_torques.__globals__['bounded_inverse_kinematics'] = mod.bounded_inverse_kinematics
        cls._validate_action_config.__globals__['EquilibriumPrimitiveDynamics'] = mod.EquilibriumPrimitiveDynamics
        cls._validate_action_config.__globals__['AdaptiveEquilibriumDynamics'] = mod.AdaptiveEquilibriumDynamics
        cls._validate_action_config.__globals__['ModalEquilibriumDynamics'] = mod.ModalEquilibriumDynamics
        env = cls()
        env.device, env.num_envs = 'cpu', 2
        env.cfg = SimpleNamespace(
            asset=Config.asset, noise=Config.noise,
            control=SimpleNamespace(action_scale=.5, equilibrium=Config.control.equilibrium),
            env=SimpleNamespace(play=False))
        env.ID = mod.EquilibriumPrimitiveDynamics(
            equilibrium_cfg=Config.control.equilibrium, num_envs=2, device='cpu',
            nominal_clearance=.15, max_clearance=.20)
        env.torques = torch.zeros(2, 12)
        env.commands = torch.zeros(2, 3)
        env.feet_indices = torch.arange(4)
        env.contact_forces = torch.zeros(2, 4, 3)
        env.dof_pos = torch.zeros(2, 12)
        env.dof_vel = torch.zeros(2, 12)
        env.dof_pos_hard_limits = limits()
        env.p_gains, env.d_gains = torch.full((12,), 100.), torch.full((12,), 2.)
        env.torque_limits = torch.full((12,), 33.5)
        env.base_ang_vel = torch.ones(2, 3)
        env.base_lin_vel = torch.ones(2, 3)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).expand(2, -1)
        env.root_states = torch.zeros(2, 13)
        env.default_dof_pos = torch.zeros(1, 12)
        env.actions = torch.zeros(2, 17)
        env.actions[:, 0] = torch.tensor([1., 5.])
        env.commands_scale = torch.ones(3)
        env.obs_scales = Config.normalization.obs_scales
        env.obs_buf = torch.zeros(2, 5, Config.env.num_observations)
        env.privileged_obs_buf = torch.zeros(2, Config.env.num_privileged_obs)
        env.noise_scale_vec = env._get_noise_scale_vec(env.cfg)
        return env

    def test_continuous_action_scale_does_not_scale_gait_ids(self):
        env = self.make_env()
        torque = env._compute_torques(env.actions)
        self.assertTrue(torch.isfinite(torque).all())
        self.assertTrue((torque.abs() <= 33.5).all())
        torch.testing.assert_close(env.ID._gait_ids, torch.tensor([1, 5]))

    def test_actions_drive_equilibrium_not_instant_joint_offsets(self):
        env, baseline = self.make_env(), self.make_env()
        env.actions[:, 5:] = 1.
        env._compute_torques(env.actions)
        baseline._compute_torques(baseline.actions)
        self.assertLess((env.dof_des_pos - baseline.dof_des_pos).abs().max().item(), .001)
        self.assertTrue(env.ID.offset.abs().any())
        self.assertTrue((env.dof_des_pos >= limits()[:, 0] + .01999).all())
        self.assertTrue((env.dof_des_pos <= limits()[:, 1] - .01999).all())

    def test_noise_preserves_actions_commands_and_internal_states(self):
        env = self.make_env()
        env.add_noise = False
        env.compute_observations()
        clean = env.obs_buf[:, -1].clone()
        self.assertEqual(clean.shape[1], 124)
        torch.testing.assert_close(clean[:, -70:], env.ID.controller_observations())
        env.add_noise = True
        env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:, -1, 30:], clean[:, 30:])
        self.assertFalse(torch.equal(env.obs_buf[:, -1, :30], clean[:, :30]))
        torch.testing.assert_close(env.privileged_obs_buf[:, -70:], clean[:, -70:])

    def test_saved_old_observation_contract_is_rejected_before_simulation(self):
        env = self.make_env()
        env.cfg.env = SimpleNamespace(**{
            key: getattr(Config.env, key) for key in dir(Config.env) if not key.startswith('_')})
        env._validate_action_config()
        env.cfg.env.num_observations = 54
        with self.assertRaisesRegex(ValueError, '124'):
            env._validate_action_config()
        del env.cfg.env.action_space
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            env._validate_action_config()


if __name__ == '__main__':
    unittest.main()
