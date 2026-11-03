"""Rear leg equilibrium limits for the current MP_full_a1 action layout."""
import unittest

import torch

from test_equilibrium_dynamics import module
from test_interoceptive_rewards import load_class


Config = load_class('mp_full_config.py', 'MotorPrimitivesCfg')


class RearEquilibriumTests(unittest.TestCase):
    def make(self):
        obj = module().AdaptiveEquilibriumDynamics(
            equilibrium_cfg=Config.control.equilibrium,
            timing_cfg=Config.control.timing,
            num_envs=1, device='cpu', time_step=Config.sim.dt,
            nominal_clearance=Config.asset.ground_clearance,
            max_clearance=Config.asset.max_ground_clearance)
        obj.oscillator = lambda: None
        return obj

    def test_rear_x_target_and_state_stay_in_configured_interval(self):
        lower, upper = Config.control.equilibrium.rear_x_offset_range
        self.assertEqual((lower, upper), (-.07, .03))
        obj = self.make()
        self.assertTrue(((obj.offset[0, 2:, 0] >= lower) &
                         (obj.offset[0, 2:, 0] <= upper)).all())
        actions = torch.zeros(1, Config.env.num_actions)
        command = torch.zeros(1, 3)
        for value, target in [(1., upper), (-1., lower), (0., (lower + upper) / 2),
                              (1., upper)]:
            actions[:, 11:13] = value
            for _ in range(250):
                obj.intrinsic_dynamics(actions, command)
                self.assertTrue(((obj.offset[0, 2:, 0] >= lower - 1e-6) &
                                 (obj.offset[0, 2:, 0] <= upper + 1e-6)).all())
            torch.testing.assert_close(obj.offset_target[0, 2:, 0],
                                       torch.full((2,), target))
        obj.reset(torch.tensor([0]))
        torch.testing.assert_close(obj.offset[0, 2:, 0], torch.full((2,), upper))

    def test_controller_can_run_without_optional_contact_sensor(self):
        obj = module().AdaptiveEquilibriumDynamics(
            equilibrium_cfg=Config.control.equilibrium,
            timing_cfg=Config.control.timing,
            num_envs=1, device='cpu', time_step=Config.sim.dt,
            nominal_clearance=Config.asset.ground_clearance,
            max_clearance=Config.asset.max_ground_clearance)
        actions = torch.zeros(1, Config.env.num_actions)
        trajectory = obj.intrinsic_dynamics(actions, torch.zeros(1, 3))
        self.assertTrue(all(torch.isfinite(axis).all() for axis in trajectory))

    def test_timing_actions_change_gait_frequency(self):
        obj = self.make()
        actions = torch.zeros(1, Config.env.num_actions)
        actions[:, 0] = 1  # trot, nominally 8 Hz
        actions[:, 17] = -1  # lowest frequency scale
        for _ in range(500):
            obj.intrinsic_dynamics(actions, torch.ones(1, 3))
        self.assertLess(obj.selected_frequencies(torch.tensor([1])).item(), 4.)
        self.assertEqual(obj.controller_observations().shape[-1],
                         Config.env.num_controller_observations)

    def test_canter_can_use_longer_stride_without_changing_trot(self):
        self.assertEqual(Config.asset.max_step_length, .1)
        self.assertEqual(Config.asset.gait_step_length_scales, [1., 1., 1.5, 1.5])
        obj = module().AdaptiveEquilibriumDynamics(
            equilibrium_cfg=Config.control.equilibrium,
            timing_cfg=Config.control.timing,
            num_envs=2, device='cpu', time_step=Config.sim.dt,
            nominal_clearance=Config.asset.ground_clearance,
            max_clearance=Config.asset.max_ground_clearance)
        obj._MAX_STEP_LEN = .1
        obj._gait_step_scales = torch.tensor(Config.asset.gait_step_length_scales)
        obj._gait_ids[:] = torch.tensor([1, 2])
        obj.X[:, 0, :] = obj.mu_up
        obj.X[:, 2, :] = torch.pi
        x, _, _ = obj._foot_trajectory(torch.tensor([[1., 0., 0.]] * 2), obj.X[:, 2, :])
        torch.testing.assert_close(x[:, 0], torch.tensor([.1, .15]), atol=1e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
