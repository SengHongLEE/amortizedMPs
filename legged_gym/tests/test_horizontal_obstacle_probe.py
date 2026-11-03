"""Horizontal riser impacts must not become false support contacts."""
import importlib.util
from pathlib import Path
import unittest
import torch
from test_equilibrium_dynamics import module, Config
from test_adaptive_equilibrium import timing_config

PATH = Path(__file__).resolve().parents[1] / 'legged_gym/scripts/horizontal_obstacle_probe.py'


class HorizontalTests(unittest.TestCase):
    def test_no_horizontal_input_preserves_original_dynamics(self):
        spec = importlib.util.spec_from_file_location('horizontal_probe', PATH)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        cls = module().AdaptiveEquilibriumDynamics
        options = dict(timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
                       num_envs=2, device='cpu', nominal_clearance=.15, max_clearance=.2)
        old, new = cls(**options), cls(**options)
        probe = helper.HorizontalObstacleProbe(new)
        action = torch.zeros(2, 19)
        action[:, 0] = 2
        for _ in range(50):
            force = torch.zeros(2, 4, 3)
            force[:, :, 2] = (torch.rand(2, 4) > .5) * 20.
            probe.set_forces(force)
            a = torch.stack(old.intrinsic_dynamics(action, torch.zeros(2, 3), contact_forces=force[:, :, 2]), -1)
            b = torch.stack(new.intrinsic_dynamics(action, torch.zeros(2, 3), contact_forces=force[:, :, 2]), -1)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(old.controller_observations(), new.controller_observations(), atol=0, rtol=0)

    def test_late_swing_wall_contact_drives_obstacle_but_not_support(self):
        self.assertTrue(PATH.exists(), 'Missing horizontal collision probe')
        spec = importlib.util.spec_from_file_location('horizontal_probe', PATH)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        obj = module().AdaptiveEquilibriumDynamics(
            timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
            num_envs=2, device='cpu', nominal_clearance=.15, max_clearance=.2)
        probe = helper.HorizontalObstacleProbe(obj)
        beta = obj.selected_duty_factors(obj._gait_ids)
        obj.X[0, 2] = .85 * torch.pi * 2 * (1 - beta[0])  # late swing
        obj.X[1, 2] = torch.pi * (2 - 1.5 * beta[1])  # mid stance
        force = torch.zeros(2, 4, 3)
        force[:, :, 0] = 30.
        probe.set_forces(force)
        obj._set_contact(None, force[:, :, 2])
        for _ in range(25):
            obj._update_reflex(.001)
        self.assertTrue((obj.q_obstacle[0] > .5).all())
        self.assertFalse(obj.q_obstacle[1].any())
        self.assertFalse(obj._contact.any())
        self.assertFalse(obj.q_early.any())
        self.assertTrue((obj.q_late[1] > 0).all())
        obj.reset(torch.tensor([0]))
        self.assertFalse(probe.hit[0].any())
        self.assertTrue(probe.hit[1].all())


if __name__ == '__main__':
    unittest.main()
