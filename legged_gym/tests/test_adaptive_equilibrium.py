"""Timing and clearance controls must be bounded, observable and resettable."""
from types import SimpleNamespace
import unittest
import torch
from test_equilibrium_dynamics import module, Config


def timing_config(**overrides):
    cfg = dict(frequency_scale_range=[.35, 1.2], duty_offset_range=[-.15, .25],
               duty_factor_range=[.3, .8], clearance_range=[.06, .20],
               time_constant=.10, align_stroke=True)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class AdaptiveTests(unittest.TestCase):
    def make(self, device='cpu', **cfg):
        mod = module()
        self.assertTrue(hasattr(mod, 'AdaptiveEquilibriumDynamics'),
                        'Missing adaptive dynamical action space')
        return mod.AdaptiveEquilibriumDynamics(
            timing_cfg=timing_config(**cfg), equilibrium_cfg=Config.control.equilibrium,
            num_envs=6, device=device, nominal_clearance=.15, max_clearance=.20)

    def test_neutral_parameters_preserve_v1_when_stroke_alignment_disabled(self):
        adaptive = self.make(align_stroke=False)
        base = module().EquilibriumPrimitiveDynamics(
            equilibrium_cfg=Config.control.equilibrium, num_envs=6, device='cpu',
            nominal_clearance=.15, max_clearance=.20)
        actions = torch.zeros(6, 23)
        actions[:, 0] = torch.arange(6)
        commands = torch.tensor([[.5, 0., .1]]).expand(6, -1)
        for _ in range(40):
            actual = torch.stack(adaptive.intrinsic_dynamics(actions, commands), -1)
            expected = torch.stack(base.intrinsic_dynamics(actions[:, :17], commands), -1)
            torch.testing.assert_close(actual, expected)
        self.assertEqual(adaptive.controller_observations().shape, (6, 72))

    def test_timing_filters_and_partial_reset_cpu_cuda(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            obj = self.make(device=device)
            actions = torch.ones(6, 23, device=device)
            actions[:, 0] = torch.arange(6, device=device)
            actions[:, 21] = -1.  # slower cycle, longer stance
            cmd = torch.zeros(6, 3, device=device)
            obj.intrinsic_dynamics(actions, cmd)
            self.assertGreater(obj.frequency_scale.min().item(), .35)
            self.assertLess(obj.frequency_scale.max().item(), 1.)
            for _ in range(800):
                obj.intrinsic_dynamics(actions, cmd)
            torch.testing.assert_close(obj.frequency_scale, torch.full_like(obj.frequency_scale, .35),
                                       atol=.001, rtol=0)
            self.assertTrue((obj.selected_duty_factors(obj._gait_ids) <= .8).all())
            self.assertTrue((obj.nominal_clearance <= .20).all())
            self.assertTrue(torch.isfinite(obj.controller_observations()).all())
            others = obj.controller_observations()[1:].clone()
            obj.reset(torch.tensor([0], device=device))
            self.assertEqual(obj.frequency_scale[0].item(), 1.)
            self.assertEqual(obj.duty_offset[0].item(), 0.)
            torch.testing.assert_close(obj.controller_observations()[1:], others)

    def test_stroke_turns_at_touchdown_for_every_duty_factor(self):
        obj = self.make()
        obj.duty_offset.fill_(.1)
        obj._gait_ids[:] = torch.arange(6)
        duty = obj.selected_duty_factors(obj._gait_ids)
        touchdown = 2 * torch.pi * (1 - duty)
        obj.X[:, 0] = obj.mu_up
        cmd = torch.tensor([[1., 0., 0.]]).expand(6, -1)
        positions = []
        for delta in [-.01, 0., .01]:
            obj.X[:, 2] = touchdown + delta
            positions.append(obj._foot_trajectory(cmd, obj._vertical_phase())[0].clone())
        self.assertTrue((positions[1] > positions[0]).all())
        self.assertTrue((positions[1] > positions[2]).all())
        torch.testing.assert_close(positions[1], torch.full_like(positions[1], obj._MAX_STEP_LEN))

    def test_invalid_ranges_fail(self):
        for cfg in [dict(frequency_scale_range=[0., 1.]), dict(duty_factor_range=[.3, 1.]),
                    dict(clearance_range=[.16, .20]), dict(time_constant=0)]:
            with self.assertRaises(ValueError):
                self.make(**cfg)


if __name__ == '__main__':
    unittest.main()
