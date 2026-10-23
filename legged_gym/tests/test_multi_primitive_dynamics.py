"""Exercise the actual dynamics module without importing Isaac Gym."""
import importlib.util
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/dynamics/a1/rhythmic_dynamics.py'
spec = importlib.util.spec_from_file_location('rhythmic_dynamics', PATH)
dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamics)
Multi = dynamics.Multi_Primitive_Dynamics


class MultiPrimitiveTests(unittest.TestCase):
    def test_z_support_fraction_matches_each_primitive(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            samples = 1000
            obj = Multi(num_envs=6 * samples, device=device)
            # Freeze integration to sample the actual output mapping over a cycle.
            obj.oscillator = lambda: None
            phases = (torch.arange(samples, device=device) + .5) * (2 * torch.pi / samples)
            obj.X[:, 2] = phases.repeat(6)[:, None]
            obj.X[:, 1] = .15
            obj.stance_extension[:] = .02
            actions = torch.zeros(6 * samples, 10, device=device)
            actions[:, 0] = torch.arange(6, device=device).repeat_interleave(samples)
            z = obj.intrinsic_dynamics(actions, torch.zeros(6 * samples, 3, device=device))[2]
            fraction = (z[:, 0] < -obj._robot_height).float().view(6, samples).mean(1)
            expected = torch.tensor([.6, .5, .5, .45, .4, .35], device=device)
            torch.testing.assert_close(fraction, expected)
            # Switching IDs immediately changes the selected duty factors.
            actions[:, 0] = actions[:, 0].roll(samples)
            z = obj.intrinsic_dynamics(actions, torch.zeros(6 * samples, 3, device=device))[2]
            fraction = (z[:, 0] < -obj._robot_height).float().view(6, samples).mean(1)
            torch.testing.assert_close(fraction, expected.roll(1))

    def test_z_duty_boundaries_and_amplitude(self):
        obj = Multi(num_envs=6, device='cpu')
        obj.oscillator = lambda: None
        beta = torch.tensor([.6, .5, .5, .45, .4, .35])
        touchdown = 2 * torch.pi * (1 - beta)
        obj.X[:, 2] = torch.stack((torch.zeros(6), touchdown / 2, touchdown,
                                   touchdown + torch.pi * beta), dim=1)
        obj.X[:, 1] = .15
        obj.stance_extension[:] = .02
        actions = torch.zeros(6, 5)
        actions[:, 0] = torch.arange(6)
        z = obj.intrinsic_dynamics(actions, torch.zeros(6, 3))[2]
        expected = torch.tensor([-.3, -.15, -.3, -.32]).expand(6, -1)
        torch.testing.assert_close(z, expected)
        for eps in (-1.e-6, 1.e-6):
            obj.X[:, 2] = (touchdown[:, None] + eps).expand(-1, 4)
            z = obj.intrinsic_dynamics(actions, torch.zeros(6, 3))[2]
            torch.testing.assert_close(z, torch.full_like(z, -.3), atol=1.e-6, rtol=0)

    def test_switch_restarts_only_reference_clock_once(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            obj = Multi(num_envs=3, device=device)
            actions = torch.zeros(3, 10, device=device)
            command = torch.zeros(3, 3, device=device)
            for _ in range(13):
                obj.intrinsic_dynamics(actions, command)
            previous_phi = obj.phi.clone()
            previous_theta = obj.X[:, 2].clone()
            actions[1, 0] = 2
            obj.intrinsic_dynamics(actions, command)
            advance = obj.omega_g * obj._dt
            expected = previous_phi + advance
            expected[1] = advance[1]
            torch.testing.assert_close(obj.phi, expected)
            # Actual legs integrate forward; they must not jump to the template.
            displacement = obj.wrap_to_pi(obj.X[:, 2] - previous_theta)
            self.assertTrue((displacement > 0).all())
            self.assertTrue((displacement < 1.5 * advance).all())
            obj.intrinsic_dynamics(actions, command)
            torch.testing.assert_close(obj.phi, expected + advance)
            # Legacy extra action columns are ignored and preserve the clock.
            previous_phi = obj.phi.clone()
            actions[:, 9] = .5
            obj.intrinsic_dynamics(actions, command)
            torch.testing.assert_close(obj.phi, previous_phi + obj.omega_g * obj._dt)

    def test_all_transitions_converge_to_restarted_absolute_reference(self):
        obj = Multi(num_envs=16, device='cpu')
        actions = torch.zeros(16, 10)
        actions[:, 0] = torch.arange(4).repeat_interleave(4)
        command = torch.zeros(16, 3)
        for _ in range(137):
            obj.intrinsic_dynamics(actions, command)
        previous_phi = obj.phi.clone()
        switched = actions[:, 0] != torch.arange(4).repeat(4)
        actions[:, 0] = torch.arange(4).repeat(4)
        for _ in range(600):
            obj.intrinsic_dynamics(actions, command)
        expected_phi = previous_phi.clone()
        expected_phi[switched] = 0.
        expected_phi = (expected_phi + .6 * obj.omega_g).remainder(2 * torch.pi)
        self.assertLess(obj.wrap_to_pi(obj.phi - expected_phi).abs().max().item(), 1e-4)
        error = obj.wrap_to_pi(obj.X[:, 2] - expected_phi - obj.gait_phase_delta)
        self.assertLess(error.abs().max().item(), 1e-4)

    def test_default_and_explicit_initialization(self):
        for phase in (None, [0., -1., -1., 0.]):
            obj = Multi(num_envs=3, device='cpu', init_phase=phase)
            expected = torch.tensor(phase or [0., -1., -.5, .5]) * torch.pi
            torch.testing.assert_close(obj.X[:, 2], expected.remainder(2 * torch.pi).expand(3, -1))

    def test_batch_matches_independent_environments(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            obj = Multi(num_envs=4, device=device, time_step=.0025)
            singles = [Multi(device=device, time_step=.0025) for _ in range(4)]
            actions = torch.zeros(4, 5, device=device)
            actions[:, 0] = torch.arange(4, device=device)
            actions[:, 4] = torch.linspace(-1, 1, 4, device=device)
            command = torch.tensor([[1., 0., .2]], device=device).expand(4, -1)
            contact = torch.eye(4, 4, device=device).bool()
            for step in range(8):
                if step == 4:
                    actions[:, 0] = actions[:, 0].roll(1)
                outputs = obj.intrinsic_dynamics(actions, command, foot_contact=contact)
                for i, single in enumerate(singles):
                    expected = single.intrinsic_dynamics(actions[i:i+1], command[i:i+1], foot_contact=contact[i:i+1])
                    for actual, target in zip(outputs, expected):
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual[i:i+1], target)
            expected_phase = torch.tensor([obj.primitive_dict[int(i)] for i in actions[:, 0]], device=device) * torch.pi
            torch.testing.assert_close(obj.delta, expected_phase)
            torch.testing.assert_close(obj.target_diff, expected_phase[:, None, :] - expected_phase[:, :, None])

    def test_partial_and_empty_reset(self):
        obj = Multi(num_envs=3, device='cpu')
        actions = torch.ones(3, 5)
        actions[:, 0] = torch.tensor([1., 2., 3.])
        obj.intrinsic_dynamics(actions, torch.ones(3, 3), foot_contact=torch.ones(3, 4, dtype=torch.bool))
        before = {key: value.clone() for key, value in vars(obj).items() if isinstance(value, torch.Tensor)}
        obj.reset([])
        for key, value in before.items():
            torch.testing.assert_close(getattr(obj, key), value)
        obj.reset([1])
        fresh = Multi(device='cpu')
        for key in ('X', 'X_dot', 'd2X', 'gait_phase_delta', 'target_diff', 'delta', 'omega_g', 'nominal_clearance', 'q_early', 'q_late', 'q_obstacle', '_gait_ids'):
            value = getattr(obj, key)
            torch.testing.assert_close(value[[0, 2]], before[key][[0, 2]])
            torch.testing.assert_close(value[1:2], getattr(fresh, key))

    def test_invalid_inputs(self):
        obj = Multi(device='cpu')
        for index in (-1., 6., 1.5, float('nan'), float('inf')):
            actions = torch.zeros(1, 10)
            actions[0, 0] = index
            with self.assertRaises(ValueError):
                obj.intrinsic_dynamics(actions, torch.zeros(1, 3))
        for actions, command in ((torch.zeros(1, 4), torch.zeros(1, 3)), (torch.zeros(2, 10), torch.zeros(1, 3)), (torch.zeros(10), torch.zeros(1, 3)), (torch.zeros(1, 10), torch.zeros(1, 2))):
            with self.assertRaises(ValueError):
                obj.intrinsic_dynamics(actions, command)

    def test_parent_single_gait_still_runs(self):
        obj = dynamics.Interoceptive_Reflex_Dynamics(num_envs=3, device='cpu')
        obj.intrinsic_dynamics(torch.zeros(3, 5), torch.zeros(3, 3))
        obj.reset([1])
        self.assertEqual(obj.X.shape, (3, 3, 4))
        self.assertTrue(torch.isfinite(obj.X).all())

    def test_continuous_actions_are_clipped_without_clipping_gait_id(self):
        obj, reference = Multi(device='cpu'), Multi(device='cpu')
        actions = torch.tensor([[3., -2., 2., -3., 3.]])
        clipped = actions.clone()
        clipped[:, 1:] = clipped[:, 1:].clamp(-1, 1)
        actual = obj.intrinsic_dynamics(actions, torch.ones(1, 3))
        expected = reference.intrinsic_dynamics(clipped, torch.ones(1, 3))
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        torch.testing.assert_close(obj.gait_phase_delta[0], torch.tensor(obj.primitive_dict[3]) * torch.pi)

    def test_contact_force_and_boolean_inputs_match(self):
        obj, reference = Multi(device='cpu'), Multi(device='cpu')
        actions, command = torch.zeros(1, 10), torch.ones(1, 3)
        forces = torch.tensor([[0., 2., .5, 3.]])
        for _ in range(10):
            actual = obj.intrinsic_dynamics(actions, command, contact_forces=forces)
            expected = reference.intrinsic_dynamics(actions, command, foot_contact=forces > 1.)
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target)
        with self.assertRaises(ValueError):
            obj.intrinsic_dynamics(actions, command, foot_contact=forces.bool(), contact_forces=forces)
        with self.assertRaises(ValueError):
            obj.intrinsic_dynamics(actions, command, foot_contact=torch.zeros(1, 3))
        obj.intrinsic_dynamics(actions, command)
        self.assertFalse(obj._contact_available)


if __name__ == '__main__':
    unittest.main()
