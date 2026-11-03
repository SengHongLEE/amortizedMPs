import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rsl_rl.modules.terrain_hold_snn import (
    HoldBins, TerrainHoldSNN, verified_hold_label, overestimation_counts,
)
from rsl_rl.modules.terrain_scan_encoder import TerrainScanEncoder
from rsl_rl.modules.terrain_hold_experiment import (
    HoldClock, FixedTerrainClock, HeldPolicyController, compare_quality, candidate_result,
    planar_speed_error,
)
from rsl_rl.modules.terrain_hold_experiment import conservative_outcomes


class TerrainHoldTests(unittest.TestCase):
    def test_bins_enforce_original_call_intervals(self):
        with self.assertRaises(ValueError):
            HoldBins((14, 30), (3, 6))
        with self.assertRaises(ValueError):
            HoldBins((15, 15), (3, 6))
        bins = HoldBins((15, 30), (3, 6))
        self.assertEqual(bins.gait[0], 15)
        self.assertEqual(bins.modulation[0], 3)

    def test_label_distinguishes_failure_and_right_censoring(self):
        self.assertEqual(verified_hold_label((15, 30, 45), ('pass', 'fail', 'unobserved')),
                         (15, False, True))
        self.assertEqual(verified_hold_label((15, 30), ('pass', 'pass')),
                         (30, True, True))
        self.assertEqual(verified_hold_label((15, 30), ('pass', 'unobserved')),
                         (15, True, True))
        self.assertEqual(verified_hold_label((15, 30), ('fail', 'unobserved')),
                         (15, False, False))
        self.assertEqual(verified_hold_label((15, 30, 45), ('pass', 'fail', 'pass')),
                         (15, False, True))

    def test_censored_overestimation_is_not_called_failure(self):
        # One known overestimate; one censored unknown; one censored known safe.
        n, d, unknown = overestimation_counts(
            torch.tensor([30, 30, 15]), torch.tensor([15, 15, 30]),
            torch.tensor([False, True, True]))
        self.assertEqual((n, d, unknown), (1, 2, 1))

    def test_two_bounded_outputs_and_encoder_frozen(self):
        bins = HoldBins((15, 30), (3, 6, 9))
        encoder = TerrainScanEncoder((2, 16, 5), 32)
        model = TerrainHoldSNN(encoder, bins, observation_dim=262, hidden_dims=(16,))
        obs = torch.randn(4, 262, requires_grad=True)
        gait, modulation = model(obs)
        self.assertEqual((tuple(gait.shape), tuple(modulation.shape)), ((4, 2), (4, 3)))
        durations = model.decode(obs)
        self.assertEqual(tuple(durations.shape), (4, 2))
        self.assertTrue(torch.all((durations[:, 0] >= 15) & (durations[:, 0] <= 30)))
        self.assertTrue(torch.all((durations[:, 1] >= 3) & (durations[:, 1] <= 9)))
        (gait.sum() + modulation.sum()).backward()
        self.assertTrue(all(p.grad is None for p in model.terrain_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.snn.parameters()))
        second = model(obs.detach())
        torch.testing.assert_close(gait.detach(), second[0])
        torch.testing.assert_close(modulation.detach(), second[1])

    def test_fixed_checkpoint_gait_encoder_remains_bitwise_frozen_after_update(self):
        root = Path(__file__).resolve().parents[2]
        checkpoint = (root / 'legged_gym/logs/A1/'
                      '1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        prefix = 'actor.terrain_encoder.'
        archived = {key[len(prefix):]: value.clone() for key, value in
                    saved['primitive_selection_model_state_dict'].items()
                    if key.startswith(prefix)}
        encoder = TerrainScanEncoder((2, 16, 5), 32)
        encoder.load_state_dict(archived, strict=True)
        model = TerrainHoldSNN(encoder, HoldBins((15, 30), (3, 6)), 262,
                               hidden_dims=(8,), snn_steps=2)
        optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)
        logits = model(torch.randn(2, 262))
        (logits[0].square().mean() + logits[1].square().mean()).backward()
        optimizer.step()
        for key, value in model.terrain_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, archived[key]), key)
        self.assertFalse(any(parameter.requires_grad for parameter in
                             model.terrain_encoder.parameters()))

    def test_clock_is_per_env_and_never_exceeds_original_frequency(self):
        clock = HoldClock(2, gait_min=15, modulation_min=3, device='cpu')
        self.assertEqual(clock.due(0)[0].tolist(), [0, 1])
        clock.mark('gait', torch.tensor([0, 1]), 0, torch.tensor([15, 30]))
        clock.mark('modulation', torch.tensor([0, 1]), 0, torch.tensor([3, 6]))
        self.assertEqual(clock.due(3)[0].tolist(), [])
        self.assertEqual(clock.due(3)[1].tolist(), [0])
        self.assertEqual(clock.due(15)[0].tolist(), [0])
        with self.assertRaises(ValueError):
            clock.mark('gait', torch.tensor([0]), 15, torch.tensor([14]))
        clock.mark('gait', torch.tensor([0]), 15, torch.tensor([15]))
        clock.reset(torch.tensor([1]))
        self.assertEqual(clock.due(16)[0].tolist(), [1])

    def test_fixed_terrain_clock_wakes_new_episode_without_self_scheduling(self):
        clock = FixedTerrainClock(2, period=60, device='cpu')
        self.assertEqual(clock.due(0).tolist(), [0, 1])
        clock.mark(torch.tensor([0, 1]), 0)
        clock.reset(torch.tensor([1]))
        self.assertEqual(clock.due(1).tolist(), [1])
        clock.mark(torch.tensor([1]), 1)
        self.assertEqual(clock.due(60).tolist(), [0])
        self.assertEqual(clock.due(61).tolist(), [0, 1])

    def test_partial_policy_update_preserves_other_environment_reservoir(self):
        obs = torch.zeros(2, 5, 262)
        env = SimpleNamespace(
            num_envs=2, cfg=SimpleNamespace(control=SimpleNamespace(
                primitive_selection_update_cycle=15, modulation_param_update_cycle=3)),
            step=lambda actions: (obs, None, None, torch.zeros(2, dtype=torch.bool), {}),
        )
        runner = SimpleNamespace(
            env=env, device='cpu', actions=torch.zeros(2, 19),
            primitive_selections=torch.zeros(2, dtype=torch.long),
            last_primitive_selections=torch.zeros(2, dtype=torch.long),
            modulation_params=torch.zeros(2, 18), last_modulation_params=torch.zeros(2, 18),
            primitive_selection_reservoir_states=torch.zeros(2, 4),
            modulation_param_reservoir_states=None,
        )
        calls = {'gait': [], 'modulation': []}
        def gait(inputs, states):
            calls['gait'].append(len(inputs))
            return torch.ones(len(inputs), dtype=torch.long), states + 1
        def modulation(inputs):
            calls['modulation'].append(len(inputs))
            return torch.ones(len(inputs), 18)
        controller = HeldPolicyController(runner, gait, modulation)
        controller.step(obs, 0, torch.tensor([15, 30]), torch.tensor([3, 6]))
        controller.step(obs, 3, 15, 3)
        self.assertEqual(calls['gait'], [2])
        self.assertEqual(calls['modulation'], [2, 1])
        torch.testing.assert_close(runner.primitive_selection_reservoir_states,
                                   torch.ones(2, 4))
        self.assertEqual(controller.clock.due(15)[0].tolist(), [0])
        env.step = lambda actions: (obs, None, None, torch.tensor([False, True]), {})
        controller.step(obs, 4, 15, 3)
        self.assertEqual(controller.clock.due(5)[0].tolist(), [1])
        torch.testing.assert_close(runner.primitive_selection_reservoir_states,
                                   torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.]]))

    def test_quality_requires_more_than_survival(self):
        reference = dict(failed=False, speed_error=0.2, tilt=0.1, slip=0.05,
                         collision=0., progress=1.0)
        held = dict(reference, speed_error=0.5)
        limits = dict(speed_error=0.1, tilt=0.1, slip=0.1,
                      collision=0.1, progress=0.1)
        self.assertFalse(compare_quality(reference, held, limits))
        self.assertTrue(compare_quality(reference, reference, limits))

    def test_speed_error_keeps_linear_velocity_units(self):
        body_velocity = torch.tensor([[1., 0., 99.]])
        command = torch.tensor([[0., 0., -99.]])
        torch.testing.assert_close(planar_speed_error(body_velocity, command),
                                   torch.tensor([1.]))

    def test_multiple_disturbances_use_conservative_outcome(self):
        self.assertEqual(conservative_outcomes([
            ('pass', 'pass', 'pass'), ('pass', 'fail', 'unobserved'),
        ]), ('pass', 'fail', 'unobserved'))
        self.assertEqual(conservative_outcomes([
            ('pass', 'unobserved'), ('pass', 'pass'),
        ]), ('pass', 'unobserved'))

    def test_baseline_failure_censors_candidate_outcome(self):
        quality = dict(failed=True, timeout=False, speed_error=0., tilt=0.,
                       slip=0., collision=0., progress=0.)
        reference = dict(prefix='same', quality=[quality])
        candidate = dict(prefix='same', quality=[quality])
        limits = dict(speed_error=0., tilt=0., slip=0., collision=0., progress=0.)
        self.assertEqual(candidate_result(reference, candidate, 1, limits), 'unobserved')

    def test_task_completion_censors_unobserved_remainder(self):
        quality = dict(failed=False, timeout=False, completed=True,
                       speed_error=0., tilt=0., slip=0., collision=0., progress=3.5)
        reference = dict(prefix='same', quality=[quality, quality])
        candidate = dict(prefix='same', quality=[quality, quality])
        limits = dict(speed_error=0., tilt=0., slip=0., collision=0., progress=0.)
        self.assertEqual(candidate_result(reference, candidate, 2, limits), 'unobserved')


if __name__ == '__main__':
    unittest.main()
