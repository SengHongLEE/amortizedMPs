"""Opt-in actor feature path keeps the environment observation contract."""
import unittest

import torch

from rsl_rl.modules.reservoir_actors import (
    LIFReservoirSNNReadoutActor_SINGLE_output_head,
)


class TerrainEdgeReadoutTests(unittest.TestCase):
    def test_optional_edge_head_starts_without_changing_old_logits(self):
        torch.manual_seed(7)
        baseline = LIFReservoirSNNReadoutActor_SINGLE_output_head(
            input_dim=162, output_dim=4, reservoir_dim=8,
            readout_hidden_dims=(8,), num_snn_steps=2,
            include_input_in_readout=True)
        augmented = LIFReservoirSNNReadoutActor_SINGLE_output_head(
            input_dim=162, output_dim=4, reservoir_dim=8,
            readout_hidden_dims=(8,), num_snn_steps=2,
            include_input_in_readout=True, height_edge_head=True)
        missing = augmented.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(set(missing.missing_keys),
                         {'height_edge_head.0.weight', 'height_edge_head.0.bias',
                          'height_edge_head.2.weight', 'height_edge_head.2.bias'})
        obs = torch.randn(3, 5, 162)
        state = torch.zeros(3, baseline.reservoir_state_dim)
        original_logits, _ = baseline(obs, state)
        augmented_logits, _ = augmented(obs, state)
        self.assertTrue(torch.allclose(original_logits, augmented_logits))

    def test_edge_features_react_to_hurdle_and_gap_at_any_forward_sample(self):
        actor = LIFReservoirSNNReadoutActor_SINGLE_output_head(
            input_dim=162, output_dim=4, reservoir_dim=8,
            readout_hidden_dims=(8,), num_snn_steps=2,
            height_edge_head=True)
        obs = torch.zeros(3, 5, 162)
        heights = obs[:, -1, -60:].reshape(3, 12, 5)
        heights[1, 4:6] = -.6
        heights[2, 9:11] = 3.
        features = actor.height_edge_features(obs)
        self.assertEqual(features.shape, (3, 11))
        self.assertTrue(torch.equal(features[0], torch.zeros(11)))
        self.assertLess(features[1, 3], 0.)
        self.assertGreater(features[2, 8], 0.)

    def test_terrain_encoder_drives_reservoir_and_can_receive_offline_gradient(self):
        actor = LIFReservoirSNNReadoutActor_SINGLE_output_head(
            input_dim=162, output_dim=4, reservoir_dim=8,
            readout_hidden_dims=(8,), num_snn_steps=2,
            terrain_encoder_dim=8)
        self.assertEqual(actor.input_dim, 162)
        self.assertEqual(actor.reservoir.input_dim, 170)
        obs = torch.zeros(2, 5, 162)
        obs[1, :, -60:].reshape(5, 12, 5)[:, 5:7] = -.6
        actor.train_encoder_through_reservoir = True
        _, state = actor(obs, torch.zeros(2, actor.reservoir_state_dim))
        state.sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in actor.terrain_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
