"""A reset or side exit must never count as successful traversal."""
import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/terrain_evaluation.py'


class EvaluationTests(unittest.TestCase):
    def test_exact_stair_rows(self):
        spec = importlib.util.spec_from_file_location('evaluation_metrics', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        heights = [.06, .07, .075, .08]
        height_range, rows = module.stair_evaluation_rows(heights, .005)
        self.assertEqual(height_range, [0., .08])
        for actual, expected in zip(rows, [.75, .875, .9375, 1.]):
            self.assertAlmostEqual(actual, expected)
        for heights in ([], [0.], [float('nan')], [float('inf')],
                        [.08, .06], [.06, .06], [.063]):
            with self.subTest(heights=heights), self.assertRaises(ValueError):
                module.stair_evaluation_rows(heights, .005)
        limits, rows = module.stair_evaluation_rows([0., .02, .04, .06, .08], .005, allow_flat=True)
        self.assertEqual(limits, [0., .08])
        self.assertEqual(rows, [0., .25, .5, .75, 1.])
        for heights in ([0.], [0., 0.], [-.02, .08], [0., float('nan')]):
            with self.subTest(heights=heights), self.assertRaises(ValueError):
                module.stair_evaluation_rows(heights, .005, allow_flat=True)

    def test_generated_stair_heights(self):
        from test_blind_terrain import TerrainTests, config
        import numpy as np
        TerrainTests.setUpClass()
        spec = importlib.util.spec_from_file_location('evaluation_metrics', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        heights = [.06, .07, .075, .08]
        height_range, rows = module.stair_evaluation_rows(heights, .005)
        cfg = config(num_rows=len(heights), num_cols=2, selected='stairs',
                     terrain_length=8., terrain_width=8., vertical_scale=.005,
                     row_difficulties=rows,
                     selected_kwargs=None,
                     terrain_params={'stairs': {'step_height_range': height_range, 'step_width': .6,
                                                'step_signs': [-1, 1], 'platform_size': 3.}})
        terrain = TerrainTests.Terrain(cfg, 2 * len(heights))
        for row, height in enumerate(heights):
            for col in range(2):
                start = terrain.border + row * terrain.length_per_env_pixels
                center = terrain.border + col * terrain.width_per_env_pixels + terrain.width_per_env_pixels // 2
                profile = terrain.heightsamples[start:start + terrain.length_per_env_pixels, center]
                jumps = np.abs(np.diff(profile.astype(int)))
                np.testing.assert_allclose(jumps[jumps > 0] * cfg.vertical_scale, height)
                self.assertGreater(np.count_nonzero(jumps), 0)

    def test_first_episode_and_corridor(self):
        self.assertTrue(PATH.exists(), 'Missing traversal accumulator')
        spec = importlib.util.spec_from_file_location('evaluation_metrics', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.TraversalMetrics(4, 'cpu', distance=3.4, corridor=1.5)
        metrics.update(torch.tensor([[3.5, 0.], [3.5, 2.], [1., 0.], [1., 0.]]),
                       torch.tensor([False, False, True, False]),
                       torch.tensor([True, True, False, False]), torch.ones(4), .02)
        metrics.update(torch.tensor([[0., 0.], [4., 0.], [4., 0.], [3.5, .1]]),
                       torch.tensor([True, False, False, False]),
                       torch.zeros(4, dtype=torch.bool), torch.ones(4), .02)
        self.assertEqual(metrics.success.tolist(), [True, False, False, True])
        self.assertEqual(metrics.failed.tolist(), [False, False, True, False])
        torch.testing.assert_close(metrics.duration, torch.tensor([.02, .02, .02, .04]))
        self.assertEqual(metrics.max_forward.tolist(), [3.5, 3.5, 1., 3.5])


if __name__ == '__main__':
    unittest.main()
