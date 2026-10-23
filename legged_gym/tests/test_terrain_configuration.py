"""Configuration contracts and geometry regression tests using the real SDK."""
from copy import deepcopy
from collections import Counter
import unittest

import test_blind_terrain as baseline
import numpy as np


class TerrainConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline.TerrainTests.setUpClass()
        cls.Terrain = baseline.TerrainTests.Terrain

    def make(self, **changes):
        changes.setdefault('mesh_type', 'heightfield')
        return self.Terrain(baseline.config(**changes), 12)

    def tile(self, terrain, row=0, col=0):
        x = terrain.border + row * terrain.length_per_env_pixels
        y = terrain.border + col * terrain.width_per_env_pixels
        return terrain.heightsamples[x:x+terrain.length_per_env_pixels,
                                    y:y+terrain.width_per_env_pixels]

    def test_difficulty_reaches_both_endpoints(self):
        t = self.make(selected='slope', num_rows=3, num_cols=1,
                      terrain_params={'slope': {'slope_range': [.1, .5], 'slope_signs': [1]}})
        np.testing.assert_allclose(t.difficulties[:, 0], [0., .5, 1.])
        self.assertAlmostEqual(t.tile_parameters[2, 0]['slope_range'], .5)

    def test_weighted_columns_cover_small_positive_weights(self):
        t = self.make(selected=None, num_cols=5,
                      terrain_proportions={'flat': 100, 'rough': 1, 'stairs': 1})
        self.assertEqual(Counter(t.terrain_names[0]), {'flat': 3, 'rough': 1, 'stairs': 1})
        t = self.make(selected=None, num_cols=10, terrain_proportions={'flat': 9, 'rough': 1})
        self.assertEqual(Counter(t.terrain_names[0]), {'flat': 9, 'rough': 1})
        with self.assertRaisesRegex(ValueError, 'num_cols'):
            self.make(selected=None, num_cols=2,
                      terrain_proportions={'flat': 100, 'rough': 1, 'stairs': 1})

    def test_explicit_columns_variants_and_axis(self):
        variants = {
            'up': {'type': 'ramp', 'params': {'slope_range': .2, 'slope_signs': [1]}},
            'side': {'type': 'ramp', 'params': {'slope_range': .2, 'slope_signs': [1], 'axis': 'y'}},
        }
        t = self.make(selected=None, num_cols=2, terrain_columns=['up', 'side'],
                      terrain_variants=variants)
        self.assertEqual(t.terrain_names[0].tolist(), ['up', 'side'])
        up, side = self.tile(t), self.tile(t, col=1)
        self.assertGreater(np.ptp(up[:, 0]), 0)
        self.assertEqual(np.ptp(up[0, :]), 0)
        self.assertGreater(np.ptp(side[0, :]), 0)
        self.assertEqual(np.ptp(side[:, 0]), 0)

    def test_partial_parameters_and_unknown_keys(self):
        self.make(selected='rough', terrain_params={'rough': {'height_range': .03}})
        with self.assertRaisesRegex(ValueError, 'heigth_range'):
            self.make(selected='rough', terrain_params={'rough': {'heigth_range': .03}})

    def test_ramp_platform_connects_continuously_to_both_slopes(self):
        for axis in ('x', 'y'):
            for sign in (-1, 1):
                t = self.make(selected='ramp', num_cols=1, platform_size=1.2,
                              terrain_params={'ramp': {'slope_range': .3, 'slope_signs': [sign],
                                                       'axis': axis, 'center_platform': True}})
                height = self.tile(t) * t.vertical_scale
                profile = height[:, height.shape[1] // 2] if axis == 'x' else height[height.shape[0] // 2]
                center = len(profile) // 2
                np.testing.assert_allclose(profile[center-5:center+6], 0.)
                self.assertGreater(sign * (profile[-1] - profile[0]), .5)
                self.assertLessEqual(np.abs(np.diff(profile)).max(),
                                     .3 * t.horizontal_scale + t.vertical_scale + 1e-8)

    def test_explicit_and_random_difficulties_reproduce(self):
        t = self.make(row_difficulties=[.8, .2])
        np.testing.assert_allclose(t.difficulties, [[.8]*5, [.2]*5])
        a = self.make(difficulty_mode='random', difficulty_range=[.2, .8])
        b = self.make(difficulty_mode='random', difficulty_range=[.2, .8])
        np.testing.assert_array_equal(a.difficulties, b.difficulties)
        self.assertTrue(((a.difficulties >= .2) & (a.difficulties <= .8)).all())
        self.assertGreater(len(np.unique(a.difficulties)), 2)

    def test_new_types_have_geometry_on_rectangular_tiles(self):
        for name in ('ramp', 'stairs_linear', 'stepping_stones', 'wave',
                     'gap', 'pit', 'hurdle', 'beam'):
            with self.subTest(name=name):
                t = self.make(selected=name, difficulty_range=[1., 1.])
                self.assertEqual(t.heightsamples.dtype, np.int16)
                self.assertGreater(np.ptp(self.tile(t)), 0)
                self.assertTrue(np.isfinite(t.env_origins).all())

    def test_gap_and_hurdle_dimensions(self):
        for name, key, height in [('gap', 'depth_range', -.3), ('hurdle', 'height_range', .3)]:
            t = self.make(selected=name, num_cols=1, platform_size=1.,
                          terrain_params={name: {'width_range': .4, key: .3, 'spacing': 1.}})
            profile = self.tile(t)[:, 20] * t.vertical_scale
            np.testing.assert_allclose(np.unique(profile), sorted([0., height]))
            self.assertTrue(np.all(profile[25:35] == 0))
            self.assertEqual(np.count_nonzero(profile[35:45]), 4)

    def test_decreasing_range_and_optional_spawn_platform(self):
        t = self.make(selected='beam', num_cols=1, num_rows=2, platform_size=1.,
                      terrain_params={'beam': {'width_range': [1., .3], 'depth_range': .5}})
        self.assertGreater(np.count_nonzero(self.tile(t, 0)[0] == 0),
                           np.count_nonzero(self.tile(t, 1)[0] == 0))
        t = self.make(selected='rough', spawn_platform=True, platform_size=1.)
        patch = self.tile(t)[26:34, 16:24]
        self.assertEqual(np.ptp(patch), 0)
        self.assertAlmostEqual(t.env_origins[0, 0, 2], patch[0, 0] * t.vertical_scale)

    def test_sdk_override_does_not_require_wrapper_parameters(self):
        t = self.make(selected='rough', terrain_params={},
                      selected_kwargs={'type': 'sloped_terrain', 'slope': .2})
        self.assertGreater(np.ptp(t.heightsamples), 0)
        with self.assertRaisesRegex(ValueError, 'step'):
            self.make(selected='rough', selected_kwargs={
                'type': 'random_uniform_terrain', 'min_height': -.1, 'max_height': .1, 'step': 0})

    def test_invalid_values_fail_before_generation(self):
        cases = [
            {'selected': 'wave', 'terrain_params': {'wave': {'num_waves': 0}}},
            {'selected': 'gap', 'terrain_params': {'gap': {'width_range': .01}}},
            {'selected': 'stairs', 'terrain_params': {'stairs': {'step_width': 0}}},
            {'selected': 'ramp', 'terrain_params': {'ramp': {'slope_range': 1000.}}},
            {'difficulty_mode': 'typo'},
            {'row_difficulties': [.3]},
            {'selected': None, 'terrain_columns': ['flat']},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.make(**values)

    def test_inputs_are_not_mutated(self):
        c = baseline.config(selected='rough', terrain_params={'rough': {'height_range': [.02, .03]}})
        before = deepcopy(vars(c))
        self.Terrain(c, 12)
        self.assertEqual(vars(c), before)

    def test_all_axes_and_meshes(self):
        for mesh in ('heightfield', 'trimesh'):
            for name in baseline.TerrainTests.generators:
                with self.subTest(mesh=mesh, name=name):
                    t = self.make(mesh_type=mesh, selected=name, num_rows=1, num_cols=1,
                                  difficulty_range=[1., 1.], terrain_params={name: {'axis': 'y'}})
                    self.assertEqual(self.tile(t).shape, (60, 40))
                    if mesh == 'trimesh':
                        self.assertTrue(np.isfinite(t.vertices).all())

    def test_roughness_overlay_and_per_type_platform(self):
        clean = self.make(selected='slope', num_cols=1)
        noisy = self.make(selected='slope', num_cols=1, terrain_params={
            'slope': {'roughness': {'height_range': .03}}})
        self.assertFalse(np.array_equal(clean.heightsamples, noisy.heightsamples))
        difference = self.tile(noisy).astype(float) - self.tile(clean)
        self.assertLessEqual(np.abs(difference).max() * noisy.vertical_scale, .035)
        t = self.make(selected='rough', spawn_platform=True, terrain_params={
            'rough': {'platform_size': 1.}})
        self.assertEqual(np.ptp(self.tile(t)[25:35, 15:25]), 0)

    def test_direct_sdk_parameters_are_checked(self):
        for options in [
            {'type': 'stairs_terrain', 'step_width': 0, 'step_height': .1},
            {'type': 'sloped_terrain', 'slope': 1000},
            {'type': 'sloped_terrain', 'slope': 'bad'},
            {'type': 'random_uniform_terrain', 'min_height': 0, 'max_height': .1, 'downsampled_scale': 0},
            {'type': 'wave_terrain', 'num_waves': 1.5},
            {'type': 'stepping_stones_terrain', 'stone_size': 0, 'stone_distance': .1, 'max_height': .1},
            {'type': 'discrete_obstacles_terrain', 'max_height': .1, 'min_size': 1, 'max_size': 1, 'num_rects': 2},
            {'type': 'sloped_terrain', 'unknown': 1},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make(selected='flat', selected_kwargs=options)

    def test_actual_training_config_builds_enabled_types(self):
        values = {k: getattr(baseline.Config.terrain, k) for k in dir(baseline.Config.terrain)
                  if not k.startswith('_')}
        values['mesh_type'] = 'heightfield'
        t = self.Terrain(baseline.SimpleNamespace(**values), 4096)
        if values['selected'] is not None:
            expected = {values['selected']}
        elif values.get('terrain_columns') is not None:
            expected = set(values['terrain_columns'])
        else:
            expected = {name for name, weight in values['terrain_proportions'].items() if weight > 0}
        self.assertEqual(set(t.terrain_names.flat), expected)
        self.assertTrue(np.isfinite(t.env_origins).all())

    def test_stair_width_is_not_shortened_by_floating_point_truncation(self):
        t = self.make(selected='stairs_linear', num_rows=1, num_cols=1, terrain_length=6.1,
                      terrain_params={'stairs_linear': {
                          'step_width': .3, 'step_height_range': .05, 'step_signs': [1]}})
        profile = self.tile(t)[:, 0]
        np.testing.assert_array_equal(profile[:6], [10, 10, 10, 20, 20, 20])
        self.assertEqual(profile[-1], profile[-2], 'Partial final tread must not reset to zero')
