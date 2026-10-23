"""A stair curriculum must change height without turning ramps into walls."""
from isaacgym import terrain_utils
from copy import deepcopy
import unittest

import numpy as np
import test_blind_terrain as fixtures


class StairMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TerrainTests.setUpClass()
        cls.Terrain = fixtures.TerrainTests.Terrain

    def terrain(self, height, vertical, kind='stairs', axis='x', roughness=None):
        return self.Terrain(fixtures.config(
            num_rows=1, num_cols=2, selected=kind, selected_kwargs=None,
            terrain_length=8., terrain_width=8., horizontal_scale=.1,
            vertical_scale=.005, slope_threshold=.75,
            terrain_params={kind: {'step_height_range': height, 'step_width': .6,
                                  'step_signs': [-1, 1], 'platform_size': 3.,
                                  'axis': axis, 'roughness': roughness,
                                  'mesh_vertical_faces': vertical}}), 2)

    def test_every_stair_height_has_vertical_risers_in_both_axes(self):
        for kind in ('stairs', 'stairs_linear'):
            for axis in ('x', 'y'):
                for height in (.01, .02, .06, .075, .08):
                    with self.subTest(kind=kind, axis=axis, height=height):
                        t = self.terrain(height, True, kind, axis)
                        vertices = t.vertices.reshape(t.tot_rows, t.tot_cols, 3)
                        for col in range(2):
                            x, y = t.border, t.border + col * t.width_per_env_pixels
                            if axis == 'x':
                                profile = t.heightsamples[x:x+80, y+40]
                                coordinates = vertices[x:x+80, y+40, 0]
                            else:
                                profile = t.heightsamples[x+40, y:y+80]
                                coordinates = vertices[x+40, y:y+80, 1]
                            edges = np.flatnonzero(np.diff(profile.astype(int)))
                            self.assertGreater(len(edges), 0)
                            np.testing.assert_allclose(np.diff(coordinates)[edges], 0., atol=1e-6)

    def test_legacy_geometry_and_existing_eight_cm_walls_are_preserved(self):
        legacy = self.terrain(.075, False)
        expected, triangles = terrain_utils.convert_heightfield_to_trimesh(
            legacy.heightsamples.astype(np.int32), .1, .005, slope_threshold=.75)
        np.testing.assert_array_equal(legacy.vertices, expected)
        np.testing.assert_array_equal(legacy.triangles, triangles)
        old, new = self.terrain(.08, False), self.terrain(.08, True)
        np.testing.assert_array_equal(old.vertices, new.vertices)
        np.testing.assert_array_equal(old.triangles, new.triangles)

    def test_other_terrain_vertices_and_heightfields_do_not_change(self):
        cfg = fixtures.config(num_rows=2, num_cols=4, selected=None, selected_kwargs=None,
                     terrain_length=8., terrain_width=8., horizontal_scale=.1,
                     vertical_scale=.005, slope_threshold=.75,
                     terrain_columns=['stairs', 'ramp', 'wave', 'rough'],
                     terrain_params={'stairs': {'step_height_range': .06,
                                               'mesh_vertical_faces': False}})
        old = self.Terrain(cfg, 8)
        changed = deepcopy(cfg)
        changed.terrain_params['stairs']['mesh_vertical_faces'] = True
        new = self.Terrain(changed, 8)
        np.testing.assert_array_equal(old.heightsamples, new.heightsamples)
        np.testing.assert_array_equal(old.env_origins, new.env_origins)
        np.testing.assert_array_equal(old.triangles, new.triangles)
        before = old.vertices.reshape(old.tot_rows, old.tot_cols, 3)
        after = new.vertices.reshape(new.tot_rows, new.tot_cols, 3)
        np.testing.assert_array_equal(before[:, old.border+80:], after[:, new.border+80:])
        self.assertTrue(np.any(before[:, old.border:old.border+80] !=
                               after[:, new.border:new.border+80]))

    def test_tread_roughness_is_not_converted_into_additional_walls(self):
        plain_old, plain_new = self.terrain(.06, False), self.terrain(.06, True)
        riser_coordinates = plain_old.vertices != plain_new.vertices
        noise = {'height_range': .005, 'height_step': .005, 'downsampled_scale': .2}
        old = self.terrain(.06, False, roughness=noise)
        new = self.terrain(.06, True, roughness=noise)
        np.testing.assert_array_equal(old.heightsamples, new.heightsamples)
        np.testing.assert_array_equal(old.vertices[~riser_coordinates], new.vertices[~riser_coordinates])
        self.assertTrue(np.any(old.vertices[riser_coordinates] != new.vertices[riser_coordinates]))
        with self.assertRaisesRegex(ValueError, 'mesh_vertical_faces must be a bool'):
            self.terrain(.06, 'yes')


if __name__ == '__main__':
    unittest.main()
