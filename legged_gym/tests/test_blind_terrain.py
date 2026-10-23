"""Terrain geometry and environment integration without simulator startup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from isaacgym import terrain_utils
import numpy as np
import torch
from test_interoceptive_rewards import load_class

ROOT = Path(__file__).resolve().parents[1] / 'legged_gym'
Config = load_class('mp_multi_primitive_config.py', 'MotorPrimitivesCfg')
Env = load_class('mp_multi_primitive.py', 'MotorPrimitives', {
    '_get_terrain_heights', '_get_env_origins', 'check_termination', '_spawn_positions',
})


def config(**changes):
    values = {k: getattr(Config.terrain, k) for k in dir(Config.terrain) if not k.startswith('_')}
    values.update(mesh_type='trimesh', num_rows=2, num_cols=5, selected='slope',
                  terrain_length=6., terrain_width=4., border_size=1.)
    values.update(changes)
    return SimpleNamespace(**values)


class TerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'utils/terrains/blind_terrain.py'
        cls.path = path
        if path.exists():
            spec = importlib.util.spec_from_file_location('blind_terrain', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.Terrain = module.BlindTerrain
            cls.generators = module.TERRAIN_GENERATORS

    def test_generator_exists(self):
        self.assertTrue(self.path.exists(), 'Missing configurable blind-terrain generator')

    def terrain(self, **changes):
        if not self.path.exists():
            self.skipTest('Generator not implemented yet')
        return self.Terrain(config(**changes), 12)

    def test_rectangular_map_mesh_origins_and_safe_platform(self):
        terrain = self.terrain()
        self.assertEqual(terrain.env_origins.shape, (2, 5, 3))
        rows, cols = terrain.heightsamples.shape
        self.assertEqual(terrain.vertices.shape, (rows * cols, 3))
        self.assertEqual(terrain.triangles.shape, (2 * (rows - 1) * (cols - 1), 3))
        self.assertTrue(np.isfinite(terrain.vertices).all())
        self.assertLess(terrain.triangles.max(), len(terrain.vertices))
        triangles = terrain.vertices[terrain.triangles]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.assertTrue((normals[:, 2] >= -1.e-8).all())
        for origin in terrain.env_origins.reshape(-1, 3):
            xy = np.rint((origin[:2] + terrain.border_size) / terrain.horizontal_scale).astype(int)
            patch = terrain.heightsamples[xy[0]-3:xy[0]+4, xy[1]-3:xy[1]+4]
            np.testing.assert_allclose(patch * terrain.vertical_scale, origin[2])

    def test_all_types_and_reproducibility(self):
        for name in ('flat', 'slope', 'rough_slope', 'rough', 'rubble'):
            a = self.terrain(selected=name)
            b = self.terrain(selected=name)
            np.testing.assert_array_equal(a.heightsamples, b.heightsamples)
            self.assertEqual(set(a.terrain_names.flat), {name})
            if name != 'flat':
                self.assertGreater(np.ptp(a.heightsamples), 0)
        a, b = self.terrain(selected='rubble', seed=3), self.terrain(selected='rubble', seed=4)
        self.assertFalse(np.array_equal(a.heightsamples, b.heightsamples))

    def test_invalid_config(self):
        for values in ({'selected':'unknown_type'}, {'terrain_proportions':{'rubble':-1}},
                       {'horizontal_scale':0}, {'platform_size':10}, {'num_rows':0}):
            with self.assertRaises(ValueError):
                self.terrain(**values)

    def test_height_sampling_and_termination_use_local_ground(self):
        terrain = self.terrain(selected='slope')
        e = Env()
        e.cfg = SimpleNamespace(terrain=terrain.cfg)
        e.device, e.num_envs = 'cpu', 2
        e.terrain = terrain
        e.height_samples = torch.from_numpy(terrain.heightsamples)
        e._get_env_origins()
        e.root_states = torch.zeros(2, 13)
        e.root_states[:, :3] = e.env_origins
        e.root_states[:, 2] += torch.tensor([.32, .1])
        torch.testing.assert_close(e._get_terrain_heights(e.root_states[:, :2]), e.env_origins[:, 2])
        e.contact_forces = torch.zeros(2, 1, 3)
        e.termination_contact_indices = [0]
        e.episode_length_buf = torch.zeros(2)
        e.max_episode_length = 100
        e.check_termination()
        self.assertEqual(e.reset_buf.tolist(), [False, True])
        e.root_states[:, 2] += 1
        e.root_states[0, 0] += terrain.env_length
        e.check_termination()
        self.assertTrue(e.time_out_buf[0])

    def test_plane_termination_unchanged(self):
        if not self.path.exists():
            self.skipTest('Generator not implemented yet')
        e = Env()
        e.cfg = SimpleNamespace(terrain=config(mesh_type='plane'))
        e.root_states = torch.zeros(2, 13)
        e.root_states[:, 2] = torch.tensor([.32, .1])
        e.contact_forces = torch.zeros(2, 1, 3)
        e.termination_contact_indices = [0]
        e.episode_length_buf = torch.zeros(2)
        e.max_episode_length = 100
        e.check_termination()
        self.assertEqual(e.reset_buf.tolist(), [False, True])

    def test_spawn_height_and_initial_difficulty_on_cpu_and_cuda(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            terrain = self.terrain(selected='rough_slope', max_init_terrain_level=0)
            e = Env()
            e.cfg = SimpleNamespace(terrain=terrain.cfg)
            e.device, e.num_envs, e.terrain = device, 12, terrain
            e.height_samples = torch.tensor(terrain.heightsamples, device=device)
            e.base_init_state = torch.zeros(13, device=device)
            e.base_init_state[2] = .32
            e._get_env_origins()
            self.assertTrue((e.terrain_levels == 0).all())
            ids = torch.tensor([0, 5, 11], device=device)
            positions = e._spawn_positions(ids)
            self.assertTrue(((positions[:, :2] - e.env_origins[ids, :2]).abs() <= terrain.cfg.spawn_xy_jitter).all())
            torch.testing.assert_close(positions[:, 2], e.env_origins[ids, 2] + .32)

    def test_ground_estimate_is_conservative_near_vertical_walls(self):
        self.terrain()
        e = Env()
        e.terrain = SimpleNamespace(border_size=0., horizontal_scale=1., vertical_scale=1., tot_rows=2, tot_cols=2)
        e.height_samples = torch.tensor([[0, 2], [4, 8]], dtype=torch.int16)
        actual = e._get_terrain_heights(torch.tensor([[.75, .25], [.25, .75], [1., 1.], [-1., -1.]]))
        torch.testing.assert_close(actual, torch.zeros(4))

    def test_generation_does_not_change_global_rng(self):
        np.random.seed(123)
        before = np.random.get_state()
        self.terrain()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


    def test_trimesh_matches_official_converter_with_vertical_walls(self):
        terrain = self.terrain(selected='rubble', difficulty_range=[1., 1.])
        expected_vertices, expected_triangles = terrain_utils.convert_heightfield_to_trimesh(
            terrain.heightsamples, terrain.horizontal_scale, terrain.vertical_scale,
            slope_threshold=terrain.cfg.slope_threshold)
        np.testing.assert_array_equal(terrain.vertices, expected_vertices)
        np.testing.assert_array_equal(terrain.triangles, expected_triangles)
        tri = terrain.vertices[terrain.triangles]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vertical = (np.abs(normals[:, 2]) < 1.e-8) & (np.linalg.norm(normals[:, :2], axis=1) > 1.e-7)
        self.assertGreater(vertical.sum(), 20, 'Discrete obstacles need real vertical side faces')

    def test_rubble_keeps_discrete_levels_without_blending(self):
        terrain = self.terrain(selected='rubble', difficulty_range=[1., 1.])
        amplitude = int(terrain.cfg.terrain_params['rubble']['height_range'][1] / terrain.vertical_scale)
        allowed = {0, -amplitude, -amplitude // 2, amplitude // 2, amplitude}
        self.assertTrue(set(np.unique(terrain.heightsamples)) <= allowed)
        self.assertGreater(len(np.unique(terrain.heightsamples)), 2)

    def test_stairs_have_constant_treads(self):
        terrain = self.terrain(selected='stairs', difficulty_range=[1., 1.])
        self.assertGreater(len(np.unique(terrain.heightsamples)), 3)

    def test_selected_tile_is_unmodified_official_output(self):
        for function, kwargs in (
            ('random_uniform_terrain', dict(min_height=-.05, max_height=.05, step=.005, downsampled_scale=.2)),
            ('pyramid_sloped_terrain', dict(slope=.2, platform_size=3.)),
            ('discrete_obstacles_terrain', dict(max_height=.15, min_size=1., max_size=2., num_rects=20, platform_size=3.)),
        ):
            terrain = self.terrain(num_rows=1, num_cols=1, selected='rough',
                                   selected_kwargs=dict(type=function, **kwargs))
            reference = terrain_utils.SubTerrain(width=60, length=40, horizontal_scale=.1, vertical_scale=.005)
            state = np.random.get_state()
            try:
                np.random.seed(terrain.cfg.seed)
                getattr(terrain_utils, function)(reference, **kwargs)
            finally:
                np.random.set_state(state)
            border = terrain.border
            np.testing.assert_array_equal(terrain.heightsamples[border:border+60, border:border+40], reference.height_field_raw)
            self.assertEqual(terrain.heightsamples.shape, (80, 60))
            self.assertAlmostEqual(terrain.env_origins[0, 0, 2], reference.height_field_raw[20:40, 10:30].max() * .005)
