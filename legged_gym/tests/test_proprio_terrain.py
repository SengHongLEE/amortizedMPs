"""Blind-locomotion curriculum geometry contracts, not policy success tests."""
from isaacgym import terrain_utils  # Isaac Gym must precede torch.
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from test_interoceptive_rewards import load_class

ROOT = Path(__file__).resolve().parents[2]
Config = load_class('mp_proprio_terrain_config.py', 'MotorPrimitivesCfg')
spec = importlib.util.spec_from_file_location(
    'proprio_terrain_generator', ROOT / 'legged_gym/legged_gym/utils/terrains/blind_terrain.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def config():
    return SimpleNamespace(**{k: deepcopy(getattr(Config.terrain, k))
                              for k in dir(Config.terrain) if not k.startswith('_')})


def tiles(terrain):
    for row in range(terrain.cfg.num_rows):
        for col in range(terrain.cfg.num_cols):
            x = terrain.border + row * terrain.length_per_env_pixels
            y = terrain.border + col * terrain.width_per_env_pixels
            height = terrain.heightsamples[x:x+terrain.length_per_env_pixels,
                                          y:y+terrain.width_per_env_pixels].astype(float)
            yield row, col, height * terrain.vertical_scale


class ProprioTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = config()
        cfg.mesh_type = 'heightfield'
        cls.terrain = generator.BlindTerrain(cfg, 32)

    def test_training_does_not_require_precise_foothold_or_jump_selection(self):
        self.assertFalse(Config.terrain.measure_heights)
        self.assertFalse(set(self.terrain.terrain_names.flat) & {
            'gap', 'pit', 'beam', 'stepping_stones', 'hurdle'})

    def test_both_directions_are_represented(self):
        for name, key in [('ramp', 'slope_signs'), ('slope', 'slope_signs'),
                          ('rough_slope', 'slope_signs'), ('stairs', 'step_signs')]:
            signs = {p[key][0] for n, p in zip(self.terrain.terrain_names[0],
                                             self.terrain.tile_parameters[0]) if n == name}
            self.assertEqual(signs, {-1, 1}, name)

    def test_adjacent_heights_leave_swing_clearance_margin(self):
        for row, col, height in tiles(self.terrain):
            jump = max(np.abs(np.diff(height, axis=a)).max() for a in (0, 1))
            limit = .04 if row <= Config.terrain.curriculum_initial_level else .08
            self.assertLessEqual(jump, limit + 1e-8, (row, col, self.terrain.terrain_names[row, col], jump))

    def test_upper_levels_add_material_challenge(self):
        last = self.terrain.cfg.num_rows - 1
        for name, key, minimum in [('stairs', 'step_height_range', .08),
                                   ('ramp', 'slope_range', .3), ('rubble', 'height_range', .04)]:
            col = list(self.terrain.terrain_names[last]).index(name)
            self.assertGreaterEqual(self.terrain.tile_parameters[last, col][key], minimum)

    def test_waves_remain_gentle_and_are_not_converted_into_walls(self):
        for row, col, height in tiles(self.terrain):
            if self.terrain.terrain_names[row, col] != 'wave':
                continue
            gradient = np.gradient(height, self.terrain.horizontal_scale)
            grade = np.hypot(*gradient).max()
            self.assertLessEqual(grade, .15, grade)
            self.assertLess(grade, self.terrain.cfg.slope_threshold)

    def test_ik_lengths_match_loaded_urdf(self):
        urdf = ET.parse(Config.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', str(ROOT / 'legged_gym')))
        for leg in ('FL', 'FR', 'RL', 'RR'):
            for joint, parameter in [('calf_joint', 'thigh_link_length_a1'),
                                     ('foot_fixed', 'calf_link_length_a1')]:
                origin = urdf.find(f".//joint[@name='{leg}_{joint}']/origin")
                length = abs(float(origin.attrib['xyz'].split()[2]))
                self.assertAlmostEqual(getattr(Config.asset, parameter), length)

    def test_friction_and_spawn_height_support_initial_training(self):
        self.assertGreaterEqual(Config.domain_rand.friction_range[0], .5)
        for row, col, height in tiles(self.terrain):
            cx, cy = np.array(height.shape) // 2
            radius = int(np.ceil(Config.terrain.spawn_xy_jitter / self.terrain.horizontal_scale))
            minimum = height[cx-radius:cx+radius+1, cy-radius:cy+radius+1].min()
            self.assertLessEqual(self.terrain.env_origins[row, col, 2] - minimum, .08)

    def test_sampled_foot_targets_are_inside_urdf_joint_limits(self):
        spec = importlib.util.spec_from_file_location(
            'audit_proprio', ROOT / 'legged_gym/legged_gym/scripts/audit_proprio_terrain.py')
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        result = audit.kinematic_screen(Config)
        self.assertEqual(result['infeasible_samples'], 0, result)
        self.assertEqual(len(result['reference_speed_limit_ratio_by_gait']), 6)
        self.assertEqual(result['equilibrium_limits_m'], list(Config.control.equilibrium.offset_limits))
        self.assertGreaterEqual(result['minimum_joint_margin_rad'],
                                Config.control.equilibrium.joint_limit_margin - 1e-6)
        # Raw requested targets can exceed the workspace. More lift should
        # increase this count, while bounded IK must still produce valid joints.
        previous = audit.kinematic_screen(Config, clearance=.22)
        self.assertGreater(previous['requested_infeasible_samples'], result['requested_infeasible_samples'])
        self.assertEqual(previous['infeasible_samples'], 0)
