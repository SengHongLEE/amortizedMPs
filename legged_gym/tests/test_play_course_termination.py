"""Playback safety checks for a forward course with different terrain rows."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


MP_FULL_PATH = Path(__file__).resolve().parents[1] / 'legged_gym/envs/a1/mp_full.py'


def load_termination_class():
    tree = ast.parse(MP_FULL_PATH.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'MotorPrimitives')
    cls.bases = []
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == 'check_termination']
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(MP_FULL_PATH), 'exec'), namespace)
    return namespace['MotorPrimitives']


class PlayCourseTerminationTests(unittest.TestCase):
    def test_current_row_controls_clearance_and_lane_without_reset_at_seam(self):
        env = load_termination_class()()
        env.cfg = SimpleNamespace(terrain=SimpleNamespace(
            mesh_type='trimesh', num_rows=3, num_cols=1,
            terrain_rows=['flat', 'hurdle', 'gap'], boundary_margin=.6,
            border_size=4., min_base_clearance=.15,
            hurdle_min_base_clearance=.10, hazard_lane_half_width=1.5))
        env.terrain = SimpleNamespace(
            env_length=8., env_width=8.,
            terrain_names=np.array([['flat'], ['hurdle'], ['gap']]),
            _kind=lambda name: name)
        env.root_states = torch.zeros(4, 13)
        env.root_states[:, :3] = torch.tensor([
            [4., 4., .27], [12., 4., .27],
            [20., 5.6, .32], [8., 4., .32]])
        env.env_origins = torch.tensor([[4., 4., 0.]] * 4)
        env.terrain_types = torch.zeros(4, dtype=torch.long)
        env._gap_columns = torch.tensor([False])
        env._hurdle_columns = torch.tensor([False])
        env.contact_forces = torch.zeros(4, 1, 3)
        env.termination_contact_indices = [0]
        env.episode_length_buf = torch.ones(4, dtype=torch.long)
        env.max_episode_length = 500
        env._get_terrain_heights = lambda xy: torch.full((len(xy),), .125)

        env.check_termination()

        self.assertEqual(env.terrain_failure_buf.tolist(), [True, False, True, False])
        self.assertEqual(env.terrain_lane_exit_buf.tolist(), [False, False, True, False])
        self.assertEqual(env.time_out_buf.tolist(), [False] * 4)


if __name__ == '__main__':
    unittest.main()
