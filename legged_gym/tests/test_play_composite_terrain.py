"""Configuration checks for play.py without starting Isaac Gym."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


PLAY_PATH = Path(__file__).resolve().parents[1] / 'legged_gym/scripts/play.py'


def load_configure_composite_terrain():
    source = ast.parse(PLAY_PATH.read_text())
    function = next((node for node in source.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == 'configure_composite_terrain'), None)
    assert function is not None, 'play.py must define configure_composite_terrain'
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PLAY_PATH), 'exec'), namespace)
    return namespace['configure_composite_terrain']


def make_cfg():
    return SimpleNamespace(
        env=SimpleNamespace(num_envs=1, episode_length_s=10.),
        terrain=SimpleNamespace(
            mesh_type='trimesh', selected='flat', selected_kwargs={'type': 'sloped_terrain'},
            terrain_columns=['flat'], terrain_rows=None, terrain_length=8.,
            terrain_params={'hurdle': {'spacing': 4.}},
            terrain_variants={'gap_single': {'type': 'gap', 'params': {'pattern': 'single'}}},
            num_rows=10, num_cols=1, difficulty_range=[0., 1.], curriculum=True,
            curriculum_min_level=0, curriculum_max_level=9, fixed_terrain_level=5,
            row_difficulties=[.1] * 10, difficulty_mode='random',
        ),
    )


class CompositePlayTerrainTests(unittest.TestCase):
    def test_play_lists_parameters_for_every_course_row(self):
        source = ast.parse(PLAY_PATH.read_text())
        play = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'play')
        call = next(node for node in ast.walk(play)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'configure_composite_terrain')
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        rows = ast.literal_eval(keywords['terrain_rows'])
        params = ast.literal_eval(keywords['terrain_params'])

        self.assertTrue(set(rows) <= set(params))
        self.assertTrue(all(params[row] for row in rows))
        self.assertEqual(params['gap_continuous']['pattern'], 'centered')
        self.assertEqual(params['gap_single']['single_position'], 'center')
        self.assertEqual(params['hurdle']['pattern'], 'single')

    def test_one_column_has_ordered_terrains_along_travel_axis(self):
        configure = load_configure_composite_terrain()
        cfg = make_cfg()
        configure(cfg, ['flat', 'gap_single', 'hurdle', 'stairs'], difficulty=.8)

        self.assertEqual(cfg.terrain.terrain_rows,
                         ['flat', 'gap_single', 'hurdle', 'stairs'])
        self.assertIsNone(cfg.terrain.terrain_columns)
        self.assertEqual((cfg.terrain.num_rows, cfg.terrain.num_cols), (4, 1))
        self.assertEqual(cfg.env.num_envs, 1)
        self.assertGreaterEqual(cfg.env.episode_length_s, 4 * cfg.terrain.terrain_length / .5)
        self.assertEqual(cfg.terrain.difficulty_range, [.8, .8])
        self.assertIsNone(cfg.terrain.row_difficulties)
        self.assertEqual(cfg.terrain.difficulty_mode, 'linear')
        self.assertIsNone(cfg.terrain.selected)
        self.assertIsNone(cfg.terrain.selected_kwargs)
        self.assertFalse(cfg.terrain.curriculum)
        self.assertEqual(cfg.terrain.fixed_terrain_level, 0)
        self.assertIsNone(cfg.terrain.curriculum_max_level)
        self.assertEqual(cfg.terrain.terrain_variants['gap_single']['params'].get('single_position'),
                         'center')
        self.assertEqual(cfg.terrain.terrain_params['hurdle']['pattern'], 'single')

    def test_per_type_parameters_extend_archived_settings(self):
        configure = load_configure_composite_terrain()
        cfg = make_cfg()
        configure(cfg, ['hurdle', 'gap_single'], terrain_params={
            'hurdle': {'height_range': [.1, .2]},
            'gap_single': {'width_range': [.2, .4]},
        })

        self.assertEqual(cfg.terrain.terrain_params['hurdle'],
                         {'spacing': 4., 'pattern': 'single', 'height_range': [.1, .2]})
        self.assertEqual(cfg.terrain.terrain_variants['gap_single']['params'],
                         {'pattern': 'single', 'single_position': 'center',
                          'width_range': [.2, .4]})
        self.assertEqual(cfg.terrain.difficulty_range, [1., 1.])

    def test_rejects_empty_layout_or_invalid_difficulty(self):
        configure = load_configure_composite_terrain()
        for columns, difficulty in (([], 1.), (['flat'], -0.1), (['flat'], float('nan'))):
            with self.subTest(columns=columns, difficulty=difficulty), self.assertRaises(ValueError):
                configure(make_cfg(), columns, difficulty=difficulty)

    def test_environment_count_allows_multiple_robots_on_same_course(self):
        configure = load_configure_composite_terrain()
        cfg = make_cfg()
        configure(cfg, ['flat', 'stairs'], num_envs=4)
        self.assertEqual(cfg.env.num_envs, 4)
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                configure(make_cfg(), ['flat', 'stairs'], num_envs=count)


if __name__ == '__main__':
    unittest.main()
