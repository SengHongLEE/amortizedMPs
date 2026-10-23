"""Render the actual Isaac Gym terrain mesh, without starting a training task.

Run from the repository root in the Isaac Gym environment:
    python legged_gym/legged_gym/scripts/preview_mp_terrain.py
"""
from isaacgym import gymapi
import ast
import argparse
import math
import importlib.util
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
source = ROOT / 'legged_gym/legged_gym'
spec = importlib.util.spec_from_file_location('blind_terrain', source / 'utils/terrains/blind_terrain.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--task', choices=['multi', 'proprio'], default='multi',
                    help='Choose the task config; proprio reads mp_proprio_terrain_config.py.')
preliminary, _ = pre_parser.parse_known_args()
parser = argparse.ArgumentParser(parents=[pre_parser])
config_file = 'mp_proprio_terrain_config.py' if preliminary.task == 'proprio' else 'mp_multi_primitive_config.py'
tree = ast.parse((source / 'envs/a1' / config_file).read_text())
config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotorPrimitivesCfg')
terrain_class = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'terrain')
namespace = {}
exec(compile(ast.Module(body=[terrain_class], type_ignores=[]), '<terrain config>', 'exec'), namespace)
cfg = namespace['terrain']()
parser.add_argument('--gallery', action='store_true', help='Show each type at maximum difficulty; otherwise render the training config.')
parser.add_argument('--terrain', choices=list(module.TERRAIN_GENERATORS) + list(cfg.terrain_variants),
                    help='Preview a single type or a configured variant.')
parser.add_argument('--difficulty', type=float, help='Fixed difficulty in [0, 1].')
parser.add_argument('--row', type=int, default=-1, help='Difficulty row to render; default last row.')
parser.add_argument('--max-tiles', type=int, default=0, help='Limit displayed columns; 0 displays all.')
parser.add_argument('--heightmap', action='store_true', help='Save numerical height maps without creating a simulator.')
parser.add_argument('--output', type=Path, help='Output PNG path.')
args = parser.parse_args()
cfg.mesh_type = 'heightfield' if args.heightmap else 'trimesh'
if args.gallery and args.terrain:
    parser.error('--gallery and --terrain are mutually exclusive')
if args.max_tiles < 0:
    parser.error('--max-tiles must be nonnegative')
if args.gallery:
    names = list(module.TERRAIN_GENERATORS)
    cfg.selected, cfg.selected_kwargs = None, None
    cfg.terrain_columns = names
    cfg.num_rows, cfg.num_cols = 1, len(names)
    cfg.terrain_proportions = dict.fromkeys(names, 1.)
    cfg.difficulty_range = [1., 1.]
    cfg.row_difficulties = None
if args.terrain:
    cfg.selected, cfg.selected_kwargs = args.terrain, None
    cfg.num_cols = 1
if args.difficulty is not None:
    if not 0 <= args.difficulty <= 1:
        parser.error('--difficulty must be in [0, 1]')
    cfg.difficulty_range = [args.difficulty, args.difficulty]
    cfg.row_difficulties = None
if not -cfg.num_rows <= args.row < cfg.num_rows:
    parser.error('--row is outside the configured rows')
terrain = module.BlindTerrain(cfg, 1)
row = args.row % cfg.num_rows
count = min(cfg.num_cols, args.max_tiles or cfg.num_cols)
origins = terrain.env_origins[row, :count]
names = terrain.terrain_names[row, :count]
output = args.output or ROOT / 'docs/terrain' / (
    ('proprio_' if args.task == 'proprio' else '') +
    ('terrain_heightmaps.png' if args.heightmap else
     'configurable_terrain_gallery.png' if args.gallery else 'training_terrain_preview.png'))
output.parent.mkdir(parents=True, exist_ok=True)
for col, name in enumerate(names):
    print(f'column={col} type={name} difficulty={terrain.difficulties[row, col]:.3f} '
          f'parameters={terrain.tile_parameters[row, col]}')
if args.heightmap:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(math.ceil(count / 3), min(count, 3),
                                figsize=(5 * min(count, 3), 4 * math.ceil(count / 3)), squeeze=False)
    for col, axis in enumerate(axes.flat):
        if col >= count:
            axis.axis('off')
            continue
        x = terrain.border + row * terrain.length_per_env_pixels
        y = terrain.border + col * terrain.width_per_env_pixels
        height = terrain.heightsamples[x:x+terrain.length_per_env_pixels,
                                      y:y+terrain.width_per_env_pixels] * terrain.vertical_scale
        im = axis.imshow(height.T, origin='lower', extent=(0, terrain.env_length, 0, terrain.env_width),
                         interpolation='nearest', cmap='terrain', aspect='equal')
        axis.set_title(f'{names[col]} / d={terrain.difficulties[row, col]:.2f}')
        axis.set_xlabel('x (m)')
        axis.set_ylabel('y (m)')
        figure.colorbar(im, ax=axis, label='height (m)')
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print('Saved', output)
    sys.exit(0)
gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0., 0., -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
if sim is None:
    raise RuntimeError('Could not create PhysX/graphics context')
try:
    mesh = gymapi.TriangleMeshParams()
    mesh.nb_vertices, mesh.nb_triangles = len(terrain.vertices), len(terrain.triangles)
    mesh.transform.p = gymapi.Vec3(-terrain.border_size, -terrain.border_size, 0.)
    gym.add_triangle_mesh(sim, terrain.vertices.ravel(), terrain.triangles.ravel(), mesh)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)
    props = gymapi.CameraProperties()
    props.width, props.height = 640, 480
    props.horizontal_fov = 60.
    cameras = []
    for origin in origins:
        camera = gym.create_camera_sensor(env, props)
        if camera < 0:
            raise RuntimeError('Camera creation failed')
        x, y, z = map(float, origin)
        gym.set_camera_location(camera, env, gymapi.Vec3(x + cfg.terrain_length * .65, y - cfg.terrain_width * .65, z + 5.),
                                gymapi.Vec3(x, y, z - .05))
        cameras.append(camera)
    gym.prepare_sim(sim)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    canvas = Image.new('RGB', (3 * props.width, math.ceil(len(cameras) / 3) * (props.height + 35)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (name, camera) in enumerate(zip(names, cameras)):
        rgba = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        frame = Image.fromarray(rgba.reshape(props.height, props.width, 4)).convert('RGB')
        left, top = index % 3 * props.width, index // 3 * (props.height + 35)
        canvas.paste(frame, (left, top + 35))
        draw.text((left + 12, top + 10), name + f' / {cfg.terrain_length:g} x {cfg.terrain_width:g} m / d={terrain.difficulties[row, index]:.2f}', fill='black')
    canvas.save(output)
    print('Saved', output)
finally:
    gym.destroy_sim(sim)
