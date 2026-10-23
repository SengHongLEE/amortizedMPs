"""Explicit Isaac Gym mesh collision smoke test; not part of unittest discovery."""
from isaacgym import gymapi
import ast
import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'legged_gym'
spec = importlib.util.spec_from_file_location('blind_terrain', ROOT / 'utils/terrains/blind_terrain.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tree = ast.parse((ROOT / 'envs/a1/mp_multi_primitive_config.py').read_text())
config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotorPrimitivesCfg')
terrain_class = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'terrain')
namespace = {}
exec(compile(ast.Module(body=[terrain_class], type_ignores=[]), '<terrain config>', 'exec'), namespace)
cfg = namespace['terrain']()
parser = argparse.ArgumentParser()
parser.add_argument('--mesh', choices=['trimesh', 'heightfield'], default='trimesh')
args = parser.parse_args()
cfg.mesh_type, cfg.num_rows, cfg.num_cols = args.mesh, 1, len(module.TERRAIN_GENERATORS)
cfg.selected = None
cfg.terrain_proportions = dict.fromkeys(module.TERRAIN_GENERATORS, 1.)
cfg.difficulty_range = [1., 1.]
terrain = module.BlindTerrain(cfg, 5)
gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0., 0., -9.81)
params.dt = 1. / 120
params.physx.use_gpu = False
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
if sim is None:
    raise RuntimeError('Isaac Gym could not create the simulator')
try:
    if args.mesh == 'trimesh':
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices, mesh.nb_triangles = len(terrain.vertices), len(terrain.triangles)
        mesh.transform.p = gymapi.Vec3(-terrain.border_size, -terrain.border_size, 0.)
        mesh.static_friction, mesh.dynamic_friction = 1., 1.
        gym.add_triangle_mesh(sim, terrain.vertices.ravel(), terrain.triangles.ravel(), mesh)
    else:
        field = gymapi.HeightFieldParams()
        field.nbRows, field.nbColumns = terrain.tot_cols, terrain.tot_rows
        field.row_scale = field.column_scale = terrain.horizontal_scale
        field.vertical_scale = terrain.vertical_scale
        field.transform.p = gymapi.Vec3(-terrain.border_size, -terrain.border_size, 0.)
        gym.add_heightfield(sim, terrain.heightsamples.ravel(order="F"), field)
    sphere = gym.create_sphere(sim, .1, gymapi.AssetOptions())
    actors = []
    for origin in terrain.env_origins.reshape(-1, 3):
        env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 5)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(origin[0]), float(origin[1]), float(origin[2] + .5))
        actor = gym.create_actor(env, sphere, pose, 'probe', len(actors), 0)
        actors.append((env, actor, float(origin[2])))
    gym.prepare_sim(sim)
    for _ in range(240):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    for env, actor, ground in actors:
        position = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)['pose']['p'][0]
        height = float(position['z'])
        # Rough terrain has no artificial flat platform. Check the actual final
        # contact neighbourhood, not the central maximum used for safe spawning.
        ix = int((float(position['x']) + terrain.border_size) / terrain.horizontal_scale)
        iy = int((float(position['y']) + terrain.border_size) / terrain.horizontal_scale)
        radius = int(np.ceil(.1 / terrain.horizontal_scale)) + 1
        patch = terrain.heightsamples[ix-radius:ix+radius+1, iy-radius:iy+radius+1] * terrain.vertical_scale
        assert patch.size and np.isfinite(height)
        assert patch.min() + .08 <= height <= patch.max() + .12, (height, patch.min(), patch.max())
    print(f'PASS: {args.mesh}, {cfg.num_cols} terrain types accepted by PhysX; probes settled within local surface bounds.')

finally:
    gym.destroy_sim(sim)
