"""Geometry and joint-workspace screening; does NOT measure policy traversal success."""
from isaacgym import terrain_utils  # Must precede torch.
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'legged_gym/legged_gym'


def load_module(name, path, package=False):
    options = {'submodule_search_locations': [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config():
    path = SOURCE / 'envs/a1/mp_proprio_terrain_config.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotorPrimitivesCfg')
    cls.bases = []
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['MotorPrimitivesCfg']


def kinematic_screen(cfg, clearance=None):
    """Screen all gaits, four legs and 27 equilibrium-box positions.

    Report raw requests separately from bounded IK output. This samples static
    offsets, not the filter transient or the robot's contact dynamics.
    """
    module = load_module('audit_dynamics', SOURCE / 'utils/dynamics/a1/rhythmic_dynamics.py')
    bounded = load_module('audit_equilibrium', SOURCE / 'utils/dynamics/a1/equilibrium_dynamics.py', package=True)
    library = module.Multi_Primitive_Dynamics(num_envs=1, device='cpu')
    gait_count, phase_count = len(library.primitive_names), 128
    eq = cfg.control.equilibrium
    offsets = torch.cartesian_prod(*[torch.tensor([-v, 0., v]) for v in eq.offset_limits])
    combinations = torch.cartesian_prod(
        torch.arange(gait_count, dtype=torch.float32), torch.linspace(0., 1., 3),
        torch.tensor([cfg.asset.ground_clearance,
                      cfg.asset.max_ground_clearance if clearance is None else clearance]),
        torch.arange(phase_count) * (2*np.pi/phase_count), torch.arange(len(offsets)).float())
    n = len(combinations)
    model = module.Multi_Primitive_Dynamics(num_envs=n, device='cpu',
                                           nominal_clearance=cfg.asset.ground_clearance,
                                           max_clearance=cfg.asset.max_ground_clearance)
    model._MAX_STEP_LEN = cfg.asset.max_step_length
    model._gait_ids = combinations[:, 0].long()
    model.X[:, 0, :] = combinations[:, 1, None]
    model.X[:, 1, :] = combinations[:, 2, None]
    model.X[:, 2, :] = combinations[:, 3, None]
    model.stance_extension.fill_(model.max_stance_extension)
    command = torch.zeros(n, 3)
    command[:, 0] = cfg.commands.ranges.lin_vel_x[1]
    target = torch.stack(model._foot_trajectory(command, model._vertical_phase()), -1)
    target += offsets[combinations[:, 4].long(), None, :]
    target[:, :, 1] += target.new_tensor([1., -1., 1., -1.]) * cfg.asset.hip_link_length_a1
    x, y, z = target.unbind(-1)
    l1, l2, l3 = cfg.asset.hip_link_length_a1, cfg.asset.thigh_link_length_a1, cfg.asset.calf_link_length_a1
    cosine = (x*x + y*y + z*z - l1*l1 - l2*l2 - l3*l3) / (2*l2*l3)
    raw_joints = torch.stack([
        model.compute_inverse_kinematics(cfg.asset, leg, x[:, i], y[:, i], z[:, i])
        for i, leg in enumerate((1, 0, 3, 2))], dim=1)
    urdf_path = cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', str(ROOT / 'legged_gym'))
    urdf = ET.parse(urdf_path)
    limits = [urdf.find(f".//joint[@name='{leg}_{name}_joint']/limit")
              for leg in ('FL', 'FR', 'RL', 'RR') for name in ('hip', 'thigh', 'calf')]
    lower = torch.tensor([float(l.attrib['lower']) for l in limits]).reshape(4, 3)
    upper = torch.tensor([float(l.attrib['upper']) for l in limits]).reshape(4, 3)
    raw_margin = torch.minimum(raw_joints - lower, upper - raw_joints)
    raw_feasible = ((cosine >= -1) & (cosine <= 1) & torch.isfinite(raw_joints).all(dim=-1)
                    & (raw_margin >= 0).all(dim=-1))
    joints, achieved = bounded.bounded_inverse_kinematics(
        target, cfg.asset, torch.stack((lower, upper), -1).reshape(12, 2),
        eq.joint_limit_margin, eq.min_foot_depth, eq.min_sagittal_reach)
    joints = joints.reshape(n, 4, 3)
    margin = torch.minimum(joints - lower, upper - joints)
    feasible = (torch.isfinite(joints).all(dim=-1) & (margin >= 0).all(dim=-1))
    correction = torch.linalg.vector_norm(achieved - target, dim=-1)
    # Finite-difference reference speeds at the library frequencies, without
    # extra frequency actions or reflex phase acceleration. Not measured motor speeds.
    angle_grid = joints.reshape(gait_count, 3, 2, phase_count, len(offsets), 4, 3)
    derivative = (torch.roll(angle_grid, -1, 3) - torch.roll(angle_grid, 1, 3)) / (4*np.pi/phase_count)
    frequencies = model._primitive_frequencies[:, None, None, None, None, None, None]
    reference_speed = derivative.abs() * (2*np.pi*frequencies)
    velocity_limits = torch.tensor([float(l.attrib['velocity']) for l in limits]).reshape(4, 3)
    return dict(samples=n*4, infeasible_samples=int((~feasible).sum()),
                requested_infeasible_samples=int((~raw_feasible).sum()),
                minimum_joint_margin_rad=float(margin.min()),
                raw_cosine_range=[float(cosine.min()), float(cosine.max())],
                equilibrium_limits_m=list(eq.offset_limits),
                projected_sample_fraction=float((correction > 1e-5).float().mean()),
                max_projection_distance_m=float(correction.max()),
                reference_speed_limit_ratio=float((reference_speed / velocity_limits).max()),
                reference_speed_limit_ratio_by_gait={
                    name: float((reference_speed[i] / velocity_limits).max())
                    for i, name in enumerate(model.primitive_names)},
                speed_note='Reference-trajectory estimate at library frequency; not measured motor speed or torque feasibility.')


def terrain_screen(cfg, seeds, max_row=None):
    module = load_module('audit_terrain', SOURCE / 'utils/terrains/blind_terrain.py')
    rows = {}
    for seed in seeds:
        options = {k: deepcopy(getattr(cfg.terrain, k)) for k in dir(cfg.terrain) if not k.startswith('_')}
        options.update(mesh_type='heightfield', seed=seed)
        terrain = module.BlindTerrain(SimpleNamespace(**options), 32)
        for row in range(terrain.cfg.num_rows):
            if max_row is not None and row > max_row:
                continue
            for col in range(terrain.cfg.num_cols):
                x = terrain.border + row * terrain.length_per_env_pixels
                y = terrain.border + col * terrain.width_per_env_pixels
                height = terrain.heightsamples[x:x+terrain.length_per_env_pixels,
                                              y:y+terrain.width_per_env_pixels].astype(float) * terrain.vertical_scale
                name = terrain.terrain_names[row, col]
                record = rows.setdefault(name, dict(tiles=0, max_adjacent_height_m=0.,
                                                    max_sampled_grade=0., max_spawn_drop_m=0.))
                record['tiles'] += 1
                record['max_adjacent_height_m'] = max(record['max_adjacent_height_m'],
                    *(float(np.abs(np.diff(height, axis=a)).max()) for a in (0, 1)))
                record['max_sampled_grade'] = max(record['max_sampled_grade'],
                    float(np.hypot(*np.gradient(height, terrain.horizontal_scale)).max()))
                cx, cy = np.array(height.shape) // 2
                radius = int(np.ceil(cfg.terrain.spawn_xy_jitter / terrain.horizontal_scale))
                patch = height[cx-radius:cx+radius+1, cy-radius:cy+radius+1]
                record['max_spawn_drop_m'] = max(record['max_spawn_drop_m'],
                                                float(terrain.env_origins[row, col, 2] - patch.min()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 7, 19])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = load_config()
    report = dict(
        scope='Terrain geometry and six-gait equilibrium-box workspace, raw vs bounded IK; not certified traversal.',
        terrain=terrain_screen(cfg, args.seeds), kinematics=kinematic_screen(cfg), seeds=args.seeds)
    report['initial_terrain'] = terrain_screen(cfg, args.seeds, cfg.terrain.curriculum_initial_level)
    report['initial_geometry_and_position_screen_passed'] = (report['kinematics']['infeasible_samples'] == 0 and
                       all(r['max_adjacent_height_m'] <= .04 + 1e-8 and r['max_spawn_drop_m'] <= .08
                           for r in report['initial_terrain'].values()))
    report['challenge_geometry_screen_passed'] = all(
        r['max_adjacent_height_m'] <= .08 + 1e-8 and r['max_spawn_drop_m'] <= .08
        for r in report['terrain'].values())
    report['dynamic_tracking_certified'] = False
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    if not report['initial_geometry_and_position_screen_passed'] or not report['challenge_geometry_screen_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
