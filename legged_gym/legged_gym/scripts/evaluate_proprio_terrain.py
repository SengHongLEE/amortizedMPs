"""Evaluate first-episode traversal on every enabled terrain, with fixed commands."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'legged_gym/legged_gym/test/torch_extensions'))
from isaacgym import gymapi, gymutil
import numpy as np
import torch
from legged_gym.envs.a1.mp_proprio_terrain import MotorPrimitives
from legged_gym.envs.a1.mp_proprio_terrain_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.terrain_evaluation import TraversalMetrics, stair_evaluation_rows
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


class EvaluationEnv(MotorPrimitives):
    def _get_env_origins(self):
        super()._get_env_origins()
        ids = torch.arange(self.num_envs, device=self.device)
        self.speed_ids = ids % len(self.test_speeds)
        tile = ids // (len(self.test_speeds) * self.repeats)
        self.terrain_types[:] = tile % self.cfg.terrain.num_cols
        self.terrain_levels[:] = tile // self.cfg.terrain.num_cols
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self.eval_commands = torch.tensor(self.test_speeds, device=self.device)[self.speed_ids]

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.
        self.commands[env_ids, 0] = self.eval_commands[env_ids]

    def check_termination(self, env_ids=None):
        super().check_termination(env_ids)
        if getattr(self, 'metrics', None) is not None:
            control = torch.cat((self.ID.f_gait, self.ID.selected_duty_factors(self.ID._gait_ids),
                                 self.ID.X[:, 1]), dim=-1)
            self.control_integral += self.metrics.active[:, None] * control * self.dt
            self.metrics.update(self.root_states[:, :2] - self.env_origins[:, :2],
                                self.terrain_failure_buf, self.time_out_buf,
                                (self.base_lin_vel[:, 0] - self.eval_commands).abs(), self.dt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--seconds', type=float, default=15.)
    p.add_argument('--speeds', type=float, nargs='+', default=[.5, .8, 1.2])
    rows = p.add_mutually_exclusive_group()
    rows.add_argument('--difficulties', type=float, nargs='+', default=[.5, 1.])
    rows.add_argument('--stair-heights', type=float, nargs='+',
                      help='Exact increasing step heights in metres; requires --terrains stairs.')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--terrain-seed', type=int, default=1)
    p.add_argument('--corridor', type=float, default=1.5)
    p.add_argument('--terrains', nargs='+', help='Evaluate a subset of the enabled terrain types.')
    p.add_argument('--vertical-stairs', action='store_true',
                   help='Evaluate true stair risers at every height, overriding legacy source geometry.')
    p.add_argument('--sample-low', action='store_true', help='Diagnostic: sample the learned low-level Gaussian.')
    p.add_argument('--constant-actions', type=Path, help='Diagnostic: JSON action vector instead of either policy.')
    args = p.parse_args()
    if args.repeats < 1 or min(args.speeds) <= 0 or args.seconds <= 0 or args.corridor <= 0:
        p.error('repeats, speeds, seconds and corridor must be positive')
    checkpoint = args.checkpoint.resolve()
    sources = {}
    for base in ('legged_gym/legged_gym', 'rsl_rl/rsl_rl'):
        for path in (ROOT / base).rglob('*.py'):
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg, train = load_run_config(MotorPrimitivesCfg(), MotorPrimitivesCfgPPO(), checkpoint.parent)
    if args.vertical_stairs:
        for kind in ('stairs', 'stairs_linear'):
            cfg.terrain.terrain_params.setdefault(kind, {})['mesh_vertical_faces'] = True
    set_seed(args.seed)
    enabled = [name for name, weight in cfg.terrain.terrain_proportions.items() if weight > 0]
    selected = args.terrains or enabled
    if any(name not in enabled for name in selected) or len(set(selected)) != len(selected):
        p.error('--terrains must be distinct types enabled in the source config')
    if args.stair_heights is not None:
        if selected != ['stairs']:
            p.error('--stair-heights requires --terrains stairs')
        try:
            height_range, args.difficulties = stair_evaluation_rows(
                args.stair_heights, cfg.terrain.vertical_scale)
        except ValueError as error:
            p.error(str(error))
        cfg.terrain.terrain_params['stairs']['step_height_range'] = height_range
    columns = [name for name in selected for _ in range(2)]
    cfg.env.num_envs = len(columns) * len(args.difficulties) * args.repeats * len(args.speeds)
    cfg.env.play = False
    cfg.env.episode_length_s = args.seconds
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.terrain.curriculum = False
    cfg.terrain.fixed_terrain_level = None
    cfg.terrain.max_init_terrain_level = None
    cfg.terrain.curriculum_min_level = 0
    cfg.terrain.curriculum_max_level = None
    cfg.terrain.num_rows, cfg.terrain.num_cols = len(args.difficulties), len(columns)
    cfg.terrain.selected = cfg.terrain.selected_kwargs = None
    cfg.terrain.terrain_columns = columns
    cfg.terrain.row_difficulties = args.difficulties
    cfg.terrain.seed = args.terrain_seed
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    EvaluationEnv.repeats, EvaluationEnv.test_speeds = args.repeats, args.speeds
    env = EvaluationEnv(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        runner = MultiInteroceptivePolicyRunner(env, class_to_dict(train), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_optimizer=False)
        high, low = runner.get_inference_policy(device=env.device)
        if args.sample_low:
            low = runner.modulation_param_actor_critic.act
        constant = None
        if args.constant_actions is not None:
            if args.sample_low:
                p.error('--sample-low and --constant-actions are mutually exclusive')
            values = json.loads(args.constant_actions.read_text())
            constant = torch.tensor(values, device=env.device, dtype=torch.float)
            if constant.shape != (cfg.env.num_actions,) or not torch.isfinite(constant).all():
                p.error('constant-actions must contain a finite action vector matching the controller')
            if (constant[1:].abs() > 1).any() or constant[0] != int(constant[0]) or not 0 <= constant[0] < 6:
                p.error('constant-actions requires gait 0..5 and continuous values in [-1,1]')
            constant = constant[None].expand(env.num_envs, -1)
        # Constructing different-size networks consumes different amounts of RNG.
        # Reseed *after* construction so architecture comparisons start identically.
        set_seed(args.seed)
        obs, _ = env.reset()
        initial_root = env.root_states.detach().cpu().numpy().copy()
        initial_q = env.dof_pos.detach().cpu().numpy().copy()
        initial_state_hash = hashlib.sha256(initial_root.tobytes() + initial_q.tobytes()).hexdigest()
        env.metrics = TraversalMetrics(env.num_envs, env.device,
                                      cfg.terrain.terrain_length / 2 - cfg.terrain.boundary_margin,
                                      args.corridor)
        env.control_integral = torch.zeros(env.num_envs, 6, device=env.device)
        start = time.time()
        with open(os.devnull, 'w') as quiet, torch.inference_mode():
            for step in range(round(args.seconds / env.dt)):
                with contextlib.redirect_stdout(quiet):
                    if constant is None:
                        obs, _, _, _, _ = runner.inference_rollout(obs, high, low)
                    else:
                        obs, _, _, _, _ = env.step(constant)
                if step % 100 == 0:
                    print(f'{step * env.dt:.1f}s simulated; {time.time()-start:.1f}s wall; '
                          f'{env.metrics.active.sum().item()} active', flush=True)
                if not env.metrics.active.any():
                    break
        records = env.metrics.records()
        control_means = (env.control_integral / env.metrics.duration[:, None].clamp_min(1e-6)).cpu().tolist()
        for record, control in zip(records, control_means):
            record.update(mean_frequency_hz=control[0], mean_duty_factor=control[1],
                          mean_swing_amplitude_m=control[2:])
        groups = []
        for row, difficulty in enumerate(args.difficulties):
            for col, name in enumerate(columns):
                for speed_id, speed in enumerate(args.speeds):
                    first = (row * len(columns) + col) * args.repeats * len(args.speeds) + speed_id
                    episodes = [dict(env=i, **records[i]) for i in
                                range(first, first + args.repeats * len(args.speeds), len(args.speeds))]
                    groups.append(dict(terrain=name, column=col, difficulty=difficulty, speed=speed,
                                       parameters=env.terrain.tile_parameters[row, col], episodes=episodes,
                                       success_rate=np.mean([e['success'] for e in episodes])))
        report = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      seed=args.seed, terrain_seed=args.terrain_seed, seconds=args.seconds, corridor=args.corridor,
                      initial_state_sha256=initial_state_hash, sample_low=args.sample_low,
                      requested_stair_heights=args.stair_heights,
                      constant_actions=None if constant is None else constant[0].cpu().tolist(),
                      effective_config=class_to_dict(cfg), source_hashes=sources, groups=groups,
                      notes='High actor deterministic; low actor deterministic unless sample_low. '
                            'First episode only, fixed forward commands. No pushes, '
                            'mass/friction randomization or observation noise; initial jitter retained. '
                            'Success requires forward exit without falling or exceeding lateral corridor.')
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / 'initial_state.npz', root=initial_root, dof_pos=initial_q)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        for name in dict.fromkeys(columns):
            es = [e for g in groups if g['terrain'] == name for e in g['episodes']]
            print(name, f"{sum(e['success'] for e in es)}/{len(es)}",
                  'falls', sum(e['failed'] for e in es), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
