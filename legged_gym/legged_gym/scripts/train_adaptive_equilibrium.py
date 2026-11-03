"""Continue an 18-action dynamical proprio policy, then train a bounded run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'legged_gym/legged_gym/test/torch_extensions'))
from isaacgym import gymapi, gymutil
import torch
from legged_gym.envs.a1.mp_proprio_terrain import MotorPrimitives
from legged_gym.envs.a1.mp_proprio_terrain_config import AdaptiveMotorPrimitivesCfg, AdaptiveMotorPrimitivesCfgPPO
from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--initial-level', type=int, default=5)
    p.add_argument('--minimum-level', type=int, default=0, help='Curriculum floor for this continuation run.')
    p.add_argument('--stair-weight', type=float, help='Override stairs mixture weight, retaining other enabled types.')
    p.add_argument('--terrain-cols', type=int, help='Set enough columns to retain both directions when changing mixture weights.')
    p.add_argument('--vertical-stairs', action='store_true',
                   help='Explicitly train with true stair risers at every height, including legacy source configs.')
    p.add_argument('--legacy-stroke', action='store_true', help='Ablation: retain the old horizontal phase mapping.')
    p.add_argument('--task-aligned-selection', action='store_true',
                   help='Train gait selection on the same task/safety rewards as modulation. Otherwise preserve source configuration.')
    p.add_argument('--entropy-coef', type=float,
                   help='Override the PPO entropy coefficient for this continuation.')
    p.add_argument('--modulation-std-scale', type=float,
                   help='Explicitly scale loaded continuous exploration std; deterministic means stay unchanged.')
    args = p.parse_args()
    if args.iterations < 0 or args.num_envs < 16 or args.learning_rate <= 0:
        p.error('iterations >= 0, num-envs >= 16 and positive learning-rate are required')
    if args.entropy_coef is not None and (not math.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        p.error('entropy-coef must be finite and nonnegative')
    if args.modulation_std_scale is not None and (
            not math.isfinite(args.modulation_std_scale) or args.modulation_std_scale <= 0):
        p.error('modulation-std-scale must be finite and positive')
    if args.output.exists():
        p.error('output must be a new directory to preserve experiment provenance')
    cfg, train = load_run_config(AdaptiveMotorPrimitivesCfg(), AdaptiveMotorPrimitivesCfgPPO(),
                                args.checkpoint.resolve().parent)
    source_version = cfg.env.action_space
    if source_version not in ('cpg_equilibrium_v3', 'cpg_modal_equilibrium_v4'):
        p.error('Only new 18-action equilibrium v3 or modal v4 checkpoints are supported')
    modal = source_version == 'cpg_modal_equilibrium_v4'
    cfg.env.num_modulation_params, cfg.env.num_actions = 18, 19
    cfg.env.num_controller_observations = 46
    cfg.env.num_observations, cfg.env.num_privileged_obs = 102, 109
    cfg.env.num_envs = args.num_envs
    cfg.env.play = False
    if args.vertical_stairs:
        for kind in ('stairs', 'stairs_linear'):
            cfg.terrain.terrain_params.setdefault(kind, {})['mesh_vertical_faces'] = True
    if args.task_aligned_selection:
        cfg.rewards.primitive_selection_rewards = list(cfg.rewards.modulation_param_rewards)
        cfg.rewards.primitive_selection_scales = SimpleNamespace(**{
            name: getattr(cfg.rewards.scales, name)
            for name in cfg.rewards.modulation_param_rewards})
    cfg.control.timing.align_stroke = not args.legacy_stroke
    cfg.terrain.curriculum_initial_level = args.initial_level
    if not 0 <= args.minimum_level <= args.initial_level < cfg.terrain.num_rows:
        p.error('Require 0 <= minimum-level <= initial-level < terrain.num_rows')
    cfg.terrain.curriculum_min_level = args.minimum_level
    if args.stair_weight is not None:
        if not 0 < args.stair_weight < float('inf'):
            p.error('stair-weight must be positive and finite')
        cfg.terrain.terrain_proportions['stairs'] = args.stair_weight
    if args.terrain_cols is not None:
        cfg.terrain.num_cols = args.terrain_cols
    train.seed = cfg.seed = args.seed
    train.algorithm.learning_rate = args.learning_rate
    train.algorithm.schedule = 'fixed'
    if args.entropy_coef is not None:
        train.algorithm.entropy_coef = args.entropy_coef
    train.runner.resume = False
    train.runner.max_iterations = args.iterations
    train.runner.save_interval = 10
    train.runner.run_name = 'A1MPproprioterrain-modal-v4' if modal else 'A1MPproprioterrain-equilibrium-v3'
    args.output.mkdir(parents=True)
    config = {**class_to_dict(cfg), **class_to_dict(train)}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    provenance = dict(source_checkpoint=str(args.checkpoint.resolve()), source_version=source_version,
                      source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      source_hashes={})
    for base in ('legged_gym/legged_gym', 'rsl_rl/rsl_rl'):
        for path in (ROOT / base).rglob('*.py'):
            relative = path.relative_to(ROOT)
            saved = args.output / 'source' / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, saved)
            provenance['source_hashes'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    set_seed(args.seed)
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    env = MotorPrimitives(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        print('TERRAIN_COLUMNS', env.terrain.terrain_names[0].tolist(), flush=True)
        runner = MultiInteroceptivePolicyRunner(env, class_to_dict(train), str(args.output), device=env.device)
        source = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        for high, key, model in [
            (True, 'primitive_selection_model_state_dict', runner.primitive_selection_actor_critic),
            (False, 'modulation_param_model_state_dict', runner.modulation_param_actor_critic),
        ]:
            weights = source[key]
            if not high and args.modulation_std_scale is not None:
                weights['std'] = weights['std'] * args.modulation_std_scale
            model.load_state_dict(weights, strict=True)
        runner.save(str(args.output / 'warmstart.pt'), infos=provenance)
        print('WARMSTART', args.output / 'warmstart.pt', flush=True)
        if args.iterations:
            runner.learn(args.iterations, init_at_random_ep_len=True)
    finally:
        if 'runner' in locals() and runner.writer is not None:
            runner.writer.close()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
