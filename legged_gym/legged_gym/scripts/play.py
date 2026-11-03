# License: see [LICENSE, LICENSES/legged_gym/LICENSE]
"""Play an archived policy using the environment configuration saved with its run."""

import isaacgym  # Import before torch or any module importing torch.
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.load_config import prepare_resume_configs
from legged_gym.utils.inference_forward_hold import (
    ForwardHoldConfig, InferenceForwardHold, format_policy_update_frequency)


def configure_forward_hold_scan(args, env_cfg):
    """Opt in to the 10 Hz scan required by adaptive MP_full inference."""
    if not getattr(args, 'adaptive_forward', False):
        return
    if args.task != 'MP_full_a1':
        raise ValueError('adaptive forward/hold is available only for MP_full_a1')
    env_cfg.terrain.scan_period_steps = 5


def configure_composite_terrain(env_cfg, terrain_rows, difficulty=1.0,
                                terrain_params=None, num_envs=None):
    """Arrange named terrains along world x for one continuous checkpoint run.

    ``terrain_rows`` gives the forward traversal order. ``difficulty`` is in
    [0, 1]. ``terrain_params`` extends the archived parameters by terrain type
    or named variant (for example, ``gap_single``). A ``gap_single`` row is
    centered by default in this playback layout. A ``hurdle`` row has one
    centered obstacle by default.
    By default, one robot starts in the first row. ``num_envs`` can start more
    robots on the same course. The configuration is changed in place.
    """
    rows = list(terrain_rows)
    if not rows:
        raise ValueError('terrain_rows must contain at least one terrain')
    if not 0 <= difficulty <= 1:
        raise ValueError('difficulty must be within [0, 1]')
    if num_envs is None:
        num_envs = 1
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')

    terrain = env_cfg.terrain
    if terrain.mesh_type not in ('trimesh', 'heightfield'):
        raise ValueError('composite terrain requires trimesh or heightfield')
    terrain.selected = None
    terrain.selected_kwargs = None
    terrain.terrain_columns = None
    terrain.terrain_rows = rows
    terrain.num_rows = len(rows)
    terrain.num_cols = 1
    terrain.difficulty_range = [difficulty, difficulty]
    terrain.difficulty_mode = 'linear'
    terrain.row_difficulties = None
    terrain.curriculum = False
    terrain.curriculum_min_level = 0
    terrain.curriculum_max_level = None
    terrain.fixed_terrain_level = 0
    env_cfg.env.episode_length_s = max(
        env_cfg.env.episode_length_s, len(rows) * terrain.terrain_length / .5)
    if 'gap_single' in rows:
        variant = terrain.terrain_variants.get('gap_single')
        if variant is None:
            raise ValueError('gap_single requires a registered terrain variant')
        variant.setdefault('params', {})['single_position'] = 'center'
    if 'hurdle' in rows:
        terrain.terrain_params.setdefault('hurdle', {})['pattern'] = 'single'
    if terrain_params is not None:
        for kind, params in terrain_params.items():
            if kind in terrain.terrain_variants:
                terrain.terrain_variants[kind].setdefault('params', {}).update(params)
            else:
                terrain.terrain_params.setdefault(kind, {}).update(params)
    env_cfg.env.num_envs = num_envs


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    args.resume = True
    env_cfg, train_cfg = prepare_resume_configs(args.task, env_cfg, train_cfg, args)
    configure_forward_hold_scan(args, env_cfg)
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    if args.task == 'MP_full_a1' and env_cfg.terrain.mesh_type in ('trimesh', 'heightfield'):
        env_cfg.terrain.terrain_length = 6.
        configure_composite_terrain(
            env_cfg,
            # Robot starts in row 0 and travels through these terrains in order.
            terrain_rows=['flat', 'rough_slope', 'ramp', 'gap_continuous',
                          'gap_single', 'hurdle', 'pit', 'flat'],
            difficulty=1.0,
            # Playback overrides for every row; ranges run from easy to hard.
            terrain_params={
                'flat': {'height': 0.},
                'rough_slope': {'slope_range': [0.02, 0.20], 'slope_signs': [-1, 1],
                                'height_range': [0.005, 0.025], 'height_step': 0.005,
                                'downsampled_scale': 0.3},
                'ramp': {'slope_range': [0.02, 0.30], 'slope_signs': [-1, 1],
                         'center_platform': True},
                'gap_continuous': {'pattern': 'centered', 'width_range': [0.1, 0.25],
                                   'depth_range': [0.6, 0.6], 'num_gaps': 2,
                                   'platform_size': 1.},
                # In single/center mode, spacing is required but does not move the gap or hurdle.
                'gap_single': {'pattern': 'single', 'single_position': 'center',
                               'width_range': [0.1, 0.5], 'depth_range': [0.6, 0.6],
                               'spacing': 3., 'platform_size': 3.},
                'hurdle': {'pattern': 'single', 'width_range': [0.2, 0.2],
                           'height_range': [0.05, 0.2], 'spacing': 4.},
                'stairs': {'step_height_range': [0.01, 0.1], 'step_width': 0.3,
                           'mesh_vertical_faces': True, 'step_signs': [-1, 1]},
                'pit': {'size_range': [4., 4.], 'depth_range': [0.1, 0.2]},
            },
            num_envs=args.num_envs,
        )
    else:
        env_cfg.env.num_envs = 1 if args.num_envs is None else args.num_envs

    env, env_cfg = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env, name=args.task, args=args, train_cfg=train_cfg,
        env_cfg=env_cfg, save_cfg=False)
    if getattr(args, 'adaptive_forward', False):
        runner.inference_forward_hold = InferenceForwardHold(
            ForwardHoldConfig(gait_max_skips=args.gait_max_skips,
                              modulation_max_skips=args.modulation_max_skips,
                              horizontal_rate_limit=args.horizontal_rate_limit,
                              modulation_horizontal_rate_limit=args.modulation_horizontal_rate_limit,
                              extero_edge_trigger=args.extero_edge_trigger,
                              terrain_edge_height_m=args.terrain_edge_height_m,
                              filter_normal_sway=args.filter_normal_sway,
                              tracking_error_trigger=not getattr(args, 'no_tracking_error_trigger', False),
                              tracking_linear_error_limit=getattr(args, 'tracking_linear_error_limit', .6),
                              tracking_yaw_error_limit=getattr(args, 'tracking_yaw_error_limit', .6)),
            env.scan_cache.x, env.scan_cache.y, env.num_envs, env.device)
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    two_policy = args.task == "MP_full_a1" or isinstance(policy, tuple)
    report_policy_updates = args.task == "MP_full_a1"
    forward_counts = {'gait': 0, 'modulation': 0}
    episode_forward_counts = {
        actor: torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        for actor in forward_counts} if report_policy_updates else {}
    episode_steps = (torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
                     if report_policy_updates else None)
    episode_numbers = [1] * env.num_envs if report_policy_updates else []
    active_env_steps = 0
    try:
        while True:
            if two_policy:
                obs, _, _, dones, _ = runner.inference_rollout(obs, *policy)
                if report_policy_updates:
                    episode_steps += 1
                    for actor in forward_counts:
                        selected = runner.last_inference_decisions[actor]['forward']
                        forward_counts[actor] += selected.numel()
                        episode_forward_counts[actor][selected] += 1
                    active_env_steps += env.num_envs
                    for env_id in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                        summary = format_policy_update_frequency(
                            episode_forward_counts['gait'][env_id].item(),
                            episode_forward_counts['modulation'][env_id].item(),
                            episode_steps[env_id].item(), env.dt)
                        print(f'env {env_id} episode {episode_numbers[env_id]}: {summary}', flush=True)
                        episode_numbers[env_id] += 1
                        episode_steps[env_id] = 0
                        for actor in episode_forward_counts:
                            episode_forward_counts[actor][env_id] = 0
            elif args.task in ("MP_interoceptive_a1", "MP_refmotion_pace_a1", "MP_refmotion_trot_a1", "MP_refmotion_canter_a1"):
                obs, _, _, _, _ = runner.inference_rollout(obs, policy)
            else:
                actions = policy(obs[:, -1] if obs.ndim == 3 else obs)
                obs, _, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        if report_policy_updates:
            print(format_policy_update_frequency(
                forward_counts['gait'], forward_counts['modulation'],
                active_env_steps, env.dt), flush=True)


if __name__ == "__main__":
    play(get_args())
