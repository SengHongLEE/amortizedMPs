"""Paired first-episode evaluation of a fixed MP_full gait and its learned low policy."""
import argparse
import contextlib
from dataclasses import asdict
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

from legged_gym.envs.a1.mp_full import MotorPrimitives
from legged_gym.envs.a1.mp_full_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.inference_forward_hold import (
    ForwardHoldConfig, InferenceForwardHold, format_policy_update_frequency)
from legged_gym.utils.terrain_evaluation import TraversalMetrics
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


GAITS = ('walk', 'trot', 'canter', 'bound')


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
        if getattr(self, 'metrics', None) is None:
            return
        active = self.metrics.active.clone()
        self.active_at_step = active
        positive_power = (self.substep_torques * self.substep_dof_vel
                          + .3 * self.substep_torques.square()).clamp_min(0.)
        self.work_proxy += positive_power.sum(-1).mean(1) * self.dt * active
        self.cot_integral += self._pi_g_cost_of_transport() * self.dt * active
        self.metrics.update(self.root_states[:, :2] - self.env_origins[:, :2],
                            self.terrain_failure_buf, self.time_out_buf,
                            (self.base_lin_vel[:, 0] - self.eval_commands).abs(), self.dt)

    def compute_reward(self, env_ids=None):
        rewards = super().compute_reward(env_ids)
        if getattr(self, 'metrics', None) is not None:
            self.task_return += self.rew_buf * self.active_at_step
            self.selection_return += self.primitive_selection_rew_buf * self.active_at_step
        return rewards


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gait', choices=GAITS + ('policy',), required=True)
    p.add_argument('--terrains', nargs='+', default=['flat', 'stairs', 'hurdle', 'gap_single'])
    p.add_argument('--difficulties', type=float, nargs='+', default=[.3, .8])
    p.add_argument('--speeds', type=float, nargs='+', default=[.4, .8, 1.5, 2.2])
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--seconds', type=float, default=12.)
    p.add_argument('--seed', type=int, default=321)
    p.add_argument('--terrain-seed', type=int, default=3)
    p.add_argument('--corridor', type=float, default=1.5)
    p.add_argument('--scan-period-steps', type=int, choices=(1, 5), default=1,
                   help='1 = fresh every 20 ms; 5 = 10 Hz pose-compensated scan')
    p.add_argument('--adaptive-forward', action='store_true',
                   help='apply inference-only explicit forward/hold to both actors')
    p.add_argument('--gait-max-skips', type=int, default=2)
    p.add_argument('--modulation-max-skips', type=int, default=3)
    p.add_argument('--tilt-limit', type=float, default=.25)
    p.add_argument('--horizontal-rate-limit', type=float, default=2.)
    p.add_argument('--modulation-horizontal-rate-limit', type=float, default=4.,
                   help='override only the modulation actor horizontal angular rate threshold')
    p.add_argument('--filter-normal-sway', action='store_true',
                   help='smooth ordinary horizontal angular oscillation during inference')
    p.add_argument('--obstacle-height-m', type=float, default=.05)
    p.add_argument('--extero-edge-trigger', action='store_true',
                   help='use the default abrupt known terrain-edge trigger')
    p.add_argument('--no-extero-edge-trigger', dest='extero_edge_trigger', action='store_false',
                   help='disable the default abrupt known terrain-edge trigger')
    p.set_defaults(extero_edge_trigger=True)
    p.add_argument('--terrain-edge-height-m', type=float, default=.035)
    p.add_argument('--forward-corridor-half-width-m', type=float, default=.2)
    p.add_argument('--response-s', type=float, default=.10)
    p.add_argument('--distance-margin-m', type=float, default=.15)
    args = p.parse_args()
    if (args.repeats < 1 or args.seconds <= 0 or args.corridor <= 0
            or not args.speeds or min(args.speeds) <= 0
            or not args.difficulties or min(args.difficulties) < 0
            or max(args.difficulties) > 1 or len(set(args.terrains)) != len(args.terrains)):
        p.error('invalid repeats, seconds, corridor, speeds, difficulties or terrains')
    if args.gait_max_skips < 0 or args.modulation_max_skips < 0:
        p.error('gait-max-skips and modulation-max-skips must be nonnegative')
    if args.adaptive_forward and args.scan_period_steps != 5:
        p.error('--adaptive-forward requires --scan-period-steps 5')
    forward_config = ForwardHoldConfig(
        gait_max_skips=args.gait_max_skips,
        modulation_max_skips=args.modulation_max_skips,
        tilt_limit=args.tilt_limit,
        horizontal_rate_limit=args.horizontal_rate_limit,
        modulation_horizontal_rate_limit=args.modulation_horizontal_rate_limit,
        filter_normal_sway=args.filter_normal_sway,
        obstacle_height_m=args.obstacle_height_m,
        extero_edge_trigger=args.extero_edge_trigger,
        terrain_edge_height_m=args.terrain_edge_height_m,
        corridor_half_width_m=args.forward_corridor_half_width_m,
        response_s=args.response_s,
        distance_margin_m=args.distance_margin_m)
    checkpoint = args.checkpoint.resolve()
    sources = {}
    for name in ('legged_gym/legged_gym/envs/a1/mp_full.py',
                 'legged_gym/legged_gym/envs/a1/mp_full_config.py',
                 'legged_gym/legged_gym/utils/temporal_scan.py',
                 'legged_gym/legged_gym/utils/inference_forward_hold.py',
                 'legged_gym/legged_gym/utils/dynamics/a1/adaptive_equilibrium_dynamics.py',
                 'rsl_rl/rsl_rl/runners/multi_interoceptive_policy_runner.py',
                 'legged_gym/legged_gym/scripts/evaluate_mp_full_gaits.py'):
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    cfg, train = load_run_config(MotorPrimitivesCfg(), MotorPrimitivesCfgPPO(), checkpoint.parent)
    archived = json.loads((checkpoint.parent / 'config.json').read_text())
    if 'pi_g_cot_reference' not in archived.get('rewards', {}):
        # Older checkpoints used the unnormalized CoT reward. A new class default
        # must not silently change their evaluation reward when loading them.
        cfg.rewards.pi_g_cot_reference = 1.
    # Report task performance without the training-only, evidence-based gait bonus.
    cfg.rewards.primitive_selection_scales.pi_g_context_guidance = 0.
    cfg.terrain.scan_period_steps = args.scan_period_steps
    if any(name not in cfg.terrain.terrain_columns for name in args.terrains):
        p.error('all terrains must occur in the saved training layout')
    set_seed(args.seed)
    cfg.env.num_envs = len(args.difficulties) * len(args.terrains) * len(args.speeds) * args.repeats
    cfg.env.play = False
    cfg.env.episode_length_s = args.seconds
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows, cfg.terrain.num_cols = len(args.difficulties), len(args.terrains)
    cfg.terrain.selected = cfg.terrain.selected_kwargs = None
    cfg.terrain.terrain_columns = args.terrains
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
        if args.adaptive_forward:
            runner.inference_forward_hold = InferenceForwardHold(
                forward_config, env.scan_cache.x, env.scan_cache.y,
                env.num_envs, env.device)
        gait_id = None if args.gait == 'policy' else GAITS.index(args.gait)

        def fixed_high(observations, reservoir_states):
            return torch.full((len(observations),), gait_id,
                              device=observations.device, dtype=torch.long), reservoir_states

        # Network construction and runner reset consume random numbers. Restore the
        # same seed before each gait's initial episode for exact paired starts.
        set_seed(args.seed)
        obs, _ = env.reset()
        initial_obs = obs.detach().cpu().numpy().copy()
        initial_root = env.root_states.detach().cpu().numpy().copy()
        initial_q = env.dof_pos.detach().cpu().numpy().copy()
        initial_hash = hashlib.sha256(initial_root.tobytes() + initial_q.tobytes()).hexdigest()
        env.metrics = TraversalMetrics(env.num_envs, env.device,
                                       cfg.terrain.terrain_length / 2 - cfg.terrain.boundary_margin,
                                       args.corridor)
        env.work_proxy = torch.zeros(env.num_envs, device=env.device)
        env.cot_integral = torch.zeros_like(env.work_proxy)
        env.task_return = torch.zeros_like(env.work_proxy)
        env.selection_return = torch.zeros_like(env.work_proxy)
        gait_steps = torch.zeros(env.num_envs, len(GAITS), device=env.device)
        scan_min = torch.full((env.num_envs,), float('inf'), device=env.device)
        scan_max = torch.full((env.num_envs,), -float('inf'), device=env.device)
        scan_frames = torch.ones(env.num_envs, device=env.device, dtype=torch.long)
        scan_age_sum = torch.zeros(env.num_envs, device=env.device)
        scan_age_max = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        unknown_sum = torch.zeros(env.num_envs, device=env.device)
        active_steps = torch.zeros(env.num_envs, device=env.device)
        gait_forwards = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        gait_switches = torch.zeros_like(gait_forwards)
        modulation_forwards = torch.zeros_like(gait_forwards)
        gait_candidates = torch.zeros_like(gait_forwards)
        modulation_candidates = torch.zeros_like(gait_forwards)
        reasons = {actor: {key: torch.zeros_like(gait_forwards) for key in
                            ('first', 'cap', 'command', 'proprio', 'tracking', 'terrain', 'invalid', 'hold')}
                   for actor in ('gait', 'modulation')}
        actor_invocations = {'gait': 0, 'modulation': 0}
        actor_batch_envs = {'gait': 0, 'modulation': 0}
        env_ids = torch.arange(env.num_envs, device=env.device)
        torch.cuda.synchronize(env.device)
        started = time.perf_counter()
        simulated_steps = 0
        with open(os.devnull, 'w') as quiet, torch.inference_mode():
            for step in range(round(args.seconds / env.dt)):
                active_before = env.metrics.active.clone()
                previous_gait = runner.primitive_selections.clone()
                prior_frames = env.scan_cache.frame_count.clone() if env.scan_cache is not None else None
                with contextlib.redirect_stdout(quiet):
                    obs, _, _, _, _ = runner.inference_rollout(
                        obs, high if gait_id is None else fixed_high, low)
                simulated_steps += 1
                if step > 0:
                    gait_switches += ((runner.primitive_selections != previous_gait)
                                      & active_before).long()
                for actor, candidates, forwards in (
                        ('gait', gait_candidates, gait_forwards),
                        ('modulation', modulation_candidates, modulation_forwards)):
                    decision = runner.last_inference_decisions[actor]
                    candidate_ids = decision['candidates']
                    forward_ids = decision['forward']
                    candidates[candidate_ids] += active_before[candidate_ids].long()
                    forwards[forward_ids] += active_before[forward_ids].long()
                    actor_invocations[actor] += int(forward_ids.numel() > 0)
                    actor_batch_envs[actor] += forward_ids.numel()
                    for key, mask in decision['reasons'].items():
                        chosen = candidate_ids[mask]
                        reasons[actor][key][chosen] += active_before[chosen].long()
                scan_frames += (active_before.long() if prior_frames is None else
                                (env.scan_cache.frame_count - prior_frames) * active_before.long())
                scan_age_sum += env.scan_age_steps * active_before
                scan_age_max = torch.where(active_before,
                                           torch.maximum(scan_age_max, env.scan_age_steps), scan_age_max)
                unknown_sum += env.scan_unknown_fraction * active_before
                active_steps += active_before
                finished = active_before & ~env.metrics.active
                for i in torch.nonzero(finished, as_tuple=False).flatten().tolist():
                    summary = format_policy_update_frequency(
                        gait_forwards[i].item(), modulation_forwards[i].item(),
                        active_steps[i].item(), env.dt)
                    print(f'env {i} episode 1: {summary}', flush=True)
                if step == 0 and gait_id is not None and not (env.ID._gait_ids == gait_id).all():
                    raise AssertionError('requested gait was not executed by the controller')
                gait_steps[env_ids, env.ID._gait_ids] += env.active_at_step
                scan = obs[:, -1, -env._scan_output_dim:]
                scan_min = torch.where(env.active_at_step,
                                       torch.minimum(scan_min, scan.amin(-1)), scan_min)
                scan_max = torch.where(env.active_at_step,
                                       torch.maximum(scan_max, scan.amax(-1)), scan_max)
                if step % 100 == 0:
                    print(f'{args.gait} {step * env.dt:.1f}s simulated; '
                          f'{time.perf_counter()-started:.1f}s wall; '
                          f'{env.metrics.active.sum().item()} active', flush=True)
                if not env.metrics.active.any():
                    break
        torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - started
        for i in torch.nonzero(env.metrics.active, as_tuple=False).flatten().tolist():
            summary = format_policy_update_frequency(
                gait_forwards[i].item(), modulation_forwards[i].item(),
                active_steps[i].item(), env.dt)
            print(f'env {i} episode 1 (evaluation cutoff): {summary}', flush=True)
        print(format_policy_update_frequency(
            gait_forwards.sum().item(), modulation_forwards.sum().item(),
            active_steps.sum().item(), env.dt), flush=True)
        env.metrics.timeout |= env.metrics.active
        records = env.metrics.records()
        for i, record in enumerate(records):
            record.update(gait_fraction=(gait_steps[i] / gait_steps[i].sum().clamp_min(1)).cpu().tolist(),
                          work_proxy_J=env.work_proxy[i].item(),
                          mean_raw_cot=(env.cot_integral[i] / env.metrics.duration[i].clamp_min(1e-6)).item(),
                          scan_min=scan_min[i].item(), scan_max=scan_max[i].item(),
                          scan_frames=scan_frames[i].item(),
                          mean_scan_age_steps=(scan_age_sum[i] / active_steps[i].clamp_min(1)).item(),
                          max_scan_age_steps=scan_age_max[i].item(),
                          mean_unknown_fraction=(unknown_sum[i] / active_steps[i].clamp_min(1)).item(),
                          gait_forwards=gait_forwards[i].item(),
                          gait_switches=gait_switches[i].item(),
                          modulation_forwards=modulation_forwards[i].item(),
                          gait_candidates=gait_candidates[i].item(),
                          modulation_candidates=modulation_candidates[i].item(),
                          gait_forward_reasons={key: value[i].item() for key, value in reasons['gait'].items()},
                          modulation_forward_reasons={key: value[i].item() for key, value in reasons['modulation'].items()},
                          task_return=env.task_return[i].item(),
                          selection_return=env.selection_return[i].item())
        groups = []
        for row, difficulty in enumerate(args.difficulties):
            for col, name in enumerate(args.terrains):
                for speed_id, speed in enumerate(args.speeds):
                    first = (row * len(args.terrains) + col) * args.repeats * len(args.speeds) + speed_id
                    episodes = [dict(env=i, **records[i]) for i in
                                range(first, first + args.repeats * len(args.speeds), len(args.speeds))]
                    groups.append(dict(terrain=name, difficulty=difficulty, speed=speed,
                                       parameters=env.terrain.tile_parameters[row, col], episodes=episodes))
        report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      gait=args.gait, seed=args.seed, terrain_seed=args.terrain_seed,
                      scan_period_steps=args.scan_period_steps,
                      scan_hz=1. / (args.scan_period_steps * env.dt),
                      adaptive_forward=args.adaptive_forward,
                      forward_hold_config=asdict(forward_config) if args.adaptive_forward else None,
                      actor_invocations=actor_invocations,
                      actor_batch_envs=actor_batch_envs,
                      rollout_wall_seconds=elapsed,
                      rollout_simulated_steps=simulated_steps,
                      rollout_active_env_steps=active_steps.sum().item(),
                      seconds=args.seconds, corridor=args.corridor, initial_state_sha256=initial_hash,
                      source_hashes=sources, effective_config=class_to_dict(cfg), groups=groups,
                      notes='Same checkpoint and deterministic low policy; fixed high gait. '
                            'First episode; no noise, pushes or domain randomization. '
                            'Success is 3.4m forward progress inside 1.5m corridor without failure.')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'{args.gait}.json').write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(args.output / f'{args.gait}_initial.npz', root=initial_root,
                            dof_pos=initial_q, observation=initial_obs)
        for name in args.terrains:
            es = [e for group in groups if group['terrain'] == name for e in group['episodes']]
            print(name, sum(e['success'] for e in es), '/', len(es),
                  'falls', sum(e['failed'] for e in es), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
