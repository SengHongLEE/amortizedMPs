"""Phase 1 paired hold labels, supervised Terrain SNN, and fixed-clock evaluation.

All deadlines are environment control steps. This script deliberately rebuilds
the simulator for each branch; it refuses a label unless the intervention prefix
matches exactly. It does not claim arbitrary PhysX snapshot/restore support.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'legged_gym/legged_gym/test/torch_extensions'))
from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import torch
import torch.nn.functional as F

from legged_gym.envs.a1.mp_full_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
from legged_gym.scripts.evaluate_mp_full_gaits import EvaluationEnv
from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.terrain_evaluation import TraversalMetrics
from rsl_rl.modules.terrain_hold_experiment import (
    HeldPolicyController, FixedTerrainClock, compare_quality, conservative_outcomes, prefix_digest,
    candidate_result, planar_speed_error,
)
from rsl_rl.modules.terrain_hold_snn import (
    HoldBins, TerrainHoldSNN, overestimation_counts, verified_hold_label,
)
from rsl_rl.modules.terrain_scan_encoder import TerrainScanEncoder
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


DEFAULT_CHECKPOINT = (ROOT / 'legged_gym/logs/A1/'
                      '1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt')
FIXED_CHECKPOINT_SHA256 = '169b3d90f9fba128810fd2e5bc1be2b71a32f6de1eb18632a17b28c0eee294f2'
CONTROL_DT = (MotorPrimitivesCfg.sim.dt * MotorPrimitivesCfg.control.decimation
              * MotorPrimitivesCfg.control.cycle)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class SafetyEvaluationEnv(EvaluationEnv):
    disturbance_seed = 0
    push_start_step = 0

    def _push_robots(self):
        # A local generator makes each event independent of actor calls and resets.
        if self.common_step_counter <= self.push_start_step:
            return
        rng = np.random.default_rng(self.disturbance_seed + int(self.common_step_counter) * 104729)
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        velocity = rng.uniform(-max_vel, max_vel, (self.num_envs, 2))
        self.root_states[:, 7:9] = torch.as_tensor(velocity, dtype=self.root_states.dtype,
                                                  device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self, env_ids=None):
        super().check_termination(env_ids)
        speed_error = planar_speed_error(self.base_lin_vel, self.commands)
        collision = (torch.norm(self.contact_forces[:, self.penalised_contact_indices],
                                dim=-1) > .1).float().sum(-1)
        metrics = getattr(self, 'metrics', None)
        self.phase1_quality = dict(
            failed=self.terrain_failure_buf.clone(), timeout=self.time_out_buf.clone(),
            completed=(metrics.success.clone() if metrics is not None
                       else torch.zeros_like(self.terrain_failure_buf)),
            speed_error=speed_error.clone(),
            tilt=self.projected_gravity[:, :2].norm(dim=-1).clone(),
            slip=(self._slip_cost_integral / self.dt).clone(),
            collision=collision.clone(),
            progress=(self.root_states[:, 0] - self.env_origins[:, 0]).clone(),
        )


def load_frozen_model(checkpoint, bins, observation_dim, device, snn_steps=8):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    prefix = 'actor.terrain_encoder.'
    encoder_state = {key[len(prefix):]: value for key, value in
                     saved['primitive_selection_model_state_dict'].items()
                     if key.startswith(prefix)}
    encoder = TerrainScanEncoder((2, 16, 5), 32)
    encoder.load_state_dict(encoder_state, strict=True)
    model = TerrainHoldSNN(encoder, bins, observation_dim,
                           hidden_dims=(64,), snn_steps=snn_steps).to(device)
    return model


def build_env(checkpoint, terrain, difficulty, speed, initial_seed,
              disturbance_seed, start_step, push_every, seconds):
    cfg, train = load_run_config(MotorPrimitivesCfg(), MotorPrimitivesCfgPPO(), checkpoint.parent)
    archived = json.loads((checkpoint.parent / 'config.json').read_text())
    if 'pi_g_cot_reference' not in archived.get('rewards', {}):
        cfg.rewards.pi_g_cot_reference = 1.
    cfg.rewards.primitive_selection_scales.pi_g_context_guidance = 0.
    if terrain not in cfg.terrain.terrain_columns:
        raise ValueError(f'{terrain} is absent from the archived terrain layout')
    if cfg.control.primitive_selection_update_cycle != 15 or cfg.control.modulation_param_update_cycle != 3:
        raise ValueError('this fixed checkpoint must retain its archived 15/3-step policy cycles')
    cfg.env.num_envs = 1
    cfg.env.play = False
    cfg.env.episode_length_s = seconds
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = push_every > 0
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.selected = cfg.terrain.selected_kwargs = None
    cfg.terrain.terrain_columns = [terrain]
    cfg.terrain.row_difficulties = [difficulty]
    cfg.terrain.seed = initial_seed
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    SafetyEvaluationEnv.repeats = 1
    SafetyEvaluationEnv.test_speeds = [speed]
    SafetyEvaluationEnv.disturbance_seed = disturbance_seed
    SafetyEvaluationEnv.push_start_step = start_step
    set_seed(initial_seed)
    env = SafetyEvaluationEnv(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    if push_every:
        env.cfg.domain_rand.push_interval = push_every
    return env, train


def run_branch(args, terrain, difficulty, speed, initial_seed, disturbance_seed,
               start_step, horizon, branch='reference', hold_steps=None, model=None):
    env, train = build_env(args.checkpoint, terrain, difficulty, speed, initial_seed,
                           disturbance_seed, start_step, args.push_every, args.seconds)
    try:
        runner = MultiInteroceptivePolicyRunner(env, class_to_dict(train),
                                                log_dir=None, device=env.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        for policy in (runner.primitive_selection_actor_critic,
                       runner.modulation_param_actor_critic):
            policy.eval().requires_grad_(False)
        high, low = runner.get_inference_policy(device=env.device)
        set_seed(initial_seed)
        observation, _ = env.reset()
        env.metrics = TraversalMetrics(1, env.device,
                                       cfg_distance(env), args.corridor)
        for name in ('work_proxy', 'cot_integral', 'task_return', 'selection_return'):
            setattr(env, name, torch.zeros(1, device=env.device))
        controller = HeldPolicyController(runner, high, low)
        prefix, input_observation, quality = None, None, []
        terrain_calls = 0
        snn_clock = FixedTerrainClock(env.num_envs, args.snn_every, env.device) if model is not None else None
        reference_calls = dict(gait=0, modulation=0)
        encoder_calls = dict(gait=0, modulation=0, terrain=0)
        def count_encoder(name):
            def hook(_module, _inputs, _outputs):
                encoder_calls[name] += 1
            return hook
        hooks = [
            runner.primitive_selection_actor_critic.actor.terrain_encoder.register_forward_hook(
                count_encoder('gait')),
            runner.modulation_param_actor_critic.actor.terrain_encoder.register_forward_hook(
                count_encoder('modulation')),
        ]
        if model is not None:
            hooks.append(model.terrain_encoder.register_forward_hook(count_encoder('terrain')))
        predicted = torch.tensor([[15, 3]], device=env.device)
        with torch.inference_mode():
            for step in range(start_step + horizon):
                if step == start_step:
                    prefix = prefix_digest(env, runner, observation)
                    input_observation = observation[0, -1].detach().cpu().numpy().copy()
                if branch == 'scheduled':
                    snn_ids = snn_clock.due(step)
                    if snn_ids.numel():
                        predicted[snn_ids] = model.decode(observation[snn_ids, -1])
                        snn_clock.mark(snn_ids, step)
                        terrain_calls += 1
                gait_interval = int(predicted[0, 0]) if branch == 'scheduled' else 15
                modulation_interval = int(predicted[0, 1]) if branch == 'scheduled' else 3
                if step == start_step and branch == 'gait':
                    gait_interval = hold_steps
                if step == start_step and branch == 'modulation':
                    modulation_interval = hold_steps
                if branch == 'reference':
                    if step % 15 == 0:
                        reference_calls['gait'] += 1
                    if step % 3 == 0:
                        reference_calls['modulation'] += 1
                    observation, _, _, dones, _ = runner.inference_rollout(observation, high, low)
                else:
                    observation, _, _, dones, _ = controller.step(
                        observation, step, gait_interval, modulation_interval)
                if step >= start_step:
                    quality.append({key: value[0].item()
                                    for key, value in env.phase1_quality.items()})
                if snn_clock is not None and bool(dones.any()):
                    snn_clock.reset(torch.nonzero(dones).flatten())
                if bool(dones[0]):
                    break
        for hook in hooks:
            hook.remove()
        gait_calls = (reference_calls['gait'] if branch == 'reference'
                      else int(controller.clock.call_count['gait'][0]))
        modulation_calls = (reference_calls['modulation'] if branch == 'reference'
                            else int(controller.clock.call_count['modulation'][0]))
        if (encoder_calls['gait'] != gait_calls
                or encoder_calls['modulation'] != modulation_calls
                or encoder_calls['terrain'] != terrain_calls):
            raise RuntimeError('actual terrain encoder forward counts differ from actor/SNN call counts')
        return dict(prefix=prefix, observation=input_observation, quality=quality,
                    traversal=env.metrics.records()[0],
                    gait_calls=gait_calls, modulation_calls=modulation_calls,
                    terrain_calls=terrain_calls, encoder_calls=encoder_calls,
                    simulated_steps=step + 1,
                    dt=float(env.dt))
    finally:
        env.gym.destroy_sim(env.sim)


def cfg_distance(env):
    return env.cfg.terrain.terrain_length / 2 - env.cfg.terrain.boundary_margin


def collect(args):
    bins = HoldBins(tuple(args.gait_bins), tuple(args.modulation_bins))
    if any(step < 0 or step % 15 for step in args.start_steps):
        raise ValueError('intervention start steps must be nonnegative multiples of 15')
    if args.push_every < 0 or args.disturbance_repeats < 1:
        raise ValueError('invalid disturbance schedule')
    if args.seconds <= (max(args.start_steps) + max(bins.gait[-1], bins.modulation[-1])) * CONTROL_DT:
        raise ValueError('episode seconds must exceed the longest requested observation window')
    limits = dict(speed_error=args.max_speed_error_increase,
                  tilt=args.max_tilt_increase, slip=args.max_slip_increase,
                  collision=args.max_collision_increase,
                  progress=args.max_progress_loss)
    compare_quality(dict(failed=False, **{key: 0. for key in limits}),
                    dict(failed=False, **{key: 0. for key in limits}), limits)
    rows = []
    for terrain in args.terrains:
        for difficulty in args.difficulties:
            for speed in args.speeds:
                for initial_seed in args.seeds:
                    for start in args.start_steps:
                        replica_outcomes = {'gait': [], 'modulation': []}
                        first_reference = None
                        maximum = max(bins.gait[-1], bins.modulation[-1])
                        for replica in range(args.disturbance_repeats):
                            disturbance_seed = args.disturbance_seed + replica
                            reference = run_branch(args, terrain, difficulty, speed,
                                                   initial_seed, disturbance_seed,
                                                   start, maximum)
                            if reference['prefix'] is None:
                                if replica == 0:
                                    print(f'skipped {terrain} seed={initial_seed} step={start}: '
                                          'episode ended before intervention', flush=True)
                                    break
                                raise RuntimeError('disturbance replicas disagree on pre-intervention survival')
                            # A second identical replay detects nondeterministic PhysX or reset.
                            duplicate = run_branch(args, terrain, difficulty, speed,
                                                   initial_seed, disturbance_seed,
                                                   start, maximum)
                            if (reference['prefix'] is None
                                    or reference['prefix'] != duplicate['prefix']
                                    or reference['quality'] != duplicate['quality']):
                                differing = next((i for i, (left, right) in enumerate(zip(
                                    reference['quality'], duplicate['quality'])) if left != right), None)
                                raise RuntimeError(
                                    'reference replay is not reproducible; no labels written: '
                                    f'prefix_equal={reference["prefix"] == duplicate["prefix"]}, '
                                    f'quality_lengths={(len(reference["quality"]), len(duplicate["quality"]))}, '
                                    f'first_differing_step={differing}, '
                                    f'first_values={(reference["quality"][differing], duplicate["quality"][differing]) if differing is not None else None}')
                            if first_reference is None:
                                first_reference = reference
                            elif first_reference['prefix'] != reference['prefix']:
                                raise RuntimeError('disturbance replicas differ before intervention')
                            for name, choices in (('gait', bins.gait), ('modulation', bins.modulation)):
                                outcomes = []
                                for duration in choices:
                                    held = run_branch(args, terrain, difficulty, speed,
                                                      initial_seed, disturbance_seed, start,
                                                      duration, branch=name,
                                                      hold_steps=duration)
                                    outcomes.append(candidate_result(reference, held,
                                                                     duration, limits))
                                replica_outcomes[name].append(tuple(outcomes))
                        if first_reference is None:
                            continue
                        gait_out = conservative_outcomes(replica_outcomes['gait'])
                        mod_out = conservative_outcomes(replica_outcomes['modulation'])
                        gait_safe, gait_censored, gait_usable = verified_hold_label(bins.gait, gait_out)
                        mod_safe, mod_censored, mod_usable = verified_hold_label(
                            bins.modulation, mod_out)
                        rows.append(dict(observation=first_reference['observation'],
                                         terrain=terrain, difficulty=difficulty, speed=speed,
                                         initial_seed=initial_seed, start_step=start,
                                         prefix=first_reference['prefix'],
                                         gait_safe=gait_safe, modulation_safe=mod_safe,
                                         gait_censored=gait_censored,
                                         modulation_censored=mod_censored,
                                         gait_usable=gait_usable, modulation_usable=mod_usable,
                                         gait_outcomes=gait_out, modulation_outcomes=mod_out))
                        print(f'{terrain} difficulty={difficulty} speed={speed} '
                              f'seed={initial_seed} step={start}: '
                              f'gait={gait_safe} mod={mod_safe}', flush=True)
    if not rows:
        raise RuntimeError('no intervention state survived; no label file written')
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                    gait_bins=bins.gait, modulation_bins=bins.modulation,
                    dt_seconds=first_reference['dt'], unit='environment_control_steps',
                    safety_allowances=limits, disturbance_repeats=args.disturbance_repeats,
                    disturbance_seed=args.disturbance_seed, push_every=args.push_every,
                    pairing='exact replay prefix digest + duplicate reference trajectory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, metadata=json.dumps(metadata),
                        observation=np.stack([row['observation'] for row in rows]),
                        **{key: np.asarray([row[key] for row in rows]) for key in (
                            'terrain', 'difficulty', 'speed', 'initial_seed', 'start_step',
                            'prefix', 'gait_safe', 'modulation_safe', 'gait_censored',
                            'modulation_censored', 'gait_usable', 'modulation_usable')})
    args.output.with_suffix('.json').write_text(json.dumps([
        {k: v for k, v in row.items() if k != 'observation'} for row in rows], indent=2) + '\n')


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files if name != 'metadata'}
        metadata = json.loads(data['metadata'].item())
    return result, metadata


def prediction_report(model, observations, data, indices, device):
    model.eval()
    with torch.inference_mode():
        prediction = model.decode(torch.as_tensor(observations[indices],
                                                 dtype=torch.float32, device=device)).cpu()
    report = {}
    for terrain in sorted(set(data['terrain'][indices])):
        mask = np.asarray(data['terrain'][indices] == terrain)
        terrain_report = {}
        for column, key in enumerate(('gait', 'modulation')):
            usable = torch.as_tensor(data[f'{key}_usable'][indices][mask], dtype=torch.bool)
            predicted = prediction[mask, column][usable]
            verified = torch.as_tensor(data[f'{key}_safe'][indices][mask], dtype=torch.long)[usable]
            censored = torch.as_tensor(data[f'{key}_censored'][indices][mask], dtype=torch.bool)[usable]
            over, known, unknown = overestimation_counts(predicted, verified, censored)
            exact = ~censored
            terrain_report[key] = dict(samples=len(predicted), known=known,
                                       uncensored=int(exact.sum()),
                                       unknown_censored=unknown,
                                       overestimation_rate=(over / known if known else None),
                                       mean_absolute_error_steps=(float((predicted[exact] - verified[exact]).abs().float().mean())
                                                                  if exact.any() else None),
                                       verified_conservative_fraction=(
                                           float((predicted <= verified).float().mean())
                                           if len(predicted) else None))
        report[str(terrain)] = terrain_report
    return report


def train(args):
    data, metadata = load_dataset(args.dataset)
    if metadata['unit'] != 'environment_control_steps':
        raise ValueError('dataset deadlines are not environment control steps')
    if args.snn_every * metadata['dt_seconds'] <= 1.:
        raise ValueError('fixed Terrain SNN interval must remain below 1 Hz')
    if file_sha256(args.checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('checkpoint hash does not match paired data')
    seeds = sorted(set(data['initial_seed'].tolist()))
    if len(seeds) < 2:
        raise ValueError('at least two initial seeds are required for held-out evaluation')
    train_ids = np.flatnonzero(data['initial_seed'] != seeds[-1])
    val_ids = np.flatnonzero(data['initial_seed'] == seeds[-1])
    bins = HoldBins(tuple(metadata['gait_bins']), tuple(metadata['modulation_bins']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    model = load_frozen_model(args.checkpoint, bins, data['observation'].shape[-1],
                              device, args.snn_steps)
    optimizer = torch.optim.Adam(model.snn.parameters(), lr=args.learning_rate)
    obs = torch.as_tensor(data['observation'], dtype=torch.float32, device=device)
    targets = [torch.as_tensor([list(values).index(int(x)) for x in data[f'{key}_safe']],
                               dtype=torch.long, device=device)
               for key, values in (('gait', bins.gait), ('modulation', bins.modulation))]
    masks = [torch.as_tensor(data[f'{key}_usable'], dtype=torch.bool, device=device)
             for key in ('gait', 'modulation')]
    if not any(mask[train_ids].any() for mask in masks):
        raise ValueError('training split has no usable verified labels')
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(train_ids)
        total = 0.
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / args.batch_size)))):
            logits = model(obs[batch])
            loss_terms = []
            for head, target, mask in zip(logits, targets, masks):
                valid = mask[batch]
                if not valid.any():
                    continue
                head, target = head[valid], target[batch][valid]
                above = (torch.arange(head.shape[-1], device=device)[None, :] > target[:, None])
                loss_terms.append(F.cross_entropy(head, target)
                                  + args.overestimate_penalty
                                  * (head.softmax(-1) * above).sum(-1).mean())
            loss = sum(loss_terms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f'epoch={epoch + 1} loss_sum={total:.5f}', flush=True)
    report = prediction_report(model, data['observation'], data, val_ids, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(), checkpoint_sha256=metadata['checkpoint_sha256'],
                    gait_bins=bins.gait, modulation_bins=bins.modulation,
                    observation_dim=obs.shape[-1], snn_steps=args.snn_steps,
                    snn_every=args.snn_every), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def evaluate(args):
    saved = torch.load(args.scheduler, map_location='cpu', weights_only=True)
    if file_sha256(args.checkpoint) != saved['checkpoint_sha256']:
        raise ValueError('scheduler was trained against a different control checkpoint')
    bins = HoldBins(tuple(saved['gait_bins']), tuple(saved['modulation_bins']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_frozen_model(args.checkpoint, bins, saved['observation_dim'],
                              device, saved['snn_steps'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    args.snn_every = saved['snn_every']
    records = []
    for terrain in args.terrains:
        for difficulty in args.difficulties:
            for speed in args.speeds:
                for seed in args.seeds:
                    steps = round(args.seconds / CONTROL_DT)
                    reference = run_branch(args, terrain, difficulty, speed, seed,
                                           args.disturbance_seed, 0, steps)
                    scheduled = run_branch(args, terrain, difficulty, speed, seed,
                                           args.disturbance_seed, 0, steps,
                                           branch='scheduled', model=model)
                    if reference['prefix'] != scheduled['prefix']:
                        raise RuntimeError('online comparison initial states differ')
                    records.append(dict(terrain=terrain, difficulty=difficulty, speed=speed,
                                        seed=seed, reference={k: reference[k] for k in (
                                            'traversal', 'gait_calls', 'modulation_calls',
                                            'encoder_calls', 'simulated_steps')},
                                        scheduled={k: scheduled[k] for k in (
                                            'traversal', 'gait_calls', 'modulation_calls',
                                            'terrain_calls', 'encoder_calls',
                                            'simulated_steps')}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    print(f'wrote {len(records)} paired online comparisons to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--terrains', nargs='+', default=['flat', 'stairs', 'hurdle', 'gap_single'])
    common.add_argument('--difficulties', type=float, nargs='+', default=[.3, .8])
    common.add_argument('--speeds', type=float, nargs='+', default=[.4, 1.5])
    common.add_argument('--seeds', type=int, nargs='+', default=[321, 322])
    common.add_argument('--seconds', type=float, default=12.)
    common.add_argument('--disturbance-seed', type=int, default=8001)
    common.add_argument('--push-every', type=int, default=0,
                        help='0 disables pushes; positive control-step interval uses deterministic pushes')
    common.add_argument('--corridor', type=float, default=1.5)
    common.add_argument('--output', type=Path, required=True)
    collect_parser = sub.add_parser('collect', parents=[common])
    collect_parser.add_argument('--start-steps', type=int, nargs='+', default=[0, 60])
    collect_parser.add_argument('--gait-bins', type=int, nargs='+', default=[15, 30, 45, 60])
    collect_parser.add_argument('--modulation-bins', type=int, nargs='+', default=[3, 6, 9, 15])
    collect_parser.add_argument('--disturbance-repeats', type=int, default=1)
    for flag in ('max-speed-error-increase', 'max-tilt-increase', 'max-slip-increase',
                 'max-collision-increase', 'max-progress-loss'):
        collect_parser.add_argument('--' + flag, type=float, required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--dataset', type=Path, required=True)
    train_parser.add_argument('--output', type=Path, required=True)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--learning-rate', type=float, default=1e-3)
    train_parser.add_argument('--overestimate-penalty', type=float, default=2.)
    train_parser.add_argument('--snn-steps', type=int, default=8)
    train_parser.add_argument('--snn-every', type=int, default=60)
    train_parser.add_argument('--seed', type=int, default=1)
    sub.add_parser('evaluate', parents=[common]).add_argument('--scheduler', type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error('checkpoint does not exist')
    if file_sha256(args.checkpoint) != FIXED_CHECKPOINT_SHA256:
        parser.error('Phase 1 is pinned to model_1001_10000.pt from the specified run')
    if args.command in ('collect', 'evaluate') and (
            not args.terrains or not args.difficulties or not args.speeds or not args.seeds
            or args.seconds <= 0 or args.corridor <= 0
            or min(args.difficulties) < 0 or max(args.difficulties) > 1
            or min(args.speeds) <= 0 or min(args.seeds) < 0
            or args.disturbance_seed < 0):
        parser.error('invalid evaluation grid')
    if args.command == 'train' and (args.epochs < 1 or args.batch_size < 1
                                    or args.learning_rate <= 0 or args.snn_every < 1
                                    or args.overestimate_penalty < 0):
        parser.error('invalid training parameters or SNN interval (must be below 1 Hz)')
    {'collect': collect, 'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
