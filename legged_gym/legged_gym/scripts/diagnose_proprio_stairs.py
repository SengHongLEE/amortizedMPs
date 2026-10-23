"""Replay a saved proprio policy on controlled ascending stairs; preserve terminal samples."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'legged_gym/legged_gym/test/torch_extensions'))
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply
import numpy as np
import torch
from legged_gym.envs.a1.mp_proprio_terrain import MotorPrimitives
from legged_gym.envs.a1.mp_proprio_terrain_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.terrain_evaluation import stair_evaluation_rows
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner
from contact_phase_gate import ContactPhaseGate
from horizontal_obstacle_probe import HorizontalObstacleProbe


class ProbeEnv(MotorPrimitives):
    """Same physics/controller, with fixed commands and quiet playback-quality calculation."""
    def reset_idx(self, env_ids, completed_episode=False):
        super().reset_idx(env_ids, completed_episode=completed_episode)
        if hasattr(self, 'probe_ff_valid'):
            self.probe_ff_valid[env_ids] = False

    def _compute_torques(self, actions):
        if hasattr(self.ID, 'horizontal_obstacle_probe'):
            self.ID.horizontal_obstacle_probe.set_forces(self.contact_forces[:, self.feet_indices])
        baseline = super()._compute_torques(actions)
        if self.velocity_feedforward == 0:
            return baseline
        if not hasattr(self, 'probe_ff_previous'):
            self.probe_ff_previous = self.dof_des_pos.clone()
            self.probe_ff_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reference_velocity = (self.dof_des_pos - self.probe_ff_previous) / self.sim_params.dt
        reference_velocity = reference_velocity.clamp(-self.dof_vel_limits, self.dof_vel_limits)
        reference_velocity *= self.probe_ff_valid[:, None]
        self.probe_ff_previous.copy_(self.dof_des_pos)
        self.probe_ff_valid[:] = True
        # Diagnostic-only ablation of the PD reference velocity. The policy
        # still emits dynamical parameters; no motor action is introduced.
        torque = (self.p_gains * (self.dof_des_pos - self.dof_pos)
                  + self.d_gains * (self.velocity_feedforward * reference_velocity - self.dof_vel))
        return torque.clamp(-self.torque_limits, self.torque_limits)

    def _get_env_origins(self):
        super()._get_env_origins()
        self.terrain_levels[:] = torch.arange(self.num_envs, device=self.device) // self.repeats
        self.terrain_types[:] = torch.arange(self.num_envs, device=self.device) % self.repeats
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.
        self.commands[env_ids, 0] = self.test_speed

    def compute_reward(self, env_ids=None):
        result = super().compute_reward(env_ids)
        track = sum(getattr(self, '_reward_' + name)() * self.reward_scales[name]
                    for name in self.cfg.rewards.task_rewards)
        maximum = sum(self.reward_scales[name] for name in self.cfg.rewards.task_rewards)
        self.behavior_quality[:] = (track / (maximum + 1e-6)).clamp(0, 1) - self.projected_gravity[:, :2].square().sum(-1) + .2
        return result

    def step(self, actions):
        if self.force_gait is not None or self.disable_equilibrium:
            actions = actions.clone()
            if self.force_gait is not None:
                actions[:, 0] = self.force_gait
            if self.disable_equilibrium:
                actions[:, 5:17] = 0.
        self.probe_sat = torch.zeros(self.num_envs, device=self.device)
        self.probe_ref_speed = torch.zeros_like(self.probe_sat)
        self.probe_joint_error = torch.zeros_like(self.probe_sat)
        self.probe_projection = torch.zeros_like(self.probe_sat)
        self.probe_horizontal = torch.zeros(self.num_envs, 4, device=self.device)
        self.probe_missed = torch.zeros_like(self.probe_horizontal)
        self.probe_qobstacle = torch.zeros_like(self.probe_horizontal)
        return super().step(actions)

    def post_decimation_step(self, dec_i):
        super().post_decimation_step(dec_i)
        self.probe_sat += (self.torques.abs() >= .99 * self.torque_limits).float().mean(-1) / (self.cfg.control.decimation * self.cfg.control.cycle)
        forces = self.contact_forces[:, self.feet_indices]
        horizontal = torch.linalg.vector_norm(forces[:, :, :2], dim=-1)
        self.probe_horizontal = torch.maximum(self.probe_horizontal, horizontal)
        # Candidate missed collisions, not a certified wall-contact classifier.
        missed = (horizontal > 10.) & (forces[:, :, 2] <= 1.)
        self.probe_missed += missed.float() / (self.cfg.control.decimation * self.cfg.control.cycle)
        self.probe_qobstacle = torch.maximum(self.probe_qobstacle, self.ID.q_obstacle)
        count = self.cfg.control.decimation * self.cfg.control.cycle
        if getattr(self, 'probe_prev_target', None) is not None:
            ref_speed = (self.dof_des_pos - self.probe_prev_target).abs() / self.sim_params.dt
            self.probe_ref_speed += (ref_speed > self.dof_vel_limits).float().mean(-1) / count
        self.probe_prev_target = self.dof_des_pos.clone()
        self.probe_joint_error += (self.dof_des_pos - self.dof_pos).abs().mean(-1) / count
        if hasattr(self, 'foot_targets_requested'):
            self.probe_projection += torch.linalg.vector_norm(
                self.foot_targets_requested - self.foot_targets_reachable, dim=-1).mean(-1) / count

    def check_termination(self, env_ids=None):
        super().check_termination(env_ids)
        feet = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :3]
        q = self.dof_des_pos.reshape(self.num_envs, 4, 3)
        hip, thigh, knee = q.unbind(-1)
        side = q.new_tensor([1., -1., 1., -1.])
        zp = -.2 * (thigh.cos() + (thigh + knee).cos())
        target = torch.stack((-.2 * (thigh.sin() + (thigh + knee).sin()),
                              side * .0838 * hip.cos() - zp * hip.sin(),
                              side * .0838 * hip.sin() + zp * hip.cos()), dim=-1)
        target += q.new_tensor([[.183, .047, 0], [.183, -.047, 0], [-.183, .047, 0], [-.183, -.047, 0]])
        target = quat_apply(self.base_quat[:, None].expand(-1, 4, -1).reshape(-1, 4), target.reshape(-1, 3)).reshape(-1, 4, 3) + self.root_states[:, None, :3]
        self.probe_sample = {name: value.detach().cpu().numpy().copy() for name, value in dict(
            root=self.root_states, feet=feet, target=target, q=self.dof_pos, q_target=self.dof_des_pos,
            dq=self.dof_vel, torque=self.torques, force=self.contact_forces[:, self.feet_indices],
            actions=self.actions, phase=self.ID.X[:, 2], q_obstacle=self.ID.q_obstacle,
            q_obstacle_peak=self.probe_qobstacle, saturation=self.probe_sat,
            horizontal_peak=self.probe_horizontal, missed_candidate=self.probe_missed,
            failed=self.terrain_failure_buf, timeout=self.time_out_buf,
            episode_step=self.episode_length_buf, quality=self.behavior_quality,
            reference_speed_excess=self.probe_ref_speed,
            mean_joint_error=self.probe_joint_error, mean_projection=self.probe_projection,
            offset=self.ID.offset, offset_target=self.ID.offset_target,
            offset_velocity=self.ID.offset_velocity, amplitude=self.ID.X[:, :2],
            vertical_phase=self.ID._vertical_phase(),
            frequency_hz=self.ID.omega, q_late=self.ID.q_late,
            stance_extension=self.ID.stance_extension,
            duty_factor=self.ID.selected_duty_factors(self.ID._gait_ids)).items()}
        if hasattr(self.ID, 'contact_phase_gate'):
            gate = self.ID.contact_phase_gate
            for name, value in [('gate_waiting', gate.waiting), ('gate_elapsed', gate.elapsed),
                                ('gate_support', gate.support)]:
                self.probe_sample[name] = value.detach().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['play', 'scheduled'], default='play')
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--heights', nargs='+', type=float, default=[0., .02, .04, .06, .08],
                        help='Increasing step heights in metres; optional initial zero adds flat ground.')
    parser.add_argument('--vertical-stairs', action='store_true',
                        help='Use true risers at every height, overriding legacy source geometry.')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--frequency-scale', type=float, default=1., help='Diagnostic-only gait clock multiplier.')
    parser.add_argument('--force-gait', type=int, choices=range(6))
    parser.add_argument('--disable-equilibrium', action='store_true', help='Diagnostic-only zero equilibrium targets.')
    parser.add_argument('--joint-velocity-feedforward', type=float, default=0.,
                        help='Diagnostic-only fraction of bounded reference velocity in joint PD, 0..1.')
    parser.add_argument('--contact-wait', type=float, default=0.,
                        help='Diagnostic-only maximum wait at mid-stance for contact, seconds.')
    parser.add_argument('--horizontal-obstacle', action='store_true',
                        help='Diagnostic-only horizontal riser collision input to the swing obstacle reflex.')
    parser.add_argument('--stance-extension', type=float,
                        help='Diagnostic-only maximum missing-contact search depth in metres (0..0.08).')
    args = parser.parse_args()
    if args.frequency_scale <= 0:
        parser.error('--frequency-scale must be positive')
    if not 0 <= args.joint_velocity_feedforward <= 1:
        parser.error('--joint-velocity-feedforward must be in [0,1]')
    if not 0 <= args.contact_wait <= .2 or 0 < args.contact_wait < .025:
        parser.error('--contact-wait must be zero or in [.025,.2] seconds')
    if args.stance_extension is not None and not 0 <= args.stance_extension <= .08:
        parser.error('--stance-extension must be finite and in [0,.08] metres')
    if args.checkpoint is None:
        configs = list((ROOT / 'legged_gym/logs/A1').glob('*proprio*/config.json'))
        run = max(configs, key=lambda p: p.stat().st_mtime).parent
        checkpoint = max(run.glob('model*.pt'), key=lambda p: p.stat().st_mtime)
    else:
        checkpoint = args.checkpoint.resolve()
        run = checkpoint.parent
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, train = load_run_config(MotorPrimitivesCfg(), MotorPrimitivesCfgPPO(), run)
    if args.vertical_stairs:
        cfg.terrain.terrain_params.setdefault('stairs', {})['mesh_vertical_faces'] = True
    try:
        height_range, difficulties = stair_evaluation_rows(
            args.heights, cfg.terrain.vertical_scale, allow_flat=True)
    except ValueError as error:
        parser.error(str(error))
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    print('CHECKPOINT', checkpoint, checkpoint_hash, flush=True)
    set_seed(args.seed)
    cfg.env.num_envs = len(args.heights) * args.repeats
    cfg.env.play = False  # avoid playback console/array logging; compute its quality gate above
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
    cfg.terrain.num_rows, cfg.terrain.num_cols = len(args.heights), args.repeats
    cfg.terrain.selected = 'stairs'
    cfg.terrain.selected_kwargs = None
    cfg.terrain.terrain_columns = None
    cfg.terrain.row_difficulties = difficulties
    cfg.terrain.terrain_params['stairs'].update(step_height_range=height_range, step_signs=[-1])
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    ProbeEnv.repeats, ProbeEnv.test_speed = args.repeats, args.speed
    ProbeEnv.force_gait, ProbeEnv.disable_equilibrium = args.force_gait, args.disable_equilibrium
    ProbeEnv.velocity_feedforward = args.joint_velocity_feedforward
    env = ProbeEnv(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    env.ID._primitive_frequencies *= args.frequency_scale
    if args.stance_extension is not None:
        env.ID.max_stance_extension = args.stance_extension
    if args.contact_wait:
        ContactPhaseGate(env.ID, args.contact_wait)
    if args.horizontal_obstacle:
        HorizontalObstacleProbe(env.ID)
    writer = None
    try:
        runner = MultiInteroceptivePolicyRunner(env, class_to_dict(train), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_optimizer=False)
        high, low = runner.get_inference_policy(device=env.device)
        # Network construction consumes RNG differently across action versions.
        set_seed(args.seed)
        obs, _ = env.reset()
        initial_root = env.root_states.detach().cpu().numpy().copy()
        initial_q = env.dof_pos.detach().cpu().numpy().copy()
        initial_state_hash = hashlib.sha256(initial_root.tobytes() + initial_q.tobytes()).hexdigest()
        origins = env.env_origins.cpu().numpy().copy()
        np.savez_compressed(args.output / 'initial_state.npz', root=initial_root, q=initial_q, origins=origins)
        first_edges = []
        for row in range(len(args.heights)):
            raw = env.terrain.heightsamples
            ix = env.terrain.border + row * env.terrain.length_per_env_pixels
            iy = env.terrain.border + env.terrain.width_per_env_pixels // 2
            profile = raw[ix:ix + env.terrain.length_per_env_pixels, iy].astype(float) * cfg.terrain.vertical_scale
            center = len(profile) // 2
            edges = np.flatnonzero(profile[center:] > profile[center] + .001)
            first_edges.append(float(edges[0] * cfg.terrain.horizontal_scale) if len(edges) else None)
        camera = None
        if args.video:
            props = gymapi.CameraProperties(); props.width = 960; props.height = 540
            video_env = (len(args.heights) - 1) * args.repeats
            camera = env.gym.create_camera_sensor(env.envs[video_env], props)
            if camera < 0:
                raise RuntimeError('Camera creation failed')
            writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                       '-s', '960x540', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                       str(args.output / f'stairs_{100 * args.heights[-1]:g}cm.mp4')], stdin=subprocess.PIPE)
        trace = {}
        wall_start = time.time()
        with open(os.devnull, 'w') as quiet, torch.inference_mode():
            for step in range(round(args.seconds / env.dt)):
                if args.mode == 'scheduled':
                    # Same deterministic actors/update periods, without playback-only quality freezing.
                    env.behavior_quality.zero_()
                with contextlib.redirect_stdout(quiet):
                    obs, _, _, _, _ = runner.inference_rollout(obs, high, low)
                for name, value in env.probe_sample.items():
                    trace.setdefault(name, []).append(value)
                if camera is not None and step % 5 == 0:
                    p = env.root_states[video_env, :3].cpu().numpy()
                    env.gym.set_camera_location(camera, env.envs[video_env], gymapi.Vec3(float(p[0] + .9), float(p[1] - 2.6), float(p[2] + .7)),
                                                gymapi.Vec3(float(p[0] + .2), float(p[1]), float(p[2] - .1)))
                    env.gym.fetch_results(env.sim, True)
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)
                    rgba = env.gym.get_camera_image(env.sim, env.envs[video_env], camera, gymapi.IMAGE_COLOR).reshape(540, 960, 4)
                    from PIL import Image, ImageDraw
                    frame = Image.fromarray(rgba[:, :, :3])
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 32), fill='white')
                    draw.text((8, 8), f'{checkpoint.name} | {100 * args.heights[-1]:g} cm ascending stairs | {args.mode} | t={step * env.dt:.2f} s', fill='black')
                    writer.stdin.write(np.asarray(frame).tobytes())
                if step % 100 == 0:
                    print(f'{args.mode}: {step * env.dt:.1f}/{args.seconds}s simulated; {time.time()-wall_start:.1f}s wall', flush=True)
        arrays = {k: np.stack(v) for k, v in trace.items()}
        np.savez_compressed(args.output / 'trace.npz', **arrays, origins=origins, dt=env.dt)
        rows = []
        for row, height in enumerate(args.heights):
            results = []
            for i in range(row * args.repeats, (row + 1) * args.repeats):
                # Each seed contributes only its first episode; later resets cannot inflate success.
                end_indices = np.flatnonzero(arrays['failed'][:, i] | arrays['timeout'][:, i])
                end = int(end_indices[0] + 1) if len(end_indices) else len(arrays['root'])
                x = arrays['root'][:end, i, 0] - origins[i, 0]
                feet = arrays['feet'][:end, i]
                first = first_edges[row]
                def supported(leg_indices):
                    ok = (feet[:, leg_indices, 0] > origins[i, 0] + first + .03) & (feet[:, leg_indices, 2] > origins[i, 2] + height - .02) & (arrays['force'][:end, i, leg_indices, 2] > 5)
                    return bool(ok.any())
                results.append(dict(env=i, duration_s=end * env.dt, max_forward_m=float(x.max()),
                    final_forward_m=float(x[-1]), max_body_rise_m=float((arrays['root'][:end, i, 2] - arrays['root'][0, i, 2]).max()),
                    failed=bool(arrays['failed'][end-1, i]), crossed=bool((x > cfg.terrain.terrain_length/2 - cfg.terrain.boundary_margin).any()),
                    front_supported_on_first_step=None if first is None else supported([0, 1]),
                    rear_supported_on_first_step=None if first is None else supported([2, 3]),
                    torque_saturation_fraction=float(arrays['saturation'][:end, i].mean()),
                    foot_z_rmse_m=float(np.sqrt(np.square(arrays['feet'][:end, i, :, 2] - arrays['target'][:end, i, :, 2]).mean())),
                    equilibrium_clip_fraction=float((np.abs(arrays['actions'][:end, i, 5:17]) > .999).mean()),
                    reference_speed_excess_fraction=float(arrays['reference_speed_excess'][:end, i].mean()),
                    mean_joint_error_rad=float(arrays['mean_joint_error'][:end, i].mean()),
                    mean_projection_m=float(arrays['mean_projection'][:end, i].mean()),
                    max_lateral_m=float(np.abs(arrays['root'][:end, i, 1] - origins[i, 1]).max()),
                    mean_frequency_hz=float(arrays['frequency_hz'][:end, i].mean()),
                    mean_duty_factor=float(arrays['duty_factor'][:end, i].mean()),
                    selected_gait_counts=np.bincount(arrays['actions'][:end, i, 0].astype(int), minlength=cfg.env.num_primitives).tolist()))
            rows.append(dict(height_m=height, first_riser_x_m=first_edges[row], episodes=results))
        report = dict(checkpoint=str(checkpoint), sha256=checkpoint_hash, iteration=runner.current_learning_iteration,
                      mode=args.mode, seed=args.seed, seconds=args.seconds, repeats=args.repeats, command_speed=args.speed,
                      frequency_scale=args.frequency_scale, force_gait=args.force_gait,
                      disable_equilibrium=args.disable_equilibrium,
                      joint_velocity_feedforward=args.joint_velocity_feedforward,
                      contact_wait=args.contact_wait,
                      horizontal_obstacle=args.horizontal_obstacle,
                      stance_extension_override=args.stance_extension,
                      effective_max_stance_extension=env.ID.max_stance_extension,
                      requested_heights=args.heights,
                      initial_state_sha256=initial_state_hash,
                      gait_names=env.ID.primitive_names, effective_config=class_to_dict(cfg), rows=rows,
                      notes='First episodes only; no push/noise/mass/friction randomization; reseeded after network construction. Camera follows first robot on the highest step height. Supported means any foot with Fz>5 N beyond first riser; not sustained stable support. crossed is raw forward crossing; also inspect failed and max_lateral_m, or use the stricter multi-terrain evaluator for acceptance. Uses current controller source with saved network/config.')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('CHECKPOINT', checkpoint, 'iteration', runner.current_learning_iteration)
        for row in rows:
            es = row['episodes']
            print(f"height={row['height_m']:.3f} max_x={np.mean([e['max_forward_m'] for e in es]):.3f} falls={sum(e['failed'] for e in es)}/{len(es)} crossed={sum(e['crossed'] for e in es)} front={sum(bool(e['front_supported_on_first_step']) for e in es)} rear={sum(bool(e['rear_supported_on_first_step']) for e in es)}")
    finally:
        if writer is not None:
            writer.stdin.close(); writer.wait()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
