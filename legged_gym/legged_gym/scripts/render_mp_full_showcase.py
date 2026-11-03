"""Replay one archived MP_full policy on a continuous presentation course.

The only change to the saved task configuration is the evaluation course and
deterministic playback settings. The policy and intrinsic-dynamics code are
loaded normally. Frames and traces come from the same uncut episode.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import isaacgym  # Must precede torch.
from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.load_config import prepare_resume_configs
from legged_gym.utils.terrains import blind_terrain


RUN_NAME = '1028_00-13-26_A1MPfull-signed-dense-v1'
ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'legged_gym/logs/A1' / RUN_NAME
DEFAULT_OUTPUT = ROOT / 'docs/terrain/mp_full_group_showcase_2026-10-28'
COURSE_LENGTH = 16.0
COURSE_WIDTH = 4.0
FINISH_X = 12.5
LEG_NAMES = ('FL', 'FR', 'RL', 'RR')
RAMP_GRADE = .10
HURDLE_HEIGHT = .07
GAP_WIDTH = .20


def segments():
    return (
        ('Flat', 0.0, 2.4),
        ('Uphill ramp', 2.4, 3.4),
        ('Plateau', 3.4, 4.4),
        ('Downhill ramp', 4.4, 5.4),
        ('Flat approach', 5.4, 7.2),
        ('Hurdle', 7.2, 7.4),
        ('Flat approach', 7.4, 10.5),
        ('Gap', 10.5, 10.5 + GAP_WIDTH),
        ('Landing', 10.5 + GAP_WIDTH, FINISH_X),
    )


def configure_course(max_difficulty):
    global RAMP_GRADE, HURDLE_HEIGHT, GAP_WIDTH
    if max_difficulty:
        # Upper limits in the archived run's ramp/hurdle/gap terrain_params.
        RAMP_GRADE, HURDLE_HEIGHT, GAP_WIDTH = .30, .15, .40
    else:
        RAMP_GRADE, HURDLE_HEIGHT, GAP_WIDTH = .10, .07, .20


def course_height(x):
    """Height in metres at the centre of each longitudinal terrain cell."""
    x = np.asarray(x)
    h = np.zeros_like(x, dtype=np.float64)
    h = np.where((x >= 2.4) & (x < 3.4), RAMP_GRADE * (x - 2.4), h)
    h = np.where((x >= 3.4) & (x < 4.4), RAMP_GRADE, h)
    h = np.where((x >= 4.4) & (x < 5.4), RAMP_GRADE * (5.4 - x), h)
    h = np.where((x >= 7.2) & (x < 7.4), HURDLE_HEIGHT, h)
    h = np.where((x >= 10.5) & (x < 10.5 + GAP_WIDTH), -.6, h)
    return h


def patch_course_generator():
    original_generate = blind_terrain._generate
    original_init = blind_terrain.BlindTerrain.__init__

    def generate(terrain, name, parameters):
        if name != 'flat':
            return original_generate(terrain, name, parameters)
        x = np.arange(terrain.width) * terrain.horizontal_scale
        height = np.rint(course_height(x) / terrain.vertical_scale).astype(np.int16)
        terrain.height_field_raw[:] = height[:, None]

    def init(self, cfg, num_robots):
        original_init(self, cfg, num_robots)
        self.env_origins[0, 0] = [1., COURSE_WIDTH / 2., 0.]

    blind_terrain._generate = generate
    blind_terrain.BlindTerrain.__init__ = init


def snapshot(env, t):
    controller = env.ID
    values = {
        'time_s': np.float32(t),
        'base_pos_m': env.root_states[0, :3],
        'base_vel_mps': env.root_states[0, 7:10],
        'body_vel_mps': env.base_lin_vel[0],
        'action': env.actions[0],
        'gait_id': controller._gait_ids[0],
        'amplitude_x_state': controller.X[0, 0],
        'amplitude_z_state': controller.X[0, 1],
        'phase_rad': controller.X[0, 2],
        'frequency_hz': controller.omega[0],
        'nominal_frequency_hz': controller.f_gait[0],
        'frequency_scale': controller.frequency_scale[0],
        'duty_factor': controller.selected_duty_factors(controller._gait_ids)[0],
        'phase_correction_radps': controller.phase_correction[0],
        'offset_m': controller.offset[0],
        'offset_target_m': controller.offset_target[0],
        'foot_contact': controller._contact[0],
        'contact_force_n': torch.linalg.vector_norm(env.contact_forces[0, env.feet_indices], dim=-1),
        'foot_target_m': getattr(env, 'foot_targets_requested', env.root_states.new_zeros(1, 4, 3))[0],
        'foot_world_m': env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)[0, env.feet_indices, :3],
        'joint_pos_rad': env.dof_pos[0],
        'joint_torque_nm': env.torques[0],
    }
    return {key: value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.array(value)
            for key, value in values.items()}


def plot_results(output, trace, profile_x, profile_h, terminal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})
    t = trace['time_s']
    x = trace['base_pos_m'][:, 0]
    colors = ('#1774b6', '#d88611', '#17804e', '#a33b76')

    def bands(ax, axis='time'):
        for name, left, right in segments():
            if name not in ('Uphill ramp', 'Downhill ramp', 'Hurdle', 'Gap'):
                continue
            if axis == 'time':
                if x.max() < left:
                    continue
                limits = np.interp([left, right], x, t)
            else:
                limits = left, right
            ax.axvspan(*limits, color='#c9d6e6' if name != 'Gap' else '#f0cfb2', alpha=.34)

    fig, axs = plt.subplots(3, 1, figsize=(16, 9), constrained_layout=True,
                            gridspec_kw={'height_ratios': [2.0, 1.2, 1.0]})
    axs[0].plot(profile_x, profile_h, color='#475569', lw=2, label='Terrain height')
    axs[0].plot(x, trace['base_pos_m'][:, 2], color='#0b77bd', lw=1.6, label='Base height')
    axs[0].scatter([x[-1]], [trace['base_pos_m'][-1, 2]], color='#b91c1c', s=32, zorder=5)
    axs[0].set_xlim(0, 13.3)
    axs[0].set_ylim(-.67, .65)
    axs[0].set_ylabel('Height [m]')
    axs[0].set_xlabel('Distance along course [m]')
    axs[0].legend(loc='upper right', ncol=2)
    for name, left, right in segments():
        if name in ('Uphill ramp', 'Downhill ramp', 'Hurdle', 'Gap'):
            axs[0].annotate(name, ((left + right) / 2, course_height((left + right) / 2)),
                            xytext=(0, 16 if name != 'Gap' else 35), textcoords='offset points',
                            ha='center', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#666'})
    axs[1].plot(t, trace['body_vel_mps'][:, 0], color='#0b77bd', label='Body forward speed')
    command_speed = terminal.get('command_forward_mps', 1.)
    axs[1].plot(t, trace['action'][:, 0] * 0 + command_speed, color='#999', ls='--',
                label=f'Command: {command_speed:.1f} m/s')
    bands(axs[1]); axs[1].set_ylabel('Forward speed [m/s]'); axs[1].legend(loc='upper right')
    axs[2].step(t, trace['gait_id'], where='post', color='#254b88')
    bands(axs[2]); axs[2].set_yticks(range(4), ('walk', 'trot', 'canter', 'bound'))
    axs[2].set_xlabel('Time [s]'); axs[2].set_ylabel('Selected gait')
    fig.suptitle('Single uninterrupted MP_full A1 rollout  |  ' + terminal['outcome'])
    fig.savefig(output / '01_course_overview.png'); plt.close(fig)

    fig, axs = plt.subplots(5, 1, figsize=(16, 12), constrained_layout=True, sharex=True)
    relative_phases = np.angle(np.exp(1j * (
        trace['phase_rad'] - trace['phase_rad'][:, :1]))) / np.pi
    for i, name in enumerate(LEG_NAMES):
        axs[0].plot(t, trace['amplitude_x_state'][:, i], color=colors[i], label=name, lw=1)
        axs[1].plot(t, trace['amplitude_z_state'][:, i], color=colors[i], lw=1)
        axs[3].plot(t, 100 * trace['offset_m'][:, i, 0], color=colors[i], lw=1)
    phase_image = axs[2].imshow(relative_phases.T, origin='lower', aspect='auto',
                                extent=(float(t[0]), float(t[-1]), -.5, 3.5),
                                cmap='twilight_shifted', vmin=-1., vmax=1., interpolation='nearest')
    axs[2].set_yticks(range(4), LEG_NAMES)
    fig.colorbar(phase_image, ax=axs[2], pad=.01, label='Relative phase [π rad]')
    axs[0].set_ylabel('Horizontal oscillator\namplitude [state units]')
    axs[1].set_ylabel('Lift oscillator\namplitude [state units]')
    axs[2].set_ylabel('Phase vs FL')
    axs[3].set_ylabel('X equilibrium offset [cm]')
    axs[4].plot(t, trace['frequency_hz'].mean(axis=1), color='#254b88', label='Actual mean leg frequency')
    axs[4].plot(t, trace['nominal_frequency_hz'][:, 0], color='#dc7c13', ls='--', label='Nominal frequency')
    axs[4].set_ylabel('Frequency [Hz]'); axs[4].set_xlabel('Time [s]')
    axs[0].legend(ncol=4, loc='upper right'); axs[4].legend(loc='upper right')
    for ax in axs: bands(ax)
    fig.suptitle('Intrinsic dynamics across the course  |  shaded bands: terrain events')
    fig.savefig(output / '02_intrinsic_dynamics.png'); plt.close(fig)

    fig, axs = plt.subplots(5, 1, figsize=(16, 12), constrained_layout=True, sharex=True)
    for i, name in enumerate(LEG_NAMES):
        axs[0].plot(t, trace['action'][:, 1 + i], color=colors[i], lw=1, label=name)
        axs[1].plot(t, trace['action'][:, 5 + i], color=colors[i], lw=1)
        axs[2].plot(t, 100 * trace['offset_target_m'][:, i, 0],
                    color=colors[i], lw=1, ls='--')
        axs[2].plot(t, 100 * trace['offset_m'][:, i, 0], color=colors[i], lw=1)
    frequency_action = trace['action'][:, 17]
    frequency_target = 1 + frequency_action * np.where(frequency_action >= 0, .2, .65)
    duty_action = trace['action'][:, 18]
    duty_base = np.array([.60, .50, .45, .45])[trace['gait_id'].astype(int)]
    duty_target = duty_base + duty_action * np.where(duty_action >= 0, .25, .15)
    axs[3].plot(t, frequency_target, color='#d88611', ls='--', label='Policy target')
    axs[3].plot(t, trace['frequency_scale'][:, 0], color='#254b88', label='Filtered state')
    axs[4].plot(t, duty_target, color='#d88611', ls='--', label='Policy target')
    axs[4].plot(t, trace['duty_factor'][:, 0], color='#254b88', label='Filtered state')
    axs[0].set_ylabel('Horizontal amplitude\naction [-1, 1]')
    axs[1].set_ylabel('Lift amplitude\naction [-1, 1]')
    axs[2].set_ylabel('X equilibrium\ntarget / state [cm]')
    axs[3].set_ylabel('Shared frequency\nscale')
    axs[4].set_ylabel('Stance duty\nfactor')
    axs[4].set_xlabel('Time [s]')
    axs[0].legend(ncol=4, loc='upper right')
    axs[3].legend(loc='upper right')
    for ax in axs: bands(ax)
    fig.suptitle('Policy modulation targets and filtered controller states')
    fig.savefig(output / '04_policy_modulation.png'); plt.close(fig)

    # Show actual gap traversal with phase, contact, amplitude and equilibrium states.
    mask = (x >= 9.4) & (x <= 11.5)
    if np.count_nonzero(mask) >= 5:
        tt = t[mask]
        fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
        for i, name in enumerate(LEG_NAMES):
            axs[0].plot(tt, np.unwrap(trace['phase_rad'][mask, i]) / (2 * np.pi),
                        color=colors[i], label=name, lw=1)
            axs[1].step(tt, trace['foot_contact'][mask, i].astype(int) + i * 1.3,
                        color=colors[i], where='post', lw=1)
            axs[2].plot(tt, trace['amplitude_x_state'][mask, i], color=colors[i], lw=1)
            axs[3].plot(tt, 100 * trace['offset_m'][mask, i, 2], color=colors[i], lw=1)
        gap_times = np.interp([10.5, 10.5 + GAP_WIDTH], x, t)
        for ax in axs:
            ax.axvspan(*gap_times, color='#f0cfb2', alpha=.5, label='Base over gap')
        axs[0].set_ylabel('Unwrapped phase [cycles]'); axs[0].legend(ncol=4, loc='upper left')
        axs[1].set_ylabel('Foot contact\nFL / FR / RL / RR')
        axs[1].set_yticks(np.arange(4) * 1.3 + .5, LEG_NAMES)
        axs[2].set_ylabel('Horizontal amplitude\n[state units]')
        axs[3].set_ylabel('Z equilibrium offset [cm]'); axs[3].set_xlabel('Time [s]')
        fig.suptitle('Gap passage: measured intrinsic state and foot contact')
        fig.savefig(output / '03_gap_detail.png'); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    target = trace['foot_target_m']
    hurdle = (x >= 6.9) & (x <= 7.8)
    gap = (x >= 9.7) & (x <= 11.4)
    for i, ax in enumerate(axs.flat):
        ax.plot(target[1:, i, 0], target[1:, i, 2], color='#b7c2ce', lw=.85,
                label='Full rollout')
        ax.plot(target[hurdle, i, 0], target[hurdle, i, 2], color='#1774b6',
                lw=1.4, label='Hurdle vicinity')
        ax.plot(target[gap, i, 0], target[gap, i, 2], color='#d4770b',
                lw=1.5, label='Gap vicinity')
        ax.set_title(LEG_NAMES[i])
        ax.set_xlabel('Local foot target X [m]')
        ax.set_ylabel('Local foot target Z [m]')
        ax.grid(alpha=.2)
    axs[0, 0].legend(loc='lower left')
    fig.suptitle('Four leg target trajectories generated by the intrinsic dynamics')
    fig.savefig(output / '05_foot_target_loops.png'); plt.close(fig)


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--output', type=Path)
    own.add_argument('--no-video', action='store_true')
    own.add_argument('--no-plots', action='store_true')
    own.add_argument('--max-difficulty', action='store_true')
    own.add_argument('--resolution', choices=('540p', '1080p', '1440p'), default='540p')
    own.add_argument('--command-speed', type=float, default=1.)
    own.add_argument('--max-seconds', type=float, default=24.)
    own_args, remaining = own.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'MP_full_a1'
    args.resume = True
    args.load_run = str(RUN)
    args.checkpoint = 200 if args.checkpoint is None else args.checkpoint
    args.seed = 2026 if args.seed is None else args.seed
    args.headless = True
    if not .3 <= own_args.command_speed <= 1.5:
        raise ValueError('--command-speed must be within the saved training range [0.3, 1.5] m/s')
    configure_course(own_args.max_difficulty)
    default_output = (ROOT / 'docs/terrain/mp_full_group_showcase_max_2026-10-28'
                      if own_args.max_difficulty else DEFAULT_OUTPUT)
    output = (own_args.output or default_output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg, train_cfg = prepare_resume_configs(args.task, cfg, train_cfg, args)
    cfg.seed = train_cfg.seed = args.seed
    cfg.env.num_envs = 1
    cfg.env.play = True
    cfg.env.episode_length_s = 30.
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.terrain_length = COURSE_LENGTH
    cfg.terrain.terrain_width = COURSE_WIDTH
    cfg.terrain.selected = 'flat'
    cfg.terrain.terrain_columns = ['flat']
    cfg.terrain.curriculum = False
    cfg.terrain.fixed_terrain_level = 0
    cfg.terrain.spawn_xy_jitter = 0.
    cfg.commands.ranges.lin_vel_x = [own_args.command_speed] * 2
    cfg.commands.resampling_time = 100.
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    patch_course_generator()
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(
        env, name=args.task, args=args, train_cfg=train_cfg, env_cfg=cfg,
        save_cfg=False, log_root=str(RUN.parent))
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    env.commands[0, 0] = own_args.command_speed
    env.commands[0, 1:] = 0.
    env.root_states[0, 7:13] = 0.
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.compute_observations()
    obs = env.get_observations()

    scale = env.terrain.horizontal_scale
    profile_x = np.arange(round(COURSE_LENGTH / scale)) * scale
    profile_h = env.terrain.height_field_raw[
        env.terrain.border:env.terrain.border + len(profile_x),
        env.terrain.border + round(COURSE_WIDTH / (2 * scale))].astype(float) * env.terrain.vertical_scale
    traces = [snapshot(env, 0.)]
    terminal_events = []
    original_reset = env.reset_idx

    def track_reset(ids, completed_episode=False):
        if completed_episode and len(ids):
            terminal_events.append({
                'time_s': float(env.episode_length_buf[0].item() * env.dt),
                'position_m': env.root_states[0, :3].detach().cpu().tolist(),
                'failure': bool(env.terrain_failure_buf[0].item()),
                'timeout': bool(env.time_out_buf[0].item()),
            })
        return original_reset(ids, completed_episode=completed_episode)

    env.reset_idx = track_reset
    camera = None
    writer = None
    frames = 0
    width, height = {'540p': (960, 540), '1080p': (1920, 1080),
                     '1440p': (2560, 1440)}[own_args.resolution]
    if not own_args.no_video:
        props = gymapi.CameraProperties(); props.width = width; props.height = height
        camera = env.gym.create_camera_sensor(env.envs[0], props)
        if camera < 0:
            raise RuntimeError('Isaac Gym camera creation failed')
        writer = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', str(output / 'continuous_course.mp4')
        ], stdin=subprocess.PIPE)

    def write_frame(t):
        nonlocal frames
        from PIL import Image, ImageDraw, ImageFont
        p = env.root_states[0, :3].detach().cpu().numpy()
        high_resolution = height > 540
        env.gym.set_camera_location(camera, env.envs[0],
                                    gymapi.Vec3(float(p[0] + (.65 if high_resolution else .9)),
                                                float(p[1] - (1.8 if high_resolution else 2.6)),
                                                float(p[2] + (.55 if high_resolution else .72))),
                                    gymapi.Vec3(float(p[0] + .25), float(p[1]), float(p[2] - .08)))
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR).reshape(height, width, 4)
        frame = Image.fromarray(rgba[:, :, :3].copy())
        draw = ImageDraw.Draw(frame)
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        scale_text = height / 540
        font = ImageFont.truetype(font_path, round(20 * scale_text))
        small = ImageFont.truetype(font_path, round(14 * scale_text))
        x = float(p[0])
        segment = next((name for name, left, right in segments() if left <= x < right), 'Finish')
        draw.rectangle((0, 0, width, round(67 * scale_text)), fill=(18, 29, 48))
        draw.text((round(16 * scale_text), round(8 * scale_text)),
                  'MP_full A1  |  Flat  >  Ramps  >  Hurdle  >  Gap', fill='white', font=font)
        draw.text((round(16 * scale_text), round(39 * scale_text)),
                  f'{segment}   |   t = {t:.2f} s   |   x = {x:.2f} m   |   checkpoint 200 / 2000',
                  fill=(185, 215, 240), font=small)
        writer.stdin.write(np.asarray(frame).tobytes())
        frames += 1

    start = time.time()
    if writer is not None:
        write_frame(0.)
    outcome = 'Stopped at maximum time'
    with torch.inference_mode():
        for step in range(1, round(own_args.max_seconds / env.dt) + 1):
            obs, _, _, dones, _ = runner.inference_rollout(obs, *policy)
            if dones[0].item():
                event = terminal_events[-1] if terminal_events else {}
                outcome = 'Episode terminated before finish at x = %.2f m' % event.get('position_m', [np.nan])[0]
                print(outcome, flush=True)
                break
            t = step * env.dt
            traces.append(snapshot(env, t))
            if writer is not None and step % 2 == 0:
                write_frame(t)
            x = float(env.root_states[0, 0].item())
            if x >= FINISH_X:
                outcome = 'Completed all obstacles in one episode'
                print(outcome, flush=True)
                break
            if step % 100 == 0:
                print(f'{t:.1f} s simulated, x={x:.2f} m, {time.time() - start:.1f} s wall', flush=True)
    if writer is not None:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
    trace = {key: np.stack([row[key] for row in traces]) for key in traces[0]}
    trace['terrain_x_m'] = profile_x
    trace['terrain_height_m'] = profile_h
    np.savez_compressed(output / 'rollout_trace.npz', **trace)
    checkpoint = next(RUN.glob(f'model_{args.checkpoint}_*.pt'))
    metadata = {
        'outcome': outcome,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'config': str(RUN / 'config.json'),
        'seed': args.seed,
        'command_forward_mps': own_args.command_speed,
        'duration_s': float(trace['time_s'][-1]),
        'final_x_m': float(trace['base_pos_m'][-1, 0]),
        'max_x_m': float(trace['base_pos_m'][:, 0].max()),
        'video_frames': frames,
        'video_resolution': [width, height],
        'course_variant': 'max_difficulty' if own_args.max_difficulty else 'presentation',
        'course_parameters': {'ramp_grade': RAMP_GRADE, 'hurdle_height_m': HURDLE_HEIGHT,
                              'gap_width_m': GAP_WIDTH, 'gap_depth_m': .60},
        'leg_order': list(LEG_NAMES),
        'segments': [{'name': n, 'start_x_m': a, 'end_x_m': b} for n, a, b in segments()],
        'terminal_events': terminal_events,
        'notes': 'One continuous deterministic policy inference; no reset or edited frames before finish.',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if not own_args.no_plots:
        plot_results(output, trace, profile_x, profile_h, metadata)
    print(json.dumps({k: metadata[k] for k in ('outcome', 'duration_s', 'max_x_m', 'video_frames')}, indent=2))


if __name__ == '__main__':
    main()
