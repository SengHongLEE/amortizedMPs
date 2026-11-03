"""Record the current play.py course and controller state from one MP_full rollout."""

import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import isaacgym  # Must precede torch.
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.scripts.play import configure_composite_terrain
from legged_gym.utils import get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym.utils.load_config import prepare_resume_configs


ROOT = Path(__file__).resolve().parents[3]
PLAY_PATH = Path(__file__).with_name('play.py')
DEFAULT_OUTPUT = ROOT / 'docs/terrain/mp_full_play_showcase_100fps_2026-11-02'


def play_course_spec(source):
    """Read the literal playback settings so recording tracks edits to play.py."""
    module = ast.parse(source)
    play = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == 'play')
    call = next(node for node in ast.walk(play) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'configure_composite_terrain')
    keywords = {item.arg: item.value for item in call.keywords}
    try:
        spec = {key: ast.literal_eval(keywords[key])
                for key in ('terrain_rows', 'difficulty', 'terrain_params')}
    except (KeyError, ValueError) as exc:
        raise ValueError('play.py course settings must be literal values') from exc
    for node in ast.walk(play):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == 'terrain_length':
                spec['terrain_length'] = ast.literal_eval(node.value)
    return spec


def snapshot(env, t):
    controller = env.ID
    values = {
        'time_s': np.float32(t),
        'base_pos_m': env.root_states[0, :3],
        'base_quat_xyzw': env.root_states[0, 3:7],
        'body_lin_vel_mps': env.base_lin_vel[0],
        'body_ang_vel_radps': env.base_ang_vel[0],
        'command_mps': env.commands[0, :3],
        'gait_id': controller._gait_ids[0],
        'equilibrium_m': controller.offset[0][:, [0, 2]],
        'frequency_hz': controller.omega[0],
        'amplitude_x': controller.X[0, 0],
        'amplitude_z': controller.X[0, 1],
        'phase_rad': controller.X[0, 2],
    }
    return {key: value.detach().cpu().numpy().copy() if torch.is_tensor(value)
            else np.asarray(value).copy() for key, value in values.items()}


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    own.add_argument('--no-video', action='store_true')
    own.add_argument('--max-seconds', type=float, default=45.)
    own.add_argument('--fps', type=int, default=100)
    own.add_argument('--camera-width', type=int, default=1600)
    own.add_argument('--camera-height', type=int, default=900)
    own_args, remaining = own.parse_known_args()
    if own_args.max_seconds <= 0 or own_args.fps <= 0:
        raise ValueError('max-seconds and fps must be positive')
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'MP_full_a1'
    args.resume = True
    args.load_run = '-1'
    args.checkpoint = -1
    args.headless = True

    play_source = PLAY_PATH.read_text()
    spec = play_course_spec(play_source)
    output = own_args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg, train_cfg = prepare_resume_configs(args.task, cfg, train_cfg, args)
    if args.seed is not None:
        cfg.seed = train_cfg.seed = args.seed
    run = Path(args.load_run)
    checkpoint = Path(get_load_path(str(run.parent), re.compile(r'^model_(\d+)_\d+\.pt$'),
                                    load_run=str(run), checkpoint=-1))
    cfg.env.play = True
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    if 'terrain_length' in spec:
        cfg.terrain.terrain_length = spec['terrain_length']
    configure_composite_terrain(cfg, spec['terrain_rows'], spec['difficulty'],
                                spec['terrain_params'], num_envs=1)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(
        env, name=args.task, args=args, train_cfg=train_cfg, env_cfg=cfg,
        save_cfg=False, log_root=str(run.parent))
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    scale = env.terrain.horizontal_scale
    n_cells = cfg.terrain.num_rows * env.terrain.length_per_env_pixels
    y_cell = env.terrain.border + env.terrain.width_per_env_pixels // 2
    terrain_x = np.arange(n_cells, dtype=np.float32) * scale
    terrain_h = (env.terrain.height_field_raw[
        env.terrain.border:env.terrain.border + n_cells, y_cell].astype(np.float32)
        * env.terrain.vertical_scale)
    traces = [snapshot(env, 0.)]
    camera_steps = []
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
    physics_dt = float(env.sim_params.dt)
    capture_stride = max(1, round(1 / (own_args.fps * physics_dt)))
    actual_fps = 1 / (capture_stride * physics_dt)
    if abs(actual_fps - own_args.fps) > .01:
        raise ValueError(f'Requested fps cannot match physics dt={physics_dt}; got {actual_fps}')
    if not own_args.no_video:
        props = gymapi.CameraProperties()
        props.width, props.height = own_args.camera_width, own_args.camera_height
        camera = env.gym.create_camera_sensor(env.envs[0], props)
        if camera < 0:
            raise RuntimeError('Isaac Gym camera creation failed')
        writer = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{props.width}x{props.height}', '-r', str(own_args.fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'camera_raw.mp4')
        ], stdin=subprocess.PIPE)

    def write_frame(step):
        nonlocal frames
        p = env.root_states[0, :3].detach().cpu().numpy()
        env.gym.set_camera_location(
            camera, env.envs[0],
            gymapi.Vec3(float(p[0] - .35), float(p[1] - 1.7), float(p[2] + .9)),
            gymapi.Vec3(float(p[0] + 1.0), float(p[1]), float(p[2] - .15)))
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(
            env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR).reshape(
                own_args.camera_height, own_args.camera_width, 4)
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        if frames == 0:
            from PIL import Image
            Image.fromarray(rgb).save(output / 'camera_preview.png')
        writer.stdin.write(rgb.tobytes())
        camera_steps.append(step)
        frames += 1

    physics_steps = 0
    original_post_decimation_step = env.post_decimation_step

    def capture_substep(substep):
        nonlocal physics_steps
        original_post_decimation_step(substep)
        physics_steps += 1
        if physics_steps % capture_stride:
            return
        # A physics substep supplies a distinct image and oscillator state.
        env.gym.fetch_results(env.sim, True)
        traces.append(snapshot(env, physics_steps * physics_dt))
        if writer is not None:
            write_frame(len(traces) - 1)

    env.post_decimation_step = capture_substep

    finish_x = cfg.terrain.num_rows * env.terrain.env_length - 1.
    outcome = 'time_limit'
    start_wall = time.time()
    try:
        if writer is not None:
            write_frame(0)
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), torch.inference_mode():
            for step in range(1, round(own_args.max_seconds / env.dt) + 1):
                obs, _, _, dones, _ = runner.inference_rollout(obs, *policy)
                if dones[0].item():
                    outcome = 'terminated'
                    break
                t = step * env.dt
                x = float(env.root_states[0, 0].item())
                if x >= finish_x:
                    outcome = 'reached_finish'
                    break
                if step % 100 == 0:
                    print(f'{t:.1f}s  x={x:.2f}m  wall={time.time()-start_wall:.1f}s',
                          file=sys.stderr, flush=True)
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('ffmpeg camera encoding failed')

    trace = {key: np.stack([row[key] for row in traces]) for key in traces[0]}
    trace['terrain_x_m'] = terrain_x
    trace['terrain_height_m'] = terrain_h
    trace['camera_steps'] = np.asarray(camera_steps, dtype=np.int32)
    np.savez_compressed(output / 'rollout_trace.npz', **trace)
    gait = trace['gait_id'].astype(int)
    metadata = {
        'outcome': outcome,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'run_config': str(run / 'config.json'),
        'play_path': str(PLAY_PATH),
        'play_sha256': hashlib.sha256(play_source.encode()).hexdigest(),
        'course_spec': spec,
        'terrain_names': env.terrain.terrain_names[:, 0].tolist(),
        'terrain_length_m': env.terrain.env_length,
        'finish_x_m': finish_x,
        'seed': cfg.seed,
        'duration_s': float(trace['time_s'][-1]),
        'max_x_m': float(trace['base_pos_m'][:, 0].max()),
        'gaits_seen': sorted(set(gait.tolist())),
        'gait_transitions': int(np.count_nonzero(np.diff(gait))),
        'video_frames': frames,
        'video_fps': actual_fps,
        'physics_dt_s': physics_dt,
        'trace_rate_hz': actual_fps,
        'camera_resolution': [own_args.camera_width, own_args.camera_height],
        'terminal_events': terminal_events,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: metadata[key] for key in
                      ('outcome', 'duration_s', 'max_x_m', 'gaits_seen',
                       'gait_transitions', 'video_frames')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
