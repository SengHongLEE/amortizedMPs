"""Start the real A1 task, take steps, and reset a subset on configurable terrain."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'legged_gym/legged_gym/test/torch_extensions'))
# Isaac Gym must be imported before torch.
from isaacgym import gymapi
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', choices=['plane', 'trimesh', 'heightfield'], default='trimesh')
parser.add_argument('--device', default='cpu')
parser.add_argument('--task', choices=['multi', 'proprio'], default='multi')
parser.add_argument('--configured', action='store_true', help='Keep the task terrain config exactly as written.')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--curriculum', choices=['on', 'off'], help='Override the proprio curriculum switch.')
parser.add_argument('--fixed-level', type=int, help='Fixed row when the proprio curriculum is off.')
parser.add_argument('--all-types', action='store_true', help='Exercise every built-in terrain with two actors per type.')
parser.add_argument('--equilibrium-actions', action='store_true', help='Exercise all offset axes at their limits (proprio).')
parser.add_argument('--noise', action='store_true', help='Enable sensor noise.')
parser.add_argument('--policy-update', action='store_true', help='Run one real PPO update with the configured actors (proprio).')
parser.add_argument('--adaptive', action='store_true', help='Use the v2 dynamical action space (proprio).')
parser.add_argument('--modal', action='store_true', help='Use the coordinated v3 action space (proprio).')
args = parser.parse_args()
if args.task == 'proprio':
    from legged_gym.envs.a1.mp_proprio_terrain import MotorPrimitives
    from legged_gym.envs.a1.mp_proprio_terrain_config import MotorPrimitivesCfg
    if args.adaptive:
        from legged_gym.envs.a1.mp_proprio_terrain_config import AdaptiveMotorPrimitivesCfg as MotorPrimitivesCfg
    if args.modal:
        from legged_gym.envs.a1.mp_proprio_terrain_config import ModalMotorPrimitivesCfg as MotorPrimitivesCfg
else:
    from legged_gym.envs.a1.mp_multi_primitive import MotorPrimitives
    from legged_gym.envs.a1.mp_multi_primitive_config import MotorPrimitivesCfg
from legged_gym.utils.terrains.blind_terrain import TERRAIN_GENERATORS
if args.configured and args.all_types:
    parser.error('--configured and --all-types are mutually exclusive')
if args.steps < 1:
    parser.error('--steps must be positive')
cfg = MotorPrimitivesCfg()
cfg.env.num_envs = 8
cfg.noise.add_noise = args.noise
cfg.terrain.mesh_type = args.mesh
if args.curriculum is not None:
    cfg.terrain.curriculum = args.curriculum == 'on'
if args.fixed_level is not None:
    cfg.terrain.fixed_terrain_level = args.fixed_level
if args.configured:
    cfg.env.num_envs = 2 * cfg.terrain.num_cols
else:
    cfg.terrain.num_rows, cfg.terrain.num_cols = 2, 6
    cfg.terrain.selected = None
    cfg.terrain.terrain_columns = None
    cfg.terrain.row_difficulties = None
    cfg.terrain.terrain_proportions = dict.fromkeys(('flat', 'slope', 'rough_slope', 'rough', 'rubble', 'stairs'), 1.)
if args.all_types:
    cfg.terrain.num_cols = len(TERRAIN_GENERATORS)
    cfg.terrain.terrain_proportions = dict.fromkeys(TERRAIN_GENERATORS, 1.)
    cfg.env.num_envs = 2 * cfg.terrain.num_cols
cfg.domain_rand.push_robots = False
params = gymapi.SimParams()
params.dt = cfg.sim.dt
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0., 0., -9.81)
params.use_gpu_pipeline = args.device.startswith('cuda')
params.physx.use_gpu = params.use_gpu_pipeline
params.physx.contact_collection = gymapi.ContactCollection(2)
env = MotorPrimitives(cfg, params, gymapi.SIM_PHYSX, args.device, True)
try:
    obs, critic = env.reset()
    actions = torch.zeros(cfg.env.num_envs, cfg.env.num_actions, device=env.device)
    actions[:, 0] = torch.arange(cfg.env.num_envs, device=env.device) % cfg.env.num_primitives
    if args.equilibrium_actions:
        assert args.task == 'proprio'
        actions[:, 5:17] = (torch.arange(cfg.env.num_envs * 12, device=env.device)
                         .reshape(-1, 12) % 3 - 1).float()
        if args.adaptive or args.modal:
            actions[:, 17:] = torch.linspace(-1., 1., cfg.env.num_envs, device=env.device)[:, None]
    if args.task == 'proprio':
        assert env.ID.nominal_clearance_single == cfg.asset.ground_clearance
        assert env.ID.max_clearance == cfg.asset.max_ground_clearance
    for _ in range(args.steps):
        obs, critic, rewards, done, extras = env.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
        assert all(torch.isfinite(reward).all() for reward in rewards)
        if args.task == 'proprio':
            margin = cfg.control.equilibrium.joint_limit_margin
            assert torch.isfinite(env.dof_des_pos).all()
            assert (env.dof_des_pos >= env.dof_pos_hard_limits[:, 0] + margin - 1e-6).all()
            assert (env.dof_des_pos <= env.dof_pos_hard_limits[:, 1] - margin + 1e-6).all()
            torch.testing.assert_close(obs[:, -1, -cfg.env.num_controller_observations:], env.ID.controller_observations())
    ids = torch.tensor([0, 3], device=env.device)
    if args.task == 'proprio':
        before = env.ID.offset.clone()
        env.reset_idx(ids)
        assert not env.ID.offset[ids].any()
        assert not env.ID.offset_velocity[ids].any()
        torch.testing.assert_close(env.ID.offset[1], before[1])
        assert not env.actions[ids].any()
    if args.task == 'proprio' and env.terrain_curriculum is not None:
        curriculum = env.terrain_curriculum
        before = env.terrain_levels.clone()
        env.reset_idx(ids)
        assert torch.equal(env.terrain_levels, before), 'manual reset must not advance curriculum'
        # Inject two completed episode outcomes to verify the real reset/origin path.
        if curriculum.enabled:
            env.terrain_levels[ids] = min(curriculum.minimum + 1, curriculum.maximum)
            env.env_origins[ids] = env.terrain_origins[env.terrain_levels[ids], env.terrain_types[ids]]
            env.reset_idx(ids)
        before = env.terrain_levels.clone()
        curriculum.command_distance[ids] = 3.
        curriculum.command_displacement[ids, 0] = 3.
        env.root_states[ids, :2] = curriculum.start_xy[ids]
        env.root_states[ids, 0] += 2.7
        env.terrain_failure_buf[ids] = torch.tensor([False, True], device=env.device)
        env.episode_length_buf[ids] = 100
        env.reset_idx(ids, completed_episode=True)
        expected = before.clone()
        if curriculum.enabled:
            expected[ids] = (before[ids] + before.new_tensor([1, -1])).clamp(
                curriculum.minimum, curriculum.maximum)
        assert torch.equal(env.terrain_levels, expected)
        torch.testing.assert_close(env.env_origins[ids],
                                   env.terrain_origins[expected[ids], env.terrain_types[ids]])
        torch.testing.assert_close(curriculum.start_xy[ids], env.root_states[ids, :2])
        assert not curriculum.command_distance[ids].any()
        assert 'terrain_difficulty' in env.extras['episode']
        for _ in range(3):
            obs, critic, rewards, done, extras = env.step(actions)
            assert torch.isfinite(obs).all()
        # Exercise the post_physics_step -> automatic reset path as well.
        env.episode_length_buf[0] = int(env.max_episode_length)
        obs, critic, rewards, done, extras = env.step(actions)
        assert done[0] and env.episode_length_buf[0] == 0
        assert curriculum.command_distance[0] == 0
        assert torch.isfinite(obs).all()
        print(f'PASS: curriculum enabled={curriculum.enabled}, manual/automatic reset, episode outcomes and origin switching')
    else:
        env.reset_idx(ids)
    assert obs.shape == (cfg.env.num_envs, env.num_obs_history, cfg.env.num_observations)
    assert critic.shape == (cfg.env.num_envs, cfg.env.num_privileged_obs)
    print(f'PASS: A1 {args.task} {args.mesh} initialized, {args.steps} steps and partial reset; obs {tuple(obs.shape)}, critic {tuple(critic.shape)}')
    if args.policy_update:
        assert args.task == 'proprio'
        from legged_gym.envs.a1.mp_proprio_terrain_config import MotorPrimitivesCfgPPO
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner
        train_cfg = class_to_dict(MotorPrimitivesCfgPPO())
        train_cfg['algorithm']['num_learning_epochs'] = 1
        train_cfg['algorithm']['num_mini_batches'] = 1
        with tempfile.TemporaryDirectory(prefix='equilibrium-policy-smoke-') as log_dir:
            runner = MultiInteroceptivePolicyRunner(env, train_cfg, log_dir, device=env.device)
            models = [runner.primitive_selection_actor_critic, runner.modulation_param_actor_critic]
            before = [[p.detach().clone() for p in model.parameters()] for model in models]
            runner.learn(1)
            for initial, model in zip(before, models):
                assert all(torch.isfinite(p).all() for p in model.parameters())
                assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
            checkpoint = str(next(Path(log_dir).glob('*.pt')))
            runner.load(checkpoint, load_optimizer=False)
            high, low = runner.get_inference_policy()
            obs, critic, rewards, done, extras = runner.inference_rollout(
                env.get_observations(), high, low)
            assert torch.isfinite(obs).all()
            if runner.writer is not None:
                runner.writer.close()
        print('PASS: both configured actors updated, checkpoint reloaded, inference completed')
finally:
    env.gym.destroy_sim(env.sim)
