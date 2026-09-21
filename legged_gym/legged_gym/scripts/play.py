# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, load_run_config

import numpy as np
import torch
import torch.distributions as dist
from pathlib import Path

import time
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name

    if args.load_run is None or args.load_run == "-1":
        run_dir = sorted(
            p for p in log_root.iterdir()
            if p.is_dir() and (p / "config.json").exists()
        )[-1]
    else:
        run_dir = Path(args.load_run)
        if not run_dir.is_absolute():
            run_dir = log_root / run_dir

    env_cfg, train_cfg = load_run_config(env_cfg, train_cfg, run_dir)

    # override some parameters for testing
    number_of_robots = 1 #
    env_cfg.env.num_envs = min(env_cfg.env.num_envs,number_of_robots)
    env_cfg.env.play = True

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.border_size = 1.

    #stiffness
    env_cfg.terrain.terrain_length = 5.
    env_cfg.terrain.terrain_width = 5.

    # go2
    # env_cfg.terrain.terrain_length = 5.
    # env_cfg.terrain.terrain_width = 2.    
    
    #hurdle
    # env_cfg.terrain.terrain_length = 3.
    # env_cfg.terrain.terrain_width = 15.     

    #gap
    # env_cfg.terrain.terrain_length = 1.
    # env_cfg.terrain.terrain_width = 6.   
    
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.1
    env_cfg.control.varied = False
    # env_cfg.control.cycle = 1
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = True
    tmp = 0.
    env_cfg.domain_rand.added_mass_range = [tmp, tmp]    
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2.    #0.03 0.01
    env_cfg.domain_rand.max_push_vel_xy = 1.    #0.03 0.01
    env_cfg.commands.save_data = False #number of robot should be one
    env_cfg.commands.resampling_time = 2
    env_cfg.env.play = True 
    env_cfg.env.episode_length_s = 10
    # env_cfg.mp_modeling.process_type = 'slow'
    # env_cfg.mp_modeling.gait_model_filename = []
    # env_cfg.terrain.terrain_length = 20
    env_cfg.viewer.pos = [0., -2., 1.4]  # [m]
    env_cfg.viewer.lookat = [0., 0., .3]  # [m]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    print('obs',obs)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, env_cfg=env_cfg, save_cfg= False)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if args.task == 'MP_twin_a1':
        mu_policy, omega_policy = ppo_runner.get_inference_policy(device=env.device)
    if args.task == 'MP_adaptive_a1' or 'MP_transient_a1':
        cycles = torch.stack((ppo_runner.mu_update_cycle, ppo_runner.omega_update_cycle),
            dim=-1) 
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )

        obs = torch.cat((obs, cycles), dim=-1)
        high_policy, mu_policy, omega_policy, hip_policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    stop_rew_log = 488# env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0



    number_of_samples = 100000
    decimation = torch.ones(env.num_envs, device=env.device, dtype=torch.int)
    for i in range(1*int(number_of_samples)):

        if args.task == 'MP_twin_a1':
            obs, _, rews, dones, infos = ppo_runner.inference_rollout(obs, mu_policy, omega_policy)
        if args.task == 'MP_adaptive_a1' or 'MP_transient_a1':
            obs, _, rews, dones, infos = ppo_runner.inference_rollout(obs, high_policy, mu_policy, omega_policy, hip_policy)
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                print(filename)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            
            camera_vel = np.array([1., 1., 0.]) * env.base_lin_vel[0,:].cpu().numpy()
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()
 
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
