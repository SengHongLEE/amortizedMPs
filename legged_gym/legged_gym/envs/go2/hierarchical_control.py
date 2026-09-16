# MIT License

# Copyright (c) 2026 Seng-Hong Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os
from legged_gym.utils import task_registry

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_ import quat_apply_yaw, wrap_to_pi, quaternion2rpy_torch
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.dynamics.go2.dynamics import Amortized_Control
from legged_gym.utils.gait_model import GaitModel, env_cfg

class AmortizedMPs(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.num_gait_classes = cfg.env.num_gait_classes
        self.num_decimation = cfg.env.num_decimation

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if self.cfg.control.learnable_decimation:
            self.param_dim = self.num_actions - self.num_gait_classes - self.num_decimation
            self.num_actions -= (self.num_gait_classes - 1)
            self.num_actions -= (self.num_decimation - 1)
        else:
            self.param_dim = cfg.env.num_coupling_param + cfg.env.num_stride_param
            self.num_actions -= (self.num_decimation + self.num_gait_classes - 1)
            self.num_obs -= 1
            self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
            self.last_obs = torch.zeros(self.num_envs, 5, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.import_gait_model(self.cfg.il.gait_model_filename)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions, max_decimation):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.rew_buf[:] = 0.
        self.reset_buf[:] = 0
        self.actions.copy_(actions)

        # 滑动窗口：把历史动作往前挪
        self.decimations[:] = torch.roll(self.decimations, shifts=-1, dims=1)
        self.last_gaits[:] = torch.roll(self.last_gaits, shifts=-1, dims=1)
        if self.cfg.control.learnable_decimation:
            self.last_gaits[:, -1] = self.actions[:, 1].to(torch.int)
            self.actions[:, 2:].clamp_(-clip_actions, clip_actions)
            self.decimations[:, -1] = max_decimation
        else:
            self.last_gaits[:, -1] = self.actions[:, 0].to(torch.int)
            self.actions[:, 1:].clamp_(-clip_actions, clip_actions)
            self.decimations[:, -1] = 1     #0 for n=1  1 for n=2  2 for n=4

        ranges = self.amortized_control._scale_helper(self.actions[:,-1], 0.2, 3.)
        self.gain[:,:3,0] = ranges.unsqueeze(-1)  
            
        # step physics and render each frame
        cycles = self.cfg.control.cycle

        for cycle in range(cycles):
            self.mps_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.root_states[:, 3:7],
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.contact_forces[:, self.feet_indices, 2] > 1.
                                    ),dim=-1)

            if cycle == 0:
                self.mps_actions_primary[:, 0] = torch.clip(self.gait_models[0](torch.cat((self.mps_obs_buf,
                                                        self.gain[:, 0, :3] * self.commands_scale
                                                        ),dim=-1)[:].detach()), -clip_actions, clip_actions).to(self.device)
                self.mps_actions_primary[:, 1] = torch.clip(self.gait_models[2](torch.cat((self.mps_obs_buf,
                                            self.gain[:, 1, :3] * self.commands_scale
                                            ),dim=-1)[:].detach()), -clip_actions, clip_actions).to(self.device)
                self.mps_actions_primary[:, 2] = torch.clip(self.gait_models[4](torch.cat((self.mps_obs_buf,
                                            self.gain[:, 2, :3] * self.commands_scale
                                            ),dim=-1)[:].detach()), -clip_actions, clip_actions).to(self.device)

            gait_inputs = [
                torch.cat((self.mps_obs_buf, self.gain[:, gait_idx, :3] * self.commands_scale), dim=-1).detach()
                for gait_idx in range(3)
            ]
            env_ids = ((self.decimations[:, -1] == 2) & (self.count_buf == 1)).nonzero(as_tuple=True)[0]
            self.mps_actions.copy_(self.mps_actions_primary)

            env_ids = ((self.decimations[:, -1] == 1) & ((self.count_buf == 1) | (self.count_buf == 3))).nonzero(as_tuple=True)[0]
            self.mps_actions[env_ids, 0] = self.mps_actions_primary[env_ids, 0] + torch.clip(self.gait_models[1](gait_inputs[0][env_ids]), -clip_actions, clip_actions)
            self.mps_actions[env_ids, 1] = self.mps_actions_primary[env_ids, 1] + torch.clip(self.gait_models[3](gait_inputs[1][env_ids]), -clip_actions, clip_actions)
            self.mps_actions[env_ids, 2] = self.mps_actions_primary[env_ids, 2] + torch.clip(self.gait_models[5](gait_inputs[2][env_ids]), -clip_actions, clip_actions)
            
            env_ids = (self.decimations[:, -1] == 0).nonzero(as_tuple=True)[0]
            self.mps_actions[env_ids, 0] = self.mps_actions_primary[env_ids, 0] + torch.clip(self.gait_models[1](gait_inputs[0][env_ids]), -clip_actions, clip_actions)
            self.mps_actions[env_ids, 1] = self.mps_actions_primary[env_ids, 1] + torch.clip(self.gait_models[3](gait_inputs[1][env_ids]), -clip_actions, clip_actions)
            self.mps_actions[env_ids, 2] = self.mps_actions_primary[env_ids, 2] + torch.clip(self.gait_models[5](gait_inputs[2][env_ids]), -clip_actions, clip_actions)
            
            self.mps_actions.clamp_(-clip_actions, clip_actions)

            for substep in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(self.actions[:,self.num_actions-self.param_dim-1:], self.mps_actions[:,:3]).view(self.torques.shape)
                self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.post_decimation_step(cycle * self.cfg.control.decimation + substep)
                self.render()

            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.base_quat[:] = self.root_states[:, 3:7]
            self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        self.privileged_obs_buf = torch.clip(self.last_obs.reshape(self.num_envs, -1), -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_decimation_step(self, dec_i):
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])
        self.substep_exceed_dof_pos_limit_abs[:, dec_i, :] = torch.clip(torch.maximum(
            self.dof_pos_limits[:, 0] - self.dof_pos,
            self.dof_pos - self.dof_pos_limits[:, 1],
        ), min= 0) # make sure the value is non-negative

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf[:] += 1
        # print(f"episode~~~:{self.episode_length_buf}")
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids_reset = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids_reset)
        self.compute_observations()

        self.last_obs[:] = torch.roll(self.last_obs, shifts=-1, dims=1)
        self.last_obs[:, -1] = self.obs_buf

        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_pos[:] = self.root_states[:, 0]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= torch.abs(quaternion2rpy_torch(self.base_quat)[1]) > 0.8
        self.reset_buf |= torch.abs(quaternion2rpy_torch(self.base_quat)[0]) > 0.8
        self.reset_buf |= self.root_states[:,2] < 0.25

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False), torch.ones(self.num_envs, device=self.device).to(torch.int))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.amortized_control.reset(env_ids)

        # reset buffers
        self.last_base_pos[env_ids]= 0.
        self.last_obs[env_ids] = 0.
        self.decimations[env_ids] = 0
        self.last_gaits[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_buf[env_ids] = 0.
        # self.count[env_ids] == 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self._resample_commands(env_ids)
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf = torch.clip(self.rew_buf, min=0.)


    def compute_observations(self):
        """ Computes observations
        """
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                         self.base_ang_vel  * self.obs_scales.ang_vel,
        #                         self.projected_gravity,
        #                         self.root_states[:, 3:7],
        #                         (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                         self.dof_vel * self.obs_scales.dof_vel,
        #                         self.contact_forces[:, self.feet_indices, 2] > 1.,
        #                         self.commands[:, :3] * self.commands_scale,
        #                         self.actions
        #                         ),dim=-1)


        self.obs_buf = torch.cat((
                        self.commands[:, :3] * self.commands_scale,
                        self.actions,
                        (self.amortized_control.X[:,0,:]) * self.obs_scales.dof_pos,
                        (self.amortized_control.X[:,1,:] - np.pi) * 1/np.pi,
                        self.amortized_control.X_dot[:,0,:] * 1/30, 
                        (self.amortized_control.X_dot[:,1,:] - 15) * 1/30,
                        ),dim=-1)        

        return

    def import_gait_model(self, paths):
        self.gait_models = []
        _, train_cfg = task_registry.get_cfgs('MP_go2')
        train_cfg_dict = class_to_dict(train_cfg)
        cfg = env_cfg(num_envs=self.num_envs, num_obs=37, num_privileged_obs=None, num_actions=12)

        for path in paths:
            model = GaitModel(train_cfg_dict, cfg, None, self.device)
            model.load(path, False)
            policy = model.get_inference_policy(device=self.device)
            self.gait_models.append(policy)


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                props["stiffness"][i] = 0.0
                props["damping"][i] = 0.0

        return props

    def _process_rigid_body_props(self, props, env_id):
        # if True:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # lol = 0
        # for i in range(len(props)):
        #     lol += props[i].mass
        # breakpoint()
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def get_privileged_observations(self):
        return self.last_obs.reshape(self.num_envs, -1)


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return

        if self.cfg.env.play:
            self.commands[env_ids, 0] = 1.
        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions_scaled, mps_actions):
        """ Compute torques from mps Signals (solve Inverse kinematics and run PD Control)
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        mps_actions_scaled = mps_actions * self.cfg.control.action_scale
        actions_beta_scaled = actions_scaled[:, :-1]
        des_joint_pos = torch.zeros_like(self.torques,device=self.device)
        xs, ys, zs = self.amortized_control.update(mps_actions_scaled, actions_beta_scaled, self.gain[:, :, 0])
        for ig_idx, leg_idx in enumerate(self.leg_indices):
            y = self.side_sign[leg_idx] * self.foot_y + ys[:, leg_idx]
            des_joint_pos[:, 3 * ig_idx:3 * ig_idx + 3] = self.amortized_control.compute_inverse_kinematics(
                self.cfg.asset,
                leg_idx,
                xs[:, leg_idx],
                y,
                zs[:, leg_idx],
            )
        self.dof_des_pos = des_joint_pos
        tmp = 0.2
        gait_ids = actions_beta_scaled[:, 0].long()
        hip_offsets = mps_actions_scaled[torch.arange(self.num_envs, device=self.device), gait_ids, 8:12]
        self.dof_des_pos[:, self.hip_joint_indices] = self.default_dof_pos[:, self.hip_joint_indices] + hip_offsets * tmp

        torques = self.p_gains * (self.dof_des_pos - self.dof_pos) - self.d_gains * self.dof_vel
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos 
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            if self.cfg.env.play:
                origins_xz_indices = torch.randint(0,1, (len(env_ids), 1), device=self.device).squeeze(1)
            else: 
                origins_xz_indices = torch.randint(0, len(self.terrain_origins_x) - 1, (len(env_ids), 1), device=self.device).squeeze(1)
            rand_origins_x = self.terrain_origins_x[origins_xz_indices]
            rand_origins_y = self.terrain_origins_y[origins_xz_indices]
            rand_origins_z = self.terrain_origins_z[origins_xz_indices]
            self.env_origins[env_ids, 0] = rand_origins_x
            self.env_origins[env_ids,1] = rand_origins_y
            self.env_origins[env_ids, 2] = rand_origins_z
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 1:2] += torch_rand_float(-1., 1., (len(env_ids), 1), device=self.device)
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # breakpoint()
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) 
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) # [0,0,-1] ? 
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False) # always 12 motors... num_actions will changes based on space 
        self.dof_des_pos =torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False) # always 12 motors... num_actions will changes based on space 
        
        self.p_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_obs = torch.zeros(self.num_envs, 1, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.decimations = torch.zeros(self.num_envs, 5, dtype=torch.int, device=self.device, requires_grad=False)
        self.last_gaits = torch.zeros(self.num_envs, 5, dtype=torch.int, device=self.device, requires_grad=False)
        self.last_base_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
  
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
 
        self.amortized_control = Amortized_Control(time_step=self.sim_params.dt,num_envs=self.num_envs,device=self.device, num_of_MPs=3)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name] #+ torch_rand_float(-30.0, 30.0, (len(env_ids), self.num_dof), device=self.device)
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.gain = torch.zeros(self.num_envs, 3, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.side_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device)
        self.leg_indices = (1, 0, 3, 2)
        self.hip_joint_indices = torch.tensor([0, 3, 6, 9], device=self.device)
        self.foot_y = torch.full((self.num_envs,), self.cfg.asset.hip_link_length_a1, device=self.device)
        self.mps_obs_buf = torch.zeros(self.num_envs, 41, device=self.device, dtype=torch.float)
        self.mps_actions = torch.zeros(self.num_envs, 3, 12, device=self.device, dtype=torch.float)
        self.mps_actions_primary = torch.zeros(self.num_envs, 3, 12, device=self.device, dtype=torch.float)
        self.count_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limit_abs = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= (self.dt)
        # self.reward_scales['ref_motion'] = 1.0
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_colsmeasured_heights
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.terrain_origins_x = self.terrain_origins[:,:,0].squeeze().flatten()
            self.terrain_origins_y = self.terrain_origins[:,:,1].squeeze().flatten()
            self.terrain_origins_z = self.terrain_origins[:,:,2].squeeze().flatten()
            origins_xz_indices = torch.randint(0, len(self.terrain_origins_x) - 1, (self.num_envs, 1), device=self.device).squeeze(1)
            rand_origins_x = self.terrain_origins_x[origins_xz_indices]
            rand_origins_y = self.terrain_origins_y[origins_xz_indices]
            rand_origins_z = self.terrain_origins_z[origins_xz_indices]
            self.env_origins[:,0] = rand_origins_x
            self.env_origins[:,1] = rand_origins_y
            self.env_origins[:,2] = rand_origins_z
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt * self.cfg.control.cycle
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    #------------ reward functions---------------
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_yaw(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, 2:3]-(-1)), dim=1)

    def _reward_tracking_lin_vel(self):

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # return torch.square(self.base_lin_vel[:, 2])
        return torch.square(self.root_states[:, 9] - self.last_root_vel[:, 2])

    def _reward_ang_vel_x(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, 0:1]), dim=1)

    def _reward_ang_vel_y(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, 1:2]), dim=1)

    def _reward_tracking_ang_vel(self):
        # ang_vel_error = torch.square((self.commands[:, 2] - self.base_ang_vel[:, 2]))
        ang_vel_error = torch.square((0 - self.base_ang_vel[:, 2]))
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)


    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        diffs_decimations = self.decimations[:, 1:] - self.decimations[:, :-1]
        diffs_gaits = self.last_gaits[:, 1:] - self.last_gaits[:, :-1]
        diffs_params = torch.sum(torch.square(self.last_actions[:, 1:] - self.last_actions[:, :-1]), dim=-1)
        
        # return torch.sum(5. * torch.square(diffs_decimations), dim=-1)
        # return torch.sum(torch.square(diffs_decimations) + torch.square(diffs_gaits), dim=1)
        return torch.sum(0.0001 * torch.square(diffs_decimations) + 0.0001 * torch.square(diffs_gaits) + diffs_params, dim=1)

    def _reward_computation(self):
        return torch.sum((2 - self.decimations), dim=-1)
        # return torch.sum(torch.square(diffs_decimations) + torch.square(diffs_gaits), dim=1)
        # return torch.sum(5. * torch.square(diffs_decimations) + torch.square(diffs_gaits), dim=1)

    def _reward_hip_joint(self):
        # Penalize xy axes base angular velocity
        tmp_1 = torch.square(self.dof_pos[:,0] - self.default_dof_pos[:,0])
        tmp_2 = torch.square(self.dof_pos[:,3] - self.default_dof_pos[:,3])
        tmp_3 = torch.square(self.dof_pos[:,6] - self.default_dof_pos[:,6])
        tmp_4 = torch.square(self.dof_pos[:,9] - self.default_dof_pos[:,9])
        # print(self.dof_pos[:,0])
        return tmp_1+tmp_2+tmp_3+tmp_4

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    

    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)
    
    
    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_locomotion_distance(self):
        current_base_position =  (self.root_states[:, 0])
        forward_reward = (current_base_position - self.last_base_pos).clamp(
                            max=self.cfg.commands.max_vel_x * self.dt
                        )
        return forward_reward
