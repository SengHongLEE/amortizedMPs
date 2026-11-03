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

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.a1.mp_interoceptive import MotorPrimitives as InteroceptiveMotorPrimitives
from legged_gym.utils.terrain import Terrain
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.dynamics.a1.rhythmic_dynamics import Reference_Motion_Dynamics
from legged_gym.utils.imitation_task import ImitationTask

from legged_gym.utils.math_ import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict, plot_colored_trajectory, append_to_npy_file

from legged_gym.utils import task_registry

class MotorPrimitives(InteroceptiveMotorPrimitives):
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
        self.cfg.control.hierarchical = False
        self.cfg.control.varied_decimation = False
        # if self.cfg.mp_modeling.process_type == 'slow':
        #     self.cfg.control.cycle = 4
        # elif self.cfg.mp_modeling.process_type == 'fast':
        #     self.cfg.control.cycle = 1      # 2 for n=2  1 for n=1

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        # Preserve the reference task's initialization; only shared methods are inherited.
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        self.num_obs_history = 1

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.folder_path = '/home/ubuntu/amortizedMPs/legged_gym/legged_gym/metrics/EXP_MP'
        os.makedirs(self.folder_path, exist_ok=True)

        self.phase_history = []
        self.amplitude_history = []
        self.amplitude_dot_history = []
        self.mu_history = []
        self.omega_history = []
        self.x_history = []
        self.z_history = []
        self.x_act_history = []
        self.z_act_history = []

        self.base_pos_buffer = []
        self.base_rot_buffer = []
        self.command_buffer = []
        self.base_lin_vel_buffer = []
        self.base_ang_vel_buffer = []
        self.contact_boolean_buffer = []
        self.ee_pos_buffer = []
        self.power_buffer = []
        self.obs_buffer = []
        self.update_idx_buffer = []
        self.dof_pos_buffer = []
        self.dof_pos_ref_buffer = []
        self.count = 0
        self.distance_buffer = []


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions

        dt = self.dt * torch.ones(self.num_envs,device=self.device,requires_grad=False)
        self.imitation_task.update_(self.episode_length_buf, self.commands[:,0], self.motion_label, dt)

        self.actions[:] = torch.clip(actions, -clip_actions, clip_actions).to(self.device) 

        decimation = self.cfg.control.decimation * self.cfg.control.cycle
        self._cycle_cost_integral.zero_()
        self._slip_cost_integral.zero_()
        self._touchdown_cost.zero_()

        if self.cfg.env.play:
            print(self.actions)
            print(self.commands)
            print(self.base_lin_vel)
            # print(self.actions)
            # print(self.commands)
            # print(self.base_lin_vel)

        for substep in range(decimation):        
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # prepare quantities
            self.base_quat[:] = self.root_states[:, 3:7]
            self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

            if self.cfg.env.play:
                self.x_act_history.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 0]).detach().cpu().clone())
                self.z_act_history.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 2]).detach().cpu().clone())
                self.base_pos_buffer.append(self.root_states[:, 0:3].cpu().numpy())
                # self.base_rot_buffer.append(torch.abs(quaternion2rpy_torch(self.base_quat)).cpu().numpy())
                self.base_lin_vel_buffer.append(self.base_lin_vel[:, 0:3].cpu().numpy())
                self.base_ang_vel_buffer.append(self.base_ang_vel[:, 0:3].cpu().numpy())
                contact_booleans = self.contact_forces[:, self.feet_indices, 2] > 0.
                self.contact_boolean_buffer.append(contact_booleans.cpu().numpy())
                self.ee_pos_buffer.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 0:3]).cpu().numpy())
                self.power_buffer.append(torch.sum(torch.abs(self.torques*self.dof_vel), dim=1).cpu().detach().numpy())
                self.dof_pos_buffer.append(self.dof_pos.cpu().numpy())
                self.command_buffer.append(self.commands.cpu().numpy())
                # self.dof_pos_ref_buffer.append(self.dof_pos_ref.cpu().numpy())
            self.post_decimation_step(substep)
            self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, [self.rew_buf, self.low_rew_buf], self.reset_buf, self.extras


    def post_physics_step(self, env_ids_end=None):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)


        self.first_iteartion = False
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.episode_length_buf += 1
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() 
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_pos[:] = self.root_states[:, 0]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self, env_ids=None):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.root_states[:,2] < 0.25


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

        self.ID.reset(env_ids)
        self._resample_commands(env_ids)
        self._reset_homeostasis_buffers(env_ids)

        # reset buffers
        self.last_base_pos[env_ids]= self.root_states[env_ids, 0]
        self.obs_buf[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.actions_diff[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.env.play and self.time_out_buf[0] == 1:
            breakpoint()
            append_to_npy_file(f"{self.folder_path}/{self.count}_base_position.npy", np.array(self.base_pos_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_base_lin_vel.npy", np.array(self.base_lin_vel_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_base_ang_vel.npy", np.array(self.base_ang_vel_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_contact_boolean.npy", np.array(self.contact_boolean_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_ee_pos.npy", np.array(self.ee_pos_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_power.npy", np.array(self.power_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_dof_pos.npy", np.array(self.dof_pos_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_command.npy", np.array(self.command_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_update_idx.npy", np.array(self.update_idx_buffer))
            append_to_npy_file(f"{self.folder_path}/{self.count}_mu.npy", np.array(self.mu_history))
            append_to_npy_file(f"{self.folder_path}/{self.count}_omega.npy", np.array(self.omega_history))
            append_to_npy_file(f"{self.folder_path}/{self.count}_amplitude.npy", np.array(self.amplitude_history))
            append_to_npy_file(f"{self.folder_path}/{self.count}_phase.npy", np.array(self.phase_history))
            append_to_npy_file(f"{self.folder_path}/{self.count}_x_history.npy", np.array(self.x_history))
            append_to_npy_file(f"{self.folder_path}/{self.count}_z_history.npy", np.array(self.z_history))

            self.base_pos_buffer = []
            self.base_lin_vel_buffer = []
            self.base_ang_vel_buffer = []
            self.contact_boolean_buffer = []
            self.ee_pos_buffer = []
            self.power_buffer = []
            self.dof_pos_buffer = []
            self.command_buffer = []
            self.update_idx_buffer = []
            self.phase_history = []
            self.amplitude_history = []
            self.amplitude_dot_history = []
            self.mu_history = []
            self.omega_history = []
            self.x_history = []
            self.z_history = []
            self.count += 1
            if self.count == 5:
                print(np.mean(self.distance_buffer))
                print(np.std(self.distance_buffer))
                exit()
    
    def compute_reward(self, env_ids=None):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        low_rewards = self.cfg.rewards.low_rewards
        task_rewards = self.cfg.rewards.task_rewards
        self.rew_buf[:] = 0.
        self.low_rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        for low_name in low_rewards:
            i = self.reward_names.index(low_name)
            rew = self.reward_functions[i]() * self.reward_scales[low_name]
            self.low_rew_buf += rew

        track_max = 0.
        track_reward = torch.zeros_like(self.low_rew_buf)
        for task in task_rewards:
            i = self.reward_names.index(task)
            track_max += self.reward_scales[task]
            rew = self.reward_functions[i]() * self.reward_scales[task]
            track_reward += rew

        track_quality = torch.clamp(
            track_reward / (track_max + 1e-6),
            min=0.0,
            max=1.0,
        )
        self.behavior_quality = track_quality - torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        print(self.behavior_quality)

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.low_rew_buf[:] = torch.clip(self.low_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        return [self.rew_buf, self.low_rew_buf]


    #------------- Callbacks --------------


    

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.mp_modeling.ref_type == 'pace':
            low = 0
        elif self.cfg.mp_modeling.ref_type == 'trot':
            low = 1
        elif self.cfg.mp_modeling.ref_type == 'canter':
            low = 2
        self.motion_label[env_ids] = low
        if self.cfg.env.play:
            if self.episode_length_buf[0] < 245:
                self.commands[:, 0] = 2.
            else:
                self.commands[:, 0] = 2.
            self.commands[env_ids, 1] = -0.
        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            if self.cfg.env.play:
                self.commands[env_ids, 3] = 0.
            else:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            if self.cfg.env.play:
                self.commands[env_ids, 2] = 0.
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        return
   
    def _compute_torques(self, actions):
        """ Compute torques from mps Signals (solve Inverse kinematics and run PD Control)
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        des_joint_pos = torch.zeros_like(self.torques,device=self.device)
        xs,ys,zs = self.ID.intrinsic_dynamics(actions_scaled, self.commands[:, :3])
        
        #TODO
        if self.cfg.env.play:
            self.phase_history.append(self.ID.X[:, 1, :].detach().cpu().numpy())
            self.amplitude_history.append((torch.clip(self.ID.X[:, 0, :], self.ID.mu_low, self.ID.mu_up)).detach().cpu().numpy())
            self.amplitude_dot_history.append(self.ID.X_dot[:, 0, :].detach().cpu().numpy())
            self.mu_history.append(np.sqrt(self.ID._mu_x.detach().cpu().numpy()))
            self.omega_history.append(self.ID.X_dot[:, 1, :].detach().cpu().numpy())
            self.x_history.append(xs.detach().cpu().numpy())
            self.z_history.append(zs.detach().cpu().numpy())

        sideSign = np.array([-1,1,-1,1]) 
        foot_y = torch.ones(self.num_envs,device=self.device,requires_grad=False) * self.cfg.asset.hip_link_length_a1
        LEG_INDICES = np.array([1,0,3,2])       # FR-FL-RR-RL
        for ig_idx,i in enumerate(LEG_INDICES):
            x = xs[:,i]
            z = zs[:,i]
            y = sideSign[i] * foot_y  + ys[:,i]
            # y = ys[:,i]
            robot_length = self.cfg.asset
            des_joint_pos[:, 3*ig_idx:3*ig_idx+3] = self.ID.compute_inverse_kinematics(robot_length,i,x,y,z)
        self.dof_des_pos = des_joint_pos
        hip_factor = 1.
        # self.dof_des_pos[:,0] = self.default_dof_pos[:,0] + actions_scaled[:,8] * hip_factor
        # self.dof_des_pos[:,3] = self.default_dof_pos[:,3] + actions_scaled[:,9] * hip_factor
        # self.dof_des_pos[:,6] = self.default_dof_pos[:,6] + actions_scaled[:,10] * hip_factor
        # self.dof_des_pos[:,9] = self.default_dof_pos[:,9] + actions_scaled[:,11] * hip_factor

        torques = self.p_gains*(self.dof_des_pos - self.dof_pos) - self.d_gains*self.dof_vel 

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

        

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) 
        if self.cfg.env.play:
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, -max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs_history, self.num_obs, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) # [0,0,-1] ? 
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False) # always 12 motors... num_actions will changes based on space 
        self.dof_des_pos =torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False) # always 12 motors... num_actions will changes based on space       
        
        self.p_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_base_pos = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.first_iteartion= True
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]

        self.ID = Reference_Motion_Dynamics(time_step=self.sim_params.dt,num_envs=self.num_envs,device=self.device)
        oscillator_type = self.cfg.control.oscillator_type
        self.ID.oscillator = getattr(self.ID, oscillator_type)
        self.ID._MAX_STEP_LEN = self.cfg.asset.max_step_length
        self.ID._GROUND_CLEARANCE = self.cfg.asset.ground_clearance
        self.ID._GROUND_PENETRATION = self.cfg.asset.ground_penetration

        self.actions_slow = torch.zeros(self.num_envs, 8, device=self.device, dtype=torch.float)
        self.last_actions_slow = torch.zeros_like(self.actions_slow)

        self.actions_diff = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)

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
        self.motion_label = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.decimation = self.cfg.control.cycle * torch.ones(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

        self.imitation_task = ImitationTask(ref_motion_filenames=self.cfg.mp_modeling.ref_motion_filename, envs_frame_times=torch.full((self.num_envs,), self.dt, device=self.device), device=self.device)
        self.imitation_task.reset(self.num_envs, self.root_states[:, 0:3], self.dt)


        self.low_rew_buf = torch.zeros_like(self.rew_buf)

        self.behavior_quality = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)

        self.substep_dof_diff = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limit_abs = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._init_homeostasis_buffers()
 
    #------------ reward functions---------------
    def _reward_ref_motion(self):
        joint_ids = torch.tensor([1,2,4,5,7,8,10,11], device=self.device)
        end_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, :3]
        end_states[:, :, :2] -= self.root_states[:, None, :2]
        end_states = end_states.reshape(-1,12)
        rewards, _, _ = self.imitation_task.rewards_adaptive(self.dof_pos, self.dof_vel, self.root_states[:,0:3],\
                                                self.root_states[:,3:7], self.root_states[:,7:], end_states,\
                                                self.motion_label,
                                                joint_ids)                   #TODO
        return rewards
