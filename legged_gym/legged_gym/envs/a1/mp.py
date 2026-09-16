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

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.dynamics.a1.dynamics import Intrinsic_Dynamics
from legged_gym.utils.imitation_task import ImitationTask
from legged_gym.utils.gait_model import GaitModel, env_cfg

from legged_gym.utils.math_ import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict

from legged_gym.utils import task_registry

class MotorPrimitives(BaseTask):
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
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # filename = [f'/home/ubuntu/amortizedMPs/legged_gym/logs/A1/0701_18-13-45_A1MP_trot_slow_omegaUF-1.25e+01_muUF-1.25e+00_MF-1e+03_noResume/model_450.pt']
        self.import_gait_model(self.cfg.mp_modeling.gait_model_filename)
        # self.import_gait_model(filename)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.phase_history = []
        self.amplitude_history = []
        self.amplitude_dot_history = []
        self.mu_history = []
        self.omega_history = []
        self.x_history = []
        self.z_history = []
        self.x_act_history = []
        self.z_act_history = []
            


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        dt = self.decimation * 0.005
        self.imitation_task.update_(self.episode_length_buf, self.commands[:,0], self.motion_label, dt)        

        # if self.cfg.control.cycle == 1:
        #     ft_env_ids = torch.nonzero(self.episode_length_buf % 4 == 0, as_tuple=False).squeeze(-1).to(torch.int)
        #     self.actions_slow[ft_env_ids] = torch.clip(self.gait_models[0](self.obs_buf[ft_env_ids].detach()), -clip_actions, clip_actions).to(self.device)
        #     self.actions[:] = torch.clip(self.actions_slow + actions, -clip_actions, clip_actions).to(self.device)
        # elif self.cfg.control.cycle == 2:
        #     ft_env_ids = torch.nonzero(self.episode_length_buf % 2 == 0, as_tuple=False).squeeze(-1).to(torch.int)
        #     self.actions_slow[ft_env_ids] = torch.clip(self.gait_models[0](self.obs_buf[ft_env_ids].detach()), -clip_actions, clip_actions).to(self.device)
        #     self.actions[:] = torch.clip(self.actions_slow + actions, -clip_actions, clip_actions).to(self.device)
        # # elif self.cfg.control.cycle == 4:
        # else:
        self.actions[:] = torch.clip(actions, -clip_actions, clip_actions).to(self.device) 
        # period = 4
        # ft_env_ids = torch.nonzero(self.episode_length_buf % period == 0, as_tuple=False).squeeze(-1).to(torch.int)
        # self.last_actions_slow[ft_env_ids,4:] = self.actions_slow[ft_env_ids, 4:]
        # self.actions_slow[ft_env_ids] = torch.clip(self.gait_models[0](self.obs_buf[ft_env_ids].detach()), -clip_actions, clip_actions).to(self.device)
        # ft_env_ids = torch.nonzero(self.episode_length_buf % (5 * period) != 0, as_tuple=False).squeeze(-1).to(torch.int)
        # self.actions_slow[ft_env_ids,:4] = self.last_actions_slow[ft_env_ids, :4]
        # ft_env_ids = torch.nonzero(self.episode_length_buf % (5 * period) == 0, as_tuple=False).squeeze(-1).to(torch.int)
        # self.last_actions_slow[ft_env_ids, :4] = self.actions_slow[ft_env_ids, :4]
        
        # self.actions[:] = torch.clip(self.actions_slow + actions, -clip_actions, clip_actions).to(self.device)

        decimation = self.cfg.control.decimation * self.cfg.control.cycle

        if self.cfg.env.play:
            print('---')
            print(self.commands)
            print(self.base_lin_vel)

        ft_env_ids = torch.nonzero(self.episode_length_buf % self.cfg.control.mu_cycle != 0, as_tuple=False).squeeze(-1).to(torch.int)
        self.actions[ft_env_ids,:4] = self.last_actions[ft_env_ids, :4]

        # if self.cfg.env.play:
            # print('---')
        #     print(self.commands)
        #     print(self.base_lin_vel)
            # print(self.actions_slow)
            # print(self.last_actions_slow)
            # print(actions)
            # print(self.actions)

        for i in range(decimation):        
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            if self.cfg.env.play:
                self.x_act_history.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 0]).detach().cpu().clone())
                self.z_act_history.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 2]).detach().cpu().clone())

            self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

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

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
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

        self._resample_commands(env_ids)
        self.ID.reset(env_ids, self.motion_label)

        # reset buffers
        self.last_base_pos[env_ids]= 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
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

            phase_hist = torch.stack(self.phase_history, dim=0)
            amp_hist = torch.stack(self.amplitude_history, dim=0)
            amp_dot_hist = torch.stack(self.amplitude_dot_history, dim=0)
            mu_hist = torch.stack(self.mu_history, dim=0)
            omega_hist = torch.stack(self.omega_history, dim=0)
            x_hist = torch.stack(self.x_history, dim=0)
            z_hist = torch.stack(self.z_history, dim=0)
            x_act_hist = torch.stack(self.x_act_history, dim=0)
            z_act_hist = torch.stack(self.z_act_history, dim=0)

            env_id = 0
            leg_id = 0

            phase_np = phase_hist[:, env_id, leg_id].numpy()
            amp_np = amp_hist[:, env_id, leg_id].numpy()
            x_np = x_hist[:, env_id, leg_id].numpy()
            z_np = z_hist[:, env_id, leg_id].numpy()
            x_act_np = x_act_hist[:, env_id, leg_id].numpy()
            z_act_np = z_act_hist[:, env_id, leg_id].numpy()

            # 构造线段: shape = [T-1, 2, 2]
            x = amp_np * np.cos(phase_np)
            y = amp_np * np.sin(phase_np)
            breakpoint()
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 用时间步作为颜色
            time_idx = np.arange(len(x) - 1)
            fig, axs = plt.subplots(3,1)
            ax = axs[0]

            lc = LineCollection(
                segments,
                cmap="turbo",
                linewidth=1.2,
            )

            lc.set_array(time_idx)
            lc.set_clim(time_idx.min(), time_idx.max())

            ax.add_collection(lc)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Oscillator trajectory, env={env_id}, leg={leg_id}")
            ax.grid(True)

            # 自动调整坐标范围
            margin = 0.05
            x_range = x.max() - x.min()
            z_range = y.max() - y.min()

            ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
            ax.set_ylim(y.min() - margin * z_range, y.max() + margin * z_range)

            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("Time step")

            # 构造线段: shape = [T-1, 2, 2]
            points = np.array([x_np, z_np]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 用时间步作为颜色
            time_idx = np.arange(len(x_np) - 1)

            ax = axs[1]

            lc = LineCollection(
                segments,
                cmap="turbo",
                linewidth=1.2,
            )

            lc.set_array(time_idx)
            lc.set_clim(time_idx.min(), time_idx.max())

            ax.add_collection(lc)

            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Oscillator trajectory, env={env_id}, leg={leg_id}")
            ax.grid(True)

            # 自动调整坐标范围
            margin = 0.05
            x_range = x_np.max() - x_np.min()
            z_range = z_np.max() - z_np.min()

            ax.set_xlim(x_np.min() - margin * x_range, x_np.max() + margin * x_range)
            ax.set_ylim(z_np.min() - margin * z_range, z_np.max() + margin * z_range)

            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("Time step")

            # 构造线段: shape = [T-1, 2, 2]
            points = np.array([x_act_np, z_act_np]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 用时间步作为颜色
            time_idx = np.arange(len(x_act_np) - 1)

            ax = axs[2]

            lc = LineCollection(
                segments,
                cmap="turbo",
                linewidth=1.2,
            )

            lc.set_array(time_idx)
            lc.set_clim(time_idx.min(), time_idx.max())

            ax.add_collection(lc)

            ax.set_xlabel("x_act")
            ax.set_ylabel("z_act")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Oscillator trajectory, env={env_id}, leg={leg_id}")
            ax.grid(True)

            # 自动调整坐标范围
            margin = 0.05
            x_range = x_act_np.max() - x_act_np.min()
            z_range = z_act_np.max() - z_act_np.min()

            ax.set_xlim(x_act_np.min() - margin * x_range, x_act_np.max() + margin * x_range)
            ax.set_ylim(z_act_np.min() - margin * z_range, z_act_np.max() + margin * z_range)

            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("Time step")
            
            plt.show()

            self.phase_history = []
            self.amplitude_history = []
            self.amplitude_dot_history = []
            self.mu_history = []
            self.omega_history = []
            self.x_history = []
            self.z_history = []
    
    def compute_reward(self, env_ids=None):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.root_states[:, 3:7],
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.contact_forces[:, self.feet_indices, 2] > 1.,
                                self.commands[:, :3] * self.commands_scale,
                                ),dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        return

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

                props["driveMode"][i] = gymapi.DOF_MODE_POS
                props["stiffness"][i] = 0.0
                props["damping"][i] = 0.0

        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            # props[0].mass += mass[env_id % 11]
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

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
        random_indices = torch.randint(low=low, high=low+1, size=(len(env_ids),), device=self.device)
        self.motion_label[env_ids] = random_indices.int()

        if self.cfg.env.play:
            self.commands[env_ids, 0] = 2.
        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
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
        xs,ys,zs = self.ID.intrinsic_dynamics(actions_scaled, self.commands[:,0])
        
        #TODO
        if self.cfg.env.play:
            self.phase_history.append(self.ID.X[:, 1, :].detach().cpu().clone())
            self.amplitude_history.append((torch.clip(self.ID.X[:, 0, :], self.ID.mu_low, self.ID.mu_up)).detach().cpu().clone())
            self.amplitude_dot_history.append(self.ID.X_dot[:, 0, :].detach().cpu().clone())
            self.mu_history.append(self.ID._mu.detach().cpu().clone())
            self.omega_history.append(self.ID.phases.detach().cpu().clone())
            self.x_history.append(xs.detach().cpu().clone())
            self.z_history.append(zs.detach().cpu().clone())

        sideSign = np.array([-1,1,-1,1]) 
        foot_y = torch.ones(self.num_envs,device=self.device,requires_grad=False) * self.cfg.asset.hip_link_length_a1
        LEG_INDICES = np.array([1,0,3,2])       # FR-FL-RR-RL
        for ig_idx,i in enumerate(LEG_INDICES):
            x = xs[:,i]
            z = zs[:,i]
            y = sideSign[i] * foot_y  + ys[:,i]
            robot_length = self.cfg.asset
            des_joint_pos[:, 3*ig_idx:3*ig_idx+3] = self.ID.compute_inverse_kinematics(robot_length,i,x,y,z)
        self.dof_des_pos = des_joint_pos
        # hip_factor = .5
        # self.dof_des_pos[:,0] = self.default_dof_pos[:,0] + actions_scaled[:,8] * hip_factor
        # self.dof_des_pos[:,3] = self.default_dof_pos[:,3] + actions_scaled[:,9] * hip_factor
        # self.dof_des_pos[:,6] = self.default_dof_pos[:,6] + actions_scaled[:,10] * hip_factor
        # self.dof_des_pos[:,9] = self.default_dof_pos[:,9] + actions_scaled[:,11] * hip_factor
        
        torques = self.p_gains*(self.dof_des_pos - self.dof_pos) - self.d_gains*self.dof_vel 

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
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) 
        # self.root_states[:, 7] = torch_rand_float(-max_vel, -max_vel, (self.num_envs, 1), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:] = 0. # previous actions
        return noise_vec

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
 
        self.ID = Intrinsic_Dynamics(time_step=self.sim_params.dt,num_envs=self.num_envs,device=self.device)
        self.actions_slow = torch.zeros(self.num_envs, 8, device=self.device, dtype=torch.float)
        self.last_actions_slow = torch.zeros_like(self.actions_slow)


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
                self.reward_scales[key] *= self.dt
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

    def import_gait_model(self, paths):
        self.gait_models = []
        _, train_cfg = task_registry.get_cfgs('MP_a1')
        train_cfg_dict = class_to_dict(train_cfg)
        cfg = env_cfg(num_envs=self.num_envs, num_obs=44, num_privileged_obs=None, num_actions=8)

        for path in paths:
            model = GaitModel(train_cfg_dict, cfg, None, self.device)
            model.load(path,False)
            policy = model.get_inference_policy(device=self.device)
            self.gait_models.append(policy)

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
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

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
            max_init_level = self.cfg.terrain.max_init_terrain_level
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
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

    #------------ reward functions---------------    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.abs(self.torques*self.dof_vel), dim=1)
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_ref_motion(self):
        end_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, :3].reshape(-1,12)
        rewards, _, _ = self.imitation_task.rewards(self.dof_pos, self.dof_vel, self.root_states[:,0:3],\
                                                self.root_states[:,3:7], self.root_states[:,7:], end_states,\
                                                self.motion_label)                   #TODO
        return rewards
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
