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

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrains.blind_terrain import BlindTerrain
from legged_gym.utils.terrains.terrain_curriculum import TerrainCurriculum
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.dynamics.a1.equilibrium_dynamics import (
    EquilibriumPrimitiveDynamics, AdaptiveEquilibriumDynamics, ModalEquilibriumDynamics, bounded_inverse_kinematics,
)
from legged_gym.utils.gait_model import GaitModel, env_cfg

from legged_gym.utils.math_ import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict, append_to_npy_file

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
        self._validate_action_config()
        self.cfg.control.hierarchical = False
        self.cfg.control.varied_decimation = False
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.num_obs_history = 5
        self.num_modulation_params = self.cfg.env.num_modulation_params
        self.num_primitives = self.cfg.env.num_primitives
        self.num_ee_equilibriums = self.cfg.env.num_ee_equilibriums

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.folder_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'metrics', 'EXP_MP')
        self.count = 0
        self._playback_buffers = {
            'base_position': 'base_pos_buffer',
            'base_lin_vel': 'base_lin_vel_buffer',
            'base_ang_vel': 'base_ang_vel_buffer',
            'contact_boolean': 'contact_boolean_buffer',
            'ee_pos': 'ee_pos_buffer',
            'power': 'power_buffer',
            'dof_pos': 'dof_pos_buffer',
            'command': 'command_buffer',
            'update_idx': 'update_idx_buffer',
            'mu': 'mu_history',
            'omega': 'omega_history',
            'amplitude': 'amplitude_history',
            'phase': 'phase_history',
            'x_history': 'x_history',
            'z_history': 'z_history',
        }
        for attribute in self._playback_buffers.values():
            setattr(self, attribute, [])
        if self.cfg.env.play:
            os.makedirs(self.folder_path, exist_ok=True)

    def _validate_action_config(self):
        """Fail before creating a simulator when an old run config is loaded."""
        cfg = self.cfg
        controllers = {cls.ACTION_SPACE: cls for cls in
                       (EquilibriumPrimitiveDynamics, AdaptiveEquilibriumDynamics, ModalEquilibriumDynamics)}
        
        controller = controllers.get(getattr(cfg.env, 'action_space', None))
        if controller is None:
            raise ValueError(
                'MP_pro_terrain_a1 requires a supported equilibrium action_space (v1, v2 or modal v3). '
                'Old joint-residual configs/checkpoints are incompatible; start a new training run.')
        self._controller_class = controller
        if not np.isfinite(cfg.control.action_scale) or cfg.control.action_scale <= 0:
            raise ValueError('control.action_scale must be a positive finite scalar')
        eq = cfg.control.equilibrium
        if not np.isfinite(eq.joint_limit_margin) or eq.joint_limit_margin <= 0:
            raise ValueError('equilibrium.joint_limit_margin must be positive and finite')
        if not np.isfinite(eq.min_foot_depth) or eq.min_foot_depth <= 0:
            raise ValueError('equilibrium.min_foot_depth must be positive and finite')
        if not np.isfinite(eq.min_sagittal_reach) or eq.min_sagittal_reach <= 0:
            raise ValueError('equilibrium.min_sagittal_reach must be positive and finite')

    def _save_playback_metrics(self):
        for name, attribute in self._playback_buffers.items():
            samples = getattr(self, attribute)
            append_to_npy_file(os.path.join(self.folder_path, f"{self.count}_{name}.npy"), np.array(samples))
            samples.clear()
        self.count += 1

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions.copy_(self.ID.bounded_actions(actions.to(self.device), clip_actions))
        if self.terrain_curriculum is not None:
            self.terrain_curriculum.accumulate(self.commands, self.root_states[:, 3:7], self.dt)

        decimation = self.cfg.control.decimation * self.cfg.control.cycle
        self._slip_cost_integral.zero_()
        self._touchdown_cost.zero_()
        self._pi_g_contact_integral.zero_()

        if self.cfg.env.play:
            print(self.actions)
            print(self.commands)
            print(self.base_lin_vel)

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

            # prepare quantities
            self.base_quat[:] = self.root_states[:, 3:7]
            self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

            if self.cfg.env.play:
                self.base_pos_buffer.append(self.root_states[:, 0:3].cpu().numpy())
                self.base_lin_vel_buffer.append(self.base_lin_vel[:, 0:3].cpu().numpy())
                self.base_ang_vel_buffer.append(self.base_ang_vel[:, 0:3].cpu().numpy())
                contact_booleans = self.contact_forces[:, self.feet_indices, 2] > 0.
                self.contact_boolean_buffer.append(contact_booleans.cpu().numpy())
                self.ee_pos_buffer.append((self.rigid_body_state.view(self.num_envs, self.num_bodies, -1)[:, self.feet_indices, 0:3]).cpu().numpy())
                self.power_buffer.append(torch.sum(torch.abs(self.torques*self.dof_vel), dim=1).cpu().detach().numpy())
                self.dof_pos_buffer.append(self.dof_pos.cpu().numpy())
                self.command_buffer.append(self.commands.cpu().numpy())
            self.post_decimation_step(substep)
            self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, [self.rew_buf, self.primitive_selection_rew_buf, self.modulation_param_rew_buf, self.ee_equilibrium_rew_buf], self.reset_buf, self.extras

    def post_decimation_step(self, dec_i):
        self._update_contact_rewards()
        self._update_pi_g_metrics()
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel

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


        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.episode_length_buf += 1
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids, completed_episode=True)
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
        if self.cfg.terrain.mesh_type in ('heightfield', 'trimesh'):
            ground = self._get_terrain_heights(self.root_states[:, :2])
            half_size = self.root_states.new_tensor([self.terrain.env_length, self.terrain.env_width]) / 2
            outside = ((self.root_states[:, :2] - self.env_origins[:, :2]).abs()
                       > half_size - self.cfg.terrain.boundary_margin).any(dim=1)
            self.time_out_buf |= outside
        else:
            ground = 0.
        self.terrain_failure_buf = self.reset_buf | (self.root_states[:, 2] - ground < 0.15)
        # A fall at the boundary/time limit remains a failure, not a bootstrapped timeout.
        self.time_out_buf &= ~self.terrain_failure_buf
        self.reset_buf = self.terrain_failure_buf | self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf

    def reset_idx(self, env_ids, completed_episode=False):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        if self.terrain_curriculum is not None and completed_episode:
            self.terrain_curriculum.update(
                env_ids, self.root_states[env_ids, :2], self.terrain_failure_buf[env_ids],
                self.episode_length_buf[env_ids])
            self.env_origins[env_ids] = self.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        if self.terrain_curriculum is not None:
            self.terrain_curriculum.reset(env_ids, self.root_states[env_ids, :2])

        self.ID.reset(env_ids)
        self._resample_commands(env_ids)
        self._reset_homeostasis_buffers(env_ids)
        self._reset_pi_g_buffers(env_ids)

        # reset buffers
        self.last_base_pos[env_ids]= self.root_states[env_ids, 0]
        self.obs_buf[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.primitive_diff[env_ids] = 0.
        self.param_actions_diff[env_ids] = 0.
        self.ee_equilibriums_diff[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.behavior_quality[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        if self.terrain_curriculum is not None:
            self.extras["episode"]["terrain_level"] = self.terrain_levels.float().mean()
            self.extras["episode"]["terrain_difficulty"] = self.terrain_difficulties[
                self.terrain_levels, self.terrain_types].mean()
            self.extras["episode"]["terrain_curriculum_enabled"] = float(self.terrain_curriculum.enabled)
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # if self.cfg.env.play and self.time_out_buf[0] == 1:
        #     self._save_playback_metrics()

    def compute_reward(self, env_ids=None):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf.zero_()
        self.primitive_selection_rew_buf.zero_()
        self.modulation_param_rew_buf.zero_()
        self.ee_equilibrium_rew_buf.zero_()
        rewards = {}
        for name, function in zip(self.reward_names, self.reward_functions):
            reward = function() * self.reward_scales[name]
            rewards[name] = reward
            self.rew_buf += reward
            self.episode_sums[name] += reward

        for name, scale in self.primitive_selection_reward_scales.items():
            reward = getattr(self, "_reward_" + name)() * scale
            self.primitive_selection_rew_buf += reward
            key = 'selection_' + name if name in self.reward_scales else name
            self.episode_sums[key] += reward
        self.episode_sums["pi_g"] += self.primitive_selection_rew_buf

        for name in self.cfg.rewards.modulation_param_rewards:
            if name in rewards:
                self.modulation_param_rew_buf += rewards[name]

        for name in self.cfg.rewards.ee_equilibrium_rewards:
            if name in rewards:
                self.ee_equilibrium_rew_buf += rewards[name]

        track_reward = torch.zeros_like(self.rew_buf)
        track_max = 0.
        for name in self.cfg.rewards.task_rewards:
            if name in rewards:
                track_reward += rewards[name]
                track_max += self.reward_scales[name]

        track_quality = torch.clamp(
            track_reward / (track_max + 1e-6),
            min=0.0,
            max=1.0,
        )

        if self.cfg.env.play:
            self.behavior_quality = track_quality - torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
            self.behavior_quality += 0.2
            # print(self.behavior_quality)


        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.modulation_param_rew_buf[:] = torch.clip(self.modulation_param_rew_buf[:], min=0.)
            self.ee_equilibrium_rew_buf[:] = torch.clip(self.ee_equilibrium_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        return [self.rew_buf, self.primitive_selection_rew_buf, self.modulation_param_rew_buf, self.ee_equilibrium_rew_buf]

    def compute_observations(self):
        """ Computes observations
        """
        controller_obs = self.ID.controller_observations()
        self.obs_buf[:, :-1] = self.obs_buf[:, 1:].clone()
        self.obs_buf[:, -1,:-60] = torch.cat((
                                self.base_ang_vel * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.contact_forces[:, self.feet_indices, 2] > 1.,
                                self.actions[:,:-1],
                                self.commands[:, :3] * self.commands_scale,
                                # controller_obs,
                                ),dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf[:, -1,-60:] = heights

        # add noise if needed
        if self.add_noise:
            self.obs_buf[:, -1] += (2 * torch.rand_like(self.obs_buf[:, -1]) - 1) * self.noise_scale_vec

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:,:-60] = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.root_states[:, 3:7],
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.contact_forces[:, self.feet_indices, 2] > 1.,
                                    self.actions[:,:-1],
                                    self.commands[:, :3] * self.commands_scale,
                                    # controller_obs,
                                    ),dim=-1)

            if self.cfg.terrain.measure_heights:
                self.privileged_obs_buf[:,-60:] = heights

        return

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = BlindTerrain(self.cfg.terrain, self.num_envs)
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
            self.dof_pos_hard_limits = torch.zeros_like(self.dof_pos_limits)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_hard_limits[i] = self.dof_pos_limits[i]
                if (props["upper"][i] - props["lower"][i]
                        <= 2 * self.cfg.control.equilibrium.joint_limit_margin):
                    raise ValueError('equilibrium.joint_limit_margin leaves an empty joint range')
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

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if self.cfg.env.play:
            if self.episode_length_buf[0] < 245:
                self.commands[:, 0] = 1.5
            else:
                self.commands[:, 0] = 1.5
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
        actions_scaled = actions.clone()
        actions_scaled[:, 1:] *= self.cfg.control.action_scale

        xs,ys,zs = self.ID.intrinsic_dynamics(actions_scaled, self.commands[:, :3], contact_forces=torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
        if self.cfg.env.play:
            self.phase_history.append(self.ID.X[:, 2, :].detach().cpu().numpy().copy())
            self.amplitude_history.append((torch.clip(self.ID.X[:, 0, :], self.ID.mu_low, self.ID.mu_up)).detach().cpu().numpy())
            self.mu_history.append(np.sqrt(self.ID._mu_x.detach().cpu().numpy()))
            self.omega_history.append(self.ID.X_dot[:, 2, :].detach().cpu().numpy().copy())
            self.x_history.append(xs.detach().cpu().numpy())
            self.z_history.append(zs.detach().cpu().numpy())

        # All RL modulation happens inside the dynamical system before IK.
        # Targets include the signed anatomical hip offset, FL/FR/RL/RR.
        hip_y = ys.new_tensor([1., -1., 1., -1.]) * self.cfg.asset.hip_link_length_a1
        self.foot_targets_requested = torch.stack((xs, ys + hip_y, zs), dim=-1)
        eq = self.cfg.control.equilibrium
        self.dof_des_pos, self.foot_targets_reachable = bounded_inverse_kinematics(
            self.foot_targets_requested, self.cfg.asset, self.dof_pos_hard_limits,
            eq.joint_limit_margin, eq.min_foot_depth, eq.min_sagittal_reach)

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
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] = self._spawn_positions(env_ids)
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
        if self.cfg.env.play:
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, -max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0,0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # Contact flags, actions, commands and controller states are not noisy.
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
        if self.terrain_curriculum is not None:
            self.terrain_curriculum.reset(
                torch.arange(self.num_envs, device=self.device), self.root_states[:, :2])
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
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]

        controller_kwargs = {}
        if issubclass(self._controller_class, AdaptiveEquilibriumDynamics):
            controller_kwargs['timing_cfg'] = self.cfg.control.timing
        self.ID = self._controller_class(
            equilibrium_cfg=self.cfg.control.equilibrium,
            time_step=self.sim_params.dt, num_envs=self.num_envs, device=self.device,
            nominal_clearance=self.cfg.asset.ground_clearance,
            max_clearance=self.cfg.asset.max_ground_clearance, **controller_kwargs)
        oscillator_type = self.cfg.control.oscillator_type
        self.ID.oscillator = getattr(self.ID, oscillator_type)
        self.ID._MAX_STEP_LEN = self.cfg.asset.max_step_length
        self.ID._GROUND_CLEARANCE = self.cfg.asset.ground_clearance
        self.ID._GROUND_PENETRATION = self.cfg.asset.ground_penetration


        self.param_actions_diff = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)
        self.primitive_diff = torch.zeros_like(self.param_actions_diff)
        self.ee_equilibriums_diff = torch.zeros_like(self.param_actions_diff)

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
        self.decimation = self.cfg.control.cycle * torch.ones(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

        self.modulation_param_rew_buf = torch.zeros_like(self.rew_buf)
        self.primitive_selection_rew_buf = torch.zeros_like(self.rew_buf)
        self.ee_equilibrium_rew_buf = torch.zeros_like(self.rew_buf)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.behavior_quality = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)

        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.cycle * self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._init_homeostasis_buffers()
        self._init_pi_g_buffers()

    def _init_homeostasis_buffers(self):
        """Allocate substep contact history (no simulator handles)."""
        cfg = self.cfg.rewards
        if min(cfg.touchdown_velocity_scale, cfg.touchdown_force_scale) <= 0:
            raise ValueError('Touchdown scales must be positive')
        if not 0 <= cfg.contact_off_force < cfg.contact_on_force:
            raise ValueError('Contact thresholds must be ordered')
        if min(cfg.touchdown_min_air_time, cfg.touchdown_force_window,
               cfg.touchdown_force_weight) < 0:
            raise ValueError('Touchdown windows and force weight must be nonnegative')
        n, feet = self.num_envs, len(self.feet_indices)
        self._slip_cost_integral = torch.zeros(n, device=self.device)
        self._touchdown_cost = torch.zeros(n, device=self.device)
        self._contact_initialized = torch.zeros(n, device=self.device, dtype=torch.bool)
        self._stance_contact = torch.zeros(n, feet, device=self.device, dtype=torch.bool)
        self._foot_air_time = torch.zeros(n, feet, device=self.device)
        self._previous_foot_velocity = torch.zeros(n, feet, 3, device=self.device)
        self._touchdown_window_left = torch.zeros(n, feet, device=self.device)
        self._touchdown_peak_cost = torch.zeros(n, feet, device=self.device)

    def _reset_homeostasis_buffers(self, env_ids):
        """Invalidate only reset environments; never compare across episodes."""
        self._slip_cost_integral[env_ids] = 0.
        self._touchdown_cost[env_ids] = 0.
        self._contact_initialized[env_ids] = False
        self._stance_contact[env_ids] = False
        self._foot_air_time[env_ids] = 0.
        self._previous_foot_velocity[env_ids] = 0.
        self._touchdown_window_left[env_ids] = 0.
        self._touchdown_peak_cost[env_ids] = 0.

    def _init_pi_g_buffers(self):
        self._pi_g_contact_integral = torch.zeros(self.num_envs, device=self.device)
        self._pi_g_previous_energy = torch.zeros_like(self._pi_g_contact_integral)
        self._pi_g_previous_phase = torch.zeros_like(self._pi_g_contact_integral)
        self._pi_g_previous_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pi_g_work_sum = torch.zeros_like(self._pi_g_contact_integral)
        self._pi_g_work = torch.zeros_like(self._pi_g_contact_integral)
        self._pi_g_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pi_g_full_cycle = torch.zeros_like(self._pi_g_initialized)

    def _reset_pi_g_buffers(self, env_ids):
        self._pi_g_initialized[env_ids] = False
        self._pi_g_full_cycle[env_ids] = False
        self._pi_g_work_sum[env_ids] = 0.
        self._pi_g_work[env_ids] = 0.
        self._pi_g_contact_integral[env_ids] = 0.

    def _update_contact_rewards(self):
        """Integrate stance slip and capture touchdown speed/force at physics rate.

        Assumes a static horizontal ground: force z is the normal load and world
        velocity xy is tangential velocity. Startup contact is not a touchdown.
        """
        cfg, dt = self.cfg.rewards, self.sim_params.dt
        feet = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        velocity = feet[:, :, 7:10]
        force = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.)
        contact = (force > cfg.contact_on_force) | (self._stance_contact & (force > cfg.contact_off_force))
        touchdown = (contact & ~self._stance_contact & self._contact_initialized[:, None]
                     & (self._foot_air_time >= cfg.touchdown_min_air_time))
        self._slip_cost_integral += (velocity[:, :, :2].square().sum(dim=-1) * contact).sum(dim=-1) * dt
        descent = (-self._previous_foot_velocity[:, :, 2]).clamp_min(0.) / cfg.touchdown_velocity_scale
        self._touchdown_cost += (descent.square() * touchdown).sum(dim=-1)

        self._touchdown_window_left = torch.where(
            touchdown, torch.full_like(force, cfg.touchdown_force_window), self._touchdown_window_left)
        self._touchdown_peak_cost[touchdown] = 0.
        active = contact & ((self._touchdown_window_left > 0.) | touchdown)
        force_cost = ((force - cfg.max_contact_force).clamp_min(0.) / cfg.touchdown_force_scale).square()
        peak_increment = (force_cost - self._touchdown_peak_cost).clamp_min(0.) * active
        self._touchdown_cost += cfg.touchdown_force_weight * peak_increment.sum(dim=-1)
        self._touchdown_peak_cost += peak_increment
        self._touchdown_window_left = (self._touchdown_window_left - dt).clamp_min(0.)
        self._foot_air_time = torch.where(contact, torch.zeros_like(force), self._foot_air_time + dt)
        self._stance_contact.copy_(contact)
        self._previous_foot_velocity.copy_(velocity)
        self._contact_initialized[:] = True

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

        self.primitive_selection_reward_scales = {
            name: getattr(self.cfg.rewards.primitive_selection_scales, name) * self.dt
            for name in self.cfg.rewards.primitive_selection_rewards
            if getattr(self.cfg.rewards.primitive_selection_scales, name) != 0
        }
        for name in self.primitive_selection_reward_scales:
            getattr(self, "_reward_" + name)  # Fail early for an invalid configured term.
            key = 'selection_' + name if name in self.reward_scales else name
            self.episode_sums[key] = torch.zeros_like(self.rew_buf)
        self.episode_sums["pi_g"] = torch.zeros_like(self.rew_buf)

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
        hf_params = gymapi.HeightFieldParams()
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

        # Isaac Gym expects a flat column-major height array.
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.ravel(order="F"), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
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
        # Cache physical properties after Isaac Gym recomputes randomized inertia.
        self.body_mass = torch.zeros(self.num_envs, self.num_bodies, device=self.device)
        self.body_inertia = torch.zeros(self.num_envs, self.num_bodies, 3, 3, device=self.device)
        self.actor_handles = []
        self.envs = []
        spawn_positions = self._spawn_positions(torch.arange(self.num_envs, device=self.device))
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = spawn_positions[i]
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            self.body_mass[i] = torch.tensor([b.mass for b in body_props], device=self.device)
            self.body_inertia[i] = torch.tensor([
                [[getattr(getattr(b.inertia, row), col) for col in 'xyz']
                 for row in 'xyz'] for b in body_props], device=self.device)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
        self.robot_mass = self.body_mass.sum(dim=1)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _spawn_positions(self, env_ids):
        """Use the central maximum height, following the official terrain task reset."""
        positions = self.env_origins[env_ids].clone() + self.base_init_state[:3]
        if self.cfg.terrain.mesh_type in ('heightfield', 'trimesh'):
            jitter = self.cfg.terrain.spawn_xy_jitter
            positions[:, :2] += (2 * torch.rand(len(env_ids), 2, device=self.device) - 1) * jitter
        return positions

    def _get_terrain_heights(self, xy):
        """Conservative local ground estimate, following legged_gym height sampling.

        Official trimesh conversion can move vertices horizontally to form
        vertical walls, so regular-grid triangle interpolation is no longer
        exact. Use neighbouring minimum heights for termination only.
        Never exposed to policies.
        """
        coordinates = (xy + self.terrain.border_size) / self.terrain.horizontal_scale
        indices = coordinates.long()
        i = indices[:, 0].clamp(0, self.terrain.tot_rows - 2)
        j = indices[:, 1].clamp(0, self.terrain.tot_cols - 2)
        height = torch.minimum(self.height_samples[i, j], self.height_samples[i + 1, j])
        height = torch.minimum(height, self.height_samples[i, j + 1])
        return height.float() * self.terrain.vertical_scale

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.terrain_curriculum = None
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.terrain_curriculum = TerrainCurriculum(
                self.cfg.terrain, self.terrain.difficulties, self.num_envs, self.device,
                play=self.cfg.env.play)
            self.terrain_levels = self.terrain_curriculum.levels
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.terrain_difficulties = torch.as_tensor(
                self.terrain.difficulties, device=self.device, dtype=torch.float)
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

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

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

            if env_ids:
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
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_yaw(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, 2:3]-(-1)), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate_ps(self):
        tmp = torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        # print(tmp)
        return self.primitive_diff

    def _reward_action_rate_mp(self):
        return self.param_actions_diff

    def _reward_action_rate_eee(self):
        return self.ee_equilibriums_diff

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).mean(dim= 1)

    def _reward_stance_slip(self):
        """Time-averaged sum of stance-foot world xy speeds squared, (m/s)^2."""
        return self._slip_cost_integral / self.dt

    def _reward_touchdown_impact(self):
        """Landing event cost / dt, cancelling the reward scale's dt multiplier."""
        return self._touchdown_cost / self.dt

    def _pi_g_energy(self):
        """E_k - E_p in joules, summing rigid-body translation and rotation.

        Eq. (13) leaves the inertia model unspecified; use the simulator's
        per-link masses/inertias, including randomized mass, for this task.
        """
        bodies = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        omega = quat_rotate_inverse(bodies[:, :, 3:7].reshape(-1, 4),
                                    bodies[:, :, 10:13].reshape(-1, 3))
        omega = omega.view(self.num_envs, self.num_bodies, 3)
        kinetic = .5 * (self.body_mass * bodies[:, :, 7:10].square().sum(dim=-1)).sum(dim=1)
        kinetic += .5 * torch.einsum('nbi,nbij,nbj->n', omega, self.body_inertia, omega)
        potential = (self.body_mass * bodies[:, :, 2]).sum(dim=1) * abs(self.sim_params.gravity.z)
        return kinetic - potential

    def _update_pi_g_metrics(self):
        """Sample contact mismatch and signed cycle work (paper Eq. 13)."""
        gait_id = self.actions[:, 0].long()
        phase = self.ID.phi[:, 0]
        reference_phase = (self.ID.phi + self.ID.gait_phase_delta).remainder(2 * np.pi)
        duty = self.ID.selected_duty_factors(gait_id)
        reference_contact = reference_phase >= 2 * np.pi * (1 - duty)
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.pi_g_contact_force
        mismatch = (contact != reference_contact).float().mean(dim=1)
        self._pi_g_contact_integral += mismatch * self.sim_params.dt

        energy = self._pi_g_energy()
        continuous = self._pi_g_initialized & (gait_id == self._pi_g_previous_id)
        self._pi_g_work_sum[~continuous] = 0.
        self._pi_g_work[~continuous] = 0.
        self._pi_g_full_cycle[~continuous] = False
        self._pi_g_work_sum += torch.where(continuous, energy - self._pi_g_previous_energy, 0.)
        wrapped = continuous & (phase < self._pi_g_previous_phase)
        complete = wrapped & self._pi_g_full_cycle
        self._pi_g_work[complete] = self._pi_g_work_sum[complete]
        self._pi_g_work_sum[wrapped] = 0.
        self._pi_g_full_cycle[wrapped] = True
        self._pi_g_previous_energy.copy_(energy)
        self._pi_g_previous_phase.copy_(phase)
        self._pi_g_previous_id.copy_(gait_id)
        self._pi_g_initialized[:] = True

    @staticmethod
    def _pi_g_psi(value):
        return 1. - torch.tanh(value.square())

    def _pi_g_cost_of_transport(self):
        """Transport cost is inapplicable at zero commanded speed."""
        command_speed = torch.linalg.vector_norm(self.commands[:, :3], dim=1)
        power = (self.substep_torques * self.substep_dof_vel
                 + .3 * self.substep_torques.square()).clamp_min(0.).sum(dim=-1).mean(dim=1)
        denominator = self.robot_mass * abs(self.sim_params.gravity.z) * command_speed
        return torch.where(command_speed > 0., power / denominator.clamp_min(1.e-8), 0.)

    def _pi_g_torque_fraction(self):
        return (self.substep_torques.abs() / self.torque_limits).mean(dim=(1, 2))

    def _pi_g_motion_costs(self):
        """Unshaped transport, torque, contact and signed cycle-work costs."""
        return (self._pi_g_cost_of_transport(), self._pi_g_torque_fraction(),
                self._pi_g_contact_integral / self.dt, self._pi_g_work)

    def _reward_pi_g_utility(self):
        """Tracking utility and motion-dependent gait-switch penalty."""
        velocity_error = (self.base_lin_vel[:, :2] - self.commands[:, :2]).square().sum(dim=1)
        velocity_error += (self.base_ang_vel[:, 2] - self.commands[:, 2]).square()
        switched = (self.actions[:, 0] != self.last_actions[:, 0]) & (self.episode_length_buf > 1)
        cot, torque, contact, work = self._pi_g_motion_costs()
        smooth = -self.cfg.rewards.pi_g_switch_cost * switched.float() * self._pi_g_psi(cot + torque + contact + work)
        return self.cfg.rewards.pi_g_utility_weight * (self._pi_g_psi(velocity_error) + smooth)

    def _reward_pi_g_cot(self):
        return self._pi_g_psi(self._pi_g_cost_of_transport())

    def _reward_pi_g_torque(self):
        return self._pi_g_psi(self._pi_g_torque_fraction())

    def _reward_pi_g_contact(self):
        return self._pi_g_psi(self._pi_g_contact_integral / self.dt)

    def _reward_pi_g_work(self):
        return self._pi_g_psi(self._pi_g_work)

    def _reward_pi_g(self):
        """Unscaled legacy aggregate; no standing primitive is defined."""
        return (self._reward_pi_g_utility() + self._reward_pi_g_cot()
                + self._reward_pi_g_torque() + self._reward_pi_g_contact()
                + self._reward_pi_g_work())

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces

        tmp = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

        # print(tmp)
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)