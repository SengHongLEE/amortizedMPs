# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import logging
from legged_gym.utils import motion_data


class ImitationTask(object):
  """Imitation reference motion task."""

  def __init__(self,
               weight=1.,
               ref_motion_filenames=None,
               pose_weight=1.0,       #0.1, 0.5, 0.2
               velocity_weight=0.0,    #0.05
               end_effector_weight=0.0,  #0.2
               root_pose_weight=0.,    #0.15
               root_velocity_weight=0.,  #0.1
               pose_err_scale=5.0,      #5.0
               velocity_err_scale=0.1,   #0.5, 0.1
               end_effector_err_scale=40.0,   #40.0
               root_pose_err_scale=20.0,    #40.0 , 20.0
               root_velocity_err_scale=2.0, #0.5, 2.0
               envs_frame_times=None,
               device=None):
    """Initializes the task.

    Args:
      weight: Float. The scaling factor for the reward.
      ref_motion_filenames: List of files containing reference motion data.
      pose_weight: Pose reward weight.
      velocity_weight: Velocity reward weight.
      end_effector_weight: End effector reward weight.
      root_pose_weight: Root position and rotation reward weight.
      root_velocity_weight: Root linear and angular velocity reward weight.
      pose_err_scale: Pose error scale for calculating pose reward.
      velocity_err_scale: Velocity error scale for calculating velocity reward.
      end_effector_err_scale: End effector error scale for calculating end
        effector reward.
      end_effector_height_err_scale: End effector height error scale for
        calculating the end effector reward.
      root_pose_err_scale: Root position and rotation error scale for
        calculating root position and rotation reward.
      root_velocity_err_scale: Root linear and angular velocity error scale for
        calculating root linear and angular velocity reward.
      envs_frame_times: Timestep of each agent
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # 输出到终端
        ]
    )
    assert ref_motion_filenames is not None
    self._ref_motion_filenames = ref_motion_filenames
    self._ref_motions = None

    self._default_pose = None
    self._dof_order = None
    self._ee_order = None

    # reward function parameters
    self._weight = weight
    self._pose_weight = pose_weight
    self._velocity_weight = velocity_weight
    self._end_effector_weight = end_effector_weight
    self._root_pose_weight = root_pose_weight
    self._root_velocity_weight = root_velocity_weight

    self._pose_err_scale = pose_err_scale
    self._velocity_err_scale = velocity_err_scale
    self._end_effector_err_scale = end_effector_err_scale
    self._root_pose_err_scale = root_pose_err_scale
    self._root_velocity_err_scale = root_velocity_err_scale

    self.envs_frame_times = envs_frame_times
    self.device = device

  def reset(self, env_nums, initial_root_pose, time_step):
    """Resets the internal state of the task."""
    self.env_nums = env_nums
    self.time_step = time_step
    if (self._ref_motions is None):
      self._ref_motions = self._load_ref_motions(self._ref_motion_filenames)

    if self._default_pose is None:
      self._default_pose = initial_root_pose.to(self.device) if self.device is not None else initial_root_pose

    if self._dof_order is None:
      self._dof_order = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=self.device)
    if self._ee_order is None:
      self._ee_order = torch.tensor([6, 7, 8, 0, 1, 2, 9, 10, 11, 3, 4, 5], device=self.device)
    return
  
  def update_(self, progress_buf, command_x, motion_label, time_step=None):
    self.progress_buf = progress_buf
    self.command_x = command_x
    self.motion_label = motion_label
    if time_step is not None:
      self.time_step = time_step

    indices_pace = torch.nonzero(self.motion_label == 0, as_tuple=False).squeeze(-1).int()
    self.envs_frame_times[indices_pace] = (0.75 * 0.02) / (self.command_x[indices_pace])

    indices_trot = torch.nonzero(self.motion_label == 1, as_tuple=False).squeeze(-1).int()
    self.envs_frame_times[indices_trot] = (1.21 * 0.02) / (self.command_x[indices_trot])

    indices_canter = torch.nonzero(self.motion_label == 2, as_tuple=False).squeeze(-1).int()
    self.envs_frame_times[indices_canter] = (2.21 * 0.02) / (self.command_x[indices_canter])
    
    return

  def _reorder_dof(self, dof_tensor):
    return dof_tensor.index_select(1, self._dof_order)

  def _reorder_ee(self, ee_tensor):
    return ee_tensor.index_select(1, self._ee_order)

  def rewards(self, dof_pos, dof_vel, torso_position, torso_rotation, root_velocities, end_state, motion_label):
    reward = torch.zeros(self.env_nums, device=self.device, requires_grad=False)
    dof_pos_ref_all = torch.zeros_like(dof_pos)
    root_pose_ref_all = torch.zeros_like(torso_position)
    time = self._get_motion_time()
    for i in range(len(self._ref_motions)):
      mask = (motion_label == i)
      if not mask.any().item():
        continue
      motion = self.get_active_motion(i)
      indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).int()
      frame = motion.calc_frame(time[indices], self.envs_frame_times[indices])
      vel = motion.calc_frame_vel(time[indices], self.envs_frame_times[indices])
      pose_reward, dof_pos_ref = self.reward_dof_pos(indices, frame, dof_pos)
      vel_reward = self.reward_dof_vel(indices, vel, dof_vel)
      root_pose_reward, root_pose_ref = self.reward_root_state(indices, frame, torso_position, torso_rotation)
      root_velocity_reward = self.reward_root_vel(indices, vel, root_velocities)
      end_effector_reward = self.reward_ee_state(indices, frame, end_state)

      reward_tmp = self._pose_weight * pose_reward + self._root_pose_weight * root_pose_reward + \
                self._velocity_weight * vel_reward + self._root_velocity_weight * root_velocity_reward + \
                self._end_effector_weight * end_effector_reward

      reward[indices] = reward_tmp.float()
      dof_pos_ref_all[indices] = dof_pos_ref.to(dof_pos_ref_all.dtype)
      root_pose_ref_all[indices] = root_pose_ref.to(root_pose_ref_all.dtype)

    return reward * self._weight, dof_pos_ref_all, root_pose_ref_all

  def rewards_adaptive(self, dof_pos, dof_vel, torso_position, torso_rotation, root_velocities, end_state, motion_label, joint_ids):
    reward = torch.zeros(self.env_nums, device=self.device, requires_grad=False)
    dof_pos_ref_all = torch.zeros_like(dof_pos)
    root_pose_ref_all = torch.zeros_like(torso_position)
    time = self._get_motion_time()
    for i in range(len(self._ref_motions)):
      mask = (motion_label == i)
      if not mask.any().item():
        continue
      motion = self.get_active_motion(i)
      indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).int()
      frame = motion.calc_frame(time[indices], self.envs_frame_times[indices])
      vel = motion.calc_frame_vel(time[indices], self.envs_frame_times[indices])
      pose_reward, dof_pos_ref = self.reward_dof_pos_adaptive(indices, frame, dof_pos, joint_ids)
      vel_reward = self.reward_dof_vel(indices, vel, dof_vel)
      root_pose_reward, root_pose_ref = self.reward_root_state(indices, frame, torso_position, torso_rotation)
      root_velocity_reward = self.reward_root_vel(indices, vel, root_velocities)
      end_effector_reward = self.reward_ee_state(indices, frame, end_state)

      reward_tmp = self._pose_weight * pose_reward + self._root_pose_weight * root_pose_reward + \
                self._velocity_weight * vel_reward + self._root_velocity_weight * root_velocity_reward + \
                self._end_effector_weight * end_effector_reward

      reward[indices] = reward_tmp.float()
      dof_pos_ref_all[indices] = dof_pos_ref.to(dof_pos_ref_all.dtype)
      root_pose_ref_all[indices] = root_pose_ref.to(root_pose_ref_all.dtype)

    return reward * self._weight, dof_pos_ref_all, root_pose_ref_all

  def reward_dof_pos_adaptive(self, indices, frame, dof_pos, joint_ids):
      j_pose_ref = frame[:,7:19].clone()
      # alignment between api of simulator and ref motion
      j_pose_ref_tmp = self._reorder_dof(j_pose_ref)

      j_pose_diff = j_pose_ref_tmp - dof_pos.index_select(0, indices)
      pose_err = torch.sum(torch.square(j_pose_diff[:, joint_ids]), dim=1)
      pose_reward = torch.exp(-self._pose_err_scale * pose_err)

      return pose_reward, j_pose_ref_tmp

  def reward_dof_pos(self, indices, frame, dof_pos):
      j_pose_ref = frame[:,7:19].clone()
      # alignment between api of simulator and ref motion
      j_pose_ref_tmp = self._reorder_dof(j_pose_ref)

      j_pose_diff = j_pose_ref_tmp - dof_pos.index_select(0, indices)
      j_pose_diff[:,0] = 0
      j_pose_diff[:,3] = 0
      j_pose_diff[:,6] = 0
      j_pose_diff[:,9] = 0
      pose_err = torch.sum(torch.square(j_pose_diff), dim=1)
      pose_reward = torch.exp(-self._pose_err_scale * pose_err)

      return pose_reward, j_pose_ref_tmp

  def reward_dof_vel(self, indices, vel, dof_vel):
      j_vel_ref = vel[:,6:18].clone()

      # alignment between api of simulator and ref motion
      j_vel_ref_tmp = self._reorder_dof(j_vel_ref)

      j_vel_diff = j_vel_ref_tmp - dof_vel.index_select(0, indices)
      vel_err = torch.sum(torch.square(j_vel_diff), dim=1)
      vel_reward = torch.exp(-self._velocity_err_scale * vel_err)

      return vel_reward

  def reward_root_state(self, indices, frame, torso_position, torso_rotation):
      root_pos_ref = frame[:,:3].clone()
      root_pos_ref[:,0:2] += self._default_pose.index_select(0, indices)[:,0:2]
      root_pos_sim = torso_position.index_select(0, indices)
      root_pos_diff = root_pos_ref - root_pos_sim
      root_pos_err = torch.sum(torch.square(root_pos_diff), dim=1)

      root_rot_ref = frame[:,3:7].clone()
      root_rot_sim = torso_rotation.index_select(0, indices)
      root_rot_diff = root_rot_ref - root_rot_sim
      root_rot_err = torch.sum(torch.square(root_rot_diff), dim=1)

      root_pose_err = root_pos_err + 0.5 * root_rot_err
      root_pose_reward = torch.exp(-self._root_pose_err_scale * root_pose_err)

      return root_pose_reward, root_pos_ref

  def reward_root_vel(self, indices, vel, root_velocities):
      root_vel_ref = vel[:,:3].clone()
      root_vel_sim = root_velocities.index_select(0, indices)[:,0:3]
      root_vel_diff = root_vel_ref - root_vel_sim
      root_vel_err = torch.sum(torch.square(root_vel_diff), dim=1)

      root_ang_vel_ref = vel[:,3:6].clone()
      root_ang_vel_sim = root_velocities.index_select(0, indices)[:,3:6]
      root_ang_vel_diff = root_ang_vel_ref - root_ang_vel_sim
      root_ang_vel_err = torch.sum(torch.square(root_ang_vel_diff), dim=1)

      root_velocity_err = root_vel_err + 0.1 * root_ang_vel_err
      root_velocity_reward = torch.exp(-self._root_velocity_err_scale * root_velocity_err)

      return root_velocity_reward

  def reward_ee_state(self, indices, frame, end_state):
      #end effector state
      end_state_ref = frame[:,19:].clone().reshape(-1,4,3)
      end_state_ref[:,:,0:2] += self._default_pose.index_select(0, indices)[:,0:2].unsqueeze(1)
      end_state_ref = end_state_ref.reshape(-1,12)
      end_state_sim = end_state.index_select(0, indices)

      # alignment between api of simulator and ref motion
      end_state_ref_tmp = self._reorder_ee(end_state_ref)

      end_state_diff = end_state_ref_tmp - end_state_sim
      end_state_err = torch.sum(torch.square(end_state_diff), dim=1)
      end_effector_reward = torch.exp(-self._end_effector_err_scale * end_state_err)

      return end_effector_reward

  def _get_motion_time(self):
    """Get the time since the start of the reference motion."""
    time = self.progress_buf * self.time_step
    return time

  def get_active_motion(self, motion_id):
    """Get index of the active reference motion currently being imitated.

    Returns:
      Index of the active reference motion.
    """
    return self._ref_motions[motion_id]

  def _load_ref_motions(self, filenames):
    """Load reference motions.

    Args:
      dir: Directory containing the reference motion files.
      filenames: Names of files in dir to be loaded.
    Returns: List of reference motions loaded from the files.
    """
    num_files = len(filenames)
    if num_files == 0:
      raise ValueError("No reference motions specified.")

    total_time = 0.0
    motions = []
    for filename in filenames:
      curr_motion = motion_data.MotionData(filename, self.device)
      curr_duration = curr_motion.get_duration()
      total_time += curr_duration
      motions.append(curr_motion)

    logging.info("Loaded {:d} motion clips with {:.3f}s of motion data.".format(
        num_files, total_time))
    return motions
