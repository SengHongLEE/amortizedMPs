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

import numpy as np
import torch


class _DynamicsBase:
    """Shared utilities for A1 oscillator-based dynamics controllers."""

    LEG_INDICES = np.array([1, 0, 3, 2])
    _INTEGRATION_DT = 0.001
    _OSCILLATOR_GAIN = 150.0
    _TWO_PI = 2 * np.pi

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Linearly scale from [-1, 1] to [lower_lim, upper_lim]."""
        scaled = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        return torch.clip(scaled, lower_lim, upper_lim)

    def _reset_main_state(self, env_ids):
        self._mu[env_ids, :] = 0
        self._omega_residuals[env_ids, :] = 0
        self.X[env_ids, 0, :] = torch.rand(len(env_ids), 4, device=self._device) * 0.1
        self.X[env_ids, 1, :] = self.PHI[0, :]
        self.X_dot[env_ids, :, :] = 0.0
        self.d2X[env_ids, :, :] = 0.0

    def integrate_oscillator_equations(self):
        dt = self._INTEGRATION_DT
        steps = max(1, int(self._dt / dt))
        oscillator_gain = self._OSCILLATOR_GAIN
        half_dt = dt * 0.5
        quarter_gain = oscillator_gain * 0.25

        X = self.X
        X_dot = self.X_dot
        d2X = self.d2X
        sqrt_mu = torch.sqrt(self._mu)

        for _ in range(steps):
            prev_d2X = d2X
            prev_X_dot = X_dot
            d2X = (oscillator_gain * (quarter_gain * (sqrt_mu - X[:, 0, :]) - prev_X_dot[:, 0, :])).unsqueeze(1)
            X_dot = torch.empty_like(prev_X_dot)
            X_dot[:, 1, :] = self._omega_residuals
            X_dot[:, 0, :] = prev_X_dot[:, 0, :] + (prev_d2X[:, 0, :] + d2X[:, 0, :]) * half_dt
            X = X + (prev_X_dot + X_dot) * half_dt
            X[:, 1, :] = torch.remainder(X[:, 1, :], self._TWO_PI)

        self.X = X
        self.X_dot = X_dot
        self.d2X = d2X

    def compute_inverse_kinematics(self, robot, legID, x, y, z):
        l1 = robot.hip_link_length_a1
        l2 = robot.thigh_link_length_a1
        l3 = robot.calf_link_length_a1

        D = (y**2 + (-z)**2 - l1**2 + (-x)**2 - l2**2 - l3**2) / (2 * l3 * l2)
        D = torch.clip(D, -1.0, 1.0)

        sideSign = -1 if legID == 0 or legID == 2 else 1
        knee_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z)**2 - l1**2
        hip_roll_angle = -1 * (
            -torch.atan2(z, y)
            - torch.atan2(torch.sqrt(sqrt_component), sideSign * l1 * torch.ones_like(x))
        )
        hip_thigh_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) - 1 * torch.atan2(
            l3 * torch.sin(knee_angle),
            l2 + l3 * torch.cos(knee_angle),
        )
        return torch.stack([hip_roll_angle, hip_thigh_angle, knee_angle], dim=-1)


class Intrinsic_Dynamics(_DynamicsBase):
    """Intrinsic dynamics controller"""

    _MAX_STEP_LEN = 0.15
    _GROUND_CLEARANCE = 0.1
    _GROUND_PENETRATION = 0.01

    def __init__(
        self,
        time_step=0.001,
        robot_height=0.3,
        num_envs=1,
        device=None,
        mu_low=0.5,
        mu_up=4.0,
    ):
        self._device = device
        self.num_envs = num_envs
        self._robot_height = robot_height
        self.mu_low = mu_low
        self.mu_up = mu_up
        self._dt = time_step

        self.X = torch.zeros(num_envs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.X_dot = torch.zeros(num_envs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.d2X = torch.zeros(num_envs, 1, 4, dtype=torch.float, device=device, requires_grad=False)
        self._mu = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._omega_residuals = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)

        self.y = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.x = torch.zeros_like(self.y)
        self.z = torch.zeros_like(self.y)

        self.PHI = torch.tensor(
            [[0, -1, -1.5, -1], [1, 0, 0, 1], [1, 0, 0, 1], [0, -1, -1, 0]],
            dtype=torch.float,
            device=device,
            requires_grad=False,
        ) * np.pi

        self.X[:, 0, :] = torch.rand(num_envs, 4, device=device) * 0.1
        self.X[:, 1, :] = self.PHI[0, :]

    def reset(self, env_ids, motion_label):
        self._reset_main_state(env_ids)
        self.motion_label = motion_label

    def omega_upper_limit(self, command):
        return ((14 * (command - 0.2)) / 0.8 + 30).unsqueeze(-1)

    def _apply_motion_group(self, indices, actions, command):
        if indices.numel() == 0:
            return

        omega_limit = self.omega_upper_limit(command[indices])
        self._mu[indices, :] = self._scale_helper(actions[indices, :4], self.mu_low**2, self.mu_up**2)
        self._omega_residuals[indices, :] = self._scale_helper(
            actions[indices, 4:8],
            torch.zeros_like(omega_limit),
            omega_limit,
        )

    def intrinsic_dynamics(self, actions, command):
        clipped_actions = torch.clip(actions, -1, 1)

        self._apply_motion_group((self.motion_label == 0).nonzero(as_tuple=True)[0], clipped_actions, command)
        self._apply_motion_group((self.motion_label == 1).nonzero(as_tuple=True)[0], clipped_actions, command)
        self._apply_motion_group((self.motion_label == 2).nonzero(as_tuple=True)[0], clipped_actions, command)

        self.integrate_oscillator_equations()

        phase = self.X[:, 1, :]
        sin_phase = torch.sin(phase)
        self.x = torch.clip(self.X[:, 0, :], self.mu_low, self.mu_up)
        self.x = self._MAX_STEP_LEN * (self.x - self.mu_low) / (self.mu_up - self.mu_low)
        self.x = -self.x * torch.cos(phase)
        self.z = torch.where(
            sin_phase > 0,
            -self._robot_height + self._GROUND_CLEARANCE * sin_phase,
            -self._robot_height + self._GROUND_PENETRATION * sin_phase,
        )
        return self.x, self.y, self.z


class Amortized_Control(_DynamicsBase):
    """Amortized motor-primitive controller."""

    _MU_LOW = 0.5
    _MU_UPP = 4.0
    _ROBOT_HEIGHT = 0.3
    _MAX_STEP_LEN = 0.1
    _GROUND_CLEARANCE = 0.15
    _GROUND_PENETRATION = 0.01

    def __init__(
        self,
        time_step=0.001,
        num_envs=1,
        device=None,
        num_of_MPs=3,
    ):
        self._device = device
        self.num_envs = num_envs
        self._dt = time_step
        self.num_of_MPs = num_of_MPs

        self.X = torch.zeros(num_envs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.X_dot = torch.zeros(num_envs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.d2X = torch.zeros(num_envs, 1, 4, dtype=torch.float, device=device, requires_grad=False)
        self.Xs = torch.zeros(num_envs, num_of_MPs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.X_dots = torch.zeros(num_envs, num_of_MPs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.d2Xs = torch.zeros(num_envs, num_of_MPs, 1, 4, dtype=torch.float, device=device, requires_grad=False)

        self._mu = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._omega_residuals = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._mu_s = torch.zeros(num_envs, num_of_MPs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._omega_residuals_s = torch.zeros(num_envs, num_of_MPs, 4, dtype=torch.float, device=device, requires_grad=False)

        self.y = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.x = torch.zeros_like(self.y)
        self.z = torch.zeros_like(self.y)

        self.PHI = torch.tensor(
            [[0, -1, -1.5, -1], [1, 0, 0, 1], [1, 0, 0, 1], [0, -1, -1, 0]],
            dtype=torch.float,
            device=device,
            requires_grad=False,
        ) * np.pi

        self.X[:, 0, :] = torch.rand(num_envs, 4, device=device) * 0.1
        self.X[:, 1, :] = self.PHI[0, :]

    def reset(self, env_ids):
        self._reset_main_state(env_ids)
        self._mu_s[env_ids, :, :] = 0
        self._omega_residuals_s[env_ids, :, :] = 0
        self.Xs[env_ids, :, 0, :] = torch.rand(len(env_ids), self.num_of_MPs, 4, device=self._device) * 0.1
        self.Xs[env_ids, :, 1, :] = self.PHI[0, :]
        self.X_dots[env_ids, :, :, :] = 0.0
        self.d2Xs[env_ids, :, :, :] = 0.0

    def omega_upper_limit(self, command):
        return (14 * (command.unsqueeze(-1).expand(-1, -1, 4) - 0.2)) / 0.8 + 30

    def _apply_coupling(self, omega_residuals, phase, sub_phase, sub_amplitude, index_tensor, source_a, source_b):
        if index_tensor.numel() == 0:
            return

        selected_phase = phase[index_tensor]
        coupling_1 = self.beta[index_tensor, 1].unsqueeze(-1) * sub_amplitude[index_tensor, source_a]
        coupling_2 = self.beta[index_tensor, 2].unsqueeze(-1) * sub_amplitude[index_tensor, source_b]
        omega_residuals[index_tensor] += coupling_1 * torch.sin(selected_phase - sub_phase[index_tensor, source_a])
        omega_residuals[index_tensor] += coupling_2 * torch.sin(selected_phase - sub_phase[index_tensor, source_b])

    def integrate_oscillator_equations(self):
        dt = self._INTEGRATION_DT
        steps = max(1, int(self._dt / dt))
        oscillator_gain = self._OSCILLATOR_GAIN
        half_dt = dt * 0.5
        quarter_gain = oscillator_gain * 0.25

        X = self.X
        X_dot = self.X_dot
        d2X = self.d2X
        Xs = self.Xs
        X_dots = self.X_dots
        d2Xs = self.d2Xs
        sqrt_mu = torch.sqrt(self._mu)
        sqrt_mu_s = torch.sqrt(self._mu_s)

        for _ in range(steps):
            prev_d2Xs = d2Xs
            prev_X_dots = X_dots
            d2Xs = (
                oscillator_gain * (quarter_gain * (sqrt_mu_s - Xs[:, :, 0, :]) - prev_X_dots[:, :, 0, :])
            ).unsqueeze(1)
            X_dots = torch.empty_like(prev_X_dots)
            X_dots[:, :, 1, :] = self._omega_residuals_s
            X_dots[:, :, 0, :] = prev_X_dots[:, :, 0, :] + (prev_d2Xs[:, :, 0, :] + d2Xs[:, :, 0, :]) * half_dt
            Xs = Xs + (prev_X_dots + X_dots) * half_dt
            Xs[:, :, 1, :] = torch.remainder(Xs[:, :, 1, :], self._TWO_PI)

            prev_d2X = d2X
            prev_X_dot = X_dot
            d2X = (oscillator_gain * (quarter_gain * (sqrt_mu - X[:, 0, :]) - prev_X_dot[:, 0, :])).unsqueeze(1)

            omega_residuals = self._omega_residuals.clone()
            phase = X[:, 1, :]
            sub_phase = Xs[:, :, 1, :]
            sub_amplitude = Xs[:, :, 0, :]
            self._apply_coupling(omega_residuals, phase, sub_phase, sub_amplitude, self.indices_pace, 1, 2)
            self._apply_coupling(omega_residuals, phase, sub_phase, sub_amplitude, self.indices_trot, 0, 2)
            self._apply_coupling(omega_residuals, phase, sub_phase, sub_amplitude, self.indices_canter, 0, 1)
            self._omega_residuals = omega_residuals

            X_dot = torch.empty_like(prev_X_dot)
            X_dot[:, 1, :] = omega_residuals
            X_dot[:, 0, :] = prev_X_dot[:, 0, :] + (prev_d2X[:, 0, :] + d2X[:, 0, :]) * half_dt
            X = X + (prev_X_dot + X_dot) * half_dt
            X[:, 1, :] = torch.remainder(X[:, 1, :], self._TWO_PI)

        self.X = X
        self.X_dot = X_dot
        self.d2X = d2X
        self.Xs = Xs
        self.X_dots = X_dots
        self.d2Xs = d2Xs

    def update(self, primitive_actions, coupling_actions, command):
        self.beta = coupling_actions
        max_idx = self.beta[:, 0]
        groups = {i: (max_idx == i).nonzero(as_tuple=True)[0] for i in [0, 1, 2]}
        self.indices_pace = groups[0]
        self.indices_trot = groups[1]
        self.indices_canter = groups[2]

        action_primitive = torch.clip(primitive_actions, -1, 1)
        omega_limit = self.omega_upper_limit(command)
        self._mu_s = self._scale_helper(action_primitive[:, :, :4], self._MU_LOW**2, self._MU_UPP**2)
        self._omega_residuals_s = self._scale_helper(
            action_primitive[:, :, 4:8],
            torch.zeros_like(omega_limit),
            omega_limit,
        )

        self._mu[self.indices_pace] = self._mu_s[self.indices_pace, 0]
        self._omega_residuals[self.indices_pace] = self._omega_residuals_s[self.indices_pace, 0]
        self._mu[self.indices_trot] = self._mu_s[self.indices_trot, 1]
        self._omega_residuals[self.indices_trot] = self._omega_residuals_s[self.indices_trot, 1]
        self._mu[self.indices_canter] = self._mu_s[self.indices_canter, 2]
        self._omega_residuals[self.indices_canter] = self._omega_residuals_s[self.indices_canter, 2]

        self.integrate_oscillator_equations()

        phase = self.X[:, 1, :]
        sin_phase = torch.sin(phase)
        self.z = torch.where(
            sin_phase > 0,
            -self._ROBOT_HEIGHT + self._GROUND_CLEARANCE * sin_phase,
            -self._ROBOT_HEIGHT + self._GROUND_PENETRATION * sin_phase,
        )
        self.x = torch.clip(self.X[:, 0, :], self._MU_LOW, self._MU_UPP)
        self.x = self._MAX_STEP_LEN * (self.x - self._MU_LOW) / (self._MU_UPP - self._MU_LOW)
        self.x = -self.x * torch.cos(phase)
        return self.x, self.y, self.z
