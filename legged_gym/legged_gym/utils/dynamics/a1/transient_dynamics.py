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
        self._mu_x[env_ids, :] = 0.
        self._mu_z[env_ids, :] = 0.
        self._omega[env_ids, :] = 0.
        # self.X[env_ids, 0, :] = torch.rand(len(env_ids), 4, device=self._device) * 0.1
        self.X[env_ids, :, :] = 0.
        self.X_dot[env_ids, :, :] = 0.0
        self.d2X[env_ids, :, :] = 0.0
        self.phi[env_ids, :] = self.phi_init
        self.delta[env_ids, :] = 0.0
        self.active[env_ids, :] = True

    def wrap_to_pi(self, x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def integrate_oscillator_equations(self):
        dt = self._INTEGRATION_DT
        steps = max(1, int(self._dt / dt))
        oscillator_gain = self._OSCILLATOR_GAIN
        half_dt = dt * 0.5
        quarter_gain = oscillator_gain * 0.25

        X = self.X
        X_dot = self.X_dot
        d2X = self.d2X

        f_gait = 4.
        f_modulation = 2.
        omega_g = 2 * torch.pi * (f_gait + self._omega * f_modulation)
        omega_g = torch.clamp(omega_g, 2 * torch.pi * (f_gait - f_modulation) , 2 * torch.pi * (f_gait + f_modulation))

        for _ in range(steps):
            target_amp_x = torch.where(
                self.active,
                torch.sqrt(self._mu_x),
                torch.zeros_like(self._mu_x),
            )
            target_amp_z = torch.where(
                self.active,
                torch.sqrt(self._mu_z),
                0. * torch.ones_like(self._mu_z),
            )
            prev_d2X = d2X.clone()
            prev_X_dot = X_dot.clone()

            d2X[:, 0, :] = (oscillator_gain * (quarter_gain * (target_amp_x - X[:, 0, :]) - prev_X_dot[:, 0, :]))
            d2X[:, 1, :] = (oscillator_gain * (quarter_gain * (target_amp_z - X[:, 1, :]) - prev_X_dot[:, 1, :]))

            X_dot = torch.empty_like(prev_X_dot)
            X_dot[:, :2, :] = prev_X_dot[:, :2, :] + (prev_d2X[:, :2, :] + d2X[:, :2, :]) * half_dt
            X_dot[:, 2, :] = torch.where(
                self.active,
                omega_g,
                torch.zeros_like(omega_g),
            )

            X[:, :2, :] += (prev_X_dot[:, :2, :] + X_dot[:, :2, :]) * half_dt
            X[:, 2, :] += X_dot[:, 2, :] * dt

            # -----------------------------------
            # one-shot termination
            # -----------------------------------
            finished = X[:, 2, :] >= 2.0 * torch.pi

            X[:, 2, :] = torch.clamp(
                X[:, 2, :],
                max=2.0 * torch.pi,
            )

            self.active = self.active & (~finished)

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


class Transient_Dynamics(_DynamicsBase):
    """Transient dynamics controller"""

    _MAX_STEP_LEN = 0.15
    _LOAD_COMPRESSION = 0.05
    _PUSH_EXTENSION = 0.1
    _TUCK_HEIGHT = 0.15

    def __init__(
        self,
        time_step=0.001,
        robot_height=0.3,
        num_envs=1,
        device=None,
        mu_low=0.5,
        mu_up=4.0,
        L =  0.2,
        W = 0.1
    ):
        self._device = device
        self.num_envs = num_envs
        self._robot_height = robot_height
        self.mu_low = mu_low
        self.mu_up = mu_up
        self._dt = time_step

        self._FOOT_X = torch.tensor([ L,  L, -L, -L], device=self._device)
        self._FOOT_Y = torch.tensor([ W,  -W, W, -W], device=self._device)

        self.X = torch.zeros(num_envs, 3, 4, dtype=torch.float, device=device, requires_grad=False)
        self.X_dot = torch.zeros(num_envs, 3, 4, dtype=torch.float, device=device, requires_grad=False)
        self.d2X = torch.zeros(num_envs, 2, 4, dtype=torch.float, device=device, requires_grad=False)
        self.phi = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.delta = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)

        self._mu_x = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._mu_z = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self._omega = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)

        self.active = torch.ones_like(self._mu_x) == 1

        self.y = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.x = torch.zeros_like(self.y)
        self.z = torch.zeros_like(self.y)

        self.phi_init = torch.tensor(
            [0, 0., 0., 0.],
            dtype=torch.float,
            device=device,
            requires_grad=False,
        ) * np.pi

        self.X[:] = 0.
        self.phi[:] = self.phi_init

    def reset(self, env_ids, motion_label):
        self._reset_main_state(env_ids)
        self.motion_label = motion_label


    def intrinsic_dynamics(self, actions, command):
        clipped_actions = torch.clip(actions, -1, 1)

        self._mu_x[:] = self._scale_helper(clipped_actions[:, :4], self.mu_low**2, self.mu_up**2)
        self._mu_z[:] = self._scale_helper(clipped_actions[:, 4:8], self.mu_low**2, self.mu_up**2)
        self._omega[:] = clipped_actions[:, 8:9]

        self.integrate_oscillator_equations()

        phase = self.X[:, 2, :]      
        amplitude_x = self.X[:, 0, :]
        amplitude_z = self.X[:, 1, :]
        amp_clip_x = torch.clip(amplitude_x, self.mu_low, self.mu_up)
        amp_norm_x = (amp_clip_x - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_x = torch.clamp(amp_norm_x, 0.0, 1.0)

        amp_clip_z = torch.clip(amplitude_z, self.mu_low, self.mu_up)
        amp_norm_z = (amp_clip_z - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_z = torch.clamp(amp_norm_z, 0.0, 1.0)

        amp_load_x = 0. * self._MAX_STEP_LEN * amp_norm_x
        amp_push_x = self._MAX_STEP_LEN * amp_norm_x
        amp_load_z = self._LOAD_COMPRESSION * amp_norm_z
        amp_push_z = self._PUSH_EXTENSION * amp_norm_z
        amp_tuck_z = torch.where(self.active, self._TUCK_HEIGHT * torch.ones_like(amp_norm_z),  self._TUCK_HEIGHT * amp_norm_z)

        # ============================================================
        # Stage 1: preload, 0 -> pi
        # nominal -> slightly forward
        # ============================================================
        load_phase = torch.clamp(
            phase,
            min=0.0,
            max=torch.pi,
        )
        load_basis = torch.sin(load_phase).square()
        x_load = amp_load_x * load_basis

        # ============================================================
        # Stage 2: push, pi -> 1.5 pi
        # forward preload -> backward extension
        # ============================================================
        push_phase = torch.clamp(
            (phase - torch.pi),
            min=0.0,
            max=0.5 * torch.pi,
        )
        push_basis = torch.sin(push_phase).square()

        x_push = (
            amp_load_x
            - (amp_load_x + amp_push_x) * push_basis
        )
        sin_phase = torch.sin(phase)
        self.x = torch.where(
            phase < torch.pi,
            # torch.zeros_like(self.x),
            amp_load_x * torch.sin(0.5 * phase).square(),
            amp_load_x - (amp_push_x + amp_load_x) * sin_phase.square(),
        )
        #####
        # self.z = torch.where(
        #     sin_phase > 0,
        #     -self._robot_height + amp_load_z * sin_phase.square(),
        #     -self._robot_height - amp_push_z * sin_phase.square(),
        # )



        # ============================================================
        # Stage 1: preload, 0 -> pi
        # ============================================================
        load_basis = torch.sin(0.5 * phase).square()

        z_load = (
            -self._robot_height
            + amp_load_z * load_basis
        )

        # ============================================================
        # Stage 2: push,  pi -> 1.5 pi
        # monotonic nominal -> maximum extension
        # ============================================================
        push_phase = torch.clamp(
            (phase - torch.pi),
            min=0.0,
            max=0.5 * torch.pi,
        )
        push_basis = torch.sin(push_phase).square()

#         push_progress = torch.clamp(
#             (phase - 0.5 * torch.pi) / torch.pi,
#             min=0.0,
#             max=1.0,
# )

#         push_basis = (
#             20.0 * push_progress.pow(3)
#             - 45.0 * push_progress.pow(4)
#             + 36.0 * push_progress.pow(5)
#             - 10.0 * push_progress.pow(6)
#         )

        z_push = (
            -self._robot_height + amp_load_z
            - (amp_push_z + amp_load_z) * push_basis
        )

        # ============================================================
        # Stage 3: tuck, 1.5 pi -> 2 pi
        # maximum extension -> tucked configuration
        # ============================================================
        tuck_progress = torch.clamp(
            (phase - 1.5 * torch.pi) / (0.001 * torch.pi),
            min=0.0,
            max=1.0,
        )

        smooth_tuck = (
            3.0 * tuck_progress.square()
            - 2.0 * tuck_progress.pow(3)
        )

        z_tuck = (
            -self._robot_height
            - amp_push_z
            + (amp_push_z + amp_tuck_z) * smooth_tuck
        )

        # self.x = torch.where(
        #     phase < torch.pi,
        #     x_load,
        #     x_push
        # )

        self.z = torch.where(
            phase < torch.pi,
            z_load,
            torch.where(
                phase < 1.5 * torch.pi,
                z_push,
                z_tuck,
            ),
        )
        return self.x, self.y, self.z