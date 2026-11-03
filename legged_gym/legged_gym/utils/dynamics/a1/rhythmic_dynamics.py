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
        self.phases[env_ids, :] = 0.
        self.omega[env_ids, :] = 0.
        self.X[env_ids, 0, :] = torch.rand(len(env_ids),4,device=self._device) * 0.1
        self.X[env_ids, 1, :] = torch.rand(len(env_ids),4,device=self._device) * 0.1
        self.X[env_ids, 2, :] = 0.
        self.X_dot[env_ids, :, :] = 0.0
        self.d2X[env_ids, :, :] = 0.0
        self.phi[env_ids, :] = 0.
        self.delta[env_ids, :] = 0.0
        self.theta_unwrapped[env_ids, :] = 0.
        self.delta_unwrapped[env_ids, :] = 0.
        self.prev_delta_raw[env_ids, :] = 0.
        self.phase_correction[env_ids, :] = 0.

    def wrap_to_pi(self, x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def update_unwrapped_phase_command(self):
        # Current policy output -> wrapped relative phase
        delta_raw = torch.pi * self.phases[:, 1:]

        # Difference from previous raw command
        raw_diff = delta_raw - self.prev_delta_raw

        # Shortest displacement on S1
        delta_step = torch.remainder(
            raw_diff + torch.pi,
            2.0 * torch.pi
        ) - torch.pi

        # ----------------------------------------------------------
        # Exact pi is fundamentally ambiguous.
        # Preserve the direction implied by raw action change.
        # ----------------------------------------------------------
        eps = 1e-6
        at_antipode = (
            torch.abs(torch.abs(raw_diff) - torch.pi) < eps
        )
        direction = torch.where(
            raw_diff >= 0.0,
            torch.ones_like(raw_diff),
            -torch.ones_like(raw_diff)
        )
        delta_step = torch.where(
            at_antipode,
            direction * torch.pi,
            delta_step
        )

        # Lift S1 command to R
        self.delta_unwrapped[:, 1:] += delta_step
        self.delta_unwrapped[:, 0] = 0.0

        self.prev_delta_raw.copy_(delta_raw)


    def phase_oscillator(self):
        dt = self._INTEGRATION_DT

        f_gait = 8.
        f_modulation = 2.
        omega_g = 2 * torch.pi * (f_gait + self.phases[:,:1] * f_modulation)
        omega_g = torch.clamp(omega_g, 2 * torch.pi * (f_gait - f_modulation) , 2 * torch.pi * (f_gait + f_modulation))
        # f_max = self.omega_upper_limit(self.cmd[:,0])
        # f_min = self.omega_lower_limit(self.cmd[:,0])
        # omega_g = 2 * torch.pi * self._scale_helper(self.phases[:,:1], f_min, f_max) 
        omega_step = omega_g * dt

        # self.update_unwrapped_phase_command()
        delta_raw = torch.pi * self.phases[:, 1:]

        self.delta[:, 0] = 0.
        self.delta[:, 1:] = delta_raw
        theta_target = self.phi + self.delta
        phase_error = self.wrap_to_pi(theta_target - self.X[:,2,:])
        # ============================================================
        # Anti-podal branch hysteresis
        # ============================================================
        phase_margin = 0.05 * torch.pi
        zero_eps = 1e-6
        near_antipodal = (
            torch.abs(phase_error)
            > torch.pi - phase_margin
        )
        candidate_sign = torch.sign(phase_error)
        # Do NOT allow zero to erase the stored direction
        candidate_sign = torch.where(
            torch.abs(phase_error) > zero_eps,
            candidate_sign,
            self.prev_phase_correction_sign,
        )
        # Keep previous convergence direction near +/- pi
        stable_sign = torch.where(
            near_antipodal,
            self.prev_phase_correction_sign,
            candidate_sign,
        )
        phase_error_stable = torch.where(
            near_antipodal,
            stable_sign * torch.abs(phase_error),
            phase_error,
        )
        # Update memory only after leaving ambiguous region
        self.prev_phase_correction_sign = torch.where(
            near_antipodal,
            self.prev_phase_correction_sign,
            candidate_sign,
        )

        omega_corr_max = 0.3 * omega_g            #2. * torch.pi * 4.
        kappa = 0.6 * omega_g
        phase_correction_des = (
            omega_corr_max
            * torch.tanh(
                kappa
                * phase_error_stable
                / omega_corr_max
            )
        )
        # ============================================================
        # Smooth correction dynamics
        # tau_corr controls how rapidly convergence direction can change
        # ============================================================
        tau_corr = 0.02

        alpha_corr = dt / (tau_corr + dt)

        self.phase_correction = (
            self.phase_correction
            + alpha_corr
            * (
                phase_correction_des
                - self.phase_correction
            )
        )
        omega = omega_g + self.phase_correction
        self.integrate_oscillator_equations(omega)

        self.phi.add_(omega_step)
        self.phi = torch.remainder(self.phi, self._TWO_PI)


    def omega_oscillator(self):
        dt = self._INTEGRATION_DT
        steps = max(1, int(self._dt / dt))
        oscillator_gain = self._OSCILLATOR_GAIN
        half_dt = dt * 0.5
        quarter_gain = oscillator_gain * 0.25

        omega_g = 2.0 * torch.pi * self.omega
        self.integrate_oscillator_equations(omega_g)
        

    def integrate_oscillator_equations(self, omega):
        dt = self._INTEGRATION_DT
        steps = max(1, int(self._dt / dt))

        oscillator_gain = self._OSCILLATOR_GAIN
        half_dt = dt * 0.5
        quarter_gain = oscillator_gain * 0.25

        d2X = self.d2X.clone()
        X_dot = self.X_dot.clone() 
        for _ in range(steps):
            sqrt_mu = torch.sqrt(self._mu_x)

            prev_d2X = self.d2X.clone()
            prev_X_dot = self.X_dot.clone()
            prev_X = self.X.clone()

            d2X[:, 0, :] = (oscillator_gain * (quarter_gain * (sqrt_mu - prev_X[:, 0, :]) - prev_X_dot[:, 0, :]))
            X_dot[:, 0, :] = prev_X_dot[:, 0, :] + (prev_d2X[:, 0, :] + d2X[:, 0, :]) * half_dt
            X_dot[:, 2, :] = omega

            X = prev_X + (prev_X_dot + X_dot) * half_dt
            X[:, 2, :] = torch.remainder(X[:, 2, :], self._TWO_PI)

            self.X = X.clone()
            self.X_dot = X_dot.clone()
            self.d2X = d2X.clone()

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

    _MAX_STEP_LEN = 0.1
    _GROUND_CLEARANCE = 0.15
    _GROUND_PENETRATION = 0.0       #canter train with 0.01, infer with 0.0

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
        self.phases = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.omega = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.omega_base = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)

        self.y = torch.zeros(num_envs, 4, dtype=torch.float, device=device, requires_grad=False)
        self.x = torch.zeros_like(self.y)
        self.z = torch.zeros_like(self.y)

        self.phi_init = torch.tensor(
            [0., 0., 0., 0.],
            # [0, -1., -1., 0.],
            # [0, -0.5, -1., 0.5],
            dtype=torch.float,
            device=device,
            requires_grad=False,
        ) * np.pi

        self.X[:, 0, :] = torch.rand(num_envs, 4, device=device) * 0.1
        self.X[:, 2, :] = self.phi_init
        self.phi[:] = self.phi_init

        self.theta_unwrapped = torch.zeros(self.num_envs, 4, device=device)
        self.delta_unwrapped = torch.zeros(self.num_envs, 4, device=device)
        self.phase_correction = torch.zeros(self.num_envs, 4, device=device)
        self.prev_delta_raw = torch.zeros(self.num_envs, 3, device=device)

        self.prev_phase_correction_sign = torch.ones(self.num_envs, 4, device=device)

        self.oscillator = None

    def reset(self, env_ids, motion_label):
        self._reset_main_state(env_ids)
        self.motion_label = motion_label

    def omega_upper_limit(self, command):
        cmd_x_min = 0.2
        f_max = 10.
        # return (((14 * (command)) / 0.8 + 30) / (2 * torch.pi)).unsqueeze(-1)
        return (7 + 1.1 * (command - cmd_x_min)).unsqueeze(-1) 

    def omega_lower_limit(self, command):
        cmd_x_min = 0.2
        f_min = 5.
        # return (((14 * (command)) / 0.8 + 30) / (2 * torch.pi)).unsqueeze(-1)
        return (5 + 1.1 * (command - cmd_x_min)).unsqueeze(-1)  


    def _apply_motion_group(self, indices, actions, command):
        if indices.numel() == 0:
            return
        self._mu_x[indices, :] = self._scale_helper(actions[indices, :4], self.mu_low**2, self.mu_up**2)
        self._mu_z[indices, :] = self._scale_helper(actions[indices, 4:8], self.mu_low**2, self.mu_up**2)
        self.phases[indices, :] = actions[indices, 4:8]
        self.omega[indices, :] = self._scale_helper(actions[indices, 4:8], self.omega_lower_limit(command[indices]), self.omega_upper_limit(command[indices]))
        self.omega_base[indices, :] =  0.5 * (self.omega_lower_limit(command[indices]) + self.omega_upper_limit(command[indices]))
        # self.phases[indices, 1] = -1.
        # self.phases[indices, 2] = -1.
        # self.phases[indices, 3] = 0.

    def intrinsic_dynamics(self, actions, command):
        clipped_actions = torch.clip(actions, -1, 1)
        self._apply_motion_group((self.motion_label == 0).nonzero(as_tuple=True)[0], clipped_actions, command[:,0])
        self._apply_motion_group((self.motion_label == 1).nonzero(as_tuple=True)[0], clipped_actions, command[:,0])
        self._apply_motion_group((self.motion_label == 2).nonzero(as_tuple=True)[0], clipped_actions, command[:,0])

        # self.integrate_oscillator_equations_trot()
        # self.integrate_oscillator_equations_canter()
        self.cmd = command
        self.oscillator()

        phase = self.X[:, 2, :]      

        amplitude_x = self.X[:, 0, :]
        amp_clip_x = torch.clip(amplitude_x, self.mu_low, self.mu_up)
        amp_norm_x = (amp_clip_x - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_x = torch.clamp(amp_norm_x, 0.0, 1.0)

        amplitude_z = self.X[:, 1, :]
        amp_clip_z = torch.clip(amplitude_z, self.mu_low, self.mu_up)
        amp_norm_z = (amp_clip_z - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_z = torch.clamp(amp_norm_z, 0.0, 1.0)

        amp_x = self._MAX_STEP_LEN * amp_norm_x
        amp_z = self._GROUND_CLEARANCE * amp_norm_z
        # self.x = -self.x * torch.cos(phase)


        #####
        cmd_x = command[:, 0:1]
        cmd_yaw = command[:, 2:3]

        foot_x = self._FOOT_X.unsqueeze(0)
        foot_y = self._FOOT_Y.unsqueeze(0)

        vx_leg = (
            cmd_x
            - cmd_yaw * foot_y
        )

        vy_leg = (
            cmd_yaw * foot_x
        )

        motion_norm = torch.sqrt(
                vx_leg.square()
                + vy_leg.square()
            )

        moving = motion_norm > 1e-4

        norm_safe = motion_norm.clamp_min(1e-4)

        dir_x = vx_leg / norm_safe
        dir_y = vy_leg / norm_safe
        stroke = -amp_x * torch.cos(phase)

        self.x = stroke * dir_x
        self.y = stroke * dir_y

        self.x = torch.where(
            moving,
            self.x,
            torch.zeros_like(self.x),
        )
        self.y = torch.where(
            moving,
            self.y,
            torch.zeros_like(self.y),
        )
        #####
        sin_phase = torch.sin(phase)
        self.z = torch.where(
            sin_phase > 0,
            -self._robot_height + self._GROUND_CLEARANCE * sin_phase,
            -self._robot_height + self._GROUND_PENETRATION * sin_phase,
        )
        return self.x, self.y, self.z

class Reference_Motion_Dynamics(Intrinsic_Dynamics):
    def __init__(
        self,
        time_step=0.001,
        robot_height=0.3,
        num_envs=1,
        device=None,
        mu_low=0.5,
        mu_up=4.0,
        L =  0.2,
        W = 0.1,
        init_phase = [0, 0, 0, 0]
    ):
        super().__init__(time_step, robot_height, num_envs, device, mu_low, mu_up, L, W)
        self.phi_init = torch.tensor(init_phase,
                                     dtype=torch.float,
                                     device=device,
                                     requires_grad=False,)
        self.X[:, 2, :] = torch.tensor([0, 0, 0, 0],
                                     dtype=torch.float,
                                     device=device,
                                     requires_grad=False,)
        self.phi[:] = self.phi_init * torch.pi

    def reset(self, env_ids):
        self._reset_main_state(env_ids)

    def intrinsic_dynamics(self, actions, command):
        self.mu_low = 0.
        self.mu_up = 1.
        clipped_actions = torch.clip(actions, -1, 1)
        self._mu_x = self._scale_helper(clipped_actions[:, :4], self.mu_low**2, self.mu_up**2)

        self.phases[:, :4] = clipped_actions[:, 4:8]
        # self.phases[:,1] = 1
        # self.phases[:,2] = 0
        # self.phases[:,3] = 1 
        # print(self._MAX_STEP_LEN, self._GROUND_CLEARANCE, self._GROUND_PENETRATION)
        self.cmd = command
        self.oscillator()

        phase = self.X[:, 2, :]      
        # print(self.wrap_to_pi(phase[:,1] - phase[:,0]), self.wrap_to_pi(phase[:,2] - phase[:,0]), self.wrap_to_pi(phase[:,3] - phase[:,0]))
        amplitude_x = self.X[:, 0, :]
        amp_clip_x = torch.clip(amplitude_x, self.mu_low, self.mu_up)
        amp_norm_x = (amp_clip_x - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_x = torch.clamp(amp_norm_x, 0.0, 1.0)
        amp_x = self._MAX_STEP_LEN * amp_norm_x

        #####
        cmd_x = command[:, 0:1]
        cmd_yaw = command[:, 2:3]

        foot_x = self._FOOT_X.unsqueeze(0)
        foot_y = self._FOOT_Y.unsqueeze(0)

        vx_leg = cmd_x - cmd_yaw * foot_y
        vy_leg = cmd_yaw * foot_x

        motion_norm = torch.sqrt(
                vx_leg.square()
                + vy_leg.square()
            )

        moving = motion_norm > 1e-4

        norm_safe = motion_norm.clamp_min(1e-4)

        dir_x = vx_leg / norm_safe
        dir_y = vy_leg / norm_safe
        stroke = -amp_x * torch.cos(phase)

        self.x = stroke * dir_x
        self.y = stroke * dir_y

        self.x = torch.where(
            moving,
            self.x,
            torch.zeros_like(self.x),
        )
        self.y = torch.where(
            moving,
            self.y,
            torch.zeros_like(self.y),
        )
        #####
        sin_phase = torch.sin(phase)
        self.z = torch.where(
            sin_phase > 0,
            -self._robot_height + self._GROUND_CLEARANCE * sin_phase,
            -self._robot_height + self._GROUND_PENETRATION * sin_phase,
        )
        return self.x, self.y, self.z

class Interoceptive_Dynamics(Intrinsic_Dynamics):
    def __init__(
        self,
        time_step=0.001,
        robot_height=0.3,
        num_envs=1,
        device=None,
        mu_low=0.5,
        mu_up=4.0,
        L =  0.2,
        W = 0.1,
        init_phase = [0, 0, 0, 0]
    ):
        super().__init__(time_step, robot_height, num_envs, device, mu_low, mu_up, L, W)
        self.phi_init = torch.tensor(init_phase,
                                     dtype=torch.float,
                                     device=device,
                                     requires_grad=False,)
        self.X[:, 2, :] = torch.tensor([0, 0, 0, 0],
                                     dtype=torch.float,
                                     device=device,
                                     requires_grad=False,)
        self.phi[:] = self.phi_init * torch.pi

    def reset(self, env_ids):
        self._reset_main_state(env_ids)

    def intrinsic_dynamics(self, actions, command, contact_forces=None):
        self.mu_low = 0.
        self.mu_up = 2.
        clipped_actions = torch.clip(actions, -1, 1)
        self._mu_x = self._scale_helper(clipped_actions[:, :4], self.mu_low**2, self.mu_up**2)
        self.phases[:, :1] = clipped_actions[:, 4:5]
        self.phases[:, 1:4] = self.phi_init[None, 1:4]
        self.cmd = command
        self.oscillator()

        phase = self.X[:, 2, :]      
        # print(self.wrap_to_pi(phase[:,1] - phase[:,0]), self.wrap_to_pi(phase[:,2] - phase[:,0]), self.wrap_to_pi(phase[:,3] - phase[:,0]))
        amplitude_x = self.X[:, 0, :]
        amp_clip_x = torch.clip(amplitude_x, self.mu_low, self.mu_up)
        amp_norm_x = (amp_clip_x - self.mu_low) / (self.mu_up - self.mu_low)
        amp_norm_x = torch.clamp(amp_norm_x, 0.0, 1.0)

        amp_x = self._MAX_STEP_LEN * amp_norm_x

        #####
        cmd_x = command[:, 0:1]
        cmd_yaw = command[:, 2:3]

        foot_x = self._FOOT_X.unsqueeze(0)
        foot_y = self._FOOT_Y.unsqueeze(0)

        vx_leg = cmd_x - cmd_yaw * foot_y
        vy_leg = cmd_yaw * foot_x

        motion_norm = torch.sqrt(
                vx_leg.square()
                + vy_leg.square()
            )

        moving = motion_norm > 1e-4

        norm_safe = motion_norm.clamp_min(1e-4)

        dir_x = vx_leg / norm_safe
        dir_y = vy_leg / norm_safe
        stroke = -amp_x * torch.cos(phase)

        self.x = stroke * dir_x
        self.y = stroke * dir_y

        self.x = torch.where(
            moving,
            self.x,
            torch.zeros_like(self.x),
        )
        self.y = torch.where(
            moving,
            self.y,
            torch.zeros_like(self.y),
        )
        #####
        sin_phase = torch.sin(phase)
        self.z = torch.where(
            sin_phase > 0,
            -self._robot_height + self._GROUND_CLEARANCE * sin_phase,
            -self._robot_height + self._GROUND_PENETRATION * sin_phase,
        )
        return self.x, self.y, self.z

class Interoceptive_Reflex_Dynamics(Intrinsic_Dynamics):
    """
    State
    -----
    X[:, 0, :] : horizontal oscillator amplitudes (legacy normalized units)
    X[:, 1, :] : swing-clearance amplitudes [m]
    X[:, 2, :] : leg phases [rad], [FL, FR, RL, RR]

    Dynamics at internal substep (default 1 ms)
    ----------------------------------------------
      theta_dot[i] = omega_g + LPF(Kuramoto[i] + PhaseReflex[i])
      r_ddot      = alpha * (alpha/4 * (sqrt(mu) - r) - r_dot)
      tau_q*q_dot = trigger - q (on/off tau selected separately)

    API compatible with original `intrinsic_dynamics(actions, command)`, with
    *optional* `foot_contact` or `contact_forces` to activate reflexes.
    
    IMPORTANT: Contact order must be [FL, FR, RL, RR].
    The original LEG_INDICES = [1,0,3,2] is not applied automatically here.
    """

    LEG_NAMES = ("FL", "FR", "RL", "RR")

    def __init__(
        self,
        time_step=0.001,
        robot_height=0.3,
        num_envs=1,
        device=None,
        mu_low=0.0,
        mu_up=1.0,
        L=0.2,
        W=0.1,
        init_phase=[0., -1., -1., 0.],
        # oscillator
        integration_dt=0.001,
        f_gait=8.0,
        f_modulation=2.0,
        coupling_gain=32.0,
        tau_phase_correction=0.012,
        max_phase_correction_ratio=0.48,
        # reflex response
        tau_reflex_on=0.008,
        tau_reflex_off=0.055,
        reflex_uncoupling=0.85,
        phase_early_ratio=0.34,
        phase_late_ratio=0.42,
        phase_obstacle_ratio=0.18,
        # amplitude / contact-search
        nominal_clearance=0.15,
        max_clearance=0.22,
        max_stance_extension=0.05,
        tau_stance_extension=0.025,
        contact_force_threshold=1.0,
    ):
        super().__init__(
            time_step=time_step,
            robot_height=robot_height,
            num_envs=num_envs,
            device=device,
            mu_low=mu_low,
            mu_up=mu_up,
            L=L,
            W=W,
        )
        self._INTEGRATION_DT = integration_dt
        self.f_gait = float(f_gait)
        self.f_modulation = float(f_modulation)
        self.K_c = float(coupling_gain)
        self.tau_phase_correction = float(tau_phase_correction)
        self.max_phase_correction_ratio = float(max_phase_correction_ratio)
        self.tau_reflex_on = float(tau_reflex_on)
        self.tau_reflex_off = float(tau_reflex_off)
        self.reflex_uncoupling = float(reflex_uncoupling)
        self.phase_early_ratio = float(phase_early_ratio)
        self.phase_late_ratio = float(phase_late_ratio)
        self.phase_obstacle_ratio = float(phase_obstacle_ratio)
        self.max_clearance = float(max_clearance)
        self.max_stance_extension = float(max_stance_extension)
        self.tau_stance_extension = float(tau_stance_extension)
        self.contact_force_threshold = float(contact_force_threshold)

        self.gait_phase_delta = torch.tensor(
            init_phase, device=self._device
        ) * np.pi
        # The original class default `init_phase=[0,0,0,0]` is NOT a
        # stable initialized Trot. Interpret that legacy default as
        # `use the Trot template` instead of keeping all legs in phase.
        
        # Symmetric all-to-all Kuramoto, normalized for K_c independent of degree.
        self.A = (torch.ones(4, 4, device=self._device) -
                  torch.eye(4, device=self._device)) / 3.0
        self.target_diff = (
            self.gait_phase_delta[None, :] - self.gait_phase_delta[:, None]
        )  # [i,j] = delta_j - delta_i

        self.q_early = torch.zeros(num_envs, 4, device=self._device)
        self.q_late = torch.zeros_like(self.q_early)
        self.q_obstacle = torch.zeros_like(self.q_early)
        self.stance_extension = torch.zeros_like(self.q_early)
        self.stance_extension_gait_id = None
        self.kuramoto_term = torch.zeros_like(self.q_early)
        self.reflex_term = torch.zeros_like(self.q_early)
        self.effective_coupling = torch.ones_like(self.q_early)
        self.omega_g = torch.full((num_envs, 1), 2 * np.pi * f_gait,
                                  device=self._device)
        self.nominal_clearance = torch.ones_like(self.q_early) * float(nominal_clearance)
        self.nominal_clearance_single = float(nominal_clearance)
        self._contact = torch.zeros(num_envs, 4, dtype=torch.bool,
                                    device=self._device)
        self._contact_available = False
        # Preserves legacy `self.oscillator()` invocation pattern.
        self.oscillator = self.phase_oscillator
        self.reset(torch.arange(num_envs, device=self._device))

    @torch.no_grad()
    def reset(self, env_ids):
        """Reset ONLY specified environments, including reflex memory."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).flatten()
        if ids.numel() == 0:
            return
        self._mu_x[ids] = (0.5 * (self.mu_low + self.mu_up)) ** 2
        self._mu_z[ids] = self.nominal_clearance_single ** 2
        self.X[ids, 0, :] = 0.5 * (self.mu_low + self.mu_up)
        self.X[ids, 1, :] = self.nominal_clearance_single
        self.X[ids, 2, :] = torch.remainder(
            self.gait_phase_delta if self.gait_phase_delta.ndim == 1
            else self.gait_phase_delta[ids], 2 * np.pi
        )
        self.X_dot[ids] = 0.0
        self.d2X[ids] = 0.0
        self.omega[ids] = self.f_gait[ids] if torch.is_tensor(self.f_gait) else self.f_gait
        self.phase_correction[ids] = 0.0
        self.q_early[ids] = 0.0
        self.q_late[ids] = 0.0
        self.q_obstacle[ids] = 0.0
        self.stance_extension[ids] = 0.0
        self.kuramoto_term[ids] = 0.0
        self.reflex_term[ids] = 0.0
        self.effective_coupling[ids] = 1.0
        self._contact[ids] = False
        # Base-state auxiliary fields are set for legacy consumers.
        self.theta_unwrapped[ids] = 0.0
        self.delta_unwrapped[ids] = 0.0
        self.prev_delta_raw[ids] = 0.0
        self.prev_phase_correction_sign[ids] = 1.0
        self.phi[env_ids, :] = 0.

    @staticmethod
    def _approach(signal, trigger, dt, tau_on, tau_off):
        tau = torch.where(trigger, tau_on, tau_off)
        blend = 1.0 - torch.exp(-dt / tau)
        return signal + blend * (trigger.to(signal.dtype) - signal)

    def _vertical_phase(self):
        """Trajectory phase: [0, pi) swing, [pi, 2*pi) stance."""
        return self.X[:, 2, :]

    @torch.no_grad()
    def _update_reflex(self, dt):
        theta = self._vertical_phase()
        pi = np.pi
        if not self._contact_available:
            # No sensor supplied: never infer "missing contact" from zeros.
            trig_early = trig_late = trig_obstacle = torch.zeros_like(self._contact)
        else:
            contact = self._contact
            # Early swing collision -> clearance, not touchdown.
            trig_obstacle = (contact & (theta > 0.15 * pi)
                             & (theta < 0.5 * pi))
            # Trusted contact only near the expected swing -> stance boundary.
            trig_early = (contact & (theta >= 0.15 * pi)
                          & (theta < 0.8 * pi))
            # self.X[:,2,:][trig_early] = pi
            # self.X[:,0,:][trig_early] = self.mu_low
            # Missing contact after nominal touchdown -> delay + ground search.
            trig_late = ((~contact) & (theta >= 1.12 * pi)
                         & (theta <= 1.74 * pi))
            # self.X[:,0,:][trig_late] = self.mu_low
            # self.X[:,1,:][trig_late] = self.mu_up
            # trig_late = torch.zeros_like(self._contact)


        tau_on = torch.as_tensor(self.tau_reflex_on, device=self._device)
        tau_off = torch.as_tensor(self.tau_reflex_off, device=self._device)
        # self.q_early = self._approach(
        #     self.q_early, trig_early, dt, tau_on, tau_off)
        # self.q_late = self._approach(
        #     self.q_late, trig_late, dt, tau_on, tau_off)
        # self.q_obstacle = self._approach(
        #     self.q_obstacle, trig_obstacle, dt, tau_on, tau_off)
        self.q_obstacle = trig_obstacle * ((theta - 0.15 * pi) / (0.7 * pi - 0.15 * pi))
        self.q_early = trig_early * (1. - ((theta - 0.15 * pi) / (0.8 * pi - 0.15 * pi)))
        self.q_late = trig_late * ((theta - 1.12 * pi) / (1.74 * pi - 1.12 * pi))

    @torch.no_grad()
    def _kuramoto_reflex_substep(self, dt):
        self._update_reflex(dt)
        theta = self.X[:, 2, :]

        # Smooth local suppression. Symmetry g_i*g_j preserves reciprocity.
        q = torch.maximum(torch.maximum(self.q_early, self.q_late),
                          self.q_obstacle)
        self.effective_coupling = 1.0 - self.reflex_uncoupling * q
        edge_gain = torch.sqrt(
            self.effective_coupling[:, :, None]
            * self.effective_coupling[:, None, :]
        )

        # theta_j - theta_i - (delta_j - delta_i)
        phase_difference = theta[:, None, :] - theta[:, :, None]
        residual = phase_difference - self.target_diff
        self.kuramoto_term = self.K_c * (
            self.A[None] * edge_gain * torch.sin(residual)
        ).sum(dim=-1)

        self.reflex_term = self.omega_g * (
            - self.phase_early_ratio * self.q_early
            + self.phase_late_ratio * self.q_late
            - self.phase_obstacle_ratio * self.q_obstacle
        )

        # self.update_unwrapped_phase_command()
        delta_raw = self.gait_phase_delta[..., 1:]
        self.delta[:, 0] = 0.
        self.delta[:, 1:] = delta_raw
        theta_target = self.phi + self.delta
        phase_error = self.wrap_to_pi(theta_target - self.X[:,2,:])
        # ============================================================
        # Anti-podal branch hysteresis
        # ============================================================
        phase_margin = 0.05 * torch.pi
        zero_eps = 1e-6
        near_antipodal = (
            torch.abs(phase_error)
            > torch.pi - phase_margin
        )
        candidate_sign = torch.sign(phase_error)
        # Do NOT allow zero to erase the stored direction
        candidate_sign = torch.where(
            torch.abs(phase_error) > zero_eps,
            candidate_sign,
            self.prev_phase_correction_sign,
        )
        # Keep previous convergence direction near +/- pi
        stable_sign = torch.where(
            near_antipodal,
            self.prev_phase_correction_sign,
            candidate_sign,
        )
        phase_error_stable = torch.where(
            near_antipodal,
            stable_sign * torch.abs(phase_error),
            phase_error,
        )
        # Update memory only after leaving ambiguous region
        self.prev_phase_correction_sign = torch.where(
            near_antipodal,
            self.prev_phase_correction_sign,
            candidate_sign,
        )

        omega_corr_max = 0.3 * self.omega_g            #2. * torch.pi * 4.
        kappa = 0.6 * self.omega_g
        phase_correction_des = (
            omega_corr_max
            * torch.tanh(
                kappa
                * phase_error_stable
                / omega_corr_max
            )
        )


        max_correction = self.max_phase_correction_ratio * self.omega_g
        desired_correction = torch.clamp(
            phase_correction_des,
            # phase_correction_des + self.reflex_term,
            min=-max_correction, max=max_correction,
        )
        blend_phase = 1.0 - np.exp(-dt / self.tau_phase_correction)
        self.phase_correction += blend_phase * (
            desired_correction - self.phase_correction
        )
        omega = torch.clamp(
            self.omega_g + self.phase_correction,
            min=0.10 * self.omega_g,
        )
        self.omega = omega / (2.0 * np.pi)  # legacy: Hz

        # Same substep for BOTH phase and amplitude -- no dt mismatch.
        sqrt_mu = torch.stack([
            torch.sqrt(self._mu_x.clamp_min(0.0)),
            torch.sqrt(self._mu_z.clamp_min(0.0)),
        ], dim=1)
        r = self.X[:, :2, :]
        r_dot = self.X_dot[:, :2, :]
        alpha = self._OSCILLATOR_GAIN
        r_accel = alpha * (alpha * 0.25 * (sqrt_mu - r) - r_dot)
        r_dot = r_dot + dt * r_accel
        r = r + dt * r_dot
        self.X[:, :2, :] = r
        self.X_dot[:, :2, :] = r_dot
        self.d2X[:, :2, :] = r_accel
        self.X_dot[:, 2, :] = omega
        self.X[:, 2, :] = torch.remainder(theta + dt * omega, 2 * np.pi)
        self.phi = torch.remainder(self.phi + dt * self.omega_g, 2 * np.pi)

    @torch.no_grad()
    def phase_oscillator(self):
        steps = max(1, int(np.ceil(self._dt / self._INTEGRATION_DT)))
        dt = self._dt / steps  # correct even if dt not a multiple of 1 ms
        for i in range(steps):
            self._kuramoto_reflex_substep(dt)

    @torch.no_grad()
    def intrinsic_dynamics(
        self, actions, command, foot_contact=None, contact_forces=None
    ):
        """Return foot offsets [N,4], same as original.

        actions: [N, >=5]. First 4 = mu_x normalized actions, column 4 =
                 shared gait frequency adjustment. Columns 5+ are unused.
        command: [N, >=3] (vx, vy, yaw_rate), legacy mapping uses vx+yaw.
        foot_contact: [N,4] bool in [FL,FR,RL,RR] order, OR
        contact_forces: [N,4] normal forces [N] in same order.
        Passing neither disables reflex activation (no fabricated misses).
        """
        if actions.ndim != 2 or actions.shape != (self.num_envs, actions.shape[1]) or actions.shape[1] < 5:
            raise ValueError("actions must have shape [num_envs, >=5]")
        if command.ndim != 2 or command.shape[0] != self.num_envs or command.shape[1] < 3:
            raise ValueError("command must have shape [num_envs, >=3]")
        self._set_contact(foot_contact, contact_forces)

        a = actions.clamp(-1.0, 1.0)
        self._mu_x.copy_(self._scale_helper(
            a[:, :4], self.mu_low**2, self.mu_up**2
        ))

        # self.nominal_clearance.copy_(self._scale_helper(
        #     a[:, 4:8], 0.0, self._GROUND_CLEARANCE
        # ))

        freq = self.f_gait + self.f_modulation * a[:, 4:5]
        if (freq <= 0).any():
            raise ValueError("gait frequency must be positive")
        self.omega_g = 2.0 * np.pi * freq

        self.oscillator()

        return self._foot_trajectory(command, self.X[:, 2, :])

    def _set_contact(self, foot_contact, contact_forces):
        """Normalize optional contact input for the reflex integrator."""
        if foot_contact is not None and contact_forces is not None:
            raise ValueError("Pass only one of foot_contact or contact_forces")
        if foot_contact is None and contact_forces is None:
            self._contact_available = False
        else:
            source = foot_contact if foot_contact is not None else contact_forces
            source = torch.as_tensor(source, device=self._device)
            if source.shape != (self.num_envs, 4):
                raise ValueError("contact input must have shape [num_envs,4]")
            if foot_contact is not None:
                self._contact.copy_(source.bool())
            else:
                self._contact.copy_(source > self.contact_force_threshold)
            self._contact_available = True

    def _foot_trajectory(self, command, vertical_phase):
        """Map oscillator amplitudes/phases to foot offsets in leg order."""
        theta = self.X[:, 2, :]
        rx = self.X[:, 0, :].clamp(self.mu_low, self.mu_up)
        rx_norm = ((rx - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0)
        amplitude_x = self._MAX_STEP_LEN * rx_norm
        if hasattr(self, '_gait_step_scales'):
            # print(self._gait_step_scales)
            amplitude_x = amplitude_x * self._gait_step_scales[self._gait_ids, None]
        vx = command[:, 0:1] - command[:, 2:3] * self._FOOT_Y[None, :]
        vy = command[:, 2:3] * self._FOOT_X[None, :]
        planar_norm = torch.sqrt(vx.square() + vy.square())
        moving = planar_norm > 1.0e-4
        norm_safe = planar_norm.clamp_min(1.0e-4)
        stroke = -amplitude_x * torch.cos(self._horizontal_phase(theta, vertical_phase))
        self.x = torch.where(moving, stroke * (vx / norm_safe), 0.0)
        self.x[:,2:] = self.x[:,2:].clip(
            max=getattr(self, 'rear_forward_stroke_limit', 0.07))
        # print(self.x)
        self.y = torch.where(moving, stroke * (vy / norm_safe), 0.0)

        swing = torch.sin(vertical_phase).clamp_min(0.0)
        stance = (-torch.sin(vertical_phase)).clamp_min(0.0)

        rz = self.X[:, 1, :].clamp(self.mu_low, self.mu_up)
        rz_norm = ((rz - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0)
        amplitude_z_swing = (self.max_clearance - 0.1) * rz_norm + 0.1
        amplitude_z_stance = (self.max_stance_extension) * rz_norm
        self.stance_extension.copy_(amplitude_z_stance)
        self.z = (
            -self._robot_height
            + amplitude_z_swing * swing
            - 0. * stance
        )
        return self.x, self.y, self.z

    def _horizontal_phase(self, theta, vertical_phase):
        return theta

class Multi_Primitive_Dynamics(Interoceptive_Reflex_Dynamics):
    def __init__(self, time_step=0.001,
                 robot_height=0.3,
                 num_envs=1,
                 device=None,
                 mu_low=0,
                 mu_up=1,
                 L=0.2,
                 W=0.1,
                 init_phase=None,
                 integration_dt=0.001,
                 f_gait=8,
                 f_modulation=2,
                 coupling_gain=32,
                 tau_phase_correction=0.012,
                 max_phase_correction_ratio=0.48,
                 tau_reflex_on=0.008,
                 tau_reflex_off=0.055,
                 reflex_uncoupling=0.85,
                 phase_early_ratio=5.,
                 phase_late_ratio=0.1,
                 phase_obstacle_ratio=0.1,
                 nominal_clearance=0.15,
                 max_clearance=0.2,
                 max_stance_extension=0.02,
                 tau_stance_extension=0.025,
                 contact_force_threshold=1):

        # (name, phase offsets / pi in FL/FR/RL/RR order, stance fraction, Hz)
        primitives = (
            ("walk", (0., -1., -.5, .5), .6, 6.),
            # ("pace", (0., -1., 0., -1.), .50, 6.),
            ("trot", (0., -1., -1., 0.), .50, 8.),
            ("canter", (0., .6, .6, -.8), .45, 8.),
            ("bound", (0., 0., .5, .5), .45, 10.),
            # A longer stance gives the synchronized legs time to push off.
            # ("pronk", (0., 0., 0., 0.), .45, 6.),
            # ("gallop", (0., -.2, .8, -1.), .35, 10.),
        )
        self.primitive_names = tuple(row[0] for row in primitives)
        self.primitive_dict = {i: list(row[1]) for i, row in enumerate(primitives)}
        self.primitive_duty_factors = {i: row[2] for i, row in enumerate(primitives)}
        self.primitive_frequencies = {row[0]: row[3] for row in primitives}
        # Library values supersede the legacy scalar constructor argument.
        f_gait = self.primitive_frequencies[self.primitive_names[0]]
        if init_phase is None:
            init_phase = self.primitive_dict[0]

        super().__init__(
            time_step=time_step,
            robot_height=robot_height,
            num_envs=num_envs,
            device=device,
            mu_low=mu_low,
            mu_up=mu_up,
            L=L,
            W=W,
            init_phase=init_phase,
            integration_dt=integration_dt,
            f_gait=f_gait,
            f_modulation=f_modulation,
            tau_phase_correction=tau_phase_correction,
            max_phase_correction_ratio=max_phase_correction_ratio,
            tau_reflex_on=tau_reflex_on,
            tau_reflex_off=tau_reflex_off,
            reflex_uncoupling=reflex_uncoupling,
            phase_early_ratio=phase_early_ratio,
            phase_late_ratio=phase_late_ratio,
            phase_obstacle_ratio=phase_obstacle_ratio,
            nominal_clearance=nominal_clearance,
            max_clearance=max_clearance,
            max_stance_extension=max_stance_extension,
            tau_stance_extension=tau_stance_extension,
            contact_force_threshold=contact_force_threshold,
        )

        self._primitive_phases = torch.tensor(
            list(self.primitive_dict.values()), device=self._device
        ) * np.pi
        self._primitive_duty_factors = torch.tensor(
            [self.primitive_duty_factors[i] for i in self.primitive_dict],
            device=self._device,
        )
        self._primitive_frequencies = torch.tensor(
            [self.primitive_frequencies[name] for name in self.primitive_names],
            device=self._device,
        )
        self.f_gait = self._primitive_frequencies[:1].expand(num_envs, 1).clone()
        self.gait_phase_delta = self.gait_phase_delta.expand(num_envs, -1).clone()
        self.target_diff = (
            self.gait_phase_delta[:, None, :] - self.gait_phase_delta[:, :, None]
        )
        self.delta.copy_(self.gait_phase_delta)
        self._gait_ids = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self.omega_g = torch.full((num_envs, 4), 2 * np.pi * f_gait,
                                  device=self._device)
        self.pronk_preload_depth = 0.05
        self.pronk_preload_fraction = 0.35
        self.pronk_retract_fraction = 0.1

    def _vertical_phase(self):
        """Map actual phase to the shared vertical-trajectory/reflex phase."""
        theta = torch.remainder(self.X[:, 2, :], 2 * np.pi)
        beta = self.selected_duty_factors(self._gait_ids)
        swing_end = 2 * np.pi * (1.0 - beta)
        return torch.where(
            theta < swing_end,
            theta / (2.0 * (1.0 - beta)),
            np.pi + (theta - swing_end) / (2.0 * beta),
        )

    def _foot_trajectory(self, command, vertical_phase):
        """Use a preload, sustained push-off and smooth flight tuck for pronk."""
        x, y, ordinary_z = super()._foot_trajectory(command, vertical_phase)
        pronk = (self._gait_ids == 4)[:, None]

        def smoothstep(progress):
            progress = progress.clamp(0.0, 1.0)
            return progress.square() * (3.0 - 2.0 * progress)

        stance_progress = ((vertical_phase - np.pi) / np.pi).clamp(0.0, 1.0)
        # Retain the full extension through the nominal takeoff boundary.
        preload = self.pronk_preload_depth * smoothstep(
            stance_progress / self.pronk_preload_fraction)
        push = (self.pronk_preload_depth + self.stance_extension) * smoothstep(
            (stance_progress - self.pronk_preload_fraction)
            / (1.0 - self.pronk_preload_fraction))
        stance_z = -self._robot_height + preload - push

        swing_progress = (vertical_phase / np.pi).clamp(0.0, 1.0)
        retract = smoothstep(swing_progress / self.pronk_retract_fraction)
        rz = self.X[:, 1, :].clamp(self.mu_low, self.mu_up)
        rz_norm = ((rz - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0)
        clearance = 0.1 + (self.max_clearance - 0.1) * rz_norm
        swing_z = (-self._robot_height
                   - self.stance_extension * (1.0 - retract)
                   + clearance * torch.sin(vertical_phase).square())
        pronk_z = torch.where(vertical_phase >= np.pi, stance_z, swing_z)
        self.z = torch.where(pronk, pronk_z, ordinary_z)
        return x, y, self.z

    def selected_duty_factors(self, gait_ids):
        return self._primitive_duty_factors[gait_ids, None]

    def selected_frequencies(self, gait_ids):
        return self._primitive_frequencies[gait_ids, None]

    @torch.no_grad()
    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).flatten()
        if ids.numel() == 0:
            return
        # The parent constructor calls reset before the batched table exists.
        if self.gait_phase_delta.ndim == 1:
            super().reset(ids)
            return
        self.gait_phase_delta[ids] = self._primitive_phases[0]
        self._gait_ids[ids] = 0
        self.f_gait[ids] = self._primitive_frequencies[0]
        super().reset(ids)
        self.delta[ids] = self.gait_phase_delta[ids]
        self.target_diff[ids] = (
            self.gait_phase_delta[ids, None, :] - self.gait_phase_delta[ids, :, None]
        )
        self.omega_g[ids] = 2.0 * np.pi * self.f_gait[ids]
        self.nominal_clearance[ids] = self.nominal_clearance_single

    @torch.no_grad()
    def intrinsic_dynamics(
        self, actions, command, foot_contact=None, contact_forces=None
    ):
        """Actions: [N, >=9] = gait ID, four stride and four clearance values.

        Extra columns are ignored; frequency and clearance use library defaults.

        Continuous actions are normalized to [-1, 1]; gait IDs are integers.
        A gait ID change restarts that environment's reference clock at zero.
        Actual leg phases and correction memory remain continuous.
        """
        if actions.ndim != 2 or actions.shape[0] != self.num_envs or actions.shape[1] < 9:
            raise ValueError("actions must have shape [num_envs, >=9]")
        if command.ndim != 2 or command.shape[0] != self.num_envs or command.shape[1] < 3:
            raise ValueError("command must have shape [num_envs, >=3]")
        gait_ids = actions[:, 0]
        if not (torch.isfinite(gait_ids) & (gait_ids == gait_ids.long())
                & (gait_ids >= 0) & (gait_ids < len(self.primitive_dict))).all():
            raise ValueError(f"gait IDs must be integers in [0, {len(self.primitive_dict) - 1}]")
        gait_ids = gait_ids.long()
        a = actions[:, 1:].clamp(-1.0, 1.0)
        self._set_contact(foot_contact, contact_forces)
        self.gait_phase_delta.copy_(self._primitive_phases[gait_ids])
        self.target_diff.copy_(
            self.gait_phase_delta[:, None, :] - self.gait_phase_delta[:, :, None]
        )
        self._mu_x.copy_(self._scale_helper(
            a[:, :4], self.mu_low**2, self.mu_up**2
        ))
        # self.nominal_clearance -= self.offset_target[:,:,2]
        self._mu_z.copy_(self._scale_helper(
            a[:, 4:8], self.mu_low**2, self.mu_up**2
        ))
        self.f_gait.copy_(self.selected_frequencies(gait_ids))
        self.omega_g.copy_(2.0 * np.pi * self.f_gait)
        # self.omega_g += a[:,8:9] * 2.

        # Start the new absolute phase template at the transition instant.
        # Repeated commands for the same gait must keep the clock running.
        switched = gait_ids != self._gait_ids
        self.phi[switched] = 0.0
        self._gait_ids.copy_(gait_ids)

        self.oscillator()

        return self._foot_trajectory(command, self._vertical_phase())
