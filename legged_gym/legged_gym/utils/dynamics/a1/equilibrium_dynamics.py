"""Dynamical trajectory equilibria for proprioceptive A1 locomotion."""
import math
import numpy as np
import torch

from .rhythmic_dynamics import Multi_Primitive_Dynamics


class EquilibriumPrimitiveDynamics(Multi_Primitive_Dynamics):
    """Actions: gait, four amplitudes, then bx/by/bz targets (FL/FR/RL/RR).

    Offsets are in hip-frame metres, before the anatomical lateral hip offset.
    The target drives a critically damped second-order system. Contact or the
    nominal stance phase limits offset speed; it does not freeze the CPG stroke.
    """

    ACTION_SPACE = 'cpg_equilibrium_v1'
    NUM_ACTIONS = 18
    NUM_OBSERVATIONS = 70

    def __init__(self, equilibrium_cfg, **kwargs):
        super().__init__(**kwargs)
        cfg = equilibrium_cfg
        self.offset_limits = self._positive_xyz(cfg.offset_limits, 'offset_limits')
        self.swing_velocity_limits = self._positive_xyz(
            cfg.swing_velocity_limits, 'swing_velocity_limits')
        self.stance_velocity_limits = self._positive_xyz(
            cfg.stance_velocity_limits, 'stance_velocity_limits')
        self.acceleration_limits = self._positive_xyz(
            cfg.acceleration_limits, 'acceleration_limits')
        if (self.stance_velocity_limits > self.swing_velocity_limits).any():
            raise ValueError('stance_velocity_limits must not exceed swing_velocity_limits')
        self.equilibrium_frequency = float(cfg.frequency_hz)
        if not math.isfinite(self.equilibrium_frequency) or not 0 < self.equilibrium_frequency <= 20:
            raise ValueError('equilibrium frequency_hz must be in (0, 20]')
        if not math.isfinite(self._dt) or self._dt <= 0:
            raise ValueError('time_step must be positive and finite')
        self.offset = torch.zeros(self.num_envs, 4, 3, device=self._device)
        self.offset_velocity = torch.zeros_like(self.offset)
        self.offset_target = torch.zeros_like(self.offset)

    def _positive_xyz(self, values, name):
        result = torch.as_tensor(values, dtype=torch.float, device=self._device)
        if result.shape != (3,) or not (torch.isfinite(result) & (result >= 0)).all():
            raise ValueError(f'{name} must contain three positive finite XYZ values')
        return result

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        # The parent constructor calls reset before these states are allocated.
        if hasattr(self, 'offset'):
            self.offset[env_ids] = 0.
            self.offset_velocity[env_ids] = 0.
            self.offset_target[env_ids] = 0.

    @torch.no_grad()
    def _kuramoto_reflex_substep(self, dt):
        self._update_reflex(dt)
        theta = self.X[:, 2, :]

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
        target_nominal_clearance = (self.max_clearance - 0.1) * ((torch.sqrt(self.nominal_clearance.clamp_min(0.0)) - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0) + 0.1
        target_z = target_nominal_clearance
        self._mu_z = target_z.square()
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
    def intrinsic_dynamics(self, actions, command, foot_contact=None, contact_forces=None):
        if actions.shape != (self.num_envs, EquilibriumPrimitiveDynamics.NUM_ACTIONS):
            raise ValueError('equilibrium actions must have shape [num_envs, 17]')
        nominal = torch.stack(super().intrinsic_dynamics(
            actions[:, :9], command, foot_contact, contact_forces), dim=-1)

        equilibrium_x = self._scale_helper(actions[:, 9:13], self.mu_low ** 2, self.mu_up ** 2)
        target_equilibrium_x = (self.offset_limits[0]) * (2 * ((torch.sqrt(equilibrium_x.clamp_min(0.0)) - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0) - 1)
        self.offset_target[:,:,:1].copy_(
            target_equilibrium_x.reshape(-1, 1, 4).transpose(1, 2))

        equilibrium_y = torch.ones_like(actions[:, 9:13]) * self.offset_limits[1]
        self.offset_target[:,:,1:2].copy_(
            equilibrium_y.reshape(-1, 1, 4).transpose(1, 2))

        equilibrium_z = self._scale_helper(actions[:, 13:17], self.mu_low ** 2, self.mu_up ** 2)
        target_equilibrium_z = (self.offset_limits[2]) * ((torch.sqrt(equilibrium_z.clamp_min(0.0)) - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0.0, 1.0)
        self.offset_target[:,:,2:].copy_(
           target_equilibrium_z.reshape(-1, 1, 4).transpose(1, 2))

        stance = (self._vertical_phase() >= math.pi) | self._contact


        omega = 2. * math.pi * self.equilibrium_frequency
        # self.swing_velocity_limits[0] = 2.
        # self.swing_velocity_limits[2] = 2.
        # self.stance_velocity_limits[0] = 0.8
        # self.stance_velocity_limits[2] = 0.8

        speed_limit = torch.where(
            stance[..., None], self.stance_velocity_limits, self.swing_velocity_limits)


        steps = max(1, math.ceil(self._dt / .001))
        dt = self._dt / steps
        for _ in range(2 * steps):
            acceleration = (omega**2 * (self.offset_target - self.offset)
                            - 2. * omega * self.offset_velocity)
            acceleration = torch.clamp(
                acceleration, -self.acceleration_limits, self.acceleration_limits)
            velocity = torch.clamp(
                self.offset_velocity + dt * acceleration, -speed_limit, speed_limit)
            candidate = self.offset + dt * velocity
            bounded = torch.clamp(candidate, -self.offset_limits, self.offset_limits)
            # Anti-windup at position limits. A contact transition may also
            # instantly reduce speed; acceleration limits apply between gates.
            self.offset_velocity.copy_(torch.where(
                candidate == bounded, velocity, torch.zeros_like(velocity)))
            self.offset.copy_(bounded)

        # print(self.offset)
        feet = nominal + self.offset
        return feet.unbind(dim=-1)

    def controller_observations(self):
        """70 dimensionless controller states; no exteroceptive information."""
        phase = self.X[:, 2]
        amplitude_scale = self.X.new_tensor([self.mu_up, self.max_clearance])[None, :, None]
        # alpha/2 is the amplitude filter's natural angular frequency.
        amplitude_rate_scale = amplitude_scale * (self._OSCILLATOR_GAIN / 2.)
        return torch.cat((
            phase.sin(), phase.cos(),                       # 8
            self.phi[:, :1].sin(), self.phi[:, :1].cos(),      # 2 (shared clock)
            (self.X[:, :2] / amplitude_scale).flatten(1),     # 8
            (self.X_dot[:, :2] / amplitude_rate_scale).flatten(1),  # 8
            self.phase_correction / self.omega_g,            # 4
            self.q_early, self.q_late, self.q_obstacle,       # 12
            self.stance_extension / max(self.max_stance_extension, 1e-6),  # 4
            (self.offset / self.offset_limits).transpose(1, 2).flatten(1),  # 12
            (self.offset_velocity / self.swing_velocity_limits).transpose(1, 2).flatten(1),  # 12
        ), dim=-1)

    def continuous_action_change_cost(self, previous, current):
        return (current - previous).square().sum(-1)

    def bounded_actions(self, actions, limit):
        """Bound applied actions and their observation history in one place."""
        bounded = actions.clone()
        bounded[:, 1:].clamp_(-limit, limit)
        return bounded


class AdaptiveEquilibriumDynamics(EquilibriumPrimitiveDynamics):
    """v1 plus four lift targets, a clock multiplier and a stance-duty offset.

    New actions are neutral at zero. Timing parameters follow a first-order
    dynamical system; lift uses the existing second-order amplitude dynamics.
    All state is proprioceptive/controller-internal, without terrain lookahead.
    """

    ACTION_SPACE = 'cpg_equilibrium_v2'
    NUM_ACTIONS = 23
    NUM_OBSERVATIONS = 72

    def __init__(self, timing_cfg, **kwargs):
        super().__init__(**kwargs)
        cfg = timing_cfg
        def interval(values, name):
            if len(values) != 2 or not all(math.isfinite(float(x)) for x in values) or values[0] >= values[1]:
                raise ValueError(f'{name} must be an increasing finite pair')
            return tuple(float(x) for x in values)
        self.frequency_range = interval(cfg.frequency_scale_range, 'frequency_scale_range')
        self.duty_offset_range = interval(cfg.duty_offset_range, 'duty_offset_range')
        self.duty_range = interval(cfg.duty_factor_range, 'duty_factor_range')
        self.clearance_range = interval(cfg.clearance_range, 'clearance_range')
        self.timing_tau = float(cfg.time_constant)
        if not (0 < self.frequency_range[0] <= 1 <= self.frequency_range[1] <= 2):
            raise ValueError('frequency_scale_range must contain neutral 1 within (0, 2]')
        if not (self.duty_offset_range[0] <= 0 <= self.duty_offset_range[1]):
            raise ValueError('duty_offset_range must contain neutral 0')
        if not (0 < self.duty_range[0] < self.duty_range[1] < 1):
            raise ValueError('duty_factor_range must lie strictly inside (0, 1)')
        if not (0 < self.clearance_range[0] <= self.nominal_clearance_single <= self.clearance_range[1] <= self.max_clearance):
            raise ValueError('clearance_range must contain nominal clearance and not exceed max_clearance')
        if not math.isfinite(self.timing_tau) or self.timing_tau <= 0:
            raise ValueError('time_constant must be positive and finite')
        self.align_stroke = bool(cfg.align_stroke)
        self.frequency_scale = torch.ones(self.num_envs, 1, device=self._device)
        self.duty_offset = torch.zeros_like(self.frequency_scale)

    @staticmethod
    def _neutral_map(actions, low, neutral, high):
        return neutral + actions * torch.where(actions >= 0, high - neutral, neutral - low)

    def selected_frequencies(self, gait_ids):
        return super().selected_frequencies(gait_ids) * self.frequency_scale

    def selected_duty_factors(self, gait_ids):
        return (super().selected_duty_factors(gait_ids) + self.duty_offset).clamp(*self.duty_range)

    def _horizontal_phase(self, theta, vertical_phase):
        return vertical_phase if self.align_stroke else theta

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, 'frequency_scale'):
            self.frequency_scale[env_ids] = 1.
            self.duty_offset[env_ids] = 0.

    @torch.no_grad()
    def intrinsic_dynamics(self, actions, command, foot_contact=None, contact_forces=None):
        if actions.shape != (self.num_envs, self.NUM_ACTIONS):
            raise ValueError('adaptive equilibrium actions must have shape [num_envs, 23]')
        a = actions[:, 17:].clamp(-1., 1.)
        lift = self._neutral_map(a[:, :4], self.clearance_range[0],
                                 self.nominal_clearance_single, self.clearance_range[1])
        frequency = self._neutral_map(a[:, 4:5], self.frequency_range[0], 1., self.frequency_range[1])
        duty = self._neutral_map(a[:, 5:6], self.duty_offset_range[0], 0., self.duty_offset_range[1])
        rate = -math.expm1(-self._dt / self.timing_tau)
        self.frequency_scale.lerp_(frequency, rate)
        self.duty_offset.lerp_(duty, rate)
        self.nominal_clearance.copy_(lift)
        return super().intrinsic_dynamics(actions[:, :17], command, foot_contact, contact_forces)

    def controller_observations(self):
        frequency = self.frequency_scale - 1.
        frequency = frequency / torch.where(frequency >= 0, self.frequency_range[1] - 1.,
                                             1. - self.frequency_range[0]).clamp_min(1e-6)
        duty = self.duty_offset / torch.where(self.duty_offset >= 0, self.duty_offset_range[1],
                                             -self.duty_offset_range[0]).clamp_min(1e-6)
        return torch.cat((super().controller_observations(), frequency, duty), dim=-1)


class ModalEquilibriumDynamics(AdaptiveEquilibriumDynamics):
    """Axis-wise common/front-back/left-right/diagonal equilibrium modes.

    The four modes span the same four leg targets. Saturation happens after
    conversion to Cartesian targets, preserving physical position/speed bounds.
    This changes coordinates and exploration structure, not nominal trajectories.
    """
    ACTION_SPACE = 'cpg_modal_equilibrium_v3'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode_to_leg = self.X.new_tensor([
            [1., 1., 1., 1.], [1., 1., -1., -1.],
            [1., -1., 1., -1.], [1., -1., -1., 1.]])

    @torch.no_grad()
    def intrinsic_dynamics(self, actions, command, foot_contact=None, contact_forces=None):
        if actions.shape != (self.num_envs, self.NUM_ACTIONS):
            raise ValueError('modal equilibrium actions must have shape [num_envs, 23]')
        cartesian = actions.clone()
        cartesian[:, 5:17] = (actions[:, 5:17].reshape(-1, 3, 4) @ self.mode_to_leg.T).flatten(1)
        return super().intrinsic_dynamics(cartesian, command, foot_contact, contact_forces)

    def continuous_action_change_cost(self, previous, current):
        # H.T @ H = 4I: preserve the v2 Cartesian action-rate penalty.
        delta = current - previous
        return delta.square().sum(-1) + 3. * delta[:, 4:16].square().sum(-1)

    def bounded_actions(self, actions, limit):
        bounded = super().bounded_actions(actions, limit)
        # The old policy observes clipped leg actions. Preserve that history
        # under the basis change, including when raw Gaussian samples saturate.
        leg = actions[:, 5:17].reshape(-1, 3, 4) @ self.mode_to_leg.T
        leg = leg.clamp(-limit, limit)
        bounded[:, 5:17] = (leg @ self.mode_to_leg / 4.).flatten(1)
        return bounded


def bounded_inverse_kinematics(feet, robot, joint_limits, margin,
                              min_foot_depth=.06, min_sagittal_reach=.06):
    """Project hip-frame targets, solve IK, and return bounded q and its FK.

    Physical order is FL/FR/RL/RR. Input includes the signed anatomical hip
    offset. Radial projection avoids invalid square roots and leg extension
    singularities; final joint clipping enforces the URDF box with a margin.
    This guarantees kinematic joint bounds, not collision freedom or stability.
    """
    l1 = robot.hip_link_length_a1
    l2 = robot.thigh_link_length_a1
    l3 = robot.calf_link_length_a1
    lower = (joint_limits[:, 0] + margin).reshape(4, 3)
    upper = (joint_limits[:, 1] - margin).reshape(4, 3)
    x, y, z = feet.unbind(-1)
    z = z.clamp(max=-min_foot_depth)
    yz_radius = torch.sqrt(y.square() + z.square()).clamp_min(1e-6)
    # Keep a physical distance from the hip-axis singularity, not merely an
    # epsilon that avoids NaNs but amplifies tiny Cartesian changes.
    safe_radius = yz_radius.clamp_min(math.sqrt(l1*l1 + min_sagittal_reach**2))
    y, z = y * safe_radius / yz_radius, z * safe_radius / yz_radius
    sagittal = (safe_radius.square() - l1*l1).clamp_min(1e-6).sqrt()
    length = (x.square() + sagittal.square()).sqrt()
    # A1 uses the negative-knee IK branch; its limits lie inside (-pi, 0).
    min_length = (l2*l2 + l3*l3 + 2*l2*l3*lower[:, 2].cos()).sqrt()
    max_length = (l2*l2 + l3*l3 + 2*l2*l3*upper[:, 2].cos()).sqrt()
    length_scale = torch.clamp(length, min_length, max_length) / length
    x, sagittal = x * length_scale, sagittal * length_scale
    projected_radius = (sagittal.square() + l1*l1).sqrt()
    y, z = y * projected_radius / safe_radius, z * projected_radius / safe_radius
    d = ((x.square() + sagittal.square() - l2*l2 - l3*l3)
         / (2*l2*l3)).clamp(-1.+1e-6, 1.-1e-6)
    knee = -torch.acos(d)
    side = feet.new_tensor([1., -1., 1., -1.])
    hip = torch.atan2(z, y) + torch.atan2(sagittal, side * l1)
    thigh = torch.atan2(-x, sagittal) - torch.atan2(
        l3 * knee.sin(), l2 + l3 * knee.cos())
    q = torch.clamp(torch.stack((hip, thigh, knee), -1), lower, upper)
    h, t, k = q.unbind(-1)
    sagittal_z = -l2 * t.cos() - l3 * (t + k).cos()
    achieved = torch.stack((
        -l2 * t.sin() - l3 * (t + k).sin(),
        side * l1 * h.cos() - sagittal_z * h.sin(),
        side * l1 * h.sin() + sagittal_z * h.cos(),
    ), -1)
    return q.flatten(1), achieved
