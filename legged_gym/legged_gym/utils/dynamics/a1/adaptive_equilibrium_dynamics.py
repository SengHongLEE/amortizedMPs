"""The four-gait, 19-action A1 controller used by the 1028 MP_full archive.

This controller owns its oscillator, contact response, trajectory and equilibrium
filter.  It inherits only the state allocation and scalar utilities from
``Intrinsic_Dynamics``; no other gait controller participates in its behavior.
"""

import math

import torch

from .rhythmic_dynamics import Intrinsic_Dynamics


class AdaptiveEquilibriumDynamics(Intrinsic_Dynamics):
    ACTION_SPACE = 'cpg_equilibrium_v3'
    NUM_ACTIONS = 19
    NUM_OBSERVATIONS = 46
    _GAITS = (
        ('walk', (0., -1., -.5, .5), .60, 6.),
        ('trot', (0., -1., -1., 0.), .50, 8.),
        ('canter', (0., .6, .6, -.8), .45, 8.),
        ('bound', (0., 0., .5, .5), .45, 10.),
    )

    def __init__(self, equilibrium_cfg, timing_cfg, *, integration_dt=.001,
                 nominal_clearance=.15, max_clearance=.2,
                 contact_force_threshold=1., tau_phase_correction=.012,
                 max_phase_correction_ratio=.48, phase_early_ratio=5.,
                 phase_late_ratio=.1, phase_obstacle_ratio=.1,
                 tau_reflex_on=.008, tau_reflex_off=.055,
                 max_stance_extension=.02, **kwargs):
        kwargs.setdefault('mu_low', 0.)
        kwargs.setdefault('mu_up', 1.)
        super().__init__(**kwargs)

        self.primitive_names = tuple(gait[0] for gait in self._GAITS)
        self.primitive_dict = {i: list(gait[1]) for i, gait in enumerate(self._GAITS)}
        self.primitive_duty_factors = {i: gait[2] for i, gait in enumerate(self._GAITS)}
        self.primitive_frequencies = {gait[0]: gait[3] for gait in self._GAITS}
        self._primitive_phases = torch.tensor(
            [gait[1] for gait in self._GAITS], device=self._device) * math.pi
        self._primitive_duty_factors = torch.tensor(
            [gait[2] for gait in self._GAITS], device=self._device)
        self._primitive_frequencies = torch.tensor(
            [gait[3] for gait in self._GAITS], device=self._device)
        self._gait_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.gait_phase_delta = self._primitive_phases[0].expand(self.num_envs, -1).clone()
        self.f_gait = self._primitive_frequencies[:1].expand(self.num_envs, 1).clone()
        self.omega_g = torch.full((self.num_envs, 4), 2 * math.pi * 6., device=self._device)
        self._INTEGRATION_DT = integration_dt
        self.tau_phase_correction = float(tau_phase_correction)
        self.max_phase_correction_ratio = float(max_phase_correction_ratio)
        self.phase_early_ratio = float(phase_early_ratio)
        self.phase_late_ratio = float(phase_late_ratio)
        self.phase_obstacle_ratio = float(phase_obstacle_ratio)
        self.tau_reflex_on = float(tau_reflex_on)
        self.tau_reflex_off = float(tau_reflex_off)
        self.contact_force_threshold = float(contact_force_threshold)
        self._contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self._device)
        self._contact_available = False
        self.q_early = torch.zeros(self.num_envs, 4, device=self._device)
        self.q_late = torch.zeros_like(self.q_early)
        self.q_obstacle = torch.zeros_like(self.q_early)
        self.reflex_term = torch.zeros_like(self.q_early)
        self.stance_extension = torch.zeros_like(self.q_early)
        self.nominal_clearance_single = float(nominal_clearance)
        self.nominal_clearance = torch.full_like(self.q_early, nominal_clearance)
        self.max_clearance = float(max_clearance)
        self.max_stance_extension = float(max_stance_extension)
        self.stance_extension_gait_id = None

        eq = equilibrium_cfg
        self.offset_limits = self._positive_xyz(eq.offset_limits, 'offset_limits')
        self.rear_x_offset_range = tuple(float(x) for x in getattr(
            eq, 'rear_x_offset_range', (-float(self.offset_limits[0]), 0.)))
        self.swing_velocity_limits = self._positive_xyz(
            eq.swing_velocity_limits, 'swing_velocity_limits')
        self.stance_velocity_limits = self._positive_xyz(
            eq.stance_velocity_limits, 'stance_velocity_limits')
        self.acceleration_limits = self._positive_xyz(
            eq.acceleration_limits, 'acceleration_limits')
        if (self.stance_velocity_limits > self.swing_velocity_limits).any():
            raise ValueError('stance_velocity_limits must not exceed swing_velocity_limits')
        self.equilibrium_frequency = float(eq.frequency_hz)
        if not math.isfinite(self.equilibrium_frequency) or not 0 < self.equilibrium_frequency <= 20:
            raise ValueError('equilibrium frequency_hz must be in (0, 20]')
        if not math.isfinite(self._dt) or self._dt <= 0:
            raise ValueError('time_step must be positive and finite')
        self.offset = torch.zeros(self.num_envs, 4, 3, device=self._device)
        self.offset_velocity = torch.zeros_like(self.offset)
        self.offset_target = torch.zeros_like(self.offset)

        timing = timing_cfg
        self.frequency_range = self._interval(timing.frequency_scale_range, 'frequency_scale_range')
        self.duty_offset_range = self._interval(timing.duty_offset_range, 'duty_offset_range')
        self.duty_range = self._interval(timing.duty_factor_range, 'duty_factor_range')
        self.timing_tau = float(timing.time_constant)
        if not (0 < self.frequency_range[0] <= 1 <= self.frequency_range[1] <= 2):
            raise ValueError('frequency_scale_range must contain neutral 1 within (0, 2]')
        if not (self.duty_offset_range[0] <= 0 <= self.duty_offset_range[1]):
            raise ValueError('duty_offset_range must contain neutral 0')
        if not (0 < self.duty_range[0] < self.duty_range[1] < 1):
            raise ValueError('duty_factor_range must lie strictly inside (0, 1)')
        if not math.isfinite(self.timing_tau) or self.timing_tau <= 0:
            raise ValueError('time_constant must be positive and finite')
        self.align_stroke = bool(timing.align_stroke)
        self.frequency_scale = torch.ones(self.num_envs, 1, device=self._device)
        self.duty_offset = torch.zeros_like(self.frequency_scale)
        self.oscillator = self.phase_oscillator
        self.reset(torch.arange(self.num_envs, device=self._device))

    def _positive_xyz(self, values, name):
        result = torch.as_tensor(values, dtype=torch.float, device=self._device)
        if result.shape != (3,) or not (torch.isfinite(result) & (result >= 0)).all():
            raise ValueError(f'{name} must contain three positive finite XYZ values')
        return result

    @staticmethod
    def _interval(values, name):
        if len(values) != 2 or not all(math.isfinite(float(x)) for x in values) or values[0] >= values[1]:
            raise ValueError(f'{name} must be an increasing finite pair')
        return tuple(float(x) for x in values)

    @staticmethod
    def _neutral_map(actions, low, neutral, high):
        return neutral + actions * torch.where(actions >= 0, high - neutral, neutral - low)

    @torch.no_grad()
    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).flatten()
        if ids.numel() == 0:
            return
        self.gait_phase_delta[ids] = self._primitive_phases[0]
        self._gait_ids[ids] = 0
        self.f_gait[ids] = self._primitive_frequencies[0]
        self._mu_x[ids] = (.5 * (self.mu_low + self.mu_up)) ** 2
        self._mu_z[ids] = self.nominal_clearance_single ** 2
        self.X[ids, 0] = .5 * (self.mu_low + self.mu_up)
        self.X[ids, 1] = self.nominal_clearance_single
        self.X[ids, 2] = torch.remainder(self.gait_phase_delta[ids], 2 * math.pi)
        self.X_dot[ids] = 0.
        self.d2X[ids] = 0.
        self.omega[ids] = self.f_gait[ids]
        self.omega_g[ids] = 2 * math.pi * self.f_gait[ids]
        self.phi[ids] = 0.
        self.delta[ids] = self.gait_phase_delta[ids]
        self.phase_correction[ids] = 0.
        self.prev_phase_correction_sign[ids] = 1.
        self._contact[ids] = False
        self.q_early[ids] = 0.
        self.q_late[ids] = 0.
        self.q_obstacle[ids] = 0.
        self.reflex_term[ids] = 0.
        self.stance_extension[ids] = 0.
        self.nominal_clearance[ids] = self.nominal_clearance_single
        self.theta_unwrapped[ids] = 0.
        self.delta_unwrapped[ids] = 0.
        self.prev_delta_raw[ids] = 0.
        self.offset[ids] = 0.
        self.offset_velocity[ids] = 0.
        self.offset_target[ids] = 0.
        self.offset[ids, 2:, 0] = self.rear_x_offset_range[1]
        self.offset_target[ids, 2:, 0] = self.rear_x_offset_range[1]
        self.frequency_scale[ids] = 1.
        self.duty_offset[ids] = 0.

    def selected_frequencies(self, gait_ids):
        return self._primitive_frequencies[gait_ids, None] * self.frequency_scale

    def selected_duty_factors(self, gait_ids):
        return (self._primitive_duty_factors[gait_ids, None] + self.duty_offset).clamp(*self.duty_range)

    def _vertical_phase(self):
        theta = torch.remainder(self.X[:, 2], 2 * math.pi)
        beta = self.selected_duty_factors(self._gait_ids)
        swing_end = 2 * math.pi * (1. - beta)
        return torch.where(theta < swing_end,
                           theta / (2. * (1. - beta)),
                           math.pi + (theta - swing_end) / (2. * beta))

    def _set_contact(self, foot_contact, contact_forces):
        if foot_contact is not None and contact_forces is not None:
            raise ValueError('Pass only one of foot_contact or contact_forces')
        if foot_contact is None and contact_forces is None:
            self._contact_available = False
            return
        source = torch.as_tensor(
            foot_contact if foot_contact is not None else contact_forces, device=self._device)
        if source.shape != (self.num_envs, 4):
            raise ValueError('contact input must have shape [num_envs,4]')
        self._contact.copy_(source.bool() if foot_contact is not None
                            else source > self.contact_force_threshold)
        self._contact_available = True

    @staticmethod
    def _approach(signal, trigger, dt, tau_on, tau_off):
        tau = torch.where(trigger, tau_on, tau_off)
        blend = 1. - torch.exp(-dt / tau)
        return signal + blend * (trigger.to(signal.dtype) - signal)

    @torch.no_grad()
    def _update_reflex(self, dt):
        theta = self._vertical_phase()
        pi = math.pi
        if not self._contact_available:
            trig_early = trig_late = trig_obstacle = torch.zeros_like(self._contact)
        else:
            contact = self._contact
            trig_obstacle = contact & (theta > .15 * pi) & (theta < .5 * pi)
            trig_early = contact & (theta >= .15 * pi) & (theta < .8 * pi)
            trig_late = ~contact & (theta >= 1.12 * pi) & (theta <= 1.74 * pi)
        self.q_obstacle = trig_obstacle * ((theta - .15 * pi) / (.7 * pi - .15 * pi))
        self.q_early = trig_early * (1. - ((theta - .15 * pi) / (.8 * pi - .15 * pi)))
        self.q_late = trig_late * ((theta - 1.12 * pi) / (1.74 * pi - 1.12 * pi))

    @torch.no_grad()
    def _kuramoto_reflex_substep(self, dt):
        self._update_reflex(dt)
        theta = self.X[:, 2]
        self.reflex_term = self.omega_g * (
            -self.phase_early_ratio * self.q_early
            + self.phase_late_ratio * self.q_late
            - self.phase_obstacle_ratio * self.q_obstacle)
        self.delta[:, 0] = 0.
        self.delta[:, 1:] = self.gait_phase_delta[:, 1:]
        phase_error = self.wrap_to_pi(self.phi + self.delta - self.X[:, 2])
        near_antipodal = phase_error.abs() > math.pi - .05 * math.pi
        candidate_sign = torch.where(phase_error.abs() > 1e-6,
                                     phase_error.sign(), self.prev_phase_correction_sign)
        stable_sign = torch.where(near_antipodal, self.prev_phase_correction_sign, candidate_sign)
        phase_error_stable = torch.where(near_antipodal,
                                         stable_sign * phase_error.abs(), phase_error)
        self.prev_phase_correction_sign = torch.where(
            near_antipodal, self.prev_phase_correction_sign, candidate_sign)
        omega_corr_max = .3 * self.omega_g
        kappa = .6 * self.omega_g
        phase_correction_des = omega_corr_max * torch.tanh(
            kappa * phase_error_stable / omega_corr_max)
        max_correction = self.max_phase_correction_ratio * self.omega_g
        desired_correction = torch.clamp(
            phase_correction_des, min=-max_correction, max=max_correction)
        blend_phase = 1. - math.exp(-dt / self.tau_phase_correction)
        self.phase_correction += blend_phase * (desired_correction - self.phase_correction)
        omega = torch.clamp(self.omega_g + self.phase_correction, min=.10 * self.omega_g)
        self.omega = omega / (2. * math.pi)
        sqrt_mu = torch.stack((self._mu_x.clamp_min(0.).sqrt(),
                               self._mu_z.clamp_min(0.).sqrt()), dim=1)
        r = self.X[:, :2]
        r_dot = self.X_dot[:, :2]
        alpha = self._OSCILLATOR_GAIN
        r_accel = alpha * (alpha * .25 * (sqrt_mu - r) - r_dot)
        r_dot = r_dot + dt * r_accel
        r = r + dt * r_dot
        self.X[:, :2] = r
        self.X_dot[:, :2] = r_dot
        self.d2X[:, :2] = r_accel
        self.X_dot[:, 2] = omega
        self.X[:, 2] = torch.remainder(theta + dt * omega, 2 * math.pi)
        self.phi = torch.remainder(self.phi + dt * self.omega_g, 2 * math.pi)

    @torch.no_grad()
    def phase_oscillator(self):
        steps = max(1, math.ceil(self._dt / self._INTEGRATION_DT))
        dt = self._dt / steps
        for _ in range(steps):
            self._kuramoto_reflex_substep(dt)

    def _foot_trajectory(self, command, vertical_phase):
        theta = self.X[:, 2]
        rx = self.X[:, 0].clamp(self.mu_low, self.mu_up)
        amplitude_x = self._MAX_STEP_LEN * (
            (rx - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0., 1.)
        if hasattr(self, '_gait_step_scales'):
            amplitude_x = amplitude_x * self._gait_step_scales[self._gait_ids, None]
        vx = command[:, :1] - command[:, 2:3] * self._FOOT_Y[None]
        vy = command[:, 2:3] * self._FOOT_X[None]
        planar_norm = torch.sqrt(vx.square() + vy.square())
        moving = planar_norm > 1e-4
        stroke_phase = vertical_phase if self.align_stroke else theta
        stroke = -amplitude_x * torch.cos(stroke_phase)
        self.x = torch.where(moving, stroke * (vx / planar_norm.clamp_min(1e-4)), 0.)
        self.x[:, 2:] = self.x[:, 2:].clip(max=getattr(self, 'rear_forward_stroke_limit', .07))
        self.y = torch.where(moving, stroke * (vy / planar_norm.clamp_min(1e-4)), 0.)
        swing = torch.sin(vertical_phase).clamp_min(0.)
        rz = self.X[:, 1].clamp(self.mu_low, self.mu_up)
        rz_norm = ((rz - self.mu_low) / (self.mu_up - self.mu_low)).clamp(0., 1.)
        amplitude_z_swing = (self.max_clearance - .1) * rz_norm + .1
        self.stance_extension.copy_(self.max_stance_extension * rz_norm)
        self.z = -self._robot_height + amplitude_z_swing * swing
        return self.x, self.y, self.z

    @torch.no_grad()
    def intrinsic_dynamics(self, actions, command, foot_contact=None, contact_forces=None):
        if actions.shape != (self.num_envs, self.NUM_ACTIONS):
            raise ValueError('adaptive equilibrium actions must have shape [num_envs, 19]')
        if command.ndim != 2 or command.shape[0] != self.num_envs or command.shape[1] < 3:
            raise ValueError('command must have shape [num_envs, >=3]')
        timing = actions[:, 17:].clamp(-1., 1.)
        frequency = self._neutral_map(timing[:, :1], *(
            self.frequency_range[0], 1., self.frequency_range[1]))
        duty = self._neutral_map(timing[:, 1:2], *(
            self.duty_offset_range[0], 0., self.duty_offset_range[1]))
        rate = -math.expm1(-self._dt / self.timing_tau)
        self.frequency_scale.lerp_(frequency, rate)
        self.duty_offset.lerp_(duty, rate)

        gait_ids = actions[:, 0]
        if not (torch.isfinite(gait_ids) & (gait_ids == gait_ids.long())
                & (gait_ids >= 0) & (gait_ids < len(self._GAITS))).all():
            raise ValueError(f'gait IDs must be integers in [0, {len(self._GAITS) - 1}]')
        gait_ids = gait_ids.long()
        a = actions[:, 1:17].clamp(-1., 1.)
        self._set_contact(foot_contact, contact_forces)
        self.gait_phase_delta.copy_(self._primitive_phases[gait_ids])
        self._mu_x.copy_(self._scale_helper(a[:, :4], self.mu_low**2, self.mu_up**2))
        self._mu_z.copy_(self._scale_helper(a[:, 4:8], self.mu_low**2, self.mu_up**2))
        self.f_gait.copy_(self.selected_frequencies(gait_ids))
        self.omega_g.copy_(2. * math.pi * self.f_gait)
        switched = gait_ids != self._gait_ids
        self.phi[switched] = 0.
        self._gait_ids.copy_(gait_ids)
        self.oscillator()
        nominal = torch.stack(self._foot_trajectory(command, self._vertical_phase()), dim=-1)

        equilibrium_x = self._scale_helper(actions[:, 9:13], self.mu_low**2, self.mu_up**2)
        target_x = self.offset_limits[0] * (2 * (
            (equilibrium_x.clamp_min(0.).sqrt() - self.mu_low)
            / (self.mu_up - self.mu_low)).clamp(0., 1.) - 1)
        target_x[:, :2] = target_x[:, :2].clip(min=-.05)
        rear_low, rear_high = self.rear_x_offset_range
        target_x[:, 2:] = rear_low + .5 * (
            actions[:, 11:13].clamp(-1., 1.) + 1.) * (rear_high - rear_low)
        self.offset_target[:, :, 0] = target_x
        self.offset_target[:, :, 1] = self.offset_limits[1]
        equilibrium_z = self._scale_helper(actions[:, 13:17], self.mu_low**2, self.mu_up**2)
        target_z = self.offset_limits[2] * (2 * (
            (equilibrium_z.clamp_min(0.).sqrt() - self.mu_low)
            / (self.mu_up - self.mu_low)).clamp(0., 1.) - 1)
        self.offset_target[:, :, 2] = target_z
        stance = (self._vertical_phase() >= math.pi) | self._contact
        omega = 2. * math.pi * self.equilibrium_frequency
        speed_limit = torch.where(
            stance[..., None], self.stance_velocity_limits, self.swing_velocity_limits)
        steps = max(1, math.ceil(self._dt / .001))
        dt = self._dt / steps
        for _ in range(steps):
            acceleration = omega**2 * (self.offset_target - self.offset) - 2. * omega * self.offset_velocity
            acceleration = torch.clamp(acceleration, -self.acceleration_limits, self.acceleration_limits)
            velocity = torch.clamp(self.offset_velocity + dt * acceleration, -speed_limit, speed_limit)
            candidate = self.offset + dt * velocity
            bounded = torch.clamp(candidate, -self.offset_limits, self.offset_limits)
            bounded[:, 2:, 0].clamp_(*self.rear_x_offset_range)
            self.offset_velocity.copy_(torch.where(candidate == bounded, velocity, torch.zeros_like(velocity)))
            self.offset.copy_(bounded)
        return (nominal + self.offset).unbind(dim=-1)

    def controller_observations(self):
        phase = self.X[:, 2]
        amplitude_scale = self.X.new_tensor([self.mu_up, self.max_clearance])[None, :, None]
        base = torch.cat((
            phase.sin(), phase.cos(),
            (self.X[:, :2] / amplitude_scale).flatten(1),
            self.phase_correction / self.omega_g,
            (self.offset / self.offset_limits).transpose(1, 2).flatten(1),
            (self.offset_velocity / self.swing_velocity_limits).transpose(1, 2).flatten(1),
        ), dim=-1)
        frequency = self.frequency_scale - 1.
        frequency = frequency / torch.where(
            frequency >= 0, self.frequency_range[1] - 1.,
            1. - self.frequency_range[0]).clamp_min(1e-6)
        duty = self.duty_offset / torch.where(
            self.duty_offset >= 0, self.duty_offset_range[1],
            -self.duty_offset_range[0]).clamp_min(1e-6)
        return torch.cat((base, frequency, duty), dim=-1)

    def continuous_action_change_cost(self, previous, current):
        return (current - previous).square().sum(-1)

    def bounded_actions(self, actions, limit):
        bounded = actions.clone()
        bounded[:, 1:].clamp_(-limit, limit)
        return bounded
