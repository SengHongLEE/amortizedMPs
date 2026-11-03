"""Explicit inference-only actor decisions from observed terrain and proprioception."""

from dataclasses import dataclass
import math
from typing import Optional

import torch


_SAFETY_LOOKAHEAD_SKIPS = {'gait': 2, 'modulation': 3}


def format_policy_update_frequency(gait_forwards, modulation_forwards, active_env_steps, dt):
    """Summarize actual actor forwards per simulated environment second."""
    env_seconds = active_env_steps * dt
    gait_hz = gait_forwards / env_seconds if env_seconds > 0 else 0.
    modulation_hz = modulation_forwards / env_seconds if env_seconds > 0 else 0.
    return ('Average policy update frequency (per env, simulated time): '
            f'gait: {gait_hz:.2f} Hz ({gait_forwards} forwards), '
            f'modulation: {modulation_hz:.2f} Hz ({modulation_forwards} forwards)')


@dataclass
class ForwardHoldConfig:
    gait_max_skips: int = 2
    modulation_max_skips: int = 3
    tilt_limit: float = .25
    horizontal_rate_limit: float = 2.
    modulation_horizontal_rate_limit: Optional[float] = 4.
    filter_normal_sway: bool = False
    sway_filter_alpha: float = .5
    sway_tilt_guard: float = .1
    horizontal_rate_emergency_limit: float = 2.
    tracking_error_trigger: bool = True
    tracking_linear_error_limit: float = .6
    tracking_yaw_error_limit: float = .6
    tracking_filter_alpha: float = .1
    tracking_warmup_s: float = .5
    obstacle_height_m: float = .05
    extero_edge_trigger: bool = True
    terrain_edge_height_m: float = .035
    corridor_half_width_m: float = .2
    response_s: float = .10
    distance_margin_m: float = .15

    def __post_init__(self):
        if type(self.filter_normal_sway) is not bool:
            raise ValueError('filter_normal_sway must be a boolean')
        if type(self.extero_edge_trigger) is not bool:
            raise ValueError('extero_edge_trigger must be a boolean')
        if type(self.tracking_error_trigger) is not bool:
            raise ValueError('tracking_error_trigger must be a boolean')
        if not math.isfinite(self.sway_filter_alpha) or not 0 < self.sway_filter_alpha <= 1:
            raise ValueError('sway_filter_alpha must be finite and within (0, 1]')
        if (not math.isfinite(self.horizontal_rate_emergency_limit)
                or self.horizontal_rate_emergency_limit < 0):
            raise ValueError('horizontal_rate_emergency_limit must be finite and nonnegative')
        if (self.modulation_horizontal_rate_limit is not None
                and (not math.isfinite(self.modulation_horizontal_rate_limit)
                     or self.modulation_horizontal_rate_limit < 0)):
            raise ValueError('modulation_horizontal_rate_limit must be finite and nonnegative')
        if not math.isfinite(self.tracking_filter_alpha) or not 0 < self.tracking_filter_alpha <= 1:
            raise ValueError('tracking_filter_alpha must be finite and within (0, 1]')
        if not math.isfinite(self.terrain_edge_height_m) or self.terrain_edge_height_m <= 0:
            raise ValueError('terrain_edge_height_m must be finite and positive')
        for name in ('gait_max_skips', 'modulation_max_skips'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('tilt_limit', 'horizontal_rate_limit',
                     'tracking_linear_error_limit', 'tracking_yaw_error_limit',
                     'tracking_warmup_s',
                     'sway_tilt_guard',
                     'obstacle_height_m', 'corridor_half_width_m',
                     'response_s', 'distance_margin_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


class InferenceForwardHold:
    """Filter fixed-clock actor candidates; never sample terrain or update training state."""

    def __init__(self, config, x, y, num_envs, device):
        self.config = config
        self.x = torch.as_tensor(x, device=device)
        self.y = torch.as_tensor(y, device=device)
        if self.x.ndim != 1 or self.y.ndim != 1 or self.x.numel() < 2 or self.y.numel() < 2:
            raise ValueError('dense scan axes must be one-dimensional and nonempty')
        grid_x, grid_y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.grid_x = grid_x
        self.corridor = grid_y.abs() <= config.corridor_half_width_m + 1e-6
        self.center_cols = self.y.abs() <= .1 + 1e-6
        self.near = (grid_x <= .3 + 1e-6) & (grid_y.abs() <= .1 + 1e-6)
        if not self.near.any():
            raise ValueError('scan grid needs a near-field reference patch')
        self.skipped = {name: torch.zeros(num_envs, device=device, dtype=torch.long)
                        for name in ('gait', 'modulation')}
        self.has_forward = {name: torch.zeros(num_envs, device=device, dtype=torch.bool)
                            for name in ('gait', 'modulation')}
        self.last_command = {name: torch.zeros(num_envs, 3, device=device)
                             for name in ('gait', 'modulation')}
        self.sway_filtered = torch.zeros(num_envs, 2, device=device)
        self.sway_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.tracking_linear_filtered = torch.zeros(num_envs, 2, device=device)
        self.tracking_yaw_filtered = torch.zeros(num_envs, device=device)
        self.tracking_last_command = torch.zeros(num_envs, 3, device=device)
        self.tracking_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def observe(self, episode_steps, ang_vel):
        """Update the inference-only horizontal angular velocity filter every control step."""
        if not self.config.filter_normal_sway:
            return
        current = ang_vel[:, :2]
        restart = (episode_steps == 0) | ~self.sway_valid
        alpha = self.config.sway_filter_alpha
        self.sway_filtered[:] = torch.where(
            restart[:, None], current,
            alpha * current + (1. - alpha) * self.sway_filtered)
        self.sway_valid[:] = torch.isfinite(current).all(1)

    def observe_tracking(self, episode_steps, commands, lin_vel, ang_vel):
        """Filter body-frame command tracking errors at the inference control rate."""
        if not self.config.tracking_error_trigger:
            return
        command = commands[:, :3]
        linear_error = command[:, :2] - lin_vel[:, :2]
        yaw_error = command[:, 2] - ang_vel[:, 2]
        restart = ((episode_steps == 0) | ~self.tracking_valid
                   | ((command - self.tracking_last_command).abs().amax(1) > 1e-3))
        alpha = self.config.tracking_filter_alpha
        self.tracking_linear_filtered[:] = torch.where(
            restart[:, None], linear_error,
            alpha * linear_error + (1. - alpha) * self.tracking_linear_filtered)
        self.tracking_yaw_filtered[:] = torch.where(
            restart, yaw_error,
            alpha * yaw_error + (1. - alpha) * self.tracking_yaw_filtered)
        self.tracking_last_command[:] = command
        self.tracking_valid[:] = (torch.isfinite(command).all(1)
                                  & torch.isfinite(linear_error).all(1)
                                  & torch.isfinite(yaw_error))

    def select(self, actor, candidates, episode_steps, commands, lin_vel, ang_vel,
               gravity, low, high, known, scan_age_steps, scan_period_steps, dt,
               period_steps):
        """Return ids to forward and overlapping reason masks for candidate ids."""
        if actor not in self.skipped:
            raise ValueError('actor must be gait or modulation')
        ids = candidates.flatten()
        if ids.numel() == 0:
            return ids, {}
        cap = getattr(self.config, f'{actor}_max_skips')
        steps = episode_steps[ids]
        command = commands[ids][:, (0, 1, 3)]
        first = (steps == 0) | ~self.has_forward[actor][ids]
        cap_reached = self.skipped[actor][ids] >= cap
        command_change = (command - self.last_command[actor][ids]).abs().amax(1) > 1e-3
        horizontal_rate = ang_vel[ids, :2].norm(dim=1)
        tilt = gravity[ids, :2].norm(dim=1)
        rate_limit = (self.config.modulation_horizontal_rate_limit
                      if actor == 'modulation' and self.config.modulation_horizontal_rate_limit is not None
                      else self.config.horizontal_rate_limit)
        if self.config.filter_normal_sway:
            filtered_rate = torch.where(
                self.sway_valid[ids, None], self.sway_filtered[ids], ang_vel[ids, :2]
            ).norm(dim=1)
            horizontal_trigger = ((filtered_rate > rate_limit)
                                  | (horizontal_rate > max(self.config.horizontal_rate_emergency_limit,
                                                           rate_limit))
                                  | ((horizontal_rate > rate_limit)
                                     & (tilt > self.config.sway_tilt_guard)))
        else:
            horizontal_trigger = horizontal_rate > rate_limit
        proprio = (tilt > self.config.tilt_limit) | horizontal_trigger
        if self.config.tracking_error_trigger:
            linear_error = torch.where(
                self.tracking_valid[ids, None], self.tracking_linear_filtered[ids],
                commands[ids, :2] - lin_vel[ids, :2]).norm(dim=1)
            yaw_error = torch.where(
                self.tracking_valid[ids], self.tracking_yaw_filtered[ids],
                commands[ids, 2] - ang_vel[ids, 2]).abs()
            tracking = ((steps * dt >= self.config.tracking_warmup_s)
                        & ((linear_error > self.config.tracking_linear_error_limit)
                           | (yaw_error > self.config.tracking_yaw_error_limit)))
        else:
            tracking = torch.zeros_like(first)
        terrain_invalid = known is None or low is None or high is None
        if terrain_invalid:
            hazard = torch.ones_like(first)
            invalid = torch.ones_like(first)
        else:
            low_part, high_part, known_part = low[ids], high[ids], known[ids]
            finite = (torch.isfinite(low_part).flatten(1).all(1)
                      & torch.isfinite(high_part).flatten(1).all(1))
            near_known = known_part[:, self.near].all(1)
            reference = ((low_part[:, self.near] + high_part[:, self.near]) / 2).median(1).values
            abnormal = ((low_part - reference[:, None, None]).abs() > self.config.obstacle_height_m
                        ) | ((high_part - reference[:, None, None]).abs() > self.config.obstacle_height_m)
            nearest_obstacle = torch.where(
                abnormal & known_part & self.corridor, self.grid_x, float('inf')).flatten(1).amin(1)
            if self.config.extero_edge_trigger:
                center_low = low_part[:, :, self.center_cols]
                center_high = high_part[:, :, self.center_cols]
                center_known = known_part[:, :, self.center_cols]
                profile = ((center_low + center_high) / 2).median(-1).values
                edge = (profile[:, 1:] - profile[:, :-1]).abs() >= self.config.terrain_edge_height_m
                edge_known = center_known[:, 1:].all(-1) & center_known[:, :-1].all(-1)
                nearest_edge = torch.where(
                    edge & edge_known, self.x[None, :-1], float('inf')).amin(1)
            nearest_unknown = torch.where(
                ~known_part & self.corridor, self.grid_x, float('inf')).flatten(1).amin(1)
            nearest_unknown = torch.minimum(nearest_unknown, self.x[-1])
            speed = torch.maximum(lin_vel[ids, 0], commands[ids, 0]).clamp_min(0.)
            # Candidate decisions still run at the fixed actor clock. A large
            # hold cap must not turn unseen terrain several seconds away into
            # an immediate forward trigger.
            remaining = (cap - self.skipped[actor][ids]).clamp_min(0)
            lookahead_skips = remaining.clamp_max(_SAFETY_LOOKAHEAD_SKIPS[actor])
            lookahead_s = (lookahead_skips * period_steps * dt + scan_period_steps * dt
                           + scan_age_steps[ids] * dt + self.config.response_s)
            distance = speed * lookahead_s + self.config.distance_margin_m
            obstacle_hazard = nearest_obstacle <= distance
            if self.config.extero_edge_trigger:
                obstacle_hazard &= nearest_edge <= distance
            hazard = obstacle_hazard | (nearest_unknown <= distance)
            invalid = ~finite | ~near_known
        finite_proprio = (torch.isfinite(command).all(1)
                          & torch.isfinite(lin_vel[ids]).all(1)
                          & torch.isfinite(ang_vel[ids]).all(1)
                          & torch.isfinite(gravity[ids]).all(1))
        invalid = invalid | ~finite_proprio
        reasons = dict(first=first, cap=cap_reached, command=command_change,
                       proprio=proprio, tracking=tracking, terrain=hazard, invalid=invalid)
        forward = torch.stack(list(reasons.values())).any(0)
        self.skipped[actor][ids] = torch.where(forward, 0, self.skipped[actor][ids] + 1)
        selected = ids[forward]
        self.has_forward[actor][selected] = True
        self.last_command[actor][selected] = command[forward]
        reasons['hold'] = ~forward
        return selected, reasons
