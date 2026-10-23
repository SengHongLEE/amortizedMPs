"""Per-environment terrain levels driven by completed locomotion episodes."""
import numpy as np
import torch


class TerrainCurriculum:
    def __init__(self, cfg, difficulties, num_envs, device, play=False):
        if not isinstance(cfg.curriculum, bool):
            raise ValueError('terrain.curriculum must be True or False')
        self.enabled = cfg.curriculum and not play
        self.cfg = cfg

        def level(value, name):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < cfg.num_rows:
                raise ValueError(f'{name} must be an integer row index in [0, {cfg.num_rows - 1}]')
            return value

        self.minimum = level(cfg.curriculum_min_level, 'curriculum_min_level')
        self.maximum = level(cfg.num_rows - 1 if cfg.curriculum_max_level is None
                             else cfg.curriculum_max_level, 'curriculum_max_level')
        if self.minimum > self.maximum:
            raise ValueError('curriculum_min_level must not exceed curriculum_max_level')
        if cfg.fixed_terrain_level is not None:
            level(cfg.fixed_terrain_level, 'fixed_terrain_level')
        if self.enabled:
            initial_max = level(cfg.curriculum_initial_level, 'curriculum_initial_level')
            if not self.minimum <= initial_max <= self.maximum:
                raise ValueError('curriculum_initial_level must lie between the curriculum level limits')
            if cfg.difficulty_mode == 'random' and cfg.row_difficulties is None:
                raise ValueError('curriculum requires linear difficulty or explicit nondecreasing row_difficulties')
            if not np.isfinite(difficulties).all() or np.any(np.diff(difficulties, axis=0) < 0):
                raise ValueError('curriculum requires nondecreasing difficulty rows in every column')
            distance = cfg.curriculum_promote_distance
            available = min(cfg.terrain_length, cfg.terrain_width) / 2 - cfg.boundary_margin - cfg.spawn_xy_jitter
            if not np.isfinite(distance) or not 0 < distance < available:
                raise ValueError('curriculum_promote_distance must be positive and inside the usable tile half-length')
            up, down = cfg.curriculum_promote_ratio, cfg.curriculum_demote_ratio
            if not np.isfinite([up, down]).all() or not 0 <= down < up <= 1:
                raise ValueError('curriculum ratios must satisfy 0 <= demote < promote <= 1')
            self.levels = torch.randint(self.minimum, initial_max + 1, (num_envs,), device=device)
        elif cfg.fixed_terrain_level is not None:
            self.levels = torch.full((num_envs,), cfg.fixed_terrain_level, device=device, dtype=torch.long)
        else:
            initial_max = level(cfg.num_rows - 1 if cfg.max_init_terrain_level is None
                                else cfg.max_init_terrain_level, 'max_init_terrain_level')
            self.levels = torch.randint(0, initial_max + 1, (num_envs,), device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.command_displacement = torch.zeros_like(self.start_xy)
        self.command_distance = torch.zeros(num_envs, device=device)

    def reset(self, env_ids, xy):
        self.start_xy[env_ids] = xy
        self.command_displacement[env_ids] = 0.
        self.command_distance[env_ids] = 0.

    def accumulate(self, commands, quaternion, dt):
        """Integrate the command used by this step, before command resampling."""
        if not self.enabled:
            return
        x, y, z, w = quaternion.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx, vy = commands[:, 0], commands[:, 1]
        self.command_displacement[:, 0] += (c * vx - s * vy) * dt
        self.command_displacement[:, 1] += (s * vx + c * vy) * dt
        self.command_distance += torch.linalg.vector_norm(commands[:, :2], dim=1) * dt

    def update(self, env_ids, xy, failed, episode_steps):
        """Update only completed episodes; stay at the hardest level after success."""
        if not self.enabled or len(env_ids) == 0:
            return
        desired = self.command_displacement[env_ids]
        length = torch.linalg.vector_norm(desired, dim=1)
        direction = desired / length.clamp_min(1e-6).unsqueeze(1)
        progress = ((xy - self.start_xy[env_ids]) * direction).sum(dim=1)
        commanded = self.command_distance[env_ids]
        moving = (commanded > 1e-6) & (length > 1e-6)
        promote = (~failed & moving & (progress >= self.cfg.curriculum_promote_distance)
                   & (progress >= self.cfg.curriculum_promote_ratio * commanded))
        demote = failed | (moving & (progress < self.cfg.curriculum_demote_ratio * commanded))
        change = (promote.long() - demote.long()) * (episode_steps > 0).long()
        self.levels[env_ids] = (self.levels[env_ids] + change).clamp(self.minimum, self.maximum)
