"""First-episode traversal metrics, sampled before simulator auto-reset."""
import math
import torch


def resolved_band_width(width, horizontal_scale):
    """Return the band width after the terrain generator rounds to grid cells."""
    if width <= 0 or horizontal_scale <= 0:
        raise ValueError('width and horizontal_scale must be positive')
    return round(width / horizontal_scale) * horizontal_scale


def first_band_clear_distance(platform_size, obstacle_width, rear_margin=0.35):
    """Forward travel from the tile centre until the first band is behind the robot."""
    if platform_size <= 0 or obstacle_width <= 0 or rear_margin <= 0:
        raise ValueError('platform_size, obstacle_width and rear_margin must be positive')
    return platform_size / 2 + obstacle_width + rear_margin


def second_band_clear_distance(platform_size, obstacle_width, spacing, rear_margin=0.35):
    """Forward travel until the rear feet also clear the next repeated band."""
    if spacing <= obstacle_width:
        raise ValueError('spacing must exceed obstacle_width')
    return first_band_clear_distance(platform_size, obstacle_width, rear_margin) + spacing


def next_tile_second_band_clear_distance(tile_length, platform_size, obstacle_width,
                                         spacing, rear_margin=0.35):
    """Travel from one tile centre past the second band in the following tile."""
    if not math.isfinite(tile_length) or tile_length <= 0:
        raise ValueError('tile_length must be finite and positive')
    return tile_length + second_band_clear_distance(
        platform_size, obstacle_width, spacing, rear_margin)


def stair_evaluation_rows(heights, vertical_scale, allow_flat=False):
    """Return a height range and row difficulties with exact grid-resolved steps."""
    if not heights or any(not math.isfinite(h) or h < 0 or (h == 0 and not allow_flat)
                          for h in heights) or max(heights) <= 0:
        raise ValueError('stair heights must be finite and positive (an initial zero is allowed only for flat diagnostics)')
    if any(b <= a for a, b in zip(heights, heights[1:])):
        raise ValueError('stair heights must be strictly increasing')
    if not math.isfinite(vertical_scale) or vertical_scale <= 0:
        raise ValueError('vertical_scale must be finite and positive')
    maximum = heights[-1]
    difficulties = [height / maximum for height in heights]
    # Isaac Gym truncates step_height / vertical_scale. Check the actual
    # interpolation too, so a floating-point boundary cannot silently lose a cell.
    for height, difficulty in zip(heights, difficulties):
        resolved = int(maximum * difficulty / vertical_scale) * vertical_scale
        if not math.isclose(resolved, height, rel_tol=0., abs_tol=1e-10):
            raise ValueError('stair height cannot be represented exactly at this vertical_scale')
    return [0., maximum], difficulties


class TraversalMetrics:
    def __init__(self, count, device, distance, corridor):
        self.distance, self.corridor = distance, corridor
        self.active = torch.ones(count, device=device, dtype=torch.bool)
        self.success = torch.zeros_like(self.active)
        self.failed = torch.zeros_like(self.active)
        self.timeout = torch.zeros_like(self.active)
        self.duration = torch.zeros(count, device=device)
        self.max_forward = torch.full_like(self.duration, -float('inf'))
        self.max_lateral = torch.zeros_like(self.duration)
        self.speed_error_integral = torch.zeros_like(self.duration)
        self.final_forward = torch.zeros_like(self.duration)
        self.final_lateral = torch.zeros_like(self.duration)

    def update(self, xy, failed, timeout, speed_error, dt):
        active = self.active.clone()
        self.duration += active * dt
        self.max_forward = torch.where(active, torch.maximum(self.max_forward, xy[:, 0]), self.max_forward)
        self.final_forward = torch.where(active, xy[:, 0], self.final_forward)
        self.final_lateral = torch.where(active, xy[:, 1], self.final_lateral)
        self.max_lateral = torch.where(active, torch.maximum(self.max_lateral, xy[:, 1].abs()), self.max_lateral)
        self.speed_error_integral += torch.where(active, speed_error * dt, 0.)
        crossed = (xy[:, 0] >= self.distance) & (self.max_lateral <= self.corridor) & ~failed
        self.success |= active & crossed
        self.failed |= active & failed
        self.timeout |= active & timeout
        self.active &= ~(crossed | failed | timeout)

    def records(self):
        values = {k: getattr(self, k).cpu().tolist() for k in (
            'success', 'failed', 'timeout', 'duration', 'max_forward', 'max_lateral',
            'final_forward', 'final_lateral')}
        values['mean_speed_error'] = (self.speed_error_integral / self.duration.clamp_min(1e-6)).cpu().tolist()
        return [{key: value[i] for key, value in values.items()} for i in range(len(values['success']))]
