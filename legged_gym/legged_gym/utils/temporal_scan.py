"""Reproject a sampled static terrain patch between exteroceptive frames."""

import torch


def scan_due(episode_steps, period):
    """Select environments completing their next full scan period."""
    if type(period) is not int or period < 1:
        raise ValueError('scan period must be a positive integer')
    return (episode_steps + 1).remainder(period) == 0


def _yaw(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


class TemporalScanCache:
    """World-fixed low/high height samples and their capture pose per environment."""

    def __init__(self, x, y, num_envs, device):
        if (x.ndim != 1 or y.ndim != 1 or x.numel() < 2 or y.numel() < 2
                or not torch.all(x[1:] > x[:-1]) or not torch.all(y[1:] > y[:-1])):
            raise ValueError('scan axes must be strictly increasing and contain two points')
        self.x = x.to(device)
        self.y = y.to(device)
        self.num_envs = num_envs
        self.low = torch.zeros(num_envs, len(x), len(y), device=device, dtype=x.dtype)
        self.high = torch.zeros_like(self.low)
        self.xy = torch.zeros(num_envs, 2, device=device, dtype=x.dtype)
        self.yaw = torch.zeros(num_envs, device=device, dtype=x.dtype)
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.frame_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.frame_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    def capture(self, env_ids, low, high, xy, quat, step):
        ids = torch.as_tensor(env_ids, device=self.x.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        shape = (len(ids), self.x.numel() * self.y.numel())
        if low.shape != shape or high.shape != shape or xy.shape != (len(ids), 2) or quat.shape != (len(ids), 4):
            raise ValueError('capture shape does not match environment ids or scan grid')
        self.low[ids] = low.reshape(len(ids), len(self.x), len(self.y))
        self.high[ids] = high.reshape(len(ids), len(self.x), len(self.y))
        self.xy[ids] = xy
        self.yaw[ids] = _yaw(quat)
        self.frame_step[ids] = torch.as_tensor(step, device=self.x.device, dtype=torch.long)
        self.frame_count[ids] += 1
        self.valid[ids] = True

    def project(self, xy, quat):
        if not self.valid.all():
            raise RuntimeError('capture a terrain frame for every environment before project')
        if xy.shape != (self.num_envs, 2) or quat.shape != (self.num_envs, 4):
            raise ValueError('project pose shape does not match environments')
        local_x, local_y = torch.meshgrid(self.x, self.y, indexing='ij')
        current_yaw = _yaw(quat)
        c = current_yaw.cos()[:, None, None]
        s = current_yaw.sin()[:, None, None]
        world_x = xy[:, :1, None] + c * local_x - s * local_y
        world_y = xy[:, 1:, None] + s * local_x + c * local_y
        dx = world_x - self.xy[:, :1, None]
        dy = world_y - self.xy[:, 1:, None]
        c0 = self.yaw.cos()[:, None, None]
        s0 = self.yaw.sin()[:, None, None]
        old_x = c0 * dx + s0 * dy
        old_y = -s0 * dx + c0 * dy
        known = ((old_x >= self.x[0] - 1e-6) & (old_x <= self.x[-1] + 1e-6)
                 & (old_y >= self.y[0] - 1e-6) & (old_y <= self.y[-1] + 1e-6))
        ix1 = torch.searchsorted(self.x, old_x.contiguous()).clamp(max=len(self.x) - 1)
        iy1 = torch.searchsorted(self.y, old_y.contiguous()).clamp(max=len(self.y) - 1)
        ix0 = (ix1 - 1).clamp_min(0)
        iy0 = (iy1 - 1).clamp_min(0)
        ix1 = torch.where((old_x - self.x[ix0]).abs() <= 1e-5, ix0, ix1)
        iy1 = torch.where((old_y - self.y[iy0]).abs() <= 1e-5, iy0, iy1)
        ix0 = torch.where((old_x - self.x[ix1]).abs() <= 1e-5, ix1, ix0)
        iy0 = torch.where((old_y - self.y[iy1]).abs() <= 1e-5, iy1, iy0)
        ix0 = torch.where(old_x >= self.x[-1], ix1, ix0)
        iy0 = torch.where(old_y >= self.y[-1], iy1, iy0)

        def corners(values):
            flat = values.flatten(1)
            return torch.stack([
                flat.gather(1, (ix * len(self.y) + iy).flatten(1)).reshape_as(old_x)
                for ix, iy in ((ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1))
            ])

        return corners(self.low).amin(0), corners(self.high).amax(0), known
