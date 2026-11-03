"""Compact a dense terrain scan while keeping both raised and recessed surfaces."""

import torch


def scan_bin_ids(x, y, bins_x, bins_y):
    """Map a body-frame x/y grid to fixed spatial bins in row-major order."""
    if x.ndim != 1 or y.ndim != 1 or x.numel() < bins_x or y.numel() < bins_y:
        raise ValueError('scan grid must have at least one point per bin on each axis')
    if bins_x < 1 or bins_y < 1:
        raise ValueError('scan bin counts must be positive')
    if not torch.all(x[1:] > x[:-1]) or not torch.all(y[1:] > y[:-1]):
        raise ValueError('scan axes must be strictly increasing')
    x_step = (x[1:] - x[:-1]).median() if len(x) > 1 else x.new_tensor(1.)
    y_step = (y[1:] - y[:-1]).median() if len(y) > 1 else y.new_tensor(1.)
    bx = ((x - x[0]) * bins_x / (x[-1] - x[0] + x_step)).floor().long().clamp(0, bins_x - 1)
    by = ((y - y[0]) * bins_y / (y[-1] - y[0] + y_step)).floor().long().clamp(0, bins_y - 1)
    ids = (bx[:, None] * bins_y + by[None, :]).reshape(-1)
    if torch.unique(ids).numel() != bins_x * bins_y:
        raise ValueError('dense scan grid leaves an empty compact bin')
    return ids


def signed_scan_observations(low_heights, high_heights, base_height,
                             bin_ids, bins_x, bins_y, scale):
    """Return low/high observed heights, preserving positive and negative terrain.

    `low_heights` and `high_heights` contain the local minimum and maximum
    terrain height around every dense point. The output is two channels, each
    ordered as x-major/y-minor spatial bins: raised terrain first, recesses second.
    """
    if (low_heights.shape != high_heights.shape or low_heights.ndim != 2
            or bin_ids.numel() != low_heights.shape[1]):
        raise ValueError('height tensors and bin ids must have compatible shapes')
    count = bins_x * bins_y
    if count < 1 or bin_ids.min() < 0 or bin_ids.max() >= count:
        raise ValueError('scan bin ids are out of range')
    ids = bin_ids.to(device=low_heights.device).expand(low_heights.shape[0], -1)
    minimum = torch.full((low_heights.shape[0], count), float('inf'),
                         device=low_heights.device, dtype=low_heights.dtype)
    maximum = torch.full_like(minimum, -float('inf'))
    minimum.scatter_reduce_(1, ids, low_heights, reduce='amin', include_self=True)
    maximum.scatter_reduce_(1, ids, high_heights, reduce='amax', include_self=True)
    raised = (base_height - maximum).clamp(-1., 1.) * scale
    recessed = (base_height - minimum).clamp(-1., 1.) * scale
    return torch.cat((raised, recessed), dim=-1)
