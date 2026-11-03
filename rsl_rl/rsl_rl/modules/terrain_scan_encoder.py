"""Small learned encoder for a bounded two-channel terrain scan."""

import torch
import torch.nn as nn


class TerrainScanEncoder(nn.Module):
    def __init__(self, scan_shape, output_dim):
        super().__init__()
        if (len(scan_shape) != 3 or scan_shape[0] != 2
                or any(not isinstance(size, int) or size < 1 for size in scan_shape)
                or output_dim < 3):
            raise ValueError('expected two scan channels and at least three encoded features')
        self.scan_shape = tuple(scan_shape)
        self.scan_dim = scan_shape[0] * scan_shape[1] * scan_shape[2]
        self.output_dim = output_dim
        spatial_features = 8 * scan_shape[1] * scan_shape[2]
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(spatial_features, output_dim - 2), nn.Tanh(),
        )

    def forward(self, scan):
        if tuple(scan.shape[-3:]) != self.scan_shape:
            raise ValueError('terrain scan shape does not match encoder layout')
        leading = scan.shape[:-3]
        flat = scan.reshape(-1, *self.scan_shape)
        reference = flat[:, :, 0, :].mean(dim=(1, 2))
        raised_peak = (reference[:, None, None] - flat[:, 0]).amax(dim=(1, 2)).clamp_min(0.)
        recessed_peak = (flat[:, 1] - reference[:, None, None]).amax(dim=(1, 2)).clamp_min(0.)
        learned = self.features(flat)
        return torch.cat((raised_peak[:, None], recessed_peak[:, None], learned), -1).reshape(
            *leading, self.output_dim)


class TerrainEncodedActor(nn.Module):
    """Present a compact terrain feature vector to an ordinary actor."""

    def __init__(self, actor, input_dim, scan_shape, encoder_dim):
        super().__init__()
        self.actor = actor
        self.input_dim = input_dim
        self.terrain_encoder = TerrainScanEncoder(scan_shape, encoder_dim)
        if input_dim <= self.terrain_encoder.scan_dim:
            raise ValueError('actor observation must include non-terrain features')

    def forward(self, observations):
        if observations.shape[-1] != self.input_dim:
            raise ValueError('actor observation dimension does not match terrain layout')
        scan = observations[..., -self.terrain_encoder.scan_dim:].reshape(
            *observations.shape[:-1], *self.terrain_encoder.scan_shape)
        compact = torch.cat((observations[..., :-self.terrain_encoder.scan_dim],
                             self.terrain_encoder(scan)), dim=-1)
        return self.actor(compact)
