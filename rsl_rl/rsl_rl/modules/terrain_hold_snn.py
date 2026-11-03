"""Phase 1: frozen gait scan encoder and two supervised hold-time heads.

Durations are integer environment control steps, not physics substeps or seconds.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .snn import SNNActor


@dataclass(frozen=True)
class HoldBins:
    gait: tuple
    modulation: tuple

    def __post_init__(self):
        for name, values, minimum in (
            ('gait', self.gait, 15), ('modulation', self.modulation, 3),
        ):
            if (not values or any(type(v) is not int for v in values)
                    or values[0] < minimum
                    or any(a >= b for a, b in zip(values, values[1:]))):
                raise ValueError(f'{name} bins must be increasing control steps >= {minimum}')


def verified_hold_label(bins, outcomes):
    """Return (largest verified safe step count, right_censored, usable).

    An unobserved candidate includes a window cut short by episode end. A failure
    stops the safe prefix, even if a longer candidate happens to pass.
    """
    if len(bins) != len(outcomes) or any(s not in ('pass', 'fail', 'unobserved') for s in outcomes):
        raise ValueError('one pass/fail/unobserved result is required per bin')
    safe = bins[0]
    usable = False
    for step, result in zip(bins, outcomes):
        if result == 'pass':
            safe, usable = step, True
        else:
            return safe, result == 'unobserved', usable
    return safe, True, usable


def overestimation_counts(predicted, verified_safe, right_censored):
    """Return known overestimates, identifiable examples, unknown examples.

    For a censored sample, a prediction at or below its verified lower bound is
    known safe; a larger prediction is unknown rather than an overestimate.
    """
    if predicted.shape != verified_safe.shape or predicted.shape != right_censored.shape:
        raise ValueError('prediction and label shapes must match')
    unknown = right_censored.bool() & (predicted > verified_safe)
    over = ~right_censored.bool() & (predicted > verified_safe)
    return int(over.sum()), int((~unknown).sum()), int(unknown.sum())


class TerrainHoldSNN(nn.Module):
    """One internally multi-step SNN with two discrete deadline heads.

    The gait encoder is part of the terrain channel. Its weight is frozen, but
    it is evaluated only when this model is called by the fixed Phase 1 clock.
    SNNActor resets membrane within each call, so there is no cross-episode SNN
    state. Only policy reservoir state needs per-environment maintenance.
    """

    def __init__(self, gait_encoder, bins, observation_dim, hidden_dims=(64,), snn_steps=8):
        super().__init__()
        if not hasattr(gait_encoder, 'scan_dim') or not hasattr(gait_encoder, 'output_dim'):
            raise ValueError('expected the gait TerrainScanEncoder')
        self.bins = bins
        self.observation_dim = observation_dim
        self.terrain_encoder = gait_encoder
        self.terrain_encoder.requires_grad_(False)
        self.terrain_encoder.eval()
        input_dim = observation_dim - gait_encoder.scan_dim + gait_encoder.output_dim
        if input_dim <= gait_encoder.output_dim:
            raise ValueError('observation does not contain proprioception')
        self.snn = SNNActor(input_dim, hidden_dims,
                            len(bins.gait) + len(bins.modulation),
                            num_snn_steps=snn_steps)

    def train(self, mode=True):
        super().train(mode)
        self.terrain_encoder.eval()
        return self

    def forward(self, observation):
        if observation.shape[-1] != self.observation_dim:
            raise ValueError('observation dimension does not match checkpoint')
        scan = observation[..., -self.terrain_encoder.scan_dim:].reshape(
            *observation.shape[:-1], *self.terrain_encoder.scan_shape)
        with torch.no_grad():
            terrain = self.terrain_encoder(scan)
        logits = self.snn(torch.cat((observation[..., :-self.terrain_encoder.scan_dim], terrain), -1))
        return logits.split((len(self.bins.gait), len(self.bins.modulation)), dim=-1)

    @torch.no_grad()
    def decode(self, observation):
        gait, modulation = self(observation)
        return torch.stack((
            torch.as_tensor(self.bins.gait, device=gait.device)[gait.argmax(-1)],
            torch.as_tensor(self.bins.modulation, device=modulation.device)[modulation.argmax(-1)],
        ), -1)
