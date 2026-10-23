"""Strict, explicit warm-start mapping for the proprio v1 and v2 SNN policies."""
import math
import torch


def with_timing_prior(source, timing_cfg, nominal_clearance, prior, std=.2):
    """Explicitly reinitialize six timing/lift heads, preserving spatial actions.

    This changes the initial physical policy; it is not an equivalent migration.
    Prior order is frequency multiplier, duty offset, clearance in metres.
    """
    if len(prior) != 3 or not all(math.isfinite(x) for x in prior):
        raise ValueError('timing prior must contain three finite numbers')
    if not math.isfinite(std) or std <= 0:
        raise ValueError('timing exploration std must be finite and positive')
    normalized = []
    for value, bounds, neutral in zip(prior, (
            timing_cfg.frequency_scale_range, timing_cfg.duty_offset_range,
            timing_cfg.clearance_range), (1., 0., nominal_clearance)):
        low, high = bounds
        if not low <= value <= high:
            raise ValueError(f'timing prior {value} is outside [{low},{high}]')
        span = high - neutral if value >= neutral else neutral - low
        normalized.append((value - neutral) / span if span else 0.)
    if source['actor.output_layer.weight'].shape[0] != 22:
        raise ValueError('timing prior requires the 22-output SNN modulation actor')
    result = {key: value.clone() for key, value in source.items()}
    result['actor.output_layer.weight'][16:] = 0.
    result['actor.output_layer.bias'][16:] = result['std'].new_tensor(
        [normalized[2]] * 4 + normalized[:2])
    result['std'][16:] = std
    return result


def migrate_state_dict(source, target, high, new_std=.2):
    """Copy existing functions; ignore new inputs and initialize new means to zero.

    Supports the configured SNN low actor and LIF reservoir high actor only.
    Optimizer state is deliberately not migrated across changing dimensions.
    """
    if not math.isfinite(new_std) or new_std <= 0:
        raise ValueError('new_std must be positive and finite')
    if source.keys() != target.keys():
        raise ValueError('Source and destination network keys differ')
    input_key = 'actor.reservoir.w_in' if high else 'actor.linear_layers.0.weight'
    required = {input_key: (124, 132), 'critic.0.weight': (131, 139)}
    for key, (old, new) in required.items():
        if source[key].ndim != 2 or target[key].ndim != 2 or source[key].shape[1] != old or target[key].shape[1] != new:
            raise ValueError(f'Unexpected v1/v2 dimensions for {key}')
    result = {}
    for key, destination in target.items():
        value = source[key].to(destination)
        if key in required:
            before = 51 if key == input_key else 58
            if value.shape[0] != destination.shape[0]:
                raise ValueError(f'Hidden dimensions differ: {key}')
            mapped = torch.zeros_like(destination)
            mapped[:, :before] = value[:, :before]
            mapped[:, before + 6:-2] = value[:, before:]
        elif not high and key in ('std', 'actor.output_layer.weight', 'actor.output_layer.bias'):
            if value.shape[0] != 16 or destination.shape[0] != 22 or value.shape[1:] != destination.shape[1:]:
                raise ValueError(f'Unexpected action output dimensions: {key}')
            mapped = torch.full_like(destination, new_std) if key == 'std' else torch.zeros_like(destination)
            mapped[:16] = value
        elif value.shape == destination.shape:
            mapped = value.clone()
        else:
            raise ValueError(f'Unsupported network change: {key}: {value.shape} -> {destination.shape}')
        result[key] = mapped
    return result


def migrate_modal_state_dict(source, high):
    """v2 -> v3 basis change for history inputs, physical means and critic.

    Diagonal Gaussian marginal variances are transformed, but a diagonal
    distribution cannot preserve all old cross-mode covariances. Deterministic
    functions are preserved; stochastic exploration is intentionally different.
    """
    input_key = 'actor.reservoir.w_in' if high else 'actor.linear_layers.0.weight'
    if source[input_key].shape[1] != 132 or source['critic.0.weight'].shape[1] != 139:
        raise ValueError('Modal migration requires the v2 132/139 observation contract')
    result = {key: value.clone() for key, value in source.items()}
    h = source[input_key].new_tensor([[1., 1., 1., 1.], [1., 1., -1., -1.],
                                     [1., -1., 1., -1.], [1., -1., -1., 1.]])
    transform = torch.block_diag(h, h, h)
    for key, start in [(input_key, 39), ('critic.0.weight', 46)]:
        result[key][:, start:start + 12] = source[key][:, start:start + 12] @ transform
    if not high:
        if source['actor.output_layer.weight'].shape[0] != 22:
            raise ValueError('Modal migration requires 22 continuous outputs')
        inverse = transform.T / 4.
        result['actor.output_layer.weight'][4:16] = inverse @ source['actor.output_layer.weight'][4:16]
        result['actor.output_layer.bias'][4:16] = inverse @ source['actor.output_layer.bias'][4:16]
        result['std'][4:16] = (inverse.square() @ source['std'][4:16].square()).sqrt()
    return result
