import math
from collections.abc import Sequence
from numbers import Real
from typing import List

import torch
import torch.nn as nn


def _normalize_finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    try:
        normalized_value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite real number, got {value!r}."
        ) from error
    if not math.isfinite(normalized_value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    return normalized_value


def _validate_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_hidden_dims(hidden_dims: Sequence) -> List[int]:
    if (
        isinstance(hidden_dims, (str, bytes))
        or not isinstance(hidden_dims, Sequence)
        or len(hidden_dims) == 0
    ):
        raise ValueError("hidden_dims must contain at least one layer.")

    validated_dims = list(hidden_dims)
    for index, hidden_dim in enumerate(validated_dims):
        _validate_dimension(f"hidden_dims[{index}]", hidden_dim)
    return validated_dims


class SurrogateSpike(torch.autograd.Function):
    """Hard threshold with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(
        ctx,
        membrane_over_threshold: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(membrane_over_threshold)
        ctx.alpha = alpha
        return (membrane_over_threshold >= 0.0).to(
            membrane_over_threshold.dtype
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane_over_threshold,) = ctx.saved_tensors
        surrogate_gradient = 1.0 / (
            1.0 + ctx.alpha * membrane_over_threshold.abs()
        ).pow(2)
        return grad_output * surrogate_gradient, None


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with detached reset."""

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 5.0,
        reset_mode: str = "subtract",
    ):
        super().__init__()
        beta = _normalize_finite_real("beta", beta)
        threshold = _normalize_finite_real("threshold", threshold)
        surrogate_alpha = _normalize_finite_real(
            "surrogate_alpha",
            surrogate_alpha,
        )
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}.")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}.")
        if surrogate_alpha <= 0.0:
            raise ValueError(
                f"surrogate_alpha must be positive, got {surrogate_alpha}."
            )
        if reset_mode not in ("subtract", "zero"):
            raise ValueError("reset_mode must be 'subtract' or 'zero'.")

        self.beta = beta
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha
        self.reset_mode = reset_mode

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ):
        membrane = self.beta * membrane + current
        spike = SurrogateSpike.apply(
            membrane - self.threshold,
            self.surrogate_alpha,
        )
        detached_spike = spike.detach()
        if self.reset_mode == "subtract":
            membrane = membrane - detached_spike * self.threshold
        else:
            membrane = membrane * (1.0 - detached_spike)
        return spike, membrane


class SNNActor(nn.Module):
    """A stateless-across-calls, internally multi-step SNN actor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence,
        output_dim: int,
        num_snn_steps: int = 4,
        lif_beta: float = 0.9,
        lif_threshold: float = 1.0,
        surrogate_alpha: float = 5.0,
        reset_mode: str = "subtract",
        input_scale: float = 1.0,
    ):
        super().__init__()
        _validate_dimension("input_dim", input_dim)
        _validate_dimension("output_dim", output_dim)
        hidden_dims = _validate_hidden_dims(hidden_dims)
        _validate_dimension("num_snn_steps", num_snn_steps)
        input_scale = _normalize_finite_real("input_scale", input_scale)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_snn_steps = num_snn_steps
        self.input_scale = input_scale
        self.last_spike_rates = []

        layer_dims = [input_dim, *hidden_dims]
        self.linear_layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(
                layer_dims[:-1], layer_dims[1:]
            )
        )
        self.lif_layers = nn.ModuleList(
            LIFNeuron(
                beta=lif_beta,
                threshold=lif_threshold,
                surrogate_alpha=surrogate_alpha,
                reset_mode=reset_mode,
            )
            for _ in hidden_dims
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        for layer in self.linear_layers:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            actual_dim = None if observations.ndim == 0 else observations.shape[-1]
            raise ValueError(
                f"Expected observation dimension {self.input_dim}, "
                f"got {actual_dim}."
            )

        leading_shape = observations.shape[:-1]
        observations = observations.reshape(-1, self.input_dim)
        batch_size = observations.shape[0]
        membrane_states = [
            observations.new_zeros(batch_size, hidden_dim)
            for hidden_dim in self.hidden_dims
        ]
        action_sum = observations.new_zeros(batch_size, self.output_dim)
        spike_sums = [
            observations.new_zeros(()) for _ in self.hidden_dims
        ]
        scaled_observations = observations * self.input_scale

        for _ in range(self.num_snn_steps):
            x = scaled_observations
            for index, (linear_layer, lif_layer) in enumerate(
                zip(self.linear_layers, self.lif_layers)
            ):
                current = linear_layer(x)
                spike, membrane = lif_layer(
                    current,
                    membrane_states[index],
                )
                membrane_states[index] = membrane
                if spike.numel() == 0:
                    spike_rate = spike.detach().new_zeros(())
                else:
                    spike_rate = spike.detach().mean()
                spike_sums[index] = spike_sums[index] + spike_rate
                x = spike
            action_sum = action_sum + self.output_layer(x)

        action_mean = action_sum / float(self.num_snn_steps)
        self.last_spike_rates = [
            spike_sum / float(self.num_snn_steps)
            for spike_sum in spike_sums
        ]
        return action_mean.reshape(*leading_shape, self.output_dim)
