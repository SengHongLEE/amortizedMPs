import math
from collections.abc import Sequence
from typing import List

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Return the activation module identified by ``name``."""
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        activation = activations[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported activation: {name!r}.") from None
    return activation()


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


class MLPActor(nn.Module):
    """A reusable feed-forward actor with a small final-layer gain."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence,
        output_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        _validate_dimension("input_dim", input_dim)
        _validate_dimension("output_dim", output_dim)
        hidden_dims = _validate_hidden_dims(hidden_dims)

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        layer_dims = [input_dim, *hidden_dims, output_dim]
        for index, (in_features, out_features) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            linear = nn.Linear(in_features, out_features)
            gain = 0.01 if index == len(layer_dims) - 2 else math.sqrt(2.0)
            nn.init.orthogonal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(layer_dims) - 2:
                layers.append(get_activation(activation))

        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            actual_dim = None if observations.ndim == 0 else observations.shape[-1]
            raise ValueError(
                f"Expected observation dimension {self.input_dim}, "
                f"got {actual_dim}."
            )
        return self.network(observations)
