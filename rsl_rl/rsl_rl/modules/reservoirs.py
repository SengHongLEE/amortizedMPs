import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actors import get_activation
from .snn import LIFNeuron


_FLOAT32_MAX = torch.finfo(torch.float32).max
_FLOAT32_UNIFORM_LIMIT = _FLOAT32_MAX / 2.0


def _is_finite_float32(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        if not math.isfinite(value):
            return False
        converted = torch.tensor(value, dtype=torch.float32).item()
        return math.isfinite(converted)
    except (TypeError, ValueError, RuntimeError):
        return False


class _FixedReservoirBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        input_scale: float = 0.5,
        bias_scale: float = 0.1,
        num_reservoir_steps: int = 3,
        train_reservoir: bool = False,
    ):
        super().__init__()

        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim <= 0
        ):
            raise ValueError("input_dim must be a positive integer.")
        if (
            isinstance(reservoir_dim, bool)
            or not isinstance(reservoir_dim, int)
            or reservoir_dim < 2
        ):
            raise ValueError("reservoir_dim must be an integer of at least 2.")
        if (
            not _is_finite_float32(connectivity)
            or not 0.0 < connectivity <= 1.0
        ):
            raise ValueError("connectivity must be in (0, 1].")
        if (
            not _is_finite_float32(spectral_radius)
            or spectral_radius <= 0.0
        ):
            raise ValueError("spectral_radius must be positive.")
        if (
            not _is_finite_float32(input_scale)
            or input_scale < 0.0
            or input_scale > _FLOAT32_UNIFORM_LIMIT
        ):
            raise ValueError(
                "input_scale must be finite, nonnegative, and representable."
            )
        if (
            not _is_finite_float32(bias_scale)
            or bias_scale < 0.0
            or bias_scale > _FLOAT32_UNIFORM_LIMIT
        ):
            raise ValueError(
                "bias_scale must be finite, nonnegative, and representable."
            )
        if (
            isinstance(num_reservoir_steps, bool)
            or not isinstance(num_reservoir_steps, int)
            or num_reservoir_steps < 1
        ):
            raise ValueError(
                "num_reservoir_steps must be a positive integer."
            )
        if train_reservoir:
            raise ValueError(
                "train_reservoir=True is not supported because PPO "
                "updates use stored reservoir states."
            )

        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.feature_dim = reservoir_dim
        self.num_reservoir_steps = num_reservoir_steps

        w_in = torch.empty(reservoir_dim, input_dim)
        nn.init.uniform_(w_in, -input_scale, input_scale)

        reservoir_bias = torch.empty(reservoir_dim)
        nn.init.uniform_(reservoir_bias, -bias_scale, bias_scale)

        expected_fan_in = reservoir_dim * connectivity
        w_res = self._initialize_recurrent_matrix(
            reservoir_dim=reservoir_dim,
            connectivity=connectivity,
            expected_fan_in=expected_fan_in,
            spectral_radius=spectral_radius,
        )

        self.register_buffer("w_in", w_in)
        self.register_buffer("w_res", w_res)
        self.register_buffer("reservoir_bias", reservoir_bias)

    @classmethod
    @torch.no_grad()
    def _initialize_recurrent_matrix(
        cls,
        reservoir_dim: int,
        connectivity: float,
        expected_fan_in: float,
        spectral_radius: float,
    ) -> torch.Tensor:
        fan_in_scale = expected_fan_in ** 0.5
        for _ in range(8):
            matrix = torch.randn(reservoir_dim, reservoir_dim)
            sparse_mask = torch.rand_like(matrix) < connectivity
            matrix.mul_(sparse_mask)
            matrix.div_(fan_in_scale)
            matrix.fill_diagonal_(0.0)
            try:
                return cls._scale_spectral_radius(
                    matrix,
                    target_radius=spectral_radius,
                )
            except RuntimeError:
                continue

        matrix = torch.zeros(reservoir_dim, reservoir_dim)
        matrix[0, 1] = 1.0 / fan_in_scale
        matrix[1, 0] = 1.0 / fan_in_scale
        return cls._scale_spectral_radius(
            matrix,
            target_radius=spectral_radius,
        )

    @staticmethod
    @torch.no_grad()
    def _scale_spectral_radius(
        matrix: torch.Tensor,
        target_radius: float,
    ) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvals(matrix.to(torch.float64))
        current_radius = eigenvalues.abs().max().real
        if not torch.isfinite(current_radius) or current_radius <= 1e-12:
            raise RuntimeError(
                "Invalid reservoir spectral radius. "
                "Try increasing connectivity or changing the random seed."
            )
        scale = target_radius / current_radius.to(matrix.dtype)
        scaled_matrix = matrix * scale
        if not torch.isfinite(scaled_matrix).all():
            raise ValueError(
                "spectral_radius produces non-finite float32 weights."
            )
        return scaled_matrix

    def input_projection(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_observations(observations)
        return F.linear(
            observations,
            self.w_in,
            self.reservoir_bias,
        )

    def _validate_observations(
        self,
        observations: torch.Tensor,
    ) -> None:
        if (
            observations.ndim != 2
            or observations.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"Expected observations with shape "
                f"[batch, {self.input_dim}], but received "
                f"{tuple(observations.shape)}."
            )

    def _validate_states(
        self,
        states: torch.Tensor,
    ) -> None:
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected states with shape "
                f"[batch, {self.state_dim}], but received "
                f"{tuple(states.shape)}."
            )

    def _validate_inputs(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        self._validate_observations(observations)
        self._validate_states(states)
        if observations.shape[0] != states.shape[0]:
            raise ValueError(
                "observations and states must have the same batch size."
            )


class AnalogReservoir(_FixedReservoirBase):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        input_scale: float = 0.5,
        bias_scale: float = 0.1,
        leak_rate: float = 0.5,
        num_reservoir_steps: int = 3,
        activation: str = "tanh",
        train_reservoir: bool = False,
    ):
        if (
            not _is_finite_float32(leak_rate)
            or not 0.0 < leak_rate <= 1.0
        ):
            raise ValueError("leak_rate must be in (0, 1].")

        try:
            reservoir_activation = get_activation(activation)
        except ValueError:
            raise ValueError(
                f"Invalid reservoir activation: {activation}."
            ) from None

        super().__init__(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            spectral_radius=spectral_radius,
            connectivity=connectivity,
            input_scale=input_scale,
            bias_scale=bias_scale,
            num_reservoir_steps=num_reservoir_steps,
            train_reservoir=train_reservoir,
        )

        self.state_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.activation = reservoir_activation

    def update_state(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(observations, states)
        input_drive = self.input_projection(observations)

        for _ in range(self.num_reservoir_steps):
            candidate_state = self.activation(
                input_drive + F.linear(states, self.w_res)
            )
            states = (
                (1.0 - self.leak_rate) * states
                + self.leak_rate * candidate_state
            )

        return states.detach()

    def features_from_state(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_states(states)
        return states


class LIFReservoir(_FixedReservoirBase):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        input_scale: float = 0.5,
        bias_scale: float = 0.1,
        lif_beta: float = 0.9,
        lif_threshold: float = 1.0,
        num_reservoir_steps: int = 3,
        surrogate_alpha: float = 5.0,
        reset_mode: str = "subtract",
        train_reservoir: bool = False,
    ):
        if (
            not _is_finite_float32(lif_beta)
            or not 0.0 <= lif_beta < 1.0
        ):
            raise ValueError(f"lif_beta must be in [0, 1), got {lif_beta}.")
        if (
            not _is_finite_float32(lif_threshold)
            or lif_threshold <= 0.0
        ):
            raise ValueError(
                f"lif_threshold must be positive, got {lif_threshold}."
            )
        if (
            not _is_finite_float32(surrogate_alpha)
            or surrogate_alpha <= 0.0
        ):
            raise ValueError("surrogate_alpha must be positive.")
        if reset_mode not in ("subtract", "zero"):
            raise ValueError("reset_mode must be 'subtract' or 'zero'.")

        super().__init__(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            spectral_radius=spectral_radius,
            connectivity=connectivity,
            input_scale=input_scale,
            bias_scale=bias_scale,
            num_reservoir_steps=num_reservoir_steps,
            train_reservoir=train_reservoir,
        )

        self.state_dim = 2 * reservoir_dim
        self.lif_neuron = LIFNeuron(
            beta=lif_beta,
            threshold=lif_threshold,
            surrogate_alpha=surrogate_alpha,
            reset_mode=reset_mode,
        )

    def update_state(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(observations, states)
        input_drive = self.input_projection(observations)
        membrane, spikes = states.chunk(2, dim=-1)

        for _ in range(self.num_reservoir_steps):
            recurrent_drive = F.linear(spikes, self.w_res)
            spikes, membrane = self.lif_neuron(
                input_drive + recurrent_drive,
                membrane,
            )

        return torch.cat(
            (membrane.detach(), spikes.detach()),
            dim=-1,
        )

    def features_from_state(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_states(states)
        _, spikes = states.chunk(2, dim=-1)
        return spikes
