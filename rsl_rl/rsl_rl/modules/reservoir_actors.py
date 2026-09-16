from collections.abc import Sequence

import torch
import torch.nn as nn

from .actors import MLPActor
from .reservoirs import AnalogReservoir, LIFReservoir
from .snn import SNNActor, SNNActor_TWIN_output_head


class _ReservoirReadoutActor(nn.Module):
    """Compose fixed reservoir dynamics with a trainable actor readout."""

    is_reservoir = True

    def __init__(
        self,
        reservoir: nn.Module,
        readout: nn.Module,
        output_dim: int,
        include_input_in_readout: bool = False,
        is_discrete: bool = False,
    ):
        super().__init__()

        readout_input_dim = reservoir.feature_dim
        if include_input_in_readout:
            readout_input_dim += reservoir.input_dim

        self.reservoir = reservoir
        self.readout = readout
        self.input_dim = reservoir.input_dim
        self.output_dim = output_dim
        self.reservoir_dim = reservoir.reservoir_dim
        self.reservoir_state_dim = reservoir.state_dim
        self.include_input_in_readout = include_input_in_readout
        self.is_discrete = is_discrete

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected observations[..., {self.input_dim}], "
                f"but received {tuple(observations.shape)}."
            )

    def readout_process(
        self,
        observations: torch.Tensor,
        reservoir_states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_observations(observations)
        if (
            reservoir_states.ndim == 0
            or reservoir_states.shape[-1] != self.reservoir_state_dim
        ):
            raise ValueError(
                f"Expected reservoir_states[..., "
                f"{self.reservoir_state_dim}], but received "
                f"{tuple(reservoir_states.shape)}."
            )
        if observations.shape[:-1] != reservoir_states.shape[:-1]:
            raise ValueError(
                "observations and reservoir_states must have matching "
                "leading dimensions."
            )

        leading_shape = observations.shape[:-1]
        flat_observations = observations.reshape(-1, self.input_dim)
        flat_states = reservoir_states.reshape(
            -1,
            self.reservoir_state_dim,
        )
        features = self.reservoir.features_from_state(flat_states)
        expected_feature_shape = (
            flat_states.shape[0],
            self.reservoir.feature_dim,
        )
        if features.shape != expected_feature_shape:
            raise ValueError(
                f"Expected reservoir features with shape "
                f"{expected_feature_shape}, but received "
                f"{tuple(features.shape)}."
            )

        if self.include_input_in_readout:
            features = torch.cat((flat_observations, features), dim=-1)

        if self.is_discrete:
            logits_1, logits_2 = self.readout(features)
            return logits_1, logits_2

        else:
            actions = self.readout(features)
            expected_action_shape = (flat_states.shape[0], self.output_dim)
            if actions.shape != expected_action_shape:
                raise ValueError(
                    f"Expected readout actions with shape "
                    f"{expected_action_shape}, but received "
                    f"{tuple(actions.shape)}."
                )
            return actions.reshape(*leading_shape, self.output_dim)

    def forward(
        self,
        observations: torch.Tensor,
        reservoir_states: torch.Tensor = None,
    ):
        self._validate_observations(observations)
        leading_shape = observations.shape[:-1]
        expected_state_shape = (*leading_shape, self.reservoir_state_dim)
        if reservoir_states is None:
            reservoir_states = observations.new_zeros(expected_state_shape)
        # elif reservoir_states.shape != expected_state_shape:
        #     raise ValueError(
        #         f"Expected reservoir_states with shape "
        #         f"{expected_state_shape}, but received "
        #         f"{tuple(reservoir_states.shape)}."
        #     )

        # flat_observations = observations.reshape(-1, self.input_dim)
        flat_states = reservoir_states.reshape(
            -1,
            self.reservoir_state_dim,
        )
        updated_states = self.reservoir.update_state(
            observations,
            flat_states,
        )
        if self.is_discrete:
            logits_1, logits_2 = self.readout_process(
                observations[:,-1],
                updated_states,
            )
            return (
                logits_1,
                logits_2,
                updated_states
            )

        else:
            actions = self.readout_process(
                flat_observations,
                updated_states,
            )
            return (
                actions.reshape(*leading_shape, self.output_dim),
                updated_states.reshape(
                    *leading_shape,
                    self.reservoir_state_dim,
                ),
            )


def _initialize_reservoir_readout_actor(
    actor: _ReservoirReadoutActor,
    reservoir_type,
    readout_type,
    *,
    input_dim: int,
    output_dim: int,
    output_dim_1: int,
    output_dim_2: int,
    reservoir_dim: int,
    reservoir_connectivity: float,
    spectral_radius: float,
    reservoir_input_scale: float,
    reservoir_bias_scale: float,
    num_reservoir_steps: int,
    leak_rate: float,
    reservoir_activation: str,
    reservoir_lif_beta: float,
    reservoir_lif_threshold: float,
    reservoir_surrogate_alpha: float,
    reservoir_reset_mode: str,
    readout_hidden_dims: Sequence,
    readout_activation: str,
    include_input_in_readout: bool,
    num_snn_steps: int,
    snn_lif_beta: float,
    snn_lif_threshold: float,
    snn_surrogate_alpha: float,
    snn_reset_mode: str,
    snn_input_scale: float,
    train_reservoir: bool,
    is_discrete: bool,
) -> None:
    reservoir_kwargs = {
        "input_dim": input_dim,
        "reservoir_dim": reservoir_dim,
        "connectivity": reservoir_connectivity,
        "spectral_radius": spectral_radius,
        "input_scale": reservoir_input_scale,
        "bias_scale": reservoir_bias_scale,
        "num_reservoir_steps": num_reservoir_steps,
        "train_reservoir": train_reservoir,
    }
    if reservoir_type is AnalogReservoir:
        reservoir_kwargs.update(
            leak_rate=leak_rate,
            activation=reservoir_activation,
        )
    else:
        reservoir_kwargs.update(
            lif_beta=reservoir_lif_beta,
            lif_threshold=reservoir_lif_threshold,
            surrogate_alpha=reservoir_surrogate_alpha,
            reset_mode=reservoir_reset_mode,
        )
    reservoir = reservoir_type(**reservoir_kwargs)

    readout_input_dim = reservoir.feature_dim
    if include_input_in_readout:
        readout_input_dim += input_dim
    readout_kwargs = {
        "input_dim": readout_input_dim,
        "hidden_dims": readout_hidden_dims,
        "output_dim": output_dim,
    }
    if readout_type is MLPActor:
        readout_kwargs["activation"] = readout_activation
    elif readout_type is SNNActor:
        readout_kwargs.update(
            num_snn_steps=num_snn_steps,
            lif_beta=snn_lif_beta,
            lif_threshold=snn_lif_threshold,
            surrogate_alpha=snn_surrogate_alpha,
            reset_mode=snn_reset_mode,
            input_scale=snn_input_scale,
        )
    else:
        readout_kwargs.update(
            num_snn_steps=num_snn_steps,
            lif_beta=snn_lif_beta,
            lif_threshold=snn_lif_threshold,
            surrogate_alpha=snn_surrogate_alpha,
            reset_mode=snn_reset_mode,
            input_scale=snn_input_scale,
            output_dim_1 = output_dim_1,
            output_dim_2 = output_dim_2
        )
        readout_kwargs.pop("output_dim")
    readout = readout_type(**readout_kwargs)

    _ReservoirReadoutActor.__init__(
        actor,
        reservoir=reservoir,
        readout=readout,
        output_dim=output_dim,
        include_input_in_readout=include_input_in_readout,
        is_discrete=is_discrete
    )


class AnalogReservoirMLPReadoutActor(_ReservoirReadoutActor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int = 16,
        reservoir_connectivity: float = 0.5,
        spectral_radius: float = 0.9,
        reservoir_input_scale: float = 0.5,
        reservoir_bias_scale: float = 0.1,
        num_reservoir_steps: int = 3,
        leak_rate: float = 0.5,
        reservoir_activation: str = "tanh",
        reservoir_lif_beta: float = 0.9,
        reservoir_lif_threshold: float = 1.0,
        reservoir_surrogate_alpha: float = 5.0,
        reservoir_reset_mode: str = "subtract",
        readout_hidden_dims: Sequence = (32,),
        readout_activation: str = "elu",
        include_input_in_readout: bool = False,
        num_snn_steps: int = 4,
        snn_lif_beta: float = 0.9,
        snn_lif_threshold: float = 1.0,
        snn_surrogate_alpha: float = 5.0,
        snn_reset_mode: str = "subtract",
        snn_input_scale: float = 1.0,
        train_reservoir: bool = False,
    ):
        _initialize_reservoir_readout_actor(
            self,
            AnalogReservoir,
            MLPActor,
            **{
                name: value
                for name, value in locals().items()
                if name != "self"
            },
        )


class AnalogReservoirSNNReadoutActor(_ReservoirReadoutActor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int = 16,
        reservoir_connectivity: float = 0.5,
        spectral_radius: float = 0.9,
        reservoir_input_scale: float = 0.5,
        reservoir_bias_scale: float = 0.1,
        num_reservoir_steps: int = 3,
        leak_rate: float = 0.5,
        reservoir_activation: str = "tanh",
        reservoir_lif_beta: float = 0.9,
        reservoir_lif_threshold: float = 1.0,
        reservoir_surrogate_alpha: float = 5.0,
        reservoir_reset_mode: str = "subtract",
        readout_hidden_dims: Sequence = (32,),
        readout_activation: str = "elu",
        include_input_in_readout: bool = False,
        num_snn_steps: int = 4,
        snn_lif_beta: float = 0.9,
        snn_lif_threshold: float = 1.0,
        snn_surrogate_alpha: float = 5.0,
        snn_reset_mode: str = "subtract",
        snn_input_scale: float = 1.0,
        train_reservoir: bool = False,
    ):
        _initialize_reservoir_readout_actor(
            self,
            AnalogReservoir,
            SNNActor,
            **{
                name: value
                for name, value in locals().items()
                if name != "self"
            },
        )


class LIFReservoirMLPReadoutActor(_ReservoirReadoutActor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int = 16,
        reservoir_connectivity: float = 0.5,
        spectral_radius: float = 0.9,
        reservoir_input_scale: float = 0.5,
        reservoir_bias_scale: float = 0.1,
        num_reservoir_steps: int = 3,
        leak_rate: float = 0.5,
        reservoir_activation: str = "tanh",
        reservoir_lif_beta: float = 0.9,
        reservoir_lif_threshold: float = 1.0,
        reservoir_surrogate_alpha: float = 5.0,
        reservoir_reset_mode: str = "subtract",
        readout_hidden_dims: Sequence = (32,),
        readout_activation: str = "elu",
        include_input_in_readout: bool = False,
        num_snn_steps: int = 4,
        snn_lif_beta: float = 0.9,
        snn_lif_threshold: float = 1.0,
        snn_surrogate_alpha: float = 5.0,
        snn_reset_mode: str = "subtract",
        snn_input_scale: float = 1.0,
        train_reservoir: bool = False,
    ):
        _initialize_reservoir_readout_actor(
            self,
            LIFReservoir,
            MLPActor,
            **{
                name: value
                for name, value in locals().items()
                if name != "self"
            },
        )


class LIFReservoirSNNReadoutActor(_ReservoirReadoutActor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int = 16,
        reservoir_connectivity: float = 0.5,
        spectral_radius: float = 0.9,
        reservoir_input_scale: float = 0.5,
        reservoir_bias_scale: float = 0.1,
        num_reservoir_steps: int = 3,
        leak_rate: float = 0.5,
        reservoir_activation: str = "tanh",
        reservoir_lif_beta: float = 0.9,
        reservoir_lif_threshold: float = 1.0,
        reservoir_surrogate_alpha: float = 5.0,
        reservoir_reset_mode: str = "subtract",
        readout_hidden_dims: Sequence = (32,),
        readout_activation: str = "elu",
        include_input_in_readout: bool = False,
        num_snn_steps: int = 4,
        snn_lif_beta: float = 0.9,
        snn_lif_threshold: float = 1.0,
        snn_surrogate_alpha: float = 5.0,
        snn_reset_mode: str = "subtract",
        snn_input_scale: float = 1.0,
        train_reservoir: bool = False,
    ):
        _initialize_reservoir_readout_actor(
            self,
            LIFReservoir,
            SNNActor,
            **{
                name: value
                for name, value in locals().items()
                if name != "self"
            },
        )

class LIFReservoirSNNReadoutActor_TWIN_output_head(LIFReservoirSNNReadoutActor):
    def __init__(self, input_dim, output_dim_1, output_dim_2, reservoir_dim = 16, reservoir_connectivity = 0.5, spectral_radius = 0.9, reservoir_input_scale = 0.5, reservoir_bias_scale = 0.1, num_reservoir_steps = 3, leak_rate = 0.5, reservoir_activation = "tanh", reservoir_lif_beta = 0.9, reservoir_lif_threshold = 1, reservoir_surrogate_alpha = 5, reservoir_reset_mode = "subtract", readout_hidden_dims = (32, ), readout_activation = "elu", include_input_in_readout = False, num_snn_steps = 4, snn_lif_beta = 0.9, snn_lif_threshold = 1, snn_surrogate_alpha = 5, snn_reset_mode = "subtract", snn_input_scale = 1, train_reservoir = False):
        output_dim = None
        is_discrete = True
        _initialize_reservoir_readout_actor(
            self,
            LIFReservoir,
            SNNActor_TWIN_output_head,
            **{
                name: value
                for name, value in locals().items()
                if name != "self"
            },
        )