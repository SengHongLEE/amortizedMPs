from collections.abc import Sequence

import torch
import torch.nn as nn

from .actors import MLPActor
from .reservoirs import AnalogReservoir, LIFReservoir
from .snn import SNNActor, SNNActor_TWIN_output_head, SNNActor_SINGLE_output_head
from .terrain_scan_encoder import TerrainScanEncoder


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
        height_edge_head: bool = False,
        external_input_dim: int = None,
        terrain_encoder_dim: int = 0,
        terrain_scan_shape=None,
    ):
        super().__init__()

        readout_input_dim = reservoir.feature_dim
        if include_input_in_readout:
            readout_input_dim += reservoir.input_dim

        self.reservoir = reservoir
        self.readout = readout
        self.input_dim = reservoir.input_dim if external_input_dim is None else external_input_dim
        self.output_dim = output_dim
        self.reservoir_dim = reservoir.reservoir_dim
        self.reservoir_state_dim = reservoir.state_dim
        self.include_input_in_readout = include_input_in_readout
        self.is_discrete = is_discrete
        if readout.input_dim != readout_input_dim:
            raise ValueError('readout input_dim does not match reservoir features')
        if output_dim is not None and getattr(readout, 'output_dim', output_dim) != output_dim:
            raise ValueError('readout output_dim does not match actor output')
        self.train_encoder_through_reservoir = False
        if terrain_scan_shape is not None:
            self.terrain_encoder = TerrainScanEncoder(terrain_scan_shape, terrain_encoder_dim)
            self.terrain_scan_shape = self.terrain_encoder.scan_shape
            self.terrain_scan_dim = self.terrain_encoder.scan_dim
            if (not isinstance(reservoir, LIFReservoir)
                    or self.input_dim <= self.terrain_scan_dim
                    or reservoir.input_dim != self.input_dim - self.terrain_scan_dim
                    + 2 * self.terrain_scan_shape[1]):
                raise ValueError('dense terrain encoder does not match reservoir input')
            if is_discrete:
                self.terrain_readout_head = nn.Sequential(
                    nn.Linear(terrain_encoder_dim, 16), nn.ReLU(),
                    nn.Linear(16, output_dim))
                nn.init.normal_(self.terrain_readout_head[-1].weight, std=0.01)
                nn.init.zeros_(self.terrain_readout_head[-1].bias)
        elif terrain_encoder_dim:
            if (not isinstance(reservoir, LIFReservoir)
                    or self.input_dim != 162
                    or reservoir.input_dim != self.input_dim + terrain_encoder_dim):
                raise ValueError('terrain encoder requires the 162-input LIF actor')
            self.terrain_encoder = nn.Sequential(
                nn.Conv1d(5, 8, kernel_size=3, padding=1),
                nn.ReLU(), nn.Flatten(), nn.Linear(8 * 12, terrain_encoder_dim),
                nn.Tanh())
            nn.init.zeros_(self.terrain_encoder[-2].weight)
            nn.init.zeros_(self.terrain_encoder[-2].bias)
        if height_edge_head:
            if not is_discrete or output_dim is None or reservoir.input_dim != 162:
                raise ValueError('height edge head requires a 162-input single discrete actor')
            self.height_edge_head = nn.Sequential(
                nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, output_dim))
            nn.init.zeros_(self.height_edge_head[-1].weight)
            nn.init.zeros_(self.height_edge_head[-1].bias)

    def height_edge_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Signed centerline height differences from the existing actor input."""
        if observations.ndim != 3 or observations.shape[-1] != 162:
            raise ValueError('expected actor observation history [N, T, 162]')
        center = observations[:, -1, -60:].reshape(-1, 12, 5)[:, :, 2]
        return center[:, 1:] - center[:, :-1]

    def terrain_logits_from_observation(self, observations: torch.Tensor) -> torch.Tensor:
        """Trainable terrain contribution, also used by saved-state PPO replay."""
        scan = observations[..., -self.terrain_scan_dim:].reshape(
            *observations.shape[:-1], *self.terrain_scan_shape)
        return self.terrain_readout_head(self.terrain_encoder(scan))

    def command_logits_from_observation(self, observations: torch.Tensor) -> torch.Tensor:
        start = self.command_obs_start
        return self.command_readout_head(observations[..., start:start + 3])

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
            return self.readout(features)

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
        if not self.is_discrete:
            leading_shape = observations.shape[:-1]
            expected_state_shape = (*leading_shape, self.reservoir_state_dim)
            if reservoir_states is None:
                reservoir_states = observations.new_zeros(expected_state_shape)
            elif reservoir_states.shape != expected_state_shape:
                raise ValueError('reservoir state shape must match observation leading dimensions')
            updated = self.reservoir.update_state(
                observations.reshape(-1, self.input_dim),
                reservoir_states.reshape(-1, self.reservoir_state_dim))
            updated = updated.reshape(*expected_state_shape)
            return self.readout_process(observations, updated), updated

        if observations.ndim != 3:
            raise ValueError('discrete reservoir actor expects history [N, T, input_dim]')
        leading_shape = observations.shape[:-1]
        expected_state_shape = (observations.shape[0], self.reservoir_state_dim)
        if reservoir_states is None:
            reservoir_states = observations.new_zeros(expected_state_shape)
        elif reservoir_states.shape != expected_state_shape:
            raise ValueError('discrete reservoir state must have shape [N, state_dim]')

        # flat_observations = observations.reshape(-1, self.input_dim)
        flat_states = reservoir_states.reshape(
            -1,
            self.reservoir_state_dim,
        )
        reservoir_observations = observations
        if hasattr(self, 'terrain_scan_shape'):
            scan = observations[..., -self.terrain_scan_dim:].reshape(
                *observations.shape[:2], *self.terrain_scan_shape)
            # Reservoir states saved in PPO are treated as fixed features. Keep
            # this temporal input deterministic; the learned spatial path is
            # replayed from the stored latest observation below.
            fixed_scan = torch.cat((scan[:, :, 0].amin(-1),
                                    scan[:, :, 1].amax(-1)), dim=-1)
            reservoir_observations = torch.cat((
                observations[..., :-self.terrain_scan_dim], fixed_scan), dim=-1)
            updated_states = self.reservoir.update_state(
                reservoir_observations, flat_states)
        elif hasattr(self, 'terrain_encoder'):
            if observations.ndim != 3:
                raise ValueError('terrain encoder requires observation history [N, T, 162]')
            height_grid = observations[..., -60:].reshape(
                *observations.shape[:2], 12, 5)
            encoded = self.terrain_encoder(
                height_grid.permute(0, 1, 3, 2).reshape(-1, 5, 12))
            encoded = encoded.reshape(*observations.shape[:2], -1)
            reservoir_observations = torch.cat((observations, encoded), dim=-1)
            updated_states = self.reservoir.update_state(
                reservoir_observations, flat_states,
                detach_state=not self.train_encoder_through_reservoir)
        else:
            updated_states = self.reservoir.update_state(
                reservoir_observations, flat_states)
        if self.is_discrete:
            logits = self.readout_process(
                observations[:, -1],
                updated_states,
            )

            if not isinstance(logits, tuple):
                logits = (logits,)
            if hasattr(self, 'height_edge_head'):
                edge_logits = self.height_edge_head(
                    self.height_edge_features(observations))
                logits = (logits[0] + edge_logits, *logits[1:])
            if hasattr(self, 'terrain_readout_head'):
                logits = (logits[0] + self.terrain_logits_from_observation(
                    observations[:, -1]),
                          *logits[1:])
            if hasattr(self, 'command_readout_head'):
                logits = (logits[0] + self.command_logits_from_observation(
                    observations[:, -1]), *logits[1:])

            return (*logits, updated_states)
        else:
            actions = self.readout_process(
                observations[:, -1],
                updated_states,
            )
            return (actions, updated_states)


def _initialize_reservoir_readout_actor(
    actor: _ReservoirReadoutActor,
    reservoir_type,
    readout_type,
    *,
    input_dim: int,
    output_dim: int,
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
    output_dim_1: int = None,
    output_dim_2: int = None,
    is_discrete: bool = False,
    height_edge_head: bool = False,
    terrain_encoder_dim: int = 0,
    terrain_scan_shape=None,
) -> None:
    if terrain_encoder_dim and reservoir_type is not LIFReservoir:
        raise ValueError('terrain encoder requires a LIF reservoir')
    reservoir_kwargs = {
        "input_dim": (input_dim -
                      (terrain_scan_shape[0] * terrain_scan_shape[1] * terrain_scan_shape[2])
                      + 2 * terrain_scan_shape[1] if terrain_scan_shape is not None
                      else input_dim + terrain_encoder_dim),
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
    elif readout_type is SNNActor_SINGLE_output_head:
        readout_kwargs.update(
            num_snn_steps=num_snn_steps,
            lif_beta=snn_lif_beta,
            lif_threshold=snn_lif_threshold,
            surrogate_alpha=snn_surrogate_alpha,
            reset_mode=snn_reset_mode,
            input_scale=snn_input_scale,
            output_dim = output_dim,
        )
    elif readout_type is SNNActor_TWIN_output_head:
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
        is_discrete=is_discrete,
        height_edge_head=height_edge_head,
        external_input_dim=input_dim,
        terrain_encoder_dim=terrain_encoder_dim,
        terrain_scan_shape=terrain_scan_shape,
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
        output_dim_1: int = output_dim
        output_dim_2: int = output_dim
        is_discrete: bool = False
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

class LIFReservoirSNNReadoutActor_SINGLE_output_head(LIFReservoirSNNReadoutActor):
    def __init__(self, input_dim, output_dim, reservoir_dim = 16, reservoir_connectivity = 0.5, spectral_radius = 0.9, reservoir_input_scale = 0.5, reservoir_bias_scale = 0.1, num_reservoir_steps = 3, leak_rate = 0.5, reservoir_activation = "tanh", reservoir_lif_beta = 0.9, reservoir_lif_threshold = 1, reservoir_surrogate_alpha = 5, reservoir_reset_mode = "subtract", readout_hidden_dims = (32, ), readout_activation = "elu", include_input_in_readout = False, num_snn_steps = 4, snn_lif_beta = 0.9, snn_lif_threshold = 1, snn_surrogate_alpha = 5, snn_reset_mode = "subtract", snn_input_scale = 1, train_reservoir = False, height_edge_head = False, terrain_encoder_dim = 0, terrain_scan_shape = None, command_obs_start = None):
        output_dim_1 = None
        output_dim_2 = None
        is_discrete = True
        _initialize_reservoir_readout_actor(
            self,
            LIFReservoir,
            SNNActor_SINGLE_output_head,
            **{
                name: value
                for name, value in locals().items()
                if name not in ("self", "command_obs_start")
            },
        )
        if command_obs_start is not None:
            if (not isinstance(command_obs_start, int) or command_obs_start < 0
                    or command_obs_start + 3 > input_dim):
                raise ValueError('command_obs_start must identify three observation columns')
            self.command_obs_start = command_obs_start
            self.command_readout_head = nn.Linear(3, output_dim)
            nn.init.zeros_(self.command_readout_head.weight)
            nn.init.zeros_(self.command_readout_head.bias)
