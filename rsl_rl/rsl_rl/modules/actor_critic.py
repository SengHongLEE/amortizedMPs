# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .actors import MLPActor, MLPActor_TWIN_output_head, get_activation
from .reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
    LIFReservoirSNNReadoutActor_TWIN_output_head,
    LIFReservoirSNNReadoutActor_SINGLE_output_head

)
from .snn import SNNActor, SNNActor_TWIN_output_head, SNNActor_SINGLE_output_head
from .terrain_scan_encoder import TerrainEncodedActor

ACTOR_REGISTRY = {
    "mlp": MLPActor,
    "snn": SNNActor,
    "snn_single_output_head": SNNActor_SINGLE_output_head,
    "analog_reservoir_mlp": AnalogReservoirMLPReadoutActor,
    "analog_reservoir_snn": AnalogReservoirSNNReadoutActor,
    "lif_reservoir_mlp": LIFReservoirMLPReadoutActor,
    "lif_reservoir_snn": LIFReservoirSNNReadoutActor,

    "mlp_twin_ouput": MLPActor_TWIN_output_head,
    "snn_twin_output": SNNActor_TWIN_output_head,
    "lif_reservoir_snn_twin_output": LIFReservoirSNNReadoutActor_TWIN_output_head,
    "lif_reservoir_snn_single_output": LIFReservoirSNNReadoutActor_SINGLE_output_head,
}


class ActorCritic(nn.Module):
    is_recurrent = False
    is_reservoir = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        actor_type="mlp",
        reservoir_dim=16,
        reservoir_connectivity=0.5,
        spectral_radius=0.9,
        reservoir_input_scale=0.5,
        reservoir_bias_scale=0.1,
        num_reservoir_steps=3,
        leak_rate=0.5,
        reservoir_activation="tanh",
        reservoir_lif_beta=0.9,
        reservoir_lif_threshold=1.0,
        reservoir_surrogate_alpha=5.0,
        reservoir_reset_mode="subtract",
        readout_hidden_dims=(32,),
        readout_activation="elu",
        include_input_in_readout=False,
        num_snn_steps=4,
        snn_lif_beta=0.9,
        snn_lif_threshold=1.0,
        snn_surrogate_alpha=5.0,
        snn_reset_mode="subtract",
        snn_input_scale=1.0,
        train_reservoir=False,
        squash_actions=False,
        min_action_std=0.05,
        max_action_std=0.6,
        terrain_scan_shape=None,
        terrain_encoder_dim=0,
        **kwargs,
    ):
        try:
            actor_class = ACTOR_REGISTRY[actor_type]
        except (KeyError, TypeError):
            supported = ", ".join(ACTOR_REGISTRY)
            raise ValueError(
                f"Unsupported actor_type {actor_type!r}. "
                f"Expected one of: {supported}."
            ) from None

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will "
                f"be ignored for compatibility: {list(kwargs)}"
            )
        super().__init__()

        self.mlp_input_dim_a = num_actor_obs
        self.mlp_input_dim_c = num_critic_obs
        if terrain_scan_shape is not None:
            if actor_class is not SNNActor or terrain_encoder_dim < 3:
                raise ValueError('dense terrain encoder requires the SNN modulation actor')
            scan_dim = math.prod(terrain_scan_shape)
            compact_actor_obs = num_actor_obs - scan_dim + terrain_encoder_dim
            if compact_actor_obs <= terrain_encoder_dim:
                raise ValueError('dense terrain scan exceeds actor observation size')
        else:
            compact_actor_obs = num_actor_obs

        if actor_class is MLPActor:
            actor = actor_class(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim=num_actions,
                activation=activation,
            )
        elif actor_class is SNNActor:
            actor = actor_class(
                input_dim=compact_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim=num_actions,
                num_snn_steps=num_snn_steps,
                lif_beta=snn_lif_beta,
                lif_threshold=snn_lif_threshold,
                surrogate_alpha=snn_surrogate_alpha,
                reset_mode=snn_reset_mode,
                input_scale=snn_input_scale,
            )
        else:
            actor = actor_class(
                input_dim=num_actor_obs,
                output_dim=num_actions,
                reservoir_dim=reservoir_dim,
                reservoir_connectivity=reservoir_connectivity,
                spectral_radius=spectral_radius,
                reservoir_input_scale=reservoir_input_scale,
                reservoir_bias_scale=reservoir_bias_scale,
                num_reservoir_steps=num_reservoir_steps,
                leak_rate=leak_rate,
                reservoir_activation=reservoir_activation,
                reservoir_lif_beta=reservoir_lif_beta,
                reservoir_lif_threshold=reservoir_lif_threshold,
                reservoir_surrogate_alpha=reservoir_surrogate_alpha,
                reservoir_reset_mode=reservoir_reset_mode,
                readout_hidden_dims=readout_hidden_dims,
                readout_activation=readout_activation,
                include_input_in_readout=include_input_in_readout,
                num_snn_steps=num_snn_steps,
                snn_lif_beta=snn_lif_beta,
                snn_lif_threshold=snn_lif_threshold,
                snn_surrogate_alpha=snn_surrogate_alpha,
                snn_reset_mode=snn_reset_mode,
                snn_input_scale=snn_input_scale,
                train_reservoir=train_reservoir,
            )
        if terrain_scan_shape is not None:
            actor = TerrainEncodedActor(actor, num_actor_obs,
                                        terrain_scan_shape, terrain_encoder_dim)
        self.actor_type = actor_type
        self.actor = actor
        self.is_reservoir = getattr(actor, "is_reservoir", False)

        critic_layers = []
        critic_dims = [num_critic_obs, *critic_hidden_dims, 1]
        for index, (source, target) in enumerate(
            zip(critic_dims[:-1], critic_dims[1:])
        ):
            critic_layers.append(nn.Linear(source, target))
            if index < len(critic_dims) - 2:
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.squash_actions = bool(squash_actions)
        self.min_action_std = float(min_action_std)
        self.max_action_std = float(max_action_std)
        if self.squash_actions and not (0 < self.min_action_std <= self.max_action_std):
            raise ValueError("squashed action std bounds must satisfy 0 < min <= max")
        self.distribution = None

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    @staticmethod
    def get_activation(act_name):
        return get_activation(act_name)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.squash_actions:
            # Entropy in the controller's action space, including the tanh
            # change of variables. rsample keeps gradients through the mean.
            latent = self.distribution.rsample()
            log_jacobian = 2.0 * (
                math.log(2.0) - latent - F.softplus(-2.0 * latent)
            )
            return (self.distribution.entropy() + log_jacobian).sum(dim=-1)
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, reservoir_states=None):
        if self.is_reservoir:
            if reservoir_states is None:
                raise ValueError("reservoir_states are required.")
            mean, reservoir_states = self.actor(
                observations,
                reservoir_states,
            )
        else:
            mean = self.actor(observations)
        if self.squash_actions:
            with torch.no_grad():
                self.std.clamp_(self.min_action_std, self.max_action_std)
        self.distribution = Normal(
            mean,
            mean * 0.0 + self.std,
            validate_args=False,
        )
        return reservoir_states

    def act(self, observations, reservoir_states=None, **kwargs):
        next_states = self.update_distribution(
            observations,
            reservoir_states,
        )
        actions = self.distribution.sample()
        if self.squash_actions:
            actions = torch.tanh(actions).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        if self.is_reservoir:
            return actions.detach(), next_states
        return actions

    def get_actions_log_prob(self, actions):
        if self.squash_actions:
            bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            latent = torch.atanh(bounded)
            return (
                self.distribution.log_prob(latent)
                - torch.log1p(-bounded.square())
            ).sum(dim=-1)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(
        self,
        observations,
        reservoir_states=None,
        **kwargs,
    ):
        if self.is_reservoir:
            if reservoir_states is None:
                raise ValueError("reservoir_states are required.")
            mean, next_states = self.actor(observations, reservoir_states)
            return (torch.tanh(mean) if self.squash_actions else mean), next_states
        mean = self.actor(observations)
        return torch.tanh(mean) if self.squash_actions else mean

    def act_for_ppo_update(self, observations, reservoir_states):
        if not self.is_reservoir:
            raise RuntimeError(
                "act_for_ppo_update requires a reservoir actor."
            )
        mean = self.actor.readout_process(
            observations,
            reservoir_states,
        )
        if self.squash_actions:
            with torch.no_grad():
                self.std.clamp_(self.min_action_std, self.max_action_std)
        self.distribution = Normal(
            mean,
            mean * 0.0 + self.std,
            validate_args=False,
        )
        actions = self.distribution.sample()
        if self.squash_actions:
            actions = torch.tanh(actions).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return actions

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

class ActorCriticDiscrete(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions_1, num_actions_2, actor_hidden_dims=(256, 256, 256), critic_hidden_dims=(256, 256, 256), activation="elu", init_noise_std=1, high_actor_type="mlp_twin_ouput", primitive_uniform_mix=0., primitive_temperature=1., command_obs_start=None, reservoir_dim=16, reservoir_connectivity=0.5, spectral_radius=0.9, reservoir_input_scale=0.5, reservoir_bias_scale=0.1, num_reservoir_steps=3, leak_rate=0.5, reservoir_activation="tanh", reservoir_lif_beta=0.9, reservoir_lif_threshold=1, reservoir_surrogate_alpha=5, reservoir_reset_mode="subtract", readout_hidden_dims=(32, ), readout_activation="elu", include_input_in_readout=False, num_snn_steps=4, snn_lif_beta=0.9, snn_lif_threshold=1, snn_surrogate_alpha=5, snn_reset_mode="subtract", snn_input_scale=1, train_reservoir=False, height_edge_head=False, terrain_encoder_dim=0, terrain_scan_shape=None, **kwargs):
        try:
            actor_class = ACTOR_REGISTRY[high_actor_type]
        except (KeyError, TypeError):
            supported = ", ".join(ACTOR_REGISTRY)
            raise ValueError(
                f"Unsupported actor_type {high_actor_type!r}. "
                f"Expected one of: {supported}."
            ) from None
        if ((height_edge_head or terrain_encoder_dim or terrain_scan_shape is not None
             or command_obs_start is not None)
                and actor_class is not LIFReservoirSNNReadoutActor_SINGLE_output_head):
            raise ValueError('terrain or command readout options require the single-output LIF reservoir actor')

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will "
                f"be ignored for compatibility: {list(kwargs)}"
            )
        nn.Module.__init__(self)
        if not math.isfinite(primitive_uniform_mix) or not 0 <= primitive_uniform_mix < 1:
            raise ValueError('primitive_uniform_mix must be in [0, 1)')
        if not math.isfinite(primitive_temperature) or primitive_temperature <= 0:
            raise ValueError('primitive_temperature must be positive and finite')
        self.primitive_uniform_mix = float(primitive_uniform_mix)
        self.primitive_temperature = float(primitive_temperature)

        self.actor_class = actor_class
        self.mlp_input_dim_a = num_actor_obs
        self.mlp_input_dim_c = num_critic_obs

        if actor_class is MLPActor_TWIN_output_head:
            actor = actor_class(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim_1=num_actions_1,
                output_dim_2=num_actions_2,
                activation=activation,
            )
        elif actor_class is SNNActor_TWIN_output_head:
            actor = actor_class(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim_1=num_actions_1,
                output_dim_2=num_actions_2,
                num_snn_steps=num_snn_steps,
                lif_beta=snn_lif_beta,
                lif_threshold=snn_lif_threshold,
                surrogate_alpha=snn_surrogate_alpha,
                reset_mode=snn_reset_mode,
                input_scale=snn_input_scale,
            )
        elif actor_class is SNNActor_SINGLE_output_head:
            actor = actor_class(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim=num_actions_1,
                num_snn_steps=num_snn_steps,
                lif_beta=snn_lif_beta,
                lif_threshold=snn_lif_threshold,
                surrogate_alpha=snn_surrogate_alpha,
                reset_mode=snn_reset_mode,
                input_scale=snn_input_scale,
            )

        elif actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:
            actor = actor_class(
                input_dim=num_actor_obs,
                output_dim_1=num_actions_1,
                output_dim_2=num_actions_2,
                reservoir_dim=reservoir_dim,
                reservoir_connectivity=reservoir_connectivity,
                spectral_radius=spectral_radius,
                reservoir_input_scale=reservoir_input_scale,
                reservoir_bias_scale=reservoir_bias_scale,
                num_reservoir_steps=num_reservoir_steps,
                leak_rate=leak_rate,
                reservoir_activation=reservoir_activation,
                reservoir_lif_beta=reservoir_lif_beta,
                reservoir_lif_threshold=reservoir_lif_threshold,
                reservoir_surrogate_alpha=reservoir_surrogate_alpha,
                reservoir_reset_mode=reservoir_reset_mode,
                readout_hidden_dims=readout_hidden_dims,
                readout_activation=readout_activation,
                include_input_in_readout=include_input_in_readout,
                num_snn_steps=num_snn_steps,
                snn_lif_beta=snn_lif_beta,
                snn_lif_threshold=snn_lif_threshold,
                snn_surrogate_alpha=snn_surrogate_alpha,
                snn_reset_mode=snn_reset_mode,
                snn_input_scale=snn_input_scale,
                train_reservoir=train_reservoir,
            )

        elif actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head:
            actor = actor_class(
                input_dim=num_actor_obs,
                output_dim=num_actions_1,
                reservoir_dim=reservoir_dim,
                reservoir_connectivity=reservoir_connectivity,
                spectral_radius=spectral_radius,
                reservoir_input_scale=reservoir_input_scale,
                reservoir_bias_scale=reservoir_bias_scale,
                num_reservoir_steps=num_reservoir_steps,
                leak_rate=leak_rate,
                reservoir_activation=reservoir_activation,
                reservoir_lif_beta=reservoir_lif_beta,
                reservoir_lif_threshold=reservoir_lif_threshold,
                reservoir_surrogate_alpha=reservoir_surrogate_alpha,
                reservoir_reset_mode=reservoir_reset_mode,
                readout_hidden_dims=readout_hidden_dims,
                readout_activation=readout_activation,
                include_input_in_readout=include_input_in_readout,
                num_snn_steps=num_snn_steps,
                snn_lif_beta=snn_lif_beta,
                snn_lif_threshold=snn_lif_threshold,
                snn_surrogate_alpha=snn_surrogate_alpha,
                snn_reset_mode=snn_reset_mode,
                snn_input_scale=snn_input_scale,
                train_reservoir=train_reservoir,
                height_edge_head=height_edge_head,
                terrain_encoder_dim=terrain_encoder_dim,
                terrain_scan_shape=terrain_scan_shape,
                command_obs_start=command_obs_start,
            )

        self.actor_type = high_actor_type
        self.actor = actor
        self.is_reservoir = getattr(actor, "is_reservoir", False)
        self.num_actions_1 = num_actions_1
        self.num_actions_2 = num_actions_2

        critic_layers = []
        critic_dims = [num_critic_obs, *critic_hidden_dims, 1]
        for index, (source, target) in enumerate(
            zip(critic_dims[:-1], critic_dims[1:])
        ):
            critic_layers.append(nn.Linear(source, target))
            if index < len(critic_dims) - 2:
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions_1 + num_actions_2))
        self.distribution = None

    @property
    def entropy(self):
        if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head or self.actor_class is SNNActor_SINGLE_output_head:
            return self.distribution_1.entropy()
        
        elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:   
            entropy_1 = self.distribution_1.entropy()
            entropy_2 = self.distribution_2.entropy()
            # return entropy_2
            return entropy_1 + entropy_2

    def _gait_distribution(self, logits):
        logits = logits / self.primitive_temperature
        if self.primitive_uniform_mix == 0:
            return Categorical(logits=logits)
        probabilities = F.softmax(logits, dim=-1)
        probabilities = (1. - self.primitive_uniform_mix) * probabilities
        probabilities = probabilities + self.primitive_uniform_mix / logits.shape[-1]
        return Categorical(probs=probabilities)

    def update_distribution(self, observations, reservoir_states=None):
        if self.is_reservoir:
            if reservoir_states is None:
                raise ValueError("reservoir_states are required.")
            if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head:
                logits_1, reservoir_states = self.actor(
                    observations,
                    reservoir_states,
                )
                self.distribution_1 = self._gait_distribution(logits_1)
                return reservoir_states, logits_1
            elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:                
                logits_1, logits_2, reservoir_states = self.actor(
                    observations,
                    reservoir_states,
                )
                self.distribution_1 = self._gait_distribution(logits_1)
                self.distribution_2 = Categorical(logits=logits_2)
                return reservoir_states, logits_1, logits_2
        else:
            if self.actor_class is SNNActor_SINGLE_output_head:
                logits_1 = self.actor(observations)
                self.distribution_1 = self._gait_distribution(logits_1)
                return logits_1

    def act(self, observations, reservoir_states=None, **kwargs):
        if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head:
            next_states, logits_1 = self.update_distribution(
                observations,
                reservoir_states,
            )
            actions_1 = self.distribution_1.sample()
            return actions_1.unsqueeze(-1), logits_1, next_states
        elif self.actor_class is SNNActor_SINGLE_output_head:
            logits_1 = self.update_distribution(observations)
            actions_1 = self.distribution_1.sample()
            return actions_1.unsqueeze(-1), logits_1
        elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:   
            next_states, logits_1, logits_2 = self.update_distribution(
                observations,
                reservoir_states,
            )
            actions_1 = self.distribution_1.sample()
            actions_2 = self.distribution_2.sample()
            return torch.cat((actions_1.unsqueeze(-1), actions_2.unsqueeze(-1)), dim=-1), torch.cat((logits_1, logits_2), dim=-1), next_states

        return torch.cat((actions_1.unsqueeze(-1), actions_2.unsqueeze(-1)), dim=-1), torch.cat((logits_1, logits_2), dim=-1)

    def get_actions_log_prob(self, actions):
        if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head or self.actor_class is SNNActor_SINGLE_output_head:
            log_prob_1 = self.distribution_1.log_prob(actions[:,0])
            joint_log_prob = log_prob_1
            return joint_log_prob
        
        elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:   
            log_prob_1 = self.distribution_1.log_prob(actions[:,0])
            log_prob_2 = self.distribution_2.log_prob(actions[:,1])

            joint_log_prob = log_prob_1 + log_prob_2

            return joint_log_prob

    def act_inference(
        self,
        observations,
        reservoir_states=None,
        **kwargs,
    ):
        if self.is_reservoir:
            if reservoir_states is None:
                raise ValueError("reservoir_states are required.")
            if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head:
                logits_1, reservoir_states = self.actor(observations, reservoir_states)
                action_1 = torch.argmax(logits_1, dim=-1)
                return action_1, reservoir_states
            
            elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:   
                logits_1, logits_2, reservoir_states = self.actor(observations, reservoir_states)
                action_1 = torch.argmax(logits_1, dim=-1)
                action_2 = torch.argmax(logits_2, dim=-1)
                return action_1, action_2, reservoir_states
        else:
            if self.actor_class is SNNActor_SINGLE_output_head:
                logits_1 = self.actor(observations)
                action_1 = torch.argmax(logits_1, dim=-1)
                return action_1


    def act_for_ppo_update(self, observations, reservoir_states):
        if not self.is_reservoir:
            raise RuntimeError(
                "act_for_ppo_update requires a reservoir actor."
            )
        if self.actor_class is LIFReservoirSNNReadoutActor_SINGLE_output_head:
            logits_1 = self.actor.readout_process(
                observations,
                reservoir_states,
            )
            if hasattr(self.actor, 'terrain_readout_head'):
                logits_1 = logits_1 + self.actor.terrain_logits_from_observation(
                    observations)
            if hasattr(self.actor, 'command_readout_head'):
                logits_1 = logits_1 + self.actor.command_logits_from_observation(
                    observations)
            self.distribution_1 = self._gait_distribution(logits_1)
        elif self.actor_class is LIFReservoirSNNReadoutActor_TWIN_output_head:
            logits_1, logits_2 = self.actor.readout_process(
                observations,
                reservoir_states,
            )
            self.distribution_1 = self._gait_distribution(logits_1)
            self.distribution_2 = Categorical(logits=logits_2)
        return logits_1
