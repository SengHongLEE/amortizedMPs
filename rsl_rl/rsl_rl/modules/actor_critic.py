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

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from .actors import MLPActor, MLPActor_TWIN_output_head, get_activation
from .reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
    LIFReservoirSNNReadoutActor_TWIN_output_head,

)
from .snn import SNNActor, SNNActor_TWIN_output_head

ACTOR_REGISTRY = {
    "mlp": MLPActor,
    "snn": SNNActor,
    "analog_reservoir_mlp": AnalogReservoirMLPReadoutActor,
    "analog_reservoir_snn": AnalogReservoirSNNReadoutActor,
    "lif_reservoir_mlp": LIFReservoirMLPReadoutActor,
    "lif_reservoir_snn": LIFReservoirSNNReadoutActor,

    "mlp_twin_ouput": MLPActor_TWIN_output_head,
    "snn_twin_output": SNNActor_TWIN_output_head,
    "lif_reservoir_snn_twin_output": LIFReservoirSNNReadoutActor_TWIN_output_head,
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

        if actor_class is MLPActor:
            actor = actor_class(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                output_dim=num_actions,
                activation=activation,
            )
        elif actor_class is SNNActor:
            actor = actor_class(
                input_dim=num_actor_obs,
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
        if self.is_reservoir:
            return actions.detach(), next_states
        return actions

    def get_actions_log_prob(self, actions):
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
            return self.actor(observations, reservoir_states)
        return self.actor(observations)

    def act_for_ppo_update(self, observations, reservoir_states):
        if not self.is_reservoir:
            raise RuntimeError(
                "act_for_ppo_update requires a reservoir actor."
            )
        mean = self.actor.readout_process(
            observations,
            reservoir_states,
        )
        self.distribution = Normal(
            mean,
            mean * 0.0 + self.std,
            validate_args=False,
        )
        return self.distribution.sample()

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

class ActorCriticDiscrete(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions_1, num_actions_2, actor_hidden_dims=(256, 256, 256), critic_hidden_dims=(256, 256, 256), activation="elu", init_noise_std=1, high_actor_type="mlp_twin_ouput", reservoir_dim=16, reservoir_connectivity=0.5, spectral_radius=0.9, reservoir_input_scale=0.5, reservoir_bias_scale=0.1, num_reservoir_steps=3, leak_rate=0.5, reservoir_activation="tanh", reservoir_lif_beta=0.9, reservoir_lif_threshold=1, reservoir_surrogate_alpha=5, reservoir_reset_mode="subtract", readout_hidden_dims=(32, ), readout_activation="elu", include_input_in_readout=False, num_snn_steps=4, snn_lif_beta=0.9, snn_lif_threshold=1, snn_surrogate_alpha=5, snn_reset_mode="subtract", snn_input_scale=1, train_reservoir=False, **kwargs):
        try:
            actor_class = ACTOR_REGISTRY[high_actor_type]
        except (KeyError, TypeError):
            supported = ", ".join(ACTOR_REGISTRY)
            raise ValueError(
                f"Unsupported actor_type {high_actor_type!r}. "
                f"Expected one of: {supported}."
            ) from None

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will "
                f"be ignored for compatibility: {list(kwargs)}"
            )
        nn.Module.__init__(self)

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

        else:
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
        entropy_1 = self.distribution_1.entropy()
        entropy_2 = self.distribution_2.entropy()
        # return entropy_2
        return entropy_1 + entropy_2

    def update_distribution(self, observations, reservoir_states=None):
        if self.is_reservoir:
            if reservoir_states is None:
                raise ValueError("reservoir_states are required.")
            logits_1, logits_2, reservoir_states = self.actor(
                observations,
                reservoir_states,
            )
        else:
            logits_1, logits_2 = self.actor(observations)

        self.distribution_1 = Categorical(logits=logits_1)
        self.distribution_2 = Categorical(logits=logits_2)
        return reservoir_states, logits_1, logits_2

    def act(self, observations, reservoir_states=None, **kwargs):
        next_states, logits_1, logits_2 = self.update_distribution(
            observations,
            reservoir_states,
        )
        actions_1 = self.distribution_1.sample()
        actions_2 = self.distribution_2.sample()
        if self.is_reservoir:
            return torch.cat((actions_1.unsqueeze(-1), actions_2.unsqueeze(-1)), dim=-1), torch.cat((logits_1, logits_2), dim=-1), next_states
        return torch.cat((actions_1.unsqueeze(-1), actions_2.unsqueeze(-1)), dim=-1), torch.cat((logits_1, logits_2), dim=-1)

    def get_actions_log_prob(self, actions):
        log_prob_1 = self.distribution_1.log_prob(actions[:,0])
        log_prob_2 = self.distribution_2.log_prob(actions[:,1])

        joint_log_prob = log_prob_1 + log_prob_2
        # joint_log_prob = log_prob_2

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

            logits_1, logits_2, reservoir_states = self.actor(observations, reservoir_states)
            action_1 = torch.argmax(logits_1, dim=-1)
            action_2 = torch.argmax(logits_2, dim=-1)
            return action_1, action_2, reservoir_states
        
        logits_1, logits_2 = self.actor(observations)

        action_1 = torch.argmax(logits_1, dim=-1)
        action_2 = torch.argmax(logits_2, dim=-1)

        return action_1, action_2


    def act_for_ppo_update(self, observations, reservoir_states):
        if not self.is_reservoir:
            raise RuntimeError(
                "act_for_ppo_update requires a reservoir actor."
            )
        logits_1, logits_2 = self.actor.readout_process(
            observations,
            reservoir_states,
        )
        self.distribution_1 = Categorical(logits=logits_1)
        self.distribution_2 = Categorical(logits=logits_2)