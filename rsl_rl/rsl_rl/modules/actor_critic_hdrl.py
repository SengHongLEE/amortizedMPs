# MIT License

# Copyright (c) 2026 Seng-Hong Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class SoftAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 得到 (0,1) 权重
        )
        
    def forward(self, x):
        weights = self.attn(x)      # shape: (batch_size, input_dim)
        return x * weights          # element-wise multiply

class ActorNetwork(nn.Module):
    def __init__(self, activation,
                    mlp_input_dim_a, 
                    actor_hidden_dims,
                    gait_classes,
                    param_dim,
                    decimation,
                    learnable_decimation,
                    hidden_dim=128):
        super().__init__()
        self.activation = activation
        self.learnable_decimation = learnable_decimation
        self.soft_attn = SoftAttention(mlp_input_dim_a)
        
        hidden_layers = []
        hidden_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        hidden_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], gait_classes))
            else:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                hidden_layers.append(activation)
                # hidden_layers.append(nn.BatchNorm1d(actor_hidden_dims[l + 1]))
        self.G_hidden_layers = nn.Sequential(*hidden_layers)
        
        # self.gait_embedding = nn.Embedding(gait_classes, dim_embedding)
        
        hidden_layers = []
        hidden_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        hidden_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], param_dim))
                # hidden_layers.append(nn.Tanh())
            else:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                hidden_layers.append(activation)
                # hidden_layers.append(nn.BatchNorm1d(actor_hidden_dims[l + 1]))
        self.M_hidden_layers = nn.Sequential(*hidden_layers)

        hidden_layers = []
        hidden_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        hidden_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], decimation))
                # hidden_layers.append(nn.Tanh())
            else:
                hidden_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                hidden_layers.append(activation)
                # hidden_layers.append(nn.BatchNorm1d(actor_hidden_dims[l + 1]))
        self.T_hidden_layers = nn.Sequential(*hidden_layers)


            
    def forward(self, x):
        output_after_attn = self.soft_attn(x)      # shape: (batch_size, input_dim)
        # output_after_attn = x
        gait_logits = self.G_hidden_layers(output_after_attn)
        
        coupling_param = self.M_hidden_layers(output_after_attn)

        if self.learnable_decimation:
            decimation_logits = self.T_hidden_layers(output_after_attn)
            return torch.cat([decimation_logits, gait_logits, coupling_param], dim=-1)
        else:
            return torch.cat([gait_logits, coupling_param], dim=-1)



class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        gait_classes,
                        param_dim,
                        decimation,
                        learnable_decimation = False,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        self.actor = ActorNetwork(activation, mlp_input_dim_a, actor_hidden_dims, gait_classes, param_dim, decimation, learnable_decimation)
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                # critic_layers.append(nn.Sigmoid())
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions - 6))
        # self.std = init_noise_std * torch.ones(1,dtype=torch.float, device='cuda:0', requires_grad=False)
        self.distribution = None
        self.learnable_decimation = learnable_decimation
        self.gait_classes = gait_classes
        self.param_dim = param_dim
        self.decimation = decimation
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.param_dist.mean

    @property
    def action_std(self):
        return self.param_dist.stddev
    
    @property
    def entropy(self):
        gait_entropy = self.gait_dist.entropy().sum(dim=-1)
        param_entropy = self.param_dist.entropy().sum(dim=-1)
        if self.learnable_decimation:
            decimation_entropy = self.decimation_dist.entropy().sum(dim=-1)
            return param_entropy + 0.001 * (gait_entropy + decimation_entropy)
        return param_entropy + 0.001 * gait_entropy
    
    @property
    def logits(self):
        return self.gait_logits

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.learnable_decimation:
            decimation_dist = Categorical(logits=mean[:, :self.decimation])
            gait_dist = Categorical(logits=mean[:, self.decimation:self.decimation+self.gait_classes])
            param_action = mean[:, self.decimation+self.gait_classes:]
            param_dist = Normal(param_action, param_action*0. + self.std)
            self.decimation_dist = decimation_dist
            self.gait_logits = mean[:, :self.decimation+self.gait_classes]
        else:
            gait_dist = Categorical(logits=mean[:, :self.gait_classes])
            param_action = mean[:, self.gait_classes:]
            param_dist = Normal(param_action, param_action*0. + self.std)
            self.gait_logits = mean[:, :self.gait_classes]

        self.gait_dist = gait_dist
        self.param_dist = param_dist
        

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        gait_action = self.gait_dist.sample().unsqueeze(-1)
        param_action = self.param_dist.sample()
        if self.learnable_decimation:
            decimation_action = self.decimation_dist.sample().unsqueeze(-1)
            return torch.cat([decimation_action, gait_action, param_action], dim=-1)
        else:
            return torch.cat([gait_action, param_action], dim=-1)
    
    def get_actions_log_prob(self, actions):
        if self.learnable_decimation:
            gait_logprob = self.gait_dist.log_prob(actions[:, 1])
            param_logprob = self.param_dist.log_prob(actions[:, 2:]).sum(dim=-1)
            decimation_logprob = self.decimation_dist.log_prob(actions[:, 0])
            return param_logprob + gait_logprob + decimation_logprob
        else:
            gait_logprob = self.gait_dist.log_prob(actions[:, 0])
            param_logprob = self.param_dist.log_prob(actions[:, 1:]).sum(dim=-1)
            return param_logprob + gait_logprob

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
