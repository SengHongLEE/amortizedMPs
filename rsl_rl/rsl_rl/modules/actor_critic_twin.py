import torch

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.actors import MLPActor


class ActorCriticTwin(ActorCritic):
    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions=4,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1,
                 **kwargs):
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            **kwargs,
        )
        if isinstance(self.actor, MLPActor):
            self.actor = self.actor.network
        self.is_reservoir = False

    @property
    def mu_actions_mean(self):
        return self.action_mean

    @property
    def mu_actions_std(self):
        return self.action_std

    @property
    def mu_entropy(self):
        return self.entropy

    @property
    def omega_actions_mean(self):
        return self.action_mean

    @property
    def omega_actions_std(self):
        return self.action_std

    @property
    def omega_entropy(self):
        return self.entropy

    def act_mu(self, observations, **kwargs):
        return self.act(observations, **kwargs)

    def get_mu_actions_log_prob(self, actions):
        return self.get_actions_log_prob(actions)

    def act_omega(self, observations, **kwargs):
        return self.act(observations, **kwargs)

    def get_omega_actions_log_prob(self, actions):
        return self.get_actions_log_prob(actions)

    @torch.no_grad()
    def act_mu_inference(self, observations):
        return self.act_inference(observations)

    @torch.no_grad()
    def act_omega_inference(self, observations):
        return self.act_inference(observations)
