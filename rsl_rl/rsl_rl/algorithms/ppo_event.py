from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage_event, RolloutStorage_event_discrete

import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

class PPOevent(PPO):
    def __init__(self, actor_critic, 
                 num_learning_epochs=1, 
                 num_mini_batches=1, 
                 clip_param=0.2, 
                 gamma=0.998, 
                 lam=0.95, 
                 value_loss_coef=1, 
                 entropy_coef=0, 
                 learning_rate=0.001, 
                 max_grad_norm=1,
                 use_clipped_value_loss=True, 
                 schedule="fixed", 
                 desired_kl=0.01, 
                 device='cpu'):

        super().__init__(actor_critic,
                         num_learning_epochs,
                         num_mini_batches,
                         clip_param,
                         gamma,
                         lam,
                         value_loss_coef,
                         entropy_coef,
                         learning_rate,
                         max_grad_norm,
                         use_clipped_value_loss,
                         schedule,
                         desired_kl,
                         device
                         )
        self.start_transition = RolloutStorage_event.StartTransition()
        self.end_transition = RolloutStorage_event.EndTransition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        self.storage = RolloutStorage_event(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, self.device)

    def clear_start_transition(self):
        self.start_transition.clear()

    def act(self, env_ids, step_ids, obs, critic_obs, reservoir_states=None):
        obs = obs[env_ids]
        critic_obs = critic_obs[env_ids]
        self.start_transition.env_ids = env_ids
        self.start_transition.step_ids = step_ids
        
        if getattr(self.actor_critic, "is_reservoir", False):
            reservoir_states = reservoir_states[env_ids]
            self.start_transition.actions, self.start_transition.reservoir_states = self.actor_critic.act(obs, reservoir_states)
        else:
            self.start_transition.actions = self.actor_critic.act(obs).detach()

        self.start_transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.start_transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.start_transition.actions).detach()
        self.start_transition.action_mean = self.actor_critic.action_mean.detach()
        self.start_transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.start_transition.observations = obs
        self.start_transition.critic_observations = critic_obs

        self.storage.add_start_transitions(self.start_transition)

        if getattr(self.actor_critic, "is_reservoir", False):
            return (
                self.start_transition.actions,
                self.start_transition.reservoir_states,
            )
        return self.start_transition.actions

    def process_env_step(self, current_step, env_ids, rewards, dones,
                         critic_obs, infos):
        pending_mask, tracked_env_ids, transition_ids = (
            self.storage.get_pending_transitions(env_ids)
        )
        if tracked_env_ids.numel() == 0:
            self.actor_critic.reset(dones[env_ids])
            return

        durations = (
            current_step - self.storage.step_ids[transition_ids]
        ).clamp_min(1)
        self.end_transition.rewards = (
            rewards[tracked_env_ids] / durations
        ).clone()
        self.end_transition.dones = dones[tracked_env_ids]
        self.end_transition.critic_observations = critic_obs[
            tracked_env_ids
        ]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.end_transition.rewards += self.gamma * torch.squeeze(
                self.storage.values[transition_ids]
                * infos['time_outs'][tracked_env_ids]
                .unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_end_transitions(
            tracked_env_ids,
            self.end_transition,
        )
        self.end_transition.clear()
        self.actor_critic.reset(dones[env_ids])
    
    def compute_returns(self, last_critic_obs):
        bootstrap_env_ids = torch.nonzero(
            self.storage.bootstrap_valid,
            as_tuple=False,
        ).squeeze(-1)
        last_values = torch.zeros(
            self.storage.num_envs,
            1,
            device=self.device,
        )
        last_values[bootstrap_env_ids] = self.actor_critic.evaluate(
            self.storage.bootstrap_observations[bootstrap_env_ids]
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_masked_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch, update_mask_batch):

        if not update_mask_batch.any():
            return actions_log_prob_batch.sum() * 0.0

        ratio = torch.exp(actions_log_prob_batch[update_mask_batch] - torch.squeeze(old_actions_log_prob_batch[update_mask_batch]))

        surrogate = -torch.squeeze(advantages_batch[update_mask_batch]) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch[update_mask_batch]) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        return surrogate_loss
    
    def masked_mean_or_zero(self, x, mask):
        if mask.any():
            return x[mask].mean()
        else:
            return x.sum() * 0.0

    def update(self):
        if getattr(self.actor_critic, "is_reservoir", False):
            mean_value_loss = 0
            mean_surrogate_loss = 0
            generator = self.storage.reservoir_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, reservoir_states_batch in generator:
                
                self.actor_critic.act_for_ppo_update(obs_batch, reservoir_states_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            self.storage.clear()

            return mean_value_loss, mean_surrogate_loss

        else:
            return super().update()

class PPOevent_discrete(PPOevent):
    def __init__(self, actor_critic, num_learning_epochs=1, num_mini_batches=1, clip_param=0.2, gamma=0.998, lam=0.95, value_loss_coef=1, entropy_coef=0, learning_rate=0.001, max_grad_norm=1, use_clipped_value_loss=True, schedule="fixed", desired_kl=0.01, device='cpu'):
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, logits_shape):
        self.storage = RolloutStorage_event_discrete(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, logits_shape, self.device)

    def act(self, env_ids, step_ids, obs, critic_obs, reservoir_states=None):
        obs = obs[env_ids]
        critic_obs = critic_obs[env_ids]
        self.start_transition.env_ids = env_ids
        self.start_transition.step_ids = step_ids
        
        if getattr(self.actor_critic, "is_reservoir", False):
            reservoir_states = reservoir_states[env_ids]
            self.start_transition.actions, self.start_transition.logits, self.start_transition.reservoir_states = self.actor_critic.act(obs, reservoir_states)
        else:
            self.start_transition.actions, self.start_transition.logits = self.actor_critic.act(obs)

        self.start_transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.start_transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.start_transition.actions).detach()
        self.start_transition.action_mean = self.start_transition.actions.clone()
        self.start_transition.action_sigma = self.start_transition.actions.clone()
        # need to record obs and critic_obs before env.step()
        self.start_transition.observations = obs[:,-1]
        self.start_transition.critic_observations = critic_obs

        self.storage.add_start_transitions(self.start_transition)

        if getattr(self.actor_critic, "is_reservoir", False):
            return (
                self.start_transition.actions.detach(),
                self.start_transition.reservoir_states,
            )
        return self.start_transition.actions.detach()
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if getattr(self.actor_critic, "is_reservoir", False):
            generator = self.storage.reservoir_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, logits_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
        old_mu_batch, old_sigma_batch, reservoir_states_batch, _ in generator:

            if getattr(self.actor_critic, "is_reservoir", False):
                self.actor_critic.act_for_ppo_update(obs_batch, reservoir_states_batch)
            else:
                self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            # mu_batch = self.actor_critic.action_mean
            # sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    old_dist = Categorical(logits=logits_batch[:, :self.actor_critic.num_actions_1])
                    new_dist = self.actor_critic.distribution_1
                    kl_mu = kl_divergence(old_dist, new_dist)

                    old_dist = Categorical(logits=logits_batch[:, self.actor_critic.num_actions_1: self.actor_critic.num_actions_1 + self.actor_critic.num_actions_2])
                    new_dist = self.actor_critic.distribution_2
                    kl_omega = kl_divergence(old_dist, new_dist)

                    kl_mean = (kl_mu + kl_omega).mean()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss