from rsl_rl.runners import TwinPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticTwin, ActorCriticDiscrete
from rsl_rl.algorithms import PPOevent, PPOevent_discrete

import time
import os
from collections import deque
import statistics
import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch


_POLICY_CLASSES = {
    "ActorCritic": ActorCritic,
    "ActorCriticTwin": ActorCriticTwin,
    "ActorCriticDiscrete": ActorCriticDiscrete
}
_ALGORITHM_CLASSES = {
    "PPOevent": PPOevent,
    "PPOeventDiscrete": PPOevent_discrete,
}


def _resolve_class(name, classes, class_kind):
    try:
        return classes[name]
    except (KeyError, TypeError):
        supported = ", ".join(classes)
        raise ValueError(
            f"Unsupported {class_kind} {name!r}. "
            f"Supported values: {supported}."
        ) from None


class MPPolicyRunner(TwinPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        low_actor_critic_class = _resolve_class(
            self.cfg["low_level_policy_class_name"],
            _POLICY_CLASSES,
            "policy class",
        )
        low_alg_class = _resolve_class(
            self.cfg["low_algorithm_class_name"],
            _ALGORITHM_CLASSES,
            "algorithm class",
        )

        hip_actor_critic: ActorCritic = low_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.hip_shape,
                                            **self.policy_cfg).to(self.device)

        self.hip_alg: PPOevent = low_alg_class(hip_actor_critic, device=self.device, **self.alg_cfg)

        mu_actor_critic: ActorCritic = low_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.mu_shape,
                                            **self.policy_cfg).to(self.device)

        self.mu_alg: PPOevent = low_alg_class(mu_actor_critic, device=self.device, **self.alg_cfg)

        omega_actor_critic: ActorCritic = low_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.omega_shape,
                                            **self.policy_cfg).to(self.device)

        self.omega_alg: PPOevent = low_alg_class(omega_actor_critic, device=self.device, **self.alg_cfg)

        high_actor_critic_class = _resolve_class(
            self.cfg["high_level_policy_class_name"],
            _POLICY_CLASSES,
            "policy class",
        )
        high_alg_class = _resolve_class(
            self.cfg["high_algorithm_class_name"],
            _ALGORITHM_CLASSES,
            "algorithm class",
        )

        high_actor_critic: ActorCritic = high_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.mu_freq_idx_shape,
                                            self.env.omega_freq_idx_shape,
                                            **self.policy_cfg).to(self.device)

        self.high_alg: PPOevent_discrete = high_alg_class(high_actor_critic, device=self.device, **self.alg_cfg)


        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #Storage
        self.hip_alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.hip_shape])
        self.mu_alg.init_storage(self.env.num_envs,
                              int(self.num_steps_per_env),
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.mu_shape])
        self.omega_alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.omega_shape])

        self.high_alg.init_storage(self.env.num_envs,
                              int(self.num_steps_per_env),
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [2],
                              [self.env.mu_freq_idx_shape + self.env.omega_freq_idx_shape])

        #Twin Rate
        self.mu_cached = torch.zeros(self.env.num_envs, self.env.mu_shape, device=self.device)
        self.omega_cached = torch.zeros(self.env.num_envs, self.env.omega_shape, device=self.device)
        self.hip_cached = torch.zeros(self.env.num_envs, self.env.hip_shape, device=self.device)
        self.actions = torch.zeros(
            self.env.num_envs,
            self.env.mu_shape + self.env.omega_shape + self.env.hip_shape,
            device=self.device,
        )
        self.mu_update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * min(self.env.cfg.control.mu_freq_idx)
        self.omega_update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * min(self.env.cfg.control.omega_freq_idx)

        self.mu_reservoir_states = None
        self.omega_reservoir_states = None
        self.high_reservoir_states =None
        self.hip_reservoir_states =None

        if getattr(self.high_alg.actor_critic, "is_reservoir", False):
            self.high_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.high_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )
        if getattr(self.mu_alg.actor_critic, "is_reservoir", False):
            self.mu_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.mu_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )
        if getattr(self.omega_alg.actor_critic, "is_reservoir", False):
            self.omega_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.omega_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )
        if getattr(self.hip_alg.actor_critic, "is_reservoir", False):
            self.hip_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.hip_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )

        self.current_hip_step = 0
        self.current_high_step = 0
        self.current_mu_step = 0
        self.current_omega_step = 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.hip_tot_timesteps = 0
        self.hip_tot_time = 0
        self.high_tot_timesteps = 0
        self.high_tot_time = 0
        self.mu_tot_timesteps = 0
        self.mu_tot_time = 0
        self.omega_tot_timesteps = 0
        self.omega_tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    @staticmethod
    def get_update_env_ids(episode_length_buf, update_period):
        update_period = torch.as_tensor(
            update_period,
            dtype=episode_length_buf.dtype,
            device=episode_length_buf.device,
        )
        update_mask = torch.remainder(
            episode_length_buf,
            update_period,
        ) == 0
        return torch.nonzero(
            update_mask,
            as_tuple=False,
        ).squeeze(-1)

    @torch.no_grad
    def inference_rollout(self, obs, high_policy, mu_policy, omega_policy, hip_policy):
        high_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.high_cycle,
        )

        if getattr(self, "high_reservoir_states", None) is not None:
            mu_update_freq_idx, omega_update_freq_idx, reservoir_states = high_policy(obs[high_update_env_ids], self.high_reservoir_states[high_update_env_ids])
            self.high_reservoir_states[high_update_env_ids] = reservoir_states
        else:
            mu_update_freq_idx, omega_update_freq_idx = high_policy(obs[high_update_env_ids,-1])
        mu_map = torch.tensor(
            self.env.cfg.control.mu_freq_idx,
            device=mu_update_freq_idx.device,
            dtype=self.mu_update_cycle.dtype,
        )
        omega_map = torch.tensor(
            self.env.cfg.control.omega_freq_idx,
            device=omega_update_freq_idx.device,
            dtype=self.omega_update_cycle.dtype,
        )

        self.mu_update_cycle[high_update_env_ids] = mu_map[mu_update_freq_idx.long()]
        self.omega_update_cycle[high_update_env_ids] = omega_map[omega_update_freq_idx.long()]


        mu_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.mu_update_cycle,
        )
        omega_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.omega_update_cycle,
        )
        hip_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.hip_cycle,
        )

        if getattr(self, "mu_reservoir_states", None) is not None:
            mu_actions, reservoir_states = mu_policy(obs[mu_update_env_ids,-1], self.mu_reservoir_states[mu_update_env_ids])
            self.mu_reservoir_states[mu_update_env_ids] = reservoir_states
        else:
            mu_actions = mu_policy(obs[mu_update_env_ids,-1])

        if getattr(self, "omega_reservoir_states", None) is not None:
            omega_actions, reservoir_states = omega_policy(obs[omega_update_env_ids,-1], self.omega_reservoir_states[omega_update_env_ids])
            self.omega_reservoir_states[omega_update_env_ids] = reservoir_states
        else:
            omega_actions = omega_policy(obs[omega_update_env_ids,-1])

        if getattr(self, "hip_reservoir_states", None) is not None:
            hip_actions, reservoir_states = hip_policy(obs[hip_update_env_ids,-1], self.hip_reservoir_states[hip_update_env_ids])
            self.hip_reservoir_states[hip_update_env_ids] = reservoir_states
        else:
            hip_actions = hip_policy(obs[hip_update_env_ids,-1])


        self.mu_cached[mu_update_env_ids] = mu_actions
        self.omega_cached[omega_update_env_ids] = omega_actions
        self.hip_cached[hip_update_env_ids] = hip_actions
        self.actions[:, :self.env.mu_shape] = self.mu_cached
        self.actions[:, self.env.mu_shape:self.env.mu_shape + self.env.omega_shape] = self.omega_cached
        self.actions[:, self.env.mu_shape + self.env.omega_shape:] = self.hip_cached

        print(self.mu_update_cycle, self.omega_update_cycle)    
        self.env.update_idx_buffer.append([torch.concat((self.mu_update_cycle, self.omega_update_cycle), dim=-1).cpu().numpy()])

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)
        cycles = torch.stack((self.mu_update_cycle, self.omega_update_cycle),
            dim=-1) 
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )

        obs = torch.cat((obs, cycles), dim=-1)
        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)

        if getattr(self, "high_reservoir_states", None) is not None:
            self.high_reservoir_states[dones] = 0.0
        if getattr(self, "mu_reservoir_states", None) is not None:
            self.mu_reservoir_states[dones] = 0.0
        if getattr(self, "omega_reservoir_states", None) is not None:
            self.omega_reservoir_states[dones] = 0.0
        if getattr(self, "hip_reservoir_states", None) is not None:
            self.hip_reservoir_states[dones] = 0.0

        return obs, critic_obs, rewards, dones, infos

    @torch.no_grad
    def training_rollout(self, obs, critic_obs, current_high_step, current_mu_step, current_omega_step, current_hip_step):
        high_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.high_cycle,
        )
        high_step_ids = torch.full(
            (high_update_env_ids.shape[0],),
            current_high_step,
            dtype=torch.long,
            device=self.device,
        )

        if getattr(self.high_alg.actor_critic, "is_reservoir", False):
            if high_update_env_ids.numel() > 0:
                update_freq_idx, reservoir_states = self.high_alg.act(high_update_env_ids, high_step_ids, obs, critic_obs, self.high_reservoir_states)
                self.high_reservoir_states[high_update_env_ids] = reservoir_states
                mu_map = torch.tensor(
                self.env.cfg.control.mu_freq_idx,
                device=self.device,
                dtype=self.mu_update_cycle.dtype,
                )
                omega_map = torch.tensor(
                self.env.cfg.control.omega_freq_idx,
                device=self.device,
                dtype=self.omega_update_cycle.dtype,
                )

                self.mu_update_cycle[high_update_env_ids] = mu_map[update_freq_idx[:, 0].long()]
                self.omega_update_cycle[high_update_env_ids] = omega_map[update_freq_idx[:, 1].long()]
        else:
            update_freq_idx = self.high_alg.act(high_update_env_ids, high_step_ids, obs[:, -1], critic_obs)
        # mu_map = torch.tensor(
        #     [50, 100, 200, 400],
        #     device=self.device,
        #     dtype=self.mu_update_cycle.dtype,
        # )
        # omega_map = torch.tensor(
        #     [4, 10, 100, 200],
        #     device=self.device,
        #     dtype=self.omega_update_cycle.dtype,
        # )

        # self.mu_update_cycle[high_update_env_ids] = mu_map[update_freq_idx[:, 0].long()]
        # self.omega_update_cycle[high_update_env_ids] = omega_map[update_freq_idx[:, 1].long()]

        self.mu_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.mu_update_cycle,
        )
        self.omega_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.omega_update_cycle,
        )
        mu_step_ids = torch.full(
            (self.mu_update_env_ids.shape[0],),
            current_mu_step,
            dtype=torch.long,
            device=self.device,
        )
        omega_step_ids = torch.full(
            (self.omega_update_env_ids.shape[0],),
            current_omega_step,
            dtype=torch.long,
            device=self.device,
        )

        hip_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.hip_cycle,
        )
        hip_step_ids = torch.full(
            (hip_update_env_ids.shape[0],),
            current_hip_step,
            dtype=torch.long,
            device=self.device,
        )

        if getattr(self.mu_alg.actor_critic, "is_reservoir", False):
            mu_actions, reservoir_states = self.mu_alg.act(self.mu_update_env_ids, mu_step_ids, obs[:, -1], critic_obs, self.mu_reservoir_states)
            self.mu_reservoir_states[self.mu_update_env_ids] = reservoir_states
        else:
            mu_actions = self.mu_alg.act(self.mu_update_env_ids, mu_step_ids, obs[:, -1], critic_obs)

        if getattr(self.omega_alg.actor_critic, "is_reservoir", False):
            omega_actions, reservoir_states = self.omega_alg.act(self.omega_update_env_ids, omega_step_ids, obs[:, -1], critic_obs, self.omega_reservoir_states)
            self.omega_reservoir_states[self.omega_update_env_ids] = reservoir_states
        else:
            omega_actions = self.omega_alg.act(self.omega_update_env_ids, omega_step_ids, obs[:, -1], critic_obs)

        # if getattr(self.hip_alg.actor_critic, "is_reservoir", False):
        #     hip_actions, reservoir_states = self.hip_alg.act(hip_update_env_ids, hip_step_ids, obs[:, -1], critic_obs, self.hip_reservoir_states)
        #     self.hip_reservoir_states[hip_update_env_ids] = reservoir_states
        # else:
        #     hip_actions = self.hip_alg.act(hip_update_env_ids, hip_step_ids, obs[:, -1], critic_obs)

        self.mu_cached[self.mu_update_env_ids] = mu_actions
        self.omega_cached[self.omega_update_env_ids] = omega_actions
        # self.hip_cached[hip_update_env_ids] = hip_actions
        self.actions[:, :self.env.mu_shape] = self.mu_cached
        self.actions[:, self.env.mu_shape:self.env.mu_shape + self.env.omega_shape] = self.omega_cached
        self.actions[:, self.env.mu_shape + self.env.omega_shape:] = self.hip_cached

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)
        cycles = torch.stack((self.mu_update_cycle, self.omega_update_cycle),
            dim=-1) 
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )

        obs = torch.cat((obs, cycles), dim=-1)
        critic_obs = privileged_obs if privileged_obs is not None else obs[:, -1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)
        rewards = [reward.to(self.device) for reward in rewards]
        self.high_alg.clear_start_transition()
        self.mu_alg.clear_start_transition()
        self.omega_alg.clear_start_transition()
        self.hip_alg.clear_start_transition()

        return obs, critic_obs, rewards, dones, infos


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        cycles = torch.stack((self.mu_update_cycle, self.omega_update_cycle),
            dim=-1) 
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )

        obs = torch.cat((obs, cycles), dim=-1)
        privileged_obs = self.env.get_privileged_observations()
        if self.env.cfg.control.hierarchical:
            critic_obs = privileged_obs
        else:
            critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.high_alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.mu_alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.omega_alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.hip_alg.actor_critic.train() # switch to train mode (for dropout for example)
        high_ep_infos = []
        mu_ep_infos = []
        omega_ep_infos = []
        hip_ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        #Accumulate Rewards
        high_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)
        mu_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)
        omega_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)
        hip_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        high_it = self.current_learning_iteration
        mu_it = self.current_learning_iteration
        omega_it = self.current_learning_iteration
        hip_it = self.current_learning_iteration

        # Rollout
        high_collection_start = time.time()
        mu_collection_start = time.time()
        omega_collection_start = time.time()
        hip_collection_start = time.time()
        while True:

            obs, critic_obs, rewards, dones, infos = self.training_rollout(obs, critic_obs, self.current_high_step, self.current_mu_step, self.current_omega_step, self.current_hip_step)
            mu_acc_rewards += rewards[2]
            omega_acc_rewards += rewards[3]
            # hip_acc_rewards += rewards[4]
            high_acc_rewards += rewards[1]

            c_mu = torch.where(self.mu_update_cycle==0, 1., ((self.mu_update_cycle - min(self.env.cfg.control.mu_freq_idx)) / (max(self.env.cfg.control.mu_freq_idx) - min(self.env.cfg.control.mu_freq_idx))))
            c_omega = torch.where(self.omega_update_cycle==0, 1., ((self.omega_update_cycle - min(self.env.cfg.control.omega_freq_idx)) / (max(self.env.cfg.control.omega_freq_idx) - min(self.env.cfg.control.omega_freq_idx))))

            track_max = self.env.reward_scales["tracking_lin_vel"] + self.env.reward_scales["tracking_ang_vel"]
            track_quality = torch.clamp(
                rewards[1] / (track_max + 1e-6),
                min=0.0,
                max=1.0,
            )
            q_bad = self.env.cfg.rewards.q_bad
            q_good = self.env.cfg.rewards.q_good
            failure_cost = (
                torch.relu(q_bad - track_quality) / q_bad
            ).square()

            success_cost = (torch.relu(track_quality - q_good) / (1.0 - q_good)).square()

            lambda_mu_dist = 0.5
            lambda_omega_dist = 0.5

            high_acc_rewards += self.env.reward_scales['disturbance_penalty'] * failure_cost * (lambda_mu_dist * c_mu + lambda_omega_dist * c_omega)
            high_acc_rewards += self.env.reward_scales['mu_decision'] * success_cost * (1 - lambda_mu_dist) * (1 - c_mu)
            high_acc_rewards += self.env.reward_scales['omega_decision'] * success_cost * (1 - lambda_omega_dist) * (1 - c_omega)

            self.env.episode_sums['disturbance_penalty'] = self.env.reward_scales['disturbance_penalty'] * failure_cost * c_omega
            self.env.episode_sums['mu_decision'][self.mu_update_env_ids] = self.env.reward_scales['mu_decision'] * (torch.relu(track_quality - q_good) / (1.0 - q_good)).square()[self.mu_update_env_ids]
            self.env.episode_sums['omega_decision'][self.omega_update_env_ids] = self.env.reward_scales['omega_decision']* (torch.relu(track_quality - q_good) / (1.0 - q_good)).square()[self.omega_update_env_ids]
            
            env_ids = dones.nonzero(as_tuple=False).flatten()
            infos['episode']['rew_disturbance_penalty'] = torch.mean(self.env.episode_sums['disturbance_penalty'][env_ids]) / self.env.max_episode_length_s
            infos['episode']['rew_mu_decision'] = torch.mean(self.env.episode_sums['mu_decision'][env_ids]) / self.env.max_episode_length_s
            infos['episode']['rew_omega_decision'] = torch.mean(self.env.episode_sums['omega_decision'][env_ids]) / self.env.max_episode_length_s
            self.env.episode_sums['disturbance_penalty'][env_ids] = 0.
            self.env.episode_sums['mu_decision'][env_ids] = 0.
            self.env.episode_sums['omega_decision'][env_ids] = 0.


            self.current_high_step += 1
            self.current_mu_step += 1
            self.current_omega_step += 1
            self.current_hip_step += 1

            high_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.env.cfg.control.high_cycle,
            )
            mu_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.mu_update_cycle,
            )
            omega_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.omega_update_cycle,
            )
            hip_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.env.cfg.control.hip_cycle,
            )

            if self.hip_reservoir_states is not None:
                self.hip_reservoir_states[dones] = 0.
            if self.omega_reservoir_states is not None:
                self.omega_reservoir_states[dones] = 0.
            if self.mu_reservoir_states is not None:
                self.mu_reservoir_states[dones] = 0.
            if self.high_reservoir_states is not None:
                self.high_reservoir_states[dones] = 0.

            # self.hip_alg.process_env_step(
            #     self.current_hip_step,
            #     hip_update_env_ids,
            #     hip_acc_rewards,
            #     dones,
            #     critic_obs,
            #     infos,
            # )
            self.high_alg.process_env_step(
                self.current_high_step,
                high_update_env_ids,
                high_acc_rewards,
                dones,
                critic_obs,
                infos,
            )
            self.mu_alg.process_env_step(
                self.current_mu_step,
                mu_update_env_ids,
                mu_acc_rewards,
                dones,
                critic_obs,
                infos,
            )
            self.omega_alg.process_env_step(
                self.current_omega_step,
                omega_update_env_ids,
                omega_acc_rewards,
                dones,
                critic_obs,
                infos,
            )

            if self.log_dir is not None:
                # Book keeping
                if 'episode' in infos:
                    hip_ep_infos.append(infos['episode'])
                    high_ep_infos.append(infos['episode'])
                    mu_ep_infos.append(infos['episode'])
                    omega_ep_infos.append(infos['episode'])
                cur_reward_sum += rewards[0]
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            hip_acc_rewards[hip_update_env_ids] = 0.
            high_acc_rewards[high_update_env_ids] = 0.
            mu_acc_rewards[mu_update_env_ids] = 0.
            omega_acc_rewards[omega_update_env_ids] = 0.

            if self.high_alg.storage.ready:
                high_collection_time = time.time() - high_collection_start
                high_learn_start = time.time()
                self.high_alg.compute_returns(critic_obs)
                mean_high_value_loss, mean_high_surrogate_loss = self.high_alg.update()
                high_learn_stop = time.time()
                high_learn_time = high_learn_stop - high_learn_start
                if self.log_dir is not None:
                    self.high_log(locals())
                if high_it % self.save_interval == 0:
                    self.save_model(self.log_dir, high_it, hip_it, mu_it, omega_it)

                high_it += 1
                high_collection_start = time.time()
                high_ep_infos.clear()
            # if self.hip_alg.storage.ready:
            if self.current_hip_step % self.num_steps_per_env == 0:
                hip_collection_time = time.time() - hip_collection_start
                hip_learn_start = time.time()
                # self.hip_alg.compute_returns(critic_obs)
                # mean_hip_value_loss, mean_hip_surrogate_loss = self.hip_alg.update()
                mean_hip_value_loss, mean_hip_surrogate_loss = 0., 0.
                hip_learn_stop = time.time()
                hip_learn_time = hip_learn_stop - hip_learn_start
                if self.log_dir is not None:
                    self.hip_log(locals())
                if hip_it % self.save_interval == 0:
                    self.save_model(self.log_dir, high_it, hip_it, mu_it, omega_it)

                hip_it += 1
                hip_collection_start = time.time()
                hip_ep_infos.clear()
            if self.mu_alg.storage.ready:
                mu_collection_time = time.time() - mu_collection_start
                mu_learn_start = time.time()
                self.mu_alg.compute_returns(critic_obs)
                mean_mu_value_loss, mean_mu_surrogate_loss = self.mu_alg.update()
                mu_learn_stop = time.time()
                mu_learn_time = mu_learn_stop - mu_learn_start
                if self.log_dir is not None:
                    self.mu_log(locals())
                if mu_it % self.save_interval == 0:
                    self.save_model(self.log_dir, high_it, hip_it, mu_it, omega_it)

                mu_it += 1
                mu_collection_start = time.time()
                mu_ep_infos.clear()

            # torch.nonzero(self.omega_alg.storage.pending,as_tuple=False).squeeze(-1)
            if self.omega_alg.storage.ready:
                omega_collection_time = time.time() - omega_collection_start
                omega_learn_start = time.time()
                self.omega_alg.compute_returns(critic_obs)
                mean_omega_value_loss, mean_omega_surrogate_loss = self.omega_alg.update()
                omega_learn_stop = time.time()
                omega_learn_time = omega_learn_stop - omega_learn_start
                if self.log_dir is not None:
                    self.omega_log(locals())
                if omega_it % self.save_interval == 0:
                    self.save_model(self.log_dir,high_it, hip_it, mu_it, omega_it)

                omega_it += 1
                omega_collection_start = time.time()
                omega_ep_infos.clear()

            if high_it >= tot_iter:
                break

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir,
                               f"model_last_high-{high_it}-{mu_it}_omega-{omega_it}.pt"))
        self.final_log()

    def save_model(self, log_dir, high_it, hip_it, mu_it, omega_it):
        log_dir = Path(log_dir)

        pattern = re.compile(r"^model_(\d+)_high-\d+_hip-\d+_mu-\d+_omega-\d+\.pt$")
        existing_indices = []
        for file in log_dir.glob("*.pt"):
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        model_idx = max(existing_indices, default=-1) + 1

        save_path = log_dir / (
            f"model_{model_idx}_high-{high_it}_hip-{hip_it}_mu-{mu_it}_omega-{omega_it}.pt"
        )

        self.save(str(save_path))


    def final_log(self, width=40, pad=35):
        log_string = (f"""{'#' * width}\n"""
                        f"""Training done""")
        print(log_string)

    def high_log(self, locs, width=40, pad=35):
        self.high_tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.high_tot_time += locs['high_collection_time'] + locs['high_learn_time']
        iteration_time = locs['high_collection_time'] + locs['high_learn_time']

        ep_string = f''
        if locs['high_ep_infos']:
            for key in locs['high_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['high_ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/high/' + key, value, locs['high_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.high_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/high/value_function', locs['mean_high_value_loss'], locs['high_it'])
        self.writer.add_scalar('Loss/high/surrogate', locs['mean_high_surrogate_loss'], locs['high_it'])
        self.writer.add_scalar('Loss/high/learning_rate', self.high_alg.learning_rate, locs['high_it'])
        self.writer.add_scalar('Policy/high/mean_noise_std', mean_std.item(), locs['high_it'])
        self.writer.add_scalar('Perf/high/total_fps', fps, locs['high_it'])
        self.writer.add_scalar('Perf/high/collection time', locs['high_collection_time'], locs['high_it'])
        self.writer.add_scalar('Perf/high/learning_time', locs['high_learn_time'], locs['high_it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/high/mean_reward', statistics.mean(locs['rewbuffer']), locs['high_it'])
            self.writer.add_scalar('Train/high/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['high_it'])
            self.writer.add_scalar('Train/high/mean_reward/time', statistics.mean(locs['rewbuffer']), self.high_tot_time)
            self.writer.add_scalar('Train/high/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.high_tot_time)

        str = f" \033[1m Learning iteration {locs['high_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width} high {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'high_collection_time']:.3f}s, learning {locs['high_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_high_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_high_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'high_collection_time']:.3f}s, learning {locs['high_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_high_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_high_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.high_tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.high_tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.high_tot_time / (locs['high_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['high_it']):.1f}s\n""")
        print(log_string)

    def mu_log(self, locs, width=40, pad=35):
        self.mu_tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.mu_tot_time += locs['mu_collection_time'] + locs['mu_learn_time']
        iteration_time = locs['mu_collection_time'] + locs['mu_learn_time']

        ep_string = f''
        if locs['mu_ep_infos']:
            for key in locs['mu_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['mu_ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/mu/' + key, value, locs['mu_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.mu_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/mu/value_function', locs['mean_mu_value_loss'], locs['mu_it'])
        self.writer.add_scalar('Loss/mu/surrogate', locs['mean_mu_surrogate_loss'], locs['mu_it'])
        self.writer.add_scalar('Loss/mu/learning_rate', self.mu_alg.learning_rate, locs['mu_it'])
        self.writer.add_scalar('Policy/mu/mean_noise_std', mean_std.item(), locs['mu_it'])
        self.writer.add_scalar('Perf/mu/total_fps', fps, locs['mu_it'])
        self.writer.add_scalar('Perf/mu/collection time', locs['mu_collection_time'], locs['mu_it'])
        self.writer.add_scalar('Perf/mu/learning_time', locs['mu_learn_time'], locs['mu_it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mu/mean_reward', statistics.mean(locs['rewbuffer']), locs['mu_it'])
            self.writer.add_scalar('Train/mu/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['mu_it'])
            self.writer.add_scalar('Train/mu/mean_reward/time', statistics.mean(locs['rewbuffer']), self.mu_tot_time)
            self.writer.add_scalar('Train/mu/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.mu_tot_time)

        str = f" \033[1m Learning iteration {locs['mu_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width} mu {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'mu_collection_time']:.3f}s, learning {locs['mu_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_mu_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_mu_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'mu_collection_time']:.3f}s, learning {locs['mu_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_mu_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_mu_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.mu_tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.mu_tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.mu_tot_time / (locs['mu_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['mu_it']):.1f}s\n""")
        print(log_string)

    def omega_log(self, locs, width=40, pad=35):
        self.omega_tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.omega_tot_time += locs['omega_collection_time'] + locs['omega_learn_time']
        iteration_time = locs['omega_collection_time'] + locs['omega_learn_time']

        ep_string = f''
        if locs['omega_ep_infos']:
            for key in locs['omega_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['omega_ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/omega/' + key, value, locs['omega_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.omega_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/omega/value_function', locs['mean_omega_value_loss'], locs['omega_it'])
        self.writer.add_scalar('Loss/omega/surrogate', locs['mean_omega_surrogate_loss'], locs['omega_it'])
        self.writer.add_scalar('Loss/omega/learning_rate', self.omega_alg.learning_rate, locs['omega_it'])
        self.writer.add_scalar('Policy/omega/mean_noise_std', mean_std.item(), locs['omega_it'])
        self.writer.add_scalar('Perf/omega/total_fps', fps, locs['omega_it'])
        self.writer.add_scalar('Perf/omega/collection time', locs['omega_collection_time'], locs['omega_it'])
        self.writer.add_scalar('Perf/omega/learning_time', locs['omega_learn_time'], locs['omega_it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/omega/mean_reward', statistics.mean(locs['rewbuffer']), locs['omega_it'])
            self.writer.add_scalar('Train/omega/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['omega_it'])
            self.writer.add_scalar('Train/omega/mean_reward/time', statistics.mean(locs['rewbuffer']), self.omega_tot_time)
            self.writer.add_scalar('Train/omega/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.omega_tot_time)

        str = f" \033[1m Learning iteration {locs['omega_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width} omega {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'omega_collection_time']:.3f}s, learning {locs['omega_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_omega_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_omega_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'omega_collection_time']:.3f}s, learning {locs['omega_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_omega_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_omega_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.omega_tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.omega_tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.omega_tot_time / (locs['omega_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['omega_it']):.1f}s\n""")
        print(log_string)

    def hip_log(self, locs, width=40, pad=35):
        self.hip_tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.hip_tot_time += locs['hip_collection_time'] + locs['hip_learn_time']
        iteration_time = locs['hip_collection_time'] + locs['hip_learn_time']

        ep_string = f''
        if locs['hip_ep_infos']:
            for key in locs['hip_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['hip_ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/hip/' + key, value, locs['hip_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.hip_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/hip/value_function', locs['mean_hip_value_loss'], locs['hip_it'])
        self.writer.add_scalar('Loss/hip/surrogate', locs['mean_hip_surrogate_loss'], locs['hip_it'])
        self.writer.add_scalar('Loss/hip/learning_rate', self.hip_alg.learning_rate, locs['hip_it'])
        self.writer.add_scalar('Policy/hip/mean_noise_std', mean_std.item(), locs['hip_it'])
        self.writer.add_scalar('Perf/hip/total_fps', fps, locs['hip_it'])
        self.writer.add_scalar('Perf/hip/collection time', locs['hip_collection_time'], locs['hip_it'])
        self.writer.add_scalar('Perf/hip/learning_time', locs['hip_learn_time'], locs['hip_it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/hip/mean_reward', statistics.mean(locs['rewbuffer']), locs['hip_it'])
            self.writer.add_scalar('Train/hip/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['hip_it'])
            self.writer.add_scalar('Train/hip/mean_reward/time', statistics.mean(locs['rewbuffer']), self.hip_tot_time)
            self.writer.add_scalar('Train/hip/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.hip_tot_time)

        str = f" \033[1m Learning iteration {locs['hip_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width} hip {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'hip_collection_time']:.3f}s, learning {locs['hip_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_hip_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_hip_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'hip_collection_time']:.3f}s, learning {locs['hip_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_hip_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_hip_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.hip_tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.hip_tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.hip_tot_time / (locs['hip_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['hip_it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'high_model_state_dict': self.high_alg.actor_critic.state_dict(),
            'mu_model_state_dict': self.mu_alg.actor_critic.state_dict(),
            'omega_model_state_dict': self.omega_alg.actor_critic.state_dict(),
            'hip_model_state_dict': self.hip_alg.actor_critic.state_dict(),
            'high_optimizer_state_dict': self.high_alg.optimizer.state_dict(),
            'mu_optimizer_state_dict': self.mu_alg.optimizer.state_dict(),
            'omega_optimizer_state_dict': self.omega_alg.optimizer.state_dict(),
            'hip_optimizer_state_dict': self.hip_alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.high_alg.actor_critic.load_state_dict(
            loaded_dict['high_model_state_dict'],
            strict=True,
        )
        self.hip_alg.actor_critic.load_state_dict(
            loaded_dict['hip_model_state_dict'],
            strict=True,
        )
        self.mu_alg.actor_critic.load_state_dict(
            loaded_dict['mu_model_state_dict'],
            strict=True,
        )
        self.omega_alg.actor_critic.load_state_dict(
            loaded_dict['omega_model_state_dict'],
            strict=True,
        )
        if load_optimizer:
            if 'high_optimizer_state_dict' in loaded_dict:
                self.high_alg.optimizer.load_state_dict(
                    loaded_dict['high_optimizer_state_dict']
                )
            if 'mu_optimizer_state_dict' in loaded_dict:
                self.mu_alg.optimizer.load_state_dict(
                    loaded_dict['mu_optimizer_state_dict']
                )
            if 'omega_optimizer_state_dict' in loaded_dict:
                self.omega_alg.optimizer.load_state_dict(
                    loaded_dict['omega_optimizer_state_dict']
                )
            if 'hip_optimizer_state_dict' in loaded_dict:
                self.hip_alg.optimizer.load_state_dict(
                    loaded_dict['hip_optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.high_alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        self.hip_alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.high_alg.actor_critic.to(device)
            self.hip_alg.actor_critic.to(device)

        mu_policy, omega_policy = super().get_inference_policy()

        return self.high_alg.actor_critic.act_inference, mu_policy, omega_policy, self.hip_alg.actor_critic.act_inference