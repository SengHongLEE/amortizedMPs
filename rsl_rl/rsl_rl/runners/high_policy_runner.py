from rsl_rl.runners import OnPolicyRunner
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


class HighPolicyRunner(OnPolicyRunner):
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
        self.low_actor_critic: ActorCritic = low_actor_critic_class( self.env.num_obs - 1,
                                            num_critic_obs,
                                            self.env.num_actions,
                                            **self.policy_cfg).to(self.device)

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

        self.high_actor_critic: ActorCritic = high_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.low_freq_idx_shape,
                                            self.env.low_freq_idx_shape,
                                            **self.policy_cfg).to(self.device)
        self.high_alg: PPOevent_discrete = high_alg_class(self.high_actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #Storage
        self.high_alg.init_storage(self.env.num_envs,
                            int(5),
                            [self.env.num_obs],
                            [self.env.num_privileged_obs],
                            [1],
                            [self.env.low_freq_idx_shape])

        #Twin Rate
        self.actions = torch.zeros(
            self.env.num_envs,
            self.env.num_actions,
            device=self.device,
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.low_update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * min(self.env.cfg.control.low_freq_idx)

        self.high_reservoir_states = None
        self.low_reservoir_states = None

        if getattr(self.high_alg.actor_critic, "is_reservoir", False):
            self.high_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.high_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )
        if getattr(self.low_actor_critic, "is_reservoir", False):
            self.low_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.low_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )

        self.current_high_step = 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.high_tot_timesteps = 0
        self.high_tot_time = 0
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
    def inference_rollout(self, obs, high_policy, low_policy):
        low_map = torch.tensor(
            self.env.cfg.control.low_freq_idx,
            device=self.low_update_cycle.device,
            dtype=self.low_update_cycle.dtype,
        )
        high_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.high_cycle,
        )
        low_update_freq_idx = (self.low_update_cycle.unsqueeze(-1) == low_map).long().argmax(dim=-1)

        cycles = low_update_freq_idx.unsqueeze(-1)
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )
        high_obs = torch.cat((obs, cycles), dim=-1)
        if getattr(self, "high_reservoir_states", None) is not None:
            low_update_freq_idx, reservoir_states = high_policy(high_obs[high_update_env_ids], self.high_reservoir_states[high_update_env_ids])
            self.high_reservoir_states[high_update_env_ids] = reservoir_states
        else:
            low_update_freq_idx = high_policy(high_obs[high_update_env_ids,-1])

        self.low_update_cycle[high_update_env_ids] = low_map[low_update_freq_idx.long()]

        low_update_freq_idx = (self.low_update_cycle.unsqueeze(-1) == low_map).long().argmax(dim=-1)

        low_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.low_update_cycle,
        )

        # cycles = low_update_freq_idx.unsqueeze(-1)
        # cycles = cycles.unsqueeze(1).expand(
        #     -1, obs.size(1), -1
        # )
        # obs = torch.cat((obs[:, :, :-1], cycles), dim=-1)

        if getattr(self, "low_reservoir_states", None) is not None:
            low_actions, reservoir_states = low_policy(obs[low_update_env_ids,-1], self.low_reservoir_states[low_update_env_ids])
            self.low_reservoir_states[low_update_env_ids] = reservoir_states
        else:
            low_actions = low_policy(obs[low_update_env_ids,-1])

        self.actions[low_update_env_ids] = low_actions
 
        self.env.update_idx_buffer.append([self.low_update_cycle.cpu().numpy()])

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)

        print(self.low_update_cycle)
        # cycles = low_update_freq_idx.unsqueeze(-1)
        # cycles = cycles.unsqueeze(1).expand(
        #     -1, obs.size(1), -1
        # )

        # obs = torch.cat((obs, cycles), dim=-1)
        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)

        if getattr(self, "high_reservoir_states", None) is not None:
            self.high_reservoir_states[dones] = 0.0
        if getattr(self, "low_reservoir_states", None) is not None:
            self.low_reservoir_states[dones] = 0.0

        return obs, critic_obs, rewards, dones, infos

    @torch.no_grad
    def training_rollout(self, obs, critic_obs, current_high_step):
        self.low_actor_critic.eval()
        low_policy = self.low_actor_critic.act_inference
        low_map = torch.tensor(
                self.env.cfg.control.low_freq_idx,
                device=self.device,
                dtype=self.low_update_cycle.dtype,
                )
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

        low_update_freq_idx = (self.low_update_cycle.unsqueeze(-1) == low_map).long().argmax(dim=-1)

        cycles = low_update_freq_idx.unsqueeze(-1)
        cycles = cycles.unsqueeze(1).expand(
            -1, obs.size(1), -1
        )
        high_obs = torch.cat((obs, cycles), dim=-1)

        if getattr(self.high_alg.actor_critic, "is_reservoir", False):
            if high_update_env_ids.numel() > 0:
                update_freq_idx, reservoir_states = self.high_alg.act(high_update_env_ids, high_step_ids, high_obs, critic_obs, self.high_reservoir_states)
                self.high_reservoir_states[high_update_env_ids] = reservoir_states

                self.low_update_cycle[high_update_env_ids] = low_map[update_freq_idx[:, 0].long()]
        else:
            update_freq_idx = self.high_alg.act(high_update_env_ids, high_step_ids, high_obs[:, -1], critic_obs)

        self.low_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.low_update_cycle,
        )

        if getattr(self, "low_reservoir_states", None) is not None:
            low_actions, reservoir_states = low_policy(obs[self.low_update_env_ids,-1], self.low_reservoir_states[low_update_env_ids])
            self.low_reservoir_states[self.low_update_env_ids] = reservoir_states
        else:
            low_actions = low_policy(obs[self.low_update_env_ids,-1])

        self.last_actions[self.low_update_env_ids] = self.actions[self.low_update_env_ids]
        self.actions[self.low_update_env_ids] = low_actions
        self.env.actions_diff[self.low_update_env_ids] = torch.sum(torch.square(self.last_actions[self.low_update_env_ids] - self.actions[self.low_update_env_ids]), dim=1)

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)
        # cycles = low_update_freq_idx.unsqueeze(-1)
        # cycles = cycles.unsqueeze(1).expand(
        #     -1, obs.size(1), -1
        # )

        # obs = torch.cat((obs, cycles), dim=-1)
        critic_obs = privileged_obs if privileged_obs is not None else obs[:, -1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)
        rewards = [reward.to(self.device) for reward in rewards]
        self.high_alg.clear_start_transition()

        return obs, critic_obs, rewards, dones, infos


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        low_map = torch.tensor(
            self.env.cfg.control.low_freq_idx,
            device=self.low_update_cycle.device,
            dtype=self.low_update_cycle.dtype,
        )
        low_update_freq_idx = (self.low_update_cycle.unsqueeze(-1) == low_map).long().argmax(dim=-1)
        # cycles = low_update_freq_idx.unsqueeze(-1)
        # cycles = cycles.unsqueeze(1).expand(
        #     -1, obs.size(1), -1
        # )

        # obs = torch.cat((obs, cycles), dim=-1)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.high_alg.actor_critic.train()

        high_ep_infos = []
        high_ep_counts = []
        high_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_high_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        decision_episode_sum = torch.zeros(self.env.num_envs, device=self.device)

        #Accumulate Rewards
        high_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        high_it = self.current_learning_iteration

        # Rollout
        high_collection_start = time.time()
        while True:
            obs, critic_obs, rewards, dones, infos = self.training_rollout(obs, critic_obs, self.current_high_step)

            track_max = self.env.reward_scales["tracking_lin_vel"] + self.env.reward_scales["tracking_ang_vel"]
            track_quality = torch.clamp(
                rewards[1] / (track_max + 1e-6),
                min=0.0,
                max=1.0,
            )

            eta = 1.
            threshold_25 = 0.8
            threshold_50 = 0.9
            tmp_10 = (self.low_update_cycle == 10).float() * torch.clip(track_quality, min=0., max=threshold_25)
            tmp_25 = (self.low_update_cycle == 25).float() * torch.clip(track_quality + eta * (track_quality - threshold_25), min=0., max=(threshold_25 + eta * (threshold_50 - threshold_25)))
            tmp_50 = (self.low_update_cycle == 50).float() * torch.clip(track_quality + eta * (track_quality - threshold_25) + eta * (track_quality - threshold_50), min=0.)

            high_reward = track_max * (tmp_10 + tmp_25 + tmp_50)

            decision_reward = high_reward
            high_acc_rewards += high_reward

            env_ids = dones.nonzero(as_tuple=False).flatten()

            self.current_high_step += 1

            high_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.env.cfg.control.high_cycle,
            )

            if self.high_reservoir_states is not None:
                self.high_reservoir_states[dones] = 0.
            if self.low_reservoir_states is not None:
                self.low_reservoir_states[dones] = 0.

            self.high_alg.process_env_step(
                self.current_high_step,
                high_update_env_ids,
                high_acc_rewards,
                dones,
                critic_obs,
                infos,
            )

            if self.log_dir is not None:
                # Book keeping
                cur_high_reward_sum += high_reward

                cur_episode_length += 1
                # Keep runner rewards separate from the sums cleared by env.reset_idx().
                decision_episode_sum += decision_reward
                if env_ids.numel() > 0:
                    episode_info = infos.get('episode', {}).copy()
                    episode_info['rew_decision'] = torch.mean(decision_episode_sum[env_ids]) / self.env.max_episode_length_s
                    high_ep_infos.append(episode_info)
                    high_ep_counts.append(env_ids.numel())
                    high_rewbuffer.extend(cur_high_reward_sum[env_ids].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[env_ids].cpu().numpy().tolist())
                    cur_high_reward_sum[env_ids] = 0.
                    cur_episode_length[env_ids] = 0.
                    decision_episode_sum[env_ids] = 0.

            high_acc_rewards[high_update_env_ids] = 0.

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
                    self.save_model(self.log_dir, high_it, 0)

                high_it += 1
                high_collection_start = time.time()
                high_ep_infos.clear()
                high_ep_counts.clear()

            if high_it >= tot_iter:
                break

        self.current_learning_iteration += num_learning_iterations
        self.save_model(self.log_dir, high_it, 0)
        self.final_log()

    def save_model(self, log_dir, high_it, low_it):
        log_dir = Path(log_dir)

        pattern = re.compile(r"^model_(\d+)_high-\d+_low-\d+\.pt$")
        existing_indices = []
        for file in log_dir.glob("*.pt"):
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        model_idx = max(existing_indices, default=-1) + 1

        save_path = log_dir / (
            f"model_{model_idx}_high-{high_it}_low-{low_it}.pt"
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
            episode_counts = torch.tensor(locs['high_ep_counts'], device=self.device)
            for key in locs['high_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['high_ep_infos']:
                    ep_value = torch.as_tensor(ep_info[key], device=self.device).float().mean().view(1)
                    infotensor = torch.cat((infotensor, ep_value))
                value = torch.sum(infotensor * episode_counts) / episode_counts.sum()
                self.writer.add_scalar('Episode/high/' + key, value, locs['high_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_entropy = self.high_alg.mean_entropy
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/high/value_function', locs['mean_high_value_loss'], locs['high_it'])
        self.writer.add_scalar('Loss/high/surrogate', locs['mean_high_surrogate_loss'], locs['high_it'])
        self.writer.add_scalar('Loss/high/learning_rate', self.high_alg.learning_rate, locs['high_it'])
        self.writer.add_scalar('Policy/high/entropy', mean_entropy, locs['high_it'])
        self.writer.add_scalar('Perf/high/total_fps', fps, locs['high_it'])
        self.writer.add_scalar('Perf/high/collection time', locs['high_collection_time'], locs['high_it'])
        self.writer.add_scalar('Perf/high/learning_time', locs['high_learn_time'], locs['high_it'])
        if len(locs['high_rewbuffer']) > 0:
            self.writer.add_scalar('Train/high/mean_reward', statistics.mean(locs['high_rewbuffer']), locs['high_it'])
            self.writer.add_scalar('Train/high/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['high_it'])
            self.writer.add_scalar('Train/high/mean_reward/time', statistics.mean(locs['high_rewbuffer']), self.high_tot_time)
            self.writer.add_scalar('Train/high/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.high_tot_time)

        str = f" \033[1m Learning iteration {locs['high_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['high_rewbuffer']) > 0:
            log_string = (f"""{'#' * width} high {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'high_collection_time']:.3f}s, learning {locs['high_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_high_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_high_surrogate_loss']:.4f}\n"""
                          f"""{'Mean policy entropy:':>{pad}} {mean_entropy:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['high_rewbuffer']):.2f}\n"""
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
                          f"""{'Mean policy entropy:':>{pad}} {mean_entropy:.4f}\n""")
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

    def low_log(self, locs, width=40, pad=35):
        self.low_tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.low_tot_time += locs['low_collection_time'] + locs['low_learn_time']
        iteration_time = locs['low_collection_time'] + locs['low_learn_time']

        ep_string = f''
        if locs['low_ep_infos']:
            episode_counts = torch.tensor(locs['low_ep_counts'], device=self.device)
            for key in locs['low_ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['low_ep_infos']:
                    ep_value = torch.as_tensor(ep_info[key], device=self.device).float().mean().view(1)
                    infotensor = torch.cat((infotensor, ep_value))
                value = torch.sum(infotensor * episode_counts) / episode_counts.sum()
                self.writer.add_scalar('Episode/low/' + key, value, locs['low_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.low_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/low/value_function', locs['mean_low_value_loss'], locs['low_it'])
        self.writer.add_scalar('Loss/low/surrogate', locs['mean_low_surrogate_loss'], locs['low_it'])
        self.writer.add_scalar('Loss/low/learning_rate', self.low_alg.learning_rate, locs['low_it'])
        self.writer.add_scalar('Policy/low/mean_noise_std', mean_std.item(), locs['low_it'])
        self.writer.add_scalar('Perf/low/total_fps', fps, locs['low_it'])
        self.writer.add_scalar('Perf/low/collection time', locs['low_collection_time'], locs['low_it'])
        self.writer.add_scalar('Perf/low/learning_time', locs['low_learn_time'], locs['low_it'])
        if len(locs['low_rewbuffer']) > 0:
            self.writer.add_scalar('Train/low/mean_reward', statistics.mean(locs['low_rewbuffer']), locs['low_it'])
            self.writer.add_scalar('Train/low/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['low_it'])
            self.writer.add_scalar('Train/low/mean_reward/time', statistics.mean(locs['low_rewbuffer']), self.low_tot_time)
            self.writer.add_scalar('Train/low/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.low_tot_time)

        str = f" \033[1m Learning iteration {locs['low_it']}/{locs['tot_iter']} \033[0m "

        if len(locs['low_rewbuffer']) > 0:
            log_string = (f"""{'#' * width} low {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'low_collection_time']:.3f}s, learning {locs['low_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_low_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_low_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['low_rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'low_collection_time']:.3f}s, learning {locs['low_learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_low_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_low_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.low_tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.low_tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.low_tot_time / (locs['low_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['low_it']):.1f}s\n""")
        print(log_string)


    def save(self, path, infos=None):
        torch.save({
            'high_model_state_dict': self.high_actor_critic.state_dict(),
            'low_model_state_dict': self.low_actor_critic.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.high_actor_critic.load_state_dict(
            loaded_dict['high_model_state_dict'],
            strict=True,
        )
        self.low_actor_critic.load_state_dict(
            loaded_dict['low_model_state_dict'],
            strict=True,
        )
        if load_optimizer:
            if 'high_optimizer_state_dict' in loaded_dict:
                self.high_alg.optimizer.load_state_dict(
                    loaded_dict['high_optimizer_state_dict']
                )
            if 'low_optimizer_state_dict' in loaded_dict:
                self.low_alg.optimizer.load_state_dict(
                    loaded_dict['low_optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.high_actor_critic.eval() # switch to evaluation mode (dropout for example)
        self.low_actor_critic.eval()
        if device is not None:
            self.high_actor_critic.to(device)
            self.low_actor_critic.to(device)
        high_policy = self.high_actor_critic.act_inference
        low_policy = self.low_actor_critic.act_inference

        return high_policy, low_policy
