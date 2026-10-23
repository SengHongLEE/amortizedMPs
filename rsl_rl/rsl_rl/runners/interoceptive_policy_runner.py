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


class InteroceptivePolicyRunner(OnPolicyRunner):
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

        actor_critic_class = _resolve_class(
            self.cfg["policy_class_name"],
            _POLICY_CLASSES,
            "policy class",
        )
        alg_class = _resolve_class(
            self.cfg["algorithm_class_name"],
            _ALGORITHM_CLASSES,
            "algorithm class",
        )
        self.modulation_param_actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                            num_critic_obs,
                                            self.env.num_actions,
                                            **self.policy_cfg).to(self.device)
        
        self.modulation_param_alg: PPOevent = alg_class(self.modulation_param_actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #Storage
        self.modulation_param_alg.init_storage(self.env.num_envs,
                              int(self.num_steps_per_env),
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.num_actions])

        self.actions = torch.zeros(
            self.env.num_envs,
            self.env.num_actions,
            device=self.device,
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * self.env.cfg.control.update_cycle

        self.modulation_param_reservoir_states = None

        if getattr(self.modulation_param_alg.actor_critic, "is_reservoir", False):
            self.modulation_param_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.modulation_param_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )

        self.current_step = 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.num = 0

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
    def inference_rollout(self, obs, policy):
        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.update_cycle,
        )
        update_env_ids = next_update_env_ids[
            self.env.behavior_quality[next_update_env_ids] < self.env.cfg.control.update_quality_threshold
        ]
        if getattr(self, "reservoir_states", None) is not None:
            actions, reservoir_states = policy(obs[update_env_ids,-1], self.modulation_param_reservoir_states[update_env_ids])
            self.modulation_param_reservoir_states[update_env_ids] = reservoir_states
        else:
            actions = policy(obs[update_env_ids,-1])

        self.last_actions[update_env_ids] = self.actions[update_env_ids]
        self.actions[update_env_ids] = actions
        self.env.actions_diff[update_env_ids] = torch.sum(torch.square(self.last_actions[update_env_ids] - self.actions[update_env_ids]), dim=1)
 
        self.env.update_idx_buffer.append([self.update_cycle.cpu().numpy()])

        if update_env_ids.numel() > 0:
            self.num += 1
        print(self.num)

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)

        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)

        if getattr(self, "eservoir_states", None) is not None:
            self.modulation_param_reservoir_states[dones] = 0.0

        return obs, critic_obs, rewards, dones, infos

    @torch.no_grad
    def training_rollout(self, obs, critic_obs, current_step):
        
        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.update_cycle)
        self.update_env_ids = next_update_env_ids

        step_ids = torch.full(
            (self.update_env_ids.shape[0],),
            current_step,
            dtype=torch.long,
            device=self.device,
        )
        if getattr(self.modulation_param_alg.actor_critic, "is_reservoir", False):
            actions, reservoir_states = self.modulation_param_alg.act(self.update_env_ids, step_ids, obs[:, -1], critic_obs, self.modulation_param_reservoir_states)
            self.modulation_param_reservoir_states[self.update_env_ids] = reservoir_states
        else:
            actions = self.modulation_param_alg.act(self.update_env_ids, step_ids, obs[:, -1], critic_obs)

        self.last_actions[self.update_env_ids] = self.actions[self.update_env_ids]
        self.actions[self.update_env_ids] = actions
        self.env.actions_diff[self.update_env_ids] = torch.sum(torch.square(self.last_actions[self.update_env_ids] - self.actions[self.update_env_ids]), dim=1)

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)

        critic_obs = privileged_obs if privileged_obs is not None else obs[:, -1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)
        rewards = [reward.to(self.device) for reward in rewards]
        self.modulation_param_alg.clear_start_transition()

        return obs, critic_obs, rewards, dones, infos

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        ep_infos = []
        ep_counts = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        #Accumulate Rewards
        acc_rewards = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = num_learning_iterations
        it = 0

        collection_start = time.time()
        self.modulation_param_alg.actor_critic.train()

        while True:
            obs, critic_obs, acc_rewards, cur_reward_sum, cur_episode_length, ep_infos, ep_counts, rewbuffer, lenbuffer = self.low_learn(obs, critic_obs, acc_rewards, cur_reward_sum, cur_episode_length, ep_infos, ep_counts, rewbuffer, lenbuffer)
            if self.modulation_param_alg.storage.ready:
                collection_time = time.time() - collection_start
                learn_start = time.time()
                self.modulation_param_alg.compute_returns(critic_obs)
                mean_value_loss, mean_surrogate_loss = self.modulation_param_alg.update()
                learn_stop = time.time()
                learn_time = learn_stop - learn_start
                if self.log_dir is not None:
                    it += 1
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save_model(self.log_dir, it)

                collection_start = time.time()
                ep_infos.clear()
                ep_counts.clear()
            if it >= tot_iter:
                break

        self.save_model(self.log_dir, it)       
        self.final_log()

    def low_learn(self, obs, critic_obs, acc_rewards, cur_reward_sum, cur_episode_length, ep_infos, ep_counts, rewbuffer, lenbuffer):
        obs, critic_obs, rewards, dones, infos = self.training_rollout(obs, critic_obs, self.current_step)
    
        env_ids = dones.nonzero(as_tuple=False).flatten()
        self.current_step += 1
        storage = self.modulation_param_alg.storage
        pending_env_ids = storage.pending.nonzero(as_tuple=False).flatten()
        acc_rewards[pending_env_ids] += rewards[1][pending_env_ids]

        # Settle the old action before the next rollout can call act again.
        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.update_cycle,
        )
        completed_mask = dones.bool().clone()
        completed_mask[next_update_env_ids] = True
        completed_env_ids = pending_env_ids[completed_mask[pending_env_ids]]

        if completed_env_ids.numel() > 0:
            durations = (
                self.current_step - storage.step_ids[storage.env_ptrs[completed_env_ids]]
            ).clamp_min(1)
            # Cancel process_env_step's duration averaging; keep the reward sum.
            scaled_rewards = acc_rewards.clone()
            # scaled_rewards[completed_env_ids] *= durations
            self.modulation_param_alg.process_env_step(
                self.current_step,
                completed_env_ids,
                scaled_rewards,
                dones,
                critic_obs,
                infos,
            )
        acc_rewards[completed_env_ids] = 0.

        if self.modulation_param_reservoir_states is not None:
            self.modulation_param_reservoir_states[dones] = 0.

        if self.log_dir is not None:
            # Book keeping
            cur_reward_sum += rewards[1]
            cur_episode_length += 1
            if env_ids.numel() > 0:
                episode_info = infos.get('episode', {}).copy()
                ep_infos.append(episode_info)
                ep_counts.append(env_ids.numel())

                rewbuffer.extend(cur_reward_sum[env_ids].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[env_ids].cpu().numpy().tolist())
                cur_reward_sum[env_ids] = 0.
                cur_episode_length[env_ids] = 0.

        return obs, critic_obs, acc_rewards, cur_reward_sum, cur_episode_length, ep_infos, ep_counts, rewbuffer, lenbuffer


    def save_model(self, log_dir, it):
        log_dir = Path(log_dir)

        pattern = re.compile(r"^model_(\d+)_\d+\.pt$")
        existing_indices = []
        for file in log_dir.glob("*.pt"):
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        model_idx = max(existing_indices, default=-1) + 1

        save_path = log_dir / (
            f"model_{model_idx}_{it}.pt"
        )

        self.save(str(save_path))


    def final_log(self, width=40, pad=35):
        log_string = (f"""{'#' * width}\n"""
                        f"""Training done""")
        print(log_string)

    def log(self, locs, width=40, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            episode_counts = torch.tensor(locs['ep_counts'], device=self.device)
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    ep_value = torch.as_tensor(ep_info[key], device=self.device).float().mean().view(1)
                    infotensor = torch.cat((infotensor, ep_value))
                value = torch.sum(infotensor * episode_counts) / episode_counts.sum()
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.modulation_param_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.modulation_param_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width} Training {'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'--' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)


    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.modulation_param_actor_critic.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.modulation_param_actor_critic.load_state_dict(
            loaded_dict['model_state_dict'],
            strict=True,
        )
        if load_optimizer:
            if 'optimizer_state_dict' in loaded_dict:
                self.modulation_param_alg.optimizer.load_state_dict(
                    loaded_dict['optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.modulation_param_actor_critic.eval()
        if device is not None:
            self.modulation_param_actor_critic.to(device)
        policy = self.modulation_param_actor_critic.act_inference

        return policy
