from rsl_rl.runners import OnPolicyRunner, InteroceptivePolicyRunner
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


class MultiInteroceptivePolicyRunner(OnPolicyRunner):
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
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #Primitive Selection
        primitive_actor_critic_class = _resolve_class(
            self.cfg["primitive_policy_class_name"],
            _POLICY_CLASSES,
            "policy class",
        )
        primitive_alg_class = _resolve_class(
            self.cfg["primitive_algorithm_class_name"],
            _ALGORITHM_CLASSES,
            "algorithm class",
        )
        self.primitive_selection_actor_critic: ActorCriticDiscrete = primitive_actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.num_primitives,
                                            self.env.num_primitives,
                                            **self.policy_cfg).to(self.device)
        
        self.primitive_selection_alg: PPOevent_discrete = primitive_alg_class(self.primitive_selection_actor_critic, device=self.device, **self.alg_cfg)

        self.primitive_selection_reservoir_states = None
        if getattr(self.primitive_selection_alg.actor_critic, "is_reservoir", False):
            self.primitive_selection_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.primitive_selection_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )

        policy_frequency_ratio = self.env.cfg.control.modulation_param_update_cycle / self.env.cfg.control.primitive_selection_update_cycle
        self.primitive_selection_alg.init_storage(self.env.num_envs,
                              int(policy_frequency_ratio * self.num_steps_per_env),
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [1],
                              [self.env.num_primitives])

        #Modulation_Parameter
        modulation_param_actor_critic_class = _resolve_class(
            self.cfg["modulation_param_policy_class_name"],
            _POLICY_CLASSES,
            "policy class",
        )
        modulation_param_alg_class = _resolve_class(
            self.cfg["modulation_param_algorithm_class_name"],
            _ALGORITHM_CLASSES,
            "algorithm class",
        )
        self.modulation_param_actor_critic: ActorCritic = modulation_param_actor_critic_class(self.env.num_obs,
                                            num_critic_obs,
                                            self.env.num_modulation_params,
                                            **self.policy_cfg).to(self.device)
        
        self.modulation_param_alg: PPOevent = modulation_param_alg_class(self.modulation_param_actor_critic, device=self.device, **self.alg_cfg)

        self.modulation_param_reservoir_states = None
        if getattr(self.modulation_param_alg.actor_critic, "is_reservoir", False):
            self.modulation_param_reservoir_states = torch.zeros(
                self.env.num_envs,
                self.modulation_param_alg.actor_critic.actor.reservoir_state_dim,
                device=self.device,
            )

        self.modulation_param_alg.init_storage(self.env.num_envs,
                              int(self.num_steps_per_env),
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.num_modulation_params])

        # Action
        self.primitive_selections = torch.zeros(self.env.num_envs,
                                                dtype=torch.long,
                                               device=self.device)
        self.last_primitive_selections = torch.zeros_like(self.primitive_selections)
        self.modulation_params = torch.zeros(self.env.num_envs,
                                            self.env.num_modulation_params,
                                            device=self.device)
        self.last_modulation_params = torch.zeros_like(self.modulation_params)
        self.actions = torch.zeros(
            self.env.num_envs,
            self.env.num_actions,
            device=self.device,
        )
        self.last_actions = torch.zeros_like(self.actions)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.num = 0
        self.current_step = 0
        self.primitive_selection_update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * self.env.cfg.control.primitive_selection_update_cycle
        self.modulation_param_update_cycle = torch.ones(self.env.num_envs, dtype=torch.long, device=self.device) * self.env.cfg.control.modulation_param_update_cycle

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
    def inference_rollout(self, obs, primitive_selection_actor, modulation_parameter_actor):
        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.primitive_selection_update_cycle,
        )
        update_env_ids = next_update_env_ids
        # update_env_ids = next_update_env_ids[
        #     self.env.behavior_quality[next_update_env_ids] < self.env.cfg.control.update_quality_threshold
        # ]
        if update_env_ids.numel() == 0:
            primitive_selections = self.primitive_selections[update_env_ids]
        elif getattr(self, "primitive_selection_reservoir_states", None) is not None:
            primitive_selections, reservoir_states = primitive_selection_actor(obs[update_env_ids], self.primitive_selection_reservoir_states[update_env_ids])
            self.primitive_selection_reservoir_states[update_env_ids] = reservoir_states
        else:
            primitive_selections = primitive_selection_actor(obs[update_env_ids,-1])
        self.last_primitive_selections[update_env_ids] = self.primitive_selections[update_env_ids]
        self.primitive_selections[update_env_ids] = primitive_selections

        self.actions[update_env_ids, 0] = self.primitive_selections[update_env_ids].float()

        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.modulation_param_update_cycle,
        )
        update_env_ids = next_update_env_ids
        # update_env_ids = next_update_env_ids[
        #     self.env.behavior_quality[next_update_env_ids] < self.env.cfg.control.update_quality_threshold
        # ]
        if update_env_ids.numel() == 0:
            modulation_params = self.modulation_params[update_env_ids]
        elif getattr(self, "modulation_param_reservoir_states", None) is not None:
            modulation_params, reservoir_states = modulation_parameter_actor(obs[update_env_ids,-1], self.modulation_param_reservoir_states[update_env_ids])
            self.modulation_param_reservoir_states[update_env_ids] = reservoir_states
        else:
            modulation_params = modulation_parameter_actor(obs[update_env_ids,-1])

        self.last_modulation_params[update_env_ids] = self.modulation_params[update_env_ids]
        self.modulation_params[update_env_ids] = modulation_params

        self.actions[update_env_ids, 1:] = self.modulation_params[update_env_ids]


        # self.last_actions[update_env_ids] = self.actions[update_env_ids]
        # self.actions[update_env_ids, 0] = self.primitive_selections[update_env_ids].float()
        # self.actions[update_env_ids, 1:] = self.modulation_params[update_env_ids]
        # self.env.primitive_diff[update_env_ids] = torch.sum(torch.square(self.last_actions[update_env_ids, :1] - self.actions[update_env_ids, :1]), dim=1)
        # self.env.param_actions_diff[update_env_ids] = torch.sum(torch.square(self.last_actions[update_env_ids, 1:] - self.actions[update_env_ids, 1:]), dim=1)

        # self.env.update_idx_buffer.append([self.update_cycle.cpu().numpy()])

        if update_env_ids.numel() > 0:
            self.num += 1
        print(self.num)

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)

        critic_obs = privileged_obs if privileged_obs is not None else obs[:,-1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)

        if getattr(self, "primitive_selection_reservoir_states", None) is not None:
            self.primitive_selection_reservoir_states[dones] = 0.0
        if getattr(self, "modulation_param_reservoir_states", None) is not None:
            self.modulation_param_reservoir_states[dones] = 0.0

        return obs, critic_obs, rewards, dones, infos

    @torch.no_grad
    def training_rollout(self, obs, critic_obs, current_step):
        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.primitive_selection_update_cycle)

        step_ids = torch.full(
            (next_update_env_ids.shape[0],),
            current_step,
            dtype=torch.long,
            device=self.device,
        )

        if next_update_env_ids.numel() == 0:
            primitive_selections = self.primitive_selections[next_update_env_ids, None]
        elif getattr(self.primitive_selection_alg.actor_critic, "is_reservoir", False):
            primitive_selections, reservoir_states = self.primitive_selection_alg.act(next_update_env_ids, 
                                                                            step_ids, 
                                                                            obs, 
                                                                            critic_obs, 
                                                                            self.primitive_selection_reservoir_states)
            self.primitive_selection_reservoir_states[next_update_env_ids] = reservoir_states
        else:
            primitive_selections = self.primitive_selection_alg.act(next_update_env_ids, 
                                                          step_ids, 
                                                          obs[:, -1], 
                                                          critic_obs)
        self.last_primitive_selections[next_update_env_ids] = self.primitive_selections[next_update_env_ids]
        self.primitive_selections[next_update_env_ids] = primitive_selections[:, 0]

        self.last_actions[next_update_env_ids, 0] = self.actions[next_update_env_ids, 0]
        self.actions[next_update_env_ids, 0] = self.primitive_selections[next_update_env_ids].float()
        self.env.primitive_diff[next_update_env_ids] = torch.sum(torch.square(self.last_actions[next_update_env_ids, :1] - self.actions[next_update_env_ids, :1]), dim=1)

        next_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.modulation_param_update_cycle)

        step_ids = torch.full(
            (next_update_env_ids.shape[0],),
            current_step,
            dtype=torch.long,
            device=self.device,
        )

        if next_update_env_ids.numel() == 0:
            modulation_params = self.modulation_params[next_update_env_ids]
        elif getattr(self.modulation_param_alg.actor_critic, "is_reservoir", False):
            modulation_params, reservoir_states = self.modulation_param_alg.act(next_update_env_ids, 
                                                                                step_ids, 
                                                                                obs[:, -1], 
                                                                                critic_obs, 
                                                                                self.modulation_param_reservoir_states)
            self.modulation_param_reservoir_states[next_update_env_ids] = reservoir_states
        else:
            modulation_params = self.modulation_param_alg.act(next_update_env_ids, 
                                                              step_ids, 
                                                              obs[:, -1],
                                                              critic_obs)
        self.last_modulation_params[next_update_env_ids] = self.modulation_params[next_update_env_ids]
        self.modulation_params[next_update_env_ids] = modulation_params

        self.last_actions[next_update_env_ids, 1:] = self.actions[next_update_env_ids, 1:]
        self.actions[next_update_env_ids, 1:] = self.modulation_params[next_update_env_ids]
        previous = self.last_actions[next_update_env_ids, 1:]
        current = self.actions[next_update_env_ids, 1:]
        action_cost = getattr(getattr(self.env, 'ID', None), 'continuous_action_change_cost', None)
        self.env.param_actions_diff[next_update_env_ids] = (
            action_cost(previous, current) if action_cost is not None
            else (current - previous).square().sum(-1))

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)

        critic_obs = privileged_obs if privileged_obs is not None else obs[:, -1]
        obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)
        rewards = [reward.to(self.device) for reward in rewards]
        self.primitive_selection_alg.clear_start_transition()
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
        cur_primitive_selection_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_modulation_param_reward_sum = torch.zeros_like(cur_primitive_selection_reward_sum)
        cur_reward_sum = torch.zeros_like(cur_primitive_selection_reward_sum)
        cur_episode_length = torch.zeros_like(cur_primitive_selection_reward_sum)

        #Accumulate Rewards
        primitive_selection_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)
        modulation_param_acc_rewards = torch.zeros_like(primitive_selection_acc_rewards)

        tot_iter = num_learning_iterations
        it = 0

        collection_start = time.time()
        self.primitive_selection_alg.actor_critic.train()
        self.modulation_param_alg.actor_critic.train()

        while True:
            obs, critic_obs, rewards, dones, infos = self.training_rollout(obs, critic_obs, self.current_step)
    
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self.current_step += 1

            #modulation_parameter
            storage = self.modulation_param_alg.storage
            pending_env_ids = storage.pending.nonzero(as_tuple=False).flatten()
            modulation_param_acc_rewards[pending_env_ids] += rewards[2][pending_env_ids]

            next_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.modulation_param_update_cycle,
            )
            completed_mask = dones.bool().clone()
            completed_mask[next_update_env_ids] = True
            completed_env_ids = pending_env_ids[completed_mask[pending_env_ids]]

            if completed_env_ids.numel() > 0:
                self.modulation_param_alg.process_env_step(
                    self.current_step,
                    completed_env_ids,
                    modulation_param_acc_rewards,
                    dones,
                    critic_obs,
                    infos,
                )
            modulation_param_acc_rewards[completed_env_ids] = 0.

            if self.modulation_param_reservoir_states is not None:
                self.modulation_param_reservoir_states[dones] = 0.

            #Primitive Selection
            storage = self.primitive_selection_alg.storage
            pending_env_ids = storage.pending.nonzero(as_tuple=False).flatten()
            primitive_selection_acc_rewards[pending_env_ids] += rewards[1][pending_env_ids]

            next_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.primitive_selection_update_cycle,
            )
            completed_mask = dones.bool().clone()
            completed_mask[next_update_env_ids] = True
            completed_env_ids = pending_env_ids[completed_mask[pending_env_ids]]
            
            if completed_env_ids.numel() > 0:
                self.primitive_selection_alg.process_env_step(
                    self.current_step,
                    completed_env_ids,
                    primitive_selection_acc_rewards,
                    dones,
                    critic_obs,
                    infos,
                )
            primitive_selection_acc_rewards[completed_env_ids] = 0.

            if self.primitive_selection_reservoir_states is not None:
                self.primitive_selection_reservoir_states[dones] = 0.

            # Book keeping
            if self.log_dir is not None:
                cur_reward_sum += rewards[0]
                cur_primitive_selection_reward_sum += rewards[1]
                cur_modulation_param_reward_sum += rewards[2]
                cur_episode_length += 1
                if env_ids.numel() > 0:
                    episode_info = infos.get('episode', {}).copy()
                    ep_infos.append(episode_info)
                    ep_counts.append(env_ids.numel())

                    rewbuffer.extend(cur_reward_sum[env_ids].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[env_ids].cpu().numpy().tolist())
                    cur_primitive_selection_reward_sum[env_ids] = 0.
                    cur_modulation_param_reward_sum[env_ids] = 0.
                    cur_reward_sum[env_ids] = 0.
                    cur_episode_length[env_ids] = 0.

            if self.primitive_selection_alg.storage.ready:
                self.primitive_selection_alg.compute_returns(critic_obs)
                mean_primitive_selection_value_loss, mean_primitive_selection_surrogate_loss = self.primitive_selection_alg.update()

            
            if self.modulation_param_alg.storage.ready:
                collection_time = time.time() - collection_start
                learn_start = time.time()
                self.modulation_param_alg.compute_returns(critic_obs)
                mean_modulation_param_value_loss, mean_modulation_param_surrogate_loss = self.modulation_param_alg.update()
                learn_stop = time.time()
                learn_time = learn_stop - learn_start

                it += 1
                self.current_learning_iteration += 1
                if self.log_dir is not None:
                    self.log(locals())
                    if it % self.save_interval == 0:
                        self.save_model(self.log_dir, self.current_learning_iteration)

                collection_start = time.time()
                ep_infos.clear()
                ep_counts.clear()
            if it >= tot_iter:
                break

        if self.log_dir is not None:
            self.save_model(self.log_dir, self.current_learning_iteration)
        self.final_log()


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
        # mean_primitive_selection_entropy = self.primitive_selection_alg.mean_entropy
        mean_modulation_param_std = self.modulation_param_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/value_function', (locs['mean_modulation_param_value_loss']), locs['it'])
        self.writer.add_scalar('Loss/surrogate', (locs['mean_modulation_param_surrogate_loss']), locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.modulation_param_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_modulation_param_std.item(), locs['it'])
        # self.writer.add_scalar('Policy/entropy', mean_primitive_selection_entropy, locs['it'])
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
                          f"""{'Value function loss:':>{pad}} {(locs['mean_modulation_param_value_loss']):.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {(locs['mean_modulation_param_surrogate_loss']):.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_modulation_param_std.item():.2f}\n"""
                        #   f"""{'Mean policy entropy:':>{pad}} {mean_primitive_selection_entropy:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(2*width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {0.5 * (locs['mean_primitive_selection_value_loss'] + locs['mean_modulation_param_value_loss']):.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {0.5 * (locs['mean_primitive_selection_surrogate_loss'] + locs['mean_modulation_param_surrogate_loss']):.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_modulation_param_std.item():.2f}\n"""
                        #   f"""{'Mean policy entropy:':>{pad}} {mean_primitive_selection_entropy:.4f}\n"""
                         )
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
            'primitive_selection_model_state_dict': self.primitive_selection_actor_critic.state_dict(),
            'modulation_param_model_state_dict': self.modulation_param_actor_critic.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.primitive_selection_actor_critic.load_state_dict(
            loaded_dict['primitive_selection_model_state_dict'],
            strict=True,
        )
        self.modulation_param_actor_critic.load_state_dict(
            loaded_dict['modulation_param_model_state_dict'],
            strict=True,
        )
        if load_optimizer:
            if 'primitive_selection_optimizer_state_dict' in loaded_dict:
                self.primitive_selection_alg.optimizer.load_state_dict(
                    loaded_dict['primitive_selection_optimizer_state_dict']
                )
            if 'modulation_param_optimizer_state_dict' in loaded_dict:
                self.modulation_param_alg.optimizer.load_state_dict(
                    loaded_dict['modulation_param_optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.primitive_selection_actor_critic.eval()
        self.modulation_param_actor_critic.eval()
        if device is not None:
            self.primitive_selection_actor_critic.to(device)
            self.modulation_param_actor_critic.to(device)
        primitive_selection_actor = self.primitive_selection_actor_critic.act_inference
        modulation_param_actor = self.modulation_param_actor_critic.act_inference

        return primitive_selection_actor, modulation_param_actor
