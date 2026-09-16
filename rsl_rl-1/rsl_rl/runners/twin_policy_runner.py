from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticTwin
from rsl_rl.algorithms import PPOtwin

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
}
_ALGORITHM_CLASSES = {
    "PPOtwin": PPOtwin,
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


class TwinPolicyRunner(OnPolicyRunner):
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

        mu_actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.mu_shape,
                                            **self.policy_cfg).to(self.device)

        self.mu_alg: PPOtwin = alg_class(mu_actor_critic, device=self.device, **self.alg_cfg)

        omega_actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                            num_critic_obs,
                                            self.env.omega_shape,
                                            **self.policy_cfg).to(self.device)

        self.omega_alg: PPOtwin = alg_class(omega_actor_critic, device=self.device, **self.alg_cfg)


        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #Storage
        self.mu_alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.mu_shape])
        self.omega_alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.omega_shape])

        #Twin Rate
        self.mu_cached = torch.zeros(self.env.num_envs, self.env.mu_shape, device=self.device)
        self.omega_cached = torch.zeros(self.env.num_envs, self.env.omega_shape, device=self.device)
        self.actions = torch.zeros(
            self.env.num_envs,
            self.env.mu_shape + self.env.omega_shape,
            device=self.device,
        )
        self.mu_update_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.omega_update_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        self.mu_reservoir_states = None
        self.omega_reservoir_states = None

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

        self.current_mu_step = 0
        self.current_omega_step = 0


        # Log
        self.log_dir = log_dir
        self.writer = None
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
    def inference_rollout(self, obs, mu_policy, omega_policy):
        mu_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.mu_cycle,
        )
        omega_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.omega_cycle,
        )




        if getattr(self, "mu_reservoir_states", None) is not None:
            mu_actions, reservoir_states = mu_policy(obs[mu_update_env_ids], self.mu_reservoir_states[mu_update_env_ids])
            self.mu_reservoir_states[mu_update_env_ids] = reservoir_states
        else:
            mu_actions = mu_policy(obs[mu_update_env_ids])

        if getattr(self, "omega_reservoir_states", None) is not None:
            omega_actions, reservoir_states = omega_policy(obs[omega_update_env_ids], self.omega_reservoir_states[omega_update_env_ids])
            self.omega_reservoir_states[omega_update_env_ids] = reservoir_states
        else:
            omega_actions = omega_policy(obs[omega_update_env_ids])



        self.mu_cached[mu_update_env_ids] = mu_actions
        self.omega_cached[omega_update_env_ids] = omega_actions
        self.actions[:, :self.env.mu_shape] = self.mu_cached
        self.actions[:, self.env.mu_shape:] = self.omega_cached

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

        if getattr(self, "mu_reservoir_states", None) is not None:
            self.mu_reservoir_states[dones] = 0.0
        if getattr(self, "omega_reservoir_states", None) is not None:
            self.omega_reservoir_states[dones] = 0.0

        return obs, critic_obs, rewards, dones, infos

    @torch.no_grad
    def training_rollout(self, obs, critic_obs, current_mu_step, current_omega_step):
        mu_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.mu_cycle,
        )
        omega_update_env_ids = self.get_update_env_ids(
            self.env.episode_length_buf,
            self.env.cfg.control.omega_cycle,
        )

        mu_step_ids = torch.full(
            (mu_update_env_ids.shape[0],),
            current_mu_step,
            dtype=torch.long,
            device=self.device,
        )
        omega_step_ids = torch.full(
            (omega_update_env_ids.shape[0],),
            current_omega_step,
            dtype=torch.long,
            device=self.device,
        )

        if getattr(self.mu_alg.actor_critic, "is_reservoir", False):
            mu_actions, reservoir_states = self.mu_alg.act(mu_update_env_ids, mu_step_ids, obs, critic_obs, self.mu_reservoir_states)
            self.mu_reservoir_states[mu_update_env_ids] = reservoir_states
        else:
            mu_actions = self.mu_alg.act(mu_update_env_ids, mu_step_ids, obs, critic_obs)

        if getattr(self.omega_alg.actor_critic, "is_reservoir", False):
            omega_actions, reservoir_states = self.omega_alg.act(omega_update_env_ids, omega_step_ids, obs, critic_obs, self.omega_reservoir_states)
            self.omega_reservoir_states[omega_update_env_ids] = reservoir_states
        else:
            omega_actions = self.omega_alg.act(omega_update_env_ids, omega_step_ids, obs, critic_obs)

        self.mu_cached[mu_update_env_ids] = mu_actions
        self.omega_cached[omega_update_env_ids] = omega_actions
        self.actions[:, :self.env.mu_shape] = self.mu_cached
        self.actions[:, self.env.mu_shape:] = self.omega_cached

        obs, privileged_obs, rewards, dones, infos = self.env.step(self.actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

        self.mu_alg.clear_start_transition()
        self.omega_alg.clear_start_transition()

        return obs, critic_obs, rewards, dones, infos


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        if self.env.cfg.control.hierarchical:
            critic_obs = privileged_obs
        else:
            critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.mu_alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.omega_alg.actor_critic.train() # switch to train mode (for dropout for example)
        mu_ep_infos = []
        omega_ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        #Accumulate Rewards
        mu_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)
        omega_acc_rewards = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        mu_it = self.current_learning_iteration
        omega_it = self.current_learning_iteration

        # Rollout
        mu_collection_start = time.time()
        omega_collection_start = time.time()
        while True:

            obs, critic_obs, rewards, dones, infos = self.training_rollout(obs, critic_obs, self.current_mu_step, self.current_omega_step)

            mu_acc_rewards += rewards
            omega_acc_rewards += rewards

            self.current_mu_step += 1
            self.current_omega_step += 1

            mu_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.env.cfg.control.mu_cycle,
            )
            omega_update_env_ids = self.get_update_env_ids(
                self.env.episode_length_buf,
                self.env.cfg.control.omega_cycle,
            )

            if self.omega_reservoir_states is not None:
                self.omega_reservoir_states[dones] = 0.
            if self.mu_reservoir_states is not None:
                self.mu_reservoir_states[dones] = 0.

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
                    mu_ep_infos.append(infos['episode'])
                    omega_ep_infos.append(infos['episode'])
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0


            mu_acc_rewards[mu_update_env_ids] = 0.
            omega_acc_rewards[omega_update_env_ids] = 0.


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
                    self.save_model(self.log_dir, mu_it, omega_it)

                mu_it += 1
                mu_collection_start = time.time()
                mu_ep_infos.clear()

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
                    self.save_model(self.log_dir, mu_it, omega_it)

                omega_it += 1
                omega_collection_start = time.time()
                omega_ep_infos.clear()

            if mu_it >= tot_iter and omega_it >= tot_iter:
                break

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir,
                               f"model_mu-{mu_it}_omega-{omega_it}.pt"))
        self.final_log()

    def save_model(self, log_dir, mu_it, omega_it):
        log_dir = Path(log_dir)

        pattern = re.compile(r"^model_(\d+)_mu-\d+_omega-\d+\.pt$")
        existing_indices = []
        for file in log_dir.glob("*.pt"):
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        model_idx = max(existing_indices, default=-1) + 1

        save_path = log_dir / (
            f"model_{model_idx}_mu-{mu_it}_omega-{omega_it}.pt"
        )

        self.save(str(save_path))


    def final_log(self, width=40, pad=35):
        log_string = (f"""{'#' * width}\n"""
                        f"""Training done""")
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

    def save(self, path, infos=None):
        torch.save({
            'mu_model_state_dict': self.mu_alg.actor_critic.state_dict(),
            'omega_model_state_dict': self.omega_alg.actor_critic.state_dict(),
            'mu_optimizer_state_dict': self.mu_alg.optimizer.state_dict(),
            'omega_optimizer_state_dict': self.omega_alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
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
            if 'mu_optimizer_state_dict' in loaded_dict:
                self.mu_alg.optimizer.load_state_dict(
                    loaded_dict['mu_optimizer_state_dict']
                )
            if 'omega_optimizer_state_dict' in loaded_dict:
                self.omega_alg.optimizer.load_state_dict(
                    loaded_dict['omega_optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.mu_alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        self.omega_alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.mu_alg.actor_critic.to(device)
            self.omega_alg.actor_critic.to(device)
        return self.mu_alg.actor_critic.act_inference, self.omega_alg.actor_critic.act_inference
