from rsl_rl.storage import RolloutStorage

import torch


class RolloutStorage_event(RolloutStorage):
    class StartTransition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_ids = None
            self.step_ids = None
            self.reservoir_states = None

        def clear(self):
            self.__init__()

    class EndTransition:
        def __init__(self):
            self.rewards = None
            self.dones = None
            self.critic_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 privileged_obs_shape,
                 actions_shape,
                 device='cpu'):

        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.capacity = num_transitions_per_env * num_envs

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        critic_obs_shape = privileged_obs_shape
        if privileged_obs_shape[0] is None:
            critic_obs_shape = obs_shape

        # Core
        self.observations = torch.zeros(
            self.capacity,
            *obs_shape,
            device=self.device,
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                self.capacity,
                *privileged_obs_shape,
                device=self.device,
            )
        else:
            self.privileged_observations = None

        self.bootstrap_observations = torch.zeros(
            num_envs,
            *critic_obs_shape,
            device=self.device,
        )
        self.bootstrap_valid = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.actions = torch.zeros(
            self.capacity,
            *actions_shape,
            device=self.device,
        )
        self.rewards = torch.zeros(self.capacity, 1, device=self.device)
        self.dones = torch.zeros(
            self.capacity,
            1,
            dtype=torch.bool,
            device=self.device,
        )

        # For PPO
        self.actions_log_prob = torch.zeros(
            self.capacity,
            1,
            device=self.device,
        )
        self.values = torch.zeros(self.capacity, 1, device=self.device)
        self.returns = torch.zeros(self.capacity, 1, device=self.device)
        self.advantages = torch.zeros(self.capacity, 1, device=self.device)
        self.mu = torch.zeros(
            self.capacity,
            *actions_shape,
            device=self.device,
        )
        self.sigma = torch.zeros(
            self.capacity,
            *actions_shape,
            device=self.device,
        )

        self.env_ids = torch.zeros(
            self.capacity,
            dtype=torch.long,
            device=self.device,
        )
        self.step_ids = torch.zeros(
            self.capacity,
            dtype=torch.long,
            device=self.device,
        )

        self.storage_done = torch.zeros(
            self.capacity,
            dtype=torch.bool,
            device=self.device,
        )
        self.env_ptrs = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.pending = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.reservoir_states = None

        self.counter = 0
        self.completed_count = 0
        self.full = False

    @property
    def ready(self):
        return self.full and self.completed_count == self.capacity

    def clear(self):
        self.counter = 0
        self.completed_count = 0
        self.storage_done.zero_()
        self.env_ptrs.fill_(-1)
        self.pending.zero_()
        self.bootstrap_valid.zero_()
        self.full = False

    def add_start_transitions(self, transition):
        available = self.capacity - self.counter
        batch_size = min(transition.observations.shape[0], available)
        if batch_size == 0:
            return transition.env_ids[:0].to(dtype=torch.long)

        idx = slice(self.counter, self.counter + batch_size)
        env_ids = transition.env_ids[:batch_size].to(dtype=torch.long)

        self.observations[idx].copy_(
            transition.observations[:batch_size]
        )
        if self.privileged_observations is not None:
            self.privileged_observations[idx].copy_(
                transition.critic_observations[:batch_size]
            )
        self.actions[idx].copy_(transition.actions[:batch_size])
        self.values[idx].copy_(transition.values[:batch_size])
        self.actions_log_prob[idx].copy_(
            transition.actions_log_prob[:batch_size].view(-1, 1)
        )
        self.mu[idx].copy_(transition.action_mean[:batch_size])
        self.sigma[idx].copy_(transition.action_sigma[:batch_size])
        self.env_ids[idx].copy_(env_ids)
        self.step_ids[idx].copy_(
            transition.step_ids[:batch_size].to(dtype=torch.long)
        )

        idx_tensor = torch.arange(
            idx.start,
            idx.stop,
            device=self.device,
            dtype=torch.long,
        )
        self.env_ptrs[env_ids] = idx_tensor
        self.pending[env_ids] = True

        if transition.reservoir_states is not None:
            self.save_reservoir_states(
                idx,
                transition.reservoir_states[:batch_size],
            )


        self.counter += batch_size
        self.full = self.counter == self.capacity
        return env_ids

    def save_reservoir_states(self, idx, reservoir_states):
        if reservoir_states is None:
            return

        # initialize if needed 
        if self.reservoir_states is None:
            self.reservoir_states = torch.zeros(self.observations.shape[0], reservoir_states.shape[-1], device=self.device)
        # copy the states
        self.reservoir_states[idx].copy_(reservoir_states)

    def get_pending_transitions(self, env_ids):
        env_ids = env_ids.to(dtype=torch.long)
        pending_mask = self.pending[env_ids]
        tracked_env_ids = env_ids[pending_mask]
        transition_ids = self.env_ptrs[tracked_env_ids]
        return pending_mask, tracked_env_ids, transition_ids

    def add_end_transitions(self, env_ids, transition):
        pending_mask, tracked_env_ids, idx = self.get_pending_transitions(
            env_ids
        )
        if tracked_env_ids.numel() == 0:
            return tracked_env_ids

        self.rewards[idx] = transition.rewards[pending_mask].view(-1, 1)
        self.dones[idx] = transition.dones[pending_mask].view(-1, 1)
        self.bootstrap_observations[tracked_env_ids] = (
            transition.critic_observations[pending_mask]
        )
        self.bootstrap_valid[tracked_env_ids] = True
        self.storage_done[idx] = True

        self.pending[tracked_env_ids] = False
        self.env_ptrs[tracked_env_ids] = -1
        self.completed_count += tracked_env_ids.numel()
        return tracked_env_ids

    def compute_returns(self, last_values, gamma, lam):
        batch_size = self.counter
        if batch_size == 0:
            return
        if self.completed_count != batch_size:
            raise RuntimeError(
                "Cannot compute returns with unfinished transitions"
            )

        env_ids = self.env_ids[:batch_size]
        order = torch.argsort(env_ids, stable=True)
        sorted_env_ids = env_ids[order]
        counts = torch.bincount(
            sorted_env_ids,
            minlength=self.num_envs,
        )
        offsets = counts.cumsum(0) - counts
        max_events = int(counts.max().item())

        sorted_values = self.values[:batch_size][order]
        sorted_rewards = self.rewards[:batch_size][order]
        sorted_dones = self.dones[:batch_size][order].float()

        next_values = last_values[sorted_env_ids].clone()
        same_env = sorted_env_ids[:-1] == sorted_env_ids[1:]
        next_values[:-1][same_env] = sorted_values[1:][same_env]

        deltas = (
            sorted_rewards
            + (1.0 - sorted_dones) * gamma * next_values
            - sorted_values
        )
        continuation = (1.0 - sorted_dones) * gamma * lam

        sorted_advantages = torch.zeros_like(deltas)
        running_advantage = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
        )

        for event_rank in range(max_events - 1, -1, -1):
            active_env_ids = torch.nonzero(
                counts > event_rank,
                as_tuple=False,
            ).squeeze(-1)
            sorted_indices = offsets[active_env_ids] + event_rank
            current_advantage = (
                deltas[sorted_indices]
                + continuation[sorted_indices]
                * running_advantage[active_env_ids]
            )
            running_advantage[active_env_ids] = current_advantage
            sorted_advantages[sorted_indices] = current_advantage

        advantages = torch.empty_like(sorted_advantages)
        advantages[order] = sorted_advantages
        self.returns[:batch_size].copy_(
            advantages + self.values[:batch_size]
        )

        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
        self.advantages[:batch_size].copy_(normalized_advantages)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.counter
        mini_batch_size = batch_size // num_mini_batches
        usable_batch_size = num_mini_batches * mini_batch_size

        observations = self.observations[:batch_size]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[:batch_size]
        else:
            critic_observations = observations

        actions = self.actions[:batch_size]
        values = self.values[:batch_size]
        returns = self.returns[:batch_size]
        old_actions_log_prob = self.actions_log_prob[:batch_size]
        advantages = self.advantages[:batch_size]
        old_mu = self.mu[:batch_size]
        old_sigma = self.sigma[:batch_size]

        for epoch in range(num_epochs):
            indices = torch.randperm(
                batch_size,
                requires_grad=False,
                device=self.device,
            )[:usable_batch_size]
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
                

    def reservoir_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.counter
        mini_batch_size = batch_size // num_mini_batches
        usable_batch_size = num_mini_batches * mini_batch_size

        observations = self.observations[:batch_size]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[:batch_size]
        else:
            critic_observations = observations

        actions = self.actions[:batch_size]
        values = self.values[:batch_size]
        returns = self.returns[:batch_size]
        old_actions_log_prob = self.actions_log_prob[:batch_size]
        advantages = self.advantages[:batch_size]
        old_mu = self.mu[:batch_size]
        old_sigma = self.sigma[:batch_size]
        reservoir_states = self.reservoir_states[:batch_size]

        for epoch in range(num_epochs):
            indices = torch.randperm(
                batch_size,
                requires_grad=False,
                device=self.device,
            )[:usable_batch_size]
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                reservoir_states_batch = reservoir_states[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, reservoir_states_batch


class RolloutStorage_event_discrete(RolloutStorage_event):
    class StartTransition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.logits = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_ids = None
            self.step_ids = None
            self.reservoir_states = None

        def clear(self):
            self.__init__()

    class EndTransition:
        def __init__(self):
            self.rewards = None
            self.dones = None
            self.critic_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 privileged_obs_shape,
                 actions_shape,
                 logits_shape,
                 device='cpu'):

        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)

        self.logits = torch.zeros(
            self.capacity,
            *logits_shape,
            device=self.device,
        )


    def add_start_transitions(self, transition):
        available = self.capacity - self.counter
        batch_size = min(transition.observations.shape[0], available)
        if batch_size == 0:
            return transition.env_ids[:0].to(dtype=torch.long)

        idx = slice(self.counter, self.counter + batch_size)
        env_ids = transition.env_ids[:batch_size].to(dtype=torch.long)

        self.observations[idx].copy_(
            transition.observations[:batch_size]
        )
        if self.privileged_observations is not None:
            self.privileged_observations[idx].copy_(
                transition.critic_observations[:batch_size]
            )
        self.actions[idx].copy_(transition.actions[:batch_size])
        self.logits[idx].copy_(transition.logits[:batch_size])
        self.values[idx].copy_(transition.values[:batch_size])
        self.actions_log_prob[idx].copy_(
            transition.actions_log_prob[:batch_size].view(-1, 1)
        )
        self.mu[idx].copy_(transition.action_mean[:batch_size])
        self.sigma[idx].copy_(transition.action_sigma[:batch_size])
        self.env_ids[idx].copy_(env_ids)
        self.step_ids[idx].copy_(
            transition.step_ids[:batch_size].to(dtype=torch.long)
        )

        idx_tensor = torch.arange(
            idx.start,
            idx.stop,
            device=self.device,
            dtype=torch.long,
        )
        self.env_ptrs[env_ids] = idx_tensor
        self.pending[env_ids] = True

        if transition.reservoir_states is not None:
            self.save_reservoir_states(
                idx,
                transition.reservoir_states[:batch_size],
            )


        self.counter += batch_size
        self.full = self.counter == self.capacity
        return env_ids


    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.counter
        mini_batch_size = batch_size // num_mini_batches
        usable_batch_size = num_mini_batches * mini_batch_size

        observations = self.observations[:batch_size]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[:batch_size]
        else:
            critic_observations = observations

        actions = self.actions[:batch_size]
        logits = self.logits[:batch_size]
        values = self.values[:batch_size]
        returns = self.returns[:batch_size]
        old_actions_log_prob = self.actions_log_prob[:batch_size]
        advantages = self.advantages[:batch_size]
        old_mu = self.mu[:batch_size]
        old_sigma = self.sigma[:batch_size]

        for epoch in range(num_epochs):
            indices = torch.randperm(
                batch_size,
                requires_grad=False,
                device=self.device,
            )[:usable_batch_size]
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                logits_batch = logits[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, logits_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
                

    def reservoir_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.counter
        mini_batch_size = batch_size // num_mini_batches
        usable_batch_size = num_mini_batches * mini_batch_size

        observations = self.observations[:batch_size]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[:batch_size]
        else:
            critic_observations = observations

        actions = self.actions[:batch_size]
        logits = self.logits[:batch_size]
        values = self.values[:batch_size]
        returns = self.returns[:batch_size]
        old_actions_log_prob = self.actions_log_prob[:batch_size]
        advantages = self.advantages[:batch_size]
        old_mu = self.mu[:batch_size]
        old_sigma = self.sigma[:batch_size]
        reservoir_states = self.reservoir_states[:batch_size]

        for epoch in range(num_epochs):
            indices = torch.randperm(
                batch_size,
                requires_grad=False,
                device=self.device,
            )[:usable_batch_size]
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                logits_batch = logits[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                reservoir_states_batch = reservoir_states[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch,logits_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, reservoir_states_batch, None