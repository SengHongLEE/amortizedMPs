"""Small, simulator-independent pieces for Phase 1 hold experiments."""

import hashlib
import math

import torch


def prefix_digest(env, runner, observation):
    """Hash physical, controller, observation, action, reservoir and RNG state."""
    digest = hashlib.sha256()
    values = {
        'observation': observation,
        'root': env.root_states, 'dof': env.dof_state,
        'bodies': env.rigid_body_state, 'contacts': env.contact_forces,
        'commands': env.commands, 'episode_step': env.episode_length_buf,
        'terrain_type': env.terrain_types, 'terrain_level': env.terrain_levels,
        'runner_actions': runner.actions,
        'runner_gait': runner.primitive_selections,
        'runner_last_gait': runner.last_primitive_selections,
        'runner_modulation': runner.modulation_params,
        'runner_last_modulation': runner.last_modulation_params,
        'torch_rng': torch.get_rng_state(),
    }
    for name in ('actions', 'last_actions', 'last_dof_vel', 'last_root_vel',
                 'base_lin_vel', 'base_ang_vel', 'projected_gravity',
                 'measured_heights', '_stance_contact', '_foot_air_time',
                 '_previous_foot_velocity', '_touchdown_window_left',
                 '_touchdown_peak_cost', '_contact_initialized',
                 '_slip_cost_integral', '_touchdown_cost'):
        state = getattr(env, name, None)
        if isinstance(state, torch.Tensor):
            values[f'env.{name}'] = state
    if torch.cuda.is_available():
        for i, state in enumerate(torch.cuda.get_rng_state_all()):
            values[f'cuda_rng_{i}'] = state
    for name in ('primitive_selection_reservoir_states', 'modulation_param_reservoir_states'):
        state = getattr(runner, name)
        if state is not None:
            values[name] = state
    for name, state in vars(env.ID).items():
        if isinstance(state, torch.Tensor):
            values[f'ID.{name}'] = state
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    terrain = getattr(env, 'terrain', None)
    height_field = getattr(terrain, 'height_field_raw', None)
    if height_field is not None:
        digest.update(b'terrain.height_field_raw')
        digest.update(str(height_field.shape).encode())
        digest.update(str(height_field.dtype).encode())
        digest.update(height_field.tobytes())
    return digest.hexdigest()


class HoldClock:
    """Independent per-environment policy deadlines in environment control steps."""

    def __init__(self, num_envs, gait_min, modulation_min, device):
        if gait_min < 15 or modulation_min < 3:
            raise ValueError('policy calls cannot exceed the archived 15/3-step frequencies')
        self.minimum = {'gait': gait_min, 'modulation': modulation_min}
        self.next_step = {name: torch.full((num_envs,), -1, dtype=torch.long, device=device)
                          for name in self.minimum}
        self.last_step = {name: torch.full((num_envs,), -1, dtype=torch.long, device=device)
                          for name in self.minimum}
        self.call_count = {name: torch.zeros(num_envs, dtype=torch.long, device=device)
                           for name in self.minimum}

    def due(self, step):
        return tuple(torch.nonzero(self.next_step[name] <= step).flatten()
                     for name in ('gait', 'modulation'))

    def mark(self, name, env_ids, step, durations):
        if name not in self.minimum:
            raise ValueError('unknown policy clock')
        durations = torch.as_tensor(durations, device=self.next_step[name].device,
                                    dtype=torch.long).expand(env_ids.numel())
        if (durations < self.minimum[name]).any():
            raise ValueError('hold deadline exceeds maximum policy frequency')
        previous = self.last_step[name][env_ids]
        if ((previous >= 0) & (step - previous < self.minimum[name])).any():
            raise ValueError('actual policy calls exceeded maximum frequency')
        self.last_step[name][env_ids] = step
        self.next_step[name][env_ids] = step + durations
        self.call_count[name][env_ids] += 1

    def reset(self, env_ids):
        for name in self.minimum:
            self.next_step[name][env_ids] = -1
            self.last_step[name][env_ids] = -1


class FixedTerrainClock:
    """Fixed Phase 1 SNN period; reset wakes a new episode immediately."""

    def __init__(self, num_envs, period, device):
        if type(period) is not int or period < 1:
            raise ValueError('fixed SNN period must be positive control steps')
        self.period = period
        self.next_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def due(self, step):
        return torch.nonzero(self.next_step <= step).flatten()

    def mark(self, env_ids, step):
        self.next_step[env_ids] = step + self.period

    def reset(self, env_ids):
        self.next_step[env_ids] = -1


def compare_quality(reference, held, allowances):
    """Apply explicit absolute allowances to paired observable quantities."""
    required = ('speed_error', 'tilt', 'slip', 'collision', 'progress')
    if any(key not in allowances or not math.isfinite(allowances[key])
           or allowances[key] < 0 for key in required):
        raise ValueError('all five finite nonnegative safety allowances are required')
    if held['failed'] or reference['failed']:
        return False
    return all((held[key] - reference[key]) <= allowances[key]
               for key in required[:-1]) and (
                   reference['progress'] - held['progress'] <= allowances['progress'])


def planar_speed_error(body_linear_velocity, command):
    """Planar linear velocity tracking error in metres per second."""
    return (body_linear_velocity[..., :2] - command[..., :2]).norm(dim=-1)


def conservative_outcomes(replicates):
    """A candidate passes only when every matched disturbance replica passes."""
    if not replicates or len({len(row) for row in replicates}) != 1:
        raise ValueError('replicas must be nonempty and have equal bin counts')
    results = []
    for column in zip(*replicates):
        if 'fail' in column:
            results.append('fail')
        elif 'unobserved' in column:
            results.append('unobserved')
        elif all(value == 'pass' for value in column):
            results.append('pass')
        else:
            raise ValueError('unexpected replica outcome')
    return tuple(results)


def candidate_result(reference, candidate, duration, limits):
    """Classify a paired hold window without attributing baseline failure to hold."""
    if reference['prefix'] is None or candidate['prefix'] != reference['prefix']:
        raise RuntimeError('paired prefix mismatch; no counterfactual label may be written')
    count = min(duration, len(reference['quality']), len(candidate['quality']))
    for index in range(count):
        ref, held = reference['quality'][index], candidate['quality'][index]
        if ref['failed'] or ref['timeout'] or held['timeout']:
            return 'unobserved'
        if held['failed'] or not compare_quality(ref, held, limits):
            return 'fail'
        if index + 1 < duration and (ref.get('completed', False)
                                     or held.get('completed', False)):
            return 'unobserved'
    return 'pass' if count == duration else 'unobserved'


class HeldPolicyController:
    """Run archived actors with independent cached outputs and reservoir states."""

    def __init__(self, runner, gait_actor, modulation_actor):
        self.runner = runner
        self.env = runner.env
        self.gait_actor = gait_actor
        self.modulation_actor = modulation_actor
        self.clock = HoldClock(self.env.num_envs,
                               self.env.cfg.control.primitive_selection_update_cycle,
                               self.env.cfg.control.modulation_param_update_cycle,
                               runner.device)

    @torch.no_grad()
    def step(self, observation, global_step, gait_interval, modulation_interval):
        runner = self.runner
        gait_ids, modulation_ids = self.clock.due(global_step)
        if gait_ids.numel():
            if runner.primitive_selection_reservoir_states is not None:
                value, state = self.gait_actor(
                    observation[gait_ids], runner.primitive_selection_reservoir_states[gait_ids])
                runner.primitive_selection_reservoir_states[gait_ids] = state
            else:
                value = self.gait_actor(observation[gait_ids, -1])
            runner.last_primitive_selections[gait_ids] = runner.primitive_selections[gait_ids]
            runner.primitive_selections[gait_ids] = value
            runner.actions[gait_ids, 0] = value.float()
            self.clock.mark('gait', gait_ids, global_step, gait_interval)
        if modulation_ids.numel():
            if runner.modulation_param_reservoir_states is not None:
                value, state = self.modulation_actor(
                    observation[modulation_ids, -1],
                    runner.modulation_param_reservoir_states[modulation_ids])
                runner.modulation_param_reservoir_states[modulation_ids] = state
            else:
                value = self.modulation_actor(observation[modulation_ids, -1])
            runner.last_modulation_params[modulation_ids] = runner.modulation_params[modulation_ids]
            runner.modulation_params[modulation_ids] = value
            runner.actions[modulation_ids, 1:] = value
            self.clock.mark('modulation', modulation_ids, global_step, modulation_interval)
        next_obs, privileged, rewards, dones, infos = self.env.step(runner.actions)
        reset_ids = torch.nonzero(dones).flatten()
        if reset_ids.numel():
            self.clock.reset(reset_ids)
            if runner.primitive_selection_reservoir_states is not None:
                runner.primitive_selection_reservoir_states[reset_ids] = 0
            if runner.modulation_param_reservoir_states is not None:
                runner.modulation_param_reservoir_states[reset_ids] = 0
        return next_obs, privileged, rewards, dones, infos
