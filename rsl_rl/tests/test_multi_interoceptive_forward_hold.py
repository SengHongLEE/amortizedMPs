"""Inference filtering must hold actions and reservoir state without touching training."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


def make_runner():
    runner = MultiInteroceptivePolicyRunner.__new__(MultiInteroceptivePolicyRunner)
    runner.device = 'cpu'
    runner.actions = torch.ones(2, 17)
    runner.actions[:, 0] = 2
    runner.last_actions = runner.actions.clone()
    runner.primitive_selections = torch.full((2,), 2, dtype=torch.long)
    runner.last_primitive_selections = runner.primitive_selections.clone()
    runner.modulation_params = runner.actions[:, 1:].clone()
    runner.last_modulation_params = runner.modulation_params.clone()
    runner.primitive_selection_update_cycle = torch.full((2,), 15)
    runner.modulation_param_update_cycle = torch.full((2,), 3)
    runner.primitive_selection_reservoir_states = torch.zeros(2, 2)
    runner.modulation_param_reservoir_states = torch.zeros(2, 2)
    obs = torch.zeros(2, 5, 12)
    scan = SimpleNamespace(x=torch.tensor([.1, .2]), y=torch.tensor([-.1, 0., .1]))
    runner.env = SimpleNamespace(
        num_envs=2, episode_length_buf=torch.zeros(2, dtype=torch.long),
        scan_cache=scan, scan_known_mask=torch.ones(2, 6, dtype=torch.bool),
        _dense_lower_heights=torch.zeros(2, 6), _dense_upper_heights=torch.zeros(2, 6),
        scan_age_steps=torch.zeros(2, dtype=torch.long),
        cfg=SimpleNamespace(
            terrain=SimpleNamespace(scan_period_steps=5),
            control=SimpleNamespace(primitive_selection_update_cycle=15,
                                    modulation_param_update_cycle=3)), dt=.02,
        commands=torch.zeros(2, 4), base_lin_vel=torch.zeros(2, 3),
        base_ang_vel=torch.zeros(2, 3), projected_gravity=torch.zeros(2, 3),
        primitive_diff=torch.zeros(2), param_actions_diff=torch.zeros(2),
        step=lambda actions: (obs, None, [torch.zeros(2)] * 3,
                              torch.zeros(2, dtype=torch.bool), {}),
    )
    return runner, obs


class FirstOnlyGate:
    def __init__(self):
        self.calls = []
        self.observations = []

    def observe(self, episode_steps, ang_vel):
        self.observations.append((episode_steps.clone(), ang_vel.clone()))

    def select(self, actor, candidates, **kwargs):
        self.calls.append((actor, candidates.tolist()))
        return candidates[:1], {'hold': torch.arange(len(candidates)) > 0}


def test_inference_filters_each_actor_and_preserves_held_reservoir():
    runner, obs = make_runner()
    gate = FirstOnlyGate()
    runner.inference_forward_hold = gate
    calls = []

    def high(values, states):
        calls.append(('gait', len(values)))
        return torch.full((len(values),), 3, dtype=torch.long), states + 1

    def low(values, states):
        calls.append(('modulation', len(values)))
        return torch.full((len(values), 16), 4.), states + 2

    runner.inference_rollout(obs, high, low)
    assert calls == [('gait', 1), ('modulation', 1)]
    assert gate.calls == [('gait', [0, 1]), ('modulation', [0, 1])]
    assert len(gate.observations) == 1
    assert runner.actions[:, 0].tolist() == [3., 2.]
    assert runner.actions[:, 1].tolist() == [4., 1.]
    assert runner.primitive_selection_reservoir_states.tolist() == [[1., 1.], [0., 0.]]
    assert runner.modulation_param_reservoir_states.tolist() == [[2., 2.], [0., 0.]]
    assert runner.last_inference_decisions['gait']['forward'].tolist() == [0]
    assert runner.last_inference_decisions['modulation']['forward'].tolist() == [0]


def test_training_keeps_fixed_candidates_with_inference_gate_attached():
    runner, obs = make_runner()
    gate = FirstOnlyGate()
    runner.inference_forward_hold = gate
    runner.primitive_selection_reservoir_states = None
    runner.modulation_param_reservoir_states = None
    trained = []

    def high_act(ids, *args, **kwargs):
        trained.append(('gait', ids.tolist()))
        return torch.ones(len(ids), 1, dtype=torch.long)

    def low_act(ids, *args, **kwargs):
        trained.append(('modulation', ids.tolist()))
        return torch.ones(len(ids), 16)

    runner.primitive_selection_alg = SimpleNamespace(
        actor_critic=SimpleNamespace(is_reservoir=False), act=high_act,
        clear_start_transition=lambda: None)
    runner.modulation_param_alg = SimpleNamespace(
        actor_critic=SimpleNamespace(is_reservoir=False), act=low_act,
        clear_start_transition=lambda: None)
    runner.training_rollout(obs, obs[:, -1], 0)
    assert trained == [('gait', [0, 1]), ('modulation', [0, 1])]
    assert gate.calls == []
    assert gate.observations == []
