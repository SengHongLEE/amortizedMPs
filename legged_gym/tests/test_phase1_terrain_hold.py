from types import SimpleNamespace
import numpy as np

from rsl_rl.modules.terrain_hold_experiment import prefix_digest
import torch


def test_prefix_digest_includes_contact_memory():
    env = SimpleNamespace(
        root_states=torch.zeros(1, 13), dof_state=torch.zeros(1, 12, 2),
        rigid_body_state=torch.zeros(1, 13), contact_forces=torch.zeros(1, 3),
        commands=torch.zeros(1, 4), episode_length_buf=torch.zeros(1, dtype=torch.long),
        terrain_types=torch.zeros(1, dtype=torch.long),
        terrain_levels=torch.zeros(1, dtype=torch.long),
        _stance_contact=torch.zeros(1, 4, dtype=torch.bool),
        terrain=SimpleNamespace(height_field_raw=np.zeros((4, 4), dtype=np.int16)),
        ID=SimpleNamespace(phi=torch.zeros(1, 4)),
    )
    runner = SimpleNamespace(
        actions=torch.zeros(1, 19), primitive_selections=torch.zeros(1, dtype=torch.long),
        last_primitive_selections=torch.zeros(1, dtype=torch.long),
        modulation_params=torch.zeros(1, 18),
        last_modulation_params=torch.zeros(1, 18),
        primitive_selection_reservoir_states=torch.zeros(1, 32),
        modulation_param_reservoir_states=None,
    )
    first = prefix_digest(env, runner, torch.zeros(1, 5, 262))
    env._stance_contact[0, 0] = True
    assert prefix_digest(env, runner, torch.zeros(1, 5, 262)) != first
    env._stance_contact[0, 0] = False
    env.terrain.height_field_raw[0, 0] = 1
    assert prefix_digest(env, runner, torch.zeros(1, 5, 262)) != first
