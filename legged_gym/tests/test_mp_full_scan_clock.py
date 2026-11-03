"""The visual clock is independent of the trained actor clocks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_interoceptive_rewards import load_class


PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/temporal_scan.py'
spec = importlib.util.spec_from_file_location('temporal_scan_clock', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scan_due_after_five_completed_steps_per_environment():
    due = module.scan_due
    assert [bool(due(torch.tensor([step]), 5)[0]) for step in range(10)] == [
        False, False, False, False, True,
        False, False, False, False, True,
    ]
    assert due(torch.tensor([0, 4, 9]), 5).tolist() == [False, True, True]
    assert due(torch.tensor([0, 1, 2]), 1).tolist() == [True, True, True]


def test_scan_period_rejects_invalid_values():
    with pytest.raises(ValueError, match='positive integer'):
        module.scan_due(torch.tensor([0]), 0)
    with pytest.raises(ValueError, match='positive integer'):
        module.scan_due(torch.tensor([0]), 2.5)


def test_environment_samples_only_due_ids_and_refreshes_reset_ids():
    Env = load_class('mp_full.py', 'MotorPrimitives', {'_refresh_height_scan'})
    env = Env()
    env._refresh_height_scan.__globals__['scan_due'] = module.scan_due
    env.cfg = SimpleNamespace(terrain=SimpleNamespace(scan_period_steps=5))
    env.num_envs = 2
    env.device = 'cpu'
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.root_states = torch.zeros(2, 13)
    env.root_states[:, 6] = 1.
    env.base_quat = env.root_states[:, 3:7]
    env.measured_heights = torch.zeros(2, 9)
    env._dense_lower_heights = torch.zeros(2, 9)
    env._dense_upper_heights = torch.zeros(2, 9)
    env.scan_age_steps = torch.zeros(2, dtype=torch.long)
    env.scan_unknown_fraction = torch.zeros(2)
    axis = torch.tensor([-.1, 0., .1])
    env.scan_cache = module.TemporalScanCache(axis, axis, 2, 'cpu')
    called = []

    def sample(ids):
        called.append(ids.tolist())
        values = ids[:, None].float().expand(-1, 9) + len(called)
        env._dense_lower_heights[ids] = values
        env._dense_upper_heights[ids] = values + .1
        return values

    env._get_heights = sample
    env._refresh_height_scan(torch.tensor([0, 1]))
    assert called == [[0, 1]]
    assert env.scan_known_mask.shape == (2, 9)
    assert env.scan_known_mask.all()
    assert env.scan_cache.frame_step.tolist() == [0, 0]
    env.episode_length_buf[:] = torch.tensor([0, 4])
    env._refresh_height_scan()
    assert called == [[0, 1], [1]]
    assert env.scan_cache.frame_step.tolist() == [0, 5]
    assert env.scan_cache.frame_count.tolist() == [1, 2]
    assert env.scan_age_steps.tolist() == [1, 0]
    env._refresh_height_scan(torch.tensor([0]))
    assert called[-1] == [0]
    assert env.scan_cache.frame_step.tolist() == [0, 5]


def test_scan_period_defaults_to_every_step():
    Config = load_class('mp_full_config.py', 'MotorPrimitivesCfg')
    assert Config.terrain.scan_period_steps == 1
