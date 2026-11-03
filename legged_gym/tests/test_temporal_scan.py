"""Pose compensation for a slowly sampled, world-stationary terrain scan."""

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/temporal_scan.py'


def temporal_scan_cache(*args):
    assert PATH.exists(), 'temporal_scan.py must implement the scan cache'
    spec = importlib.util.spec_from_file_location('temporal_scan', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalScanCache(*args)


def quat(yaw):
    angle = torch.as_tensor(yaw) / 2
    return torch.stack((torch.zeros_like(angle), torch.zeros_like(angle),
                        angle.sin(), angle.cos()), -1)


def test_capture_projects_same_pose_and_preserves_world_heights():
    x = torch.tensor([-.1, 0., .1])
    y = torch.tensor([-.1, 0., .1])
    cache = temporal_scan_cache(x, y, 1, 'cpu')
    low = torch.arange(9.).reshape(1, 9)
    high = low + .2
    xy = torch.tensor([[2., 3.]])
    q = quat(torch.tensor([.4]))
    cache.capture(torch.tensor([0]), low, high, xy, q, torch.tensor([0]))
    actual_low, actual_high, known = cache.project(xy, q)
    torch.testing.assert_close(actual_low.flatten(1), low)
    torch.testing.assert_close(actual_high.flatten(1), high)
    assert known.all()
    assert cache.frame_step.tolist() == [0]
    assert cache.frame_count.tolist() == [1]


def test_translation_reprojects_linear_field_and_marks_unseen_border():
    x = torch.tensor([0., .1, .2])
    y = torch.tensor([-.1, 0., .1])
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    terrain = 2 * grid_x + 3 * grid_y
    cache = temporal_scan_cache(x, y, 1, 'cpu')
    cache.capture(torch.tensor([0]), terrain.reshape(1, -1),
                  (terrain + .1).reshape(1, -1),
                  torch.zeros(1, 2), quat(torch.zeros(1)), 0)
    actual_low, actual_high, known = cache.project(
        torch.tensor([[.05, 0.]]), quat(torch.zeros(1)))
    expected_low = terrain
    expected_high = 2 * torch.tensor([.1, .2, .2])[:, None] + 3 * grid_y + .1
    torch.testing.assert_close(actual_low[0], expected_low, atol=1e-6, rtol=0)
    torch.testing.assert_close(actual_high[0], expected_high, atol=1e-6, rtol=0)
    assert known[0, :2].all()
    assert not known[0, 2].any()


def test_half_cell_motion_keeps_narrow_recess_and_raised_edge_visible():
    x = torch.tensor([0., .1, .2])
    y = torch.tensor([-.1, 0., .1])
    cache = temporal_scan_cache(x, y, 1, 'cpu')
    low = torch.zeros(1, 3, 3)
    high = torch.zeros_like(low)
    low[0, 1, 1] = -.6
    high[0, 1, 1] = .2
    cache.capture(torch.tensor([0]), low.flatten(1), high.flatten(1),
                  torch.zeros(1, 2), quat(torch.zeros(1)), 0)
    shifted_low, shifted_high, _ = cache.project(
        torch.tensor([[.05, 0.]]), quat(torch.zeros(1)))
    assert shifted_low[0, 0, 1] == -.6
    assert shifted_high[0, 0, 1] == .2


def test_yaw_reprojects_into_capture_frame():
    x = torch.tensor([-1., 0., 1.])
    y = torch.tensor([-1., 0., 1.])
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    cache = temporal_scan_cache(x, y, 1, 'cpu')
    low = grid_x + 2 * grid_y
    cache.capture(torch.tensor([0]), low.reshape(1, -1), low.reshape(1, -1),
                  torch.zeros(1, 2), quat(torch.zeros(1)), 0)
    actual, _, known = cache.project(torch.zeros(1, 2), quat(torch.tensor([torch.pi / 2])))
    torch.testing.assert_close(actual[0], -grid_y + 2 * grid_x, atol=1e-6, rtol=0)
    assert known.all()


def test_capture_updates_only_selected_environment():
    axis = torch.tensor([-1., 0., 1.])
    cache = temporal_scan_cache(axis, axis, 2, 'cpu')
    ids = torch.tensor([0, 1])
    cache.capture(ids, torch.stack((torch.zeros(9), torch.ones(9))),
                  torch.stack((torch.zeros(9), torch.ones(9))),
                  torch.zeros(2, 2), quat(torch.zeros(2)), torch.zeros(2, dtype=torch.long))
    cache.capture(torch.tensor([0]), torch.full((1, 9), 4.), torch.full((1, 9), 5.),
                  torch.zeros(1, 2), quat(torch.zeros(1)), 7)
    actual_low, actual_high, _ = cache.project(torch.zeros(2, 2), quat(torch.zeros(2)))
    assert actual_low[:, 1, 1].tolist() == [4., 1.]
    assert actual_high[:, 1, 1].tolist() == [5., 1.]
    assert cache.frame_step.tolist() == [7, 0]
    assert cache.frame_count.tolist() == [2, 1]


def test_project_requires_a_frame():
    axis = torch.tensor([-1., 0., 1.])
    cache = temporal_scan_cache(axis, axis, 1, 'cpu')
    with pytest.raises(RuntimeError, match='capture'):
        cache.project(torch.zeros(1, 2), quat(torch.zeros(1)))
