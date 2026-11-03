"""Inference-only scheduling from already observed terrain and proprioception."""

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / 'legged_gym/utils/inference_forward_hold.py'


def load_rule_module():
    assert PATH.exists(), 'inference_forward_hold.py must provide the rule'
    spec = importlib.util.spec_from_file_location('inference_forward_hold', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(num_envs=2):
    x = torch.arange(1, 34).float() / 10
    y = torch.arange(-3, 4).float() / 10
    shape = (num_envs, len(x), len(y))
    state = dict(
        episode_steps=torch.zeros(num_envs, dtype=torch.long),
        commands=torch.zeros(num_envs, 4),
        lin_vel=torch.zeros(num_envs, 3),
        ang_vel=torch.zeros(num_envs, 3),
        gravity=torch.tensor([[0., 0., -1.]]).repeat(num_envs, 1),
        low=torch.zeros(shape), high=torch.zeros(shape),
        known=torch.ones(shape, dtype=torch.bool),
        scan_age_steps=torch.zeros(num_envs, dtype=torch.long),
        scan_period_steps=5, dt=.02, period_steps=3,
    )
    return x, y, state


def choose(rule, actor, state, ids=None):
    ids = torch.arange(state['episode_steps'].numel()) if ids is None else ids
    return rule.select(actor, ids, **state)


def test_independent_skip_caps_and_episode_first_forward():
    rule_module = load_rule_module()
    x, y, state = make_inputs()
    config = rule_module.ForwardHoldConfig(gait_max_skips=2, modulation_max_skips=3)
    rule = rule_module.InferenceForwardHold(config, x, y, 2, 'cpu')
    selected, _ = choose(rule, 'gait', state)
    assert selected.tolist() == [0, 1]
    state['episode_steps'][:] = 15
    state['period_steps'] = 15
    assert choose(rule, 'gait', state)[0].numel() == 0
    state['episode_steps'][:] = 30
    assert choose(rule, 'gait', state)[0].numel() == 0
    state['episode_steps'][:] = 45
    assert choose(rule, 'gait', state)[0].tolist() == [0, 1]
    assert rule.skipped['gait'].tolist() == [0, 0]
    state['period_steps'] = 3
    state['episode_steps'][:] = 0
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1]
    for step in (3, 6, 9):
        state['episode_steps'][:] = step
        assert choose(rule, 'modulation', state)[0].numel() == 0
    state['episode_steps'][:] = 12
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1]


def test_large_skip_caps_can_hold_on_visible_flat_ground():
    rule_module = load_rule_module()
    x, y, state = make_inputs(3)
    state['commands'][:, 0] = torch.tensor([.8, 1.5, 2.2])
    state['lin_vel'][:, 0] = state['commands'][:, 0]
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(gait_max_skips=15, modulation_max_skips=75),
        x, y, 3, 'cpu')
    choose(rule, 'gait', state)
    choose(rule, 'modulation', state)
    for candidate in range(1, 76):
        state['episode_steps'][:] = candidate * 3
        selected, reasons = choose(rule, 'modulation', state)
        assert selected.numel() == 0
        assert not reasons['terrain'].any()
    state['episode_steps'][:] = 76 * 3
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1, 2]
    state['period_steps'] = 15
    for candidate in range(1, 16):
        state['episode_steps'][:] = candidate * 15
        selected, reasons = choose(rule, 'gait', state)
        assert selected.numel() == 0
        assert not reasons['terrain'].any()
    state['episode_steps'][:] = 16 * 15
    assert choose(rule, 'gait', state)[0].tolist() == [0, 1, 2]


def test_large_skip_caps_keep_obstacle_lookahead_at_original_safe_horizon():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = 1.5
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(gait_max_skips=15, modulation_max_skips=75),
        x, y, 1, 'cpu')
    choose(rule, 'gait', state)
    choose(rule, 'modulation', state)
    state['low'][0, 10:] = .08  # obstacle at 1.1 m, ahead of the next gait candidate horizon
    state['high'][0] = state['low'][0]
    state['episode_steps'][:] = 15
    state['period_steps'] = 15
    assert choose(rule, 'gait', state)[0].tolist() == [0]
    state['low'].zero_()
    state['low'][0, 6:] = .08  # obstacle at 0.7 m, ahead of the next modulation horizon
    state['high'][0] = state['low'][0]
    state['episode_steps'][:] = 3
    state['period_steps'] = 3
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_proprio_command_and_known_obstacle_force_forward():
    rule_module = load_rule_module()
    x, y, state = make_inputs()
    state['commands'][:, 0] = .8
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(horizontal_rate_limit=1.,
                                      modulation_horizontal_rate_limit=1.,
                                      extero_edge_trigger=False), x, y, 2, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['gravity'][0, 0] = .3
    state['commands'][1, 0] = 1.
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1]
    state['episode_steps'][:] = 6
    state['gravity'][0, 0] = 0.
    state['high'][0, 2, 3] = .2
    state['low'][1, 2, 3] = -.2
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1]
    state['episode_steps'][:] = 9
    state['low'].zero_(); state['high'].zero_()
    state['ang_vel'][0, 0] = 1.2
    state['ang_vel'][1, 2] = 1.8
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_default_thresholds_and_yaw_rotation_does_not_force_forward():
    rule_module = load_rule_module()
    config = rule_module.ForwardHoldConfig()
    assert config.horizontal_rate_limit == 2.
    assert config.modulation_horizontal_rate_limit == 4.
    assert config.extero_edge_trigger is True
    assert not hasattr(config, 'yaw_rate_limit')
    x, y, state = make_inputs(1)
    rule = rule_module.InferenceForwardHold(config, x, y, 1, 'cpu')
    choose(rule, 'gait', state)
    choose(rule, 'modulation', state)
    state['ang_vel'][0] = torch.tensor([1.9, 0., 6.])
    state['episode_steps'][:] = 3
    assert choose(rule, 'modulation', state)[0].numel() == 0
    state['episode_steps'][:] = 15
    state['period_steps'] = 15
    assert choose(rule, 'gait', state)[0].numel() == 0
    state['ang_vel'][0, 0] = 2.1
    state['episode_steps'][:] = 30
    assert choose(rule, 'gait', state)[0].tolist() == [0]
    state['period_steps'] = 3
    state['ang_vel'][0, 0] = 4.1
    state['episode_steps'][:] = 6
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_sustained_linear_and_yaw_tracking_errors_trigger_after_warmup():
    rule_module = load_rule_module()
    x, y, state = make_inputs(2)
    state['commands'][0, 0] = state['lin_vel'][0, 0] = .8
    state['commands'][1, 2] = state['ang_vel'][1, 2] = 1.
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(modulation_max_skips=100), x, y, 2, 'cpu')
    rule.observe_tracking(state['episode_steps'], state['commands'],
                          state['lin_vel'], state['ang_vel'])
    choose(rule, 'modulation', state)
    state['lin_vel'][0, 0] = 0.
    state['ang_vel'][1, 2] = 0.
    for step in range(1, 28):
        state['episode_steps'][:] = step
        rule.observe_tracking(state['episode_steps'], state['commands'],
                              state['lin_vel'], state['ang_vel'])
        if step == 24:
            selected, reasons = choose(rule, 'modulation', state)
            assert selected.numel() == 0
            assert not reasons['tracking'].any()
    selected, reasons = choose(rule, 'modulation', state)
    assert selected.tolist() == [0, 1]
    assert reasons['tracking'].tolist() == [True, True]
    assert reasons['proprio'].tolist() == [False, False]


def test_tracking_filter_ignores_alternating_gait_motion_and_matching_yaw_target():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = state['lin_vel'][0, 0] = .8
    state['commands'][0, 2] = state['ang_vel'][0, 2] = 1.
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(modulation_max_skips=100), x, y, 1, 'cpu')
    rule.observe_tracking(state['episode_steps'], state['commands'],
                          state['lin_vel'], state['ang_vel'])
    choose(rule, 'modulation', state)
    for step in range(1, 46):
        state['episode_steps'][:] = step
        state['lin_vel'][0, 0] = .8 + (1. if step % 2 else -1.)
        rule.observe_tracking(state['episode_steps'], state['commands'],
                              state['lin_vel'], state['ang_vel'])
    selected, reasons = choose(rule, 'modulation', state)
    assert selected.numel() == 0
    assert not reasons['tracking'].any()


def test_modulation_rate_override_leaves_gait_proprio_trigger_intact():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    config = rule_module.ForwardHoldConfig(
        horizontal_rate_limit=1., modulation_horizontal_rate_limit=4.)
    rule = rule_module.InferenceForwardHold(config, x, y, 1, 'cpu')
    choose(rule, 'gait', state)
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['ang_vel'][0, 0] = 2.
    assert choose(rule, 'gait', state)[0].tolist() == [0]
    assert choose(rule, 'modulation', state)[0].numel() == 0


def test_modulation_rate_override_validates_finite_nonnegative_value():
    rule_module = load_rule_module()
    with pytest.raises(ValueError, match='modulation_horizontal_rate_limit'):
        rule_module.ForwardHoldConfig(modulation_horizontal_rate_limit=float('nan'))
    assert rule_module.ForwardHoldConfig(horizontal_rate_limit=2.5).horizontal_rate_limit == 2.5


def test_unknown_and_stale_frame_limit_verified_clear_distance():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    rule = rule_module.InferenceForwardHold(rule_module.ForwardHoldConfig(), x, y, 1, 'cpu')
    choose(rule, 'gait', state)
    state['episode_steps'][:] = 15
    state['period_steps'] = 15
    state['commands'][0, 0] = 2.
    choose(rule, 'gait', state)  # command change establishes the new reference
    state['episode_steps'][:] = 30
    state['known'][0, 14, 3] = False  # 1.5 m, inside the bounded gait safety horizon
    assert choose(rule, 'gait', state)[0].tolist() == [0]
    state['known'].fill_(True)
    state['episode_steps'][:] = 45
    state['scan_age_steps'][:] = 4
    state['known'][0, 17, 3] = False  # scan age expands the horizon from 1.75 to 1.91 m
    assert choose(rule, 'gait', state)[0].tolist() == [0]


def test_unseen_reference_and_nonfinite_data_force_forward():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    rule = rule_module.InferenceForwardHold(rule_module.ForwardHoldConfig(), x, y, 1, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['known'][0, 0, 3] = False
    assert choose(rule, 'modulation', state)[0].tolist() == [0]
    state['episode_steps'][:] = 6
    state['known'].fill_(True)
    state['low'][0, 0, 3] = float('nan')
    assert choose(rule, 'modulation', state)[0].tolist() == [0]
    state['episode_steps'][:] = 9
    state['low'].zero_()
    state['known'] = None
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_unknown_outer_edge_does_not_block_clear_central_corridor():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = .8
    rule = rule_module.InferenceForwardHold(rule_module.ForwardHoldConfig(), x, y, 1, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['known'][0, 0, 0] = False  # y=-.3 m, outside the .2 m travel half-width
    assert choose(rule, 'modulation', state)[0].numel() == 0
    state['episode_steps'][:] = 6
    state['known'][0, 0, 3] = False  # same x, now in the central path
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_edge_extero_ignores_smooth_ramp_and_rough_but_detects_obstacles():
    rule_module = load_rule_module()
    x, y, state = make_inputs(6)
    state['commands'][:, 0] = .8
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(extero_edge_trigger=True,
                                      terrain_edge_height_m=.035), x, y, 6, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['low'][0] = x[:, None] * .3  # smooth ramp exceeds the old 5 cm absolute threshold
    state['high'][0] = state['low'][0]
    state['low'][1, 1:] = .025  # small roughness without a sharp edge
    state['high'][1] = state['low'][1]
    for env, height in ((2, .08), (3, -.6), (4, .04), (5, -.1)):
        state['low'][env, 2:] = height
        state['high'][env] = state['low'][env]
    state['low'][4, 5:] = .08  # two shallow stair risers accumulate past 5 cm
    state['high'][4] = state['low'][4]
    state['lin_vel'][4, 0] = 1.5
    selected, reasons = choose(rule, 'modulation', state)
    assert selected.tolist() == [2, 3, 4, 5]
    assert reasons['terrain'].tolist() == [False, False, True, True, True, True]


def test_edge_extero_preserves_unknown_boundary_trigger():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = .8
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(extero_edge_trigger=True), x, y, 1, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['known'][0, 2, 3] = False
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_edge_extero_keeps_legacy_distance_when_scan_bounds_span_an_edge():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = .2
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(extero_edge_trigger=True), x, y, 1, 'cpu')
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    state['low'][0, 1] = -.08  # conservative bounds detect the edge at x=.2 m
    state['high'][0, 1] = .08
    state['low'][0, 2:] = .08
    state['high'][0, 2:] = .08
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_lookahead_beyond_scan_range_is_unknown():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    state['commands'][0, 0] = 7.  # even one-period lookahead exceeds the 3.3 m scan
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(gait_max_skips=10), x, y, 1, 'cpu')
    choose(rule, 'gait', state)
    state['episode_steps'][:] = 15
    state['period_steps'] = 15
    assert choose(rule, 'gait', state)[0].tolist() == [0]


def test_skip_limits_must_be_nonnegative():
    rule_module = load_rule_module()
    with pytest.raises(ValueError, match='max_skips'):
        rule_module.ForwardHoldConfig(gait_max_skips=-1)
    with pytest.raises(ValueError, match='finite'):
        rule_module.ForwardHoldConfig(tilt_limit=float('nan'))
    with pytest.raises(ValueError, match='positive'):
        rule_module.ForwardHoldConfig(terrain_edge_height_m=0.)


def test_policy_update_frequency_uses_actual_forwards_and_active_simulated_time():
    rule_module = load_rule_module()
    # 30 active environment steps at 20 ms give 0.6 simulated env-seconds.
    summary = rule_module.format_policy_update_frequency(
        gait_forwards=2, modulation_forwards=9, active_env_steps=30, dt=.02)
    assert 'gait: 3.33 Hz (2 forwards)' in summary
    assert 'modulation: 15.00 Hz (9 forwards)' in summary
    assert 'per env, simulated time' in summary


def test_policy_update_frequency_handles_interruption_before_first_step():
    rule_module = load_rule_module()
    summary = rule_module.format_policy_update_frequency(0, 0, 0, .02)
    assert 'gait: 0.00 Hz' in summary
    assert 'modulation: 0.00 Hz' in summary


def test_sway_filter_ignores_alternating_motion_but_detects_sustained_rotation():
    rule_module = load_rule_module()
    x, y, state = make_inputs()
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(filter_normal_sway=True,
                                      sway_filter_alpha=.5,
                                      modulation_horizontal_rate_limit=1.,
                                      horizontal_rate_emergency_limit=2.),
        x, y, 2, 'cpu')
    rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].tolist() == [0, 1]
    for step, alternating in ((1, 1.4), (2, -1.4), (3, 1.4)):
        state['episode_steps'][:] = step
        state['ang_vel'][:, 0] = torch.tensor([alternating, 1.4])
        rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].tolist() == [1]
    for step in (4, 5):
        state['episode_steps'][:] = step
        state['ang_vel'].zero_()
        rule.observe(state['episode_steps'], state['ang_vel'])
    state['episode_steps'][:] = 6
    state['ang_vel'][0, 0] = 2.1
    rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].tolist() == [0]


def test_sway_filter_clears_previous_episode_history():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(filter_normal_sway=True,
                                      modulation_horizontal_rate_limit=1.), x, y, 1, 'cpu')
    rule.observe(state['episode_steps'], state['ang_vel'])
    choose(rule, 'modulation', state)
    for step in (1, 2, 3):
        state['episode_steps'][:] = step
        state['ang_vel'][0, 0] = 1.4
        rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].tolist() == [0]
    state['episode_steps'][:] = 0
    state['ang_vel'].zero_()
    rule.observe(state['episode_steps'], state['ang_vel'])
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 3
    rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].numel() == 0


def test_sway_filter_does_not_hide_rotation_with_emerging_tilt():
    rule_module = load_rule_module()
    x, y, state = make_inputs(1)
    rule = rule_module.InferenceForwardHold(
        rule_module.ForwardHoldConfig(filter_normal_sway=True,
                                      modulation_horizontal_rate_limit=1.,
                                      sway_tilt_guard=.1), x, y, 1, 'cpu')
    rule.observe(state['episode_steps'], state['ang_vel'])
    choose(rule, 'modulation', state)
    state['episode_steps'][:] = 1
    state['ang_vel'][0, 0] = 1.4
    rule.observe(state['episode_steps'], state['ang_vel'])
    state['episode_steps'][:] = 2
    state['ang_vel'][0, 0] = -1.4
    rule.observe(state['episode_steps'], state['ang_vel'])
    state['episode_steps'][:] = 3
    state['ang_vel'][0, 0] = 1.4
    state['gravity'][0, 0] = .15
    rule.observe(state['episode_steps'], state['ang_vel'])
    assert choose(rule, 'modulation', state)[0].tolist() == [0]
