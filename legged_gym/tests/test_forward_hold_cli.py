"""Adaptive inference is an explicit runtime option in evaluation and playback."""

import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / 'legged_gym/legged_gym/scripts/evaluate_mp_full_gaits.py'
PLAY = ROOT / 'legged_gym/legged_gym/scripts/play.py'
HELPERS = ROOT / 'legged_gym/legged_gym/utils/helpers.py'


def test_play_prints_policy_frequencies_when_interrupted():
    source = ast.parse(PLAY.read_text())
    play = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == 'play')
    namespace = {}
    exec(compile(ast.Module(body=[play], type_ignores=[]), str(PLAY), 'exec'), namespace)
    cfg = SimpleNamespace(
        env=SimpleNamespace(play=False, num_envs=2),
        noise=SimpleNamespace(add_noise=True),
        domain_rand=SimpleNamespace(push_robots=True),
        terrain=SimpleNamespace(mesh_type='plane'))
    env = SimpleNamespace(num_envs=2, device='cpu', dt=.02,
                          scan_cache=SimpleNamespace(x=[], y=[]),
                          get_observations=lambda: None)

    class Runner:
        calls = 0

        def get_inference_policy(self, device):
            return (object(), object())

        def inference_rollout(self, obs, high, low):
            self.calls += 1
            if self.calls > 4:
                raise KeyboardInterrupt
            decisions = (([0], [0, 1]), ([0, 1], [1]),
                         ([1], [0, 1]), ([0], [0]))[self.calls - 1]
            dones = ((False, False), (True, False),
                     (False, True), (True, False))[self.calls - 1]
            self.last_inference_decisions = {
                actor: {'forward': torch.tensor(ids, dtype=torch.long)}
                for actor, ids in zip(('gait', 'modulation'), decisions)}
            return None, None, None, torch.tensor(dones), None

    runner = Runner()
    namespace.update(
        task_registry=SimpleNamespace(
            get_cfgs=lambda task: (cfg, object()),
            make_env=lambda task, args, env_cfg: (env, env_cfg),
            make_alg_runner=lambda *args, **kwargs: (runner, None)),
        prepare_resume_configs=lambda task, env_cfg, train_cfg, args: (env_cfg, train_cfg),
        configure_forward_hold_scan=lambda args, env_cfg: None,
        ForwardHoldConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        InferenceForwardHold=lambda config, *args: SimpleNamespace(config=config),
        torch=torch,
        format_policy_update_frequency=lambda gait, modulation, steps, dt:
            f'{gait / (steps * dt):.2f} Hz, {modulation / (steps * dt):.2f} Hz')
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace['play'](SimpleNamespace(task='MP_full_a1', num_envs=2,
                                         adaptive_forward=True, gait_max_skips=2,
                                         modulation_max_skips=3,
                                         horizontal_rate_limit=1.15,
                                         modulation_horizontal_rate_limit=4.,
                                         extero_edge_trigger=True,
                                         terrain_edge_height_m=.04,
                                         filter_normal_sway=True))
    assert 'env 0 episode 1: 50.00 Hz, 25.00 Hz' in output.getvalue()
    assert 'env 1 episode 1: 33.33 Hz, 50.00 Hz' in output.getvalue()
    assert 'env 0 episode 2: 25.00 Hz, 50.00 Hz' in output.getvalue()
    assert '31.25 Hz, 37.50 Hz' in output.getvalue()
    assert runner.inference_forward_hold.config.horizontal_rate_limit == 1.15
    assert runner.inference_forward_hold.config.modulation_horizontal_rate_limit == 4.
    assert runner.inference_forward_hold.config.extero_edge_trigger is True
    assert runner.inference_forward_hold.config.terrain_edge_height_m == .04
    assert runner.inference_forward_hold.config.filter_normal_sway is True


def test_evaluator_prints_policy_frequencies_after_rollout():
    source = ast.parse(EVALUATOR.read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'format_policy_update_frequency'
               for node in ast.walk(main))


def test_evaluator_prints_episode_frequency_during_rollout():
    source = ast.parse(EVALUATOR.read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    rollout = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == 'step')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'format_policy_update_frequency'
               for node in ast.walk(rollout))


def test_evaluator_exposes_adaptive_options_and_rejects_negative_limits():
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    help_result = subprocess.run([sys.executable, str(EVALUATOR), '--help'],
                                 capture_output=True, text=True, env=env)
    assert help_result.returncode == 0, help_result.stderr
    assert '--adaptive-forward' in help_result.stdout
    assert '--gait-max-skips' in help_result.stdout
    assert '--modulation-max-skips' in help_result.stdout
    assert '--filter-normal-sway' in help_result.stdout
    assert '--modulation-horizontal-rate-limit' in help_result.stdout
    assert '--extero-edge-trigger' in help_result.stdout
    assert '--no-extero-edge-trigger' in help_result.stdout
    assert '--tracking-linear-error-limit' in help_result.stdout
    assert '--tracking-yaw-error-limit' in help_result.stdout
    assert '--no-tracking-error-trigger' in help_result.stdout
    bad = subprocess.run([sys.executable, str(EVALUATOR), '--checkpoint', '/missing',
                          '--output', '/tmp/unused-forward-hold', '--gait', 'policy',
                          '--adaptive-forward', '--gait-max-skips', '-1'],
                         capture_output=True, text=True, env=env)
    assert bad.returncode != 0
    assert 'max-skips' in bad.stderr


def test_play_exposes_horizontal_rate_limit_option():
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    result = subprocess.run([sys.executable, str(PLAY), '--help'],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert '--horizontal_rate_limit' in result.stdout
    assert '--modulation_horizontal_rate_limit' in result.stdout
    assert '--extero_edge_trigger' in result.stdout
    assert '--no_extero_edge_trigger' in result.stdout
    assert '--terrain_edge_height_m' in result.stdout
    assert '--filter_normal_sway' in result.stdout
    assert '--tracking_linear_error_limit' in result.stdout
    assert '--tracking_yaw_error_limit' in result.stdout
    assert '--no_tracking_error_trigger' in result.stdout


def test_play_forward_hold_defaults_and_edge_opt_out():
    code = '''
from isaacgym import gymutil
import ast
import sys
from pathlib import Path
source = ast.parse(Path(sys.argv[1]).read_text())
function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == 'get_args')
scope = {'gymutil': gymutil}
exec(compile(ast.Module(body=[function], type_ignores=[]), sys.argv[1], 'exec'), scope)
for flags in ([], ['--no_extero_edge_trigger']):
    sys.argv = ['play.py'] + flags
    args = scope['get_args']()
    print(args.horizontal_rate_limit, args.modulation_horizontal_rate_limit,
          args.extero_edge_trigger)
'''
    result = subprocess.run([sys.executable, '-c', code, str(HELPERS)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '2.0 4.0 True' in result.stdout
    assert '2.0 4.0 False' in result.stdout


def test_play_enables_10hz_scan_only_for_adaptive_mp_full():
    source = ast.parse(PLAY.read_text())
    function = next((node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'configure_forward_hold_scan'), None)
    assert function is not None, 'play.py needs an opt-in scan configuration helper'
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PLAY), 'exec'), namespace)
    configure = namespace['configure_forward_hold_scan']
    cfg = SimpleNamespace(terrain=SimpleNamespace(scan_period_steps=1))
    configure(SimpleNamespace(task='MP_full_a1', adaptive_forward=False), cfg)
    assert cfg.terrain.scan_period_steps == 1
    configure(SimpleNamespace(task='MP_full_a1', adaptive_forward=True), cfg)
    assert cfg.terrain.scan_period_steps == 5
