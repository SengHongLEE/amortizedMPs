"""The two A1 MP tasks must retain the archived network and resume contracts."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch.

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path, update_cfg_from_args
from legged_gym.utils.load_config import prepare_resume_configs


LOG_ROOT = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "A1"


def _runs():
    return sorted(path for path in LOG_ROOT.iterdir() if (path / "config.json").is_file())


def _saved_config(run):
    return json.loads((run / "config.json").read_text())


def test_only_two_a1_mp_tasks_are_registered():
    tasks = {name for name in task_registry.task_classes if name.startswith("MP_") and name.endswith("_a1")}
    assert tasks == {"MP_full_a1", "MP_interoceptive_a1"}


def test_full_default_matches_latest_archived_model_contract():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    saved = _saved_config(run)
    env, train = task_registry.get_cfgs("MP_full_a1")
    actual_env, actual_train = class_to_dict(env), class_to_dict(train)
    for key in ("num_actions", "num_modulation_params", "num_observations",
                "num_privileged_obs", "num_controller_observations", "action_space"):
        assert actual_env["env"][key] == saved["env"][key]
    for key in ("scan_mode", "scan_bins_x", "scan_bins_y", "measured_points_x", "measured_points_y"):
        assert actual_env["terrain"][key] == saved["terrain"][key]
    assert actual_train["policy"]["terrain_scan_shape"] == saved["policy"]["terrain_scan_shape"]
    assert actual_train["policy"]["terrain_encoder_dim"] == saved["policy"]["terrain_encoder_dim"]
    assert actual_train["runner"]["run_name"] == saved["runner"]["run_name"]


def test_resume_restores_each_archived_run_configuration():
    for run in _runs():
        task = "MP_full_a1" if "MPfull" in run.name else "MP_interoceptive_a1"
        env, train = task_registry.get_cfgs(task)
        args = SimpleNamespace(resume=True, load_run=run.name, checkpoint=-1)
        restored_env, restored_train = prepare_resume_configs(task, env, train, args)
        saved = _saved_config(run)
        if task == "MP_interoceptive_a1":
            assert restored_env.control.gait_type == saved["control"]["gait_type"]
        assert class_to_dict(restored_env)["env"]["num_observations"] == saved["env"]["num_observations"]
        assert class_to_dict(restored_train)["policy"] == saved["policy"]
        assert Path(restored_train.runner.load_run) == run
        assert restored_train.runner.resume is True


def test_full_resume_uses_current_hurdle_spacing():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    env, train = task_registry.get_cfgs("MP_full_a1")
    expected = env.terrain.terrain_params['hurdle']['spacing']
    args = SimpleNamespace(resume=True, load_run=run.name, checkpoint=-1)
    restored_env, _ = prepare_resume_configs("MP_full_a1", env, train, args)
    assert restored_env.terrain.terrain_params['hurdle']['spacing'] == expected


def test_explicit_checkpoint_uses_archived_filename():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    expected = next(run.glob("model_40_*.pt"))
    selected = get_load_path(str(LOG_ROOT), re.compile(r"^model_(\d+)_\d+\.pt$"),
                             load_run=run.name, checkpoint=40)
    assert Path(selected) == expected


def test_explicit_run_loads_without_experiment_directory():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    selected = get_load_path("/tmp/nonexistent_mp_experiment_directory",
                             re.compile(r"^model_(\d+)_\d+\.pt$"),
                             load_run=str(run), checkpoint=40)
    assert Path(selected) == next(run.glob("model_40_*.pt"))


def test_missing_checkpoint_fails_before_runner_load():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    try:
        get_load_path(str(LOG_ROOT), re.compile(r"^model_(\d+)_\d+\.pt$"),
                      load_run=str(run), checkpoint=9999)
    except FileNotFoundError as exc:
        assert "9999" in str(exc)
    else:
        assert False, "Missing checkpoint should be reported"


def test_task_configs_are_independent_across_runs():
    first_env, first_train = task_registry.get_cfgs("MP_interoceptive_a1")
    first_env.control.gait_type = "gallop"
    first_train.runner.load_run = "an-old-run"
    next_env, next_train = task_registry.get_cfgs("MP_interoceptive_a1")
    assert next_env.control.gait_type == "walk"
    assert next_train.runner.load_run == -1


def test_interoceptive_legacy_checkpoint_restores_iteration():
    from rsl_rl.runners.interoceptive_policy_runner import InteroceptivePolicyRunner

    run = next(path for path in _runs() if "MPinteroceptive-walk" in path.name)
    runner = InteroceptivePolicyRunner.__new__(InteroceptivePolicyRunner)
    runner.device = "cpu"
    runner.modulation_param_actor_critic = SimpleNamespace(load_state_dict=lambda *args, **kwargs: None)
    runner.load(str(run / "model_40_200.pt"), load_optimizer=False)
    assert runner.current_learning_iteration == 200


def test_automatic_resume_selects_latest_run_for_each_task():
    for task in ("MP_full_a1", "MP_interoceptive_a1"):
        env, train = task_registry.get_cfgs(task)
        args = SimpleNamespace(resume=True, load_run=None, checkpoint=-1)
        _, restored_train = prepare_resume_configs(task, env, train, args)
        selected = Path(restored_train.runner.load_run)
        saved = _saved_config(selected)
        assert saved["runner_class_name"] == train.runner_class_name


def test_wrong_task_run_is_rejected():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    env, train = task_registry.get_cfgs("MP_interoceptive_a1")
    args = SimpleNamespace(resume=True, load_run=run.name, checkpoint=-1)
    try:
        prepare_resume_configs("MP_interoceptive_a1", env, train, args)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        assert False, "Expected task mismatch to fail before simulation creation"


def test_resume_source_survives_cli_experiment_override():
    run = next(path for path in _runs() if "MPfull-signed-dense" in path.name)
    env, train = task_registry.get_cfgs("MP_full_a1")
    args = SimpleNamespace(
        resume=True, load_run=run.name, checkpoint=40,
        experiment_name="/tmp/mp_archive_smoke", run_name=None,
        max_iterations=1, seed=None)
    _, train = prepare_resume_configs("MP_full_a1", env, train, args)
    _, train = update_cfg_from_args(None, train, args)
    assert Path(train.runner.load_run) == run


def test_every_archived_checkpoint_loads_strictly_into_its_saved_runner():
    from rsl_rl.runners.interoceptive_policy_runner import InteroceptivePolicyRunner
    from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner

    for run in _runs():
        saved = _saved_config(run)
        env_data, control = saved["env"], saved["control"]
        fake_env = SimpleNamespace(
            num_envs=1,
            num_obs=env_data["num_observations"],
            num_privileged_obs=env_data["num_privileged_obs"],
            num_actions=env_data["num_actions"],
            num_primitives=env_data.get("num_primitives"),
            num_modulation_params=env_data.get("num_modulation_params"),
            cfg=SimpleNamespace(control=SimpleNamespace(**control)),
            reset=lambda: (None, None),
        )
        runner_class = (MultiInteroceptivePolicyRunner if "MPfull" in run.name
                        else InteroceptivePolicyRunner)
        runner = runner_class(fake_env, saved, device="cpu")
        checkpoints = sorted(run.glob("model_*.pt"))
        assert checkpoints
        for checkpoint in checkpoints:
            runner.load(str(checkpoint), load_optimizer=False)
