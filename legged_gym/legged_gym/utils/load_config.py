import json
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

TRAIN_KEYS = {"seed", "runner_class_name", "policy", "algorithm", "runner"}
SKIP_KEYS = {"init_member_classes"}

def merge_cfg_obj(obj, data):
    for key, value in data.items():
        if key in SKIP_KEYS:
            continue

        current = getattr(obj, key, None)
        if isinstance(value, dict) and current is not None and hasattr(current, "__dict__"):
            merge_cfg_obj(current, value)
        elif hasattr(obj, key):
            setattr(obj, key, value)
    return obj

def load_run_config(env_cfg, train_cfg, run_dir):
    config_path = Path(run_dir) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    train_data = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    env_data = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS and k not in SKIP_KEYS}

    merge_cfg_obj(train_cfg, train_data)
    merge_cfg_obj(env_cfg, env_data)
    # A selector reset applies only to the training invocation that requested it.
    # Archived configs are also loaded for evaluation and later resumes.
    if hasattr(train_cfg.runner, 'reset_primitive_selection_on_resume'):
        train_cfg.runner.reset_primitive_selection_on_resume = False

    # task_registry.get_cfgs() 原本会做这个同步；load 后也同步一次
    env_cfg.seed = train_cfg.seed
    return env_cfg, train_cfg


def prepare_resume_configs(task_name, env_cfg, train_cfg, args):
    """Restore the archived environment before it or its policy is constructed."""
    if not args.resume:
        return env_cfg, train_cfg
    if task_name not in ("MP_full_a1", "MP_interoceptive_a1"):
        return env_cfg, train_cfg

    selector_reset_terrain_columns = (list(env_cfg.terrain.terrain_columns)
                                      if task_name == 'MP_full_a1'
                                      and getattr(args, 'reset_primitive_selection', False)
                                      else None)
    hurdle_spacing = (env_cfg.terrain.terrain_params['hurdle']['spacing']
                      if task_name == 'MP_full_a1' else None)

    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    requested = args.load_run
    if requested is None or requested == "-1":
        candidates = sorted(path for path in log_root.iterdir()
                            if path.is_dir() and (path / "config.json").is_file())
        for path in reversed(candidates):
            saved = json.loads((path / "config.json").read_text())
            if saved.get("runner_class_name") == train_cfg.runner_class_name:
                run_dir = path
                break
        else:
            raise FileNotFoundError(f"No archived run for {task_name} under {log_root}")
    else:
        run_dir = Path(requested)
        if not run_dir.is_absolute():
            run_dir = log_root / run_dir
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing run config: {run_dir / 'config.json'}")
    saved = json.loads((run_dir / "config.json").read_text())
    if saved.get("runner_class_name") != train_cfg.runner_class_name:
        raise ValueError(f"Run {run_dir.name} is incompatible with {task_name}")
    env_cfg, train_cfg = load_run_config(env_cfg, train_cfg, run_dir)
    if task_name == 'MP_full_a1':
        # Training resumes use the current hurdle course; archived configs stay intact.
        env_cfg.terrain.terrain_params.setdefault('hurdle', {})['spacing'] = hurdle_spacing
        if selector_reset_terrain_columns is not None:
            env_cfg.terrain.terrain_columns = selector_reset_terrain_columns
        guidance = 'pi_g_context_guidance'
        if guidance not in env_cfg.rewards.primitive_selection_rewards:
            env_cfg.rewards.primitive_selection_rewards = [
                *env_cfg.rewards.primitive_selection_rewards, guidance]
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = str(run_dir)
    args.load_run = str(run_dir)
    return env_cfg, train_cfg
