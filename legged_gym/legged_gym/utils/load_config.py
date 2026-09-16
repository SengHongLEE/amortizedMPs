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

    # task_registry.get_cfgs() 原本会做这个同步；load 后也同步一次
    env_cfg.seed = train_cfg.seed
    return env_cfg, train_cfg