# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
import re
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, pattern, load_run=-1, checkpoint=-1):
    if load_run in (-1, "-1"):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"No runs in this directory: {root}")
        runs = sorted(entry.path for entry in os.scandir(root)
                      if entry.is_dir() and entry.name != "exported")
        if not runs:
            raise FileNotFoundError(f"No runs in this directory: {root}")
        run_dir = runs[-1]
    else:
        run_dir = os.path.join(root, str(load_run))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    models = [file for file in os.listdir(run_dir) if pattern.fullmatch(file)]
    if checkpoint == -1:
        if not models:
            raise FileNotFoundError(f"No valid model files found in {run_dir}")
        model = max(models, key=lambda file: int(pattern.fullmatch(file).group(1)))
    else:
        matches = [file for file in models
                   if int(pattern.fullmatch(file).group(1)) == checkpoint]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one checkpoint with index {checkpoint} in {run_dir}; found {len(matches)}")
        model = matches[0]
    return os.path.join(run_dir, model)

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if getattr(args, 'reset_primitive_selection', False):
            if not args.resume:
                raise ValueError('--reset_primitive_selection requires --resume')
            cfg_train.runner.reset_primitive_selection_on_resume = True
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--reset_primitive_selection", "action": "store_true", "default": False,
         "help": "When resuming, initialize a new primitive selection policy while loading the low policy"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--adaptive_forward", "action": "store_true", "default": False,
         "help": "Enable inference-only forward/hold for MP_full_a1 playback."},
        {"name": "--gait_max_skips", "type": int, "default": 2,
         "help": "Maximum consecutive scheduled gait updates to hold during adaptive playback."},
        {"name": "--modulation_max_skips", "type": int, "default": 3,
         "help": "Maximum consecutive scheduled modulation updates to hold during adaptive playback."},
        {"name": "--horizontal_rate_limit", "type": float, "default": 2.0,
         "help": "Inference-only horizontal angular velocity trigger in rad/s for adaptive MP_full_a1 playback."},
        {"name": "--modulation_horizontal_rate_limit", "type": float, "default": 4.0,
         "help": "Override only the modulation actor horizontal angular velocity trigger in rad/s."},
        {"name": "--extero_edge_trigger", "action": "store_true",
         "help": "Use the default abrupt-terrain-edge trigger during adaptive playback."},
        {"name": "--no_extero_edge_trigger", "action": "store_true",
         "help": "Disable the default abrupt-terrain-edge trigger during adaptive playback."},
        {"name": "--terrain_edge_height_m", "type": float, "default": 0.035,
         "help": "Minimum adjacent scan-row height jump in metres for an extero terrain edge."},
        {"name": "--filter_normal_sway", "action": "store_true", "default": False,
         "help": "Filter ordinary horizontal angular velocity oscillation during adaptive MP_full_a1 playback."},
        {"name": "--tracking_linear_error_limit", "type": float, "default": 0.6,
         "help": "Filtered planar linear velocity tracking error trigger in m/s."},
        {"name": "--tracking_yaw_error_limit", "type": float, "default": 0.6,
         "help": "Filtered yaw angular velocity tracking error trigger in rad/s."},
        {"name": "--no_tracking_error_trigger", "action": "store_true",
         "help": "Disable inference-only command tracking error triggers."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    args.extero_edge_trigger = args.extero_edge_trigger or not args.no_extero_edge_trigger

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
def plot_colored_trajectory(
    ax,
    x,
    y,
    cmap="viridis",
    norm=None,
    linewidth=1.5,
):
    """
    Plot a 2-D trajectory colored according to time.

    x, y: shape [T]
    """

    points = np.column_stack([x, y]).reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    if norm is None:
        norm = Normalize(0, len(x) - 1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )

    lc.set_array(np.arange(len(x) - 1))

    ax.add_collection(lc)

    return lc

def append_to_npy_file(file_path, new_data):
        # 读取现有的 .npy 文件
        try:
            existing_data = np.load(file_path)
        except FileNotFoundError:
            print("File not found. A new file will be created.")
            existing_data = np.empty((0,) + new_data.shape[1:])

        # 检查是否文件为空，并进行适当的拼接
        if existing_data.size == 0:
            updated_data = new_data
        else:
            updated_data = np.concatenate((existing_data, new_data))

        # 将更新后的数据保存回 .npy 文件
        np.save(file_path, updated_data)
        # print("Data has been appended and saved successfully.")
