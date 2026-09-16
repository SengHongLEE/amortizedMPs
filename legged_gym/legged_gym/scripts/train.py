# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import numpy as np
import os
from datetime import datetime
import random
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import sys

def train(args):
    # args.task = 'HDRLgap'
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = get_args()

    train(args)
