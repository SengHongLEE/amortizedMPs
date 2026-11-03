# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.load_config import prepare_resume_configs

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = prepare_resume_configs(args.task, env_cfg, train_cfg, args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, env_cfg=env_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()

    train(args)
