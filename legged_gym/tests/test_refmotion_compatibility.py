"""The reference-motion task keeps its gait logic while using the shared runner API."""

import unittest

import isaacgym  # noqa: F401 - import Isaac Gym before torch-based modules.

from legged_gym.envs import task_registry
from legged_gym.envs.a1.mp_interoceptive import MotorPrimitives as InteroceptiveEnv
from legged_gym.envs.a1.mp_interoceptive_config import MotorPrimitivesCfgPPO as InteroceptivePPO
from legged_gym.envs.a1.mp_refmotion import MotorPrimitives as ReferenceEnv
from legged_gym.envs.a1.mp_refmotion_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
from legged_gym.utils.helpers import class_to_dict


class ReferenceMotionCompatibilityTests(unittest.TestCase):
    def test_reference_observation_and_runner_contract(self):
        env = MotorPrimitivesCfg()
        train = MotorPrimitivesCfgPPO()
        self.assertEqual(env.mp_modeling.ref_type, '')
        self.assertEqual((env.env.num_actions, env.env.num_observations,
                          env.env.num_privileged_obs), (8, 45, 52))
        self.assertEqual(env.control.update_cycle, env.control.low_freq_idx[0])
        self.assertEqual(env.rewards.scales.ref_motion, 100.)
        self.assertNotIn('ref_motion', env.rewards.low_rewards)
        self.assertEqual(train.runner_class_name, 'InteroceptivePolicyRunner')
        self.assertEqual(train.runner.policy_class_name, 'ActorCritic')
        self.assertEqual(train.runner.algorithm_class_name, 'PPOevent')
        self.assertEqual((train.runner.num_steps_per_env, train.runner.save_interval), (24, 20))
        self.assertEqual(class_to_dict(train)['policy'], class_to_dict(InteroceptivePPO())['policy'])
        self.assertEqual(class_to_dict(train)['algorithm'], class_to_dict(InteroceptivePPO())['algorithm'])

    def test_gaits_are_registered_without_changing_default_config(self):
        default_cfg, _ = task_registry.get_cfgs('MP_refmotion_a1')
        self.assertEqual(default_cfg.mp_modeling.ref_type, '')
        for gait in ('pace', 'trot', 'canter'):
            name = f'MP_refmotion_{gait}_a1'
            cfg, train = task_registry.get_cfgs(name)
            self.assertIs(task_registry.get_task_class(name), ReferenceEnv)
            self.assertEqual(cfg.mp_modeling.ref_type, gait)
            self.assertIn(gait, train.runner.run_name)

    def test_shared_environment_methods_are_inherited(self):
        self.assertTrue(issubclass(ReferenceEnv, InteroceptiveEnv))
        for name in ('compute_observations', '_update_cycle_consistency',
                     '_update_contact_rewards', '_create_envs'):
            self.assertNotIn(name, ReferenceEnv.__dict__)
            self.assertIs(getattr(ReferenceEnv, name), getattr(InteroceptiveEnv, name))
        for name in ('step', 'compute_reward', '_reward_ref_motion',
                     '_compute_torques', '_resample_commands'):
            self.assertIn(name, ReferenceEnv.__dict__)


if __name__ == '__main__':
    unittest.main()
