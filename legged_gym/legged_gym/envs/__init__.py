# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.a1.mp_interoceptive import MotorPrimitives
from legged_gym.envs.a1.mp_interoceptive_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_interoceptive_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_refmotion import MotorPrimitives as ReferenceMotionMotorPrimitives
from legged_gym.envs.a1.mp_refmotion_config import (
    MotorPrimitivesCfg as ReferenceMotionCfg,
    MotorPrimitivesCfgPPO as ReferenceMotionCfgPPO,
)
task_registry.register('MP_refmotion_a1', ReferenceMotionMotorPrimitives,
                       ReferenceMotionCfg(), ReferenceMotionCfgPPO())
for gait in ('pace', 'trot', 'canter'):
    env_cfg = ReferenceMotionCfg()
    env_cfg.mp_modeling.ref_type = gait
    if env_cfg.mp_modeling.ref_type == 'canter':
        env_cfg.control.update_cycle = 2
    train_cfg = ReferenceMotionCfgPPO()
    train_cfg.runner.run_name = train_cfg.runner.run_name.replace('A1MPrm--', f'A1MPrm-{gait}-', 1)
    task_registry.register(f'MP_refmotion_{gait}_a1', ReferenceMotionMotorPrimitives, env_cfg, train_cfg)

from legged_gym.envs.a1.mp_full import MotorPrimitives
from legged_gym.envs.a1.mp_full_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_full_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.go2.mp import MotorPrimitives
from legged_gym.envs.go2.mp_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_go2", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.go2.hierarchical_control import AmortizedMPs
from legged_gym.envs.go2.hierarchical_control_config import AmortizedMPsCfg, AmortizedMPsCfgPPO
task_registry.register( "Amortized_go2", AmortizedMPs, AmortizedMPsCfg(), AmortizedMPsCfgPPO())
