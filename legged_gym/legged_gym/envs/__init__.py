# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

# from legged_gym.envs.a1.RL import QuadrupedIL
# from legged_gym.envs.a1.RL_config import LeggedRobotCfg, LeggedRobotCfgPPO
# task_registry.register( "RL", QuadrupedIL, LeggedRobotCfg(), LeggedRobotCfgPPO())

from legged_gym.envs.a1.mp import MotorPrimitives
from legged_gym.envs.a1.mp_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_twin import MotorPrimitives
from legged_gym.envs.a1.mp_twin_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_twin_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_adaptive import MotorPrimitives
from legged_gym.envs.a1.mp_adaptive_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_adaptive_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())


from legged_gym.envs.a1.hierarchical_control import AmortizedMPs
from legged_gym.envs.a1.hierarchical_control_config import AmortizedMPsCfg, AmortizedMPsCfgPPO
task_registry.register( "Amortized_a1", AmortizedMPs, AmortizedMPsCfg(), AmortizedMPsCfgPPO())

from legged_gym.envs.go2.mp import MotorPrimitives
from legged_gym.envs.go2.mp_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_go2", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.go2.hierarchical_control import AmortizedMPs
from legged_gym.envs.go2.hierarchical_control_config import AmortizedMPsCfg, AmortizedMPsCfgPPO
task_registry.register( "Amortized_go2", AmortizedMPs, AmortizedMPsCfg(), AmortizedMPsCfgPPO())