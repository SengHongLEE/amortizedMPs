# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

# from legged_gym.envs.a1.RL import QuadrupedIL
# from legged_gym.envs.a1.RL_config import LeggedRobotCfg, LeggedRobotCfgPPO
# task_registry.register( "RL", QuadrupedIL, LeggedRobotCfg(), LeggedRobotCfgPPO())

from legged_gym.envs.a1.mp_preset import MotorPrimitives
from legged_gym.envs.a1.mp_preset_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_preset_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_refmotion import MotorPrimitives
from legged_gym.envs.a1.mp_refmotion_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_refmotion_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_interoceptive import MotorPrimitives
from legged_gym.envs.a1.mp_interoceptive_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_interoceptive_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_multi_primitive import MotorPrimitives
from legged_gym.envs.a1.mp_multi_primitive_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_multi_primitive_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_proprio_terrain import MotorPrimitives
from legged_gym.envs.a1.mp_proprio_terrain_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_pro_terrain_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())
from legged_gym.envs.a1.mp_proprio_terrain_config import AdaptiveMotorPrimitivesCfg, AdaptiveMotorPrimitivesCfgPPO
task_registry.register("MP_pro_terrain_v2_a1", MotorPrimitives, AdaptiveMotorPrimitivesCfg(), AdaptiveMotorPrimitivesCfgPPO())
from legged_gym.envs.a1.mp_proprio_terrain_config import ModalMotorPrimitivesCfg, ModalMotorPrimitivesCfgPPO
task_registry.register("MP_pro_terrain_modal_a1", MotorPrimitives, ModalMotorPrimitivesCfg(), ModalMotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_full import MotorPrimitives
from legged_gym.envs.a1.mp_full_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_full_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_omega import MotorPrimitives
from legged_gym.envs.a1.mp_omega_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_omega_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_adaptive import MotorPrimitives
from legged_gym.envs.a1.mp_adaptive_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_adaptive_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.a1.mp_transient import MotorPrimitives
from legged_gym.envs.a1.mp_transient_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_transient_a1", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.go2.mp import MotorPrimitives
from legged_gym.envs.go2.mp_config import MotorPrimitivesCfg, MotorPrimitivesCfgPPO
task_registry.register( "MP_go2", MotorPrimitives, MotorPrimitivesCfg(), MotorPrimitivesCfgPPO())

from legged_gym.envs.go2.hierarchical_control import AmortizedMPs
from legged_gym.envs.go2.hierarchical_control_config import AmortizedMPsCfg, AmortizedMPsCfgPPO
task_registry.register( "Amortized_go2", AmortizedMPs, AmortizedMPsCfg(), AmortizedMPsCfgPPO())
