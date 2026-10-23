# MIT License

# Copyright (c) 2026 Seng-Hong Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_config import BaseConfig

class MotorPrimitivesCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_modulation_params = 4
        num_primitives = 4
        num_actions = num_modulation_params + 1

        num_observations = 37 + num_modulation_params + 1
        num_privileged_obs = num_observations + 3 + 4 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise

        send_timeouts = True # send time out information to the algorithm
        env_spacing = 3.
        episode_length_s = 10 # episode length in seconds
        play = False

    class terrain:
        # All terrain options live here. See docs/terrain/mp_multi_primitive.md.
        mesh_type = 'plane'  # plane, trimesh, heightfield, or None
        horizontal_scale = 0.1  # m, official SubTerrain grid spacing
        vertical_scale = 0.005  # m
        slope_threshold = 0.3  # dz/dxy above this becomes vertical in the official converter
        border_size = 4.0  # m, outer flat border
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        terrain_length = 8.0  # x extent of each tile, m
        terrain_width = 8.0    # y extent of each tile, m
        num_rows = 10           # difficulty levels
        num_cols = 8          # >= number of nonzero mixture types; one type per column
        max_init_terrain_level = None  # None samples all rows; otherwise inclusive maximum
        difficulty_range = [0.0, 1.0]  # inclusive endpoints; one row uses the first value
        difficulty_mode = 'linear'    # linear rows or independent random difficulty per tile
        row_difficulties = None       # optional explicit values, e.g. [0., .2, .5, 1.]
        seed = 1
        selected = None  # None uses mixture/layout; otherwise any type or named variant below
        # Optional direct SDK call for selected terrain; None uses terrain_params.
        # Example: {'type': 'pyramid_sloped_terrain', 'slope': 0.2, 'platform_size': 3.0}
        selected_kwargs = None
        # Optional exact column order (length must equal num_cols), overrides weights.
        terrain_columns = None  # e.g. ['uphill', 'downhill'] with num_cols=2
        terrain_variants = {
            'uphill': {'type': 'ramp', 'params': {'slope_range': [0.05, 0.3], 'slope_signs': [1]}},
            'downhill': {'type': 'ramp', 'params': {'slope_range': [0.05, 0.3], 'slope_signs': [-1]}},
            'side_slope': {'type': 'ramp', 'params': {'slope_range': [0.05, 0.2], 'axis': 'y'}},
        }
        terrain_proportions = {
            'flat': 1., 'slope': 1., 'rough_slope': 1., 'rough': 1.,
            'rubble': 1., 'stairs': 1., 'ramp': 1., 'stairs_linear': 0.,
            'stepping_stones': 0., 'wave': 1., 'gap': 0., 'pit': 0.,
            'hurdle': 0., 'beam': 0.,
        }
        # *_range accepts a fixed scalar or [easy, hard]; descending ranges are allowed.
        # Exception: rubble.stone_size_range is the random min/max block size.
        # Any type/variant also accepts axis ('x'/'y'), platform_size, height_offset,
        # and roughness={'height_range': [0., .02], 'height_step': .005,
        #                'downsampled_scale': .2} for additive surface noise.
        terrain_params = {
            'flat': {'height': 0.},
            'slope': {'slope_range': [0.0, 0.3], 'slope_signs': [-1, 1]},
            'ramp': {'slope_range': [0.0, 0.3], 'slope_signs': [-1, 1]},
            'rough_slope': {'slope_range': [0.05, 0.20], 'slope_signs': [-1, 1],
                            'height_range': [0.005, 0.025], 'height_step': 0.005, 'downsampled_scale': 0.2},
            'rough': {'height_range': [0.01, 0.04], 'height_step': 0.005, 'downsampled_scale': 0.2},
            # NVIDIA discrete_obstacles_terrain: overlapping rectangular blocks/depressions.
            'rubble': {'height_range': [0.04, 0.12], 'stone_size_range': [1.0, 2.0],
                       'num_rects': 20},
            'stairs': {'step_height_range': [0.05, 0.1], 'step_width': 0.3,
                       'step_signs': [-1, 1]},
            'stairs_linear': {'step_height_range': [0.05, 0.1], 'step_width': 0.3,
                              'step_signs': [-1, 1]},
            'stepping_stones': {'stone_size_range': [0.8, 0.3], 'stone_distance_range': [0.1, 0.3],
                               'height_range': [0., 0.05], 'depth_range': [0.2, 0.6]},
            'wave': {'amplitude_range': [0.15, 0.5], 'num_waves': 3},
            'gap': {'width_range': [0.1, 0.5], 'depth_range': [0.2, 0.6], 'spacing': 1.5},
            'pit': {'size_range': [2.5, 3.0], 'depth_range': [0.1, 0.5]},
            'hurdle': {'width_range': [0.2, 0.3], 'height_range': [0.05, 0.2], 'spacing': 1.5},
            'beam': {'width_range': [1.0, 0.3], 'depth_range': [0.2, 0.6]},
        }
        platform_size = 3.0   # passed to official generators that support a platform
        spawn_platform = False  # True explicitly flattens the central platform on every type
        origin_sample_size = 2.0  # side length of central square used for spawn-height maximum
        spawn_xy_jitter = 0.2
        boundary_margin = 0.6  # reaching tile edge ends the rollout as a timeout
        measure_heights = False  # no terrain samples in actor or critic observations

    class commands:
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        max_vel_x = 2.5
        resampling_time = 5 # time before command are changed[s]
        heading_command = True #True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [.3, 3.] # min max [m/s]
            lin_vel_y = [-0., 0.0]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [-0.0, 0.0]


    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        pos = [.0, 0.0, 0.32] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control:
        action_scale = 1.
        decimation= 10
        cycle = 2

        primitive_selection_update_cycle = 25
        modulation_param_update_cycle = 5
        update_quality_threshold = 0.85

        oscillator_type = "phase_oscillator"
        
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 2.0}     # [N*m*s/rad]

    class asset:
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_updated_v2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 #
        hip_link_length_a1 = 0.0838
        thigh_link_length_a1 = 0.213
        calf_link_length_a1 = 0.213
        max_step_length = 0.12
        ground_clearance = 0.1
        ground_penetration = 0.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1,1.]
        randomize_base_mass = True 
        added_mass_range = [-2., 3.]     #0. 3.
        push_robots = True 
        push_interval_s = 5    #5
        max_push_vel_xy = 1.    #0.5    

    class rewards:
        class scales:
            ang_vel_xy = -0.1
            orientation = -2.
            orientation_yaw = -2.
            action_rate_ps = -0.1
            action_rate_mp = -0.1
            exceed_torque_limits_l1norm = -0.4
            dof_vel_limits = -0.4
            # stance_slip = -0.2
            # touchdown_impact = -0.002

            tracking_lin_vel = 3.
            tracking_ang_vel = 1.5

        task_rewards = ['tracking_lin_vel', 'tracking_ang_vel']
        # DOI: 10.1038/s42256-025-01065-z, Eqs. (14)--(15).
        # Selection terms use the same named-function routing as modulation.
        primitive_selection_rewards = [
            'pi_g_utility', 'pi_g_cot', 'pi_g_torque', 'pi_g_contact', 'pi_g_work',
        ]
        class primitive_selection_scales:
            pi_g_utility = 1.
            pi_g_cot = 1.
            pi_g_torque = 1.
            pi_g_contact = 1.
            pi_g_work = 1.

        pi_g_utility_weight = 0.4
        pi_g_switch_cost = 2.  # motion-dependent switch cost inside utility, before dt scaling
        pi_g_contact_force = 1.0  # N; same contact threshold as the dynamics
        modulation_param_rewards = ['tracking_lin_vel', 'tracking_ang_vel', 
                                    'orientation', 'orientation_yaw', 
                                    'ang_vel_xy', 'action_rate_mp', 
                                    # 'stance_slip',
                                    'exceed_torque_limits_l1norm', 'dof_vel_limits']
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = .9
        soft_torque_limit = .9
        max_contact_force = 180. # forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32

        contact_on_force = 5.0            # N, vertical force; hysteresis
        contact_off_force = 1.0           # N
        touchdown_min_air_time = 0.02     # s; reject contact chatter
        touchdown_force_window = 0.02     # s after touchdown; capture a delayed peak
        touchdown_velocity_scale = 1.0   # m/s
        touchdown_force_scale = 180.0     # N, excess above max_contact_force
        touchdown_force_weight = 0.25

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.5
            dof_vel = 1.5 
            lin_vel = 0.1
            ang_vel = 0.4
            gravity = 0.1 
            height_measurements = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [1.8, 0.6, .4]  # [m]
        lookat = [1.8, 0.8, .3]  # [m]

    class sim:
        dt = 0.001
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

PRIMITIVE_SELECTION_ACTOR_TYPE = 'lif_reservoir_snn_single_output' #mlp, snn, analog_reservoir_mlp, analog_reservoir_snn, lif_reservoir_mlp, lif_reservoir_snn
MODULATION_PARAM_ACTOR_TYPE = 'snn'
class MotorPrimitivesCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'MultiInteroceptivePolicyRunner'
    class policy:
        actor_type = MODULATION_PARAM_ACTOR_TYPE      
        high_actor_type = PRIMITIVE_SELECTION_ACTOR_TYPE
        init_noise_std = 1.0
        actor_hidden_dims = [64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # Reservoir parameters
        reservoir_dim = 16
        reservoir_connectivity = 0.5
        spectral_radius = 0.9
        reservoir_input_scale = 0.5
        reservoir_bias_scale = 0.1
        leak_rate = 1.
        num_reservoir_steps = 1
        reservoir_activation = "tanh"

        reservoir_lif_beta = 0.9
        reservoir_lif_threshold = 1.0
        reservoir_surrogate_alpha = 5.0
        reservoir_reset_mode = "subtract"

        include_input_in_readout = False
        readout_hidden_dims = [64,32]
        readout_activation ="elu"

        #SNN
        num_snn_steps = 8
        snn_lif_beta = 0.9
        snn_lif_threshold = 1.0
        snn_surrogate_alpha = 5.0
        snn_reset_mode = "subtract"
        snn_input_scale = 1.0

        critic_hidden_dims = [64, 32, 16]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 10
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

        primitive_policy_class_name = 'ActorCriticDiscrete'
        primitive_algorithm_class_name = 'PPOeventDiscrete'

        modulation_param_policy_class_name = 'ActorCritic'
        modulation_param_algorithm_class_name = 'PPOevent'


        num_steps_per_env = 12  #24 # per iteration  24
        max_iterations = 1000 # number of policy updates
        # logging
        save_interval = 10 # check for potential saves every this many iterations
        experiment_name = 'A1'
        run_name = "".join([
            "A1MPmultiprimitive-",
            MotorPrimitivesCfg.terrain.mesh_type,
            ("-psF_" + np.format_float_scientific(1 / (MotorPrimitivesCfg.control.primitive_selection_update_cycle * MotorPrimitivesCfg.control.cycle * MotorPrimitivesCfg.control.decimation * MotorPrimitivesCfg.sim.dt), precision= 3, trim= "-")),
            ("-mpF_" + np.format_float_scientific(1 / (MotorPrimitivesCfg.control.modulation_param_update_cycle * MotorPrimitivesCfg.control.cycle * MotorPrimitivesCfg.control.decimation * MotorPrimitivesCfg.sim.dt), precision= 3, trim= "-")),
            ("-MF_" + np.format_float_scientific( 1 / MotorPrimitivesCfg.sim.dt, precision= 3, trim= "-")),
            ("-noResume" if not resume else "-from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
