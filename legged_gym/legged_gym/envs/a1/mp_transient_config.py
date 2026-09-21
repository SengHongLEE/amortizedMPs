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
        num_observations = 44 + 2
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        mu_shape = 8
        omega_shape = 1
        num_actions = mu_shape + omega_shape + 4

        mu_freq_idx_shape = 1
        omega_freq_idx_shape = 1

        send_timeouts = True # send time out information to the algorithm
        env_spacing = 3.
        episode_length_s = 3 # episode length in seconds
        play = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.01 # [m]
        border_size = 0. # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        terrain_length = 20.
        terrain_width = 20.
        num_rows= 1 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        # trimesh only:
        slope_treshold = 0.1 # slopes above this threshold will be corrected to vertical surfaces
        measure_heights = False

    class mp_modeling:
        process_type = 'slow'       #'fast' or 'slow'
        ref_type = 'trot'
        ref_motion_filename = [f'{LEGGED_GYM_ROOT_DIR}/data/a1/pace.txt',
                               f'{LEGGED_GYM_ROOT_DIR}/data/a1/trot.txt',
                               f'{LEGGED_GYM_ROOT_DIR}/data/a1/canter.txt']
        if process_type != 'slow':
            if ref_type == 'pace':
                gait_model_filename = [f'{LEGGED_GYM_ROOT_DIR}/logs/A1/pace-slow/model_500.pt']
            elif ref_type == 'trot':
                gait_model_filename = [f'{LEGGED_GYM_ROOT_DIR}/logs/A1/0702_00-43-35_A1MP_trot_slow_omegaUF-1.25e+01_muUF-2.5e+00_MF-1e+03_noResume/model_500.pt']
            elif ref_type == 'canter':
                gait_model_filename = [f'{LEGGED_GYM_ROOT_DIR}/logs/A1/canter-slow/model_500.pt']
        else:
            gait_model_filename = []
    
    class commands:
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10 # time before command are changed[s]
        heading_command = False #True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0., 1.2] # min max [m/s]
            lin_vel_y = [-0., 0.0]   # min max [m/s]
            lin_vel_z = [-0., 0.]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [-0.0, 0.0]


    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        pos = [.0, 0.0, 0.3] # x,y,z [m]
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
        hierarchical = False
        varied_decimation = False
        decimation= 10
        cycle = 2
        mu_cycle = 1
        omega_cycle = 1
        hip_cycle = 1
        high_cycle = 50

        mu_freq_idx = [5]
        omega_freq_idx = [10]

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

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1,1.]
        randomize_base_mass = True 
        added_mass_range = [-2., 3.]     #0. 3.
        push_robots = True 
        push_interval_s = 5     #5
        max_push_vel_xy = 1.    #0.5    

    class rewards:
        q_bad = 0.7
        q_good = 1. - q_bad
        class scales:
            tracking_lin_vel = 30.
            ang_vel_xy = -0.5
            # action_rate_omega = -0.1
            orientation_yaw = -1.
            # action_rate_mu = -0.1
            exceed_torque_limits_l1norm = -0.4
            dof_vel_limits = -0.4

        high_rewards = ['tracking_lin_vel']
        mu_rewards = ['tracking_lin_vel', 'ang_vel_xy', 'orientation_yaw', 'exceed_torque_limits_l1norm', 'dof_vel_limits']
        omega_rewards = ['tracking_lin_vel', 'ang_vel_xy', 'orientation_yaw', 'exceed_torque_limits_l1norm', 'dof_vel_limits']
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = .9
        soft_torque_limit = .9
        max_contact_force = 180. # forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32

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

LOW_ACTOR_TYPE = 'snn' #mlp, snn, analog_reservoir_mlp, analog_reservoir_snn, lif_reservoir_mlp, lif_reservoir_snn
HIGH_ACTOR_TYPE = 'lif_reservoir_snn_twin_output' #mlp_twin_output, snn_twin_output, lif_reservoir_snn_twin_output

class MotorPrimitivesCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'MPPolicyRunner'
    class policy:
        actor_type = LOW_ACTOR_TYPE      
        high_actor_type = HIGH_ACTOR_TYPE
        init_noise_std = 1.0
        actor_hidden_dims = [32]
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
        readout_hidden_dims = [32]
        readout_activation ="elu"

        #SNN
        num_snn_steps = 8
        snn_lif_beta = 0.9
        snn_lif_threshold = 1.0
        snn_surrogate_alpha = 5.0
        snn_reset_mode = "subtract"
        snn_input_scale = 1.0

        critic_hidden_dims = [32, 16]

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

        low_level_policy_class_name = 'ActorCritic'
        low_algorithm_class_name = 'PPOevent'

        high_level_policy_class_name = 'ActorCriticDiscrete'
        high_algorithm_class_name = 'PPOeventDiscrete'

        num_steps_per_env = 24 #64 # per iteration  24
        max_iterations = 500 # number of policy updates
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'A1'
        run_name = "".join([
            "A1MPtransient-",
            (HIGH_ACTOR_TYPE + "-"),
            (LOW_ACTOR_TYPE),
            ("-highUF_" + np.format_float_scientific(1 / (MotorPrimitivesCfg.control.high_cycle * MotorPrimitivesCfg.control.cycle * MotorPrimitivesCfg.control.decimation * MotorPrimitivesCfg.sim.dt), precision= 3, trim= "-")),
            ("-MF_" + np.format_float_scientific( 1 / MotorPrimitivesCfg.sim.dt, precision= 3, trim= "-")),
            ("-noResume" if not resume else "-from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
