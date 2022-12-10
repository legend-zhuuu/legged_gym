from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AlienGoCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        # num_observations = 36
        num_observations = 223  # for measure heights
        episode_length_s = 10  # episode length in seconds
        use_rms = True
        debug = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True  # select a unique terrain type and pass all arguments
        terrain_kwargs = {'type': 'stairs_terrain',  # "gap_terrain", "perlin_terrain", "stairs_terrain"
                          'terrain_kwargs': {
                              # "octaves": 1,
                              # "tile": (0, 3),
                              "step_width": 0.4,
                              "step_height": 0.2,
                          }
                          }  # Dict of arguments for selected terrain
        max_init_terrain_level = 3  # starting curriculum state
        terrain_length = 10.
        terrain_width = 10.
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 2  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        step_cmd = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0, 3.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.43]  # x,y,z [m]
        start_hip = 0.
        start_thigh = 0.6425
        start_calf = -1.287
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            '1FR_hip_joint': start_hip,  # [rad]
            '1FR_thigh_joint': start_thigh,  # [rad]  0.69
            '1FR_calf_joint': start_calf,  # [rad]  -1.42

            '2FL_hip_joint': start_hip,  # [rad]
            '2FL_thigh_joint': start_thigh,  # [rad]  0.69
            '2FL_calf_joint': start_calf,  # [rad]  -1.42

            '3RR_hip_joint': start_hip,  # [rad]
            '3RR_thigh_joint': start_thigh,  # [rad]  0.69
            '3RR_calf_joint': start_calf,  # [rad]  -1.42

            '4RL_hip_joint': start_hip,  # [rad]
            '4RL_thigh_joint': start_thigh,  # [rad]  0.69
            '4RL_calf_joint': start_calf,  # [rad]  -1.42
        }
        num_steps_per_policy = 5

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 60}  # [N*m/rad]
        damping = {'joint': 1}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1  # control add policy action or not
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/ETG/actuator_net.pt"
        use_plotjuggler = False
        wandb_log = True
        get_depth_img = False

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf"
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

    class camera:
        img_width = 360
        img_length = 240

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        push_robots = False
        scale_mass_range = [-0.2, 0.2]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.4
        max_contact_force = 350.
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        only_positive_rewards = True

        class scales:
            # termination = -200.
            # tracking_ang_vel = 1.0
            # torques = -5.e-6
            # dof_acc = -2.e-7
            # lin_vel_z = -0.5
            # dof_pos_limits = -1.
            # dof_vel = -0.0
            # ang_vel_xy = -0.0
            # feet_contact_forces = -0.
            # orientation = -5.0
            # feet_air_time = 2.
            # alive = 0.1
            # up = 0.6
            # height = 0.3
            # feet_vel = 2.0
            # feet_pos = 4.0
            # action_rate = 0.5
            # feet_airtime = 6.0
            # feet_slip = 0.5
            # tau = 0.02
            # badfoot = 1.0
            # footcontact = 1.0
            # done = 1
            linear_tracking = 0.6
            angular_tracking = 0.25
            torque = -1.2e-4
            body_posture = -0.2
            joint_vel = -1.0e-4
            joint_acc = -1.5e-8
            collision = -0.2
            feet_air_time = 0.4
            # air_time_consistency= -0.05
            # rhythm = -0.2
            angular_motion = -0.02
            linear_motion = -0.2
            alive = -0.
            slip = -0.02
            # joint_motion = 0.0
            # joint_deviation = 0.0
            # body_motion = 0.0

            # velx = 1.
            # contact_nums = 1.
            # contact_errs = 1.
            # contact_rate = 1.
            # energy_sum = 1.

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            imu_rpy = 0.01
            ang_vel = 0.2

    class viewer:
        ref_env = 0
        pos = [-2, -2, 1]  # [m]
        lookat = [0., 0., 0.]  # [m]

    class sim(LeggedRobotCfg.sim):
        up_axis = 1  # 0 is y, 1 is z
        dt = 0.001  # dt = 0.002 * 5 = 0.01

        class physx:
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more


class AlienGoCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256]
        critic_hidden_dims = [256, 256]
        activation = 'relu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 2e-4  # 5.e-4
        schedule = 'fixed'  # could be adaptive, fixed
        num_learning_epochs = 4
        max_grad_norm = 0.5
        num_mini_batches = 4
        gamma = 0.992

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24
        max_iterations = 2000

        # log
        run_name = ''
        experiment_name = 'rough_aliengo'
