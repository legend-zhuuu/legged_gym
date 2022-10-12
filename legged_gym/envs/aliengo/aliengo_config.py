from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AlienGoCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 12
        num_actions = 12
        num_observations = 48

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        start_hip = 0.
        start_thigh = 0.69
        start_calf = -1.42
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FR_hip_joint': start_hip,  # [rad]
            'FR_thigh_joint': start_thigh,  # [rad]  0.69
            'FR_calf_joint': start_calf,  # [rad]  -1.42

            'FL_hip_joint': start_hip,  # [rad]
            'FL_thigh_joint': start_thigh,  # [rad]  0.69
            'FL_calf_joint': start_calf,  # [rad]  -1.42

            'RR_hip_joint': start_hip,  # [rad]
            'RR_thigh_joint': start_thigh,  # [rad]  0.69
            'RR_calf_joint': start_calf,  # [rad]  -1.42

            'RL_hip_joint': start_hip,  # [rad]
            'RL_thigh_joint': start_thigh,  # [rad]  0.69
            'RL_calf_joint': start_calf,  # [rad]  -1.42
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 120.}  # [N*m/rad]
        damping = {'joint': 3.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/ETG/ac2f10ms.onnx"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf"
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.4
        max_contact_force = 350.
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        only_positive_rewards = True
        class scales ( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            dof_pos_limits = -1.
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            orientation = -5.0
            feet_air_time = 2.

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-1., 1.]

    class sim(LeggedRobotCfg.sim):
        up_axis = 1  # 0 is y, 1 is z
        dt = 0.002


class AlienGoCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_aliengo'
        num_steps_per_policy = 5
        num_steps_per_env = 100
        max_iterations = 300
