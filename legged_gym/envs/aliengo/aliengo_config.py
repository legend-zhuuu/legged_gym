from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AlienGoCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_actions = 12
        num_observations = 48

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        start_hip = 0.
        start_thigh = 0.8
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
        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/ETG/ac2f10ms.onnx"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf"
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.35
        only_positive_rewards = True
        max_contact_force = 350.
        class scales ( LeggedRobotCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-1., 1.]

    class sim(LeggedRobotCfg.sim):
        up_axis = 1  # 0 is y, 1 is z
        dt = 0.001


class AlienGoCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_aliengo'
        max_iterations = 300
