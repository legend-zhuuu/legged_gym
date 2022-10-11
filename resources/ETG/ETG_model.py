import numpy as np
from copy import copy

ACTION_DIM = 12
LENGTH_HIP_LINK = 0.083
LENGTH_UPPER_LINK = 0.25
LENGTH_LOWER_LINK = 0.25

HIP_OFFSETS = np.array([[0.2399, -0.051, 0.0], [0.2399, 0.051, 0.0],
                        [-0.2399, -0.051, 0.0], [-0.2399, 0.051, 0.0]
                        ])
BASE_FOOT = np.array([[0.2399, -0.134, -0.38], [0.2399, 0.134, -0.38],
                      [-0.2399, -0.134, -0.38], [-0.2399, 0.134, -0.38]
                      ])


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = LENGTH_UPPER_LINK
    l_low = LENGTH_LOWER_LINK
    l_hip = LENGTH_HIP_LINK * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    x += 0.008
    theta_knee = -np.arccos(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def ComputeMotorAnglesFromFootLocalPosition(leg_id,
                                            foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.
    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    num_motors = 12
    num_legs = 4
    motors_per_leg = num_motors // num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = foot_position_in_hip_frame_to_joint_angle(
        foot_local_position - HIP_OFFSETS[leg_id],
        l_hip_sign=(-1) ** (leg_id + 1))

    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()


class ETG_layer():
    def __init__(self, T, dt, H, sigma_sq, phase, amp, T2_radio):
        # T2_ratio mean the ratio forward t
        self.dt = dt
        self.T = T
        self.t = 0
        self.H = H
        self.sigma_sq = sigma_sq  # variance:0.04
        self.phase = phase
        self.amp = amp
        self.u = []
        self.omega = 2.0 * np.pi / T
        self.T2_ratio = T2_radio
        for h in range(H):
            t_now = h * self.T / (H - 0.9)
            self.u.append(self.forward(t_now))
        self.u = np.asarray(self.u).reshape(-1, 2)
        self.TD = 0

    def forward(self, t):
        x = []
        for i in range(self.phase.shape[0]):
            x.append(self.amp * np.sin(self.phase[i] + t * self.omega))
        return np.asarray(x).reshape(-1)

    def update(self, t=None):
        time = t if t is not None else self.t
        x = self.forward(time)
        self.t += self.dt
        r = []
        for i in range(self.H):
            dist = np.sum(np.power(x - self.u[i], 2)) / self.sigma_sq
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)
        return r

    def update2(self, t=None, info=None):  # calculate V(t)
        time = t if t is not None else self.t  # step: 0.002*repeat_num
        # CPG-RBF network structure
        x = self.forward(time)  # a*sin[phi+omega(t)]
        x2 = self.forward(time + self.T2_ratio * self.T)  # a*sin[phi+omega(t+B)]
        self.t += self.dt
        r = []
        for i in range(self.H):  # H=20, the number of neurons
            dist = np.sum(np.power(x - self.u[i], 2)) / self.sigma_sq  # sigma_sq: the variance between neuron outputs
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)  # [20x1]
        r2 = []
        for i in range(self.H):
            dist = np.sum(np.power(x2 - self.u[i], 2)) / self.sigma_sq
            r2.append(np.exp(-dist))
        r2 = np.asarray(r2).reshape(-1)  # [20x1]
        return (r, r2)  # [2x20]

    def observation_T(self):
        ts = np.arange(0, self.T, self.dt)
        x = {t: self.forward(t) for t in ts}
        r_all = {}
        for j in ts:
            r = []
            for i in range(self.H):
                dist = np.sum(np.power(x[j] - self.u[i], 2)) / self.sigma_sq
                r.append(np.exp(-dist))
            r_all[j] = np.asarray(r).reshape(-1)
        return r_all

    def get_phase(self):
        p = []
        p.append(self.omega * self.t % (2 * np.pi))
        p.append(self.omega * (self.t + self.T2_ratio * self.T) % (2 * np.pi))
        return np.asarray(p).reshape(-1)

    def reset(self):
        self.t = 0
        self.TD = 0


class ETG_model():
    def __init__(self, task_mode="normal", act_mode="traj", step_y=0.5):
        self.act_mode = act_mode
        self.pose_ori = np.array([0, 0.9, -1.8] * 4)
        self.task_mode = task_mode

    def forward(self, w, b, x):
        x1 = np.asarray(x[0]).reshape(-1, 1)  # 20x1
        x2 = np.asarray(x[1]).reshape(-1, 1)  # 20x1
        act1 = w.dot(x1).reshape(-1) + b
        act2 = w.dot(x2).reshape(-1) + b
        new_act = np.zeros(ACTION_DIM)
        if self.task_mode == "gallop":
            new_act[:3] = act1.copy()
            new_act[3:6] = act1.copy()
            new_act[6:9] = act2.copy()
            new_act[9:] = act2.copy()
        else:
            new_act[:3] = act1.copy()
            new_act[3:6] = act2.copy()
            new_act[6:9] = act2.copy()
            new_act[9:] = act1.copy()
        return new_act

    def act_clip(self, new_act):
        act = np.zeros(12)
        if self.act_mode == "pose":
            # joint control mode
            act = np.tanh(new_act) * np.array([0.1, 0.7, 0.7] * 4)
        elif self.act_mode == "traj":
            # foot trajectory mode
            for i in range(4):
                delta = new_act[i * 3:(i + 1) * 3].copy()
                while (1):
                    index, angle = ComputeMotorAnglesFromFootLocalPosition(
                        i, delta + BASE_FOOT[i])
                    if np.sum(np.isnan(angle)) == 0:
                        break
                    delta *= 0.95
                act[index] = np.array(angle)
        return act
