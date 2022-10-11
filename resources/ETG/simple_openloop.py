# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""Simple openloop trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import attr
from gym import spaces
import numpy as np
import copy

from .ETG_model import ETG_layer, ETG_model


class ETGOffsetGenerator(object):
    def __init__(self, ETG_T, ETG_T2=0.5, ETG_H=20, dt=0.01, act_mode="traj", task_mode="normal", step_y=0.05):
        self.ETG_T = ETG_T
        self.ETG_T2 = ETG_T2
        self.ETG_H = ETG_H
        self.dt = dt
        self.task_mode = task_mode
        self.step_y = step_y

        phase = np.array([-np.pi / 2, 0])
        self._ETG_agent = ETG_layer(self.ETG_T, self.dt, self.ETG_H, 0.04, phase, 0.2, self.ETG_T2)
        self._ETG_model = ETG_model(task_mode=self.task_mode, act_mode=act_mode, step_y=self.step_y)
        self._ETG_weight = 1.0
        self._ETG_w, self._ETG_b, prior_points = self.Opt_with_points(ETG=self._ETG_agent, ETG_T=ETG_T, ETG_H=ETG_H,
                                                                      Footheight=0.07, Steplength=0.05)
        self._first_reset = True
        self._state_table = []
        self._action_table = []

        self._init_joint_pose = self.gym_env.robot.GetDefaultInitJointPose()
        action_high = np.array([0.3, 0.4, 0.4] * 4) + self._init_joint_pose
        action_low = np.array([-0.3, -0.4, -0.4] * 4) + self._init_joint_pose
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float64)

    # solve "x" of "Ax = b"
    def LS_sol(self, A, b, precision=1e-4, alpha=0.05, lamb=1, w0=None):
        n, m = A.shape  # 6x20
        if w0 is not None:
            x = copy.copy(w0)
        else:
            x = np.zeros((m, 1))  # 20x1
        err = A.dot(x) - b  # 6x1
        err = err.transpose().dot(err)
        i = 0
        diff = 1
        while err > precision and i < 1000:
            A1 = A.transpose().dot(A)  # 20x20
            dx = A1.dot(x) - A.transpose().dot(b)
            if w0 is not None:
                dx += lamb * (x - w0)
            x = x - alpha * dx
            diff = np.linalg.norm(dx)
            err = A.dot(x) - b
            err = err.transpose().dot(err)
            i += 1
        return x

    def Opt_with_points(self, ETG, ETG_T=0.4, ETG_H=20, points=None, b0=None, w0=None, precision=1e-4, lamb=0.5,
                        **kwargs):
        ts = np.array([0.75, 0, 0.125, 0.25, 0.375, 0.5]) * ETG_T
        if points is None:
            Steplength = kwargs["Steplength"] if "Steplength" in kwargs else 0.05
            Footheight = kwargs["Footheight"] if "Footheight" in kwargs else 0.08
            Penetration = kwargs["Penetration"] if "Penetration" in kwargs else 0.01
            # [[0.0, -0.01], [-0.05, -0.005], [-0.075, 0.06], [0.0, 0.1], [0.075, 0.06], [0.05, -0.005]]
            points = np.array([[0, -Penetration],
                               [-Steplength, -Penetration * 0.5], [-Steplength * 1.0, 0.6 * Footheight],
                               [0, Footheight],
                               [Steplength * 1.0, 0.6 * Footheight], [Steplength, -Penetration * 0.5]])
        obs = []
        for t in ts:
            v = ETG.update(t)  # calculate V(t), 20 dim
            obs.append(v)
        obs = np.array(obs).reshape(-1, ETG_H)  # V(1-6), 6x(20x1)
        if b0 is None:
            b = np.mean(points, axis=0)
        else:
            b = np.array([b0[0], b0[-1]])  # 2x1
        points_t = points - b  # 6x(2x1), W*V(t)=P(t)-b
        if w0 is None:
            x1 = self.LS_sol(A=obs, b=points_t[:, 0].reshape(-1, 1), precision=precision, alpha=0.05)  # 1x20, "x" axis
            x2 = self.LS_sol(A=obs, b=points_t[:, 1].reshape(-1, 1), precision=precision, alpha=0.05)  # 1x20, "z" axis
        else:
            x1 = self.LS_sol(A=obs, b=points_t[:, 0].reshape(-1, 1), precision=precision, alpha=0.05, lamb=lamb,
                             w0=w0[0, :].reshape(-1, 1))
            x2 = self.LS_sol(A=obs, b=points_t[:, 1].reshape(-1, 1), precision=precision, alpha=0.05, lamb=lamb,
                             w0=w0[-1, :].reshape(-1, 1))
        x1 = np.zeros((20, 1))
        w_ = np.stack((x1, np.zeros((ETG_H, 1)), x2), axis=0).reshape(3, -1)  # 3x20
        b_ = np.array([b[0], 0, b[1]])  # 3x1
        return w_, b_, points

    def reset(self, **kwargs):
        # if "ETG_w" in kwargs.keys() and kwargs["ETG_w"] is not None:
        #   self._ETG_w = kwargs["ETG_w"]
        # if "ETG_b" in kwargs.keys() and kwargs["ETG_b"] is not None:
        #   self._ETG_b = kwargs["ETG_b"]
        self._ETG_agent.reset()
        # t = self.env.get_time_since_reset()
        if self._first_reset:
            self._first_reset = False
            for i in range(int(self.ETG_T / self.dt)):
                # CPG-RBF, the output of the hidden neuron: V(t), [tuple:2(20x1)]
                state = self._ETG_agent.update2(i * self.dt)
                # P(t) = W ∗ V(t) + b, W:[3x20], b:[3x1] The phase difference of TG is T/2.
                # P(t) is the local position in foot link's frame
                act_etg = self._ETG_model.forward(self._ETG_w, self._ETG_b, state)
                ### Use IK to compute the motor angles, act_ref = etg_act - init_act !
                act_ref = self._ETG_model.act_clip(act_etg)
                self._state_table.append(state)
                self._action_table.append(act_ref)

    def get_action(self, current_time=None, input_action=None):
        # lookup state and action table
        time_index = int((current_time % self.ETG_T) / self.dt)
        state = self._state_table[time_index]
        act_ref = self._action_table[time_index]
        self._last_ETG_act = (act_ref - self._init_joint_pose) * self._ETG_weight + self._init_joint_pose
        self._last_ETG_obs = state[0]
        self._ETG_phase = self._ETG_agent.get_phase()
        return self._last_ETG_act

    def get_observation(self, input_observation):
        """Get the trajectory generator's observation."""
        return input_observation
