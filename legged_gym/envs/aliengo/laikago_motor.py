# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
"""Motor model for laikago."""
import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import collections
from collections.abc import Sequence
import numpy as np
import torch
import onnxruntime as ort

NUM_MOTORS = 12

MOTOR_COMMAND_DIMENSION = 5

# These values represent the indices of each field in the motor command tuple
POSITION_INDEX = 0
POSITION_GAIN_INDEX = 1
VELOCITY_INDEX = 2
VELOCITY_GAIN_INDEX = 3
TORQUE_INDEX = 4


class ActuatorNetMotorModel(object):
    def __init__(self,
                 net_path,
                 kp=60,
                 kd=1,
                 torque_limits=33.5):
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        if torque_limits is not None:
            if isinstance(torque_limits, (Sequence, np.ndarray)):
                self._torque_limits = np.asarray(torque_limits)
            else:
                self._torque_limits = np.full(NUM_MOTORS, torque_limits)
        # self.actuator_net = ActuatorNetWithHistory()
        # self.actuator_net.load_state_dict(torch.load("data/actuator_net_with_history.pt",
        #                                              map_location=torch.device('cpu'))['model'])
        # self.actuator_net.to(torch.device("cuda"))
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # self.ort_session = ort.InferenceSession("data/actuator_net.onnx", opts)  # P150D4(-1 -2 -3)
        # self.ort_session = ort.InferenceSession("data/0821actuator_net(-3).onnx", opts)
        # self.ort_session = ort.InferenceSession("data/904f210ms.onnx", opts)  # P150D4(-1 -3)
        self.ort_session = ort.InferenceSession(net_path, opts)  # P120D3(-1 -3)

    def set_strength_ratios(self, ratios):
        """Set the strength of each motors relative to the default value.

        Args:
          ratios: The relative strength of motor output. A numpy array ranging from
            0.0 to 1.0.
        """
        self._strength_ratios = ratios

    def set_voltage(self, voltage):
        pass

    def get_voltage(self):
        return 0.0

    def set_viscous_damping(self, viscous_damping):
        pass

    def get_viscous_dampling(self):
        return 0.0

    def convert_to_torque(self,
                          motor_commands,
                          motor_angle,
                          motor_velocity):
        angle_err = motor_commands - motor_angle
        if not hasattr(self, "angle_err_buffer"):
            self.angle_err_buffer = collections.deque((angle_err,) * 10, maxlen=10)
        self.angle_err_buffer.append(angle_err)
        if not hasattr(self, "velocity_buffer"):
            self.velocity_buffer = collections.deque((motor_velocity,) * 10, maxlen=10)
        self.velocity_buffer.append(motor_velocity)
        time0 = time.time()
        motor_torques = np.zeros(motor_commands.shape)
        # actnet_input = np.array((self.angle_err_buffer[-1], self.angle_err_buffer[-2], self.angle_err_buffer[-3],
        #     self.velocity_buffer[-1], self.velocity_buffer[-2], self.velocity_buffer[-3]), dtype=np.float32).transpose()

        for i in range(motor_torques.shape[0]):
            actnet_input = np.array((self.angle_err_buffer[-1][i], self.angle_err_buffer[-2][i],
                                     self.velocity_buffer[-1][i], self.velocity_buffer[-2][i]), dtype=np.float32).transpose()
            torques = self.ort_session.run(['output'], {'input': actnet_input}, )[0]
            motor_torques[i] = torques.flatten()
        # print("actnet time: ", (time.time() - time0) * 1000)
        if self._torque_limits is not None:
            if len(self._torque_limits) != motor_torques.shape[1]:
                raise ValueError(
                    "Torque limits dimension does not match the number of motors.")
            motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                                    self._torque_limits)
        return motor_torques, motor_torques
