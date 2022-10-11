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
	  (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
	  (2 * l_low * l_up))
  l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
  theta_hip = np.arcsin(-x / l) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)
  return np.array([theta_ab, theta_hip, theta_knee])

def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
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
assert len(self._foot_link_ids) == self.num_legs
# toe_id = self._foot_link_ids[leg_id]

motors_per_leg = self.num_motors // self.num_legs
joint_position_idxs = list(
    range(leg_id * motors_per_leg,
          leg_id * motors_per_leg + motors_per_leg))

joint_angles = foot_position_in_hip_frame_to_joint_angle(
      foot_local_position - HIP_OFFSETS[leg_id],
      l_hip_sign=(-1)**(leg_id + 1))

# Return the joing index (the same as when calling GetMotorAngles) as well
# as the angles.
return joint_position_idxs, joint_angles.tolist()
    

