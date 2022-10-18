import os
import sys
import torch
import numpy as np
import pybullet
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.math import quaternion_to_matrix, quaternion_to_euler
from .aliengo_config import AlienGoCfg

from .laikago_motor import ActuatorNetMotorModel


class AlienGo(LeggedRobot):
    cfg: AlienGoCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = ActuatorNetMotorModel(actuator_network_path)

    def _compute_torques(self, actions):
        if self.cfg.control.use_actuator_network:
            torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            _actions = (actions * self.cfg.control.action_scale + self.default_dof_pos).cpu().numpy()
            dof_pos, dof_vel = self.dof_pos.cpu().numpy(), self.dof_vel.cpu().numpy()
            # print("tar, pos, vel:", _actions[0][:3], dof_pos[0][:3], dof_vel[0][:3], sep="\n")
            torques = torch.from_numpy(self.actuator_network.convert_to_torque(_actions, dof_pos, dof_vel)[0]).to(self.device).float()
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def _init_buffers(self):
        super()._init_buffers()
        self.base_pos = self.root_states[:, :3]
        self.base_quat_mat = quaternion_to_matrix(self.base_quat)
        self.base_rpy = quaternion_to_euler(self.base_quat)
        self.base_rpy_rate = self.GetTrueBaseRollPitchYawRate()
        self.last_base10 = torch.tile(self.base_pos, (10, 1, 1))  # 10 * num_envs * xyz

        net_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.body_state = gymtorch.wrap_tensor(net_body_state).view(self.num_envs, -1, 13)  # shape: num_envs, num_bodies, (position, rotation, linear velocity, angular velocity).
        self.foot_state = self.body_state[:, self.feet_indices, :]
        self.foot_position_world = self.foot_state[..., :3]
        self.foot_velocity_world = self.foot_state[..., 7:10]
        self.last_foot_velocity = torch.zeros_like(self.foot_velocity_world)
        self.last_contacts = torch.ones_like(self.last_contacts)  # 将legged_gym中的last_contacts初始值设为1
        self.real_contact = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.foot_contact_state = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.foothold = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        self.target_foot_hold = torch.zeros_like(self.foothold)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.history_action = torch.zeros(self.num_envs, 2, self.num_actions, device=self.device, requires_grad=False)
        self.energy = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.transport_cost = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.motor_power = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    def ComputeTargetPosInBase2WorldFrame(self, foot_command):  # todo: update foot hold
        """Get the robot's foothold position in the world frame."""
        base_position, base_orientation = self.base_pos, self.base_quat
        world_link_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # Projects to world space.
        for i in range(self.num_envs):
            world_link_pos[i] = torch.from_numpy(np.asarray(pybullet.multiplyTransforms(
                base_position, base_orientation, foot_command, (0, 0, 0, 1))[0]))
        return world_link_pos

    def GetFootPositionsInBaseFrame(self):
        foot_world = self.foot_position_world
        foot_base = quat_rotate_inverse(self.base_quat, foot_world)
        return foot_base

    def GetTrueBaseRollPitchYawRate(self):
        o = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        angular_velocity = torch.unbind(self.base_ang_vel)
        orientation = torch.unbind(self.base_quat)
        for i in range(self.num_envs):
            _, orientation_inversed = pybullet.invertTransform(
                [0, 0, 0], orientation[i])
            # Transform the angular_velocity at neutral orientation using a neutral
            # translation and reverse of the given orientation.
            relative_velocity, _ = pybullet.multiplyTransforms(
                [0, 0, 0], orientation_inversed, angular_velocity[i],
                pybullet.getQuaternionFromEuler([0, 0, 0]))
            o[i] = torch.from_numpy(np.asarray(relative_velocity))
        return o

    def GetFootContacts(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 10.
        return contacts

    def GetBadFootContacts(self):
        bad_num = 0
        # todo:
        return bad_num

    def GetFootContactState(self):
        contacts = self.GetFootContacts()
        first_contact = torch.logical_xor(torch.logical_or(self.last_contacts, contacts), self.last_contacts)
        lift_contact = torch.logical_xor(torch.logical_or(self.last_contacts, contacts), contacts)
        contact_state = torch.ones(self.num_envs, 4, device=self.device, requires_grad=False)
        for e in range(self.num_envs):
            for i in range(contacts.shape[1]):
                if first_contact[e, i]:
                    contact_state[e, i] = 2
                elif lift_contact[e, i]:
                    contact_state[e, i] = 3
                elif contacts[e, i]:
                    contact_state[e, i] = 1
                else:
                    contact_state[e, i] = 0
        self.last_contacts = contacts
        return contact_state

    def GetEnergyConsumptionPerControlStep(self):
        for i in range(self.num_envs):
            self.energy[i] = torch.abs(torch.dot(self.torques[i], self.dof_vel[i])) * self.dt * self.cfg.init_state.num_steps_per_policy

    def GetCostOfTransport(self):
        tv = self.torques * self.dof_vel
        tv[tv < 0] = 0
        self.transport_cost = tv.sum(1) / (torch.norm(self.base_lin_vel, dim=-1) * 20.0 * 9.8)

    def GetMotorPower(self):
        tv = self.torques * self.dof_vel
        tv[tv < 0] = 0
        self.motor_power = tv.sum(1) / 20.0

    def check_termination(self):
        # super().check_termination()
        rot_mat = self.base_quat_mat
        pose = self.base_rpy
        footposition = self.GetFootPositionsInBaseFrame()  # in base frame
        footz = footposition[:, :, -1]
        base = self.base_pos
        base_std = torch.sum(torch.std(self.last_base10, dim=0), dim=-1)
        logic_1 = torch.logical_or(rot_mat[:, -1] < 0.5, torch.mean(footz, dim=-1) > -0.1)
        logic_2 = torch.logical_or(torch.max(footz, dim=-1).values > 0, torch.logical_and(base_std <= 2e-4, self.episode_length_buf >= 10))
        self.reset_buf = torch.logical_or(logic_1, logic_2).nonzero().flatten()
        return torch.logical_or(logic_1, logic_2)

    def c_prec(self, v, t, m):
        # w = np.arctanh(np.sqrt(0.95)) / m  # 2.89 / m
        w = torch.sqrt(np.arctanh(0.95)) / m  # 1.35 / m  ???
        return torch.tanh(torch.pow((v - t) * w, 2))

    # todo: scale * dt

    def _reward_up(self):
        roll, pitch, _ = torch.unbind(self.base_rpy, -1)
        return 1 - 0.5 * self.c_prec(torch.abs(roll), 0, 0.25) - 0.5 * self.c_prec(torch.abs(pitch), 0, 0.25)

    def _reward_height(self):
        world_z = torch.mean(self.target_foot_hold[:, :, -1], dim=-1).unsqueeze(1)
        r = torch.abs(self.root_states[:, 2].unsqueeze(1) - world_z - 0.405)
        return 1 - self.c_prec(r, 0, 0.15)

    def _reward_feet_vel(self):
        contact_state = self.foot_contact_state
        foot_velocity = self.foot_velocity_world
        rew_feet_vel = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        for i in range(contact_state.shape[1]):
            env_ids = (contact_state[:, i] == 2).nonzero().flatten()  # first contact
            contact_velocity = torch.norm(self.last_foot_velocity[env_ids, i], dim=-1)
            rew_feet_vel += (-self.c_prec(contact_velocity, 0.0, 5.0))
        self.last_foot_velocity = foot_velocity
        return rew_feet_vel

    def _reward_feet_pos(self):
        contact_state = self.foot_contact_state
        contact_position = self.foot_velocity_world
        rew_feet_length = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        for i in range(contact_state.shape[1]):
            env_ids = (contact_state[:, i] == 3).nonzero().flatten()  # left contact
            if env_ids:
                self.foothold[env_ids, i] = self.target_foot_hold[env_ids, i]
            env_ids = (contact_state[:, i] == 2).nonzero().flatten()  # first contact
            if env_ids:
                contact_position[env_ids, i, 2] -= 0.02  # foot radius
                feet_errors = self.foothold[env_ids, i] - contact_position[env_ids, i]
                feet_length_err = torch.norm(feet_errors, dim=-1)
                rew = (1 - self.c_prec(feet_length_err, 0, 0.2))
                rew_feet_length += rew  # positive reward
        return rew_feet_length

    def _reward_action_rate(self):
        new_action = self.actions
        r1 = torch.sum(torch.square(new_action - self.last_actions), dim=-1)
        self.last_actions = new_action
        return 1 - self.c_prec(r1, 0, 0.2)

    def _reward_feet_airtime(self, traj_period=0.6):
        # Reward long _steps
        contact_state = self.foot_contact_state
        self.feet_air_time += self.dt
        rew_airtime = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        for i in range(self.feet_air_time.shape[1]):
            # Punishing long duration of behavior
            airtime_err = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

            flag = torch.logical_or((self.feet_air_time[:, i] % traj_period) < 1e-5, (traj_period - self.feet_air_time[:, i] % traj_period) < 1e-5)
            env_ids = flag.nonzero().flatten()
            rew_airtime[env_ids] += -1.0

            env_ids = (contact_state[:, i] == 2).nonzero().flatten()
            airtime_err[env_ids] = torch.abs(self.feet_air_time[env_ids, i] - traj_period / 2)
            rew_airtime[env_ids] += (-self.c_prec(airtime_err, 0, traj_period / 4))

            env_ids = torch.logical_or(contact_state[:, i] == 2, contact_state[:, i] == 3)
            self.feet_air_time[env_ids, i] = 0
        return rew_airtime

    def _reward_feet_slip(self):
        contact = self.GetFootContacts()
        foot_vel = self.foot_velocity_world
        foot_vel = torch.norm(foot_vel, dim=-1)
        rew = torch.sum(foot_vel * contact, dim=-1)
        return -rew

    def _reward_tau(self):
        return -self.motor_power

    def _reward_badfoot(self):
        return -self.GetBadFootContacts()

    def _reward_footcontact(self):
        lose_contact_num = torch.sum(1.0 - self.GetFootContacts(), dim=-1)
        return torch.max(lose_contact_num - 2, torch.zeros_like(lose_contact_num))

    def _reward_done(self):
        return -1 if torch.any(self.check_termination()) else 0
