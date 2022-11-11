import json
import os
import socket
import sys
from time import time
import random
import torch
import numpy as np
import wandb
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.math import quaternion_to_matrix, quaternion_to_euler
from .aliengo_config import AlienGoCfg

from .laikago_motor import ActuatorNetMotorModel
from .actuator import Actuator
from .simple_openloop import ETGOffsetGenerator


class UdpPublisher(object):
    """
    Send data stream of locomotion to outer tools such as PlotJuggler.
    """

    def __init__(self, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def send(self, data: dict):
        msg = json.dumps(data)
        ip_port = ('127.0.0.1', self.port)
        self.client.sendto(msg.encode('utf-8'), ip_port)


class AlienGo(LeggedRobot):
    cfg: AlienGoCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.BASE_FOOT = torch.tensor([[0.2399, -0.134, -0.38],
                                       [0.2399, 0.134, -0.38],
                                       [-0.2399, -0.134, -0.38],
                                       [-0.2399, 0.134, -0.38]], device=sim_device, requires_grad=False)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            # self.actuator_network = ActuatorNetMotorModel(actuator_network_path)
            self.actuator_network = Actuator(actuator_network_path, self.num_envs, self.num_actions, device=self.device)

        self.ETG = ETGOffsetGenerator(ETG_T=0.6, dt=0.01)
        self.ETG.reset()
        self.prepare_TG_table()
        self._data_publisher = UdpPublisher(9870)

    def step(self, actions):
        self.etg_time += self.dt
        policy_action_scale = torch.tensor([0.3, 0.4, 0.4], device=self.device).repeat(4)
        etg_actions = self.get_etg_actions(self.etg_time).float() - self.default_dof_pos
        actions = actions * self.cfg.control.action_scale * policy_action_scale + etg_actions + self.default_dof_pos  # abs dof pos
        super().step(actions)
        if self.cfg.control.use_plotjungler:
            state_dict = {
                'foot_command': self.foot_command.cpu().numpy().tolist(),
                'base_pos': self.base_pos.cpu().numpy().tolist(),
                'foot_pos': self.foot_position_world.cpu().numpy().tolist(),
                'base_rpy': self.base_rpy.cpu().numpy().tolist(),
                'base_rpy_rate': self.base_rpy_rate.cpu().numpy().tolist(),
                'base_lin_vel': self.base_lin_vel.cpu().numpy().tolist(),
                'real_contact': self.real_contact.cpu().numpy().tolist(),
                'dof_pos': self.dof_pos.cpu().numpy().tolist(),
                'dof_vel': self.dof_vel.cpu().numpy().tolist(),
                'feet_air_time': self.feet_air_time.cpu().numpy().tolist(),
                'actions': self.actions.cpu().numpy().tolist(),
                'target_foot_hold': self.target_foot_hold.cpu().numpy().tolist(),
                'foot_contact_state': self.foot_contact_state.cpu().numpy().tolist(),
                'torques': self.torques.cpu().numpy().tolist(),
                'foot_vel': self.foot_velocity_world.cpu().numpy().tolist(),
            }
            reward_dict = {}
            for key in self.episode_sums.keys():
                reward_dict['rew_' + key] = self.reward_plot[key].cpu().numpy().tolist()

            self._data_publisher.send({
                'state_dict': state_dict,
                'reward_dict': reward_dict,
            })
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        # 重写buf的更新
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
            dof name: [ 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
            body name: ['base',
                        'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
                        'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
                        'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot',
                        'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']
        """
        # time1 = time()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.base_quat_mat[:] = quaternion_to_matrix(self.base_quat).reshape(self.num_envs, 9)
        self.base_rpy[:] = quaternion_to_euler(self.base_quat)
        self.base_rpy_rate[:] = self.GetTrueBaseRollPitchYawRate()
        self.last_base10[1:, ...] = self.last_base10[:9, ...].clone()
        self.last_base10[0, ...] = self.base_pos

        self.body_state = gymtorch.wrap_tensor(self.net_body_state).view(self.num_envs, -1, 13)  # shape: num_envs, num_bodies, (position, rotation, linear velocity, angular velocity).
        self.foot_state = self.body_state[:, self.feet_indices, :]
        self.foot_position_world = self.foot_state[..., :3]
        self.foot_velocity_world = self.foot_state[..., 7:10]
        self.real_contact = self.GetFootContacts()
        self.foot_contact_state = self.GetFootContactState()
        self.update_target_foot_hold()  # foot_command update

        # time2 = time()
        self.foot_command = self.ComputeTargetPosInWorld2FootFrame(self.target_foot_hold)  # compute foot_command_target in foot frame
        # time3 = time()
        self.GetEnergyConsumptionPerControlStep()
        self.energy_sum += self.energy
        self.GetCostOfTransport()
        self.GetMotorPower()

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        # time4 = time()
        # compute observations, rewards, resets, ...
        self.check_termination()
        # time5 = time()
        self.compute_reward()
        # time6 = time()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # time7 = time()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # print("update1:{}, world2foot:{}, update2:{}, termi:{}, reward:{}, obser:{}".format(time2-time1, time3-time2, time4-time3,time5-time4, time6-time5, time7-time6, ))
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        rot_mat = self.base_quat_mat
        footposition = self.GetFootPositionsInBaseFrame()  # in base frame
        footz = footposition[:, :, -1]
        base_std = torch.sum(torch.std(self.last_base10, dim=0), dim=-1)
        logic_1 = torch.logical_or(rot_mat[:, -1] < 0.5, torch.mean(footz, dim=-1) > -0.1)
        logic_2 = torch.logical_or(torch.max(footz, dim=-1).values > 0, base_std <= 2e-4)
        self.reset_buf = torch.logical_or(logic_1, logic_2)
        self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.reset_buf = torch.logical_and(logic_1, logic_2)

    def compute_observations(self):
        """
        Computes observations
        """
        self.obs_buf = torch.cat((
            self.foot_command.flatten(start_dim=1),  # 12
            self.base_lin_vel,  # 3
            self.real_contact,  # 4
            self.base_rpy,  # 3
            self.base_rpy_rate,  # 3
            self.dof_pos,  # 12
            self.dof_vel,  # 12
            self.feet_air_time,  # 4
            self.actions,  # 12
        ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_base_states(env_ids)
        self.last_foot_velocity[env_ids] = torch.zeros_like(self.last_foot_velocity[env_ids])
        self.etg_time[env_ids] = 0.
        self.contact_ok_num[env_ids] = 0.
        self.last_contacts[env_ids] = torch.ones_like(self.last_contacts[env_ids])
        self.real_contact[env_ids] = torch.ones_like(self.real_contact[env_ids])
        self.foot_command_base[env_ids] = self.foot_command_rand(env_ids)
        env_ids_mesh, foot_ids = torch.meshgrid(env_ids, torch.arange(4, device=self.device))
        env_ids_mesh, foot_ids = env_ids_mesh.flatten(), foot_ids.flatten()
        if len(env_ids) != 0:
            self.target_foot_hold[env_ids_mesh, foot_ids] = self.ComputeTargetPosInBase2WorldFrame(env_ids_mesh, foot_ids)
        self.foot_command = self.ComputeTargetPosInWorld2FootFrame(self.target_foot_hold)
        self.energy_sum[env_ids] = 0.
        if self.info_statistics["sta_foot_contact_times"] == 0:
            self.extras["episode"]["sta_foot_contact_error"] = 0
        else:
            self.extras["episode"]["sta_foot_contact_error"] = self.info_statistics["sta_foot_contact_error_sum"] / self.info_statistics["sta_foot_contact_times"]
        for key in self.info_statistics.keys():
            self.info_statistics[key] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        wandb_rew_buf = dict()
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.reward_plot[name] = rew / self.dt
            wandb_rew_buf.update({name: torch.sum(rew) / self.num_envs})
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        wandb_rew_buf.update({"reward": torch.sum(self.rew_buf) / self.num_envs})
        if self.cfg.control.wandb_log:
            wandb.log(wandb_rew_buf)

    def _reset_base_states(self, env_ids):
        # self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat_mat[env_ids] = quaternion_to_matrix(self.base_quat[env_ids]).reshape(len(env_ids), 9)
        self.base_rpy[env_ids] = quaternion_to_euler(self.base_quat[env_ids])
        self.base_rpy_rate = self.GetTrueBaseRollPitchYawRate()
        self.last_base10[:, env_ids, :] = self.root_states[env_ids, :3]

    def _compute_torques(self, actions):
        if self.cfg.control.use_actuator_network:
            dof_pos, dof_vel = self.dof_pos, self.dof_vel
            # print("tar, pos, vel:", _actions[0][:3], dof_pos[0][:3], dof_vel[0][:3], sep="\n")
            torques = self.actuator_network.convert_to_torque(actions, dof_pos, dof_vel)
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def _init_buffers(self):
        super()._init_buffers()
        self.base_pos = self.root_states[:, :3]
        self.base_quat_mat = quaternion_to_matrix(self.base_quat).reshape(self.num_envs, 9)
        self.base_rpy = quaternion_to_euler(self.base_quat)
        self.base_rpy_rate = self.GetTrueBaseRollPitchYawRate()
        self.last_base10 = torch.tile(self.base_pos, (10, 1, 1))  # 10 * num_envs * xyz

        self.net_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.body_state = gymtorch.wrap_tensor(self.net_body_state).view(self.num_envs, -1, 13)  # shape: num_envs, num_bodies, (position, rotation, linear velocity, angular velocity).
        self.foot_state = self.body_state[:, self.feet_indices, :]
        self.foot_position_world = self.foot_state[..., :3]
        self.foot_velocity_world = self.foot_state[..., 7:10]
        self.foot_command_base = self.foot_command_rand(torch.arange(self.num_envs))  # num_envs * 4 * 3
        env_ids, foot_ids = torch.meshgrid(torch.arange(self.num_envs), torch.arange(4))
        env_ids, foot_ids = env_ids.flatten(), foot_ids.flatten()
        self.target_foot_hold = self.ComputeTargetPosInBase2WorldFrame(env_ids, foot_ids).reshape(self.num_envs, 4, 3)  # num_envs * 4 * 3
        self.foot_command = self.ComputeTargetPosInWorld2FootFrame(self.target_foot_hold)  # num_envs * 4 * 3
        self.last_foot_velocity = torch.zeros_like(self.foot_velocity_world)
        self.etg_time = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.contact_ok_num = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.last_contacts = torch.ones_like(self.last_contacts)  # 将legged_gym中的last_contacts初始值设为1
        self.real_contact = torch.ones(self.num_envs, 4, device=self.device, requires_grad=False)
        self.foot_contact_state = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.foothold = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        # self.history_action = torch.zeros(self.num_envs, 2, self.num_actions, device=self.device, requires_grad=False)
        self.energy = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.energy_sum = torch.zeros_like(self.energy)
        self.transport_cost = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.motor_power = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        self.reward_plot = {}
        self.init_info_statistics()

    def init_info_statistics(self):
        # all info you want to pass out
        self.info_statistics = {}
        self.info_statistics["sta_foot_contact_error_sum"] = 0
        self.info_statistics["sta_foot_contact_times"] = 0

    def foot_command_rand(self, env_ids):
        foot_dx = torch.rand(len(env_ids), 4, device=self.device, requires_grad=False) * 0.3
        # foot_dx = torch.ones(len(env_ids), 4, device=self.device, requires_grad=False) * 0.
        dy = torch.rand(len(env_ids), 2, device=self.device, requires_grad=False) * 0.2 - 0.1
        foot_dy = torch.hstack((dy, dy.flip(dims=[-1])))
        foot_dy = torch.zeros_like(foot_dy)
        foot_dz = torch.zeros(len(env_ids), 4, device=self.device, requires_grad=False)
        foot_command = torch.stack((foot_dx, foot_dy, foot_dz), dim=1).transpose(1, 2) + self.BASE_FOOT
        return foot_command

    def update_target_foot_hold(self):
        env_ids, foot_ids = (self.foot_contact_state == 3).nonzero(as_tuple=True)
        foot_world = self.foot_position_world[env_ids, foot_ids]
        foot_target = self.target_foot_hold[env_ids, foot_ids]
        foot_hold_error = torch.sum(torch.norm(foot_world - foot_target, dim=-1))
        self.info_statistics["sta_foot_contact_error_sum"] += foot_hold_error
        self.info_statistics["sta_foot_contact_times"] += len(env_ids)  # all contact foot in num_envs envs
        self.target_foot_hold[env_ids, foot_ids] = self.ComputeTargetPosInBase2WorldFrame(env_ids, foot_ids)  # update foot targets in world

    def ComputeTargetPosInBase2WorldFrame(self, env_ids, foot_ids):
        """Get the robot's foothold position in the world frame."""
        base_position, base_orientation = self.base_pos, self.base_quat
        foot_command = self.foot_command_base[env_ids, foot_ids]
        # Projects to world space.
        R_robot_in_world = quaternion_to_matrix(base_orientation[env_ids])  # foot_nums * 3 * 3
        P_robot_in_world_ori = base_position[env_ids]  # foot_nums * 3
        P_foot_in_robot = foot_command  # foot_nums * 4 * 3
        # world_link_pos = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        # for env_id in range(len(env_ids)):
        #     world_link_pos[env_id] = (torch.mm(R_robot_in_world[env_id], P_foot_in_robot.T) + P_robot_in_world_ori[env_id].unsqueeze(-1)).T  # envs_num * 4 * 3
        world_link_pos = (
                R_robot_in_world @ P_foot_in_robot.unsqueeze(-1)
                + P_robot_in_world_ori.unsqueeze(-1)
        ).transpose(1, 2)  # envs_num * 4 * 3
        world_link_pos[:, :, 2] = 0.

        return world_link_pos.squeeze(1)

    def ComputeTargetPosInWorld2FootFrame(self, target_foot_hold):
        base_position, base_orientation = self.base_pos, self.base_quat

        R_robot_in_world = quaternion_to_matrix(base_orientation)  # env_nums * 3 * 3
        P_robot_in_world_ori = base_position  # env_nums * 3
        foot_in_world = target_foot_hold  # env_nums * 4 * 3

        local_link_pos = (R_robot_in_world.inverse() @ (
                foot_in_world.transpose(1, 2) - P_robot_in_world_ori.unsqueeze(-1)
        )).transpose(1, 2) - self.BASE_FOOT
        return local_link_pos

    def ComputeFootPosInWorld2BaseFrame(self, world_foot):
        base_position, base_orientation = self.base_pos, self.base_quat

        R_robot_in_world = quaternion_to_matrix(base_orientation)  # env_nums * 3 * 3
        P_robot_in_world_ori = base_position  # env_nums * 3
        foot_in_world = world_foot  # env_nums * 4 * 3

        local_link_pos = (R_robot_in_world.inverse() @ (
                foot_in_world.transpose(1, 2) - P_robot_in_world_ori.unsqueeze(-1)
        )).transpose(1, 2)
        return local_link_pos

    def prepare_TG_table(self):
        self.act_ref = torch.as_tensor(np.array(self.ETG._action_table), device=self.device)

    def get_etg_actions(self, current_time):
        time_index = ((current_time % self.ETG.ETG_T) / self.ETG.dt).long()
        act_ref = self.act_ref[time_index]
        start_hip, start_thigh, start_calf = (
            self.cfg.init_state.start_hip,
            self.cfg.init_state.start_thigh,
            self.cfg.init_state.start_calf
        )
        _init_joint_pose = torch.tensor([start_hip, start_thigh, start_calf] * 4, device=self.device)
        _ETG_weight = 1.0
        _last_ETG_act = (act_ref - _init_joint_pose) * _ETG_weight + _init_joint_pose
        return _last_ETG_act

    def GetFootPositionsInBaseFrame(self):
        foot_world = self.foot_position_world
        foot_base = self.ComputeFootPosInWorld2BaseFrame(foot_world)
        return foot_base

    def GetTrueBaseRollPitchYawRate(self):
        return self.base_ang_vel

    def GetFootContacts(self):
        foot_contacts_force = self.contact_forces[:, self.feet_indices, 2]
        contacts = foot_contacts_force > 0.1
        return contacts

    def GetBadFootContacts(self):
        body_indices = [idx for idx in range(self.num_bodies) if idx not in self.feet_indices]
        bad_num = (torch.norm(self.contact_forces[:, body_indices, :], dim=-1) > 1.).count_nonzero(dim=-1)
        return bad_num

    def GetFootContactState(self):
        contacts = self.real_contact
        first_contact = torch.logical_xor(torch.logical_or(self.last_contacts, contacts), self.last_contacts)
        lift_contact = torch.logical_xor(torch.logical_or(self.last_contacts, contacts), contacts)
        contact_state = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        _x, _y = torch.nonzero(contacts, as_tuple=True)
        contact_state[_x, _y] = 1
        _x, _y = torch.nonzero(lift_contact, as_tuple=True)
        contact_state[_x, _y] = 3
        _x, _y = torch.nonzero(first_contact, as_tuple=True)
        contact_state[_x, _y] = 2

        # for e in range(self.num_envs):
        #     for i in range(contacts.shape[1]):
        #         if first_contact[e, i]:
        #             contact_state[e, i] = 2
        #         elif lift_contact[e, i]:
        #             contact_state[e, i] = 3
        #         elif contacts[e, i]:
        #             contact_state[e, i] = 1
        #         else:
        #             contact_state[e, i] = 0
        self.last_contacts[:] = contacts[:]
        return contact_state

    def GetEnergyConsumptionPerControlStep(self):
        self.energy = torch.sum(torch.abs(
            self.torques * self.dof_vel), dim=-1
        ) * self.dt * self.cfg.init_state.num_steps_per_policy

    def GetCostOfTransport(self):
        tv = self.torques * self.dof_vel
        tv[tv < 0] = 0
        self.transport_cost = tv.sum(1) / (torch.norm(self.base_lin_vel, dim=-1) * 20.0 * 9.8)

    def GetMotorPower(self):
        tv = self.torques * self.dof_vel
        tv[tv < 0] = 0
        self.motor_power = tv.sum(1) / 20.0

    def c_prec(self, v, t, m):
        # w = np.arctanh(np.sqrt(0.95)) / m  # 2.89 / m
        w = torch.sqrt(torch.arctanh(torch.tensor(0.95))) / m  # 1.35 / m  ???
        return torch.tanh(torch.pow((v - t) * w, 2))

    # todo: scale * dt

    def _reward_up(self):
        roll, pitch, _ = torch.unbind(self.base_rpy, -1)
        return 1 - 0.5 * self.c_prec(torch.abs(roll), 0, 0.25) - 0.5 * self.c_prec(torch.abs(pitch), 0, 0.25)

    def _reward_height(self):
        world_z = torch.mean(self.target_foot_hold[:, :, -1], dim=-1)
        r = torch.abs(self.root_states[:, 2] - world_z - 0.405)
        return 1 - self.c_prec(r, 0, 0.15)

    def _reward_feet_vel(self):
        contact_state = self.foot_contact_state
        contact_velocity = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        foot_velocity = self.foot_velocity_world
        rew_feet_vel = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        env_ids, foot_ids = (contact_state == 2).nonzero(as_tuple=True)  # first contact
        contact_velocity[env_ids, foot_ids] = torch.norm(self.last_foot_velocity[env_ids, foot_ids], dim=-1)
        rew_feet_vel[env_ids, foot_ids] += (-self.c_prec(contact_velocity[env_ids, foot_ids], 0.0, 5.0))
        self.last_foot_velocity[:] = foot_velocity[:]
        return rew_feet_vel.sum(dim=-1)

    def _reward_feet_pos(self):
        contact_state = self.foot_contact_state
        contact_position = self.foot_position_world.clone()
        feet_errors = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        feet_length_err = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        rew = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        rew_feet_length = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        env_ids, foot_ids = (contact_state == 3).nonzero(as_tuple=True)  # lift contact
        self.foothold[env_ids, foot_ids] = self.target_foot_hold[env_ids, foot_ids]

        env_ids, foot_ids = (contact_state == 2).nonzero(as_tuple=True)  # first contact
        contact_position[env_ids, foot_ids, 2] -= 0.02  # foot radius

        feet_errors[env_ids, foot_ids] = self.foothold[env_ids, foot_ids] - contact_position[env_ids, foot_ids]
        feet_length_err[env_ids, foot_ids] = torch.norm(feet_errors[env_ids, foot_ids], dim=-1)
        rew[env_ids, foot_ids] = (1 - self.c_prec(feet_length_err[env_ids, foot_ids], 0, 0.2))
        rew_feet_length[env_ids, foot_ids] += rew[env_ids, foot_ids]  # positive reward
        return rew_feet_length.sum(dim=-1)

    def _reward_action_rate(self):
        r1 = torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
        return 1 - self.c_prec(r1, 0, 0.2)

    def _reward_feet_airtime(self):
        # Reward long _steps
        contact_state = self.foot_contact_state
        self.feet_air_time += self.dt
        traj_period = self.ETG.ETG_T
        rew_airtime = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        # Punishing long duration of behavior
        airtime_err = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        feet_stand_long = torch.logical_or((self.feet_air_time % traj_period) < 1e-5, (traj_period - self.feet_air_time % traj_period) < 1e-5)
        env_ids, foot_ids = feet_stand_long.nonzero(as_tuple=True)
        rew_airtime[env_ids, foot_ids] += -1.

        env_ids, foot_ids = (contact_state == 2).nonzero(as_tuple=True)  # first contact
        airtime_err[env_ids, foot_ids] = torch.abs(self.feet_air_time[env_ids, foot_ids] - traj_period / 2)
        rew_airtime[env_ids, foot_ids] += (-self.c_prec(airtime_err[env_ids, foot_ids], 0, traj_period / 4))

        env_ids, foot_ids = torch.logical_or(contact_state == 2, contact_state == 3).nonzero(as_tuple=True)
        self.feet_air_time[env_ids, foot_ids] = 0

        return rew_airtime.sum(dim=-1)

    def _reward_feet_slip(self):
        contact = self.real_contact
        foot_vel = self.foot_velocity_world
        foot_vel = torch.norm(foot_vel, dim=-1)
        rew = torch.sum(foot_vel * contact, dim=-1)
        return -rew

    def _reward_tau(self):
        return -self.motor_power

    def _reward_badfoot(self):
        return -self.GetBadFootContacts()

    def _reward_footcontact(self):
        lose_contact_num = torch.sum(torch.logical_not(self.real_contact), dim=-1)
        return -torch.max(lose_contact_num - 2, torch.zeros_like(lose_contact_num))

    def _reward_done(self):
        done = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        ids = self.reset_buf.nonzero().flatten()
        done[ids] = -1.
        return done

    # reward not used
    # def _reward_velx(self):
    #     return self.base_lin_vel[:, 0]
    #
    # def _reward_contact_nums(self):
    #     feet_length_error = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     for i in range(self.num_envs):
    #         feet_length_error[i] = len(self.feet_length_error[i])
    #     return feet_length_error
    #
    # def _reward_contact_errs(self):
    #     contact_errs = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     for i in range(self.num_envs):
    #         if len(self.feet_length_error[i]) != 0:
    #             contact_errs[i] = np.mean(self.feet_length_error[i])
    #         else:
    #             contact_errs[i] = 1.
    #     return contact_errs
    #
    # def _reward_contact_rate(self):
    #     contact_rate = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     for i in range(self.num_envs):
    #         if len(self.feet_length_error[i]) != 0:
    #             contact_rate[i] = self.contact_ok_num[i] / len(self.feet_length_error[i])
    #         else:
    #             contact_rate[i] = 0.
    #     return contact_rate
    #
    # def _reward_energy_sum(self):
    #     return self.energy_sum
