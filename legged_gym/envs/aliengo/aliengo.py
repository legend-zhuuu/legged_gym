import os
import sys
import torch
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
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

    def check_termination(self):
        super().check_termination()
        # z < 0.2 reset
        actor_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(actor_state)
        z_limit_buf = root_states[:, 2] < 0.2
        self.reset_buf |= z_limit_buf
