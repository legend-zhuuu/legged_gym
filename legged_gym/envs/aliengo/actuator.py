import time
from collections import deque

import torch
import torch.nn as nn


class ActuatorNet(nn.Module):
    activation = nn.Softsign

    def __init__(
            self,
            input_dim=6,
            output_dim=1,
            hidden_dims=(32, 32, 32)
    ):
        super().__init__()
        layers = []
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.device = torch.device('cpu')

    def forward(self, state):
        return self.layers(state)

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)


class Actuator:
    activation = nn.Softsign

    def __init__(self, pt_path,
                 num_envs, num_actions=12,
                 input_dim=6,
                 output_dim=1,
                 hidden_dims=(32, 32, 32),
                 interval=5,
                 # num_envs,
                 # num_actions=12,
                 device="cuda:0",
                 torques_limit=40):
        self.torques_limit = torques_limit

        self.actuator = ActuatorNet(input_dim, output_dim, hidden_dims)
        self.device = torch.device(device)

        self.pt_path = pt_path
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.interval = interval
        self.dof_err_history = deque(
            (torch.zeros(self.num_envs, self.num_actions, device=self.device),) * (2 * interval + 1),
            maxlen=2 * interval + 1
        )
        self.dof_vel_history = deque(
            (torch.zeros(self.num_envs, self.num_actions, device=self.device),) * (2 * interval + 1),
            maxlen=2 * interval + 1
        )

        model = torch.load(self.pt_path, map_location=self.device)
        self.actuator.load_state_dict(model['model'])
        self.actuator.to(self.device)

    def convert_to_torque(self, actions, dof_pos, dof_vel):
        dof_err = actions - dof_pos
        self.dof_err_history.append(dof_err)
        self.dof_vel_history.append(dof_vel)
        with torch.inference_mode():
            input_tensor = torch.stack((
                self.dof_err_history[-1],
                self.dof_err_history[-1 - self.interval],
                self.dof_err_history[-1 - 2 * self.interval],
                self.dof_vel_history[-1],
                self.dof_vel_history[-1 - self.interval],
                self.dof_vel_history[-1 - 2 * self.interval],
            ), dim=-1)
            torques = self.actuator(input_tensor).squeeze(-1).to(self.device)
            torques = torch.clip(torques, -self.torques_limit, self.torques_limit)
        return torques


if __name__ == '__main__':
    act_net = Actuator('/home/zdy/legged_gym/resources/ETG/actuator_net.pt', 4)
