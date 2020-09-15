
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


# TODO

class QModel(nn.Module):

    # OPENAI image observation shape: 500 500 3

    def __init__(self, observation_shape: tuple, action_shape: tuple, args: argparse.Namespace):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            stride=2
        )

        self.conv_2 = nn.Conv2d(
            in_channels=observation_shape[-1],
            out_channels=32,
            kernel_size=5,
            stride=2
        )

        self.conv_3 = nn.Conv2d(
            in_channels=observation_shape[-1],
            out_channels=32,
            kernel_size=5,
            stride=2
        )

        # self.lin = nn.Linear()

        # self.lin1 = Linear(int(np.prod(observation_shape)), 16)
        # self.lin2 = Linear(int(np.prod(action_shape)), 16)
        # self.lin3 = Linear(32, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(state)

        return x.view(state.size(0), -1).sum(dim=1)

        #
        # s = F.relu(self.lin1(state))
        # a = F.relu(self.lin2(action))
        #
        # sa = torch.cat((s, a), dim=1)
        # out = self.lin3(sa)
        #
        # return out.squeeze()



