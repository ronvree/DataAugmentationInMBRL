
import argparse

import torch


class DummyModel(torch.nn.Module):

    def __init__(self, observation_shape: tuple, args: argparse.Namespace):
        super().__init__()

        self.ws = torch.nn.Parameter(
            torch.randn(*observation_shape),
            requires_grad=True
        )

    def forward(self, s):
        return torch.nn.functional.sigmoid(torch.sum(s * self.ws, dim=[1, 2]))
