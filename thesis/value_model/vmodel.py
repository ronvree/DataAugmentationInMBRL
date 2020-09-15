
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO -- version for state inputs?


class VModel(torch.nn.Module):

    def __init__(self, observation_shape: tuple, args: argparse.Namespace):
        super().__init__()
        c, w, h = observation_shape

        num_channels = 32  # TODO -- hyperparam
        num_lin_features = 32  # TODO -- hyperparam

        # Compute feature map dimensions
        f_map_w, f_map_h = w, h

        f_map_w = (f_map_w + 1) // 2
        f_map_h = (f_map_h + 1) // 2

        f_map_w = f_map_w // 2
        f_map_h = f_map_h // 2

        f_map_w = (f_map_w + 1) // 2
        f_map_h = (f_map_h + 1) // 2

        f_map_w = f_map_w // 2
        f_map_h = f_map_h // 2

        f_map_w = f_map_w // 2
        f_map_h = f_map_h // 2

        self.conv_1 = nn.Conv2d(
            in_channels=observation_shape[0],
            out_channels=num_channels,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.conv_2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.conv_3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.lin_1 = nn.Linear(f_map_h * f_map_w * num_channels,
                               num_lin_features,
                               )
        self.lin_2 = nn.Linear(num_lin_features,
                               1)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:

        pass  # TODO

    def forward(self, states) -> torch.Tensor:

        x = self.conv_1(states)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv_3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.flatten(start_dim=1)

        x = self.lin_1(x)
        x = F.relu(x)

        v = self.lin_2(x)

        return v.squeeze()


if __name__ == '__main__':

    _args = argparse.Namespace()

    _obs_shape = (3, 500, 500)

    _obs = torch.randn(1, *_obs_shape)

    _model = VModel(_obs_shape, _args)

    _v = _model(_obs)

    print(_v)
