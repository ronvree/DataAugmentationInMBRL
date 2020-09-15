

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.util.func import batch_tensors, sample_normal


class ObservationModel(nn.Module):

    """
    Observation model component of the RSSM

    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # Get model dimensions from args
        hs_size = args.deterministic_state_size
        ss_size = args.stochastic_state_size
        encoding_size = args.encoding_size

        self.lin = nn.Linear(hs_size + ss_size, encoding_size)
        self.conv1 = nn.ConvTranspose2d(encoding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

        self.sample_mean = args.rssm_sample_mean

    @staticmethod
    def observation_shape() -> tuple:
        return 3, 64, 64

    def forward(self, hs: torch.Tensor, ss: torch.Tensor) -> tuple:
        """
        Sample an observation from a multivariate Gaussian with mean parameterized by a deconvolutional neural network
        and identity covariance
        :param hs: a tensor containing the hidden state of the transition model of the RSSM
                    shape: (batch_size, deterministic_state_size)
        :param ss: a tensor containing the latent state of the state model of the RSSM
                    shape: (batch_size, stochastic_state_size)
        :return: a three-tuple consisting of:
                    - an observation tensor sampled from the output distribution
                        shape: (batch_size,) + observation_shape
                    - a tensor containing the mean of the (normal) distribution from which the observation was sampled
                        shape: (batch_size,) + observation_shape
                    - a tensor containing the st.dev. of the (normal) distribution from which the observation was sampled
                        shape: (batch_size,) + observation_shape
        """
        xs = torch.cat([hs, ss], dim=1)

        xs = self.lin(xs)

        xs = xs.view(-1, xs.size(1), 1, 1)

        xs = self.conv1(xs)
        xs = F.relu(xs)

        xs = self.conv2(xs)
        xs = F.relu(xs)

        xs = self.conv3(xs)
        xs = F.relu(xs)

        mean = self.conv4(xs)

        std = torch.ones_like(mean).to(mean.device)

        if self.sample_mean:
            return mean, mean, std
        else:
            return sample_normal(mean, std), mean, std


if __name__ == '__main__':

    _args = argparse.Namespace()

    _args.deterministic_state_size = 50
    _args.stochastic_state_size = 50
    _args.encoding_size = 1024

    _model = ObservationModel(_args)

    _bs = 4

    _hs = torch.randn(_bs, _args.deterministic_state_size)
    _ss = torch.randn(_bs, _args.stochastic_state_size)

    _os, _mean, _std = _model(_hs, _ss)

    print(_os.shape)


