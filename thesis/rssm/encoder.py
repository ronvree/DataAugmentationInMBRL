
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.util.func import sample_normal


class BeliefEncoder(nn.Module):

    """
    Belief encoder component of the RSSM

    Supposed to act like a Kalman filter in a non-linear setting. That is, it updates the state belief distribution
    prior to the observation to a posterior state belief distribution upon receiving an observation of the true
    environment
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        encoding_size = args.encoding_size

        hidden_size = args.deterministic_state_size
        latent_size = args.stochastic_state_size
        encoder_size = args.encoder_model_size

        self.min_std = args.state_model_min_std

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.lin_os = nn.Linear(1024, encoding_size) if encoding_size != 1024 else nn.Identity()
        self.lin_embed = nn.Linear(hidden_size + encoding_size, encoder_size)

        self.lin_mean = nn.Linear(encoder_size, latent_size)
        self.lin_std = nn.Linear(encoder_size, latent_size)

    def forward(self, hs: torch.Tensor, os: torch.Tensor) -> tuple:
        """

        :param hs: a tensor containing the hidden state of the transition model of the RSSM
            shape: (batch_size, deterministic_state_size)
        :param os: an observation tensor based on which the state belief distribution should be updated
            shape: (batch_size,) + observation_shape
        :return: a three-tuple consisting of
                    - a state tensor containing a sample obtained from the posterior state distribution
                        shape: (batch_size, stochastic_state_size)
                    - a mean tensor parameterizing the posterior (Gaussian) state distribution
                        shape: (batch_size, stochastic_state_size)
                    - a std tensor parameterizing the posterior (Gaussian) state distribution
                        shape: (batch_size, stochastic_state_size)
        """
        # Get the batch size
        bs = hs.size(0)

        # Pass the observation through the conv net
        xs = self.conv1(os)
        xs = F.relu(xs)

        xs = self.conv2(xs)
        xs = F.relu(xs)

        xs = self.conv3(xs)
        xs = F.relu(xs)

        xs = self.conv4(xs)
        xs = F.relu(xs)

        xs = xs.view(bs, -1)
        xs = self.lin_os(xs)

        xs = torch.cat([xs, hs], dim=1)

        xs = self.lin_embed(xs)
        xs = F.relu(xs)

        mean = self.lin_mean(xs)
        std = self.lin_std(xs)
        std = F.softplus(std) + self.min_std

        state = sample_normal(mean, std)

        return state, mean, std


if __name__ == '__main__':

    _args = argparse.Namespace()

    _args.deterministic_state_size = 50
    _args.stochastic_state_size = 50
    _args.encoding_size = 64
    _args.encoder_model_size = 200
    _args.state_model_min_std = 0.1

    _model = BeliefEncoder(_args)

    _bs = 4
    _observation_shape = (3, 64, 64)

    _hs = torch.randn(_bs, _args.deterministic_state_size)
    _os = torch.randn(_bs, *_observation_shape)

    _ss, _mean, _std = _model(_hs, _os)

    print(_ss.shape)






