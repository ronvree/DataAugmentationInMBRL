

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.util.func import sample_normal


class StateModel(nn.Module):

    """
    Stochastic state model in the RSSM
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        # Gaussian with mean and variance parameterized by a neural network

        self.min_std = args.state_model_min_std

        model_size = args.state_model_size

        hidden_size = args.deterministic_state_size
        state_size = args.stochastic_state_size

        self.lin_in = nn.Linear(hidden_size, model_size)
        self.lin_mean = nn.Linear(model_size, state_size)
        self.lin_std = nn.Linear(model_size, state_size)

    def forward(self, hs: torch.Tensor) -> tuple:
        """
        Sample a prior state belief from a Gaussian distribution with mean and covariance parameterized by a feed-
        forward neural network. The network input is the hidden state of the transition model
        :param hs: a tensor containing the hidden state of the transition model of the RSSM
                    shape: (batch_size, deterministic_state_size)
        :return: a three-tuple consisting of:
                    - a state tensor sampled from the output distribution
                        shape: (batch_size, stochastic_state_size)
                    - a tensor containing the mean of the (normal) distribution from which the state was sampled
                        shape: (batch_size, stochastic_state_size)
                    - a tensor containing the st.dev. of the (normal) distribution from which the state was sampled
                        shape: (batch_size, stochastic_state_size)
        """
        xs = self.lin_in(hs)
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
    _args.state_model_size = 64
    _args.state_model_min_std = 0.1

    _model = StateModel(_args)

    _bs = 4

    _hs = torch.randn(_bs, _args.deterministic_state_size)

    _ss, _mean, _std = _model(_hs)

    print(_ss.shape)


