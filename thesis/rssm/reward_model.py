

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.util.func import sample_normal


class RewardModel(nn.Module):

    """
    Reward model component in the RSSM
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # Get model dimensions from args
        hs_size = args.deterministic_state_size
        ss_size = args.stochastic_state_size
        rm_size = args.reward_model_size

        # Define two hidden layers
        self.lin1 = nn.Linear(hs_size + ss_size, rm_size)
        self.lin2 = nn.Linear(rm_size, rm_size)
        self.lin3 = nn.Linear(rm_size, 1)

        self.sample_mean = args.rssm_sample_mean

    def forward(self, hs: torch.Tensor, ss: torch.Tensor) -> tuple:
        """
        Sample a reward from a normal distribution whose mean is parameterized by a dense neural network and with unit
        variance
        :param hs: a tensor containing the hidden state of the transition model of the RSSM
                    shape: (batch_size, deterministic_state_size)
        :param ss: a tensor containing the latent state of the state model of the RSSM
                    shape: (batch_size, stochastic_state_size)
        :return: a three-tuple consisting of:
                    - a reward tensor sampled from the output distribution
                        shape: (batch_size,)
                    - a tensor containing the mean of the (normal) distribution from which the reward was sampled
                        shape: (batch_size,)
                    - a tensor containing the st.dev. of the (normal) distribution from which the reward was sampled
                        shape: (batch_size,)
        """
        xs = torch.cat([hs, ss], dim=1)

        xs = self.lin1(xs)
        xs = F.relu(xs)

        xs = self.lin2(xs)
        xs = F.relu(xs)

        mean = self.lin3(xs).squeeze(dim=1)
        std = torch.ones_like(mean)

        if self.sample_mean:
            return mean, mean, std
        else:
            return sample_normal(mean, std), mean, std


if __name__ == '__main__':

    _args = argparse.Namespace()

    _args.deterministic_state_size = 50
    _args.stochastic_state_size = 50
    _args.reward_model_size = 64

    _model = RewardModel(_args)

    _bs = 4

    _hs = torch.randn(_bs, _args.deterministic_state_size)
    _ss = torch.randn(_bs, _args.stochastic_state_size)

    _rs, _mean, _std = _model(_hs, _ss)

    print(_rs.shape)

