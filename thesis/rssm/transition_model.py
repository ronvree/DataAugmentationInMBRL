
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.rssm.state_model import StateModel


class TransitionModel(nn.Module):

    """
    State transition component of the RSSM

    Contains a recurrent neural network whose internal state represents a deterministic component of the latent state
    A dense neural network is used to parameterize a Gaussian distribution which is sampled for a stochastic component
    of the latent state
    """

    def __init__(self, action_shape: tuple, args: argparse.Namespace):
        super().__init__()
        assert len(action_shape) == 1

        hidden_size = args.deterministic_state_size
        latent_size = args.stochastic_state_size
        action_size = action_shape[-1]

        self.lin = nn.Linear(latent_size + action_size,
                             hidden_size)

        self.rnn = nn.GRUCell(input_size=hidden_size,
                              hidden_size=hidden_size)

        self.state_model = StateModel(args)

    def forward(self,
                hidden: torch.Tensor,
                state: torch.Tensor,
                action: torch.Tensor,
                ) -> tuple:
        """

        :param hidden: a tensor containing the hidden state of the transition model of the RSSM
                        shape: (batch_size, deterministic_state_size)
        :param state: a tensor containing the latent state of the state model of the RSSM
                        shape: (batch_size, stochastic_state_size)
        :param action: a tensor containing the action applied to the RSSM state
                        shape: (batch_size, action_size)
        :return: a four-tuple containing
                - a (deterministic) state tensor containing the hidden state of the recurrent neural network
                    shape: (batch_size, deterministic_state_size)
                - a (stochastic) state tensor sampled from the output distribution
                    shape: (batch_size, stochastic_state_size)
                - a tensor containing the mean of the (normal) distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                - a tensor containing the st.dev. of the (normal) distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
        """
        # Embed the state and action tensors into one tensor using a dense layer
        embedding = torch.cat([state, action], dim=1)
        embedding = self.lin(embedding)
        embedding = F.relu(embedding)
        # Pass the embedding through the RNN
        hidden = self.rnn(embedding, hidden)
        # Use the resulting RNN state to parameterize a stochastic state model and sample a state
        state_prior, mean, std = self.state_model(hidden)

        return hidden, state_prior, mean, std


if __name__ == '__main__':

    _args = argparse.Namespace()

    _args.deterministic_state_size = 50
    _args.stochastic_state_size = 50
    _args.state_model_size = 64
    _args.state_model_min_std = 0.1

    _action_shape = (6,)

    _model = TransitionModel(_action_shape, _args)

    _bs = 4

    _hs = torch.randn(_bs, _args.deterministic_state_size)
    _ss = torch.randn(_bs, _args.stochastic_state_size)
    _as = torch.randn(_bs, *_action_shape)

    _hs, _ss, _mean, _std = _model(_hs, _ss, _as)

    print(_hs.shape)
    print(_ss.shape)












