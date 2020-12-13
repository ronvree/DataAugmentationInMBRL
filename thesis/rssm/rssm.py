
import argparse

import torch
import torch.nn as nn

from thesis.environment.env import Environment

from thesis.rssm.reward_model import RewardModel
from thesis.rssm.observation_model import ObservationModel
from thesis.rssm.encoder import BeliefEncoder
from thesis.rssm.state_model import StateModel
from thesis.rssm.transition_model import TransitionModel


# TODO -- elu activation


class RSSM(Environment, nn.Module):

    """
        Recurrent State Space Model
    """

    def __init__(self,
                 action_shape: tuple,
                 args: argparse.Namespace,
                 device=torch.device('cpu')
                 ):
        super().__init__()
        assert len(action_shape) == 1

        # Store constants
        self._action_shape = action_shape

        # Build initial state
        self._hidden_state, self._belief_state = self._init_state(device, args)

        # Default batch size when resetting internal state
        self._default_batch_size = args.env_batch_size

        # Build model components
        self.reward_model = RewardModel(args)
        self.observation_model = ObservationModel(args)
        self.transition_model = TransitionModel(action_shape, args)
        self.encoder = BeliefEncoder(args)

        # Keep internal step counter
        self._t = 0

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('Recurrent State Space Model Arguments')

        parser.add_argument('--deterministic_state_size',
                            type=int,
                            default=200,  # 200 in original paper
                            help='Size of the hidden (deterministic) states in the RSSM'
                            )
        parser.add_argument('--stochastic_state_size',
                            type=int,
                            default=30,  # 30 in original paper
                            help='Size of the (stochastic) latent state in the RSSM')

        parser.add_argument('--reward_model_size',
                            type=int,
                            default=200,  # 200 in original paper
                            help='Size of the dense layers in the reward model')
        parser.add_argument('--state_model_size',
                            type=int,
                            default=200,  # 200 in original paper
                            help='Size of the dense layers in the state model')
        parser.add_argument('--encoder_model_size',
                            type=int,
                            default=200,  # 200 in original paper
                            help='Size of the dense layer in the encoder model')
        parser.add_argument('--state_model_min_std',
                            type=float,
                            default=0.1,  # 0.1 in original paper
                            help='Minimal standard deviation of the stochastic state model')
        parser.add_argument('--encoding_size',
                            type=int,
                            default=1024,  # 1024 in original paper
                            help='The size of the encoding of the history')
        parser.add_argument('--rssm_sample_mean',
                            action='store_true',
                            help='Flag that, when set, lets the RSSMs reward and observation model always return the'
                                 'mean rather than a sample of a Gaussian with mean and unit variance')

        return parser

    def _init_state(self,
                    device,
                    args: argparse.Namespace) -> tuple:
        """
        Initialize the internal state of the RSSM
        :param device: the device on which the state tensors should be initialized
        :param args: parsed arguments containing hyperparameters
        :return: a two-tuple consisting of:
                - a hidden state tensor
                    shape: (env_batch_size, deterministic_state_size)
                - a belief state tensor
                    shape: (env_batch_size, stochastic_state_size)
        """
        # State tensors are initialized as all zeroes
        hidden = torch.zeros(args.env_batch_size,
                             args.deterministic_state_size).to(device)
        belief = torch.zeros(args.env_batch_size,
                             args.stochastic_state_size).to(device)
        # Set state
        self._hidden_state, self._belief_state = hidden, belief

        return hidden, belief

    def _reset_state(self, batch_size: int = -1):
        """
        Reset the internal state of the RSSM

        The shape and device of the new state tensors are based on the old tensors
        The batch size can be overwritten by explicitly passing it as an argument. When set to -1 (by default), the
        batch size of the current internal state is taken

        States are initialized as all zeros

        :param batch_size: The batch size of the internal RSSM state
        """
        # Get which device is used
        device = self._hidden_state.device

        # Set a different batch size if required. Otherwise take default
        batch_size = self._default_batch_size if batch_size == -1 else batch_size

        # Get sizes of hidden state and belief state
        hidden_state_size = self._hidden_state.size(1)
        belief_state_size = self._belief_state.size(1)

        # Set state to all zeros of correct shape and on correct device
        self._hidden_state = torch.zeros(batch_size, hidden_state_size).to(device)
        self._belief_state = torch.zeros(batch_size, belief_state_size).to(device)

    def to(self, *args, **kwargs):  # Override .to function so internal state is also moved to correct device
        self._hidden_state = self._hidden_state.to(*args, **kwargs)
        self._belief_state = self._belief_state.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @property
    def t(self) -> int:
        return self._t

    @property
    def state_model(self) -> StateModel:
        return self.transition_model.state_model

    def forward(self,
                actions: torch.Tensor,
                no_observation: bool = False,
                true_observation: torch.Tensor = None,
                ) -> tuple:
        """
        Perform a forward pass through the RSSM model
        :param actions: the input action tensor
                        shape: (batch_size, action_size)
        :param no_observation: when set to true, no observation is generated during this forward pass. During planning
                               no observations are used, so this speeds up the planning process
        :param true_observation: if the true observation is known, these can be given to the encoder to update the
                                 state belief distribution. When there is no true observation, the belief prior is used
        :return: a six-tuple consisting of:
            - a (observation dist) three-tuple consisting of:
                - a sample observation tensor of the observation distribution obtained from the observation model
                    shape: (batch_size,) + observation_shape
                - a mean tensor parameterizing the Gaussian distribution from which the observation was sampled
                    shape: (batch_size,) + observation_shape
                - a std tensor parameterizing the Gaussian distribution from which the observation was sampled
                    shape: (batch_size,) + observation_shape
            - a (reward dist) three-tuple consisting of:
                - a sample reward tensor of the reward distribution obtained from the reward model
                    shape: (batch_size,)
                - a mean tensor parameterizing the Gaussian distribution from which the reward was sampled
                    shape: (batch_size,)
                - a std tensor parameterizing the Gaussian distribution from which the reward was sampled
                    shape: (batch_size,)
            - a hidden (deterministic) state tensor of the state transition model
                shape: (batch_size, deterministic_state_size)
            - a (current internal belief dist) three-tuple consisting of:
                - a sample state tensor of the belief distribution obtained from the state transition model
                    shape: (batch_size, stochastic_state_size)
                - a mean tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                - a std tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                This will be the posterior distribution if a true observation was given and the prior distribution
                otherwise
            - a (prior belief dist) three-tuple consisting of:
                - a sample state tensor of the belief distribution obtained from the state transition model
                    shape: (batch_size, stochastic_state_size)
                - a mean tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                - a std tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
            - a (posterior belief dist) three-tuple consisting of:
                - a sample state tensor of the belief distribution obtained from the state transition model
                    shape: (batch_size, stochastic_state_size)
                - a mean tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                - a std tensor parameterizing the Gaussian distribution from which the state was sampled
                    shape: (batch_size, stochastic_state_size)
                If no true observation was given, no posterior could be computed. All tensors in this tuple are set to
                None
        """
        # Move the actions tensor to the right device
        actions = actions.to(device=self._hidden_state.device)

        # Pass the action through the transition model
        hidden, belief_prior, belief_prior_mean, belief_prior_std = self.transition_model(self._hidden_state,
                                                                                          self._belief_state,
                                                                                          actions
                                                                                          )
        prior = belief_prior, belief_prior_mean, belief_prior_std

        # If true observation is known, use this to update belief distribution
        if true_observation is not None:
            true_observation = true_observation.to(device=self._hidden_state.device)
            belief_post, belief_post_mean, belief_post_std = self.posterior_state_belief(true_observation,
                                                                                         no_state_update=True
                                                                                         )
            posterior = belief_post, belief_post_mean, belief_post_std

            belief, belief_mean, belief_std = posterior
        else:
            belief, belief_mean, belief_std = prior  # TODO -- neater code
            posterior = (None, None, None)

        # Get a reward based on the reward model
        reward, reward_mean, reward_std = self.reward_model(hidden, belief)

        # If required, get an observation based on the observation model
        if not no_observation:
            observation, observation_mean, observation_std = self.observation_model(hidden, belief)
        else:
            observation, observation_mean, observation_std = None, None, None

        return (observation, observation_mean, observation_std),\
               (reward, reward_mean, reward_std),\
               hidden,\
               (belief, belief_mean, belief_std),\
               prior,\
               posterior

    def reset(self,
              no_observation: bool = False,
              batch_size: int = -1
              ) -> tuple:  # TODO -- pass initial observation?
        """

        :param no_observation:
        :param batch_size:
        :return:
        """

        # Reset internal state
        self._reset_state(batch_size=batch_size)
        self._t = 0

        # Get the actual batch size used
        batch_size = self._hidden_state.size(0)

        # Check if an observation is required
        if no_observation:
            observation = None
        else:
            # Use the observation model to create an initial observation
            observation, _, _ = self.observation_model(self._hidden_state, self._belief_state)

        # Create a flag tensor
        flags = torch.zeros(batch_size, dtype=torch.bool)

        return observation, flags, {}  # Empty info dict

    def get_state(self) -> tuple:
        """
        Get the internal state of this RSSM
        :return: a two-tuple containing:
                - t: the internal step counter
                - a two-tuple containing:
                    - a hidden state tensor
                        shape: (batch_size, deterministic_state_size)
                    - a belief state tensor
                        shape: (batch_size, stochastic_state_size)
        """
        return self._t, (self._hidden_state, self._belief_state)

    def set_state(self, state: tuple):
        """
        Set the internal state of this RSSM
        :param state: a two-tuple containing:
                        - t: the internal step counter
                        - a two-tuple containing:
                            - a hidden state tensor
                                shape: (batch_size, deterministic_state_size)
                            - a belief state tensor
                                shape: (batch_size, stochastic_state_size)
        """
        t, state = state
        assert t >= 0
        assert len(state) == 2

        self._t = t
        self._hidden_state, self._belief_state = state

        assert self._hidden_state.size(0) == self._belief_state.size(0)

    def get_seed(self) -> tuple:
        raise NotImplementedError

    def set_seed(self, seed: tuple):
        raise NotImplementedError

    def clone(self) -> "RSSM":
        return self

    def step(self,
             action: torch.Tensor,
             no_observation: bool = False,
             no_state_update: bool = False,
             true_observation: torch.Tensor = None,
             ) -> tuple:
        """
        Simulate a step in the learned environment model. Return the results as if this were the true environment
        :param action: The action tensor to apply to the environment
                    shape: (batch_size, action_size)
        :param no_observation: A flag that, when set, indicates that no observation needs to be generated.
                               By default set to False
        :param no_state_update: A flag that, when set, indicates that the internal state of the RSSM should not be
                                updated. By default set to False
        :param true_observation: If the true observation tensor is known, this can be passed to the belief encoder s.t.
                                 the belief distribution can be updated
                                shape: (batch_size,) + observation_shape
        :return: a four-tuple consisting of:
                - a (float) observation tensor  (None if no_observation is set to True)
                    shape: (batch_size,) + observation_shape
                - a (float) reward tensor
                    shape: (batch_size,)
                - a (bool) tensor containing flags indicating whether the environment has terminated
                    shape: (batch_size,)
                    For the RSSM these are always set to False
                - info dict possibly containing info about the step
        """
        # Simulate the MDP in the RSSM
        observation_dist, reward_dist, belief_dist, flags, info = self.simulate_step(action,
                                                                                     no_observation=no_observation,
                                                                                     no_state_update=no_state_update,
                                                                                     true_observation=true_observation,
                                                                                     )
        # Select the relevant information from the simulation step
        observation, _, _ = observation_dist
        reward, _, _ = reward_dist
        # Return the predicted observations and rewards
        return observation, reward, flags, {}

    def simulate_step(self,
                      action: torch.Tensor,
                      no_observation: bool = False,
                      no_state_update: bool = False,
                      true_observation: torch.Tensor = None
                      ) -> tuple:
        """
        Simulate a step in the learned environment model. Return all available information
        :param action: The action tensor to apply to the environment
                    shape: (batch_size, action_size)
        :param no_observation: A flag that, when set, indicates that no observation needs to be generated.
                               By default set to False
        :param no_state_update: A flag that, when set, indicates that the internal state of the RSSM should not be
                                updated. By default set to False
        :param true_observation: If the true observation tensor is known, this can be passed to the belief encoder s.t.
                                 the belief distribution can be updated
                                shape: (batch_size,) + observation_shape
        :return: a five-tuple consisting of:

        """

        # Get the batch size
        batch_size = self._hidden_state.size(0)
        # Get the current device
        device = self._hidden_state.device

        # Pass the action through the environment model
        observation_dist, reward_dist, hidden, belief_dist, prior, posterior = self(action,
                                                                                    no_observation=no_observation,
                                                                                    true_observation=true_observation,
                                                                                    )

        # Create a flag tensor
        flags = torch.zeros(batch_size, dtype=torch.bool, device=device)  # RSSM never reaches terminal state

        # Set the internal state if not disabled
        if not no_state_update:
            belief, _, _ = belief_dist
            self._hidden_state, self._belief_state = hidden, belief

        # Increment internal counter
        self._t += 1

        return observation_dist, reward_dist, belief_dist, flags, {}

    def close(self) -> tuple:
        pass  # Nothing to do here

    def sample_random_action(self) -> torch.Tensor:
        """
        Sample an action tensor with random values. May not support entire action space -- only use for testing!
        """
        return torch.randn(self._hidden_state.size(0), *self._action_shape)

    def posterior_state_belief(self,
                               observation: torch.Tensor,
                               no_state_update: bool = False) -> tuple:

        state, mean, std = self.encoder(self._hidden_state, observation)

        if not no_state_update:
            self._belief_state = state

        return state, mean, std

    @property
    def observation_shape(self) -> tuple:
        """
        Get the shape of observation tensors obtained from this model
        """
        return self.observation_model.observation_shape

    @property
    def action_shape(self) -> tuple:
        """
        Get the shape of action tensors that can be performed in this model
        """
        return self._action_shape


if __name__ == '__main__':

    _args = argparse.Namespace()

    _args.env_batch_size = 1

    _args.deterministic_state_size = 50
    _args.stochastic_state_size = 50
    _args.reward_model_size = 200
    _args.encoder_model_size = 200
    _args.encoding_size = 60
    _args.state_model_size = 64
    _args.state_model_min_std = 0.1

    _action_shape = (6,)

    _model = RSSM(_action_shape, _args)

    _bs = _args.env_batch_size

    _as = torch.randn(_bs, *_action_shape)

    _os, _rs, _fs, _ = _model.step(_as)

    print(_os.shape)
    print(_rs.shape)
    print(_fs.shape)

    _os = torch.randn(*_os.shape)

    _ss = _model.posterior_state_belief(_os)

    print(_ss.shape)



