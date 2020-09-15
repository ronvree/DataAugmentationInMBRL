import argparse

import numpy as np

import torch


class Environment:

    """
    Abstract class for defining an environment

    Functions are required to support a batch (of possibly fixed size) of arguments applied to a batch of individual
    environments of the same type

    """

    # TODO -- warning when complete observability assumption is used in get_state!

    def reset(self, no_observation: bool = False) -> tuple:
        """
        Set the batch of environments to their initial states
        :param no_observation: When set to True, the reset function does not return an observation (None)
                       This option exists, since the planner does not require observations, but rendering slows
                       it down a lot
        :return: a 3-tuple consisting of:
                    - a torch.Tensor containing a batch of initial observations
                      Shape: (batch_size,) + observation_shape
                    - a torch.Tensor containing a batch of boolean flags indicating whether the environments terminated
                    - a tuple of dicts possibly containing additional information for each environment
        """
        raise NotImplementedError

    def get_state(self) -> tuple:
        """
        Get the environment state
        :return: a two-tuple containing the internal state of the environment. The tuple consists of
                    - environment step t
                    - a tuple containing all internal states of the environments, depending on the type of environment
        """
        raise NotImplementedError

    def set_state(self, state: tuple):
        """
        Set the internal state of the environment
        :param state: Two-tuple containing the internal state of the environment, consisting of:
                    - environment step t
                    - a tuple containing all internal states of the environments, depending on the type of environment
        """
        raise NotImplementedError

    def get_seed(self) -> tuple:
        """
        :return: The environment seed
        """
        raise NotImplementedError

    def set_seed(self, seed: tuple):
        """
        Set the environment seed
        :param seed: The seed
        """
        raise NotImplementedError

    def clone(self) -> "Environment":
        """
        Get a deep copy of the environment (batch)
        :return: a deep copy of this environment (batch)
        """
        raise NotImplementedError

    def step(self,
             action: torch.Tensor,
             no_observation: bool = False,
             ) -> tuple:
        """
        Perform an action in the environment. Returns a reward and observation
        :param action: a Tensor representation of the action that should be performed in the environment
                        Shape: (batch_size,) + action_shape
        :param no_observation: When set to True, the step function does not return an observation (None)
                               This option exists, since the planner does not require observations, but rendering slows
                               it down a lot
        :return: a 4-tuple consisting of:
                    - a torch.Tensor observation
                      Shape: (batch_size,) + observation_shape
                    - a torch.Tensor reward
                      Shape: (batch_size,)
                    - a torch.Tensor boolean flag indicating whether the environment has terminated
                      Shape: (batch_size,)
                    - a tuple of dicts possibly containing additional information
        """
        raise NotImplementedError

    def close(self) -> tuple:
        """
        Close the environment
        :return: a dict possibly containing information
        """
        raise NotImplementedError

    def sample_random_action(self) -> torch.Tensor:
        """
        :return: a uniformly sampled random action from the action space
        """
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of observations
        """
        raise NotImplementedError

    @property
    def action_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of actions
        """
        raise NotImplementedError

    @property
    def t(self) -> int:
        """
        :return: the number of time steps simulated in this environment
        """
        raise NotImplementedError

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('Environment Arguments')

        parser.add_argument('--environment_name',
                            type=str,
                            default='Pendulum-v0',
                            help='The name of the environment in which the agent should be run')
        parser.add_argument('--max_episode_length',
                            type=int,
                            default=np.inf,
                            help='Max number of steps an episode can run before the environment terminates')
        parser.add_argument('--state_observations',
                            action='store_true',
                            help='A flag that controls whether images or internal states serve as observations')
        parser.add_argument('--env_batch_size',
                            type=int,
                            default=1,
                            help='The number of environments that are used simultaneously for data collection')
        parser.add_argument('--bit_depth',
                            type=int,
                            default=5,  # Value used in paper: 5
                            help='Bit depth when preprocessing the observations')

        return parser

