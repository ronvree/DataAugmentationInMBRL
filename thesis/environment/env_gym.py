
import argparse
import cv2

from copy import deepcopy

import numpy as np

import torch

from thesis.environment.env import Environment
from thesis.environment.util import preprocess_observation
from thesis.util.func import batch_tensors


# A list of all supported Gym environment names

GYM_ENVS = [
    'Pendulum-v0',
    'MountainCarContinuous-v0',
    'Ant-v2',
    'HalfCheetah-v2',
    'Hopper-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2',
    'Reacher-v2',
    'Swimmer-v2',
    'Walker2d-v2',
]


class GymEnv(Environment):

    """
    Wrapper around OpenAI Gym environments

    Contains all dirty hacks needed to make OpenAI Gym compatible with planning
    """

    def __init__(self,
                 args: argparse.Namespace
                 ):
        assert args.env_batch_size > 0
        assert args.environment_name in GYM_ENVS
        assert args.max_episode_length > 0

        import gym

        # Wrap the Gym's Viewer constructor so the Viewer invisible
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(_self, *_args, **_kwargs):
            org_constructor(_self, *_args, **_kwargs)
            _self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor

        # Initialize the OpenAI Gym environments
        self._envs = [gym.make(args.environment_name) for _ in range(args.env_batch_size)]

        # Time step counter
        self._t = 0
        # Set time step limit
        self._max_t = args.max_episode_length
        # Check whether images or states should be observed
        self._state_obs = args.state_observations

        # Get bit depth for preprocessing the observations
        self._bit_depth = args.bit_depth
        # Get the size of observations
        self._observation_size = (32, 32) if args.downscale_observations else (64, 64)

    @property
    def batch_size(self):
        return len(self._envs)

    @property
    def t(self) -> int:
        return self._t

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
        # Set internal time step counter to 0
        self._t = 0
        # Get all initial observations from resetting the environment batch
        observations = [env.reset() for env in self._envs]
        # Create a flag tensor
        flags = torch.zeros(self.batch_size, dtype=torch.bool)
        # Create an info dict for each environment
        infos = tuple([dict() for _ in range(len(self._envs))])
        # Don't return an observation if no_observation flag is set
        if no_observation:
            return None, flags, infos
        elif not self._state_obs:
            # Get raw pixel observations of the environments
            pixels_tuple = self._pixels()
            # Process the image observations
            observations = [preprocess_observation(o, self._bit_depth, self._observation_size) for o in pixels_tuple]
            # observations = [self._process_image(o) for o in pixels_tuple]
            # Add raw pixels to the info dict
            for info, pixels in zip(infos, pixels_tuple):
                info['pixels'] = pixels

        # Cast all observations to tensors
        # observations = [torch.from_numpy(o).to(dtype=torch.float) for o in observations]
        # Concatenate all tensors in a newly created batch dimension
        # Results in a single observation tensor of shape: (batch_size,) + observation_shape
        observations = batch_tensors(*observations)  # TODO -- cast states to tensors!
        # Return the results
        return observations, flags, infos

    def get_state(self) -> tuple:
        """
        Get the environment state
        :return: a two-tuple containing the internal state of the environment. The tuple consists of
                    - environment step t
                    - a tuple containing all internal states of the environments, depending on the type of environment
        """
        return self._t, tuple(deepcopy(tuple(env.unwrapped.state for env in self._envs)))

    def set_state(self, state: tuple):
        """
        Set the internal state of the environment
        :param state: Two-tuple containing the internal state of the environment, consisting of:
                    - environment step t
                    - a tuple containing all internal states of the environments, depending on the type of environment
        """
        self._t, state = state
        for s, env in zip(state, self._envs):
            env.unwrapped.state = deepcopy(s)

    def get_seed(self):  # TODO -- why are openai seeds stored in lists??
        """
        :return: The environment seed
        """
        return tuple([env.seed()[0] for env in self._envs])

    def set_seed(self, seeds: tuple):
        """
        Set the environment seed
        :param seeds: The seed per environment in the batch
        """
        for env, seed in zip(self._envs, seeds):
            env.seed(seed)

    def clone(self) -> "Environment":
        """
        Get a deep copy of the environment (batch)
        :return: a deep copy of this environment (batch)
        """
        copy = deepcopy(self)
        return copy

    def step(self, action: torch.Tensor, no_observation: bool = False) -> tuple:
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
        # Increment the internal time step counter
        self._t += 1
        # Convert the tensor to suitable input
        action = action.detach().numpy()
        # Execute the actions in the environments
        results = [env.step(a) for a, env in zip(action, self._envs)]

        if no_observation:
            # Convert the results to tensors
            results = [(None,
                        torch.tensor(r, dtype=torch.float),
                        torch.tensor(f),
                        info)
                       for o, r, f, info in results]
            # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
            results = [*zip(*results)]
            # Merge all rewards to one tensor
            results[1] = batch_tensors(*results[1])
            # Merge all flags to one tensor
            results[2] = batch_tensors(*results[2])
            return None, results[1], results[2], results[3]

        # Check if an image observation should be made
        if not self._state_obs:
            # Get raw pixel observations of the environments
            pixels_tuple = self._pixels()
            # Convert them to suitable observations
            observations = [preprocess_observation(o, self._bit_depth, self._observation_size) for o in pixels_tuple]
            # Merge the observations in the results
            results = [(o,) + result[1:] for o, result in zip(observations, results)]

            # Add all raw pixel observations to the info dicts
            for result, pixels in zip(results, pixels_tuple):
                result[3]['pixels'] = pixels

        # Convert the results to tensors
        # results = [(torch.from_numpy(o).to(dtype=torch.float),
        results = [(o,
                    torch.tensor(r, dtype=torch.float),
                    torch.tensor(f),
                    info)
                   for o, r, f, info in results]

        # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = batch_tensors(*results[0])
        # Merge all rewards to one tensor
        results[1] = batch_tensors(*results[1])
        # Merge all flags to one tensor
        results[2] = batch_tensors(*results[2])

        # Check max episode length condition. Update flags if required
        if self._t >= self._max_t:
            results[2] |= True  # Set all flags to true if max episode length is reached

        # Return all results as a tuple
        return tuple(results)

    def close(self) -> tuple:
        """
        Close the environment
        :return: a dict possibly containing information
        """
        return tuple(env.close() for env in self._envs)

    def render(self, **kwargs):
        for env in self._envs:
            env.render(**kwargs)

    def _pixels(self) -> tuple:
        return tuple([env.render(mode='rgb_array').copy() for env in self._envs])

    def sample_random_action(self) -> torch.Tensor:
        """
        :return: a uniformly sampled random action from the action space
        """
        actions = [env.action_space.sample() for env in self._envs]
        actions = [torch.from_numpy(a) for a in actions]
        actions = batch_tensors(*actions)
        return actions

    @property
    def observation_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of observations
        """
        if self._state_obs:
            return self._envs[0].observation_space.shape
        else:
            return (3,) + self._observation_size

    @property
    def action_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of actions
        """
        return self._envs[0].action_space.shape


if __name__ == '__main__':

    _args = argparse.Namespace()
    _args.environment_name = GYM_ENVS[0]
    _args.env_batch_size = 3
    _args.max_episode_length = 1000
    _args.state_observations = False
    _args.bit_depth = 5

    _env = GymEnv(_args)

    for _ in range(3):
        _env.reset()
        for _ in range(250):
            _env.render(mode='rgb_array')
            _env.step(_env.sample_random_action())

    _env.close()





