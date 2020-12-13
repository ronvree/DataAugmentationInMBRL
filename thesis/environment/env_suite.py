import argparse

from copy import deepcopy

import numpy as np

import torch

from thesis.environment.env import Environment
from thesis.environment.util import preprocess_observation
from thesis.util.func import batch_tensors


# List of all supported control suite environments names
CONTROL_SUITE_ENVS = [
    'cartpole-balance',
    'cartpole-swingup',
    'reacher-easy',
    'finger-spin',
    'cheetah-run',
    'ball_in_cup-catch',
    'walker-walk',
    'acrobot-swingup'
]


class ControlSuiteEnvironment(Environment):

    """
    Wrapper around Control Suite environments

    https://github.com/deepmind/dm_control
    """

    # TODO -- docs

    def __init__(self, args: argparse.Namespace):
        assert args.env_batch_size > 0
        assert args.environment_name in CONTROL_SUITE_ENVS
        assert args.max_episode_length > 0

        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        domain, task = args.environment_name.split('-')

        self._envs = [suite.load(domain_name=domain,
                                 task_name=task,
                                 task_kwargs={'time_limit': np.inf}
                                 )
                      for _ in range(args.env_batch_size)]

        self._envs = [pixels.Wrapper(env,
                                     render_kwargs={'camera_id': 0}
                                     ) for env in self._envs]

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

    @property
    def observation_shape(self) -> tuple:
        return 3, 64, 64

    @property
    def action_shape(self) -> tuple:
        return self._envs[0].action_spec().shape

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
        # Reset all environments
        results = [env.reset() for env in self._envs]
        # Process the control suite results
        results = [self._process_result(result) for result in results]
        # Filter the non-existent rewards
        results = [(o, t, info) for o, r, t, info in results]

        # Don't return an observation if no_observation flag is set
        if no_observation:
            # Unzip all tuples into 3 tuples containing the observations, flags and info dicts, respectively
            results = [*zip(*results)]
            # No observation is returned
            results[0] = None
            # Merge all flags to one tensor
            results[1] = batch_tensors(*results[1])
            # Return all results as a tuple
            return tuple(results)
        # If required, set observation to image observations
        elif not self._state_obs:
            # Get raw pixel observations
            pixels_tuple = self._pixels()
            # Preprocess all observations
            results = [(preprocess_observation(image, self._bit_depth, self._observation_size), t, info)
                       for image, (_, t, info) in zip(pixels_tuple, results)]

            # Add raw pixels to all info dicts
            for pixels, (_, _, info) in zip(pixels_tuple, results):
                info['pixels'] = pixels

        # Unzip all tuples into 3 tuples containing the observations, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = batch_tensors(*results[0])
        # Merge all flags to one tensor
        results[1] = batch_tensors(*results[1])

        # Return all results as a tuple
        return tuple(results)

    def get_state(self) -> tuple:
        states = []
        for env in self._envs:
            states.append(
                (
                    np.array(env.physics.data.qpos),
                    np.array(env.physics.data.qvel),
                    np.array(env.physics.data.ctrl),
                )
            )
        return self._t, tuple(states)

    def set_state(self, state: tuple):
        self._t, state = state
        for env, (pos, vel, ctrl) in zip(self._envs, state):
            with env.physics.reset_context():
                env.physics.data.qpos[:] = pos
                env.physics.data.qvel[:] = vel
                env.physics.data.ctrl[:] = ctrl

    def get_seed(self) -> tuple:
        return tuple(env.task.random.seed() for env in self._envs)

    def set_seed(self, seed: tuple):
        for s, env in zip(seed, self._envs):
            env.task.random.seed(s)

    def clone(self) -> "Environment":
        return deepcopy(self)

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
        # Process the control suite results
        results = [self._process_result(result) for result in results]

        # Don't return an observation if no_observation flag is set
        if no_observation:
            # Unzip all tuples into 3 tuples containing the observations, flags and info dicts, respectively
            results = [*zip(*results)]
            # No observation is returned
            results[0] = None
            # Merge all rewards to one tensor
            results[1] = batch_tensors(*results[1])
            # Merge all flags to one tensor
            results[2] = batch_tensors(*results[2])
            # Return all results as a tuple
            return tuple(results)
        # If required, set observation to image observations
        elif not self._state_obs:
            # Get raw pixels from all environments
            pixels_tuple = self._pixels()
            # Convert them to suitable observations
            observations = [preprocess_observation(o, self._bit_depth, self._observation_size) for o in pixels_tuple]
            # Merge the observations in the results
            results = [(o,) + result[1:] for o, result in zip(observations, results)]

            # Add all raw pixel observations to the info dicts
            for result, pixels in zip(results, pixels_tuple):
                result[3]['pixels'] = pixels

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
        return tuple(env.close() for env in self._envs)

    def sample_random_action(self):
        actions = []
        for env in self._envs:
            spec = env.action_spec()
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
            actions += [torch.from_numpy(action).to(torch.float32)]
        actions = batch_tensors(*actions)
        return actions

    def _pixels(self) -> tuple:
        return tuple([env.physics.render(camera_id=0) for env in self._envs])

    def _process_result(self, result) -> tuple:
        """

        :param result:
        :return:
        """
        observation = [np.asarray([obs]) if isinstance(obs, float) else obs for obs in result.observation.values()]
        observation = np.concatenate(observation)
        observation = torch.FloatTensor(observation)

        reward = result.reward
        reward = torch.tensor(reward) if reward is not None else torch.tensor(0)

        terminal = result.last()
        terminal = torch.tensor(terminal)

        return observation, reward, terminal, {}


if __name__ == '__main__':

    from dm_control import viewer

    _args = argparse.Namespace()
    _args.environment_name = CONTROL_SUITE_ENVS[0]
    _args.env_batch_size = 3
    _args.max_episode_length = 1000
    _args.state_observations = False
    _args.bit_depth = 5

    _env = ControlSuiteEnvironment(_args)

    _env.reset()

    # for _i in range(_args.batch_size):
    #     viewer.launch(_env._envs[_i], policy=lambda t: _env.sample_random_action()[_i])

    for _ in range(10000):
        _env.step(_env.sample_random_action())

    _env.close()
