
import argparse
from collections import deque

import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset

from thesis.data.episode import Episode
from thesis.util.func import batch_tensors
from thesis.environment.util import preprocess_observation_tensor, postprocess_observation


class ExperienceReplay:

    """
        Implementation of a (possibly infinite) buffer of data collected from experience

        Can be converted to a PyTorch DataSet

        Data can be added in fixed-size Episode objects


    """

    def __init__(self, episode_length: int, args: argparse.Namespace):
        """
        Create a new ExperienceReplay dataset
        :param episode_length: the length of the episodes stored in this dataset
                    Episodes can't have a different length
        :param args: argparse.Namespace object containing hyperparameters
        """
        self._max_num_episodes = args.max_episodes_buffer
        self._bit_depth = args.bit_depth

        self._episode_length = episode_length

        self._data = deque(maxlen=self._max_num_episodes)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser('Experience Replay Arguments')

        parser.add_argument('--max_episodes_buffer',
                            type=int,
                            default=np.inf,
                            help='Limit on the number of episodes stored in this dataset')

        return parser

    @staticmethod
    def episode_to_samples(episode: Episode) -> tuple:
        """
        Transform an episode to a tuple of (o, a, r, o', a') data samples
        """
        # Store samples in a list
        samples = list()
        # An episode consists of:
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for i in range(0, len(episode), 3):  # Jump from observation to observation in the episodes
            # Get the next sequence of (o, a, r, o', a')
            sample = episode[i:i+5]
            # Check if the sample is complete (this does not happen at the end of the episode)
            if len(sample) == 5:
                samples.append(tuple(sample))
        # Return all samples
        return tuple(samples)

    @property
    def num_episodes(self) -> int:
        return len(self._data)

    def append_episode(self, episode: Episode):
        """
        Add an episode to the dataset
        :param episode: Episode object to append to the dataset
        """
        assert len(episode) == self._episode_length
        episode.to(torch.device('cpu'))
        # Preprocess the images so they require less memory
        episode.process_observations(lambda o: postprocess_observation(o, self._bit_depth))
        # Add the episode to the dataset
        self._data.append(episode)

    def append_episodes(self, episodes: iter):
        """
        Add multiple episodes to the dataset
        :param episodes: A collection of Episode objects to add to the dataset
        """
        for episode in episodes:
            self.append_episode(episode)

    def as_episode_dataset(self) -> TensorDataset:
        """
        Get the dataset as a PyTorch TensorDataset containing episodes of data

        Each entry in the dataset is a 5-tuple consisting of:
            - an observation tensor containing all observations obtained during the episode (at some time t)
                shape: (episode_length,) + observation_shape
            - an action tensor containing all actions performed during the episode (at some time t)
                shape: (episode_length,) + action_shape
            - a reward tensor containing all rewards obtained during the episode (at some time t + 1)
                shape: (episode_length,)
            - an observation tensor containing all subsequent observations obtained during the episode
              (at some time t + 1)
                shape: (episode_length,) + observation shape
            - an action tensor containing all subsequent actions performed during the episode (at some time t + 1)
                shape: (episode_length,) + action_shape

        The two observation tensors and two action tensors have shared storage

        All tensors are ordered in the way that data was collected

        :return a TensorDataset containing all episode data
        """
        # Group episode data
        episode_data = []
        for episode in self._data:

            # Get all episode data in single tensors
            observations = episode.get_observations_as_tensor()
            actions = episode.get_actions_as_tensor()
            rewards = episode.get_rewards_as_tensor()

            num_samples = rewards.size(0) - 1

            # Slice the corresponding tensors
            o = observations.narrow(dim=0, start=0, length=num_samples)
            a = actions.narrow(dim=0, start=0, length=num_samples)
            r = rewards.narrow(dim=0, start=0, length=num_samples)
            o_ = observations.narrow(dim=0, start=1, length=num_samples)
            a_ = actions.narrow(dim=0, start=1, length=num_samples)

            # Preprocess the images
            o = preprocess_observation_tensor(o, self._bit_depth)
            o_ = preprocess_observation_tensor(o_, self._bit_depth)

            episode_data.append((o, a, r, o_, a_))

        # Concatenate all episode data over a new dataset dimension
        tensors = [batch_tensors(*ts) for ts in (*zip(*episode_data),)]
        # Use these tensors to create a TensorDataset
        return TensorDataset(*tensors)

    def as_sample_dataset(self) -> TensorDataset:

        pass  # TODO

    @staticmethod
    def _batch_samples(samples: tuple) -> tuple:
        """
        Transform a tuple of sample tuples to a single batched sample tuple

        The batch size is the length of the tuple of samples

        :param samples: a tuple of individual samples
            Each individual sample is a tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: (1,) )
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: action_shape)

        :return: a single sample batch tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)
        """
        # Separate the samples into five tuples of all (o, a, r, o', a')
        o, a, r, o_, a_ = tuple(*zip(*samples))
        # For all five tuples, concatenate the entries over one batch dimension
        o, a, r, o_, a_ = tuple(batch_tensors(*ts) for ts in (o, a, r, o_, a_))
        # Return as a single batched sample tuple
        return o, a, r, o_, a_
