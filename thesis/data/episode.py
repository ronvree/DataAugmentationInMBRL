
import torch

from thesis.util.func import batch_tensors


class Episode:

    """

        Convenience class for storing episodes

        Episodes are stored as sequences of:

        o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T

        where
            o denotes an observation
            a denotes an action
            r denotes a reward

        o_t denotes the observation at time step t

        All entries are stored as tensors (without batch dimension)


    """

    def __init__(self):
        self._episode = []

    def __len__(self):  # Generally, (3 x the number of steps simulated) + 1
        return len(self._episode)

    def __getitem__(self, index: int):
        return self._episode[index]

    @property
    def T(self) -> int:
        """
        Get the number of time steps simulated in this episode
        """
        return (len(self._episode) + 1) // 3

    @staticmethod
    def from_tuple(episode: tuple) -> "Episode":
        """
        Create a new Episode from a data sequence of
        o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        :param episode: data sequence tuple
        :return: an Episode object containing the data
        """
        e = Episode()
        e.append_all(episode)
        return e

    def to(self, *args, **kwargs):
        """
        Call .to function on all tensors in this episode
        :param args: the args to pass to the .to function
        :param kwargs: the kwargs to pass to the .to function
        """
        self._episode = [tensor.to(*args, **kwargs) for tensor in self._episode]

    def as_tuple(self) -> tuple:
        """
        Get the episode data in a tuple
        :return: a tuple containing a data sequence of
                    o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        """
        return tuple(self._episode)

    def append(self, tensor):
        """
        Append an entry to this episode
        :param tensor: the entry to be added
        """
        self._episode.append(tensor)

    def append_all(self, *tensors):
        """
        Append multiple entries to this episode
        :param tensors: the entries to be added
        """
        self._episode.extend([*tensors])

    def process_observations(self, f: callable) -> None:
        """
        Apply a function to all observations. All observations are replaced with the output of the function
        :param f: a callable to modify the observations. Should accept one argument: the original image tensor
        """
        # An episode consists of:
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for t in range(0, len(self), 3):
            self._episode[t] = f(self._episode[t])

    @property
    def observations(self) -> tuple:
        """
        Get all observation tensors in a tuple
        """
        observations = []
        # An episode consists of:
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for t in range(0, len(self), 3):
            observations.append(self._episode[t])
        return tuple(observations)

    @property
    def actions(self) -> tuple:
        """
        Get all action tensors in a tuple
        """
        actions = []
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for t in range(0, len(self) - 3, 3):
            actions.append(self._episode[t + 1])
        return tuple(actions)

    @property
    def rewards(self) -> tuple:
        """
        Get all reward tensors in a tuple
        """
        rewards = []
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for t in range(0, len(self) - 3, 3):
            rewards.append(self._episode[t + 2])
        return tuple(rewards)

    def get_observations_as_tensor(self) -> torch.Tensor:
        """
        Get all observations in one single tensor
            shape: (episode_length + 1,) + observation_shape

        The +1 is due to the initial observation

        :return: a torch.FloatTensor containing all obtained observations
        """
        # Concatenate or 'batch' the observations along a newly created episode_length dimension
        return batch_tensors(*self.observations)

    def get_actions_as_tensor(self) -> torch.Tensor:
        """
        Get all actions in one single tensor
            shape: (episode_length,) + action_shape

        :return: a torch.FloatTensor containing all performed actions
        """
        # Concatenate or 'batch' the actions along a newly created episode_length dimension
        return batch_tensors(*self.actions)

    def get_rewards_as_tensor(self) -> torch.Tensor:
        """
        Get all rewards in one single tensor
            shape: (episode_length,)

        :return: a torch.FloatTensor containing all obtained rewards
        """
        # Concatenate or 'batch' the rewards along a newly created episode_length dimension
        return batch_tensors(*self.rewards)

