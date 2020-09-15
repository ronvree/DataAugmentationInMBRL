
from thesis.environment.env import Environment

import torch


class Planner:

    """
    Abstract class for defining a planner
    """

    def plan(self,
             env: Environment,
             device=torch.device('cpu')) -> tuple:
        """
        Plan an action based on simulated data in the environment model
        :param env: The environment model in which data can be simulated
        :param device: the device on which the resulting action tensor should be stored
        :return: a two-tuple containing:
                    - an action tensor
                        Shape: (batch_size,) + action_shape
                    - an info dict
        """
        raise NotImplementedError
