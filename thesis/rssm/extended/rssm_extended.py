import argparse

import torch

from thesis.rssm.extended.value_model import QModel
from thesis.rssm.rssm import RSSM


class ERSSM(RSSM):

    def __init__(self, action_shape: tuple, args: argparse.Namespace, device=torch.device('cpu')):
        super().__init__(action_shape, args, device)

        self._q_model = QModel(action_shape[-1], args)

    def state_action_value(self, action: torch.Tensor) -> torch.Tensor:
        hs, ss = self._hidden_state, self._belief_state
        return self._q_model.forward(hs, ss, action)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        parser = RSSM.get_argument_parser()
        parser.add_argument('--value_model_size',
                            type=int,
                            default=200,
                            help='Size of the value function model of the RSSM')

        return parser
