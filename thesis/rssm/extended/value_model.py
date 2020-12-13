import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# from thesis.util.func import sample_normal


class QModel(nn.Module):

    def __init__(self, action_size: int, args: argparse.Namespace):
        super().__init__()

        hs_size = args.deterministic_state_size
        ss_size = args.stochastic_state_size
        qm_size = args.value_model_size

        self.lin1 = nn.Linear(hs_size + ss_size + action_size, qm_size)
        self.lin2 = nn.Linear(qm_size, qm_size)
        self.lin3 = nn.Linear(qm_size, 1)

    def forward(self,
                hs: torch.Tensor,
                ss: torch.Tensor,
                action: torch.Tensor
                ) -> torch.Tensor:

        xs = torch.cat([hs, ss, action], dim=1)

        xs = self.lin1(xs)
        xs = F.relu(xs)

        xs = self.lin2(xs)
        xs = F.relu(xs)

        xs = self.lin3(xs).squeeze(dim=1)

        return xs
