
import argparse
import tqdm

from thesis.data.experience_replay import ExperienceReplay


import torch
from torch.nn import Module

from torch.utils.data import DataLoader

from thesis.value_model.qmodel import QModel
from thesis.value_model.util import DummyModel
from thesis.value_model.vmodel import VModel


class Trainer:

    # TODO -- data augmentation

    # TODO -- C, by custom sampler

    # TODO -- logging here?

    def __init__(self, dataset: ExperienceReplay, args: argparse.Namespace):

        # Store a reference to the dataset
        self._dataset = dataset
        # Store which loss function should be used
        self._loss = args.loss

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('Trainer Arguments')

        parser.add_argument('loss',
                            type=str,
                            choices=['l1', 'mse'],
                            default='mse',
                            help='The loss function to use when training')

        return parser

    def train(self,
              model: Module,
              optimizer: torch.optim.Optimizer,
              batch_size: int,
              num_iters: int = 1,
              device=torch.device('cpu')
              ) -> tuple:
        """
        TODO -- complete comments
        :param model:
        :param optimizer:
        :param batch_size:
        :param num_iters:
        :param device:
        :return:
        """

        # Create a dict for storing info about this train procedure
        info = {'epochs': []}

        # Convert the dataset to a torch.utils.data.TensorDataSet
        dataset = self._dataset.as_dataset()
        # Build an iterable over the dataset
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 )

        # Move the model to the correct device if necessary
        model = model.to(device)

        # Set the model to train mode
        model.train()
        # Do optimization loops
        for i_opt in range(num_iters):

            # Create a dict for storing info about this epoch
            epoch_info = {'index': i_opt, 'batches': []}
            info['epochs'].append(epoch_info)

            # Build progress bar
            loop = tqdm.tqdm(enumerate(dataloader),
                             total=len(dataloader),
                             desc='Train value function')
            # Loop over the data
            for i_batch, (o, a, r, o_, a_) in loop:
                # Move the data sample to the correct device
                o, a, r, o_, a_ = [tensor.to(device) for tensor in (o, a, r, o_, a_)]

                # Create a dict for storing info about this batch
                batch_info = {'index': i_batch}
                epoch_info['batches'].append(batch_info)

                # Compute value estimates
                predictions = self._compute_predictions(model, (o, a, r, o_, a_))
                # Compute targets from the dataset
                targets = self._compute_targets(model, (o, a, r, o_, a_))

                # Reset all gradients
                optimizer.zero_grad()
                # Compute the loss
                if self._loss == 'l1':
                    loss = torch.nn.functional.l1_loss(predictions, targets)
                elif self._loss == 'mse':
                    loss = torch.nn.functional.mse_loss(predictions, targets)
                else:
                    raise Exception('Unsupported loss function!')

                # Compute the gradient
                loss.backward()
                # Use the gradient to optimize the parameters
                optimizer.step()

                # Update progress bar
                loop.set_postfix_str(
                    ''.join([
                        f'iter {i_opt}/{num_iters}',
                        f', batch {i_batch}/{len(dataloader)}',
                        f', {self._loss} loss: {loss.item():.3f} '
                    ])
                )

                # Update info dicts
                batch_info['loss'] = loss.item()

        return model, info

    def _compute_predictions(self, model: torch.nn.Module, sample: tuple) -> torch.Tensor:

        o, a, r, o_, a_ = sample

        if isinstance(model, QModel):
            # Estimate q values
            predictions = model.forward(o, a)
        elif isinstance(model, VModel):
            # Estimate values
            predictions = model.forward(o)
        elif isinstance(model, DummyModel):
            predictions = model(o)
        else:
            raise Exception('Unsupported model type!')

        return predictions

    def _compute_targets(self, model: torch.nn.Module, sample: tuple) -> torch.Tensor:

        o, a, r, o_, a_ = sample

        if isinstance(model, QModel):
            # Compute targets from the dataset
            with torch.no_grad():
                targets = r + model.forward(o_, a_)
        elif isinstance(model, VModel):
            # Compute targets from the dataset
            with torch.no_grad():
                targets = r + model.forward(o_)
        elif isinstance(model, DummyModel):
            with torch.no_grad():
                targets = r + model.forward(o_)
        else:
            raise Exception('Unsupported model type!')

        return targets



