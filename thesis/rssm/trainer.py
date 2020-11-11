import argparse

from torch.distributions import Normal, kl_divergence
from torch.optim import Adam
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from thesis.data.augmentation.data_augmentation import from_keyword
from thesis.data.experience_replay_episodes import ExperienceReplay
from thesis.rssm.rssm import RSSM
from thesis.util.func import batch_tensors
from thesis.util.logging import Log


class Trainer:
    """
        Object for training a Recurrent State Space Model
    """

    def __init__(self, dataset: ExperienceReplay, args: argparse.Namespace):

        # Store a reference to the dataset
        self._dataset = dataset

        # Set how many times sequences should be sampled during training
        self._num_sequences = args.num_train_sequences

        # Set optimizer hyperparameters
        self._learning_rate = args.rssm_optimizer_lr
        self._epsilon = args.rssm_optimizer_epsilon

        # Set regularization hyperparameters
        self._kl_free_nats = args.free_nats
        self._grad_clip_norm = args.grad_clip_norm

        # Set relative weights of loss terms
        self._c_r_loss = args.rssm_reward_loss_weight
        self._c_o_loss = args.rssm_observation_loss_weight
        self._c_kl_loss = args.rssm_kl_loss_weight

        self._data_augmentations = args.data_augmentations or []
        self._state_action_augmentations = args.state_action_augmentations or []

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('RSSM Trainer Arguments')

        parser.add_argument('--num_train_sequences',
                            type=int,
                            default=50,
                            # TODO -- value in original paper -- i think it was 200 but they use smaller seq lengths for training
                            help='Number of episode batches sampled during the training phase of the RSSM. Refered to'
                                 'as C in the original paper')

        parser.add_argument('--rssm_optimizer_lr',
                            type=float,
                            default=1e-3,  # Value used in original paper: 1e-3
                            help='Learning rate of the optimizer used to train the RSSM')

        parser.add_argument('--rssm_optimizer_epsilon',
                            type=float,
                            default=1e-4,  # Value used in original paper: 1e-4
                            help='Epsilon parameter in the Adam optimizer used to train the RSSM')

        parser.add_argument('--free_nats',
                            type=int,
                            default=3,  # Value used in original paper: 3
                            help='Free nats for KL divergence loss when training the RSSM')

        parser.add_argument('--grad_clip_norm',
                            type=int,
                            default=1000,  # Value used in original paper: 1000
                            help='Gradient clipping norm')

        parser.add_argument('--rssm_reward_loss_weight',
                            type=float,
                            default=1,  # Value used in original paper: 1
                            help='Weighing factor of the reward loss when training the RSSM')

        parser.add_argument('--rssm_observation_loss_weight',
                            type=float,
                            default=1,  # Value used in original paper: 1
                            help='Weighing factor of the observation loss when training the RSSM')

        parser.add_argument('--rssm_kl_loss_weight',
                            type=float,
                            default=1,  # Value used in original paper: 1
                            help='Weighing factor of the KL divergence loss when training the RSSM')

        parser.add_argument('--data_augmentations',
                            nargs='+',
                            help='Specify keywords for data augmentations that should be used when training the encoder'
                            )

        parser.add_argument('--state_action_augmentations',
                            nargs='+',
                            help='Specify keywords for data augmentations applied to the state action pairs')

        return parser

    def train(self,
              model: RSSM,
              batch_size: int,
              log: Log = None,
              log_prefix: str = 'log_train_rssm',
              progress_desc: str = 'Train RSSM',
              device=torch.device('cpu')
              ) -> tuple:
        """
        Train a Recurrent State Space Model (RSSM)
        :param model: the RSSM model
        :param batch_size: batch size to use when training
        :param log: a Log object for logging info about the training procedure
        :param log_prefix: Prefix string for any created log files
        :param progress_desc: Description string of the progress bar
        :param device: the device (CPU/GPU) on which the training procedure should be run
        :return: a two-tuple consisting of:
                    - a reference to the model that was trained
                    - a dict containing info about the training procedure
        """

        # Create a log for storing info about this train procedure
        train_log = f'{log_prefix}_losses'
        if log is not None:
            log.create_log(train_log, 'batch', 'total loss', 'reward loss', 'observation loss', 'KL loss')
        # Create a dict for storing info about this train procedure
        info = {}

        # Convert the dataset to a torch.utils.data.TensorDataSet
        # dataset = self._dataset.as_episode_dataset()
        dataset = self._dataset

        # Define a sampling strategy for the dataloader
        sampler = BatchSampler(
            sampler=RandomSampler(
                data_source=dataset,
                replacement=True,
                num_samples=self._num_sequences * batch_size
            ),
            batch_size=batch_size,
            drop_last=False
        )

        # Build a dataloader for the dataset
        dataloader = DataLoader(dataset, batch_sampler=sampler)

        # Move the model to the correct device if necessary
        model = model.to(device)

        # Set the model to train mode
        model.train()

        # Define optimizer
        optimizer = Adam(model.parameters(),
                         lr=self._learning_rate,
                         eps=self._epsilon,
                         )

        # Build progress bar
        progress = tqdm(enumerate(dataloader),
                        total=len(dataloader),
                        desc=progress_desc)

        # Do optimization loops
        for i_batch, (o, a, r, o_, a_) in progress:

            # Move the data sample to the correct device
            o, a, r, o_, a_ = episode = tuple([tensor.to(device) for tensor in (o, a, r, o_, a_)])

            # Reset all gradients
            optimizer.zero_grad()

            # Compute the loss over the episode
            loss, batch_info = self._compute_loss(model, episode)

            # Compute the gradient
            loss.backward()

            # Clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip_norm, norm_type=2)

            # Use the gradient to optimize the parameters
            optimizer.step()

            # Write to log
            if log is not None:
                log.log_values(train_log,
                               i_batch,
                               batch_info['loss'],
                               batch_info['reward_loss'],
                               batch_info['observation_loss'],
                               batch_info['belief_loss'],
                               )

            # Update progress bar
            progress.set_postfix_str(
                self._build_status_bar(
                    i_batch,
                    len(dataloader),
                    info,
                    batch_info
                )
            )

        # Return (a reference to) the trained model and an info dict
        return model, info

    def _compute_loss(self, model: RSSM, episode: tuple) -> tuple:
        """
        Compute the loss of the RSSM over a single batch of episodes
        :param model: the RSSM model
        :param episode: a five-tuple consisting of:
                - a tensor containing all observations obtained during the episode
                    shape: (batch_size, episode_length,) + observation_shape
                - a tensor containing all actions performed during the episode
                    shape: (batch_size, episode_length, action_size)
                - a tensor containing all rewards obtained during the episode
                    shape: (batch_size, episode_length)
                - a tensor containing all subsequent observations obtained during the episode
                    (has shared storage with the first observation tensor)
                    shape: (batch_size, episode_length,) + observation_shape
                - a tensor containing all subsequent actions performed during the episode
                    (has shared storage with the first action tensor)
                    shape: (batch_size, episode_length, action_size)
        :return: a two-tuple containing:
                    - a loss tensor
                        shape: (1,)
                    - a dict containing info about the loss computation
        """
        # Keep info dict
        info = {}

        # Get the batch size used
        batch_size = episode[0].size(0)

        # Reset RSSM initial state
        model.reset(batch_size=batch_size)

        # Average losses over all time steps
        reward_losses = []
        observation_losses = []
        belief_losses = []

        # Switch the episode and batch dimensions
        episode = tuple([Trainer._switch_dims(xs) for xs in episode])

        # Apply state-action augmentations
        for aug in self._state_action_augmentations:
            episode = from_keyword(aug)(*episode)

        # Apply data augmentation to the observation tensor
        o_augmented = episode[3]  # Shape: (T, batch_size,) + observation_shape
        for aug in self._data_augmentations:
            o_augmented = from_keyword(aug)(o_augmented)  # Shape (T, batch_size,) + augmented_observation_shape

        # Simulate the trajectory in the model
        for t, (o, a, r, o_, a_, o_aug) in enumerate(zip(*episode, o_augmented)):
            # Get prediction (distributions) from the environment model
            predicted_o_, predicted_r, predicted_s, _, _ = model.simulate_step(a)

            # Only observation sample is required, omit dist params
            predicted_o_, _, _ = predicted_o_
            # Only reward sample is required, omit dist params
            predicted_r, _, _ = predicted_r
            # Get belief distribution parameters
            predicted_s, state_prior_mean, state_prior_std = predicted_s

            # Compute reward loss
            reward_loss = F.mse_loss(predicted_r, r)
            reward_losses.append(reward_loss)

            # Compute observation loss
            observation_loss = F.mse_loss(predicted_o_, o_, reduction='none').sum(dim=(1, 2, 3))
            observation_losses.append(observation_loss)

            # Get the prior and posterior belief distributions
            prior = Normal(state_prior_mean, state_prior_std)
            # Get an estimate of the posterior belief using the encoder
            _, state_posterior_mean, state_posterior_std = model.posterior_state_belief(o_aug)
            posterior = Normal(state_posterior_mean, state_posterior_std)

            # Allowed deviation in KL divergence
            free_nats = torch.ones(1, dtype=torch.float32, device=o.device) * self._kl_free_nats

            # Compute KL loss
            belief_loss = kl_divergence(posterior, prior).sum(dim=1)
            # Bound by free nats
            belief_loss = torch.max(belief_loss, free_nats)
            # Add to all losses
            belief_losses.append(belief_loss)

        # Compute total loss components
        reward_loss = torch.mean(batch_tensors(*reward_losses)) * self._c_r_loss
        observation_loss = torch.mean(batch_tensors(*observation_losses)) * self._c_o_loss
        belief_loss = torch.mean(batch_tensors(*belief_losses)) * self._c_kl_loss
        # Compute total loss
        loss = reward_loss + observation_loss + belief_loss

        # Store relevant information in info dict
        info['reward_loss'] = reward_loss.item()
        info['observation_loss'] = observation_loss.item()
        info['belief_loss'] = belief_loss.item()
        info['loss'] = loss.item()

        return loss, info

    @staticmethod
    def _switch_dims(xs: torch.Tensor):
        """
        Switch the 'episode' and 'batch' dimensions of the given tensor
        Makes it easier to iterate through the episode dimension
        :param xs: a tensor whose first and second dimensions are 'batch' and 'episode', respectively
        :return: the same tensor, but shaped (episode_dim, batch_dim, ...)
        """
        assert len(xs.shape) >= 2
        # Permute the tensor
        remainder_dims = list(range(len(xs.shape)))[2:]
        xs = xs.permute(1, 0, *remainder_dims)
        return xs

    def _build_status_bar(self,
                          i: int,
                          total: int,
                          info: dict,
                          batch_info: dict
                          ) -> str:

        r_loss = batch_info['reward_loss']
        o_loss = batch_info['observation_loss']
        b_loss = batch_info['belief_loss']
        loss = batch_info['loss']

        s = f'Loss: {loss:.3f} = R: {r_loss:.3f} + O: {o_loss:.3f} + KL: {b_loss:.3f}'

        return s
