import argparse

import numpy as np

import torch
from tqdm import tqdm

from thesis.data.episode import Episode
from thesis.data.experience_replay import ExperienceReplay
from thesis.environment.env import Environment
from thesis.environment.util import postprocess_observation
from thesis.planner.planner import Planner
from thesis.rssm.rssm import RSSM
from thesis.util.logging import Log

"""
    Helper functions for data collection
"""


def generate_random_episode_batch(env: Environment) -> list:
    """
    Generate a random trajectory in the environment using a uniformly distributed policy
    :param env: The environment in which the trajectory should be generated
    :return: a list of Episode objects containing the simulated data
    """
    # Reset the environment
    observation, _, _ = env.reset()
    # Get the batch size
    batch_size = observation.size(0)
    # Store the episode data in a list
    data = [observation]
    # Generate the trajectory
    env_terminated = torch.zeros(batch_size).bool()
    while not all(env_terminated):
        # Sample a random action from the environment action space
        action = env.sample_random_action()
        # Obtain o_{t+1}, r_{t+1}, f_{t+1} by executing a_t in the environment
        observation, reward, flag, info = env.step(action)
        # Extend the episode using the obtained information
        data.extend([action, reward, flag, observation])
        # Set environment termination flags
        env_terminated = flag
    # Return the data as episodes
    return data_to_episodes(data)


def data_to_episodes(data: list) -> list:
    """
    Convert data sequences obtained from the environment to Episode objects
    :param data: A list containing data obtained from the environment model
                 The list consists of
                     o_0, a_0, r_1, f_1, o_1, a_1, r_2, ... ,a_{T-1}, r_T, f_T, o_T
                 where (at time t)
                     o_t is an observation tensor
                        Shape: (batch_size,) + observation_shape
                     a_t is an action tensor
                        Shape: (batch_size,) + action_shape
                     r_t is a reward tensor
                        Shape: (batch_size,)
                     f_t is a termination flag tensor
                        Shape: (batch_size,)
    :return: a list of Episode objects containing all data
    """
    if len(data) == 0:
        return list()
    # Get the batch size
    batch_size = data[0].size(0)
    # Convert the data to Episode objects
    episodes = list()
    # Data is stored as:
    # o_0, a_0, r_1, f_1, o_1, a_1, r_2, ... ,a_{T-1}, r_T, f_T, o_T
    # Only o_t, a_t, r_{t+1} are relevant
    # For each episode in the batch the termination flags need to be checked
    for i in range(batch_size):
        episode = Episode()
        for j in range(0, len(data), 4):
            # Get the relevant subsequence of the data
            seq = data[j:j+4]
            # Check whether the end of the data has been reached
            if len(seq) == 4:
                o, a, r, f = seq
                # Get the data relevant to this batch index
                o, a, r, f = o[i], a[i], r[i], f[i]
                # Add the data to the episode
                episode.append_all(o, a, r)
                # Check if the episode terminated. If so, add final observation and go to next episode
                if f:
                    # Final observation is the first entry after this sequence
                    o = data[j+4][i]
                    episode.append(o)
                    break
            else:  # Procedure only enters this code block if the episode data is incomplete (termination flag not set)
                # Subsequence only contains the final observation
                o = seq[i]
                # Append the final observation
                episode.append(o)
        # Add the episodes to the collection
        episodes.append(episode)
    # Return all episode objects
    return episodes


def init_data(dataset: ExperienceReplay,
              env: Environment,
              args: argparse.Namespace) -> tuple:
    """
    Add data to the dataset by simulating data using the environment model and a uniformly distributed policy
    :param dataset: The dataset that should be filled
    :param env: The environment model
    :param args: An argparse.Namespace object containing parsed arguments
    :return: a two-tuple consisting of
                - a reference to the dataset
                - a dict containing info about the computation
    """
    info = dict()
    # Get the number of environments that are simulated at once
    bs = args.env_batch_size

    # Determine how many rollouts need to happen with the environment batch size
    num_iters = args.num_seed_episodes // bs

    # Build a progress bar
    iters = tqdm(range(num_iters),
                 total=num_iters,
                 desc='Generating random episodes')
    iters.set_postfix_str(f'completed 0/{num_iters * bs}')

    # Simulate data
    with torch.no_grad():
        for i in iters:
            # Generate a random episode batch
            episodes = generate_random_episode_batch(env)
            # Add the episodes to the dataset
            dataset.append_episodes(episodes)

            # Update progress bar
            iters.set_postfix_str(f'completed {(i + 1) * bs}/{num_iters * bs}')

    return dataset, info


def collect_data(env: Environment,
                 planner: Planner,
                 env_planning: Environment,
                 args: argparse.Namespace,
                 log: Log = None,
                 fixed_start: tuple = None,
                 log_prefix: str = 'data_collection',
                 progress_desc: str = 'Collecting episode',
                 ) -> tuple:
    """
    Execute an episode (batch) in the environment. Actions are selected by the specified planner, which plans in a
    (possibly learned) model of the environment. The obtained data is returned in Episode objects
    :param env: The environment in which data should be collected
    :param planner: The planner that selects actions to execute in the environment
    :param env_planning: The environment model that is used by the planner to select actions
    :param args: argparse.Namespace object containing hyperparameters
    :param log: (optional) a Log object for logging info about the data collection
    :param fixed_start: (optional) can be specified to set the environment model to some initial state.
                        two-tuple containing:
                - a torch.Tensor containing the initial observation
                    shape: (batch_size,)+ observation_shape
                - an object containing the initial state to which the environment should be set
    :param log_prefix: (optional) string specifying the prefix of the names of the log files
    :param progress_desc: (optional) string specifying the description in the progress bar
    :return: a two-tuple consisting of:
                - a list of Episode objects containing the data that was collected. The number of Episode objects
                  corresponds to the planning batch size
    """

    # Log the predicted and true rewards during data collection
    reward_log = f'{log_prefix}_rewards'
    if log is not None:
        log.create_log(reward_log, 't', 'true reward', 'predicted reward', 'difference')
    # Log the predicted and true observations during data collection
    observation_log = f'{log_prefix}_observations'
    if log is not None:
        log.create_image_folder(observation_log)

    # Build a progress bar for data collection (if episode length is known)
    progress = tqdm(desc=progress_desc,
                    total=args.max_episode_length) \
        if args.max_episode_length != np.inf else None

    with torch.no_grad():
        # Reset the environment, set to initial state
        observation, env_terminated, info = env.reset()
        if fixed_start is not None:
            init_observation, init_state = fixed_start
            observation = init_observation
            env.set_state(init_state)

        # Reset the planning environment
        env_planning.reset()
        # If complete observability is assumed, make sure the planning env shares state
        if type(env) == type(env_planning):
            env_planning.set_state(env.get_state())
            env_planning.set_seed(env.get_seed())  # TODO -- seed properly

        # Execute an episode (batch) in the environment
        episode_data = [observation]
        while not all(env_terminated):
            # Plan action a_t in an environment model
            action, plan_info = planner.plan(env_planning)
            # Obtain o_{t+1}, r_{t+1}, f_{t+1} by executing a_t in the environment
            observation, reward, env_terminated, info = env.step(action)

            # Perform the action in the planning environment as well
            if isinstance(env_planning, RSSM):
                observation_prediction, reward_prediction, _, _ = env_planning.step(action,
                                                                                    true_observation=observation)
                observation_prediction.cpu()
                reward_prediction.cpu()
            else:
                observation_prediction, reward_prediction, _, _ = env_planning.step(action)

            # Log the obtained and predicted reward
            # TODO -- support logging for environment batches
            r_true, r_pred = reward[0].item(), reward_prediction[0].item()
            if log is not None:
                log.log_values(reward_log, env.t, r_true, r_pred, abs(r_true - r_pred))
            # Log the obtained and predicted observation
            o_true = postprocess_observation(observation, args.bit_depth, dtype=torch.float32)[0]
            o_pred = postprocess_observation(observation_prediction, args.bit_depth, dtype=torch.float32)[0]
            if log is not None:
                log.log_observations(observation_log, f'observations_t_{env.t}', o_true, o_pred)

            # Extend the episode using the obtained information
            episode_data.extend([action, reward, env_terminated, observation])

            # Update progress bar
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f't: {env.t}, '
                    f'r_true: {reward[0].item():.3f}, '
                    f'r_pred: {reward_prediction[0].item():.3f}, ',
                    refresh=True
                )

        if progress is not None:
            progress.close()
            print('\n')

    # Return the collected data
    return data_to_episodes(episode_data), {}

