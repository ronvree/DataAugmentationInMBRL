import argparse

import torch
from tqdm import tqdm

from thesis.data.episode import Episode
from thesis.data.experience_replay import ExperienceReplay
from thesis.environment.env import Environment


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
