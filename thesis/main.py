import argparse

from tqdm import tqdm

import numpy as np

import torch
from torch.nn import Module

from thesis.environment.util import postprocess_observation
from thesis.main_args import get_parsed_arguments
from thesis.main_helpers_data import init_data, data_to_episodes
from thesis.main_helpers_init import select_device, init_env_from_args, init_planner_env_from_args, \
    init_value_model_from_args, \
    init_planner_from_args, init_value_model_trainer_from_args, init_planner_env_trainer_from_args

from thesis.data.experience_replay_episodes import ExperienceReplay
from thesis.rssm.rssm import RSSM

from thesis.util.logging import Log


# TODO -- exploration noise
# TODO -- argument controlling output device of environments
# TODO -- argument skipping planning for first couple of iterations
# TODO -- evaluation


def save_checkpoint(path: str,
                    args: argparse.Namespace,
                    iter_count: int,
                    init_state: tuple,
                    init_observation: torch.Tensor,
                    dataset: ExperienceReplay,
                    env_model: torch.nn.Module = None,
                    value_model: torch.nn.Module = None,
                    ):

    checkpoint = {
        'args': args,
        'iter_count': iter_count,
        'init_state': init_state,
        'init_observation': init_observation,
        'dataset': dataset,
    }

    if env_model is not None:
        checkpoint['env_model'] = env_model
    if value_model is not None:
        checkpoint['value_model'] = value_model

    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location=torch.device('cpu'))


def run(args: argparse.Namespace):
    # If no args are explicitly given, use the arguments passed to the program
    args = args or get_parsed_arguments()

    '''
    
        INITIALIZATION (POSSIBLY FROM CHECKPOINT)
    
    '''

    # Check if the run should be initialized from a checkpoint
    if hasattr(args, 'checkpoint'):
        # Get the checkpoint
        path_to_checkpoint = args.checkpoint
        checkpoint = load_checkpoint(path_to_checkpoint)

        args = checkpoint['args']

        # Create a log
        log = Log(args)
        log.log_message('')
        log.log_message(f'Starting from checkpoint {path_to_checkpoint}')
        log.log_message('')

        # Select the device on which models are run
        use_cuda, device = select_device(args, ctx=locals())

        dataset = checkpoint['dataset']

        # Build the environment used for data collection
        env = init_env_from_args(args, ctx=locals())

        # Build an additional environment for planning
        env_planning = checkpoint.get('env_model',
                                      init_planner_env_from_args(args, ctx=locals())
                                      )
        if isinstance(env_planning, torch.nn.Module):
            env_planning = env_planning.to(device)

        # Optionally, initialize a value function model
        value_model = checkpoint.get('value_model',
                                     init_value_model_from_args(args, ctx=locals())  # Can be None
                                     )
        if isinstance(value_model, torch.nn.Module):
            value_model = value_model.to(device)

        # Initialize the planner
        planner = init_planner_from_args(args, ctx=locals())

        # The PlaNet model assumes some known initial state
        init_observation = checkpoint['init_observation']
        init_state = checkpoint['init_state']

        # Keep iteration counter
        iter_count = checkpoint['iter_count'] + 1

    else:

        # Create a log
        log = Log(args)
        # Log the description of this run
        log.log_message(args.desc)
        # Log the args/hyperparameters
        log.log_args(args)

        # Select the device on which models are run
        use_cuda, device = select_device(args, ctx=locals())

        # Build the environment used for data collection
        env = init_env_from_args(args, ctx=locals())
        # Build an additional environment for planning
        env_planning = init_planner_env_from_args(args, ctx=locals())

        # Initialize the dataset
        dataset = ExperienceReplay(
            args.max_episode_length * 3 + 1,  # Each step 3 relevant tensors (a, o, r) are added + 1 initial observation
            args)

        # Add random seed episodes to the dataset
        log.log_message('Generating random episodes')
        init_data(dataset, env, args)

        # Optionally, initialize a value function model
        value_model = init_value_model_from_args(args, ctx=locals())  # Can be None

        # Initialize the planner
        planner = init_planner_from_args(args, ctx=locals())

        # The PlaNet model assumes some known initial state
        init_observation, _, _ = env.reset()
        init_state = env.get_state()

        # Keep iteration counter
        iter_count = 1

    '''
    
        CHECKPOINT-INDEPENDENT INITIALIZATION
    
    '''

    # Initialize trainers for the models used
    # If the models do not need to be trained (e.g. with true environment model), the trainers are set to None
    value_model_trainer = init_value_model_trainer_from_args(args, ctx=locals())
    env_planning_trainer = init_planner_env_trainer_from_args(args, ctx=locals())

    '''
        
        START MAIN LOOP
        
    '''

    # Start main loop
    log.log_message('Starting main loop')
    converged = False
    while not converged:

        '''
        
            TRAINING PHASE
        
        '''

        # TRAINING PHASE
        log.log_message(f'Training phase {iter_count}')

        if not args.disable_training and isinstance(value_model, Module):
            # Train the value model
            value_model_trainer.train(value_model)  # TODO -- pass all arguments -- optimizer in trainer

        if not args.disable_training and isinstance(env_planning, RSSM):
            # Train the RSSM
            env_planning_trainer.train(env_planning,
                                       args.train_batch_size,
                                       log=log,
                                       device=device)

        '''
        
            DATA COLLECTION PHASE
        
        '''

        # DATA COLLECTION PHASE
        log.log_message(f'Data collection phase {iter_count}')

        # Log the predicted and true rewards during data collection
        reward_log = f'iter_{iter_count}_rewards'
        log.create_log(reward_log, 't', 'true reward', 'predicted reward', 'difference')
        # Log the predicted and true observations during data collection
        observation_log = f'iter_{iter_count}_observations'
        log.create_image_folder(observation_log)

        # Build a progress bar for data collection (if episode length is known)
        progress = tqdm(desc=f'Data Collection {iter_count}',
                        total=args.max_episode_length)\
            if args.max_episode_length != np.inf else None

        with torch.no_grad():
            # Reset the environment, set to initial state
            observation, env_terminated, info = env.reset()
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
                # TODO -- support for environment batches
                r_true, r_pred = reward[0].item(), reward_prediction[0].item()
                log.log_values(reward_log, env.t, r_true, r_pred, abs(r_true - r_pred))
                # Log the obtained and predicted observation
                o_true = postprocess_observation(observation, args.bit_depth, dtype=torch.float32)[0]
                o_pred = postprocess_observation(observation_prediction, args.bit_depth, dtype=torch.float32)[0]
                log.log_observations(observation_log, f'observations_t_{env.t}', o_true, o_pred)

                # Extend the episode using the obtained information
                episode_data.extend([action, reward, env_terminated, observation])

                if progress is not None:  # TODO -- cleaner
                    progress.update(1)
                    progress.set_postfix_str(
                        f't: {env.t}, '
                        f'r_true: {reward[0].item():.3f}, '
                        f'r_pred: {reward_prediction[0].item():.3f}, '
                        f'dataset size: {dataset.num_episodes}',
                        refresh=True
                    )

            # Append the collected data to the dataset
            dataset.append_episodes(data_to_episodes(episode_data))

            if progress is not None:
                progress.close()
                print('\n')

        # Save checkpoint if required
        if iter_count % args.checkpoint_period == 0:
            save_checkpoint(
                path=f'{args.log_directory}/iter_{iter_count}_checkpoint.pth',
                args=args,
                iter_count=iter_count,
                init_state=init_state,
                init_observation=init_observation,
                dataset=dataset,
                env_model=env_planning if isinstance(env_planning, torch.nn.Module) else None,
                value_model=value_model if isinstance(env_planning, torch.nn.Module) else None,
            )

        # Check whether the algorithm should terminate
        converged = False  # TODO -- stop criterion
        # Raise the iteration counter
        iter_count += 1


if __name__ == '__main__':

    from thesis.environment.env_gym import GYM_ENVS

    from thesis.main_args import get_placeholder_args

    _args = get_placeholder_args()

    _args.max_episode_length = 200
    # _args.max_episode_length = 20

    # _args.num_seed_episodes = 100

    # _args.experience_size = 10000

    # _args.environment_name = CONTROL_SUITE_ENVS[1]
    _args.environment_name = GYM_ENVS[0]

    _args.plan_env_type = 'rssm'
    # _args.plan_env_type = 'true'
    _args.disable_cuda = True

    # _args.planning_horizon = 8
    # _args.num_plan_iter = 1
    # _args.num_plan_candidates = 40
    # _args.num_plan_top_candidates = int(_args.num_plan_candidates // 1.3)

    # _args.planner = 'cem'
    _args.planner = 'random'
    # _args.disable_cuda = False

    _args.rssm_sample_mean = True

    # _args.checkpoint_period = 1
    _args.checkpoint_period = 20

    # _args.planner = 'cem'
    # _args.state_observations = True

    # _args.checkpoint = '../logs/log_test/iter_1_checkpoint.pth'

    run(_args)
