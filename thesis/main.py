import argparse

from tqdm import tqdm

import numpy as np

import torch
from torch.nn import Module

from thesis.environment.util import postprocess_observation
from thesis.main_args import get_parsed_arguments
from thesis.main_helpers_data import init_data, data_to_episodes, collect_data
from thesis.main_helpers_init import select_device, init_env_from_args, init_planner_env_from_args, \
    init_value_model_from_args, \
    init_planner_from_args, init_value_model_trainer_from_args, init_planner_env_trainer_from_args,\
    init_eval_planner_from_args

from thesis.data.experience_replay_episodes import ExperienceReplay
from thesis.rssm.rssm import RSSM

from thesis.util.logging import Log


# TODO -- exploration noise
# TODO -- argument controlling output device of environments


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
    if not hasattr(args, 'checkpoint'):
        # No checkpoint given

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

        # Initialize the planner used for data collection
        planner = init_planner_from_args(args, ctx=locals())
        # Initialize the planner used for evaluation
        planner_eval = init_eval_planner_from_args(args, ctx=locals())

        # The PlaNet model assumes some known initial state
        init_observation, _, _ = env.reset()
        init_state = env.get_state()

        # Keep iteration counter
        iter_count = 1

    else:
        # User specified a checkpoint

        # Get the checkpoint
        path_to_checkpoint = args.checkpoint
        checkpoint = load_checkpoint(path_to_checkpoint)

        # Get the args from the checkpoint
        original_args = checkpoint['args']

        # Verify that the old log files are not overwritten
        # That is, the user should have specified a different log file
        if not hasattr(args, 'log_directory') or\
                getattr(args, 'log_directory') == getattr(original_args, 'log_directory'):
            raise Exception('Old log files should not be overwritten! Specify a different log directory using the '
                            '`--log_directory` argument!')

        # Create a log
        log = Log(args)
        log.log_message('')
        log.log_message(f'Starting from checkpoint {path_to_checkpoint}')
        log.log_message('')

        # Any args specified for this run will overwrite the old args
        for attr, val in vars(args).items():
            original_val = getattr(original_args, attr) if hasattr(original_args, attr) else '<Not specified>'
            if original_val != val:
                setattr(original_args, attr, val)
                log.log_message(f'  Argument {attr} overwritten from {original_val} to {val}')
        args = original_args

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

        # Initialize the planner used for data collection
        planner = init_planner_from_args(args, ctx=locals())
        # Initialize the planner used for evaluation
        planner_eval = init_eval_planner_from_args(args, ctx=locals())

        # The PlaNet model assumes some known initial state
        init_observation = checkpoint['init_observation']
        init_state = checkpoint['init_state']

        # Keep iteration counter
        iter_count = checkpoint['iter_count'] + 1

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
                                       log_prefix=f'iter_{iter_count}',
                                       device=device)

        '''
        
            DATA COLLECTION PHASE
        
        '''

        # DATA COLLECTION PHASE
        log.log_message(f'Data collection phase {iter_count}')
        log.log_message(f'      Dataset size: [{dataset.num_episodes}/{args.max_episodes_buffer}]')

        if not args.disable_data_collection:
            episodes, info = collect_data(
                env=env,
                planner=planner,
                env_planning=env_planning,
                args=args,
                log=log,
                fixed_start=(init_observation, init_state),
                log_prefix=f'iter_{iter_count}_collect',
                progress_desc='Data collection'
            )
            dataset.append_episodes(episodes)

        '''
        
            EVALUATION PHASE
            
        '''

        if iter_count % args.evaluation_period == 0:

            log.log_message(f'Evaluation phase')
            collect_data(
                env=env,
                planner=planner_eval,
                env_planning=env_planning,
                args=args,
                log=log,
                fixed_start=(init_observation, init_state),
                log_prefix=f'iter_{iter_count}_eval',
                progress_desc='Evaluation'
            )

        # Save checkpoint if required
        if iter_count % args.checkpoint_period == 0:
            log.log_message('Saving checkpoint')
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
    from thesis.environment.env_suite import CONTROL_SUITE_ENVS

    from thesis.main_args import get_placeholder_args

    _args = get_placeholder_args()

    _args.max_episode_length = 200
    # _args.max_episode_length = 20

    # _args.num_seed_episodes = 100

    # _args.experience_size = 10000

    _args.environment_name = CONTROL_SUITE_ENVS[7]
    # _args.environment_name = GYM_ENVS[0]

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
    _args.checkpoint_period = np.inf
    _args.evaluation_period = 50

    # _args.disable_data_collection = True
    # _args.disable_training = True
    _args.num_train_sequences = 1

    # _args.planner = 'cem'
    # _args.state_observations = True

    # _args.data_augmentations = []
    # _args.data_augmentations = ['random_translate']

    # _args.checkpoint = '../logs/log_test/iter_1_checkpoint.pth'

    run(_args)
