
import argparse

import numpy as np

from thesis.data.experience_replay_episodes import ExperienceReplay
from thesis.environment.env import Environment
from thesis.main_helpers_init import VALUE_MODEL_KEYWORDS, ACTION_VALUE_MODEL_KEYWORDS, DUMMY_VALUE_MODEL_KEYWORDS
from thesis.planner.cem import CEM, QCEM, VCEM
from thesis.rssm.extended.rssm_extended import ERSSM
from thesis.rssm.rssm import RSSM
from thesis.value_model.trainer import Trainer as ValueTrainer
from thesis.rssm.trainer import Trainer as RSSMTrainer
from thesis.util.logging import Log


def get_main_argument_parser() -> argparse.ArgumentParser:
    """
    Get an argparse.ArgumentParser object for parsing command line arguments for controlling the main program
    :return:
    """
    parser = argparse.ArgumentParser('Main Arguments')

    parser.add_argument('--checkpoint',
                        type=str,
                        required=False,
                        help='Load a checkpoint from the specified file')
    parser.add_argument('--state_checkpoint',
                        type=str,
                        required=False,
                        help='Load a checkpoint from the specified file, but only set the initial environment state')

    parser.add_argument('--desc',
                        type=str,
                        default='Run PlaNet',
                        help='Description of the run to be added to the log')

    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='When set, the GPU will be disabled')
    parser.add_argument('--disable_training',
                        action='store_true',
                        help='When set, the training procedure will be skipped')
    parser.add_argument('--disable_data_collection',
                        action='store_true',
                        help='When set, the data collection procedure will be skipped')
    parser.add_argument('--evaluation_period',
                        type=int,
                        default=np.inf,
                        help='Every `evaluation_period` iterations, the model is evaluated')
    parser.add_argument('--checkpoint_period',
                        type=int,
                        default=10,
                        help='Every `checkpoint_period` iterations a checkpoint of the run is saved to disk')
    parser.add_argument('--num_main_loops',
                        type=int,
                        default=np.inf,
                        help='The number of main loops that should be executed')
    parser.add_argument('--downscale_observations',
                        action='store_true',
                        help='Environment observations are scaled to 32x32 rather than 64x64. Encoder/decoder '
                             'architectures are modified accordingly.')

    parser.add_argument('--planner',
                        type=str,
                        choices=['debug', 'random', 'cem', 'qcem', 'vcem', 'ecem'],
                        default='cem',
                        help='Select which planning algorithm is used for data collection')
    parser.add_argument('--eval_planner',
                        type=str,
                        choices=['debug', 'random', 'cem', 'qcem', 'vcem', 'ecem'],
                        default='cem',
                        help='Select which planning algorithm is used for evaluation')
    parser.add_argument('--plan_env_type',
                        type=str,
                        choices=['true', 'rssm', 'erssm'],
                        default='rssm',
                        help='Controls whether a true environment model is used for planning, or a learned '
                             'approximation (RSSM)')
    parser.add_argument('--value_model',
                        type=str,
                        default='none',
                        help='Select which (if any) model should be used for value function estimation')

    parser.add_argument('--num_seed_episodes',
                        type=int,
                        default=5,  # Value in PlaNet paper: 5
                        help='Number of episodes generated with a random policy to create an initial experience dataset'
                        )
    parser.add_argument('--num_data_episodes',  # TODO
                        type=int,
                        default=1,
                        help='The number of episodes of data that are collected during the data collection phase')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=32,  # Value in PlaNet paper: 50
                        help='The batch size used during training')

    return parser


def get_parsed_arguments() -> argparse.Namespace:
    """
    Parse all command line arguments

    Uses multiple parsers corresponding to the components used in the main program

    :return: an argparser.Namespace object containing all parsed arguments
    """
    # Parse args of the main program
    args, _ = get_main_argument_parser().parse_known_args()

    # Handle Logging args
    Log.get_argument_parser().parse_known_args(namespace=args)

    # Handle Experience Replay args
    ExperienceReplay.get_argument_parser().parse_known_args(namespace=args)

    # Handle Environment args
    Environment.get_argument_parser().parse_known_args(namespace=args)

    # Handle planner args
    if args.planner == 'cem':
        CEM.get_argument_parser().parse_known_args(namespace=args)
    if args.planner == 'qcem':
        QCEM.get_argument_parser().parse_known_args(namespace=args)
    if args.planner == 'vcem':
        VCEM.get_argument_parser().parse_known_args(namespace=args)

    # Handle planner environment model args
    if args.plan_env_type == 'rssm':
        RSSM.get_argument_parser().parse_known_args(namespace=args)
    if args.plan_env_type == 'erssm':
        ERSSM.get_argument_parser().parse_known_args(namespace=args)

    # Handle planner environment trainer args
    if args.plan_env_type == 'rssm' or args.plan_env_type == 'erssm':
        RSSMTrainer.get_argument_parser().parse_known_args(namespace=args)

    # Handle value function model args
    if args.value_model in VALUE_MODEL_KEYWORDS:
        pass  # NO VALUE MODEL ARGS YET
    if args.value_model in ACTION_VALUE_MODEL_KEYWORDS:
        pass  # NO VALUE MODEL ARGS YET

    # Handle value function model trainer args
    if args.value_model in VALUE_MODEL_KEYWORDS + ACTION_VALUE_MODEL_KEYWORDS + DUMMY_VALUE_MODEL_KEYWORDS:
        ValueTrainer.get_argument_parser().parse_known_args(namespace=args)

    return args


def get_placeholder_args() -> argparse.Namespace:
    """
    FOR TESTING PURPOSES
    MAINLY IMPLEMENTED FOR CONVENIENCE, SINCE ARGUMENT PARSERS CANT BE USED IN GOOGLE COLAB

    Get an argparse.Namespace object containing 'default' values for each of the required arguments of the main program

    For required arguments, 'reasonable' values are chosen

    :return: argparse.Namespace object
    """
    # Create namespace object to store arguments
    args = argparse.Namespace()

    # MAIN PROGRAM ARGS
    # Get the main argument parser
    main_parser = get_main_argument_parser()
    # Get placeholder values from parser
    args.desc = main_parser.get_default('desc')
    args.disable_cuda = True  # main_parser.get_default('disable_cuda')
    args.disable_training = main_parser.get_default('disable_training')
    args.disable_data_collection = main_parser.get_default('disable_data_collection')
    args.evaluation_period = main_parser.get_default('evaluation_period')
    args.checkpoint_period = main_parser.get_default('checkpoint_period')
    args.num_main_loops = main_parser.get_default('num_main_loops')
    args.plan_env_type = main_parser.get_default('plan_env_type')
    args.planner = main_parser.get_default('planner')
    args.eval_planner = main_parser.get_default('eval_planner')
    args.value_model = main_parser.get_default('value_model')
    args.downscale_observations = main_parser.get_default('downscale_observations')
    # args.optimizer = main_parser.get_default('optimizer')
    # args.num_train_iter = main_parser.get_default('num_train_iter')
    args.num_seed_episodes = main_parser.get_default('num_seed_episodes')
    args.num_data_episodes = main_parser.get_default('num_data_episodes')
    args.train_batch_size = main_parser.get_default('train_batch_size')

    # LOGGER ARGS
    log_parser = Log.get_argument_parser()
    args.log_directory = '../logs/log_test'
    # args.log_directory = log_parser.get_default('log_directory')
    args.print_log = True
    # args.print_log = log_parser.get_default('print_log')

    # ENVIRONMENT ARGS
    env_parser = Environment.get_argument_parser()
    args.environment_name = env_parser.get_default('environment_name')
    args.max_episode_length = 200
    # args.max_episode_length = env_parser.get_default('max_episode_length')
    args.state_observations = env_parser.get_default('state_observations')
    args.env_batch_size = env_parser.get_default('env_batch_size')
    args.bit_depth = env_parser.get_default('bit_depth')

    # EXPERIENCE REPLAY DATASET ARGS
    dataset_parser = ExperienceReplay.get_argument_parser()
    args.max_episodes_buffer = 1000
    # args.max_episodes_buffer = dataset_parser.get_default('max_episodes_buffer')

    # VALUE FUNCTION MODEL ARGS
    pass  # NON-EXISTENT

    # PLANNER ARGS
    if args.planner == 'cem':
        planner_parser = CEM.get_argument_parser()

        args.planning_horizon = planner_parser.get_default('planning_horizon')
        args.num_plan_iter = planner_parser.get_default('num_plan_iter')
        args.num_plan_candidates = planner_parser.get_default('num_plan_candidates')
        args.num_plan_top_candidates = planner_parser.get_default('num_plan_top_candidates')
        args.plan_batch_size = planner_parser.get_default('plan_batch_size')

    if args.planner == 'qcem':
        planner_parser = QCEM.get_argument_parser()

        args.planning_horizon = planner_parser.get_default('planning_horizon')
        args.num_plan_iter = planner_parser.get_default('num_plan_iter')
        args.num_plan_candidates = planner_parser.get_default('num_plan_candidates')
        args.num_plan_top_candidates = planner_parser.get_default('num_plan_top_candidates')
        args.plan_batch_size = planner_parser.get_default('plan_batch_size')
    if args.planner == 'vcem':
        planner_parser = VCEM.get_argument_parser()

        args.planning_horizon = planner_parser.get_default('planning_horizon')
        args.num_plan_iter = planner_parser.get_default('num_plan_iter')
        args.num_plan_candidates = planner_parser.get_default('num_plan_candidates')
        args.num_plan_top_candidates = planner_parser.get_default('num_plan_top_candidates')
        args.plan_batch_size = planner_parser.get_default('plan_batch_size')

    # PLAN ENVIRONMENT ARGS
    if args.plan_env_type == 'rssm':
        rssm_parser = RSSM.get_argument_parser()

        args.deterministic_state_size = rssm_parser.get_default('deterministic_state_size')
        args.stochastic_state_size = rssm_parser.get_default('stochastic_state_size')
        args.reward_model_size = rssm_parser.get_default('reward_model_size')
        args.encoder_model_size = rssm_parser.get_default('encoder_model_size')
        args.state_model_size = rssm_parser.get_default('state_model_size')
        args.state_model_min_std = rssm_parser.get_default('state_model_min_std')
        args.encoding_size = rssm_parser.get_default('encoding_size')
        args.rssm_sample_mean = rssm_parser.get_default('rssm_sample_mean')
    if args.plan_env_type == 'erssm':
        rssm_parser = ERSSM.get_argument_parser()
        args.value_model_size = rssm_parser.get_default('value_model_size')
        args.deterministic_state_size = rssm_parser.get_default('deterministic_state_size')
        args.stochastic_state_size = rssm_parser.get_default('stochastic_state_size')
        args.reward_model_size = rssm_parser.get_default('reward_model_size')
        args.encoder_model_size = rssm_parser.get_default('encoder_model_size')
        args.state_model_size = rssm_parser.get_default('state_model_size')
        args.state_model_min_std = rssm_parser.get_default('state_model_min_std')
        args.encoding_size = rssm_parser.get_default('encoding_size')
        args.rssm_sample_mean = rssm_parser.get_default('rssm_sample_mean')

    # VALUE MODEL TRAINER ARGS
    if args.value_model in VALUE_MODEL_KEYWORDS + ACTION_VALUE_MODEL_KEYWORDS + DUMMY_VALUE_MODEL_KEYWORDS:
        value_trainer_parser = ValueTrainer.get_argument_parser()
        pass  # TODO

    # PLAN ENVIRONMENT TRAINER ARGS
    if args.plan_env_type == 'rssm' or args.plan_env_type == 'erssm':
        rssm_trainer_parser = RSSMTrainer.get_argument_parser()

        args.num_train_sequences = rssm_trainer_parser.get_default('num_train_sequences')

        args.rssm_optimizer_lr = rssm_trainer_parser.get_default('rssm_optimizer_lr')
        args.rssm_optimizer_epsilon = rssm_trainer_parser.get_default('rssm_optimizer_epsilon')

        args.free_nats = rssm_trainer_parser.get_default('free_nats')
        args.grad_clip_norm = rssm_trainer_parser.get_default('grad_clip_norm')

        args.rssm_reward_loss_weight = rssm_trainer_parser.get_default('rssm_reward_loss_weight')
        args.rssm_observation_loss_weight = rssm_trainer_parser.get_default('rssm_observation_loss_weight')
        args.rssm_kl_loss_weight = rssm_trainer_parser.get_default('rssm_kl_loss_weight')
        args.data_augmentations = rssm_trainer_parser.get_default('data_augmentations')
        args.state_action_augmentations = rssm_trainer_parser.get_default('state_action_augmentations')

    return args
