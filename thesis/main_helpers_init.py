
import argparse

import torch

from thesis.environment.env import Environment
from thesis.environment.env_gym import GYM_ENVS, GymEnv
from thesis.environment.env_suite import CONTROL_SUITE_ENVS, ControlSuiteEnvironment
from thesis.planner.cem import CEM, QCEM, VCEM
from thesis.planner.planner import Planner
from thesis.planner.util import RandomPlanner
from thesis.rssm.rssm import RSSM
from thesis.value_model.qmodel import QModel
from thesis.value_model.util import DummyModel
from thesis.value_model.vmodel import VModel
from thesis.value_model.trainer import Trainer as ValueTrainer
from thesis.rssm.trainer import Trainer as RSSMTrainer

"""
    Helper functions for initializing objects based on parsed arguments
"""


def select_device(args: argparse.Namespace, ctx: dict) -> tuple:
    """
    Select which device to use based on the parsed arguments (CPU/GPU)
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a tuple containing
                - a flag set to true if the GPU is used (and false otherwise)
                - the PyTorch device selected
    """
    use_cuda = torch.cuda.is_available() and not args.disable_cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    log = ctx['log']

    # Write device selection to log
    log.log_message(f'GPU available: {torch.cuda.is_available()}, GPU selected: {not args.disable_cuda}')
    log.log_message(f'Device used: {"cpu" if not use_cuda else torch.cuda.get_device_name(device)}')

    return use_cuda, device


def init_env_from_args(args: argparse.Namespace, ctx: dict) -> Environment:
    """
    Build an environment from parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: an Environment built from the parsed arguments
    """
    name = args.environment_name

    if name in GYM_ENVS:
        return GymEnv(args)
    if name in CONTROL_SUITE_ENVS:
        return ControlSuiteEnvironment(args)
    raise Exception('Environment name not recognized!')


def init_planner_env_from_args(args: argparse.Namespace, ctx: dict) -> Environment:
    """
    Build an environment used for planning from parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: an Environment built from the parsed arguments
    """
    key = args.plan_env_type

    if key == 'true':
        environment = ctx['env']
        plan_env = init_env_from_args(args, ctx)
        plan_env.set_state(environment.get_state())
        plan_env.set_seed(environment.get_seed())
        return plan_env
        # return environment.clone()
    if key == 'rssm':
        action_shape = ctx['env'].action_shape
        device = ctx['device']
        return RSSM(action_shape, args, device=device)
    raise Exception('Unknown plan environment type!')


def init_planner_env_trainer_from_args(args: argparse.Namespace, ctx: dict):
    """
    Build a trainer for the planner environment model (if required)
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a suitable trainer object for training the environment model (None if using true environment)
    """
    key = args.plan_env_type

    if key == 'true':
        return None
    if key == 'rssm':
        dataset = ctx['dataset']
        return RSSMTrainer(dataset, args)

    raise Exception('No trainer known for this plan environment type!')


NO_MODEL_KEYWORDS = ['default', 'none']
VALUE_MODEL_KEYWORDS = ['v', 'V', 'value']
ACTION_VALUE_MODEL_KEYWORDS = ['q', 'Q', 'action-value']
DUMMY_VALUE_MODEL_KEYWORDS = ['debug', 'dummy']


def init_value_model_from_args(args: argparse.Namespace, ctx: dict):
    """
    Build a model from parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a torch.nn.Module built from the parsed arguments
    """
    model_name = args.value_model

    if model_name in NO_MODEL_KEYWORDS:
        return None
    if model_name in VALUE_MODEL_KEYWORDS:
        env = ctx['env']
        device = ctx['device']
        return VModel(
            env.observation_shape,
            args
        ).to(device)
    if model_name in ACTION_VALUE_MODEL_KEYWORDS:
        env = ctx['env']
        device = ctx['device']
        return QModel(
            env.observation_shape,
            env.action_shape,
            args
        ).to(device)
    if model_name in DUMMY_VALUE_MODEL_KEYWORDS:
        env = ctx['env']
        device = ctx['device']
        return DummyModel(env.observation_shape,
                          args).to(device)

    raise Exception('No suitable model found!')


def init_value_model_trainer_from_args(args: argparse.Namespace, ctx: dict):
    """
    Build a value function model trainer based on the parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a suitable trainer object for training the value model (None if no value model is used)
    """
    model_name = args.value_model

    if model_name in NO_MODEL_KEYWORDS:
        return None
    if model_name in VALUE_MODEL_KEYWORDS:
        dataset = ctx['dataset']
        return ValueTrainer(dataset, args)
    if model_name in ACTION_VALUE_MODEL_KEYWORDS:
        dataset = ctx['dataset']
        return ValueTrainer(dataset, args)
    if model_name in DUMMY_VALUE_MODEL_KEYWORDS:
        dataset = ctx['dataset']
        return ValueTrainer(dataset, args)

    raise Exception('Unrecognized value function model setting')


def init_planner_from_args(args: argparse.Namespace, ctx: dict) -> Planner:
    """
    Build a planner from parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a Planner built from the parsed arguments
    """
    name = args.planner

    if name == 'cem':
        return CEM(args)
    if name == 'qcem':
        assert args.value_model in ACTION_VALUE_MODEL_KEYWORDS
        model = ctx['model']
        return QCEM(model, args)
    if name == 'vcem':
        assert args.value_model in VALUE_MODEL_KEYWORDS
        model = ctx['model']
        return VCEM(model, args)
    if name in ['debug', 'random']:
        return RandomPlanner()
    raise Exception('Planner name not recognized!')


def init_eval_planner_from_args(args: argparse.Namespace, ctx: dict) -> Planner:
    """
    Build a planner from parsed arguments
    :param args: an argparse.Namespace object containing parsed arguments
    :param ctx: dict containing the scope of the main program
                (might be required when initializing the objects)
    :return: a Planner built from the parsed arguments
    """
    name = args.eval_planner

    # Eval planner hyperparameters are hardcoded:
    args_ = argparse.Namespace()
    args_.planning_horizon = 12
    args_.num_plan_iter = 10
    args_.num_plan_candidates = 100
    args_.num_plan_top_candidates = 10
    args_.plan_batch_size = 1

    if name == 'cem':
        return CEM(args_)
    if name == 'qcem':
        assert args.value_model in ACTION_VALUE_MODEL_KEYWORDS
        model = ctx['model']
        return QCEM(model, args_)
    if name == 'vcem':
        assert args.value_model in VALUE_MODEL_KEYWORDS
        model = ctx['model']
        return VCEM(model, args_)
    if name in ['debug', 'random']:
        return RandomPlanner()
    raise Exception('Planner name not recognized!')


