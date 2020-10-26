import argparse

import numpy as np

from thesis.environment.env_gym import GYM_ENVS
from thesis.environment.env_suite import CONTROL_SUITE_ENVS

DESCRIPTION = "The Cross-Entropy Method is used to solve the environments. The true environment model is used."

ARGS = argparse.Namespace(
    desc=DESCRIPTION,

    disable_cuda=True,
    disable_training=True,
    disable_data_collection=True,
    evaluation_period=1,
    checkpoint_period=np.inf,
    num_main_loops=200,
    planner='cem',
    eval_planner='cem',
    plan_env_type='true',
    value_model='none',

    num_seed_episodes=0,
    num_data_episodes=0,
    train_batch_size=0,

    log_directory='../../logs/log_experiment_1_temp',
    print_log=True,

    environment_name=GYM_ENVS[0],
    max_episode_length=200,
    state_observations=False,
    env_batch_size=1,
    bit_depth=5,

    max_episodes_buffer=0,

    planning_horizon=12,
    num_plan_iter=10,
    num_plan_candidates=100,
    num_plan_top_candidates=10,
    plan_batch_size=1,
)

if __name__ == '__main__':
    import copy

    from thesis.main import run

    _args = copy.deepcopy(ARGS)

    run(_args)
