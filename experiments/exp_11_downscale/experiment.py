import argparse

from thesis.environment.env_gym import GYM_ENVS

DESCRIPTION = "The Cross-Entropy Method is used to solve the environments in a learned environment model. A learned " \
              "environment model is used. While the model was trained, data augmentation was used."

ARGS = argparse.Namespace(

    desc=DESCRIPTION,

    disable_cuda=False,
    disable_training=False,
    disable_data_collection=False,
    evaluation_period=50,
    checkpoint_period=10,
    num_main_loops=200,
    planner='random',
    eval_planner='cem',
    plan_env_type='rssm',
    value_model='none',
    downscale_observations=True,

    num_seed_episodes=10,
    num_data_episodes=1,
    train_batch_size=32,

    log_directory='./logs/log_experiment_11',
    print_log=True,

    environment_name=GYM_ENVS[0],
    max_episode_length=200,
    state_observations=False,
    env_batch_size=1,
    bit_depth=5,

    max_episodes_buffer=1000,

    deterministic_state_size=32,
    stochastic_state_size=16,
    reward_model_size=32,
    state_model_size=32,
    encoder_model_size=32,
    state_model_min_std=0.1,
    encoding_size=256,
    rssm_sample_mean=True,

    num_train_sequences=50,
    rssm_optimizer_lr=1e-3,
    rssm_optimizer_epsilon=1e-4,
    free_nats=3,
    grad_clip_norm=1000,
    rssm_reward_loss_weight=1,
    rssm_observation_loss_weight=1,
    rssm_kl_loss_weight=1,
    data_augmentations=['random_translate'],
    state_action_augmentations=[],

)

if __name__ == '__main__':
    from thesis.main import run
    from copy import deepcopy

    _args = deepcopy(ARGS)

    _args.log_directory = '../../logs/test'

    run(_args)
