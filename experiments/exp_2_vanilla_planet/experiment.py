
import argparse


from thesis.environment.env_gym import GYM_ENVS


DESCRIPTION = "The Cross-Entropy Method is used to solve the environments in a learned environment model. No data" \
              " augmentation is used. The data used for training the environment model obtained under a uniformly" \
              " random policy."


def get_args() -> argparse.Namespace:
    args = argparse.Namespace()

    with open('./experiments/exp_2_vanilla_planet/description.txt', 'r') as f:
        args.desc = f.read()

    args.disable_cuda = False
    args.disable_training = False
    args.disable_data_collection = False
    args.evaluation_period = 50
    args.checkpoint_period = 10
    args.planner = 'random'
    args.eval_planner = 'cem'
    args.plan_env_type = 'rssm'
    args.value_model = 'none'

    args.num_seed_episodes = 200
    args.num_data_episodes = 1
    args.train_batch_size = 32

    args.log_directory = './logs/log_experiment_2'
    args.print_log = True

    args.environment_name = GYM_ENVS[0]
    args.max_episode_length = 200
    args.state_observations = False
    args.env_batch_size = 1
    args.bit_depth = 5

    args.max_episodes_buffer = 200

    args.deterministic_state_size = 200
    args.stochastic_state_size = 30
    args.reward_model_size = 200
    args.state_model_size = 200
    args.encoder_model_size = 200
    args.state_model_min_std = 0.1
    args.encoding_size = 1024
    args.rssm_sample_mean = True

    args.num_train_sequences = 50
    args.rssm_optimizer_lr = 1e-3
    args.rssm_optimizer_epsilon = 1e-4
    args.free_nats = 3
    args.grad_clip_norm = 1000
    args.rssm_reward_loss_weight = 1
    args.rssm_observation_loss_weight = 1
    args.rssm_kl_loss_weight = 1
    args.data_augmentations = []

    return args

