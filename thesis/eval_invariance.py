

"""

    In this test, models are evaluated for their consistency when predicting rewards for state-action pairs that are
    invariant w.r.t. the true environment's reward function

"""

if __name__ == '__main__':

    import torch
    import argparse

    from thesis.util.logging import Log
    from thesis.environment.util import postprocess_observation
    from thesis.rssm.rssm import RSSM
    from thesis.main import load_checkpoint
    from thesis.main_helpers_init import init_env_from_args
    from thesis.data.augmentation.data_augmentation import vertical_flip, negate

    # Modify vertical flip to work on single observations, rather than complete episodes
    vertical_flip_ = lambda _o: vertical_flip(_o.unsqueeze(0)).squeeze(0)

    checkpoint_path = '../logs/test_checkpoint.pth'

    # Number of episodes under which the model is evaluated
    num_episodes = 20

    # Create an argparse.Namespace object for storing program arguments and initialize a log
    args = argparse.Namespace(
        checkpoint_path=checkpoint_path,
        num_episodes=num_episodes,
        log_directory='../logs/eval_invariance',
        print_log=True,
        bit_depth=5,
    )

    # Create a log
    log = Log(args)
    log.log_args(args)
    # Create a log for all reward values
    all_reward_log = 'all_rewards'
    log.create_log(all_reward_log, 'episode',  't', 'r_true', 'r_a', 'r_b', 'diff_a', 'diff_b', 'diff_ab')

    # Get the args from checkpoint
    checkpoint_args = load_checkpoint(checkpoint_path)['args']
    checkpoint_args.downscale_observations = False  # TODO -- remove and update test checkpoint

    # Load two copies of the trained model
    model_a = load_checkpoint(checkpoint_path)['env_model']
    model_b = load_checkpoint(checkpoint_path)['env_model']

    assert isinstance(model_a, RSSM)
    assert isinstance(model_b, RSSM)

    # Initialize environment
    env = init_env_from_args(checkpoint_args, ctx=locals())

    # Run evaluation runs
    for num_iter in range(1, num_episodes + 1):
        # Create a log for this evaluation run
        reward_log = f'iter_{num_iter}_rewards'
        log.create_log(reward_log, 't', 'r_true', 'r_a', 'r_b', 'diff_a', 'diff_b', 'diff_ab')
        observation_log = f'iter_{num_iter}_observations'
        log.create_image_folder(observation_log)

        # Reset the environment models
        model_a.reset()
        model_b.reset()
        # Reset the environment, receiving an initial observation
        o, env_terminated, _ = env.reset()
        t = 0
        # Set the initial belief state of each environment model, using the initial observation
        model_a.posterior_state_belief(o)
        model_b.posterior_state_belief(vertical_flip_(o))
        # Execute an episode with randomly sampled actions
        while not all(env_terminated):
            # Sample a random action
            a = env.sample_random_action()
            # Apply it on the environment to get an observation and reward
            o, r, env_terminated, _ = env.step(a)
            # Create an equivariant o,a pair by flipping the observation and negating the action
            o_ = vertical_flip_(o)
            a_ = negate(a)

            # Apply a to model_a
            o_pred, r_pred, _, _ = model_a.step(a)
            # Apply a_ to model b
            o_pred_, r_pred_, _, _ = model_b.step(a_)

            # Update state belief distributions for both models
            model_a.posterior_state_belief(o)
            model_b.posterior_state_belief(o_)

            # Log the rewards
            log.log_values(
                reward_log,
                t,
                r.item(),
                r_pred.item(),
                r_pred_.item(),
                abs(r - r_pred).item(),
                abs(r - r_pred_).item(),
                abs(r_pred - r_pred_).item(),
            )
            log.log_values(
                all_reward_log,
                num_iter,
                t,
                r.item(),
                r_pred.item(),
                r_pred_.item(),
                abs(r - r_pred).item(),
                abs(r - r_pred_).item(),
                abs(r_pred - r_pred_).item(),
            )

            # Log the observations
            log.log_n_observations(
                observation_log,
                f'observations_t_{t}',
                [postprocess_observation(obs,
                                         args.bit_depth,
                                         dtype=torch.float32)[0] for obs in [o, o_, o_pred, o_pred_]]
            )

            t += 1



    # For num_iter in <Number of tests>
    #     Reset the models
    #     Reset the environment, giving initial observation o_0
    #     Set initial state belief for model_a using o_0
    #     Set initial state belief for model_b using psi(o_0)
    #     For t in range T
    #           sample random action a_t
    #           apply a_t to environment, giving observation o_t+1 and reward r_t+1
    #           apply a_t to model_a, giving predicted observation and reward
    #           apply psi(a_t) to model_b, giving predicted observation and reward
    #           Log (predicted) rewards and observations
    #           Update state belief distributions using o_t+1 and psi(o_t+1)
    #














