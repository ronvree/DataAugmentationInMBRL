

if __name__ == '__main__':
    import argparse

    from thesis.planner.cem import CEM
    from thesis.main import load_checkpoint
    from thesis.main_helpers_init import init_env_from_args

    path = '../logs/log_colab/iter_20_checkpoint.pth'

    checkpoint = load_checkpoint(path)

    args = checkpoint['args']
    rssm = checkpoint['env_model']

    init_state = checkpoint['init_state']

    env = init_env_from_args(args, ctx=locals())
    env.reset()
    env.set_state(init_state)
    rssm.reset()

    planner_args = argparse.Namespace()
    planner_args.planning_horizon = 8
    planner_args.num_plan_candidates = 100
    planner_args.num_plan_top_candidates = 10
    planner_args.num_plan_iter = 10
    planner_args.plan_batch_size = 1
    planner = CEM(planner_args)

    _, info = planner.plan(env, show_progress=True)

    trajectory = info['trajectory']

    print(trajectory.shape)

    pass  # TODO

