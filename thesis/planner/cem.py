
import tqdm
import argparse

import torch

from thesis.environment.env import Environment
from thesis.planner.planner import Planner


class CEM(Planner):

    """
    Cross-Entropy Method for policy optimization
    """

    def __init__(self, args: argparse.Namespace):
        assert args.planning_horizon > 0
        assert args.num_plan_iter > 0
        assert args.num_plan_candidates > 1
        assert args.num_plan_top_candidates > 1
        assert args.num_plan_candidates >= args.num_plan_top_candidates
        assert args.plan_batch_size > 0

        self.horizon_distance = args.planning_horizon
        self.num_iter = args.num_plan_iter
        self.num_candidates = args.num_plan_candidates
        self.num_top_candidates = args.num_plan_top_candidates

        self.batch_size = args.plan_batch_size

    def plan(self,
             env: Environment,
             device=torch.device('cpu'),
             show_progress: bool = False,
             ) -> tuple:
        """
        Plan an action using the Cross-Entropy Method
        :param env: the environment in which trajectories can be simulated
        :param device: the device on which the resulting action tensor should be stored
        :param show_progress: if set to true, displays a tqdm progress bar during planning
        :return: a two-tuple containing:
                    - the selected action tensor
                    - info dict
        """
        info = dict()
        # Define size variables
        batch_size = self.batch_size
        action_shape = env.action_shape

        # Initialize action belief parameters
        param_shape = (batch_size, self.horizon_distance) + action_shape
        mean = torch.zeros(param_shape, device=device)
        std = torch.ones(param_shape, device=device)

        # Build planning progress bar if required
        if show_progress:
            iters = tqdm.tqdm(range(self.num_iter),
                              total=self.num_iter,
                              desc='Planner',
                              )
            iters.set_postfix_str(f'iters done 0/{self.num_iter}')
        else:
            iters = range(self.num_iter)

        # Optimize the action belief parameters
        for i in iters:
            # Sample all action sequences of all candidates from the current action belief
            candidates = self._sample_action_candidates(mean, std)
            # Score all candidates based on their estimated return
            returns = self._evaluate_action_candidates(candidates, env)
            # Reparameterize the policy distribution based on the best candidates
            mean, std = self._update_params(candidates, returns)

            # Update progress bar
            if show_progress:
                iters.set_postfix_str(f'iters done {i + 1}/{self.num_iter}')

        if show_progress:
            iters.close()

        # Add 'optimal' trajectory to info
        info['trajectory'] = mean.cpu().detach()
        # Return first action mean (Shape: (batch_size,) + action_shape)
        return mean[:, 0], info

    def _evaluate_action_candidates(self, candidates: torch.Tensor, environment: Environment) -> torch.Tensor:
        """
        Score each of the action sequence candidates
        :param candidates: Tensor containing each of the action sequence candidates
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param environment: Environment in which the action sequences are evaluated
        :return: tensor containing scores for each of the candidates
                    shape: (batch_size, num_candidates)
        """
        # Check which device should be used (cpu/gpu)
        device = candidates.device
        # Get the initial state of the environment
        init_env_state = environment.get_state()
        # Get the batch size
        batch_size = candidates.size(0)
        # Store scores for all candidates
        candidate_returns = list()
        # Loop through all candidate tensors
        for j, candidate in enumerate(candidates.split(1, dim=1)):
            # Remove the redundant candidate dimension of the single candidate action sequence (batch)
            # Shape: (batch_size, horizon_distance,) + action_shape
            candidate = candidate.squeeze(1)
            # Keep track of the total return
            reward_total = torch.zeros(batch_size, 1, device=device)
            # Execute all actions in the environment
            for tau, action in enumerate(candidate.split(1, dim=1)):
                # Remove the redundant planning horizon dimension
                # Shape: (batch_size,) + action_shape
                action = action.squeeze(1)
                # Execute the action
                observation, reward, flag, info = environment.step(action, no_observation=True)

                if observation is not None:  # TODO -- HANDLE THIS IN THE ENVIRONMENT
                    observation = observation.to(device)
                reward = reward.to(device)

                # Add to the total reward
                reward_total += reward.view(-1, 1)
            # Store the return estimate of this candidate
            candidate_returns.append(reward_total)
            # Reset environment for the next candidate
            environment.set_state(init_env_state)
        # Concatenate the returns to one tensor
        returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)
        return returns

    def _update_params(self, candidates: torch.Tensor, scores: torch.Tensor) -> tuple:
        """
        Reparameterize the action belief based on how the action sequence candidates were evaluated
        :param candidates: The action sequence candidates in a torch.Tensor
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param scores: The scoring of each of the action candidates in a torch.Tensor
                            shape: (batch_size,)
        :return: a two-tuple containing the new action belief parameters
        """
        # Get the batch size
        batch_size = candidates.size(0)
        # Select the top K candidate sequences based on the rewards obtained
        # Shape: (batch_size, num_top_candidates)
        top_candidate_ixs = torch.argsort(scores, dim=1)[:, -self.num_top_candidates:]

        # Iterate through each item in the batch
        new_mean = [None] * batch_size
        new_std = [None] * batch_size

        for batch_i, (candidates, top_ixs) in enumerate(zip(candidates, top_candidate_ixs)):

            top_candidates = candidates.index_select(0, top_ixs)  # Shape: (num_top_candidates, horizon_distance) + action_shape

            sample_mean = torch.sum(top_candidates, dim=0) / self.num_top_candidates
            sample_std = torch.sum(torch.abs(top_candidates - sample_mean), dim=0) / (self.num_top_candidates - 1)

            new_mean[batch_i] = sample_mean.unsqueeze(0)
            new_std[batch_i] = sample_std.unsqueeze(0)

        mean = torch.cat(new_mean, dim=0)
        std = torch.cat(new_std, dim=0)

        return mean, std

    def _sample_action_candidates(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Use the given parameterization of the action belief to sample number of action sequence candidates
        :param mean: a tensor containing a batch of mean values for the action parameters
                     Shape: (batch_size, horizon_distance, ) + action_shape
        :param std: a tensor containing a batch of std values for the action parameters
                     Shape: (batch_size, horizon_distance, ) + action_shape
        :return: a tensor containing a batch of candidate action sequences, sampled from the given parameters
                     Shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        """
        # Get the batch size from the parameter tensor
        batch_size = mean.size(0)
        # Create an extra 'candidate' dimension (using unsqueeze)
        # Expand the parameter tensors over this dimension. The other dimensions remain the same
        expanded_mean = mean.unsqueeze(1).expand(batch_size, self.num_candidates, *mean.shape[1:])
        expanded_std = std.unsqueeze(1).expand(batch_size, self.num_candidates, *std.shape[1:])
        # Sample the candidate action sequences
        # Shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        action_candidate_seqs = torch.normal(expanded_mean, expanded_std)
        return action_candidate_seqs

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('Cross-Entropy Method Arguments')

        parser.add_argument('--planning_horizon',
                            type=int,
                            default=12,  # Value in PlaNet paper: 12
                            help='Planning horizon distance')
        parser.add_argument('--num_plan_iter',
                            type=int,
                            default=10,  # Value in PlaNet paper: 10
                            help='Optimization iterations')
        parser.add_argument('--num_plan_candidates',
                            type=int,
                            default=100,  # Value in PlaNet paper: 1000
                            help='Candidates per iteration')
        parser.add_argument('--num_plan_top_candidates',
                            type=int,
                            default=10,  # Value in PlaNet paper: 100
                            help='Number of top candidates to fit')
        parser.add_argument('--plan_batch_size',
                            type=int,
                            default=1,
                            help='Number of environments used simultaneouly for planning')

        return parser


class QCEM(CEM):

    """
    Cross-Entropy Method for policy optimization

    Evaluates action candidates using a learned action-value function
    """

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(args)
        self._model = model

    def plan(self, *args, **kwargs):
        self._model.eval()
        return super(QCEM, self).plan(*args, **kwargs)

    def _evaluate_action_candidates(self, candidates: torch.Tensor, environment: Environment):
        """
        Score each of the action sequence candidates
        :param candidates: Tensor containing each of the action sequence candidates
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param environment: Environment in which the action sequences are evaluated
        :return: tensor containing scores for each of the candidates
                    shape: (batch_size, num_candidates)
        """
        # Check which device should be used (cpu/gpu)
        device = candidates.device
        # Get the initial state of the environment
        init_env_state = environment.get_state()
        # Get the batch size
        batch_size = candidates.size(0)
        # Store scores for all candidates
        candidate_returns = list()
        # Loop through all candidate tensors
        for j, candidate in enumerate(candidates.split(1, dim=1)):
            # Remove the redundant candidate dimension of the single candidate action sequence (batch)
            # Shape: (batch_size, horizon_distance,) + action_shape
            candidate = candidate.squeeze(1)
            # Keep track of the total return
            reward_total = torch.zeros(batch_size, 1, device=device)
            # Separate the tensor into multiple action tensors
            actions = [action.squeeze(1) for action in candidate.split(1, dim=1)]
            # Execute all but the last action in the environment
            for tau, action in enumerate(actions[:-1]):
                # Execute the action
                observation, reward, flag, info = environment.step(action, no_observation=bool(len(actions) - tau - 2))  # TODO -- only get observation at last iter -- neater

                if observation is not None:  # TODO -- HANDLE THIS IN THE ENVIRONMENT
                    observation = observation.to(device)
                reward = reward.to(device)

                # Add to the total reward
                reward_total += reward.view(-1, 1)
            # Use the q-model to estimate the return of the final action
            reward_total += self._model(observation, actions[-1]).to(device)
            # Store the return estimate of this candidate
            candidate_returns.append(reward_total)
            # Reset environment for the next candidate
            environment.set_state(init_env_state)
        # Concatenate the returns to one tensor
        returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)
        return returns


class VCEM(CEM):

    """
    Cross-Entropy Method for policy optimization

    Evaluates action candidates using a learned value function
    """

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(args)
        self._model = model

    def plan(self, *args, **kwargs):
        self._model.eval()
        return super(VCEM, self).plan(*args, **kwargs)

    def _evaluate_action_candidates(self, candidates: torch.Tensor, environment: Environment):
        """
        Score each of the action sequence candidates
        :param candidates: Tensor containing each of the action sequence candidates
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param environment: Environment in which the action sequences are evaluated
        :return: tensor containing scores for each of the candidates
                    shape: (batch_size, num_candidates)
        """
        # Check which device should be used (cpu/gpu)
        device = candidates.device
        # Get the initial state of the environment
        init_env_state = environment.get_state()
        # Get the batch size
        batch_size = candidates.size(0)
        # Store scores for all candidates
        candidate_returns = list()
        # Loop through all candidate tensors
        for j, candidate in enumerate(candidates.split(1, dim=1)):
            # Remove the redundant candidate dimension of the single candidate action sequence (batch)
            # Shape: (batch_size, horizon_distance,) + action_shape
            candidate = candidate.squeeze(1)
            # Keep track of the total return
            reward_total = torch.zeros(batch_size, 1, device=device)
            # Separate the tensor into multiple action tensors
            actions = [action.squeeze(1) for action in candidate.split(1, dim=1)]
            # Execute all but the last action in the environment
            for tau, action in enumerate(actions):
                # Execute the action
                observation, reward, flag, info = environment.step(action, no_observation=bool(len(actions) - tau - 1))  # TODO -- only get observation at last iter -- neater

                if observation is not None:  # TODO -- HANDLE THIS IN THE ENVIRONMENT
                    observation = observation.to(device)
                reward = reward.to(device)

                # Add to the total reward
                reward_total += reward.view(-1, 1)
            # Use the v-model to estimate the return of the final action
            reward_total += self._model(observation).to(device)
            # Store the return estimate of this candidate
            candidate_returns.append(reward_total)
            # Reset environment for the next candidate
            environment.set_state(init_env_state)
        # Concatenate the returns to one tensor
        returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)
        return returns


if __name__ == '__main__':

    from dm_control import viewer

    from thesis.environment.util import build_env_from_args  # TODO -- deprecated
    from thesis.environment.env_suite import CONTROL_SUITE_ENVS
    from thesis.environment.env_gym import GYM_ENVS

    _args = argparse.Namespace()
    # _args.env_name = CONTROL_SUITE_ENVS[0]
    _args.env_name = GYM_ENVS[0]
    _args.env_batch_size = 1
    _args.plan_batch_size = 1
    _args.max_episode_length = 1000
    _args.state_observations = True

    _args.planning_horizon = 20
    _args.num_plan_iter = 5
    _args.num_plan_candidates = 20
    _args.num_plan_top_candidates = int(_args.num_plan_candidates // 1.3)

    _cem = CEM(_args)

    _env = build_env_from_args(_args)

    _env_clone = _env.clone()

    _env.reset()
    _env_clone.reset()

    _env_clone.set_state(_env.get_state())
    _env_clone.set_seed(_env.get_seed())

    with torch.no_grad():

        for _ in range(10000):
            # _env_clone.render()

            _action, _ = _cem.plan(_env_clone)

            _env_clone.step(_action)

            _result = _env.step(_action)

            print(_result[1])

        # _env.reset()
        # viewer.launch(_env._envs[0], policy=lambda t: _cem.plan(deepcopy(_env))[0])


