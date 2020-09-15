
from thesis.planner.planner import Planner


class RandomPlanner(Planner):  # For debugging purposes

    def plan(self, env, *args, **kwargs) -> tuple:
        return env.sample_random_action(), {}
