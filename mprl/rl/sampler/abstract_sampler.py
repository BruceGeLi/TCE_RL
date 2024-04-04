import multiprocessing
from abc import ABC
from abc import abstractmethod


class AbstractSampler(ABC):
    def __init__(self, *args, **kwargs):
        self.train_envs = None
        self.test_envs = None
        self.debug_env = None

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the sampler to collect some data
        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            rollout results
        """
        pass

    @property
    def num_cpus(self):
        return multiprocessing.cpu_count()

    @property
    def observation_space(self):
        return self.debug_env.observation_space

    @property
    def observation_shape(self):
        return self.observation_space.shape

    @property
    def action_space(self):
        return self.debug_env.action_space

    @property
    def spec(self):
        return self.debug_env.unwrapped.envs[0].spec
