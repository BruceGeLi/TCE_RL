from abc import ABC
from abc import abstractmethod

import torch
from torch.optim.lr_scheduler import LinearLR as LinearLR
from mprl import util
from mprl.rl.critic import AbstractCritic
from mprl.rl.policy import AbstractGaussianPolicy
from mprl.rl.sampler.abstract_sampler import AbstractSampler


class AbstractAgent(ABC):
    def __init__(self,
                 policy: AbstractGaussianPolicy,
                 critic: AbstractCritic,
                 sampler: AbstractSampler,
                 projection,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 **kwargs):
        # Agent components
        self.policy = policy
        self.critic = critic
        self.sampler = sampler
        self.projection = projection

        # Data type and device
        self.dtype, self.device = util.parse_dtype_device(dtype, device)

        # Hyperparameter
        self.lr_policy = float(kwargs["lr_policy"])
        self.lr_critic = float(kwargs["lr_critic"])
        self.wd_policy = float(kwargs["wd_policy"])
        self.wd_critic = float(kwargs["wd_critic"])
        self.schedule_lr_policy = kwargs.get("schedule_lr_policy", False)
        self.schedule_lr_critic = kwargs.get("schedule_lr_critic", False)
        self.total_iterations = kwargs.get("total_iterations",
                                           10000)  # Only used for schedule lr
        self.discount_factor = torch.tensor(float(kwargs["discount_factor"]),
                                            dtype=self.dtype,
                                            device=self.device)
        self.epochs_policy = kwargs["epochs_policy"]
        self.epochs_critic = kwargs["epochs_critic"]

        # Policy and Critic network parameters
        self.policy_net_params = None
        self.critic_net_params = None

        # Optimizers
        self.policy_optimizer, self.critic_optimizer = \
            self.get_optimizer(self.policy, self.critic)

        # Learning rate schedulers
        self.policy_lr_scheduler, self.critic_lr_scheduler = \
            self.get_lr_scheduler()

        # Number of iterations
        self.num_iterations = 0

        # Number of total environment interactions
        self.num_global_steps = 0

    def get_optimizer(self, policy, critic):
        """
        Get the policy and critic network optimizers

        Args:
            policy: policy network
            critic: critic network

        Returns:
            two optimizers
        """
        self.policy_net_params = policy.parameters
        self.critic_net_params = critic.parameters
        policy_optimizer = torch.optim.Adam(params=self.policy_net_params,
                                            lr=self.lr_policy,
                                            weight_decay=self.wd_policy)
        critic_optimizer = torch.optim.Adam(params=self.critic_net_params,
                                            lr=self.lr_critic,
                                            weight_decay=self.wd_critic)
        return policy_optimizer, critic_optimizer

    def get_lr_scheduler(self):
        """
        Get the policy and critic network learning rate schedulers

        Returns:
            two learning rate schedulers
        """
        if self.schedule_lr_policy:
            policy_lr_scheduler = \
                LinearLR(self.policy_optimizer,
                         start_factor=1,
                         end_factor=0.01,
                         total_iters=self.total_iterations)
        else:
            policy_lr_scheduler = None
        if self.schedule_lr_critic:
            critic_lr_scheduler = \
                LinearLR(self.critic_optimizer,
                         start_factor=1,
                         end_factor=0.01,
                         total_iters=self.total_iterations)
        else:
            critic_lr_scheduler = None
        return policy_lr_scheduler, critic_lr_scheduler

    def save_agent(self, log_dir: str, epoch: int):
        """
        Save agent to file

        Args:
            log_dir: directory to save
            epoch: training epoch

        Returns:
            None
        """
        # Save policy and critic nets
        self.policy.save_weights(log_dir, epoch)
        self.critic.save_weights(log_dir, epoch)

        # Get save path of the optimizers
        policy_optimizer_path = \
            util.get_training_state_save_path(log_dir, "policy_optimizer",
                                              epoch)
        critic_optimizer_path = \
            util.get_training_state_save_path(log_dir, "critic_optimizer",
                                              epoch)

        # Save optimizers
        with open(policy_optimizer_path, "wb") as policy_opt_file:
            torch.save(self.policy_optimizer.state_dict(), policy_opt_file)
        with open(critic_optimizer_path, "wb") as critic_opt_file:
            torch.save(self.critic_optimizer.state_dict(), critic_opt_file)

    def load_agent(self, log_dir: str, epoch: int):
        """
        Load agent from file
        Args:
            log_dir: directory stored agent
            epoch: training epoch

        Returns:
            None

        """
        # Load policy and critic nets
        self.policy.load_weights(log_dir, epoch)
        self.critic.load_weights(log_dir, epoch)

        # Get save path of the optimizers
        policy_optimizer_path = \
            util.get_training_state_save_path(log_dir, "policy_optimizer",
                                              epoch)
        critic_optimizer_path = \
            util.get_training_state_save_path(log_dir, "critic_optimizer",
                                              epoch)

        # Reset optimizers using the loaded nets
        self.policy_optimizer, self.critic_optimizer = \
            self.get_optimizer(self.policy, self.critic)

        # Load optimizers states
        self.policy_optimizer.load_state_dict(torch.load(policy_optimizer_path))
        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))

        # Learning rate schedulers
        self.policy_lr_scheduler, self.critic_lr_scheduler = \
            self.get_lr_scheduler()

        # Load total number of steps
        self.num_iterations = epoch

    @abstractmethod
    def step(self, *args, **kwargs):
        """
        Take a full training step, including sampling, policy and critic update

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            train_metrics: metrics of training exploration
            evaluate_metrics: metrics of evaluation
        """
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        update policy network

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            None
        """
        pass

    @abstractmethod
    def update_critic(self, *args, **kwargs):
        """
        update critic network

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            None
        """
        pass

    @torch.no_grad()
    def evaluate(self,
                 evaluate_deterministic: bool = True,
                 evaluate_stochastic: bool = False,
                 render: bool = False):
        """
        Evaluates the current policy on the test environments.
        Args:
            evaluate_deterministic: Make policy actions deterministic for
                testing (Can be used jointly with stochastic)
            evaluate_stochastic: Make policy actions stochastic for testing
                (Can be used jointly with deterministic)
            render: Render policy (if applicable)

        Returns:
            evaluation result dict
        """

        if evaluate_deterministic:
            deterministic_result_dict = \
                self.sampler.run(training=False,
                                 policy=self.policy,
                                 critic=self.critic,
                                 deterministic=evaluate_deterministic,
                                 render=render)[0]
        else:
            deterministic_result_dict = dict()
        if evaluate_stochastic:
            stochastic_result_dict = \
                self.sampler.run(training=False,
                                 policy=self.policy,
                                 critic=self.critic,
                                 deterministic=evaluate_stochastic,
                                 render=render)[0]
        else:
            stochastic_result_dict = dict()
        return deterministic_result_dict, stochastic_result_dict
