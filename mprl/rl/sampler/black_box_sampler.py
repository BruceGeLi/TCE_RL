import numpy as np
import torch

import mprl.util as util
from mprl.rl.critic import AbstractCritic
from mprl.rl.policy import AbstractGaussianPolicy
from mprl.rl.sampler.abstract_sampler import AbstractSampler
from mprl.util import assert_shape
from mprl.util import to_np
from mprl.util import to_ts
from mprl.util import make_bb_vec_env


class BlackBoxSampler(AbstractSampler):
    def __init__(self,
                 env_id: str,
                 num_env_train: int = 1,
                 num_env_test: int = 1,
                 episodes_per_train_env: int = 1,
                 episodes_per_test_env: int = 1,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 seed: int = 1,
                 **kwargs):
        super().__init__()

        # Environment
        self.env_id = env_id

        self.num_env_train = num_env_train
        self.num_env_test = num_env_test
        self.episodes_per_train_env = episodes_per_train_env
        self.episodes_per_test_env = episodes_per_test_env
        if kwargs.get("mp") is not None:
            self.mp_args = kwargs["mp"]["args"]
        else:
            self.mp_args = dict()

        self.dtype, self.device = util.parse_dtype_device(dtype, device)
        self.seed = seed

        # Map env to cpu cores
        self.cpu_cores = kwargs.get("cpu_cores", None)

        # Logging the task specified metrics
        self.task_specified_metrics = kwargs.get("task_specified_metrics", None)

        # Render the test env
        self.render_test_env = kwargs.get("render_test_env", False)

        # Get training and testing environments
        self.train_envs = self.get_env(env_type="training")
        self.test_envs = self.get_env(env_type="testing")

        # Get one more environment for debugging
        self.debug_env = self.get_env(env_type="debugging")

    def get_env(self, env_type: str = "training"):
        """
        Get training and testing environments

        Args:
            env_type: flag of training, testing or debugging

        Returns:
            environments
        """
        render = False

        if env_type == "training":
            num_env = self.num_env_train
            seed = self.seed
        elif env_type == "testing":
            num_env = self.num_env_test
            seed = self.seed + 10000
            render = self.render_test_env
        elif env_type == "debugging":
            num_env = 1
            seed = self.seed + 20000
        else:
            raise ValueError("Unknown env_type: {}".format(env_type))

        # Make envs
        envs = make_bb_vec_env(env_id=self.env_id, num_env=num_env,
                               seed=seed, render=render, mp_args=self.mp_args)

        # Map env to cpu cores to avoid cpu conflicts in HPC
        # util.assign_env_to_cpu(num_env, envs, self.cpu_cores)

        return envs

    @torch.no_grad()
    def run(self,
            training: bool,
            policy: AbstractGaussianPolicy,
            critic: AbstractCritic,
            deterministic: bool = False,
            render: bool = False,
            task_specified_metrics: list = None):
        """
        Sample trajectories

        Args:
            training: True for training, False for evaluation
            policy: policy model to get actions from
            critic: critic model to get values from
            deterministic: evaluation only, if the evaluation is deterministic
            render: evaluation only, whether render the environment
            task_specified_metrics: task specific metrics

        Returns:
            rollout results
        """
        # Training or evaluation
        if training:
            assert deterministic is False
            envs = self.train_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_train
            ep_per_env = self.episodes_per_train_env
        else:
            envs = self.test_envs
            num_env = self.num_env_test
            episode_init_state = envs.reset()
            if render and num_env == 1:
                envs.render()

            ep_per_env = self.episodes_per_test_env

        # Determine the dimensions
        dim_obs = self.observation_space.shape[-1]
        dim_mp_params = policy.dim_out

        # Storage for rollout results
        list_episode_state = list()
        list_episode_action = list()
        list_episode_reward = list()
        list_episode_done = list()
        list_episode_value = list()

        # Storage for policy results
        list_episode_log_prob = list()  # Policy log probability
        list_episode_params_mean = list()  # Policy mean
        list_episode_params_L = list()  # Policy covariance cholesky

        # Storage task specified metrics
        if self.task_specified_metrics is not None:
            dict_task_specified_metrics = dict()
            for metric in self.task_specified_metrics:
                dict_task_specified_metrics[metric] = list()
        else:
            dict_task_specified_metrics = dict()

        # Env interaction steps
        num_total_env_steps = 0

        # Main rollout loop
        for ep_idx in range(ep_per_env):

            # Initial conditions
            episode_init_state = to_ts(episode_init_state,
                                       self.dtype, self.device)

            # Policy prediction
            episode_params_mean, episode_params_L = \
                policy.policy(episode_init_state)

            assert_shape(episode_params_mean, [num_env, dim_mp_params])
            assert_shape(episode_params_L,
                         [num_env, dim_mp_params, dim_mp_params])
            list_episode_params_mean.append(episode_params_mean)
            list_episode_params_L.append(episode_params_L)

            episode_action = policy.sample(require_grad=False,
                                           params_mean=episode_params_mean,
                                           params_L=episode_params_L,
                                           use_mean=deterministic)

            episode_log_prob = policy.log_prob(episode_action,
                                               params_mean=episode_params_mean,
                                               params_L=episode_params_L)

            assert_shape(episode_action, [num_env, dim_mp_params])
            assert_shape(episode_log_prob, [num_env])
            list_episode_action.append(episode_action)
            list_episode_log_prob.append(episode_log_prob)

            # Values V(s0)
            values = critic.critic(episode_init_state).squeeze(-1)
            assert_shape(episode_init_state, [num_env, dim_obs])
            assert_shape(values, [num_env])
            list_episode_state.append(episode_init_state)
            list_episode_value.append(values)

            # Observation, reward, done, info
            # Here, the gymnasium step() interface get suppressed by sb3
            # So we get 4 return elements rather than 5
            episode_init_state, episode_reward, episode_done, episode_info = (
                envs.step(to_np(episode_action)))

            # Episode rewards
            assert_shape(episode_reward, [num_env])
            episode_reward = to_ts(np.asarray(episode_reward),
                                   self.dtype, self.device)
            list_episode_reward.append(episode_reward)

            # Episode dones
            episode_done = to_ts(np.asarray(episode_done),
                                 torch.bool, self.device)
            assert_shape(episode_done, [num_env])
            list_episode_done.append(episode_done)

            # Update training steps
            episode_length = util.get_item_from_dicts(
                episode_info, "trajectory_length")

            num_total_env_steps += np.asarray(episode_length).sum()

            # Task specified metrics
            if self.task_specified_metrics:
                for metric in self.task_specified_metrics:
                    metric_value = \
                        util.get_item_from_dicts(episode_info,
                                                 metric, lambda x: x[-1])

                    metric_value = \
                        to_ts(metric_value, self.dtype, self.device)

                    dict_task_specified_metrics[metric].append(metric_value)

        # Form up return dictionary
        results = dict()
        results["segment_state"] = torch.cat(list_episode_state, dim=0)
        results["segment_action"] = torch.cat(list_episode_action, dim=0)
        results["segment_reward"] = torch.cat(list_episode_reward, dim=0)
        results["segment_value"] = torch.cat(list_episode_value, dim=0)
        results["episode_reward"] = torch.cat(list_episode_reward, dim=0)
        results["segment_done"] = torch.cat(list_episode_done, dim=0)
        results["segment_log_prob"] = torch.cat(list_episode_log_prob, dim=0)
        results["segment_params_mean"] = torch.cat(list_episode_params_mean,
                                                   dim=0)
        results["segment_params_L"] = torch.cat(list_episode_params_L, dim=0)

        if self.task_specified_metrics:
            for metric in dict_task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric],
                                            dim=0)

        return results, num_total_env_steps
