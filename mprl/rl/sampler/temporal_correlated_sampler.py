from typing import Union

import numpy as np
import torch

import mprl.util as util
from mprl.rl.critic import AbstractCritic
from mprl.rl.policy import AbstractGaussianPolicy
from mprl.util import assert_shape
from mprl.util import to_np
from mprl.util import to_ts
from mprl.util.util_learning import select_pred_pairs
from mprl.rl.sampler import BlackBoxSampler
from mprl.util import RunningMeanStd


class TemporalCorrelatedSampler(BlackBoxSampler):
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
        super().__init__(env_id,
                         num_env_train,
                         num_env_test,
                         episodes_per_train_env,
                         episodes_per_test_env,
                         dtype,
                         device,
                         seed,
                         **kwargs)

        # Get time step and episode length
        self.dt = self.debug_env.envs[0].dt
        self.num_times = self.debug_env.envs[0].spec.max_episode_steps

        # Step observation normalization
        self.norm_step_obs = kwargs.get("norm_step_obs", False)
        if self.norm_step_obs:
            self.obs_rms = RunningMeanStd(name="obs_rms",
                                          shape=self.observation_space.shape,
                                          dtype=dtype, device=device)
        else:
            self.obs_rms = None

        # Step reward normalization
        self.norm_step_rewards = kwargs.get("norm_step_rewards", False)
        if self.norm_step_rewards:
            self.rwd_rms = RunningMeanStd(name="rwd_rms",
                                          shape=(1,),
                                          dtype=dtype, device=device)
        else:
            self.rwd_rms = None

        # Time step pairs
        self.time_pairs_config = kwargs["time_pairs_config"]
        self.pred_pairs = None

    def get_times(self, init_time, num_times):
        """
        Get time steps for traj generation

        Args:
            init_time: initial time
            num_times: number of time steps

        Returns:
            time sequence in a tensor
        """
        times = util.tensor_linspace(start=init_time + self.dt,
                                     end=init_time + num_times * self.dt,
                                     steps=num_times).T
        return times

    def get_time_pairs(self):
        pred_pairs = util.to_ts(select_pred_pairs(num_all=self.num_times,
                                                  **self.time_pairs_config),
                                torch.long, self.device)
        self.pred_pairs = pred_pairs
        return pred_pairs

    @staticmethod
    def apply_normalization(raw, rms):
        return (raw - rms.mean) / torch.sqrt(rms.var + 1e-8)

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
            assert deterministic is False and render is False
            envs = self.train_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_train
            ep_per_env = self.episodes_per_train_env
        else:
            envs = self.test_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_test
            if render and num_env == 1:
                envs.render()
            ep_per_env = self.episodes_per_test_env

        # Determine the dimensions
        dim_obs = self.observation_space.shape[-1]
        dim_mp_params = policy.dim_out

        num_times = self.num_times
        num_dof = policy.num_dof

        # Choose prediction pairs
        pred_pairs = self.get_time_pairs()
        num_pred_pairs = pred_pairs.shape[0]

        # Storage for rollout results
        list_episode_state = list()
        list_episode_init_time = list()
        list_episode_init_pos = list()
        list_episode_init_vel = list()
        list_episode_reward = list()

        list_step_states = list()
        list_step_actions = list()
        list_step_rewards = list()
        list_step_dones = list()
        list_step_values = list()

        # Storage for policy results
        list_episode_log_prob_estimate = list()  # Policy log probability
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

            episode_init_time = episode_init_state[..., -num_dof * 2 - 1]
            episode_init_pos = episode_init_state[..., -num_dof * 2: -num_dof]
            episode_init_vel = episode_init_state[..., -num_dof:]
            assert_shape(episode_init_time, [num_env])
            assert_shape(episode_init_pos, [num_env, num_dof])
            assert_shape(episode_init_vel, [num_env, num_dof])

            list_episode_init_time.append(episode_init_time)
            list_episode_init_pos.append(episode_init_pos)
            list_episode_init_vel.append(episode_init_vel)

            # Policy prediction
            # Remove the desired position and velocity from observations
            episode_params_mean, episode_params_L = \
                policy.policy(episode_init_state[..., :-num_dof * 2])
            assert_shape(episode_params_mean, [num_env, dim_mp_params])
            assert_shape(episode_params_L,
                         [num_env, dim_mp_params, dim_mp_params])
            list_episode_params_mean.append(episode_params_mean)
            list_episode_params_L.append(episode_params_L)

            # Time to trajectories and log_probabilities
            step_times = self.get_times(episode_init_time, num_times)

            # Sample a trajectory using the predicted MP parameters
            step_actions = policy.sample(require_grad=False,
                                         params_mean=episode_params_mean,
                                         params_L=episode_params_L,
                                         times=step_times,
                                         init_time=episode_init_time,
                                         init_pos=episode_init_pos,
                                         init_vel=episode_init_vel,
                                         use_mean=deterministic)

            # Get the log-prob of the sample trajectory
            # Using the pair-wise log-prob representation
            segment_log_prob_estimate = \
                policy.log_prob(step_actions,
                                params_mean=episode_params_mean,
                                params_L=episode_params_L,
                                times=step_times,
                                init_time=episode_init_time,
                                init_pos=episode_init_pos,
                                init_vel=episode_init_vel,
                                pred_pairs=pred_pairs)

            assert_shape(step_actions, [num_env, num_times, num_dof * 2])
            assert_shape(segment_log_prob_estimate, [num_env, num_pred_pairs])
            list_step_actions.append(step_actions)
            list_episode_log_prob_estimate.append(segment_log_prob_estimate)

            # Observation, reward, done, info
            # Here, the gymnasium step() interface get suppressed by sb3
            # So we get 4 return elements rather than 5
            next_episode_init_state, episode_reward, _, step_infos = \
                envs.step(to_np(step_actions))

            # Step states and values
            step_states = util.get_item_from_dicts(step_infos, "step_states")
            step_states = to_ts(np.asarray(step_states),
                                self.dtype, self.device)
            assert_shape(step_states, [num_env, num_times, dim_obs])

            # Include the initial state
            step_states = \
                torch.cat([episode_init_state[:, None], step_states], dim=-2)

            # Apply running mean std to step obs before feed it into value net
            # Only update during training
            if self.norm_step_obs and training:
                self.obs_rms.update(step_states.view(-1, dim_obs))
                norm_step_states = self.apply_normalization(step_states,
                                                            self.obs_rms)
            else:
                norm_step_states = step_states

            # Remove the desired position and velocity from observations
            step_values = critic.critic(
                norm_step_states[..., :-num_dof * 2]).squeeze(-1)
            assert_shape(step_values, [num_env, num_times + 1])
            list_step_states.append(norm_step_states)
            list_step_values.append(step_values)

            # Update the initial state
            list_episode_state.append(episode_init_state)
            episode_init_state = next_episode_init_state

            # Step rewards
            step_rewards = util.get_item_from_dicts(step_infos, "step_rewards")
            step_rewards = to_ts(np.asarray(step_rewards),
                                 self.dtype, self.device)
            assert_shape(step_rewards, [num_env, num_times])

            # Turn Non-MDP rewards into MDP rewards if necessary
            step_rewards = util.make_mdp_reward(task_id=self.env_id,
                                                step_rewards=step_rewards,
                                                step_infos=step_infos,
                                                dtype=self.dtype,
                                                device=self.device)

            # Store step rewards
            list_step_rewards.append(step_rewards)

            # Episode rewards
            assert_shape(episode_reward, [num_env])
            episode_reward = to_ts(np.asarray(episode_reward),
                                   self.dtype, self.device)
            list_episode_reward.append(episode_reward)

            # Step dones, adapt to new gymnasium interface
            step_terminations = util.get_item_from_dicts(step_infos,
                                                         "step_terminations")
            step_truncations = util.get_item_from_dicts(step_infos,
                                                        "step_truncations")

            step_terminations = to_ts(np.asarray(step_terminations),
                                      torch.bool, self.device)
            step_truncations = to_ts(np.asarray(step_truncations),
                                     torch.bool, self.device)

            step_dones = torch.logical_or(step_terminations, step_truncations)

            assert_shape(step_dones, [num_env, num_times])
            list_step_dones.append(step_dones)

            # Update training steps
            episode_length = util.get_item_from_dicts(
                step_infos, "segment_length")
            num_total_env_steps += np.asarray(episode_length).sum()

            # Task specified metrics
            if self.task_specified_metrics is not None:
                for metric in self.task_specified_metrics:
                    metric_value = \
                        util.get_item_from_dicts(step_infos,
                                                 metric, lambda x: x[-1])

                    metric_value = \
                        to_ts(metric_value, self.dtype, self.device)

                    dict_task_specified_metrics[metric].append(metric_value)

        # Form up return dictionary
        results = dict()
        results["step_actions"] = torch.cat(list_step_actions, dim=0)
        results["segment_log_prob_estimate"] = \
            torch.cat(list_episode_log_prob_estimate, dim=0)
        results["step_states"] = torch.cat(list_step_states, dim=0)[:, :-1]
        results["step_rewards"] = torch.cat(list_step_rewards, dim=0)
        results["segment_state"] = torch.cat(list_episode_state, dim=0)
        results["segment_reward"] = results["step_rewards"].sum(dim=-1)
        results["episode_reward"] = torch.cat(list_episode_reward, dim=0)
        results["step_dones"] = torch.cat(list_step_dones, dim=0)
        results["step_values"] = torch.cat(list_step_values, dim=0)
        results["segment_init_time"] = torch.cat(list_episode_init_time, dim=0)
        results["segment_init_pos"] = torch.cat(list_episode_init_pos, dim=0)
        results["segment_init_vel"] = torch.cat(list_episode_init_vel, dim=0)
        results["step_time_limit_dones"] = (
            torch.zeros_like(results["step_dones"],
                             dtype=torch.bool, device=self.device))
        results["segment_params_mean"] = \
            torch.cat(list_episode_params_mean, dim=0)
        results["segment_params_L"] = torch.cat(list_episode_params_L, dim=0)

        if self.task_specified_metrics:
            for metric in self.task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric],
                                            dim=0)

        return results, num_total_env_steps

    def save_rms(self, log_dir: str, epoch: int):
        if self.norm_step_obs:
            self.obs_rms.save(log_dir, epoch)
        if self.norm_step_rewards:
            self.rwd_rms.save(log_dir, epoch)

    def load_rms(self, log_dir: str, epoch: int):
        if self.norm_step_obs:
            self.obs_rms.load(log_dir, epoch)
        if self.norm_step_rewards:
            self.rwd_rms.load(log_dir, epoch)
