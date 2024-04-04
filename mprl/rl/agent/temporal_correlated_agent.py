import numpy as np
import torch

import mprl.rl.critic.abstract_critic as abs_critic
import mprl.rl.policy.abstract_policy as abs_policy
import mprl.rl.sampler.abstract_sampler as abs_sampler
import mprl.util as util
from mprl.rl.agent import AbstractAgent
from trust_region_projections.utils.projection_utils import gaussian_kl_details


class TemporalCorrelatedAgent(AbstractAgent):
    def __init__(self,
                 policy: abs_policy.AbstractGaussianPolicy,
                 critic: abs_critic.AbstractCritic,
                 sampler: abs_sampler.AbstractSampler,
                 projection,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 **kwargs):
        super().__init__(policy, critic, sampler, projection,
                         dtype=dtype, device=device, **kwargs)
        self.clip_critic = float(kwargs.get("clip_critic", 0.0))
        self.clip_grad_norm = float(kwargs.get("clip_grad_norm", 0.0))
        self.num_minibatchs = kwargs.get("num_minibatchs", 10)
        self.norm_advantages = kwargs.get("norm_advantages", False)
        self.clip_advantages = kwargs.get("clip_advantages", False)
        self.entropy_penalty_coef = float(kwargs.get("entropy_penalty_coef",
                                                     0.0))

        self.use_gae = kwargs.get("use_gae", True)
        self.gae_scaling = float(kwargs.get("gae_scaling", 0.95))
        self.segment_advantage = kwargs.get("segment_advantage", "accumulate")
        self.set_variance = kwargs.get("set_variance", False)
        self.balance_check = kwargs.get("balance_check", 10)
        self.evaluation_interval = kwargs.get("evaluation_interval", 1)

    def step(self):
        # Update total step count
        self.num_iterations += 1

        # Collect dataset
        util.run_time_test(lock=True, key="sampling")
        dataset, num_env_interation = \
            self.sampler.run(training=True, policy=self.policy,
                             critic=self.critic)
        self.num_global_steps += num_env_interation
        sampling_time = util.run_time_test(lock=False, key="sampling")

        # Process dataset
        util.run_time_test(lock=True, key="process_dataset")
        dataset = self.process_dataset(dataset)
        process_dataset_time = util.run_time_test(lock=False,
                                                  key="process_dataset")
        dataset_stats = \
            util.generate_many_stats(dataset, "exploration", to_np=True)

        # Update agent
        util.run_time_test(lock=True, key="update")

        util.run_time_test(lock=True, key="update critic")
        critic_loss_dict = self.update_critic(dataset)
        if self.schedule_lr_critic:
            self.critic_lr_scheduler.step()
        update_critic_time = util.run_time_test(lock=False, key="update critic")

        util.run_time_test(lock=True, key="update policy")
        policy_loss_dict = self.update_policy(dataset)
        if self.schedule_lr_policy:
            self.policy_lr_scheduler.step()
        update_policy_time = util.run_time_test(lock=False, key="update policy")

        update_time = util.run_time_test(lock=False, key="update")

        result_metrics = {
            **dataset_stats, **critic_loss_dict, **policy_loss_dict,
            "sampling_time": sampling_time,
            "process_dataset_time": process_dataset_time,
            "update_time": update_time,
            "update_critic_time": update_critic_time,
            "update_policy_time": update_policy_time,
            "num_global_steps": self.num_global_steps,
            "lr_policy": self.policy_lr_scheduler.get_last_lr()[0]
            if self.schedule_lr_policy else self.lr_policy,
            "lr_critic": self.critic_lr_scheduler.get_last_lr()[0]
            if self.schedule_lr_critic else self.lr_critic
        }

        # Evaluate agent
        if self.evaluation_interval == 1 or\
                self.num_iterations % self.evaluation_interval == 1:
            util.run_time_test(lock=True)
            evaluate_metrics = util.generate_many_stats(self.evaluate()[0],
                                                        "evaluation",
                                                        to_np=True)
            evaluation_time = util.run_time_test(lock=False)
            result_metrics.update(evaluate_metrics),
            result_metrics.update({"evaluation_time": evaluation_time}),

        return result_metrics

    def process_dataset(self, dataset):
        rewards = dataset["step_rewards"]
        old_values = dataset["step_values"]
        dones = dataset["step_dones"]
        time_limit_dones = dataset["step_time_limit_dones"]
        pred_pairs = self.sampler.pred_pairs

        advantages, returns = self.get_advantage_return(rewards, old_values,
                                                        dones, time_limit_dones)
        dataset["step_advantages"] = advantages
        dataset["step_returns"] = returns
        segment_advantage = self.get_segment_advantage(rewards, old_values,
                                                       advantages, pred_pairs)
        dataset["segment_advantage"] = segment_advantage
        return dataset

    def get_advantage_return(self, rewards, values, dones, time_limit_dones):
        """
        Directly adapt from TRPL code
        GAE style advantages and return computation
        Args:
            rewards: dataset rewards
            values: estimated values
            dones: flags of termination
            time_limit_dones: flags of reaching max horizon

            Shape of rewards:
            [num_env, num_times]

            Shape of values:
            [num_env, num_times + 1]

            Shape of dones:
            [num_env, num_times], dtype = bool

            Shape of time_limit_dones:
            [num_env, num_times], dtype = bool

        Returns:
            adv: advantages, shape: [num_env, num_times]
            ret: returns, shape: [num_env, num_times]

        Cases of done and time_limit_done:
        +-----------------+---------------------------+--------------------+
        |                 | Done=True                 | Done=False         |
        +=================+===========================+====================+
        | Time_done=True  | Done due to time limit    | ------------------ |
        +-----------------+---------------------------+--------------------+
        | Time_done=False | Done due to other reasons | intermediate steps |
        +-----------------+---------------------------+--------------------+
        """
        returns = torch.zeros_like(values)
        not_dones = torch.logical_not(dones)
        not_time_limit_dones = torch.logical_not(time_limit_dones)
        num_times = rewards.shape[1]
        discount = self.discount_factor * not_dones
        if self.use_gae:
            gae = 0
            for step in reversed(range(num_times)):
                td = rewards[..., step] + discount[..., step] * values[
                    ..., step + 1] - values[..., step]
                gae = td + discount[..., step] * self.gae_scaling * gae
                gae = gae * not_time_limit_dones[..., step]
                returns[..., step] = gae + values[..., step]
        else:
            returns[..., -1] = values[..., -1]
            # Loop from last
            # Return = Value if current step is the done step
            # Return = reward + discounted next return otherwise
            for step in reversed(range(num_times)):
                returns[..., step] = \
                    not_time_limit_dones[..., step] * \
                    (rewards[..., step] + discount[..., step] *
                     returns[..., step + 1]) + time_limit_dones[..., step] \
                    * values[..., step]

        returns = returns[..., :-1]
        advantages = returns - values[..., :-1]

        return advantages.clone().detach(), returns.clone().detach()

    def get_segment_advantage(self, rewards, values, advantages, pred_pairs,
                              **kwargs):
        """
        
        Args:
            rewards: step rewards
            values: estimated value functions of each step
            advantages: advantage function of each step
            pred_pairs: prediction pairs
            **kwargs: keyword arguments

            Shape of rewards:
            [num_sample, num_times]

            Shape of values:
            [num_sample, num_times + 1]

            Shape of advantages:
            [num_sample, num_times]

            Shape of pred_pairs:
            [num_pred_pairs, 2]

        Returns:
            segment_advantage: the advantage of each time pair 
            Shape of segment_advantage:
            [num_sample, num_pred_pairs]
        """
        if self.segment_advantage == "accumulate":
            # Normalized Advantages
            if self.norm_advantages:
                advantages = (advantages - advantages.mean()) \
                             / (advantages.std() + 1e-8)

            if self.clip_advantages > 0:
                advantages = torch.clamp(advantages, -self.clip_advantages,
                                         self.clip_advantages)

            # Make the correct shape of accumulated_advantages
            segment_advantage = torch.zeros(advantages.shape[0],
                                            pred_pairs.shape[-2],
                                            dtype=self.dtype,
                                            device=self.device)
            for i, (low, high) in enumerate(pred_pairs):
                segment_advantage[:, i] = torch.sum(
                    advantages[:, low:high + 1], dim=-1)

            if self.norm_advantages:
                segment_advantage = (segment_advantage -
                                     segment_advantage.mean()) \
                                    / (segment_advantage.std() + 1e-8)
            return segment_advantage

        elif self.segment_advantage == "value_subtraction":
            # Start and end index of the segment
            # shape: [num_pred_pairs]
            start_idx = pred_pairs[..., 0]
            end_idx = pred_pairs[..., 1]

            # Length of the segment, w.r.t. index, [num_pred_pairs]
            segment_length = end_idx - start_idx

            # Value function discount, shape: [num_sample, num_pred_pairs]
            discount_end_to_start = self.discount_factor.pow(segment_length)

            # Value function of the start and end
            # shape: [num_sample, num_pred_pairs]
            value_end = values[:, end_idx]
            value_start = values[:, start_idx]

            # Compute segment advantages without using any loop
            idx_vector = torch.arange(0, rewards.shape[-1], 1,
                                      device=self.device)
            # Apply discount to all rewards w.r.t the trajectory start
            discount_sequence = self.discount_factor.pow(idx_vector)
            discount_rewards = rewards * discount_sequence

            # Masks indicate the data between the paired time points
            onehot_masks = torch.logical_and(start_idx[:, None] <= idx_vector,
                                             idx_vector < end_idx[:,
                                                          None])
            onehot_masks = util.to_ts(onehot_masks, self.dtype, self.device)

            # Einsum shape:
            # [num_traj, num_times], [num_pred_pairs, num_times]
            # -> [num_sample, num_pred_pairs]
            accumulated_rewards = \
                torch.einsum('ik,jk->ij', discount_rewards, onehot_masks)

            # Divide the discount from the beginning of the trajectory to the
            # first time point of the pair
            discount_pair_first = self.discount_factor.pow(start_idx)

            # Segment advantage =
            #       Sum[gamma^(t-t_A) * r_t] + gamma^(t_B - t_A) * V(B) - V(A)
            segment_advantage = accumulated_rewards / discount_pair_first + \
                                discount_end_to_start * value_end - value_start

            if self.norm_advantages:
                segment_advantage = (segment_advantage -
                                     segment_advantage.mean()) \
                                    / (segment_advantage.std() + 1e-8)

            return segment_advantage

        elif self.segment_advantage == "accumulated_rewards":
            # Start and end index of the segment
            # shape: [num_pred_pairs]
            start_idx = pred_pairs[..., 0]
            end_idx = pred_pairs[..., 1]

            # Compute segment advantages without using any loop
            idx_vector = torch.arange(0, rewards.shape[-1], 1,
                                      device=self.device)
            # Apply discount to all rewards w.r.t the trajectory start
            discount_sequence = self.discount_factor.pow(idx_vector)
            discount_rewards = rewards * discount_sequence

            # Masks indicate the data between the paired time points
            onehot_masks = torch.logical_and(start_idx[:, None] <= idx_vector,
                                             idx_vector < end_idx[:,
                                                          None])
            onehot_masks = util.to_ts(onehot_masks, self.dtype, self.device)

            # Einsum shape:
            # [num_traj, num_times], [num_pred_pairs, num_times]
            # -> [num_sample, num_pred_pairs]
            accumulated_rewards = \
                torch.einsum('ik,jk->ij', discount_rewards, onehot_masks)
            accumulated_rewards_mean = accumulated_rewards.mean(dim=0)
            # Divide the discount from the beginning of the trajectory to the
            # first time point of the pair
            discount_pair_first = self.discount_factor.pow(start_idx)
            segment_advantage = \
                (accumulated_rewards - accumulated_rewards_mean) \
                / discount_pair_first
            return segment_advantage
        else:
            raise NotImplementedError

    def update_critic(self, dataset: dict):
        """
        update critic network

        Args:
            dataset: dataset dictionary

        Returns:
            critic_loss_dict: critic loss dictionary
        """
        # Break the structure of env and time sequence
        states = util.flatten_batch(dataset["step_states"])
        old_values = util.flatten_batch(dataset["step_values"][:, :-1])
        returns = util.flatten_batch(dataset["step_returns"])

        # critic loss
        critic_loss_raw = []
        critic_grad_norm = []
        clipped_critic_grad_norm = []

        for _ in range(self.epochs_critic):
            splits = util.generate_minibatches(states.shape[0],
                                               self.num_minibatchs)
            for indices in splits:
                sel_states, sel_old_values, sel_returns = \
                    util.select_batch(indices, states, old_values, returns)

                # Remove the desired position and velocity from observations
                values_new = self.critic.critic(
                    sel_states[..., :-self.policy.num_dof * 2]).squeeze(-1)

                critic_loss = self.value_loss(values_new, sel_returns,
                                              sel_old_values)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()

                # Norm clipping
                grad_norm, grad_norm_c = util.grad_norm_clip(
                    self.clip_grad_norm, self.critic_net_params)
                critic_grad_norm.append(grad_norm)
                clipped_critic_grad_norm.append(grad_norm_c)

                self.critic_optimizer.step()
                critic_loss_raw.append(critic_loss.item())

        critic_loss_raw = np.asarray(critic_loss_raw)

        # Get some statistics
        critic_loss_dict = {
            **util.generate_stats(critic_loss_raw, "critic_loss"),
            **util.generate_stats(critic_grad_norm, "critic_grad_norm"),
            **util.generate_stats(clipped_critic_grad_norm,
                                  "clipped_critic_grad_norm"),
            # "critic_loss_raw": critic_loss_raw
        }

        return critic_loss_dict

    def update_policy(self, dataset):
        """
        update policy network

        Args:
            dataset: dataset dictionary

        Returns:
            policy_loss_dict: policy loss dict
        """
        states = dataset["segment_state"]
        actions = dataset["step_actions"]
        log_probs_old = dataset["segment_log_prob_estimate"]
        params_mean_old = dataset["segment_params_mean"]
        params_L_old = dataset["segment_params_L"]
        segment_advantage = dataset["segment_advantage"]

        init_time = dataset["segment_init_time"]
        init_pos = dataset["segment_init_pos"]
        init_vel = dataset["segment_init_vel"]

        times = self.sampler.get_times(init_time, self.sampler.num_times)
        pred_pairs = self.sampler.pred_pairs

        # Loss storage
        policy_loss_raw = []
        surrogate_loss_raw = []
        entropy_loss_raw = []
        trust_region_loss_raw = []

        # Entropy storage
        entropy = []

        # Gradient storage
        policy_grad_norm = []
        clipped_policy_grad_norm = []
        surrogate_grad_norm = []
        trust_region_grad_norm = []

        # Trust Region KL storage
        tr_kl_dict = {"new_old_mean_diff": [],
                      "new_old_cov_diff": [],
                      "new_old_shape_diff": [],
                      "new_old_volume_diff": [],
                      "new_proj_mean_diff": [],
                      "new_proj_cov_diff": [],
                      "new_proj_shape_diff": [],
                      "new_proj_volume_diff": [],
                      "proj_old_mean_diff": [],
                      "proj_old_cov_diff": [],
                      "proj_old_shape_diff": [],
                      "proj_old_volume_diff": []}

        # Runtime_storage
        projection_time_storage = []

        # Set initial entropy value in first step to calculate appropriate
        # entropy decay
        if self.projection.initial_entropy is None:
            self.projection.initial_entropy = \
                self.policy.entropy([params_mean_old, params_L_old]).mean()

        # Initialize new and projected policy using old one
        params_mean_new, params_L_new = params_mean_old, params_L_old
        proj_mean, proj_L = params_mean_old, params_L_old

        if isinstance(self.balance_check, int) \
                and self.num_iterations % self.balance_check == 1:
            self.check_policy_balance = True
        else:
            self.check_policy_balance = False

        # Policy update loop
        for _ in range(self.epochs_policy):
            if self.check_policy_balance:
                ################ Check surrogate loss grad ###############
                # Get new prediction
                # Remove the desired position and velocity from observations
                params_mean_new, params_L_new = \
                    self.policy.policy(states[..., :-self.policy.num_dof * 2])
                # Projection
                proj_mean, proj_L = self.projection(self.policy,
                                                    (params_mean_new,
                                                     params_L_new),
                                                    (params_mean_old,
                                                     params_L_old),
                                                    self.num_iterations)

                # Log-prob new
                log_prob_new = self.policy.log_prob(actions,
                                                    params_mean=proj_mean,
                                                    params_L=proj_L,
                                                    times=times,
                                                    init_time=init_time,
                                                    init_pos=init_pos,
                                                    init_vel=init_vel,
                                                    pred_pairs=pred_pairs)

                # Compute surrogate_loss
                surrogate_loss, _ = self.surrogate_loss(segment_advantage,
                                                        log_prob_new,
                                                        log_probs_old)

                # Update
                self.policy_optimizer.zero_grad(set_to_none=True)
                surrogate_loss.backward()

                # Gradient clipping
                surrogate_grad_norm.append(
                    util.grad_norm_clip(0.0, self.policy_net_params)[0])

                ################ Check trust region loss grad ###############
                # Get new prediction
                # Remove the desired position and velocity from observations
                params_mean_new, params_L_new = \
                    self.policy.policy(states[..., :-self.policy.num_dof * 2])
                # Projection
                proj_mean, proj_L = self.projection(self.policy,
                                                    (params_mean_new,
                                                     params_L_new),
                                                    (params_mean_old,
                                                     params_L_old),
                                                    self.num_iterations)

                # Trust Region loss
                trust_region_loss = \
                    self.projection.get_trust_region_loss(self.policy,
                                                          (params_mean_new,
                                                           params_L_new),
                                                          (proj_mean, proj_L),
                                                          set_variance=
                                                          self.set_variance)

                # Update
                self.policy_optimizer.zero_grad(set_to_none=True)
                trust_region_loss.backward()

                # Gradient clipping
                trust_region_grad_norm.append(
                    util.grad_norm_clip(0.0, self.policy_net_params)[0])

                ################# Finish balance check ###############

            # Get new prediction
            # Remove the desired position and velocity from observations
            params_mean_new, params_L_new = \
                self.policy.policy(states[..., :-self.policy.num_dof * 2])
            # Projection
            util.run_time_test(lock=True, key="projection")
            proj_mean, proj_L = self.projection(self.policy,
                                                (params_mean_new, params_L_new),
                                                (params_mean_old, params_L_old),
                                                self.num_iterations)
            projection_time_storage.append(
                util.run_time_test(lock=False, key="projection"))

            # Log-prob new
            log_prob_new = self.policy.log_prob(actions,
                                                params_mean=proj_mean,
                                                params_L=proj_L,
                                                times=times,
                                                init_time=init_time,
                                                init_pos=init_pos,
                                                init_vel=init_vel,
                                                pred_pairs=pred_pairs)

            # Compute surrogate_loss
            surrogate_loss, surr_stats = self.surrogate_loss(segment_advantage,
                                                             log_prob_new,
                                                             log_probs_old)

            # Log KL of: old-new, old-proj, new-proj
            tr_kl_dict = self.kl_old_new_proj(params_mean_new, params_L_new,
                                              params_mean_old, params_L_old,
                                              proj_mean, proj_L, tr_kl_dict)

            # Entropy penalty loss
            entropy_loss, entropy_stats = self.entropy_loss(proj_mean, proj_L)

            # Trust Region loss
            trust_region_loss = \
                self.projection.get_trust_region_loss(self.policy,
                                                      (params_mean_new,
                                                       params_L_new),
                                                      (proj_mean, proj_L),
                                                      set_variance=
                                                      self.set_variance)
            # Total loss
            if torch.any(torch.isnan(surrogate_loss)):
                print(f"surrogate_loss NAN loss detected")
                raise Exception("NAN surrogate_loss detected")
            if torch.any(torch.isnan(entropy_loss)):
                print(f"entropy_loss NAN loss detected")
                raise Exception("NAN entropy_loss detected")
            if torch.any(torch.isnan(trust_region_loss)):
                print(f"trust_region_loss NAN loss detected")
                raise Exception("NAN trust_region_loss detected")


            policy_loss = surrogate_loss + entropy_loss + trust_region_loss

            # Update
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()

            # Gradient clipping
            grad_norm, grad_norm_c = util.grad_norm_clip(self.clip_grad_norm,
                                                         self.policy_net_params)
            self.policy_optimizer.step()

            # Logging
            surrogate_loss_raw.append(surrogate_loss.item())
            entropy_loss_raw.append(entropy_loss.item())
            trust_region_loss_raw.append(trust_region_loss.item())

            policy_loss_raw.append(policy_loss.item())
            entropy.append(entropy_stats["entropy"])
            policy_grad_norm.append(grad_norm)
            clipped_policy_grad_norm.append(grad_norm_c)

        # Store policy balance info
        if self.check_policy_balance:
            balance = {
                **util.generate_stats(surrogate_grad_norm,
                                      "surrogate_grad_norm"),
                **util.generate_stats(trust_region_grad_norm,
                                      "trust_region_grad_norm"),
            }
            balance["balance_ratio"] = balance["surrogate_grad_norm_mean"] \
                                       / balance["trust_region_grad_norm_mean"]
        else:
            balance = {}

        policy_loss_dict = {
            **util.generate_stats(surrogate_loss_raw, "surrogate_loss"),
            **util.generate_stats(entropy_loss_raw, "entropy_loss"),
            **util.generate_stats(trust_region_loss_raw, "trust_region_loss"),
            **util.generate_stats(policy_loss_raw, "policy_loss"),
            **util.generate_stats(entropy, "entropy"),
            **util.generate_stats(policy_grad_norm, "policy_grad_norm"),
            **util.generate_stats(clipped_policy_grad_norm,
                                  "clipped_policy_grad_norm"),
            **util.generate_many_stats(tr_kl_dict, "projection", True),
            "projection_time": np.sum(projection_time_storage),
            **balance}

        # In case of non-contextual policy, directly set the variance variables
        if self.set_variance and not self.policy.contextual_cov:
            # Remove the desired position and velocity from observations
            params_mean_new, params_L_new = \
                self.policy.policy(states[..., :-self.policy.num_dof * 2])
            # Projection
            proj_mean, proj_L = self.projection(self.policy,
                                                (params_mean_new, params_L_new),
                                                (params_mean_old, params_L_old),
                                                self.num_iterations)
            self.policy.set_cov_variable(proj_L[0].detach())

        return policy_loss_dict

    def kl_old_new_proj(self, params_mean_new, params_L_new,
                        params_mean_old, params_L_old,
                        proj_mean, proj_L,
                        result_dict):
        # KL(new || old)
        new_old_mean_diff, new_old_cov_diff, new_old_shape_diff, new_old_volume_diff = \
            gaussian_kl_details(self.policy,
                                (params_mean_new, params_L_new),
                                (params_mean_old, params_L_old))

        # KL(new || proj)
        new_proj_mean_diff, new_proj_cov_diff, new_proj_shape_diff, new_proj_volume_diff = \
            gaussian_kl_details(self.policy,
                                (params_mean_new, params_L_new),
                                (proj_mean, proj_L))

        # KL(proj || old)
        proj_old_mean_diff, proj_old_cov_diff, proj_old_shape_diff, proj_old_volume_diff = \
            gaussian_kl_details(self.policy,
                                (proj_mean, proj_L),
                                (params_mean_old, params_L_old))
        result_dict["new_old_mean_diff"].append(
            util.to_np(new_old_mean_diff.mean()))
        result_dict["new_old_cov_diff"].append(
            util.to_np(new_old_cov_diff.mean()))
        result_dict["new_old_shape_diff"].append(
            util.to_np(new_old_shape_diff.mean()))
        result_dict["new_old_volume_diff"].append(
            util.to_np(new_old_volume_diff.mean()))
        result_dict["new_proj_mean_diff"].append(
            util.to_np(new_proj_mean_diff.mean()))
        result_dict["new_proj_cov_diff"].append(
            util.to_np(new_proj_cov_diff.mean()))
        result_dict["new_proj_shape_diff"].append(
            util.to_np(new_proj_shape_diff.mean()))
        result_dict["new_proj_volume_diff"].append(
            util.to_np(new_proj_volume_diff.mean()))
        result_dict["proj_old_mean_diff"].append(
            util.to_np(proj_old_mean_diff.mean()))
        result_dict["proj_old_cov_diff"].append(
            util.to_np(proj_old_cov_diff.mean()))
        result_dict["proj_old_shape_diff"].append(
            util.to_np(proj_old_shape_diff.mean()))
        result_dict["proj_old_volume_diff"].append(
            util.to_np(proj_old_volume_diff.mean()))
        return result_dict

    def value_loss(self, values: torch.Tensor, returns: torch.Tensor,
                   old_vs: torch.Tensor):
        """
        Adapt from TROL
        Computes the value function loss.

        When using GAE we have L_t = ((v_t + A_t).detach() - v_{t})
        Without GAE we get L_t = (r(s,a) + y*V(s_t+1) - v_{t}) accordingly.

        Optionally, we clip the value function around the original value of v_t

        Returns:
        Args:
            values: value estimates
            returns: computed returns with GAE or n-step
            old_vs: old value function estimates from behavior policy

        Returns:
            Value function loss
        """

        vf_loss = (returns - values).pow(2)

        if self.clip_critic > 0:
            vs_clipped = old_vs + (values - old_vs).clamp(-self.clip_critic,
                                                          self.clip_critic)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = torch.max(vf_loss, vf_loss_clipped)
        return vf_loss.mean()

    @staticmethod
    def surrogate_loss(advantages, log_prob_new, log_prob_old):
        """
        Computes the surrogate reward for Importance Sampling policy gradient

        Args:
            advantages: advantages
            log_prob_new: the log probabilities from current policy
            log_prob_old: the log probabilities

        Returns:
            surrogate_loss: the surrogate loss
            stats_dict: statistics
        """

        ratio = (log_prob_new - log_prob_old).exp()

        advantage_weighted_log_prob = ratio * advantages
        surrogate_loss = advantage_weighted_log_prob.mean()

        stats_dict = {"imp_smp_ratio": ratio.mean().item()}
        return -surrogate_loss, stats_dict

    def entropy_loss(self, params_mean, params_L):
        entropy = self.policy.entropy([params_mean, params_L]).mean()
        entropy_loss = -self.entropy_penalty_coef * entropy
        stats_dict = {"entropy": entropy.item()}
        return entropy_loss, stats_dict

    def save_agent(self, log_dir: str, epoch: int):
        super().save_agent(log_dir, epoch)
        self.sampler.save_rms(log_dir, epoch)

    def load_agent(self, log_dir: str, epoch: int):
        super().load_agent(log_dir, epoch)
        self.sampler.load_rms(log_dir, epoch)
