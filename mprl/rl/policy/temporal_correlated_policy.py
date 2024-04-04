import torch
from mprl.util.util_mp import *
from .black_box_policy import BlackBoxPolicy


class TemporalCorrelatedPolicy(BlackBoxPolicy):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 mean_net_args: dict,
                 variance_net_args: dict,
                 init_method: str,
                 out_layer_gain: float,
                 act_func_hidden: str,
                 act_func_last: str,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 **kwargs):
        super().__init__(dim_in,
                         dim_out,
                         mean_net_args,
                         variance_net_args,
                         init_method,
                         out_layer_gain,
                         act_func_hidden,
                         act_func_last,
                         dtype,
                         device,
                         **kwargs)
        self.mp: ProDMP = get_mp(**kwargs["mp"])
        # self.mp.show_scaled_basis(True)
        self.num_dof = self.mp.num_dof

    def sample(self, require_grad, params_mean, params_L,
               times, init_time, init_pos, init_vel, use_mean=False):
        """
        Given a segment-wise state, rsample an action
        Args:
            require_grad: require gradient from the samples
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the ProDMP parameters covariance
            times: trajectory times points
            init_time: initial condition time
            init_pos: initial condition pos
            init_vel: initial condition vel
            use_mean: if True, return the mean action

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

            Shape of times:
            [*add_dim, num_times]

            Shape of init_time:
            [*add_dim]

            Shape of init_pos:
            [*add_dim, num_dof]

            Shape of init_vel:
            [*add_dim, num_dof]

        Returns:
            smp_pos: sampled traj pos
            smp_vel: sampled traj vel

            Shape of smp_traj:
            [*add_dim, num_times, num_dof * 2]

        """
        if not use_mean:
            # Sample trajectory
            smp_pos, smp_vel = \
                self.mp.sample_trajectories(times=times, params=params_mean,
                                            params_L=params_L,
                                            init_time=init_time,
                                            init_pos=init_pos,
                                            init_vel=init_vel,
                                            num_smp=1, flat_shape=False)

            # squeeze the dimension of sampling
            smp_pos, smp_vel = smp_pos.squeeze(-3), smp_vel.squeeze(-3)

        else:
            smp_pos = self.mp.get_traj_pos(times=times, params=params_mean,
                                           init_time=init_time,
                                           init_pos=init_pos,
                                           init_vel=init_vel, flat_shape=False)
            smp_vel = self.mp.get_traj_vel()

        # Remove gradient if necessary
        if not require_grad:
            smp_pos = smp_pos.detach()
            smp_vel = smp_vel.detach()

        # Concatenate position and velocity
        smp_traj = torch.cat([smp_pos, smp_vel], dim=-1)

        return smp_traj

    def log_prob(self, smp_traj, params_mean, params_L,
                 times, init_time, init_pos, init_vel, **kwargs):
        """
        Compute the log probability of the sampled trajectory
        Args:
            smp_traj: sampled trajectory position and velocity
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the ProDMP parameters covariance
            times: trajectory times points
            init_time: initial condition time
            init_pos: initial condition pos
            init_vel: initial condition vel
            kwargs: keyword arguments

            Shape of smp_traj:
            [*add_dim, num_times, num_dof * 2]

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

            Shape of times:
            [*add_dim, num_times]

            Shape of init_time:
            [*add_dim]

            Shape of init_pos:
            [*add_dim, num_dof]

            Shape of init_vel:
            [*add_dim, num_dof]

        Returns:
            log_prob: log probability in the form of J pairs assembled

            Shape of log_prob: [*add_dim, num_pred_pairs]
        """

        # Get J time pairs
        pred_pairs = kwargs["pred_pairs"]
        num_pred_pairs = pred_pairs.shape[0]

        # Increase dimensionality of all mp inputs to desired ones,
        # Shape of params: [*add_dim, num_dof * num_basis_g]
        # -> [*add_dim, num_pred_pairs, num_dof * num_basis_g]
        params_mean = util.add_expand_dim(params_mean, [-2], [num_pred_pairs])

        # Shape of params_L:
        # Shape of params:
        # [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        # -> [*add_dim, num_pred_pairs, num_dof * num_basis_g, num_dof * num_basis_g]
        params_L = util.add_expand_dim(params_L, [-3], [num_pred_pairs])

        # Shape of time_pairs: [*add_dim, num_times]
        # -> [*add_dim, num_pred_pairs, 2]
        time_pairs = times[:, pred_pairs]

        # Shape of init_time: [*add_dim] -> [*add_dim num_pred_pairs]
        init_time = util.add_expand_dim(init_time, [-1], [num_pred_pairs])

        # Shape of init_pos and init_vel: [*add_dim, num_dof]
        # [*add_dim, num_pred_pairs, num_dof]
        init_pos = util.add_expand_dim(init_pos, [-2], [num_pred_pairs])
        init_vel = util.add_expand_dim(init_vel, [-2], [num_pred_pairs])

        # Extract sampled position
        # Shape of sampled position
        # [*add_dim, num_times, num_dof * 2]
        # -> [*add_dim, num_pred_pairs, 2, num_dof]
        smp_pos = smp_traj[..., pred_pairs, :self.num_dof]

        # Reshape [*add_dim, num_pred_pairs, 2, num_dof]
        # -> [*add_dim, num_pred_pairs, num_dof, 2]
        smp_pos = torch.einsum('...ji->...ij', smp_pos)

        # Couple dof and time dimensions
        # Shape of smp_traj [*add_dim, num_pred_pairs, num_dof, 2]
        # -> [*add_dim, num_pred_pairs, num_dof * 2]
        smp_pos = smp_pos.reshape(*smp_pos.shape[:-2], -1)

        # Get trajectory mean and covariance
        self.mp.update_inputs(times=time_pairs, params=params_mean,
                              params_L=params_L, init_time=init_time,
                              init_pos=init_pos, init_vel=init_vel)
        traj_mean = self.mp.get_traj_pos(flat_shape=True)
        traj_cov = self.mp.get_traj_pos_cov()

        # Form up trajectory distribution
        mvn = torch.distributions.MultivariateNormal(loc=traj_mean,
                                                     covariance_matrix=traj_cov,
                                                     validate_args=False)

        # Compute log probability
        # Shape [*add_dim, num_pred_pairs]
        log_prob = mvn.log_prob(smp_pos)

        return log_prob
