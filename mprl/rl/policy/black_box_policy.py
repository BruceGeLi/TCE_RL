from mprl.rl.policy import AbstractGaussianPolicy
from mprl.util.util_nn import *


class BlackBoxPolicy(AbstractGaussianPolicy):
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

    def policy(self, obs):
        """
        Compute the mean cov of the action given state
        Args:
            obs: observation

            Shape of obs:
            [*add_dim, dim_obs]

        Returns:
            mean and cholesky of the MP parameters

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        """
        # Compute mean and variance of the MP's parameters
        params_mean = self.mean_net(obs)
        if self.contextual_cov:
            params_L = self._vector_to_cholesky(self.variance_net(obs))
        else:
            L_vector = util.add_expand_dim(self.variance_net.variable, [0],
                                           [obs.shape[0]])
            params_L = self._vector_to_cholesky(L_vector)
        return params_mean, params_L

    def sample(self, require_grad, params_mean, params_L, use_mean=False):
        """
        Given a segment-wise state, rsample an action
        Args:
            require_grad: require gradient from the samples
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the MP parameters covariance
            use_mean: if True, return the mean action

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

        Returns:
            smp_params: sampled params

            Shape of smp_params:
            [*add_dim, num_dof * num_basis_g]

        """
        if not use_mean:
            # Sample trajectory
            mvn = torch.distributions.MultivariateNormal(
                loc=params_mean, scale_tril=params_L, validate_args=False)
            smp_params = mvn.rsample([])

        else:
            smp_params = params_mean

        # Remove gradient if necessary
        if not require_grad:
            smp_params = smp_params.detach()

        return smp_params

    def log_prob(self, smp_params, params_mean, params_L, **kwargs):
        """
        Compute the log probability of the sampled action
        Args:
            smp_params: sampled params
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the ProDMP parameters covariance
            kwargs: keyword arguments

            Shape of smp_params:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

        Returns:
            log_prob: log probability of sampled parameters.

            Shape of log_prob: [*add_dim]
        """

        # Form up trajectory distribution
        mvn = torch.distributions.MultivariateNormal(loc=params_mean,
                                                     scale_tril=params_L,
                                                     validate_args=False)

        # Compute log probability
        # Shape [*add_dim, num_pred_pairs]
        log_prob = mvn.log_prob(smp_params)

        return log_prob

    def entropy(self, params: [torch.Tensor, torch.Tensor]):
        """
        Compute the entropy of the policy

        Args:
            params: a tuple of mean and cholesky of the ProDMP parameters

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

        Returns:
            entropy: entropy of the policy, scalar
        """
        # Split mean and cholesky
        params_mean, params_L = params

        # Form up a distribution
        mvn = torch.distributions.MultivariateNormal(loc=params_mean,
                                                     scale_tril=params_L,
                                                     validate_args=False)
        entropy = mvn.entropy()
        return entropy

    def covariance(self, params_L: torch.Tensor):
        """
        Compute the covariance of the policy

        Args:
            params_L: cholesky decomposition of the ProDMP parameters covariance

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

        Returns:
            params_cov: covariance of the policy

            Shape of covariance:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        """
        params_cov = torch.einsum('...ij,...kj->...ik', params_L, params_L)
        return params_cov

    def log_determinant(self, params_L: torch.Tensor):
        """
        Compute the log_determinant of the policy

        Args:
            params_L: Cholesky matrix

        Returns:
            log_det: log_determinant of the policy, the determinant of mat,
             aka product of the diagonal
        """
        log_det = 2 * params_L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return log_det

    def precision(self, params_L: torch.Tensor):
        """
        Compute the precision of the policy

        Args:
            params_L: Cholesky matrix

        Returns:
            precision: precision of the policy
        """
        precision = torch.cholesky_solve(torch.eye(params_L.shape[-1],
                                                   dtype=params_L.dtype,
                                                   device=params_L.device),
                                         params_L, upper=False)
        return precision

    def maha(self, params: torch.Tensor, params_other: torch.Tensor,
             params_L: torch.Tensor):
        """
        Compute the Mahalanobis distance of the policy

        Args:
            params:
            params_other:
            params_L: Cholesky matrix

        Returns:
            maha:  Mahalanobis distance of the policy
        """
        diff = (params - params_other)[..., None]

        # A new version of torch.triangular_solve(B, A).solution
        maha = torch.linalg.solve_triangular(params_L,
                                             diff,
                                             upper=False).pow(2).sum([-2, -1])
        return maha
