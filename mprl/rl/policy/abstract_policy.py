from abc import ABC
from abc import abstractmethod

import torch
from mprl.util import MLP, TrainableVariable

import mprl.util as util


class AbstractGaussianPolicy(ABC):
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

        # Net input and output
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.contextual_cov = variance_net_args["contextual"]
        self.std_only = variance_net_args["std_only"]
        del variance_net_args["contextual"]
        del variance_net_args["std_only"]

        # Net hidden size
        self.mean_net_args = mean_net_args
        self.variance_net_args = variance_net_args

        # Net initialization
        self.init_method = init_method
        self.out_layer_gain = out_layer_gain

        # Net activation function
        self.act_func_hidden = act_func_hidden
        self.act_func_last = act_func_last

        # dtype and device
        self.dtype, self.device = util.parse_dtype_device(dtype, device)

        # Mean and variance Net
        self.mean_net = None
        self.variance_net = None
        self._create_network()

        # action dof
        self.num_dof = dim_out

        # Minimum std
        self.min_std = float(kwargs.get("min_std", 1e-2))

    @property
    def _policy_net_type(self) -> str:
        """
        Returns: string of policy net type
        """
        return self.__class__.__name__

    def _create_network(self):
        """
        Create policy net with given configuration

        Returns:
            None
        """

        # Two separate value heads: mean_val_net + cov_val_net
        self.mean_net = MLP(name=self._policy_net_type + "_mean",
                            dim_in=self.dim_in,
                            dim_out=self.dim_out,
                            hidden_layers=util.mlp_arch_3_params(
                                **self.mean_net_args),
                            init_method=self.init_method,
                            out_layer_gain=self.out_layer_gain,
                            act_func_hidden=self.act_func_hidden,
                            act_func_last=self.act_func_last,
                            dtype=self.dtype,
                            device=self.device)

        # compute the output dimension of variance
        if self.std_only:
            # Only has diagonal elements
            dim_out_var = self.dim_out
        else:
            # Diagonal + Non-diagonal elements,
            # form up Cholesky Decomposition
            dim_out_var = \
                self.dim_out + (self.dim_out * (self.dim_out - 1)) // 2

        if self.contextual_cov:
            self.variance_net = MLP(name=self._policy_net_type + "_variance",
                                    dim_in=self.dim_in,
                                    dim_out=dim_out_var,
                                    hidden_layers=util.mlp_arch_3_params(
                                        **self.variance_net_args),
                                    init_method=self.init_method,
                                    out_layer_gain=self.out_layer_gain,
                                    act_func_hidden=self.act_func_hidden,
                                    act_func_last=self.act_func_last,
                                    dtype=self.dtype,
                                    device=self.device)

        else:
            # Variance is a trainable variable instead of a trainable network
            variance_vector = torch.zeros(dim_out_var,
                                          dtype=self.dtype, device=self.device)
            variance_vector[:self.dim_out] += \
                util.reverse_from_softplus_space(
                    torch.ones(self.dim_out, dtype=self.dtype,
                               device=self.device), lower_bound=None)

            self.variance_net = TrainableVariable(
                self._policy_net_type + "_variance", variance_vector)

    @property
    def network(self):
        """
        Return policy network

        Returns:
        """
        return self.mean_net, self.variance_net

    @property
    def parameters(self) -> []:
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.mean_net.parameters()) + \
            list(self.variance_net.parameters())

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.mean_net.save(log_dir, epoch)
        self.variance_net.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.mean_net.load(log_dir, epoch)
        self.variance_net.load(log_dir, epoch)

    def _vector_to_cholesky(self, cov_val: torch.Tensor):
        """
        Divide diagonal and off-diagonal elements of cov-net output,
        apply reverse "Log-Cholesky to diagonal elements"
        Args:
            cov_val: output of covariance network

        Returns: params_L

        """
        # Decompose diagonal and off-diagonal elements
        diag_cov_val = cov_val[..., :self.dim_out]
        off_diag_cov_val = None if self.std_only \
            else cov_val[..., self.dim_out:]

        # De-parametrize Log-Cholesky for diagonal elements
        diag_cov_val = util.to_softplus_space(diag_cov_val,
                                              lower_bound=self.min_std)
        params_L = util.build_lower_matrix(diag_cov_val, off_diag_cov_val)

        # Return
        return params_L

    def _cholesky_to_vector(self, params_L: torch.Tensor):
        diag_cov_val, off_diag_cov_val = \
            util.reverse_build_matrix(params_L, not self.std_only)
        diag_cov_val = util.reverse_from_softplus_space(
            diag_cov_val, lower_bound=self.min_std)
        if self.std_only:
            return diag_cov_val
        else:
            return torch.cat([diag_cov_val, off_diag_cov_val], dim=-1)

    @abstractmethod
    def policy(self, *args, **kwargs):
        """
        Compute the pi(action | state)

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            probability of taking an action

        """
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Sample action with re-parametrization trick
        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            sampled action

        """
        pass

    @abstractmethod
    def log_prob(self, *args, **kwargs):
        """
        Compute the log probability of the sampled action
        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            log probability

        """
        pass

    @abstractmethod
    def entropy(self, *args, **kwargs):
        """
        Compute the entropy of the policy

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            entropy: entropy of the policy
        """
        pass

    @abstractmethod
    def covariance(self, *args, **kwargs):
        """
        Compute the covariance of the policy

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            covariance: covariance of the policy
        """
        pass

    @abstractmethod
    def log_determinant(self, *args, **kwargs):
        """
        Compute the log_determinant of the policy

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            log_determinant: log_determinant of the policy
        """
        pass

    @abstractmethod
    def precision(self, *args, **kwargs):
        """
        Compute the precision of the policy

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            precision: precision of the policy
        """
        pass

    @abstractmethod
    def maha(self, *args, **kwargs):
        """
        Compute the mahalanbis distance of the policy

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            maha:  Mahalanobis distance of the policy
        """
        pass

    @property
    def contextual_std(self):
        return self.contextual_cov

    @property
    def contextual(self):
        return True

    @property
    def is_diag(self):
        return self.std_only

    def set_cov_variable(self, param_L: torch.Tensor):
        assert self.contextual_std is False,\
            "Variance is a net instead of a variable."
        self.variance_net.variable.data = \
            self._cholesky_to_vector(param_L).detach()
