from abc import ABC
from abc import abstractmethod

from mprl import util
from mprl.util import MLP


class AbstractCritic(ABC):
    def __init__(self, dim_in: int,
                 dim_out: int,
                 hidden: dict,
                 init_method: str,
                 out_layer_gain: float,
                 act_func_hidden: str,
                 act_func_last: str,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 **kwargs):
        # Net config
        self.dim_in: int = dim_in
        self.dim_out: int = dim_out

        self.hidden: dict = hidden
        self.init_method = init_method
        self.out_layer_gain = out_layer_gain

        self.act_func_hidden = act_func_hidden
        self.act_func_last = act_func_last

        # dtype and device
        self.dtype, self.device = util.parse_dtype_device(dtype, device)

        # Mean and variance Net
        self.net = None

        self._create_network()

    @property
    def _critic_net_type(self) -> str:
        """
        Returns: string of critic net type
        """
        return self.__class__.__name__

    def _create_network(self):
        """
        Create critic net with given configuration

        Returns:
            None
        """

        # Two separate value heads: mean_val_net + cov_val_net
        self.net = MLP(name=self._critic_net_type,
                       dim_in=self.dim_in,
                       dim_out=self.dim_out,
                       hidden_layers=util.mlp_arch_3_params(**self.hidden),
                       init_method=self.init_method,
                       out_layer_gain=self.out_layer_gain,
                       act_func_hidden=self.act_func_hidden,
                       act_func_last=self.act_func_last,
                       dtype=self.dtype,
                       device=self.device)

    @property
    def network(self):
        """
        Return critic network

        Returns:
        """
        return self.net

    @property
    def parameters(self) -> []:
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.net.parameters())

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.net.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.net.load(log_dir, epoch)

    @abstractmethod
    def critic(self, *args, **kwargs):
        """
        Compute the value given state

        Args:
            args: input arguments
            kwargs: keyword arguments

        Returns:
            value of state

        """
        pass
