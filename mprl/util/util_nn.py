"""
@brief:     Classes of Neural Network Bases
"""
import pickle as pkl
from typing import Callable
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F

import mprl.util as util


def get_act_func(key: str) -> Optional[Callable]:
    func_dict = dict()
    func_dict["tanh"] = torch.tanh
    func_dict["relu"] = F.relu
    func_dict["leaky_relu"] = F.leaky_relu
    func_dict["softplus"] = F.softplus
    if key is not None:
        return func_dict[key]
    else:
        return None


def initialize_weights(model: nn.Module, initialization_type: str,
                       scale: float = 2 ** 0.5, init_w=3e-3,
                       activation="relu"):
    """
    Weight initializer for the layer or model.
    Args:
        model: module to initialize
        initialization_type: type of inialization
        scale: gain or scale values for normal, xavier, and orthogonal init
        init_w: init weight for normal and uniform init
        activation: required for he init
    Returns:
    """

    for p in model.parameters():
        if initialization_type == "normal":
            if len(p.data.shape) >= 2:
                p.data.normal_(init_w, scale)  # 0.01
            else:
                p.data.zero_()
        elif initialization_type == "uniform":
            if len(p.data.shape) >= 2:
                p.data.uniform_(-init_w, init_w)
            else:
                p.data.zero_()
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_normal_(p.data, gain=scale)
            else:
                p.data.zero_()
        elif initialization_type in ['fan_in', 'fan_out']:
            if len(p.data.shape) >= 2:
                nn.init.kaiming_uniform_(p.data, mode=initialization_type,
                                         nonlinearity=activation)
            else:
                p.data.zero_()

        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError(
                "Not a valid initialization type. Choose one of 'normal', 'uniform', 'xavier', and 'orthogonal'")


class MLP(nn.Module):
    def __init__(self,
                 name: str,
                 dim_in: int,
                 dim_out: int,
                 hidden_layers: list,
                 init_method: str,
                 out_layer_gain: float,
                 act_func_hidden: str,
                 act_func_last: str,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")):
        """
        Multi-layer Perceptron Constructor

        Args:
            name: name of the MLP
            dim_in: dimension of the input
            dim_out: dimension of the output
            hidden_layers: a list containing hidden layers' dimensions
            act_func_hidden: activation function of hidden layer
            act_func_last: activation function of last layer
            dtype: data type
            device: device
        """

        super(MLP, self).__init__()

        self.mlp_name = name + "_mlp"

        # Initialize the MLP
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers
        self.act_func_hidden_type = act_func_hidden
        self.act_func_hidden = get_act_func(act_func_hidden)
        self.act_func_last_type = act_func_last
        self.act_func_last = get_act_func(act_func_last)
        self.init_method = init_method
        self.out_layer_gain = out_layer_gain

        # dtype and device
        self.dtype = dtype
        self.device = device

        # Create networks
        # Ugly but useful to distinguish networks in gradient watch
        # e.g. if self.mlp_name is "encoder_mlp"
        # Then below will lead to self.encoder_mlp = self._create_network()
        setattr(self, self.mlp_name, self._create_network())

    def _create_network(self):
        """
        Create MLP Network

        Returns:
        MLP Network
        """

        # Total layers (n+1) = hidden layers (n) + output layer (1)

        mlp = ModuleList()

        # Add first hidden layer
        input_layer = nn.Linear(in_features=self.dim_in,
                                out_features=self.hidden_layers[0],
                                dtype=self.dtype, device=self.device)
        initialize_weights(input_layer, self.init_method)
        mlp.append(input_layer)

        # Add other hidden layers
        for i in range(1, len(self.hidden_layers)):
            hidden = nn.Linear(in_features=mlp[-1].out_features,
                               out_features=self.hidden_layers[i],
                               dtype=self.dtype, device=self.device)
            initialize_weights(hidden, self.init_method)
            mlp.append(hidden)

        # Add output layer
        output_layer = nn.Linear(in_features=mlp[-1].out_features,
                                 out_features=self.dim_out,
                                 dtype=self.dtype, device=self.device)
        initialize_weights(output_layer, self.init_method,
                           scale=self.out_layer_gain)

        mlp.append(output_layer)

        return mlp

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.mlp_name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "hidden_layers": self.hidden_layers,
                "act_func_hidden_type": self.act_func_hidden_type,
                "act_func_last_type": self.act_func_last_type,
                "dtype": self.dtype,
                "device": self.device
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.mlp_name, epoch)

        # Check structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.dim_in == parameters["dim_in"] \
                   and self.dim_out == parameters["dim_out"] \
                   and self.hidden_layers == parameters["hidden_layers"] \
                   and self.act_func_hidden_type == parameters[
                       "act_func_hidden_type"] \
                   and self.act_func_last_type == parameters[
                       "act_func_last_type"] \
                   and self.dtype == parameters["dtype"] \
                   and self.device == parameters["device"], \
                "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, data):
        """
        Network forward function

        Args:
            data: input data

        Returns: MLP output

        """

        # Hidden layers (n) + output layer (1)
        mlp = eval("self." + self.mlp_name)
        for i in range(len(self.hidden_layers)):
            data = self.act_func_hidden(mlp[i](data))
        if self.act_func_last is not None:
            data = self.act_func_last(mlp[-1](data))
        else:
            data = mlp[-1](data)

        # Return
        return data


class CNNMLP(nn.Module):
    def __init__(self,
                 name: str,
                 image_size: list,
                 kernel_size: int,
                 num_cnn: int,
                 cnn_channels: list,
                 hidden_layers: list,
                 dim_out: int,
                 act_func_hidden: str,
                 act_func_last: str,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")):
        """
        CNN, MLP constructor

        Args:
            name: name of the MLP
            image_size: w and h of input images size
            kernel_size: size of cnn kernel
            num_cnn: number of cnn layers
            cnn_channels: a list containing cnn in and out channels
            hidden_layers: a list containing hidden layers' dimensions
            dim_out: dimension of the output
            act_func_hidden: activation function of hidden layer
            act_func_last: activation function of last layer
            dtype: data type
            device: device
        """
        super(CNNMLP, self).__init__()

        self.name = name
        self.cnn_mlp_name = name + "_cnn_mlp"

        self.image_size = image_size
        self.kernel_size = kernel_size
        assert num_cnn + 1 == len(cnn_channels)
        self.num_cnn = num_cnn
        self.cnn_channels = cnn_channels
        self.dim_in = self.get_mlp_dim_in()
        self.hidden_layers = hidden_layers
        self.dim_out = dim_out
        self.act_func_hidden_type = act_func_hidden
        self.act_func_hidden = get_act_func(act_func_hidden)
        self.act_func_last_type = act_func_last
        self.act_func_last = get_act_func(act_func_last)

        # dtype and device
        self.dtype = dtype
        self.device = device

        # Initialize the CNN and MLP
        setattr(self, self.cnn_mlp_name, self._create_network())

    def get_mlp_dim_in(self) -> int:
        """
        Compute the input size of mlp layers
        Returns:
            dim_in
        """
        image_out_size = \
            [util.image_output_size(size=s,
                                    num_cnn=self.num_cnn,
                                    cnn_kernel_size=self.kernel_size)
             for s in self.image_size]
        # dim_in = channel * w * h
        dim_in = self.cnn_channels[-1]
        for s in image_out_size:
            dim_in *= s
        return dim_in

    def _create_network(self):
        """
        Create CNNs and MLP

        Returns: cnn_mlp
        """
        cnn_mlp = ModuleList()
        for i in range(self.num_cnn):
            in_channel = self.cnn_channels[i]
            out_channel = self.cnn_channels[i + 1]
            cnn_mlp.append(nn.Conv2d(in_channel, out_channel, self.kernel_size,
                                     dtype=self.dtype, device=self.device))

        # Initialize the MLP
        cnn_mlp.append(MLP(name=self.name,
                           dim_in=self.dim_in,
                           dim_out=self.dim_out,
                           hidden_layers=self.hidden_layers,
                           act_func_hidden=self.act_func_hidden_type,
                           act_func_last=self.act_func_last_type,
                           dtype=self.dtype, device=self.device))
        return cnn_mlp

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.cnn_mlp_name,
                                                epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "num_cnn": self.num_cnn,
                "cnn_channels": self.cnn_channels,
                "kernel_size": self.kernel_size,
                "image_size": self.image_size,
                "dim_in": self.dim_in,
                "hidden_layers": self.hidden_layers,
                "dim_out": self.dim_out,
                "act_func_hidden_type": self.act_func_hidden_type,
                "act_func_last_type": self.act_func_last_type,
                "dtype": self.dtype,
                "device": self.device
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.cnn_mlp_name,
                                                epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.num_cnn == parameters["num_cnn"] \
                   and self.cnn_channels == parameters["cnn_channels"] \
                   and self.kernel_size == parameters["kernel_size"] \
                   and self.image_size == parameters["image_size"] \
                   and self.dim_in == parameters["dim_in"] \
                   and self.hidden_layers == parameters["hidden_layers"] \
                   and self.dim_out == parameters["dim_out"] \
                   and self.act_func_hidden_type == parameters[
                       "act_func_hidden_type"] \
                   and self.act_func_last_type == parameters[
                       "act_func_last_type"] \
                   and self.dtype == parameters["dtype"] \
                   and self.device == parameters["device"], \
                "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, data):
        """
        Network forward function

        Args:
            data: input data

        Returns: CNN + MLP output
        """

        # Reshape images batch to [num_traj * num_obs, C, H, W]
        num_traj, num_obs = data.shape[:2]
        data = data.reshape(-1, *data.shape[2:])

        cnns = eval("self." + self.cnn_mlp_name)[:-1]
        mlp = eval("self." + self.cnn_mlp_name)[-1]

        # Forward pass in CNNs
        for i in range(len(cnns) - 1):
            data = self.act_func_hidden(F.max_pool2d(cnns[i](data), 2))
        data = self.act_func_hidden(F.max_pool2d(
            F.dropout2d(cnns[-1](data), training=self.training), 2))

        # Flatten
        data = data.view(num_traj, num_obs, self.dim_in)

        # Forward pass in MLPs
        data = mlp(data)

        # Return
        return data


class TrainableVariable:
    def __init__(self, name, data):
        self.name = name
        self.variable_name = name + "_variable"
        self.variable = nn.Parameter(data=data)
        self.shape = data.shape
        self.dtype = data.dtype
        self.device = data.device

    @property
    def data(self):
        return self.variable.data

    def parameters(self):
        return [self.variable, ]

    def save(self, log_dir: str, epoch: int):
        """
        Save variables to file
        Args:
            log_dir: directory to save to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.variable_name,
                                                epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "variable_name": self.variable_name,
                "variable_shape": self.shape,
                "dtype": self.dtype,
                "device": self.device
            }
            pkl.dump(parameters, f)

        # Store variable
        with open(w_path, "wb") as f:
            torch.save(self.variable, f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.variable_name,
                                                epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.variable_name == parameters["variable_name"] \
                   and self.shape == parameters["variable_shape"] \
                   and self.dtype == parameters["dtype"] \
                   and self.device == parameters["device"], \
                f"Variable {self.variable_name}'s parameters do not match"

        # Load NN weights
        self.variable = torch.load(w_path)
