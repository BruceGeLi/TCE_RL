import torch

import mprl.util as util
from util.nn_base import CNNMLP
from util.nn_base import MLP
from util.nn_base import TrainableVariable


def test_new_mlp(tensor_order: int = 4):
    """
    Test new mlp

    Args:
        tensor_order: input tensor's order

    Returns:
        None

    """
    util.print_wrap_title("Test for MLP, order = " + str(tensor_order), wrap=2)

    hidden_layers = []
    for _ in range(tensor_order):
        hidden_layers.append(tensor_order)
    dim_tuple = tuple(hidden_layers)
    dtype = torch.float64
    device = torch.device("cuda")
    new_kwargs = {"name": "test",
                  "dim_in": tensor_order,
                  "dim_out": tensor_order,
                  "hidden_layers": hidden_layers,
                  "act_func_hidden": "tanh",
                  "act_func_last": "leaky_relu",
                  "dtype": dtype,
                  "device": device}

    util.print_line_title("Initialize new MLP")
    old_mlp = MLP(**new_kwargs)

    util.print_line_title("Given input, compute output")
    in_data = torch.ones(dim_tuple, dtype=dtype, device=device)
    out_data = old_mlp(in_data)
    print(out_data.shape)
    log_dir = util.make_log_dir_with_time_stamp("test")
    util.mkdir(log_dir, overwrite=True)
    old_mlp.save(log_dir, 0)

    new_mlp = MLP(**new_kwargs)
    new_mlp.load(log_dir, 0)


def test_cnn_mlp():
    util.print_wrap_title("test_cnn_mlp")
    dtype = torch.float64
    device = torch.device("cuda")

    kwargs = {"name": "test",
              "image_size": [40, 60],
              "kernel_size": 5,
              "num_cnn": 2,
              "cnn_channels": [1, 10, 20],
              "hidden_layers": [10, 10],
              "dim_out": 15,
              "act_func_hidden": "tanh",
              "act_func_last": "leaky_relu",
              "dtype": dtype,
              "device": device}

    util.print_line_title("Initialize new CNN MLP")
    cnnmlp = CNNMLP(**kwargs)

    util.print_line_title("Given input, compute output")
    in_data = torch.randn([10, 10, 1, 40, 60], dtype=dtype, device=device)
    out_data = cnnmlp(in_data)
    print(out_data.shape)

    log_dir = util.make_log_dir_with_time_stamp("test")
    util.mkdir(log_dir, overwrite=True)
    cnnmlp.save(log_dir, 0)

    new_cnnmlp = CNNMLP(**kwargs)
    new_cnnmlp.load(log_dir, 0)


def test_trainable_variable():
    """
    Test trainable variable

    Returns:
        None

    """
    util.print_wrap_title("Test for trainable variable")
    dtype = torch.float64
    device = torch.device("cuda")
    data = torch.normal(0.0, 1.0, size=[3], dtype=dtype, device=device)
    var = TrainableVariable(name="test", data=data)

    # Test save and load
    log_dir = util.make_log_dir_with_time_stamp("test_var")
    print(f"Variable to be saved {var.data}")
    util.mkdir(log_dir, overwrite=True)
    var.save(log_dir, 0)

    new_var = TrainableVariable(name="test", data=data+1)
    print(f"Variable before loading {new_var.data}")
    new_var.load(log_dir, 0)
    print(f"Variable after loading {new_var.data}")


def main():
    test_new_mlp(4)
    test_new_mlp(3)
    test_cnn_mlp()
    test_trainable_variable()


if __name__ == "__main__":
    main()
