"""
    Utilities of hyper-parameters and randomness
"""

import numpy as np


def mlp_arch_3_params(avg_neuron: int, num_hidden: int, shape: float) -> [int]:
    """
    3 params way of specifying dense net, mostly for hyperparameter optimization
    Originally from Optuna work

    Args:
        avg_neuron: average number of neurons per layer
        num_hidden: number of layers
        shape: parameters between -1 and 1:
            shape < 0: "contracting" network, i.e, layers  get smaller,
                        for extreme case (shape = -1):
                        first layer 2 * avg_neuron neurons,
                        last layer 1 neuron, rest interpolating
            shape 0: all layers avg_neuron neurons
            shape > 0: "expanding" network, i.e., representation gets larger,
                        for extreme case (shape = 1)
                        first layer 1 neuron,
                        last layer 2 * avg_neuron neurons, rest interpolating

    Returns:
        architecture: list of integers representing the number of neurons of
                      each layer
    """

    assert avg_neuron >= 0
    assert -1.0 <= shape <= 1.0
    assert num_hidden >= 1
    shape = shape * avg_neuron  # we want the user to provide shape \in [-1, +1]
    architecture = []
    for i in range(num_hidden):
        # compute real-valued 'position' x of current layer (x \in (-1, 1))
        x = 2 * i / (num_hidden - 1) - 1 if num_hidden != 1 else 0.0
        # compute number of units in current layer
        d = shape * x + avg_neuron
        d = int(np.floor(d))
        if d == 0:  # occurs if shape == -avg_neuron or shape == avg_neuron
            d = 1
        architecture.append(d)
    return architecture
