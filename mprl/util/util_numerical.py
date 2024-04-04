"""
    Utilities of numerical computation
"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from mprl import util


def to_log_space(data: Union[np.ndarray, torch.Tensor],
                 lower_bound: Optional[float]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    project data to log space

    Args:
        data: original data
        lower_bound: customized lower bound in runtime, will override the
                     default value

    Returns: log(data + lower_bound)

    """
    # Compute
    assert data.min() >= 0

    # actual lower bound
    actual_lower_bound = lower_bound if lower_bound is not None else 1e-8

    if isinstance(data, np.ndarray):
        log_data = np.log(data + actual_lower_bound)
    elif isinstance(data, torch.Tensor):
        log_data = torch.log(data + actual_lower_bound)
    else:
        raise NotImplementedError
    return log_data


def to_softplus_space(data: Union[np.ndarray, torch.Tensor],
                      lower_bound: Optional[float]) -> \
        Union[np.ndarray, torch.Tensor]:
    """
    Project data to softplus space

    Args:
        data: original data
        lower_bound: runtime lower bound of the result

    Returns: softplus(data) + lower_bound

    """
    # actual lower bound
    actual_lower_bound = lower_bound if lower_bound is not None else 1e-2

    # Compute
    if isinstance(data, np.ndarray):
        sp_result = np.log(1 + np.exp(data)) + actual_lower_bound
    elif isinstance(data, torch.Tensor):
        sp_result = torch.nn.functional.softplus(data) + actual_lower_bound
    else:
        raise NotImplementedError

    return sp_result


def reverse_from_softplus_space(data: Union[np.ndarray, torch.Tensor],
                                lower_bound: Optional[float]) -> \
        Union[np.ndarray, torch.Tensor]:
    """
    Project data back from softplus space

    Args:
        data: data in softplus space
        lower_bound: runtime lower bound of the result

    Returns: softplus(data) + lower_bound

    """
    # actual lower bound
    actual_lower_bound = lower_bound if lower_bound is not None else 1e-2

    # Compute
    if isinstance(data, np.ndarray):
        result = np.log(np.exp(data - actual_lower_bound) - 1)
    elif isinstance(data, torch.Tensor):
        result = torch.log(torch.exp(data - actual_lower_bound) - 1)
    else:
        raise NotImplementedError

    return result


def interpolate(x_ori: np.ndarray, y_ori: np.ndarray,
                num_tar: int) -> np.ndarray:
    """
    Interpolates trajectories to desired length and data density

    Args:
        x_ori: original data time, shape [num_x]
        y_ori: original data value, shape [num_x, dim_y]
        num_tar: number of target sequence points

    Returns:
        interpolated y data, [num_tar, dim_y]
    """

    # Setup interpolation scale
    start, stop = x_ori[0], x_ori[-1]
    x_tar = np.linspace(start, stop, num_tar)

    # check y dim
    if y_ori.ndim == 1:
        y_tar = np.interp(x_tar, x_ori, y_ori)
    else:
        # Initialize result array as shape
        y_tar = np.zeros((num_tar, y_ori.shape[1]))

        # Loop over y's dim
        for k in range(y_ori.shape[1]):
            y_tar[:, k] = np.interp(x_tar, x_ori, y_ori[:, k])

    return y_tar


def generate_stats(data: Union[np.ndarray, torch.Tensor, List],
                   name: str = None, dim: Union[int, List, Tuple] = None,
                   to_np: bool = True):
    """
    Compute the statistics of the given data array

    Args:
        data: data array, iterable
        name: string stored in the key of the returned statistics
        dim: along which dimensions to compute, None to all dimensions
        to_np: enforce converting torch.Tensor to np.ndarray in the result

    Returns:
        statistics containing mean, max, min, std, and median
    """
    if isinstance(data, list):
        data = np.asarray(data)

    if isinstance(data, np.ndarray):
        data_pkg = np
        dim = tuple(dim) if isinstance(dim, List) else dim
        dim_marker = ", axis=dim" if dim else ""
        bias_marker = ""
        data = data.astype(np.float) if data.dtype == bool else data

    elif isinstance(data, torch.Tensor):
        data_pkg = torch
        dim_marker = ", dim=dim" if dim else ""
        bias_marker = ", unbiased=False"  # To get the same result as np.std
        data = data.double() if data.dtype == torch.bool else data

    else:
        raise NotImplementedError

    mean = eval(f"data_pkg.mean(data{dim_marker})")
    maxi = eval(f"data_pkg.max(data{dim_marker})")
    mini = eval(f"data_pkg.min(data{dim_marker})")
    std = eval(f"data_pkg.std(data{dim_marker}{bias_marker})")
    med = eval(f"data_pkg.median(data{dim_marker})")

    # Handle the weird behavior in case of pytorch
    if isinstance(maxi, tuple):
        maxi = maxi[0]
    if isinstance(mini, tuple):
        mini = mini[0]
    if isinstance(med, tuple):
        med = med[0]

    if to_np:
        mean, maxi, mini, std, med = util.to_nps(mean, maxi, mini, std, med)

    # Save in dictionary
    name = name + "_" if name is not None else ""
    stats = {f"{name}mean": mean,
             f"{name}max": maxi,
             f"{name}min": mini,
             f"{name}std": std,
             f"{name}median": med}

    # Return
    return stats


def rewrite_dict(data: dict, key_prefix: str = None, key_suffix: str = None,
                 to_np: bool = True) -> dict:
    """
    Rewrite the keys of a dictionary by adding a prefix or suffix

    Args:
        data: original dictionary
        key_prefix: prefix to be added to the keys
        key_suffix: suffix to be added to the keys
        to_np: enforce converting torch.Tensor to np.ndarray in the result

    Returns:
        dictionary with rewritten keys
    """
    # Setup
    key_prefix = key_prefix + "_" if key_prefix is not None else ""
    key_suffix = "_" + key_suffix if key_suffix is not None else ""

    # Rewrite keys
    data_new = {}
    for key, value in data.items():
        data_new[key_prefix + key + key_suffix] = \
            util.to_np(value) if to_np else value

    # Return
    return data_new


def generate_many_stats(data_dict: dict,
                        name: str = None,
                        to_np: bool = False):
    """
    Compute the statistics of many given data arrays in stored in one dictionary
    Args:
        data_dict: data dictionary
        name: general name in statistics
        to_np: enforce converting torch.Tensor to np.ndarray in the result

    Returns:
        a compound statistics dictionary
    """

    name = name + "_" if name else ""

    many_stats = dict()
    for key, value in data_dict.items():
        stats = generate_stats(value, name + key, to_np=to_np)
        many_stats.update(stats)
    return many_stats


def grad_norm_clip(bound, params):
    """
    Clip gradients and return the norm before and after the clipping
    Args:
        bound: clipping upper bound
        params: NN parameters

    Returns:
        Gradients norm before and after the clipping of all params
    """
    # Store before and after clipping grad norm
    grad_norm_before = 0
    grad_norm_clipped = 0

    # Loop over params before clipping
    for p in params:
        assert p.requires_grad and p.grad is not None
        p_norm = p.grad.detach().data.norm(2).item() ** 2
        grad_norm_before += p_norm
    grad_norm_before = grad_norm_before ** 0.5

    # Clipping
    if bound > 0:
        torch.nn.utils.clip_grad_norm_(params, bound)

    # Loop over params after clipping
    for pc in params:
        pc_norm = pc.grad.detach().data.norm(2).item() ** 2
        grad_norm_clipped += pc_norm
    grad_norm_clipped = grad_norm_clipped ** 0.5

    return grad_norm_before, grad_norm_clipped


class RunningMeanStd:
    def __init__(self, name: str = "", epsilon: float = 1e-4,
                 shape: Tuple[int, ...] = (), dtype: str = "torch.float32",
                 device: str = "cpu",):
        """
        Adapted from Stablebaseline3, using Pytorch instead
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.name = "running_mean_std" if name == "" else name
        self.shape= shape
        self.dtype, self.device = util.parse_dtype_device(dtype, device)
        self.mean = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.var = torch.ones(shape, dtype=self.dtype, device=self.device)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.clone()
        new_object.var = self.var.clone()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor,
                            batch_var: torch.Tensor,
                            batch_count: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def save(self, log_dir: str, epoch: int):
        save_path = util.get_training_state_save_path(log_dir, self.name, epoch)
        mean_std_dict = {"mean": self.mean, "var": self.var, "count": self.count}
        with open(save_path, "wb") as f:
            torch.save(mean_std_dict, f)

    def load(self, log_dir: str, epoch: int):
        save_path = util.get_training_state_save_path(log_dir, self.name, epoch)
        mean_std_dict = torch.load(save_path)
        self.mean = mean_std_dict["mean"]
        self.var = mean_std_dict["var"]
        self.count = mean_std_dict["count"]
