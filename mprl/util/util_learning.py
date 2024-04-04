"""
    Utilities of learning operation
"""
from typing import Union

import numpy as np
import torch

import mprl.util as util


def joint_to_conditional(joint_mean: Union[np.ndarray, torch.Tensor],
                         joint_L: Union[np.ndarray, torch.Tensor],
                         sample_x: Union[np.ndarray, torch.Tensor]) -> \
        [Union[np.ndarray, torch.Tensor]]:
    """
    Given joint distribution p(x,y), and a sample of x, do:
    Compute conditional distribution p(y|x)
    Args:
        joint_mean: mean of joint distribution
        joint_L: cholesky distribution of joint distribution
        sample_x: samples of x

    Returns:
        conditional mean and L
    """

    # Shape of joint_mean:
    # [*add_dim, dim_x + dim_y]
    #
    # Shape of joint_L:
    # [*add_dim, dim_x + dim_y, dim_x + dim_y]
    #
    # Shape of sample_x:
    # [*add_dim, dim_x]
    #
    # Shape of conditional_mean:
    # [*add_dim, dim_y]
    #
    # Shape of conditional_cov:
    # [*add_dim, dim_y, dim_y]

    # Check dimension
    dim_x = sample_x.shape[-1]
    # dim_y = joint_mean.shape[-1] - dim_x

    # Decompose joint distribution parameters
    mu_x = joint_mean[..., :dim_x]
    mu_y = joint_mean[..., dim_x:]

    L_x = joint_L[..., :dim_x, :dim_x]
    L_y = joint_L[..., dim_x:, dim_x:]
    L_x_y = joint_L[..., dim_x:, :dim_x]

    if util.is_ts(joint_mean):
        cond_mean = mu_y + \
                    torch.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                                 torch.cholesky_inverse(L_x), sample_x - mu_x)
    elif util.is_np(joint_mean):
        # Scipy cho_solve does not support batch operation
        cond_mean = mu_y + \
                    np.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                              torch.cholesky_inverse(torch.from_numpy(
                                  L_x)).numpy(),
                              sample_x - mu_x)
    else:
        raise NotImplementedError

    cond_L = L_y

    return cond_mean, cond_L


def select_ctx_pred_pts(**kwargs):
    """
    Generate context and prediction indices
    Args:
        **kwargs: keyword arguments

    Returns:
        context indices and prediction indices

    """
    num_ctx = kwargs.get("num_ctx", None)
    num_ctx_min = kwargs.get("num_ctx_min", None)
    num_ctx_max = kwargs.get("num_ctx_max", None)
    first_index = kwargs.get("first_index", None)
    fixed_interval = kwargs.get("fixed_interval", False)
    num_all = kwargs.get("num_all", None)
    num_select = kwargs.get("num_select", None)
    ctx_before_pred = kwargs.get("ctx_before_pred", False)

    # Determine how many points shall be selected
    if num_select is None:
        assert fixed_interval is False
        assert first_index is None
        num_select = num_all
    else:
        assert num_select <= num_all

    # Determine how many context points shall be selected
    if num_ctx is None:
        num_ctx = torch.randint(low=num_ctx_min, high=num_ctx_max, size=(1,))
    assert num_ctx < num_select

    # Select points
    if fixed_interval:
        # Select using fixed interval
        interval = num_all // num_select
        residual = num_all % num_select

        if first_index is None:
            # Determine the first index
            first_index = \
                torch.randint(low=0, high=interval + residual, size=[]).item()
        else:
            # The first index is specified
            assert 0 <= first_index < interval + residual
        selected_indices = torch.arange(start=first_index, end=num_all,
                                        step=interval, dtype=torch.long)
    else:
        # Select randomly
        permuted_indices = torch.randperm(n=num_all)
        selected_indices = torch.sort(permuted_indices[:num_select])[0]

    # split ctx and pred
    if num_ctx == 0:
        # No context
        ctx_idx = []
        pred_idx = selected_indices

    else:
        # Ctx + Pred
        if ctx_before_pred:
            ctx_idx = selected_indices[:num_ctx]
            pred_idx = selected_indices[num_ctx:]
        else:
            permuted_select_indices = torch.randperm(n=num_select)
            ctx_idx = selected_indices[permuted_select_indices[:num_ctx]]
            pred_idx = selected_indices[permuted_select_indices[num_ctx:]]
    return ctx_idx, pred_idx


def select_pred_pairs(**kwargs):
    pred_index = select_ctx_pred_pts(num_ctx=0, **kwargs)[1]
    # pred_pairs = torch.combinations(pred_index, 2)
    pred_pairs = torch.zeros([pred_index.shape[0] - 1, 2])
    pred_pairs[:, 0] = pred_index[:-1]
    pred_pairs[:, 1] = pred_index[1:]
    return pred_pairs