import numpy as np
import torch

import mprl.util as util


def test_param_to_cholesky():
    util.print_wrap_title("test_param_to_cholesky")
    param_diag = torch.zeros(6) + 0.5
    param_off_diag = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15])
    param_diag = param_diag[None, None, None, :]
    param_off_diag = param_off_diag[None, None, None, :]
    lower = util.build_lower_matrix(param_diag, param_off_diag)
    # lower = lower.squeeze()
    print(lower)

    # Speed test
    diag = torch.randn([100, 5, 50])
    off_diag = torch.randn([100, 5, 1225])
    util.how_fast(1000, util.build_lower_matrix, diag, off_diag)


def test_cholesky_2_params():
    util.print_wrap_title("test_cholesky_2_params")
    param_diag = torch.zeros(6) + 0.5
    param_off_diag = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15])
    param_diag = param_diag[None, None, None, :]
    param_off_diag = param_off_diag[None, None, None, :]

    L = util.build_lower_matrix(param_diag, param_off_diag)
    param_diag_rec, param_off_diag_rec = util.reverse_build_matrix(L, True)
    print(f"param_diag_rec: \n {param_diag_rec}")
    print(f"param_off_diag_rec: \n {param_off_diag_rec}")


def test_transform_to_cholesky():
    util.print_wrap_title("test_transform_to_cholesky")
    # Speed test
    mat = torch.randn([100, 5, 50, 50])
    util.how_fast(1000, util.transform_to_cholesky, mat)


def test_add_expand_dim():
    util.print_wrap_title("test_add_expand_dim")
    data = torch.randn([2, 3, 4])
    result1 = util.add_expand_dim(data, [1, 3, 5], [5, 6, 7])
    print(result1.shape)
    result2 = util.add_expand_dim(data, [1, -3, -1], [5, 6, 7])
    print(result2.shape)

    data = util.to_np(data)
    result3 = util.add_expand_dim(data, [1, 3, 5], [5, 6, 7])
    print(result3.shape)
    result4 = util.add_expand_dim(data, [1, -3, -1], [5, 6, 7])
    print(result4.shape)

    results = util.to_nps(result1, result2, result3, result4)
    assert np.all(results[0] == results[1]) \
           and np.all(results[1] == results[2]) \
           and np.all(results[2] == results[3])


def test_to_cholesky():
    util.print_wrap_title("test_to_cholesky")
    param_diag = torch.zeros(6) + 0.5
    param_off_diag = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15])
    param_diag = param_diag[None, None, None, :]
    param_off_diag = param_off_diag[None, None, None, :]
    lower = util.build_lower_matrix(param_diag, param_off_diag)
    cov = torch.einsum('...ik,...jk->...ij', lower, lower)

    util.print_line_title("Case 2 vectors")
    L1 = \
        util.to_cholesky(diag_vector=param_diag, off_diag_vector=param_off_diag)
    print(f"Case 2 vectors: {L1}")
    util.print_line_title("Case L")
    L2 = \
        util.to_cholesky(L=lower)
    print(f"Case L: {L2}")
    util.print_line_title("Case cov")
    L3 = \
        util.to_cholesky(cov_matrix=cov)
    print(f"Case cov: {L3}")


def test_tensor_linspace():
    util.print_wrap_title("test_tensor_linspace")
    end = torch.linspace(0, 10, 11)[None, ...]
    start = torch.zeros_like(end)
    result = util.tensor_linspace(start, end, 11)
    print(result)


def test_indexing_interpolate():
    util.print_wrap_title("test_indexing_interpolate")
    data = util.tensor_linspace(torch.zeros([2]), 5, 6)
    indices = (torch.arange(5) + 0.5)[None].expand(3, -1)
    result = util.indexing_interpolate(data, indices)
    util.print_line_title(f"desired shape {[3, 5, 2]}")
    print(result.shape)
    util.print_line_title(f"desired values: sequence from 0.5 to 4.5")
    print(result)


def test_get_sub_tensor():
    util.print_wrap_title("test_get_sub_tensor")
    data = torch.arange(0, 64).reshape([4, 4, 4])
    print(data)
    print(util.get_sub_tensor(data, [0, 1, 2],
                              [[0, 1], [1, 2], [1, 2, 3]]))
    print("desired: \n"
          "tensor([[[ 5,  6,  7], "
          "[ 9, 10, 11]], "
          "[[21, 22, 23], "
          "[25, 26, 27]]])")


def main():
    test_param_to_cholesky()
    test_cholesky_2_params()
    test_transform_to_cholesky()
    test_add_expand_dim()
    test_to_cholesky()
    test_tensor_linspace()
    test_indexing_interpolate()
    test_get_sub_tensor()


if __name__ == "__main__":
    main()
