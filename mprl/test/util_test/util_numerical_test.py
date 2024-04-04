import numpy as np
import torch

import mprl.util as util


def test_to_log_space():
    util.print_wrap_title("test_to_log_space")
    data = torch.full(size=(2, 2), fill_value=1.0)
    log_data = util.to_log_space(data, lower_bound=None)
    util.print_line_title("desired value is tensor full of 0.0")
    print(log_data)

    log_data = util.to_log_space(data, lower_bound=1.71828)
    util.print_line_title("desired value is tensor full of 1.0")
    print(log_data)


def test_to_softplus_space():
    util.print_wrap_title("test_to_softplus_space")
    data = torch.full(size=(2, 2), fill_value=0.0)
    data_sp = util.to_softplus_space(data, lower_bound=None)
    util.print_line_title("desired value is tensor full of 0.7031")
    print(data_sp)

    data_sp = util.to_softplus_space(data, lower_bound=2)
    util.print_line_title("desired value is tensor full of 2.6931")
    print(data_sp)


def test_reverse_from_softplus_space():
    util.print_wrap_title("test_reverse_from_softplus_space")
    data = torch.full(size=(2, 2), fill_value=0.7031471729278564)
    data_ori = util.reverse_from_softplus_space(data, lower_bound=None)
    util.print_line_title("desired value is tensor full of 0.0")
    print(data_ori)


def test_interpolate():
    util.print_wrap_title("test_interpolate")
    x_ori = np.linspace(0, 10, 11)
    y_ori = np.linspace(0, 10, 11)
    print("x_ori: ", x_ori)
    print("y_ori: ", y_ori)
    print("y_tar: ", util.interpolate(x_ori, y_ori, num_tar=101))

    y_ori = np.stack([y_ori, y_ori]).T

    print("x_ori: ", x_ori)
    print("y_ori: ", y_ori)
    print("y_tar: ", util.interpolate(x_ori, y_ori, num_tar=101))


def test_generate_stats():
    util.print_wrap_title("test_generate_stats")
    a = [[1, 2, 2, 3], [3, 4, 4, 5], [5, 6, 6, 7]]
    print(f"a: \n {np.asarray(a)}")

    util.print_line_title("list, dim=None")
    print(util.generate_stats(a, "a"))
    util.print_line_title("np.ndarray with no name, dim=[0,1]")
    print(util.generate_stats(np.asarray(a), None, dim=[0, 1]))
    util.print_line_title("torch.Tensor with dim=1")
    print(util.generate_stats(torch.Tensor(a), "a", dim=1))
    print(util.generate_stats(torch.Tensor(a), "a", dim=1, to_np=True))


def test_generate_many_stats():
    util.print_wrap_title("test_generate_many_stats")
    a = [[1, 2, 2, 3], [3, 4, 4, 5], [5, 6, 6, 7]]
    b = np.asarray(a) + 2
    c = np.array([[True, True, False, False], [True, True, False, False]],
                 dtype=bool)
    d = torch.tensor([[True, True, False, False], [True, True, False, False]],
                     dtype=torch.bool)
    print(util.generate_many_stats({"a": a, "b": b, "c": c, "d": d},
                                   to_np=True))

    print(util.generate_many_stats({"a": a, "b": b, "c": c, "d": d},
                                   name="test", to_np=True))


def test_rewrite_dict():
    util.print_wrap_title("test_rewrite_dict")
    a = {"a": torch.Tensor([1]), "b": torch.Tensor([2])}
    print("After adding prefix and suffix",
          util.rewrite_dict(a, key_prefix="prefix", key_suffix="suffix",
                            to_np=True), sep="\n")
    print("After adding prefix",
          util.rewrite_dict(a, key_prefix="prefix", key_suffix=None,
                            to_np=True), sep="\n")
    print("After adding suffix",
          util.rewrite_dict(a, key_prefix=None, key_suffix="suffix",
                            to_np=True), sep="\n")


def test_grad_norm_clip():
    raise NotImplementedError


def main():
    test_to_log_space()
    test_to_softplus_space()
    test_reverse_from_softplus_space()
    test_interpolate()
    test_generate_stats()
    test_generate_many_stats()
    test_rewrite_dict()
    # test_grad_norm_clip()


if __name__ == "__main__":
    main()
