import numpy as np
import torch
from addict import Dict

import mprl.util as util


def test_current_device():
    util.print_wrap_title("test_current_device")
    print(f"before {util.current_device()}")
    util.use_cuda()
    print(f"switch to gpu {util.current_device()}")
    util.use_cpu()
    print(f"switch to cpu {util.current_device()}")


def test_use_cpu():
    util.print_wrap_title("test_use_cpu")
    util.use_cuda()

    a = torch.zeros((2, 2))
    print("before:", a.device)

    # Switch to CPU
    util.use_cpu()

    b = torch.zeros((2, 2))
    print("after:", b.device)


def test_use_cuda():
    util.print_wrap_title("test_use_cuda")
    util.use_cpu()
    a = torch.zeros((2, 2))
    print("before:", a.device)

    # Check if GPU is available
    util.use_cuda()

    b = torch.zeros((2, 2))
    print("after:", b.device)


def test_parse_dtype_device():
    util.print_wrap_title("test_parse_dtype_device")
    dtype_str = ["float32", "float64"]
    device_str = ["cpu", "cuda"]
    for dt in dtype_str:
        for de in device_str:
            dtype, device = util.parse_dtype_device(dt, de)
            a = torch.zeros([2], dtype=dtype, device=device)
            print(a)


def test_make_iterable():
    util.print_wrap_title("test_make_iterable")
    t = util.make_iterable(3)
    print(t)
    t = util.make_iterable(3, 'list')
    print(t)
    t = util.make_iterable([3], )
    print(t)
    t = util.make_iterable((3,))
    print(t)


def test_from_string_to_array():
    util.print_wrap_title("test_from_string_to_array")
    r = util.from_string_to_array("[1.0   2.3   4.5 \n 5.3   5.6]")
    print(r)


def test_to_np():
    util.print_wrap_title("test_to_np")
    a = np.array([1, 2, 3])
    print(util.to_np(a))

    a = torch.tensor(a, device="cpu")
    print(util.to_np(a, dtype=np.float32))

    a = a.to("cuda")
    print(util.to_np(a))


def test_to_nps():
    util.print_wrap_title("test_to_nps")
    a = np.array([1, 2, 3])
    b = torch.tensor(a, device="cpu")
    c = b.to("cuda")
    print(util.to_nps(a, b, c))


def test_to_ts():
    util.print_wrap_title("test_to_ts")

    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([1, 2, 3]).double()
    c = 3.14
    d = np.array([1, 2, 3])  # This is a float 64 array
    e = np.array([1, 2, 3], dtype=float)

    util.print_line_title("Original data")
    for data in [a, b, c, d, e]:
        print(f"data: {data}")

    for data_type in [torch.float32, torch.float64]:
        for device in ["cpu", "cuda"]:
            util.print_line_title(f"data_type: {data_type}, device: {device}")
            for data in [a, b, c, d, e]:
                tensor_data = util.to_ts(data, data_type, device)
                print(tensor_data)
                print(tensor_data.device)
                print(tensor_data.type(), "\n")


def test_to_tss():
    util.print_wrap_title("test_to_tss")
    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([1, 2, 3]).double()
    c = 3.14
    d = np.array([1, 2, 3])  # This is a float 64 array
    e = np.array([1, 2, 3], dtype=float)

    util.print_line_title("Original data")
    for data in [a, b, c, d, e]:
        print(f"data: {data}")

    util.print_line_title("Casted data")
    a, b, c, d, e = util.to_tss(a, b, c, d, e, dtype=torch.float64,
                                device="cuda")
    for data in [a, b, c, d, e]:
        util.print_line()
        print(data)
        print(data.device)
        print(data.type(), "\n")


def test_is_np_is_ts():
    util.print_wrap_title("test_is_np_is_ts")
    assert util.is_np(np.zeros([2])) is True
    assert util.is_ts(torch.zeros([2])) is True


def test_to_tensor_dict():
    util.print_wrap_title("test_to_tensor_dict")
    npd = Dict()
    npd.a = np.zeros([2, 2])
    npd.b.c = np.zeros([3, 3])
    npd.b.d = np.zeros([3, 3])
    npd.b.e.f = [1, 2, 3]

    tsd = util.to_tensor_dict(npd.to_dict())  # dict
    print(tsd)


def test_to_numpy_dict():
    util.print_wrap_title("test_to_numpy_dict")
    tsd = Dict()
    tsd.a = torch.zeros([2, 2])
    tsd.b.c = torch.zeros([3, 3])
    tsd.b.d = torch.zeros([3, 3])
    tsd.b.e.f = [1, 2, 3]

    npd = util.to_numpy_dict(tsd.to_dict())  # dict
    print(npd)


def test_conv2d_size_out():
    util.print_wrap_title("test_conv2d_size_out")
    util.print_line_title("desired 36")
    print(util.conv2d_size_out(40, 5, 1))


def test_maxpool2d_size_out():
    util.print_wrap_title("test_maxpool2d_size_out")
    util.print_line_title("desired 20")
    print(util.maxpool2d_size_out(40, 2))


def test_image_output_size():
    util.print_wrap_title("test_image_output_size")
    print(util.image_output_size(40, 2, 5, 1, True, 2, None))


def test_get_item_from_dicts():
    util.print_wrap_title("test_get_item_from_dicts")
    a = {"key1": 123, "key2": 1234}
    b = {"key1": 234, "key2": 2345}
    print(util.get_item_from_dicts([a, b], key="key2"))


def test_assert_shape():
    util.print_wrap_title("test_assert_shape")
    a = 1
    b = [[1, 2, 3], [4, 5, 6]]
    c = np.asarray(b)
    d = torch.Tensor(b)
    util.assert_shape(a, [])
    util.assert_shape(b, [2, 3])
    util.assert_shape(c, [2, 3])
    util.assert_shape(d, [2, 3])


def test_set_value_in_nest_dict():
    util.print_wrap_title("test_set_seed_in_nest_dict")
    config = {"a": 123, "b:": 456,
              "c": {"a": 123, "b:": 456,
                    "c": {"a": 123, "b:": 456, "c": None}
                    }
              }
    print(f"before: {config}")
    util.set_value_in_nest_dict(config, "a", "cool")
    print(f"after: {config}")


def main():
    test_current_device()
    test_use_cpu()
    test_use_cuda()
    test_parse_dtype_device()
    test_make_iterable()
    test_from_string_to_array()
    test_to_np()
    test_to_nps()
    test_to_ts()
    test_to_tss()
    test_is_np_is_ts()
    test_to_tensor_dict()
    test_to_numpy_dict()
    test_conv2d_size_out()
    test_maxpool2d_size_out()
    test_image_output_size()
    test_get_item_from_dicts()
    test_assert_shape()
    test_set_value_in_nest_dict()

if __name__ == "__main__":
    main()
