import numpy as np
import torch

import mprl.util as util


def test_euler2quat():
    util.print_wrap_title("test_euler2quat")
    euler = np.array([0, 0, np.pi / 2])
    util.print_line_title("Expected w,x,y,z: 0.7, 0, 0, 0.7")
    print(util.euler2quat(euler))

    euler = torch.Tensor([np.pi / 4, np.pi / 4, np.pi / 4])
    util.print_line_title("Expected w,x,y,z:  0.73 , 0.46, 0.19, 0.46")
    print(util.euler2quat(euler))


def test_mat2euler():
    mat = torch.Tensor([[0.6, 0.8, 0],
                        [-0.8, 0.6, 0],
                        [0, 0, 1]])
    util.print_wrap_title("test_mat2euler")
    util.print_line_title("0, 0, -0.927")
    print(util.mat2euler(mat))


def test_quat2mat():
    util.print_wrap_title("test_quat2mat")
    quat = torch.Tensor([0.70710678, 0.70710678, 0., -0.])
    util.print_line_title("expected: ")
    print("[  1.0000000,  0.0000000,  0.0000000;\n"
          "   0.0000000,  0.0000000, -1.0000000;\n"
          "   0.0000000,  1.0000000,  0.0000000 ]")
    print("and get: ", util.quat2mat(quat))


def test_quat2euler():
    util.print_wrap_title("test_quat2euler")
    quat = torch.Tensor([0.70710678, 0.70710678, 0., -0.])
    util.print_line_title("expected: [ x: 1.5707963, y: 0, z: 0 ]")
    print(util.quat2euler(quat))


def main():
    test_euler2quat()
    test_mat2euler()
    test_quat2mat()
    test_quat2euler()


if __name__ == "__main__":
    main()
