import torch

import mprl.util as util


def test_joint_to_conditional():
    util.print_wrap_title("test_joint_to_conditional")
    joint_mean = torch.Tensor([0, 0])[None,]
    joint_L = util.build_lower_matrix(torch.Tensor([1, 1]),
                                      torch.Tensor([-1]))[None,]
    sample_x = torch.Tensor([1])[None,]
    cond_mean, cond_L = util.joint_to_conditional(joint_mean, joint_L, sample_x)
    util.print_line_title("Expected -1")
    print("cond_mean: ", cond_mean)
    util.print_line_title("Expected 1")
    print("cond_L: ", cond_L)

    sample_x = torch.Tensor([0.5])[None,]
    cond_mean, cond_L = util.joint_to_conditional(joint_mean, joint_L, sample_x)
    util.print_line_title("Expected -0.5")
    print("cond_mean: ", cond_mean)
    util.print_line_title("Expected 1")
    print("cond_L: ", cond_L)

    joint_mean = torch.Tensor([0, 0])[None,]
    joint_L = util.build_lower_matrix(torch.Tensor([1, 1]),
                                      torch.Tensor([-0.5]))[None,]
    sample_x = torch.Tensor([0.5])[None,]
    cond_mean, cond_L = util.joint_to_conditional(joint_mean, joint_L, sample_x)
    util.print_line_title("Expected -0.25")
    print("cond_mean: ", cond_mean)
    util.print_line_title("Expected 1")
    print("cond_L: ", cond_L)

    util.print_line_title("Numpy")
    sample_x = torch.Tensor([0.5])[None,]
    cond_mean, cond_L = util.joint_to_conditional(joint_mean.numpy(),
                                                  joint_L.numpy(),
                                                  sample_x.numpy())
    util.print_line_title("Expected -0.25")
    print("cond_mean: ", cond_mean)
    util.print_line_title("Expected 1")
    print("cond_L: ", cond_L)


def main():
    test_joint_to_conditional()


if __name__ == "__main__":
    main()
