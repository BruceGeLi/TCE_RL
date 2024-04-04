import mprl.util as util


def test_mlp_arch_3_params():
    util.print_wrap_title("test_mlp_arch_3_params")
    avg = 256
    num_hidden = 5
    shape = [0.5, 0, -0.5]
    for s in shape:
        util.print_line()
        print(util.mlp_arch_3_params(avg, num_hidden, s))


def main():
    test_mlp_arch_3_params()


if __name__ == "__main__":
    main()
