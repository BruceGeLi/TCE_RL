import random
import time

import numpy as np
import torch

import mprl.util as util


def sleep():
    time.sleep(0.5)


def test_how_fast():
    util.print_wrap_title("test_how_fast")
    util.how_fast(repeat=3, func=sleep)


def test_run_time_test():
    util.print_wrap_title("test_run_time_test")
    util.print_line_title("call run_time_test with no key")
    util.run_time_test(True)
    sleep()
    util.run_time_test(False, print_duration=True)

    util.print_line_title("call run_time_test with key 1")
    util.run_time_test(True, key="func1")
    sleep()

    util.print_line_title("call run_time_test with key 2")
    util.run_time_test(True, key="func2")
    sleep()
    util.run_time_test(False, key="func2", print_duration=True)
    util.run_time_test(False, key="func1", print_duration=True)

def test_debug_plot():
    util.print_wrap_title("test_debug_plot")
    x = np.linspace(0, 2 * np.pi, 20)
    y = []
    y_t = []
    for i in range(5):
        y.append(np.sin(x + i))
        y_t.append(torch.from_numpy(np.sin(x + i)))
    util.debug_plot(x, y)
    util.debug_plot(x, y_t)


def test_set_global_random_seed():
    util.print_wrap_title("test_set_global_random_seed")
    util.set_global_random_seed(100)
    print(random.randint(1, 10))
    print(np.random.randint(1, 10))
    print(torch.randint(1, 10, []))
    util.set_global_random_seed(100)
    print(random.randint(1, 10))
    print(np.random.randint(1, 10))
    print(torch.randint(1, 10, []))


def test_is_debugging():
    util.print_wrap_title("test_is_debugging")
    if util.is_debugging():
        print('Debug mode')
    else:
        print('Not debug mode')


def main():
    test_how_fast()
    test_run_time_test()
    test_debug_plot()
    test_set_global_random_seed()
    test_is_debugging()


if __name__ == "__main__":
    main()
