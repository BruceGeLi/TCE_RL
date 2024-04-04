import matplotlib.pyplot as plt
import numpy as np

import mprl.util as util


def test_savefig():
    util.print_wrap_title("test_savefig")
    figs = []
    for i in range(5):
        fig = plt.figure()
        x = np.linspace(0, 2 * np.pi, 20)
        y = np.sin(x + i)
        plt.plot(x, y)
        figs.append(fig)
    util.savefig(figs, "sin_fig", 'pdf', overwrite=True)


def test_save_subfig():
    util.print_wrap_title("test_save_subfig")
    fig, axes = plt.subplots(2, 2, squeeze=False)
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            x = np.linspace(0, 2 * np.pi, 20)
            y = np.sin(x + i + j)

            ax.plot(x, y)
    plt.show()
    util.save_subfig(fig, axes, [[0, 0], [1, 1]], "test_save_subfig", "pdf",
                     overwrite=True)


def test_from_figures_to_video():
    util.print_wrap_title("test_from_figures_to_video")
    figs = []
    for i in range(5):
        fig = plt.figure()
        x = np.linspace(0, 2 * np.pi, 20)
        y = np.sin(x + i)
        plt.plot(x, y)
        figs.append(fig)
    util.from_figures_to_video(figs, "sin_video2", overwrite=True)


def test_fill_between():
    util.print_wrap_title("test_fill_between")
    x = np.linspace(0, np.pi, 21)
    y_mean = np.sin(x)
    y_std = np.abs(np.cos(x))
    plt.figure()
    util.fill_between(x, y_mean, y_std, draw_mean=True)
    plt.show()


def main():
    # test_savefig()
    test_save_subfig()
    # test_from_figures_to_video()
    # test_fill_between()


if __name__ == "__main__":
    main()
