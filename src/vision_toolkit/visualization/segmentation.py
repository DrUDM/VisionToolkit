# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def display_binary_segmentation(data_set, config, _ints, _color):
    path = config["display_segmentation_path"]
    x_a = data_set["x_array"]
    y_a = data_set["y_array"]

    x_b = 1000 * np.array(np.arange(0, len(x_a))) / config["sampling_frequency"]

    plt.style.use("seaborn-v0_8")

    plt.plot(x_b, x_a, linewidth=0.6, color="grey")

    for _int in _ints:
        x = (
            1000
            * np.array(np.arange(_int[0], _int[-1] + 1))
            / config["sampling_frequency"]
        )

        y = x_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=1, color=_color)

    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Horizontal position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_horizontal_segmentation", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(x_b, y_a, linewidth=0.6, color="grey")

    for _int in _ints:
        x = (
            1000
            * np.array(np.arange(_int[0], _int[-1] + 1))
            / config["sampling_frequency"]
        )

        y = y_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=1, color=_color)

    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_vertical_segmentation", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(x_a, y_a, linewidth=0.6, color="grey")

    for _int in _ints:
        x = x_a[_int[0] : _int[-1] + 1]
        y = y_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=1.5, color=_color)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_2D_segmentation", dpi=200, bbox_inches="tight")

    plt.show()

    plt.clf()


def display_ternary_segmentation(
    data_set,
    config,
    _ints,
    _color,
):
    path = config["display_segmentation_path"]
    x_a = data_set["x_array"]
    y_a = data_set["y_array"]

    x_b = (
        1000
        * np.array(np.arange(0, config["nb_samples"]))
        / config["sampling_frequency"]
    )

    plt.style.use("seaborn-v0_8")
    plt.plot(x_b, x_a, linewidth=0.8, color="grey")

    for _int in _ints:
        x = (
            1000
            * np.array(np.arange(_int[0], _int[-1] + 1))
            / config["sampling_frequency"]
        )

        y = x_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=2, color=_color)

    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Horizontal position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        x_left, x_right = plt.xlim()
        y_low, y_high = plt.ylim()
        plt.gca().set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.30)
        fig = plt.gcf()
        fig.savefig(path + "horizontal", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(x_b, y_a, linewidth=0.8, color="grey")

    for _int in _ints:
        x = (
            1000
            * np.array(np.arange(_int[0], _int[-1] + 1))
            / config["sampling_frequency"]
        )

        y = y_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=2, color=_color)

    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        x_left, x_right = plt.xlim()
        y_low, y_high = plt.ylim()
        plt.gca().set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.30)
        fig = plt.gcf()
        fig.savefig(path + "vertical", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(x_a, y_a, linewidth=0.8, color="grey")

    for _int in _ints:
        x = x_a[_int[0] : _int[-1] + 1]
        y = y_a[_int[0] : _int[-1] + 1]

        plt.plot(x, y, linewidth=2, color=_color)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.show()

    plt.clf()
