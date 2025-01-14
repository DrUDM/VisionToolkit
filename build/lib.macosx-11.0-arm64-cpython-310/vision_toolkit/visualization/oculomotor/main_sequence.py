# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_main_sequence(
    d_s,
    a_s,
    coefs_ad,
    l_p_v,
    l_a,
    coefs_pa,
    config,
):
    plt.style.use("seaborn-v0_8")

    x_ = np.linspace(0, np.max(d_s), 1000)

    y_ = np.polyval(coefs_ad, x_)

    plt.scatter(d_s, a_s, marker="P", color="purple")

    plt.plot(x_, y_, linewidth=1.2, color="black", linestyle="dashed")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Duration (s)", fontsize=14)

    if config["distance_type"] == "angular":
        plt.ylabel("Amplitude (deg)", fontsize=14)

    else:
        plt.ylabel("Amplitude (px)", fontsize=14)

    plt.show()
    plt.clf()

    x_ = np.linspace(np.min(l_a) - 0.5, np.max(l_a), 1000)

    y_ = np.polyval(coefs_pa, x_)

    plt.scatter(l_a, l_p_v, marker="P", color="purple")

    plt.plot(x_, y_, linewidth=1.2, color="black", linestyle="dashed")

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    if config["distance_type"] == "angular":
        plt.xlabel("Log-amplitude (deg)", fontsize=14)

    else:
        plt.xlabel("Log-amplitude (px)", fontsize=14)

    if config["distance_type"] == "angular":
        plt.ylabel("Log-peak velocity (deg/s)", fontsize=14)

    else:
        plt.ylabel("Log-peak velocity (px/s)", fontsize=14)

    plt.show()
    plt.clf()
