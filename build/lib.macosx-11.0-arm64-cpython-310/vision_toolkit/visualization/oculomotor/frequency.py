# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_periodogram(freqs, densities, direction="both", cross=False):
    plt.style.use("seaborn-v0_8")

    if direction == "x":
        plt.plot(
            freqs,
            densities["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("Frequency (Hz)", fontsize=14)

        if cross:
            plt.ylabel("Cross-density", fontsize=14)
        else:
            plt.ylabel("Density", fontsize=14)

        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_visible(False)

        plt.show()
        plt.clf()

    elif direction == "y":
        plt.plot(
            freqs, densities["y"], linewidth=0.5, label="Vertical axis", color="purple"
        )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("Frequency (Hz)", fontsize=14)

        if cross:
            plt.ylabel("Cross-density", fontsize=14)
        else:
            plt.ylabel("Density", fontsize=14)

        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_visible(False)

        plt.show()
        plt.clf()

    else:
        plt.plot(freqs, densities["x"], linewidth=0.5, color="darkblue")

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Density", fontsize=14)

        plt.show()
        plt.clf()

        plt.plot(freqs, densities["y"], linewidth=0.5, color="purple")

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("Frequency (Hz)", fontsize=14)

        if cross:
            plt.ylabel("Cross-density", fontsize=14)
        else:
            plt.ylabel("Density", fontsize=14)

        plt.show()
        plt.clf()
