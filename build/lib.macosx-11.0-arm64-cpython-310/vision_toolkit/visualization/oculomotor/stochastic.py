# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_msd(
    lags_MSD, MSDs, coefs, s_ts, l_ts, scaling_exponents=True, direction="both"
):
    plt.style.use("seaborn-v0_8")
    lg_msd = np.log10(MSDs)
    lg_lag = np.log10(lags_MSD)

    x_s = lg_lag[np.where(lg_lag < np.log10(s_ts[1] * 5))]

    x_l = lg_lag[
        np.where(
            np.logical_and(
                lg_lag > np.log10(l_ts[0] / 5),
                lg_lag < min(np.log10(l_ts[1] * 5), max(lg_lag)),
            )
        )
    ]

    if direction == "x":
        plt.plot(
            lg_lag,
            lg_msd[0, :],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["x"][0], x_s)
            y_l = np.polyval(coefs["x"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

    elif direction == "y":
        plt.plot(
            lg_lag, lg_msd[1, :], linewidth=0.5, label="Vertical axis", color="purple"
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["y"][0], x_s)
            y_l = np.polyval(coefs["y"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

    else:
        plt.plot(
            lg_lag,
            lg_msd[0, :],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["x"][0], x_s)
            y_l = np.polyval(coefs["x"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

        plt.plot(
            lg_lag, lg_msd[1, :], linewidth=0.5, label="Vertical axis", color="purple"
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["y"][0], x_s)
            y_l = np.polyval(coefs["y"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.xlabel("log-lag (s)", fontsize=14)
    plt.ylabel("log-MSD", fontsize=14)

    plt.show()
    plt.clf()


def plot_dacf(order, lags_DACF, DACFs, direction="both"):
    plt.style.use("seaborn-v0_8")

    if direction == "x":
        plt.plot(
            lags_DACF,
            DACFs[0, :],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

    elif direction == "y":
        plt.plot(
            lags_DACF, DACFs[1, :], linewidth=0.5, label="Vertical axis", color="purple"
        )

    else:
        plt.plot(
            lags_DACF,
            DACFs[0, :],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        plt.plot(
            lags_DACF, DACFs[1, :], linewidth=0.5, label="Vertical axis", color="purple"
        )

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.xlabel("lag (s)", fontsize=14)
    plt.ylabel("DACF of order {k}".format(k=order), fontsize=14)

    plt.show()
    plt.clf()


def plot_dfa(
    lags_DFA, fluct, coefs, s_ts, l_ts, scaling_exponents=True, direction="both"
):
    plt.style.use("seaborn-v0_8")

    lg_lag = np.log10(lags_DFA)

    x_s = lg_lag[np.where(lg_lag < np.log10(s_ts[1] * 5))]

    x_l = lg_lag[
        np.where(
            np.logical_and(
                lg_lag > np.log10(l_ts[0] / 5),
                lg_lag < min(np.log10(l_ts[1] * 5), max(lg_lag)),
            )
        )
    ]

    if direction == "x":
        plt.plot(
            np.log10(lags_DFA),
            np.log10(fluct[0:]),
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["x"][0], x_s)
            y_l = np.polyval(coefs["x"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

    elif direction == "y":
        plt.plot(
            np.log10(lags_DFA),
            np.log10(fluct[1, :]),
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["y"][0], x_s)
            y_l = np.polyval(coefs["y"][1], x_l)

            plt.plot(x_s, y_s, linewidth=0.5, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=0.5, linestyle="dotted", color="black")

    else:
        plt.plot(
            np.log10(lags_DFA),
            np.log10(fluct[0, :]),
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["x"][0], x_s)
            y_l = np.polyval(coefs["x"][1], x_l)

            plt.plot(x_s, y_s, linewidth=1.0, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=1.0, linestyle="dotted", color="black")

        plt.plot(
            np.log10(lags_DFA),
            np.log10(fluct[1, :]),
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

        if scaling_exponents:
            y_s = np.polyval(coefs["y"][0], x_s)
            y_l = np.polyval(coefs["y"][1], x_l)

            plt.plot(x_s, y_s, linewidth=1.0, linestyle="dotted", color="black")

            plt.plot(x_l, y_l, linewidth=1.0, linestyle="dotted", color="black")

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.xlabel("log-lag (s)", fontsize=14)
        plt.ylabel("log-fluctuation", fontsize=14)
        plt.legend(fontsize=10)

        plt.show()
        plt.clf()
