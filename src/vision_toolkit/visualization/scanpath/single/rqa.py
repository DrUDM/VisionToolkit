# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_RQA(r_m):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(r_m, cmap="GnBu", edgecolors="w", linewidth=0.5)

    ax.set_aspect("equal")

    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("Recurrence matrix", fontsize=18)

    plt.show()
    plt.clf()


def plot_RQA_laminarity(r_m, h_set, v_set):
    l_mat = np.zeros_like(r_m)
    for l_ in h_set:
        for i in range(len(l_)):
            l_mat[l_[i, 0], l_[i, 1]] = 1

    for l_ in v_set:
        for i in range(len(l_)):
            l_mat[l_[i, 0], l_[i, 1]] = 1

    l_mat += l_mat.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(l_mat, cmap="GnBu", edgecolors="w", linewidth=0.5)

    ax.set_aspect("equal")

    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("Laminarity matrix", fontsize=18)

    plt.show()
    plt.clf()


def plot_RQA_determinism(r_m, d_set):
    d_mat = np.zeros_like(r_m)
    for l_ in d_set:
        for i in range(len(l_)):
            d_mat[l_[i, 0], l_[i, 1]] = 1

    d_mat += d_mat.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(d_mat, cmap="GnBu", edgecolors="w", linewidth=0.5)

    ax.set_aspect("equal")

    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("Determinism matrix", fontsize=18)

    plt.show()
    plt.clf()
