# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_BCEA(scanpath, p, path):
    plt.style.use("seaborn-v0_8")

    cov = np.cov(scanpath[0], scanpath[1])

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    i = np.argmax(eigenvalues)
    i_ = np.argmin(eigenvalues)

    ei = eigenvalues[i]
    ei_ = eigenvalues[i_]

    ev = eigenvectors[:, i]
    angle = np.arctan2(ev[1], ev[0])

    if angle < 0:
        angle += 2 * np.pi

    x_m = np.mean(scanpath[0])
    y_m = np.mean(scanpath[1])

    chisquare_val = stats.chi2.ppf(p, df=2)

    a = np.sqrt(chisquare_val * ei)
    b = np.sqrt(chisquare_val * ei_)

    theta_grid = np.linspace(0, 2 * np.pi)

    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    r_ellipse = np.matmul(rot_mat, np.vstack((ellipse_x_r, ellipse_y_r)))

    plt.scatter(scanpath[0], scanpath[1], marker="P", s=35, color="darkblue")

    plt.plot(
        r_ellipse[0] + x_m,
        r_ellipse[1] + y_m,
        color="purple",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Confidence Ellipse")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_BCEA", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_voronoi_cells(scanpath, vertices, path):
    plt.style.use("seaborn-v0_8")

    plt.scatter(scanpath[0], scanpath[1], marker="P", s=35, color="darkblue")

    for poly in vertices:
        plt.fill(*zip(*poly), alpha=0.25)

    # plt.title("Voronoi Cells")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_voronoi", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_HFD(sc_b, dist_, hilbert_points, coefs, x_, l_, path):
    plt.style.use("seaborn-v0_8")

    plt.plot(hilbert_points[:, 0], hilbert_points[:, 1], linewidth=0.8, color="purple")

    plt.plot(
        sc_b[0], sc_b[1], linestyle="", marker="P", markersize=8.0, color="darkblue"
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)
    plt.gca().invert_yaxis()

    # x_left, x_right = plt.xlim()
    # y_low, y_high = plt.ylim()
    # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)
    # fig = plt.gcf()
    # fig.savefig('hfd_hilbert', dpi=200, bbox_inches='tight')

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_hilbert", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(
        dist_,
        linewidth=0.5,
        linestyle="--",
        marker="P",
        markersize=8.0,
        color="purple",
        mfc="darkblue",
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Fixation index", fontsize=12)
    plt.ylabel("Hillbert distance", fontsize=12)

    # x_left, x_right = plt.xlim()
    # y_low, y_high = plt.ylim()
    # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)
    # fig = plt.gcf()
    # fig.savefig('hfd_dist', dpi=200, bbox_inches='tight')

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_distances", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    x = np.linspace(np.min(x_), np.max(x_), 1000)
    y = np.polyval(coefs, x)

    plt.plot(x, y, linewidth=1.8, color="black", linestyle="dashed")

    plt.plot(x_, l_, linewidth=1.2, color="darkblue")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("log lengths", fontsize=12)
    plt.ylabel("log inverse time intervals", fontsize=12)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_regression", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_convex_hull(scanpath, h_a, path):
    plt.style.use("seaborn-v0_8")
    h_a = np.append(h_a, h_a[0, :].reshape(1, 2), axis=0)

    fig, ax = plt.subplots()

    ax.plot(h_a[:, 0], h_a[:, 1], linewidth=1, color="black")

    ax.plot(scanpath[0], scanpath[1], linewidth=0.8, color="purple")

    for i in range(len(scanpath[0])):
        dur = scanpath[2, i]
        circle = plt.Circle(
            (scanpath[0, i], scanpath[1, i]),
            dur * 35,
            color="darkblue",
            fill=False,
            linewidth=0.8,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_convex_hull", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
