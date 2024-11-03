# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def display_aoi_hmm(positions, clusters, centers, covars, config):
    """


    Parameters
    ----------
    positions : TYPE
        DESCRIPTION.
    clusters : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    colors_sns = sns.color_palette("pastel", n_colors=len(clusters.keys()))

    for i, k_ in enumerate(sorted(clusters.keys())):
        ax.scatter(
            positions[0, clusters[k_]], positions[1, clusters[k_]], color=colors_sns[i]
        )

        if len(clusters[k_]) == 1:
            x_m, y_m = positions[0, clusters[k_]], positions[1, clusters[k_]]
            circle = plt.Circle(
                (x_m, y_m), 1e-6, color="black", linewidth=2, linestyle="--", fill=False
            )
            ax.add_patch(circle)
            ax.text(x_m, y_m, k_, fontsize=22)

        else:
            plot_confidence_ellipse(
                positions[:, clusters[k_]], centers[i], name=k_, ax=ax
            )

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlim([0, config["size_plan_x"]])
    plt.ylim([0, config["size_plan_y"]])
    plt.gca().invert_yaxis()

    plt.show()
    plt.clf()


def plot_confidence_ellipse(positions, center, name, ax, p=0.68):
    """


    Parameters
    ----------
    positions : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is .68.

    Returns
    -------
    None.

    """

    plt.style.use("seaborn-v0_8")

    cov = np.cov(positions[0], positions[1])
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    i = np.argmax(eigenvalues)
    i_ = np.argmin(eigenvalues)

    ei = eigenvalues[i]
    ei_ = eigenvalues[i_]

    ev = eigenvectors[:, i]
    angle = np.arctan2(ev[1], ev[0])

    if angle < 0:
        angle += 2 * np.pi

    x_m = center[0]
    y_m = center[1]

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

    ax.plot(
        r_ellipse[0] + x_m,
        r_ellipse[1] + y_m,
        color="black",
        linewidth=2,
        linestyle="--",
    )

    ax.text(x_m, y_m, name, fontsize=22)
