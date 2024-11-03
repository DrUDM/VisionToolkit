# -*- coding: utf-8 -*-

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

np.random.seed(1)


def display_aoi_predefined_reference_image(positions, clusters, config, ref_image):
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
    path = config["display_AoI_path"]

    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()
    ax.imshow(ref_image, alpha=0.6)
    ax.grid(None)

    colors_sns = sns.color_palette("pastel", n_colors=len(clusters.keys()))
    # np.random.shuffle(colors_sns)

    aoi_coords = np.array(config["AoI_coordinates"])

    for i, k_ in enumerate(sorted(clusters.keys())):
        aoi_coord = aoi_coords[i]
        xy = (aoi_coord[0, 0], aoi_coord[0, 1])
        w_ = aoi_coord[1, 1] - aoi_coord[0, 1]
        h_ = aoi_coord[1, 0] - aoi_coord[0, 0]

        rect = patches.Rectangle(
            xy,
            h_,
            w_,
            linewidth=1,
            edgecolor=colors_sns[i],
            facecolor=colors_sns[i],
            alpha=0.35,
            fill=True,
        )
        ax.add_patch(rect)
        ax.scatter(
            positions[0, clusters[k_]],
            positions[1, clusters[k_]],
            color=colors_sns[i],
            marker="+",
            s=10,
        )
        x_m = (aoi_coord[1, 0] + aoi_coord[0, 0]) / 2 - 50
        y_m = (aoi_coord[1, 1] + aoi_coord[0, 1]) / 2 + 25
        ax.text(x_m, y_m, k_, fontsize=15)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # fig.savefig('example_aoi_dt.png', dpi=200)

    plt.show()
    plt.clf()


def display_aoi_identification_reference_image(positions, clusters, config, ref_image):
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
    path = config["display_AoI_path"]

    plt.style.use("seaborn-v0_8")

    if isinstance(ref_image, str):
        ref_image = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(ref_image, alpha=0.6)
    ax.grid(None)

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
            ax.text(x_m, y_m, k_, fontsize=25)

        else:
            plot_confidence_ellipse(
                positions[:, clusters[k_]], name=k_, ax=ax, color="black"
            )

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # fig.savefig('example_aoi_dt.png', dpi=200)

    plt.show()
    plt.clf()


def display_aoi_identification(
    positions,
    clusters,
    config,
):
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
    path = config["display_AoI_path"]

    # display_aoi_identification_reference_image(positions,
    #                                           clusters,
    #                                           config)
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
                positions[:, clusters[k_]], name=k_, ax=ax, color="black"
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


def plot_confidence_ellipse(
    positions,
    name,
    ax,
    color,
    p=0.68,
):
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

    x_m = np.mean(positions[0])
    y_m = np.mean(positions[1])

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
        r_ellipse[0] + x_m, r_ellipse[1] + y_m, color=color, linewidth=2, linestyle="--"
    )

    ax.text(x_m, y_m, name, fontsize=22)
