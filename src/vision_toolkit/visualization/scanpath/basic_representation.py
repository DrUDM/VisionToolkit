# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_scanpath_reference_image(values, config, ref_image):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path = config["display_scanpath_path"]

    if isinstance(ref_image, str):
        ref_image = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_image = cv2.resize(ref_image, (config["size_plan_x"], config["size_plan_y"]))

    vf_diag = np.linalg.norm(np.array([config["size_plan_x"], config["size_plan_y"]]))
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()
    ax.imshow(ref_image, alpha=0.4)
    ax.grid(None)
    # values[1] = config['size_plan_y'] - values[1]
    s_p = values.T
    ax.plot(s_p[:, 0], s_p[:, 1], linewidth=0.8, color="purple")

    for i in range(len(s_p)):
        dur = s_p[i, 2]
        circle = plt.Circle(
            (s_p[i, 0], s_p[i, 1]),
            0.05 * dur * vf_diag,
            linewidth=0.8,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_reference_image", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def display_scanpath(values, config):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path = config["display_scanpath_path"]
    vf_diag = np.linalg.norm(np.array([config["size_plan_x"], config["size_plan_y"]]))
    plt.style.use("seaborn-v0_8")
    s_p = values.T

    fig, ax = plt.subplots()

    ax.plot(s_p[:, 0], s_p[:, 1], linewidth=0.5, color="purple")

    for i in range(len(s_p)):
        dur = s_p[i, 2]
        circle = plt.Circle(
            (s_p[i, 0], s_p[i, 1]),
            0.05 * dur * vf_diag,
            linewidth=0.5,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlim([0, config["size_plan_x"]])
    plt.ylim([0, config["size_plan_y"]])
    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
