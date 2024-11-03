# -*- coding: utf-8 -*-

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vision_toolkit.aoi.basic.basic import AoIBasicAnalysis


def AoI_predefined_dwell_time(seq_, ref_image, name=None):
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
    config = seq_.config
    centers = seq_.centers

    basic_a = AoIBasicAnalysis(seq_, verbose=False)
    dur_ = basic_a.AoI_duration(get_raw=True)["proportion"]

    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()
    ax.imshow(ref_image, alpha=0.6)
    ax.grid(None)

    colors_sns = sns.color_palette("pastel", n_colors=len(centers.keys()))
    # np.random.shuffle(colors_sns)

    aoi_coords = np.array(config["AoI_coordinates"])

    for i, k_ in enumerate(sorted(centers.keys())):
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
        x_m = (aoi_coord[1, 0] + aoi_coord[0, 0]) / 2 - 50
        y_m = (aoi_coord[1, 1] + aoi_coord[0, 1]) / 2
        # ax.text(x_m, y_m, k_, fontsize = 15)
        if dur_[i] > 0:
            p_ = max(2, dur_[i] * 360)
        else:
            p_ = 0

        wedge = patches.Wedge(
            np.array([x_m + 50, y_m + 10]),
            200,
            -90,
            -90 + p_,
            edgecolor=colors_sns[i],
            facecolor=colors_sns[i],
            alpha=1,
        )
        ax.add_patch(wedge)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if name is not None:
        plt.title(name)
    # fig.savefig('example_aoi_dt.png', dpi=200)

    plt.show()
    plt.clf()
