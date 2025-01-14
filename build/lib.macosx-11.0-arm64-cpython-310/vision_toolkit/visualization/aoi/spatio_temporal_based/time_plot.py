# -*- coding: utf-8 -*-

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns
from scipy import stats

from vision_toolkit.aoi.aoi_base import AoISequence


def AoI_time_plot(input, **kwargs):
    """


    Parameters
    ----------
    input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    assert isinstance(input, AoISequence), "Input must be an AoISequence object"
    plt.style.use("seaborn-v0_8")

    seq_ = input.sequence
    dur_ = input.durations
    nb_aoi = input.nb_aoi

    colors_sns = sns.color_palette("pastel", n_colors=nb_aoi)

    c_ = sorted(set(seq_))
    i_dict = dict()
    for i, k_ in enumerate(c_):
        i_dict.update({k_: i})

    temp_bin = kwargs.get("AoI_time_plot_temporal_resolution", 0.001)
    seq_ = [
        [i_dict[seq_[i]] for _ in range(int(np.ceil(dur_[i] / temp_bin)))]
        for i in range(len(seq_))
    ]
    seq_ = list(itertools.chain(*seq_))

    for i in range(nb_aoi):
        plt.plot([i] * len(seq_), linestyle="dotted", linewidth=4, color=colors_sns[i])
    plt.plot(
        seq_,
        color="black",
        linewidth=1,
    )

    plt.xticks(np.linspace(0, len(seq_), 5), np.linspace(0, len(seq_), 5) * temp_bin)
    plt.yticks(list(range(len(c_))), list(c_))

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("AoI", fontsize=14)
    # plt.title('AoI time-plot', fontsize=18)

    dest_ = kwargs.get("AoI_time_plot_save", None)
    if dest_ is not None:
        fig = plt.gcf()
        fig.savefig(dest_ + ".png", dpi=250)

    plt.show()
    plt.clf()
