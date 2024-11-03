# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import seaborn as sns

pio.renderers.default = "browser"

import itertools

import matplotlib.pyplot as plt

from vision_toolkit.aoi.aoi_base import AoISequence


def AoI_scarf_plot(input, **kwargs):
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

    ## For multiple AoISequence inputs
    if isinstance(input, list):
        assert all(
            isinstance(in_, AoISequence) for in_ in input
        ), "Input must be a list of AoISequence objects"

        S_s = [in_.sequence for in_ in input]
        S_dur = [in_.durations for in_ in input]
        config = input[0].config

        to_df = []
        for i in range(len(S_s)):
            ## If AoISequence computed from oculomotor series
            seg_results = input[i].fixation_analysis

            if seg_results is not None:
                f_intervals = seg_results.segmentation_results["fixation_intervals"]
                s_f = config["sampling_frequency"]

                assert len(f_intervals) == len(
                    S_s[i]
                ), "'AoI_temporal_binning' must be set to 'False'"

                starts = [f_interval[0] / s_f for f_interval in f_intervals]
                ends = [f_interval[1] / s_f for f_interval in f_intervals]
            ## If AoISequence directly computed from a list of AoI strings
            else:
                assert (
                    S_dur[i] is not None
                ), "Input AoISequence does not contain duration data"

                cumsum = np.concatenate((np.array([0]), np.cumsum(S_dur[i])))
                starts = cumsum[:-1]
                ends = cumsum[1:]

            [
                to_df.append(
                    dict(
                        Task=str(i), Start=starts[k], Finish=ends[k], Resource=S_s[i][k]
                    )
                )
                for k in range(len(starts))
            ]
        df = pd.DataFrame(to_df)
        fig = plot_scarf(df, max([len(dur_) for dur_ in S_dur]))

    ## For unique AoISequence input
    else:
        assert isinstance(input, AoISequence), "Input must be an AoISequence object"

        seq_ = input.sequence
        dur_ = input.durations
        config = input.config
        seg_results = input.fixation_analysis

        ## If AoISequence computed from oculomotor series
        if seg_results is not None:
            f_intervals = seg_results.segmentation_results["fixation_intervals"]
            s_f = config["sampling_frequency"]

            assert len(f_intervals) == len(
                seq_
            ), "'AoI_temporal_binning' must be set to 'False'"

            starts = [f_interval[0] / s_f for f_interval in f_intervals]
            ends = [f_interval[1] / s_f for f_interval in f_intervals]
        ## If AoISequence directly computed from a list of AoI strings
        else:
            assert dur_ is not None, "Input AoISequence does not contain duration data"

            cumsum = np.concatenate((np.array([0]), np.cumsum(dur_)))
            starts = cumsum[:-1]
            ends = cumsum[1:]

        df = pd.DataFrame(
            [
                dict(Task="0", Start=starts[i], Finish=ends[i], Resource=seq_[i])
                for i in range(len(starts))
            ]
        )
        fig = plot_scarf(df, len(seq_))

    dest_ = kwargs.get("AoI_scarf_plot_save", None)

    if dest_ is not None:
        fig.write_image(dest_ + ".png")


def plot_scarf(df, m_len):
    """


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    m_len : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    c_ = sorted(list(set(df["Resource"].tolist())))

    colors_sns = sns.color_palette("pastel", n_colors=len(c_))
    colors = dict({})
    for i, c in enumerate(c_):
        colors.update({c: colors_sns[i]})

    ## Create a scarf plot
    fig = ff.create_gantt(
        df,
        index_col="Resource",
        bar_width=0.4,
        show_colorbar=True,
        group_tasks=True,
        colors=colors,
    )
    ## Update the layout
    fig.update_layout(
        xaxis_type="linear",
        height=400,
        width=max(300, m_len * 30),
        xaxis_title="Time (s)",
        yaxis_title="AoI sequence index",
        title_text="Scarfplot of AoI sequences",
        legend=dict(title=dict(text="AoI")),
    )

    fig.show()

    return fig
