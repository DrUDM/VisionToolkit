#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:53:54 2024

@author: quentinlaborde
"""

import copy

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns


def AoI_sankey_diagram(input, **kwargs):
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

    S_s = [in_.sequence for in_ in input]
    n_ = len(S_s)
    S_dur = [in_.durations for in_ in input]
    config = input[0].config
    centers = input[0].centers

    S_ends = dict()
    S_n = dict()

    resolution = kwargs.get("AoI_sankey_diagram_resolution", 1000)
    config.update({"AoI_sankey_diagram_resolution": resolution})

    nb_eval = kwargs.get("AoI_sankey_diagram_number_evaluations", 9)
    config.update({"AoI_sankey_diagram_number_evaluations": nb_eval})

    normalization = kwargs.get("AoI_sankey_diagram_normalization", False)
    config.update({"AoI_sankey_diagram_normalization": normalization})

    value_type = kwargs.get("AoI_sankey_diagram_interval_value", "frequent")
    config.update({"AoI_sankey_diagram_interval_value": value_type})

    for i in range(n_):
        ## If AoISequence computed from oculomotor series
        seg_results = input[i].fixation_analysis

        if seg_results is not None:
            f_intervals = seg_results.segmentation_results["fixation_intervals"]
            s_f = config["sampling_frequency"]

            ends = [f_intervals[i][0] / s_f for i in range(1, len(f_intervals))]
            ends.insert(0, 0.0)
            ends.append(f_intervals[-1][1] / s_f)

            S_ends.update({i: ends})

    if normalization:
        for i in range(n_):
            ends = S_ends[i]
            ## Normalize total duration to AoI_sankey_diagram_resolution
            ends = np.array(ends) / ends[-1] * resolution
            ## Get the index timestamp of transition between two AoI
            ends = np.ceil(ends).astype(int)
            S_ends.update({i: ends})

    else:
        max_end = max([S_ends[i][-1] for i in range(n_)])
        print(max_end)

        for i in range(n_):
            ends = S_ends[i]
            ## Normalize total duration to AoI_sankey_diagram_resolution
            ends = np.array(ends) / max_end * resolution
            ## Get the index timestamp of transition between two AoI
            ends = np.ceil(ends).astype(int)
            S_ends.update({i: ends})

    for i in range(n_):
        s_ends = S_ends[i]

        seq_ = S_s[i]
        n_seq = [seq_[0]]
        aoi_idx = 0

        for k in range(1, s_ends[-1]):
            ## If time stamp corresponds to a transition event, then increment AoI
            if k in s_ends:
                aoi_idx += 1
            n_seq.append(seq_[aoi_idx])
        S_n.update({i: n_seq})

    l_ = (resolution - 1) // nb_eval

    c_ = sorted(centers.keys())
    n_c = len(c_)
    i_dict = dict()

    for i, k_ in enumerate(c_):
        i_dict.update({k_: i})

    ## Compute transition matrix for each evaluation interval
    tm_dict = dict()

    S_n_max = dict()
    for k_ in S_n.keys():
        S_n_max.update({k_: copy.deepcopy(S_n[k_])})

    for k in range(nb_eval):
        t_m = np.zeros((n_c, n_c))

        for i in range(n_):
            s_n = S_n[i]
            ## If sequence length is not normalized
            if len(s_n) >= (k + 1) * l_ + 1:
                ## Use AoI visited at the beginning and end of each time interval
                if value_type == "last":
                    ## For the last evaluaton use the last AoI visited
                    if k == nb_eval - 1:
                        val_ = s_n[-1]
                    ## Else use the last visited AoI during the time interval
                    else:
                        val_ = s_n[(k + 1) * l_]
                    ## Update transition matrix entry
                    t_m[i_dict[s_n[k * l_]], i_dict[val_]] += 1 / n_

                ## Use most visited AoI during the time interval
                elif value_type == "frequent":
                    ## For the last evaluaton use the last AoI visited
                    if k == nb_eval - 1:
                        S_n_max[i][k * l_ + 1 :] = s_n[-1] * len(
                            S_n_max[i][k * l_ + 1 :]
                        )
                    ## Else use the most visited AoI during the time interval preceding the evaluation
                    else:
                        loc_list = s_n[k * l_ + 1 : (k + 1) * l_ + 1]
                        t_k = max(set(loc_list), key=loc_list.count)
                        S_n_max[i][k * l_ + 1 : (k + 1) * l_ + 1] = [t_k] * l_
                    ## Update transition matrix entry
                    t_m[i_dict[S_n_max[i][k * l_]], i_dict[S_n_max[i][k * l_ + 1]]] += (
                        1 / n_
                    )

                else:
                    raise ValueError(
                        "'AoI_sankey_diagram_interval_value' must be set to 'last' or 'frequent'"
                    )

        tm_dict.update({k + 1: t_m})

    fig = plot_sankey(tm_dict, S_n, c_, n_, i_dict, l_, nb_eval)
    verbose(config)

    ## save plotly figure if requested
    dest_ = kwargs.get("AoI_sankey_diagram_save", None)
    if dest_ is not None:
        fig.write_image(dest_ + ".png")


def plot_sankey(tm_dict, S_n, c_, n_, i_dict, l_, nb_eval):
    """


    Parameters
    ----------
    tm_dict : TYPE
        DESCRIPTION.
    S_n : TYPE
        DESCRIPTION.
    c_ : TYPE
        DESCRIPTION.
    n_ : TYPE
        DESCRIPTION.
    i_dict : TYPE
        DESCRIPTION.
    l_ : TYPE
        DESCRIPTION.
    nb_eval : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    pio.renderers.default = "browser"

    colors_sns = sns.color_palette("pastel", n_colors=len(c_))

    colors = []
    lab_colors = []

    for color in colors_sns:
        col = "rgba({r}, {g}, {b}, 0.3)".format(
            r=color[0] * 255, g=color[1] * 255, b=color[2] * 255
        )
        colors.append(col)
        col = "rgba({r}, {g}, {b}, 0.8)".format(
            r=color[0] * 255, g=color[1] * 255, b=color[2] * 255
        )
        lab_colors.append(col)

    y_step = 1 / (len(c_) + 1)
    x_step = 1 / (nb_eval + 1)

    labels = copy.deepcopy(c_)
    labels_color = copy.deepcopy(lab_colors)
    x_ = [1e-6] * len(c_)
    y_ = [0.0] * len(c_)

    for k_ in S_n.keys():
        y_[i_dict[S_n[k_][0]]] = y_step

    y_ = list(np.cumsum(y_))
    y_ = sorted(list(set(y_)))

    if 0.0 in y_:
        y_.remove(0.0)

    y_ = [v - y_step + 1e-6 for v in y_]
    x_ = x_[: len(y_)]

    values_, sources_, targets_ = [], [], []
    colors_ = []

    for k_ in tm_dict.keys():
        labels_color += lab_colors
        local_tm = tm_dict[k_]

        y_l = [0.0] * len(c_)
        s_, t_ = np.nonzero(local_tm)

        for i in range(len(s_)):
            y_l[t_[i]] = y_step

            sources_.append(int(len(labels) - len(c_) + s_[i]))
            targets_.append(int(len(labels) + t_[i]))
            values_.append(local_tm[s_[i], t_[i]])
            colors_.append(colors[s_[i]])

        y_l = list(np.cumsum(y_l))
        y_l = sorted(list(set(y_l)))
        if 0.0 in y_l:
            y_l.remove(0.0)

        y_l = [v - y_step + 1e-6 for v in y_l]

        x_l = [x_step * k_] * len(y_l)
        labels += c_
        x_ += x_l
        y_ += y_l

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    thickness=5,
                    x=x_,
                    y=y_,
                    pad=0,
                    label=labels,
                    color=labels_color,
                    line=dict(({"width": 0})),
                ),
                link=dict(
                    source=sources_, target=targets_, value=values_, color=colors_
                ),
            )
        ]
    )

    ## Add title, size, margin etc (Optional)
    fig.update_layout(  # title_text="<b>AoI sankey diagram</b>",
        font_size=15, width=1200, height=800, margin=dict(t=50, l=50, b=50, r=30)
    )
    fig.show()

    return fig


def verbose(config):
    """


    Parameters
    ----------
    add_ : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    if config["verbose"]:
        print("\n --- Config used: ---\n")

        for it in config.keys():
            print(
                "# {it}:{esp}{val}".format(
                    it=it, esp=" " * (50 - len(it)), val=config[it]
                )
            )
        print("\n")
