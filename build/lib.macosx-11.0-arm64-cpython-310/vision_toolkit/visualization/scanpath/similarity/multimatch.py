# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_multi_match(seq_1, seq_2, opt_links, id_1, id_2):
    trace1 = go.Scatter3d(
        x=np.arange(len(seq_1[0])),
        y=seq_1[0],
        z=seq_1[1],
        mode="lines+markers",
        marker=dict(size=5, symbol="circle", opacity=0.9),
        name="Sequence {id_1}".format(id_1=id_1),
    )

    trace2 = go.Scatter3d(
        x=np.arange(len(seq_2[0])),
        y=seq_2[0],
        z=seq_2[1],
        mode="lines+markers",
        marker=dict(size=5, symbol="circle", opacity=0.9),
        name="Sequence {id_2}".format(id_2=id_2),
    )
    to_trace = [trace1, trace2]

    for i in range(len(opt_links)):
        l_ = opt_links[i]
        to_trace.append(
            go.Scatter3d(
                x=np.array(l_[2]),
                y=np.array([l_[0, 0], l_[1, 0]]),
                z=np.array([l_[0, 1], l_[1, 1]]),
                mode="lines",
                showlegend=False,
                line=dict(color="#7f7f7f", width=1.0),
            )
        )

    fig = go.Figure(data=to_trace)

    fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=3, y=1, z=1))
    fig.update_layout(legend=dict(font=dict(size=20)))

    fig.update_layout(
        scene=dict(
            xaxis_title="Time-stamps",
            yaxis_title="Horizontal axis",
            zaxis_title="Vertical axis",
        ),
        margin=dict(r=20, b=10, l=10, t=10),
    )

    fig.show()


def plot_simplification(s_1, s_1_s, s_2, s_2_s, id_1, id_2):
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()

    ax.plot(s_1[0], s_1[1], linewidth=0.8, color="purple")

    for i in range(len(s_1[0])):
        dur = s_1[2, i]
        circle = plt.Circle(
            (s_1[0, i], s_1[1, i]),
            dur * 35,
            linewidth=0.8,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Sequence {id_1}".format(id_1=id_1), fontsize=18)
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()

    ax.plot(s_1_s[0], s_1_s[1], linewidth=0.8, color="purple")

    for i in range(len(s_1_s[0])):
        dur = s_1_s[2, i]
        circle = plt.Circle(
            (s_1_s[0, i], s_1_s[1, i]),
            dur * 35,
            linewidth=0.8,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Sequence {id_1} Simplified".format(id_1=id_1), fontsize=18)
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()

    ax.plot(s_2[0], s_2[1], linewidth=0.8, color="purple")

    for i in range(len(s_2[0])):
        dur = s_2[2, i]
        circle = plt.Circle(
            (s_2[0, i], s_2[1, i]),
            dur * 35,
            linewidth=0.8,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Sequence {id_2}".format(id_2=id_2), fontsize=18)
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()

    ax.plot(s_2_s[0], s_2_s[1], linewidth=0.8, color="purple")

    for i in range(len(s_2_s[0])):
        dur = s_2_s[2, i]
        circle = plt.Circle(
            (s_2_s[0, i], s_2_s[1, i]),
            dur * 35,
            linewidth=0.8,
            color="darkblue",
            fill=False,
        )

        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=14)
    plt.ylabel("Vertical position (px)", fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Sequence {id_2} Simplified".format(id_2=id_2), fontsize=18)
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()
