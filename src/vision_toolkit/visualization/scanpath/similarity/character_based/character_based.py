# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_character_based(seq_1, seq_2, opt_links, id_1, id_2):
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
