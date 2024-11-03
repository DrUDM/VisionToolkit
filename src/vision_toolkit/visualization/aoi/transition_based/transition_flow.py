# -*- coding: utf-8 -*-

from itertools import groupby

import holoviews as hv
import imageio as iio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from vision_toolkit.aoi.markov_based.markov_based import AoI_transition_matrix

hv.extension("bokeh")
hv.output(size=300)


def AoI_transition_flow(input):
    t_mat = AoI_transition_matrix(input)["AoI_transition_matrix"]
    seq_ = input.sequence

    c_ = sorted(list(set(seq_)))
    m_idx = []

    for c in c_:
        l_idx = np.mean([i for i, x in enumerate(seq_) if x == c])
        m_idx.append(l_idx)

    m_idx = np.array(m_idx)
    m_idx -= np.min(m_idx)

    n_layers = 4
    s_layer = np.max(m_idx + 1e-5) / n_layers
    layer = m_idx // s_layer

    w_layer = np.zeros(n_layers)
    nodePosDict = dict({})

    for i, c in enumerate(c_):
        l_ = int(layer[i])
        nodePosDict.update({c: np.array([w_layer[l_], l_])})
        w_layer[l_] += 1

    from_, to_, widths = [], [], []

    for i, a in enumerate(c_):
        for j, b in enumerate(c_):
            # if i!=j:
            if nodePosDict[b][1] > nodePosDict[a][1] and i != j:
                from_.append(a)
                to_.append(b)
                widths.append(t_mat[i, j] * 25)

    df = pd.DataFrame({"from": from_, "to": to_})
    G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph())

    nx.draw(
        G,
        with_labels=True,
        pos=nodePosDict,
        node_size=500,
        alpha=0.8,
        font_weight="bold",
        arrows=True,
        connectionstyle="arc3, rad = 0.1",
        width=widths,
    )

    plt.gca().invert_yaxis()

    fig = plt.gcf()
    fig.savefig("example_flow_diagram.png", dpi=200)

    plt.show()
    plt.clf()
