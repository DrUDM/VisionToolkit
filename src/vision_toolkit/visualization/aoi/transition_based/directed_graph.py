# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def display_transition_matrix(t_mat):
    ax = sns.heatmap(t_mat, square=True, cmap="viridis")
    for i in range(t_mat.shape[0]):
        for j in range(t_mat.shape[0]):
            val_ = np.round(t_mat[j, i], 2)
            ax.text(i + 0.5, j + 0.5, val_, color="white", ha="center", va="center")

    label_list = [chr(i + 65) for i in range(t_mat.shape[0])]

    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list, rotation=0)

    # fig = plt.gcf()
    # fig.savefig('example_transition_mat.png', dpi=200)

    plt.grid(None)
    plt.show()
    plt.clf()


def AoI_directed_graph(t_mat, centers, ref_image):
    ax = sns.heatmap(t_mat, square=True, cmap="viridis")
    for i in range(t_mat.shape[0]):
        for j in range(t_mat.shape[0]):
            val_ = np.round(t_mat[j, i], 2)
            ax.text(i + 0.5, j + 0.5, val_, color="white", ha="center", va="center")

    label_list = [chr(i + 65) for i in range(t_mat.shape[0])]

    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list, rotation=0)

    # fig = plt.gcf()
    # fig.savefig('example_transition_mat.png', dpi=200)

    plt.grid(None)
    plt.show()
    plt.clf()

    plt.imshow(ref_image, alpha=0.6)
    plt.grid(None)

    n_aoi = t_mat.shape[0]
    from_, to_, widths = [], [], []
    edge_label = dict({})

    s_ = np.sum(t_mat, axis=1)
    l_s = list(np.argwhere(s_ == 0).flatten())

    for i in range(n_aoi):
        for j in range(n_aoi):
            if i != j and i not in l_s and j not in l_s:
                from_.append(chr(i + 65))
                to_.append(chr(j + 65))
                if t_mat[i, j] > 0:
                    widths.append(t_mat[i, j] * 10)
                else:
                    widths.append(np.nan)
                edge_label.update({(chr(i + 65), chr(j + 65)): t_mat[i, j]})

    nodePosDict = dict({})

    for k in centers.keys():
        nodePosDict.update({k: centers[k]})
    # for i in range(n_aoi):
    #    nodePosDict.update({chr(i + 65): centers[i]})

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
        connectionstyle="arc3, rad = 0.2",
        width=widths,
    )

    # plt.axis('on')
    # plt.xlabel("Horizontal position (px)", fontsize =14)
    # plt.ylabel("Vertical position (px)", fontsize =14)

    # plt.xticks(fontsize = 10)
    # plt.yticks(fontsize = 10)

    # fig = plt.gcf()
    # fig.savefig('example_directed_graph.png', dpi=200)

    plt.show()
    plt.clf()
