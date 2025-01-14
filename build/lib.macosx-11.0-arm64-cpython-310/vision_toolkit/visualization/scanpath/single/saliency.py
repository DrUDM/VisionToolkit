# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns


def plot_saliency_map(s_m):
    ax = sns.heatmap(s_m, square=True)

    ax.invert_yaxis()
    plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)
    plt.yticks(rotation=0)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    plt.show()
    plt.clf()
