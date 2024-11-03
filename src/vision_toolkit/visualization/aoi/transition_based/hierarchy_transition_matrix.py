# -*- coding: utf-8 -*-

import random
import string

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform

X = np.random.randint(0, 10, size=(120, 10))
df = pd.DataFrame(X)

# Initialize figure by creating upper dendrogram
fig = ff.create_dendrogram(df.values, orientation="bottom")
fig.for_each_trace(lambda trace: trace.update(visible=False))

for i in range(len(fig["data"])):
    fig["data"][i]["yaxis"] = "y2"

# Create Side Dendrogram
# dendro_side = ff.create_dendrogram(X, orientation='right', labels = labels)
dendro_side = ff.create_dendrogram(X, orientation="right")
for i in range(len(dendro_side["data"])):
    dendro_side["data"][i]["xaxis"] = "x2"

# Add Side Dendrogram Data to Figure
for data in dendro_side["data"]:
    fig.add_trace(data)

# Create Heatmap
dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(df.values)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves, :]
heat_data = heat_data[:, dendro_leaves]

heatmap = [
    go.Heatmap(x=dendro_leaves, y=dendro_leaves, z=heat_data, colorscale="Blues")
]

heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

# Add Heatmap Data to Figure
for data in heatmap:
    fig.add_trace(data)

# Edit Layout
fig.update_layout(
    {
        "width": 800,
        "height": 800,
        "showlegend": False,
        "hovermode": "closest",
    }
)
# Edit xaxis
fig.update_layout(
    xaxis={
        "domain": [0.15, 1],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "ticks": "",
    }
)
# Edit xaxis2
fig.update_layout(
    xaxis2={
        "domain": [0, 0.15],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
)

# Edit yaxis
fig.update_layout(
    yaxis={
        "domain": [0, 1],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
)
# # Edit yaxis2
fig.update_layout(
    yaxis2={
        "domain": [0.825, 0.975],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_tickfont=dict(color="rgba(0,0,0,0)"),
)

fig.show()
