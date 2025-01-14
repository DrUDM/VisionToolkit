# -*- coding: utf-8 -*-

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
from bokeh.sampledata.les_mis import data
from holoviews import dim, opts

hv.extension("bokeh")
hv.output(size=300)


def AoI_chord_diagram(
    hmm_model,
):
    t_mat = hmm_model.transition_matrix
    label_list = [chr(i + 65) for i in range(t_mat.shape[0])]

    colors_sns = sns.color_palette("pastel", n_colors=len(label_list))
    colors = []

    for color in colors_sns:
        col = "rgba({r}, {g}, {b}, 0.3)".format(
            r=color[0] * 255, g=color[1] * 255, b=color[2] * 255
        )
        colors.append(col)

    # colors = dict({})
    # for i, c in enumerate(label_list):
    #    colors.update({c: colors_sns[i]})

    data = [dict({"name": lab_}) for lab_ in label_list]
    node = hv.Dataset(pd.DataFrame(data), "index")

    link = pd.DataFrame(dict({"source": [], "target": [], "value": []}))

    for i in range(len(t_mat)):
        for j in range(len(t_mat)):
            if t_mat[i, j] > 0:
                new_row = dict(
                    {"source": i, "target": j, "value": int(t_mat[i, j] * 1e5)}
                )
                link.loc[len(link)] = new_row

    chord = hv.Chord((link, node)).select(value=(0, None))
    chord.opts(
        opts.Chord(
            cmap="Pastel1",
            edge_cmap="Pastel1",
            edge_color=dim("source").str(),
            labels="name",
            node_color=dim("index").str(),
            label_text_font_size="20pt",
        ),
    )

    if "bokeh_server" in locals():
        bokeh_server.stop()

    bokeh_server = pn.Row(chord).show(port=12345)


"""    
t_m = np.array([[0., .1, .9], 
                [.4, 0., .6], 
                [.5, .5, 0.]])



data_bis = []
for data_ in data['nodes']:
    data_bis.append(dict({'name': data_['name']}))
    
    
    
data_test = [dict({'name': 'A'}), 
             dict({'name': 'B'}), 
             dict({'name': 'C'})]

link_test = pd.DataFrame(dict({'source':[], 
                               'target':[], 
                               'value':[]}))
for i in range(len(t_m)):
    for j in range(len(t_m)):
        if t_m[i,j]>0:
            new_row = dict({'source': i, 
                            'target': j, 
                            'value': int(t_m[i,j]*1e5)})
      #  if i!=j:
            link_test.loc[len(link_test)] = new_row

  
#nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
nodes = hv.Dataset(pd.DataFrame(data_bis), 'index')
nodes_test = hv.Dataset(pd.DataFrame(data_test), 'index')


nodes.data.head()
links = pd.DataFrame(data['links'])

chord = hv.Chord((link_test, nodes_test)).select(value=(0, None))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))
 

if ("bokeh_server" in locals()):
    bokeh_server.stop()
    
bokeh_server = pn.Row(chord).show(port=12345)

 
#bokeh_server.stop()



#print((data['nodes']))
#print()
#print(data_bis)
#print()
print(links)
print(link_test)

"""
