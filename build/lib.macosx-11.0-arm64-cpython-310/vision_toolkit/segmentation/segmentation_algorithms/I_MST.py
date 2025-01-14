# -*- coding: utf-8 -*-
import time

import networkx as nx
import numpy as np
from networkx.algorithms import tree
from scipy.spatial.distance import cdist

from vision_toolkit.utils.segmentation_utils import (
    centroids_from_ints,
    interval_merging)


def process_IMST(data_set, config):
    assert (
        config["distance_type"] == "euclidean"
    ), "'Distance type' must be set to 'euclidean"

    if config["verbose"]:
        print("Processing MST Identification...")
        start_time = time.time()

    x_a = data_set["x_array"]
    y_a = data_set["y_array"]

    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    g_p = np.concatenate((x_a.reshape(n_s, 1), y_a.reshape(n_s, 1)), axis=1)

    vareps = config["IMST_distance_threshold"]

    t_du = config["IMST_window_duration"]
    t_du = int(np.ceil(config["IMST_window_duration"] * s_f))

    i_fix = np.array([False] * n_s)

    i = 0
    while i + t_du < n_s:
        j = i + t_du

        w_gp = g_p[i:j]

        # Compute distance matrix between points of the considered data window
        d_m = cdist(w_gp, w_gp, metric="euclidean")

        g = nx.from_numpy_array(d_m, create_using=nx.MultiGraph())
        mst = tree.minimum_spanning_tree(g, algorithm="prim")

        edgelist = mst.edges(data=True)

        # Classify
        for edge in edgelist:
            w_ = edge[2]["weight"]

            i_mst = edge[0]
            j_mst = edge[1]

            # Fixation/Saccade distinction
            if w_ < vareps:
                i_fix[i + i_mst] = True
                i_fix[i + j_mst] = True

        i = j

    if config["verbose"]:
        print("Done")

    wi_fix = np.where(i_fix == True)[0]

    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
    )

    if config["verbose"]:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=config["min_sac_duration"]
            )
        )

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True] * config["nb_samples"])

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config["min_fix_duration"] * s_f))

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1] : s_int[0] + 1] = False

    if config["verbose"]:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config["min_fix_duration"]
            )
        )

    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config["min_fix_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    if config["verbose"]:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=config["status_threshold"]
            )
        )

    assert len(f_ints) == len(
        ctrds
    ), "Interval set and centroid set have different lengths"

    if config["verbose"]:
        print("\n...MST Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Keep track of index that were effectively labeled
    i_lab = np.array([False] * config["nb_samples"])

    for f_int in f_ints:
        i_lab[f_int[0] : f_int[1] + 1] = True

    for s_int in s_ints:
        i_lab[s_int[0] : s_int[1] + 1] = True

    return dict(
        {
            "is_labeled": i_lab,
            "fixation_intervals": f_ints,
            "saccade_intervals": s_ints,
            "centroids": ctrds,
        }
    )
