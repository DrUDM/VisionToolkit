# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IKM(values, config, ref_image=None):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """

    ## Get inputs and parameters
    pos_ = values[0:2]
    dur_ = values[2]

    n_st = config["AoI_IKM_cluster_number"]

    if n_st == "search":
        ## Search for best number of centers according to the silhouette score
        k_min = config["AoI_IKM_min_clusters"]
        k_max = config["AoI_IKM_max_clusters"]

        models = dict({})
        sc_ = []

        for n_ in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=n_, random_state=0, n_init="auto").fit(pos_.T)
            models[n_] = kmeans
            sc_.append(silhouette_score(pos_.T, kmeans.labels_))

        ## Keep model with max slhouette score
        n_st = np.argmax(sc_) + k_min
        kmeans = models[n_st]

        print("{n_st} centers found\n".format(n_st=n_st))

    else:
        ## Compute k-means with predefined number of centers
        kmeans = KMeans(n_clusters=n_st, random_state=0, n_init="auto").fit(pos_.T)

    seq_ = kmeans.labels_

    ## Keep track of clusters
    centers_ = dict({})
    clus_ = dict({})

    for i in range(n_st):
        vals_ = np.argwhere(np.array(seq_) == int(i)).T[0]
        clus_.update({chr(i + 65): vals_})

        ## Centers are computed as the mean position of clustered fixations
        centers_.update({chr(i + 65): np.mean(pos_[:, vals_], axis=1)})

    ## Compute final AoI sequence
    seq_, seq_dur = compute_aoi_sequence(seq_, dur_, config)

    if config["display_AoI_identification"]:
        ## Plot clusters
        if ref_image is None:
            display_aoi_identification(pos_, clus_, config)
        else:
            display_aoi_identification_reference_image(pos_, clus_, config, ref_image)

    results = dict(
        {
            "AoI_sequence": seq_,
            "AoI_durations": seq_dur,
            "centers": centers_,
            "clustered_fixations": clus_,
        }
    )

    return results
