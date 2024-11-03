# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import DBSCAN

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IDT(values, config, ref_image=None):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    ## Get inputs and parameters
    pos_ = values[0:2]
    dur_ = values[2]

    eps = config["AoI_IDT_density_threshold"]
    min_samples = config["AoI_IDT_min_samples"]

    ## Compute density based clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_.T)

    seq_ = dbscan.labels_
    t_k = seq_ >= 0

    pos_ = pos_[:, t_k]
    dur_ = dur_[t_k]
    seq_ = seq_[t_k]

    ## Keep track of clusters
    centers_ = dict({})
    clus_ = dict({})

    for i in range(list(set(seq_))[-1] + 1):
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
