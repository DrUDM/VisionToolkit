# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import MeanShift

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IMS(values, config, ref_image=None):
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
    bandwidth = config["AoI_IMS_bandwidth"]
    ## Compute mean-shift
    ms = MeanShift(bandwidth=bandwidth, cluster_all=True).fit(pos_.T)
    seq_ = ms.labels_
    t_k = seq_ >= 0
    pos_ = pos_[:, t_k]
    dur_ = dur_[t_k]
    seq_ = seq_[t_k]
    ## Keep track of clusters
    centers_ = dict({})
    clus_ = dict({})
    to_relabel = []
    i_ = 0
    ## To avoid signle point clusters
    for i in range(list(set(seq_))[-1] + 1):
        vals_ = np.argwhere(np.array(seq_) == int(i)).T[0]
        if len(vals_) >= 2:
           
            clus_.update({chr(i_ + 65): vals_})
            seq_[vals_] = i_
            centers_.update({chr(i_ + 65): np.mean(pos_[:, vals_], axis=1)})
            i_ += 1
        else:
            to_relabel += list(vals_)
    ## Relabel single points
    if to_relabel != []:
        centers_array = np.array([centers_[k_] for k_ in sorted(list(centers_.keys()))])
        for val in to_relabel:
            pos_l = pos_[:, val]
            d_ = np.sum((centers_array - pos_l) ** 2, axis=1)
            c_val = np.argmin(d_)
            seq_[val] = c_val
            old_clus = sorted(list(clus_[chr(c_val + 65)]) + [val])
            clus_[chr(c_val + 65)] = np.array(old_clus)
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
