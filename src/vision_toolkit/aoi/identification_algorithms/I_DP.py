# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gmean, norm

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IDP(values, config, ref_image=None):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """

    ## Get inputs and parameters
    pos_ = values[0:2]
    dur_ = values[2]
    n_s = len(dur_)

    k_sd = config["AoI_IDP_gaussian_kernel_sd"]
    center_method = config["AoI_IDP_centers"]

    dist_ = cdist(pos_.T, pos_.T)
    max_d = np.max(dist_)

    ## Compute local densities according to the Gaussian kernel with
    ## standard deviation set to 'k_sd'
    rho = np.array(
        [np.sum(norm.pdf(dist_[i], 0, k_sd)) - norm.pdf(0, 0, k_sd) for i in range(n_s)]
    )
    max_r = np.max(rho)
    delta = list()

    ## For each fixation, compute the distance of the closest data point
    ## of higher density
    for i in range(n_s):
        l_rho = rho[i]
        if l_rho < max_r:
            idxs = np.argwhere(rho > l_rho).T[0]
            delta.append(np.min(dist_[i, idxs]))
        ## For the max rhos value, set distance to the maximal possible distance between points
        else:
            delta.append(max_d)

    # Compute the gamma product
    delta = np.array(delta)
    gamma = rho * delta

    ## Compute the gamma threshold
    thresh = compute_threshold(gamma)

    ## Identify index of fixations with gamma value above the gamma threshold
    center_idx = np.argwhere(gamma > thresh).T[0]

    dist_f = dist_[:, center_idx]

    ## Compute the sequence of AoI-clustered fixation
    seq_ = list(np.argmin(dist_f, axis=1))

    ## Keep track of clusters
    centers_ = dict({})
    clus_ = dict({})

    for i in range(len(center_idx)):
        vals_ = np.argwhere(np.array(seq_) == int(i)).T[0]
        clus_.update({chr(i + 65): vals_})

    if center_method == "mean":
        for i in range(len(center_idx)):
            # Centers are computed as the mean position of clustered fixations
            centers_.update({chr(i + 65): np.mean(pos_[:, clus_[chr(i + 65)]], axis=1)})

    elif center_method == "raw_IDP":
        for i in range(len(center_idx)):
            # Centers are computed as the examplars from DP method
            centers_.update({chr(i + 65): pos_[:, center_idx[i]]})

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


def compute_threshold(gamma):
    """


    Parameters
    ----------
    gamma :TYPE
        DESCRIPTION.

    Returns
    -------
    thresh : TYPE
        DESCRIPTION.

    """

    ## Sort gamma in decreasing order
    gamma = np.sort(gamma)[::-1]

    n_s = len(gamma)
    weights = list()

    ## Compute weights for the weighted geometric mean
    for i in range(n_s):
        alpha = 2 ** (np.floor(np.log(n_s)) - np.ceil(np.log(i + 1)) + 1) - 1
        weights.append(alpha)

    ## Compute final threshold
    thresh = gmean(gamma, weights=weights)

    return thresh
