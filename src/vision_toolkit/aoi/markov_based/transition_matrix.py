# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.aoi.pattern_mining.n_gram import NGram


def transition_matrix(seq_, nb_aoi):
    """


    Parameters
    ----------
    seq_ : TYPE
        DESCRIPTION.
    nb_aoi : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    ## Initialize the transition matrix
    t_m = np.zeros((nb_aoi, nb_aoi))
    ## Compute N-Grams of length 2
    h_ = NGram(seq_, 2).table
    # print(seq_)
    # print(h_)
    ## Fill the transition matrix
    for k_ in h_.keys():
        val_ = h_[k_]
        g_r_ = k_.split(",")

        g_ = g_r_[0]
        r_ = g_r_[1]

        t_m[ord(g_) - 65, ord(r_) - 65] = val_
    ## Normalize raws to 1
    for i in range(nb_aoi):
        s_ = np.sum(t_m[i])
        if s_ > 0:
            t_m[i] /= s_

    return t_m
