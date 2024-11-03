# -*- coding: utf-8 -*-

import numpy as np


def modified_k_coefficient(scanpath):
    """


    Parameters
    ----------
    scanpath : TYPE
        DESCRIPTION.

    Returns
    -------
    k_c : TYPE
        DESCRIPTION.

    """

    n_ = len(scanpath.values[0])

    mu_d = np.mean(scanpath.values[2])
    std_d = np.std(scanpath.values[2])

    a_s = np.linalg.norm(scanpath.values[0:2, 1:] - scanpath.values[0:2, :-1], axis=0)

    mu_a = np.mean(a_s)
    std_a = np.std(a_s)

    k_j = (scanpath.values[2, :-1] - mu_d) / std_d - (a_s - mu_a) / std_a
    k_c = np.sum(k_j) / (n_ - 1)

    return k_c
