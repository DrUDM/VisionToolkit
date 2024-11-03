# -*- coding: utf-8 -*-

import itertools
from itertools import groupby
from operator import itemgetter

import numpy as np


def dispersion_metric(x_array, y_array):
    """

    Parameters
    ----------
    x_array : TYPE
        DESCRIPTION.
    y_array : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return max(x_array) - min(x_array) + max(y_array) - min(y_array)


def centroids_from_ints(f_ints, x_coords, y_coords):
    """

    Parameters
    ----------
    w_i : TYPE
        DESCRIPTION.
    x_coords : TYPE
        DESCRIPTION.
    y_coords : TYPE
        DESCRIPTION.
    p_o : TYPE, optional
        DESCRIPTION. The default is 0.
    s_o : TYPE, optional
        DESCRIPTION. The default is 0.
    min_int_size : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    centroids : TYPE
        DESCRIPTION.

    """
    centroids = list()

    for f_int in f_ints:
        l_ctrd = comp_centroid(
            x_coords[f_int[0] : f_int[-1] + 1], y_coords[f_int[0] : f_int[-1] + 1]
        )

        centroids.append(l_ctrd)

    return centroids


def comp_centroid(x_coords, y_coords):
    """

    Parameters
    ----------
    x_coords : TYPE
        DESCRIPTION.
    y_coords : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    _len = len(x_coords)
    centroid_x = sum(x_coords) / _len
    centroid_y = sum(y_coords) / _len

    return [centroid_x, centroid_y]


def interval_merging(
    w_i, min_int_size=0, max_int_size=np.inf, p_o=0, s_o=0, status=None, proportion=0.95
):
    """

    Parameters
    ----------
    w_i : TYPE
        DESCRIPTION.
    min_int_size : TYPE, optional
        DESCRIPTION. The default is 0.
    p_o : TYPE, optional
        DESCRIPTION. The default is 0.
    s_o : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    intervals : TYPE
        DESCRIPTION.

    """
    intervals = list()

    for k, g in groupby(enumerate(w_i), lambda ix: ix[0] - ix[1]):
        i_l = list(map(itemgetter(1), g))
        if (i_l[-1] + s_o - (i_l[0] - p_o)) >= min_int_size and (
            i_l[-1] + s_o - (i_l[0] - p_o)
        ) < max_int_size:
            ends_local = [i_l[0] - p_o, i_l[-1] + s_o]

            if not status is None:
                if (
                    np.sum(status[i_l[0] - p_o : i_l[-1] + s_o + 1])
                    / (i_l[-1] + s_o - i_l[0] - p_o)
                    > proportion
                ):
                    intervals.append(ends_local)

            else:
                intervals.append(ends_local)

    return intervals


def dict_vectorize(dict_list):
    """

    Parameters
    ----------
    dict_list : TYPE
        DESCRIPTION.

    Returns
    -------
    res_ : TYPE
        DESCRIPTION.

    """
    res_ = []

    for k in dict_list[0].keys():
        res_.append(
            np.array(
                list(itertools.chain.from_iterable([list(d[k]) for d in dict_list]))
            )
        )

    res_ = np.array(res_).T

    return res_


def standard_normalization(features, mu=None, sigma=None):
    """

    Parameters
    ----------
    features : TYPE
        DESCRIPTION.

    Returns
    -------
    features : TYPE
        DESCRIPTION.

    """
    if not isinstance(mu, np.ndarray):
        mu = np.mean(features, axis=0)

    if not isinstance(sigma, np.ndarray):
        sigma = np.std(features, axis=0)
        sigma[sigma == 0] = features[0][sigma == 0]

    features = (features - mu) / sigma

    return features, mu, sigma
