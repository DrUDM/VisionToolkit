# -*- coding: utf-8 -*-
import itertools

import numpy as np
from scipy.spatial.distance import cdist


def spatial_bin(s_, x_nb_pixels, y_nb_pixels, x_size, y_size):
    if x_size == None:
        x_size = np.max(s_[0]) + 0.001

    if y_size == None:
        y_size = np.max(s_[1]) + 0.001

    elem_sizes = np.array([[x_size / x_nb_pixels], [y_size / y_nb_pixels]])
    bin_ = (
        np.minimum(s_, np.array([[x_size - 0.001], [y_size - 0.001]])) / elem_sizes
    ).astype(int)

    return bin_


def spatial_bin_d(s_1, s_2, x_nb_pixels, y_nb_pixels, x_size, y_size):
    if x_size == None:
        x_size = max(np.max(s_1[0]), np.max(s_2[0])) + 0.001

    if y_size == None:
        y_size = max(np.max(s_1[1]), np.max(s_2[1])) + 0.001

    elem_sizes = np.array([[x_size / x_nb_pixels], [y_size / y_nb_pixels]])

    s_1_b = np.floor(
        np.minimum(s_1, np.array([[x_size - 0.001], [y_size - 0.001]])) / elem_sizes
    )
    s_2_b = np.floor(
        np.minimum(s_2, np.array([[x_size - 0.001], [y_size - 0.001]])) / elem_sizes
    )

    # We use the following convention: first letter for the row
    # and second letter for the column
    bin_ = (
        [
            chr(65 + int(s_1_b[1, i])) + chr(65 + int(s_1_b[0, i]))
            for i in range(len(s_1_b[0]))
        ],
        [
            chr(65 + int(s_2_b[1, i])) + chr(65 + int(s_2_b[0, i]))
            for i in range(len(s_2_b[0]))
        ],
        elem_sizes,
    )

    return bin_


def spatial_temp_bin_d(s_1, s_2, temp_bin, x_nb_pixels, y_nb_pixels, x_size, y_size):
    if x_size == None:
        x_size = max(np.max(s_1[0]), np.max(s_2[0])) + 0.001

    if y_size == None:
        y_size = max(np.max(s_1[1]), np.max(s_2[1])) + 0.001

    elem_sizes = np.array([[x_size / x_nb_pixels], [y_size / y_nb_pixels]])

    s_1_b = np.floor(s_1[0:2] / elem_sizes)
    s_2_b = np.floor(s_2[0:2] / elem_sizes)

    ## We use the following convention: first letter for the row
    ## and second letter for the column
    s_1_c = [
        [chr(65 + int(s_1_b[1, i])) + chr(65 + int(s_1_b[0, i]))]
        * int(np.ceil(s_1[2, i] / temp_bin))
        for i in range(len(s_1_b[0]))
    ]

    s_2_c = [
        [chr(65 + int(s_2_b[1, i])) + chr(65 + int(s_2_b[0, i]))]
        * int(np.ceil(s_2[2, i] / temp_bin))
        for i in range(len(s_2_b[0]))
    ]

    bin_ = (
        list(itertools.chain.from_iterable(s_1_c)),
        list(itertools.chain.from_iterable(s_2_c)),
        elem_sizes,
    )

    return bin_


def dist_mat(x_s, y_s, elem_sizes, normalize=True):
    n = x_s * y_s
    i_dict = dict()
    c_ = np.zeros((y_s, x_s, 2))

    for i in range(y_s):
        c_[i, :, 0] = np.arange(0, x_s) * elem_sizes[0, 0]

    for j in range(x_s):
        c_[:, j, 1] = np.arange(0, y_s) * elem_sizes[1, 0]

    c_ = c_.reshape(y_s * x_s, 2)
    d_m = cdist(c_, c_, metric="euclidean")

    for i in range(n):
        y_1 = i // x_s
        x_1 = i % x_s
        i_dict.update({chr(65 + int(y_1)) + chr(65 + int(x_1)): i})

    if normalize:
        d_m = (d_m) / np.max(d_m)

    return d_m, i_dict


def dict_bin(x_nb_pixels, y_nb_pixels, elem_sizes):
    dict_ = dict()

    for i in range(x_nb_pixels):
        for j in range(y_nb_pixels):
            dict_.update(
                {
                    chr(65 + int(j))
                    + chr(65 + int(i)): np.array(
                        [i * elem_sizes[0, 0], j * elem_sizes[1, 0]]
                    )
                }
            )

    return dict_


def aoi_dict_dist_mat(centers, normalize=True):
    c_ = sorted(centers.keys())

    i_dict = dict()
    for i, k_ in enumerate(c_):
        i_dict.update({k_: i})

    d_ = np.array([centers[k_] for k_ in c_])

    d_m = cdist(d_, d_, metric="euclidean")

    if normalize:
        d_m = (d_m) / np.max(d_m)

    return d_m, i_dict
