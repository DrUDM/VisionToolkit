# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IVMP(data_set, config):
    """
    Identifies saccades like the I-VT algorithm.
    Distinguishes pursuits from fixations using the movement
    pattern of the eye trace.
        - T_s = saccade velocity threshold.
        - n_w = temporal window size.
        – T_m = movement threshold.
    """

    if config["verbose"]:
        print("Processing VMP Identification...")
        start_time = time.time()

    # Eye movement parameters
    a_sp = data_set["absolute_speed"]
    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    x_array = data_set["x_array"]
    y_array = data_set["y_array"]

    # Algorithm parameters
    t_s = config["IVMP_saccade_threshold"]
    t_du = int(np.ceil(config["IVMP_window_duration"] * s_f))
    t_r = config["IVMP_rayleigh_threshold"]

    i_sac = np.ones(len(x_array))
    i_purs = np.zeros_like(i_sac)
    i_fix = np.zeros_like(i_sac)

    not_sac = a_sp <= t_s
    wn_saccade = np.where(not_sac == True)[0]
    w_sacc = np.where(not_sac == False)[0]
    i_sac[w_sacc] = 1

    _ints = interval_merging(wn_saccade)

    # Compute position difference vectors
    diff_ = np.zeros((2, n_s))

    diff_[0, :-1] = x_array[1:] - x_array[:-1]
    diff_[1, :-1] = y_array[1:] - y_array[:-1]
    diff_[:, -1] = diff_[:, -2]

    # Compute successive vector directions relative to the horizontal axis
    # for the Rayleigh z-score
    suc_dir = np.zeros_like(a_sp)

    # to avoid numerical instability
    diff_ += 1e-10

    _m = diff_[1, :] < 0
    _p = diff_[1, :] >= 0

    n_p = np.linalg.norm(diff_[:, _p], axis=0)
    suc_dir[_p] = np.arccos(np.divide(diff_[0, :][_p], n_p, where=n_p > 0))

    n_m = np.linalg.norm(diff_[:, _m], axis=0)
    suc_dir[_m] = 2 * np.pi - np.arccos(np.divide(diff_[0, :][_m], n_m, where=n_m > 0))

    # We now work with intersaccadic intervals
    for _int in _ints:
        # create vector of current indices
        cur_idx = np.arange(_int[0], _int[1] + 1)
        i_fix[_int[0] : _int[1] + 1] = 1
        i_sac[_int[0] : _int[1] + 1] = 0

        if len(cur_idx) > t_du:
            i = cur_idx[0]
            while (i + t_du) < _int[1]:
                j = i + t_du

                pos_unitary_circle = np.array(
                    [np.cos(suc_dir[i:j]), np.sin(suc_dir[i:j])]
                )

                rm_mat = np.sum(pos_unitary_circle, axis=1) / t_du

                # Use movement threshold to detect pursuits
                z_score = np.linalg.norm(rm_mat) ** 2

                if z_score > t_r:
                    i_purs[i:j] = 1
                    i_fix[i:j] = 0

                i += t_du

        else:
            print(cur_idx)

    if config["verbose"]:
        print("\n...VMP Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return dict(
        {
            "is_saccade": i_sac == 1,
            "saccade_intervals": interval_merging(np.where((i_sac == 1) == True)[0]),
            "is_pursuit": i_purs == 1,
            "pursuit_intervals": interval_merging(np.where((i_purs == 1) == True)[0]),
            "is_fixation": i_fix == 1,
            "fixation_intervals": interval_merging(np.where((i_fix == 1) == True)[0]),
        }
    )
