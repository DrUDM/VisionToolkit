# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import dispersion_metric, interval_merging


def process_IVDT(data_set, config):
    """
    Adapted from Komogortsev & Karpov (2013).
    Identifies saccades like the I-VT algorithm.
    Distinguishes pursuits from fixations using a
    modified version of the I-DT algorithm.
        - T_s = saccade velocity threshold.
        - n_w = temporal window size.
        – T_d = dispersion threshold.
    """

    if config["verbose"]:
        print("Processing VDT Identification...")
        start_time = time.time()

    # Eye movement inputs
    a_sp = data_set["absolute_speed"]
    s_f = config["sampling_frequency"]

    if config["distance_type"] == "euclidean":
        x_a = data_set["x_array"]
        y_a = data_set["y_array"]

    if config["distance_type"] == "angular":
        theta_coord = data_set["theta_coord"]
        x_a = theta_coord[0, :]
        y_a = theta_coord[1, :]

    # Algorithm parameters
    t_s = config["IVDT_saccade_threshold"]
    t_du = int(np.ceil(config["IVDT_window_duration"] * s_f))

    t_di = config["IVDT_dispersion_threshold"]

    i_sac = np.ones(len(x_a))
    i_purs = np.zeros_like(i_sac)
    i_fix = np.zeros_like(i_sac)

    not_sac = a_sp <= t_s
    wn_saccade = np.where(not_sac == True)[0]

    _ints = interval_merging(wn_saccade)

    # We now work with intersaccadic intervals
    for _int in _ints:
        # create vector of current indices
        cur_idx = np.arange(_int[0], _int[1] + 1)

        if len(cur_idx) > t_du:
            i = cur_idx[0]
            while (i + t_du) < _int[1]:
                j = i + t_du

                # current positions to use
                x_cur = x_a[i:j]
                y_cur = y_a[i:j]

                # Compute dispersion of points in the window
                d = dispersion_metric(x_cur, y_cur)

                # If dispersion is lower than the threshold
                # we add more points to the window until
                # the dispersion exceeds the threshold
                if d < t_di:
                    while d < t_di and j <= _int[1]:
                        j += 1

                        # update current positions
                        x_cur_u = x_a[i:j]
                        y_cur_u = y_a[i:j]

                        d = dispersion_metric(x_cur_u, y_cur_u)

                    # Mark points inside the window as fixations
                    i_fix[i : j - 1] = 1
                    i_sac[i : j - 1] = 0

                    i = j

                # Otherwise, mark the first point in the window
                # as a smooth pursuit
                else:
                    i_purs[i] = 1
                    i_sac[i] = 0
                    i += 1

            # remaining points of the interval are marked as smooth pursuit
            i_purs[i : j + 1] = 1
            i_sac[i : j + 1] = 0

    if config["verbose"]:
        print("\n...VDT Identification done\n")
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
