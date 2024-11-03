# -*- coding: utf-8 -*-

import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2

from vision_toolkit.utils.segmentation_utils import (centroids_from_ints,
                                             interval_merging)
from vision_toolkit.utils.velocity_distance_factory import absolute_angular_distance


def process_I2MC(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """

    if config["verbose"]:
        print("Processing 2MC Identification...")
        start_time = time.time()

    if config["distance_type"] == "euclidean":
        x_a = data_set["x_array"]
        y_a = data_set["y_array"]

    if config["distance_type"] == "angular":
        theta_coord = data_set["theta_coord"]
        u_v = data_set["unitary_gaze_vectors"]

        x_a = theta_coord[0, :]
        y_a = theta_coord[1, :]

    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    t_du = np.floor(config["I2MC_window_duration"] * s_f / 2)
    t_mo = np.ceil(config["I2MC_moving_threshold"] * s_f)

    # interval merging for saccade with min_int_size = t_me_du-2
    t_me_du = np.ceil(config["I2MC_merging_duration_threshold"] * s_f)
    t_me_di = config["I2MC_merging_distance_threshold"]

    i_fix = np.array([True] * config["nb_samples"])

    n_wc = int(2 * t_du + 1)
    n_wc_2 = int(np.ceil(n_wc / 2))
    n_wc_4 = int(np.ceil(n_wc / 4))

    # Initialize final weights
    f_w = np.zeros(n_s)
    norm_ = np.zeros(n_s)

    i = 0
    while i + t_mo < n_s:
        i += t_mo

        # Compute index windows of two times t_du length centered on i
        w_c = np.minimum(
            np.maximum(0, np.arange(i - t_du, i + t_du + 1, dtype=int)), n_s - 1
        )

        X_1 = np.concatenate(
            (x_a[w_c].reshape(n_wc, 1), y_a[w_c].reshape(n_wc, 1)), axis=1
        )
        _, lbl_1 = kmeans2(X_1, 2, iter=15, minit="++")

        trs_1 = np.where(np.diff(lbl_1) != 0)[0] + 1

        # Compute weigths
        w_1 = np.zeros(n_wc)
        w_1[trs_1] = 1 / len(trs_1)

        # Add weigths
        f_w[w_c] += w_1

        # Under-sample: 1 for 2
        X_2 = np.concatenate(
            (x_a[w_c][::2].reshape(n_wc_2, 1), y_a[w_c][::2].reshape(n_wc_2, 1)), axis=1
        )
        _, lbl_2 = kmeans2(X_2, 2, iter=15, minit="++")

        trs_2 = np.where(np.diff(lbl_2) != 0)[0] + 1

        # Compute weigths
        w_2 = np.zeros(n_wc_2)
        w_2[trs_2] = 1 / len(trs_2)

        # Repeat results since sequence was under-sampled and add weigth
        f_w[w_c] += np.repeat(w_2, 2)[:n_wc]

        # Under-sample: 1 for 4
        X_4 = np.concatenate(
            (x_a[w_c][::4].reshape(n_wc_4, 1), y_a[w_c][::4].reshape(n_wc_4, 1)), axis=1
        )

        _, lbl_4 = kmeans2(X_4, 2, iter=15, minit="++")

        trs_4 = np.where(np.diff(lbl_4) != 0)[0] + 1

        # Compute weigths
        w_4 = np.zeros(n_wc_4)
        w_4[trs_4] = 1 / len(trs_4)

        # Repeat results since sequence was under-sampled and add weigth
        f_w[w_c] += np.repeat(w_4, 4)[:n_wc]

        # Normalization factor to get the average
        norm_[w_c] += np.ones(n_wc)

    f_w /= norm_

    sac_thrs = np.mean(f_w) + np.std(f_w)
    wi_fix = np.where(f_w < sac_thrs)[0]

    # Keep fixation intervals with duration above I2MC_merging_duration_threshold
    f_ints = interval_merging(wi_fix, min_int_size=t_me_du)

    # Initialize final i_fix
    i_fix = np.array([False] * config["nb_samples"])

    # Merging fixations separated by a distance below I2MC_merging_distance_threshold
    for i in range(len(f_ints)):
        f_int = f_ints[i]
        i_fix[f_int[0] : f_int[1] + 1] = True

        if i > 0:
            f_int_p = f_ints[i - 1]

            if config["distance_type"] == "euclidean":
                arr_ = np.array(
                    [x_a[f_int[0]] - x_a[f_int_p[1]], y_a[f_int[0]] - y_a[f_int_p[1]]]
                )

                dist_ = np.linalg.norm(arr_)

            elif config["distance_type"] == "angular":
                uv_1 = u_v[:, f_int_p[1]]
                uv_2 = u_v[:, f_int[0]]

                dist_ = absolute_angular_distance(uv_1, uv_2)

            if dist_ < t_me_di:
                i_fix[f_int_p[1] : f_int[0]] = True

    wi_fix = np.where(i_fix == True)[0]
    i_fix = i_fix == 1.0

    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
    )

    if config["verbose"]:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=config["min_sac_duration"]
            )
        )

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True] * config["nb_samples"])

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config["min_fix_duration"] * s_f))

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1] : s_int[0] + 1] = False

    if config["verbose"]:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config["min_fix_duration"]
            )
        )

    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config["min_fix_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    if config["verbose"]:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=config["status_threshold"]
            )
        )

    assert len(f_ints) == len(
        ctrds
    ), "Interval set and centroid set have different lengths"

    if config["verbose"]:
        print("\n...2MC Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Keep track of index that were effectively labeled
    i_lab = np.array([False] * config["nb_samples"])

    for f_int in f_ints:
        i_lab[f_int[0] : f_int[1] + 1] = True

    for s_int in s_ints:
        i_lab[s_int[0] : s_int[1] + 1] = True

    return dict(
        {
            "is_labeled": i_lab,
            "fixation_intervals": f_ints,
            "saccade_intervals": s_ints,
            "centroids": ctrds,
        }
    )
