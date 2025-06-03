# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IVVT(data_set, config):
    """
    Adapted from Komogortsev & Karpov (2013).
    Modified I-VT algorithm, with a supplementary
    threshold to distinguish pursuits from fixations.
        - T_s = saccade velocity threshold.
        - T_p = saccade velocity threshold.
    """

    if config["verbose"]:
        print("Processing VVT Identification...")
        start_time = time.time()

    # Eye movement parameters
    a_sp = data_set["absolute_speed"]

    # Algorithm parameters
    T_s = config["IVVT_saccade_threshold"]
    T_p = config["IVVT_pursuit_threshold"]

    # Saccades are found like in the I-VT algorithm
    i_sac = np.where(a_sp > T_s, 1, 0)

    # An additional threshold is used for pursuits
    i_purs = np.where((a_sp > T_p) & (a_sp <= T_s), 1, 0)
    
    # The remaining points are fixations
    i_fix = np.where(a_sp <= T_p, 1, 0)
    print(a_sp)
    if config["verbose"]:
        print("\n...VVT Identification done\n")
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
