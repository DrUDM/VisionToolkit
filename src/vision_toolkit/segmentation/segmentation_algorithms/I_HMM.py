# -*- coding: utf-8 -*-


import time

import vision_toolkit.segmentation.segmentation_algorithms.c_I_HMM.c_I_HMM as c_I_HMM


def process_IHMM(data_set, config):
    if config["verbose"]:
        print("Processing HMM Identification...")
        start_time = time.time()

    out = c_I_HMM.process_IHMM(data_set, config)

    if config["verbose"]:
        print("\n...HMM Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return out
