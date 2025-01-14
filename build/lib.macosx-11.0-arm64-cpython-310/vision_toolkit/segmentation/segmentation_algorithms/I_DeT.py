# -*- coding: utf-8 -*-


import time

import vision_toolkit.segmentation.segmentation_algorithms.c_I_DeT.c_I_DeT as c_I_DeT


def process_IDeT(data_set, config):
    if config["verbose"]:
        print("Processing DeT Identification...")
        start_time = time.time()

    out = c_I_DeT.process_IDeT(data_set, config)

    if config["verbose"]:
        print("\n...DeT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return out
