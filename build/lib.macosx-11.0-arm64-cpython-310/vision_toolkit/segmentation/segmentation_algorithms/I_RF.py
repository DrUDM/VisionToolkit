# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import (
    centroids_from_ints,
    interval_merging)
from vision_toolkit.utils.velocity_distance_factory import process_speed_components


def process_IRF(data_set, config):
    assert (
        config["distance_type"] == "euclidean"
    ), "'Distance type' must be set to 'euclidean"

    if config["verbose"]:
        print("Processing KF Identification...")
        start_time = time.time()

    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    d_t = 1 / s_f
    c_wn = config["IKF_chi2_window"]

    x_a = data_set["x_array"]
    y_a = data_set["y_array"]