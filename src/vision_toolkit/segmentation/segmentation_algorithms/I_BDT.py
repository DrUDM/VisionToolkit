# -*- coding: utf-8 -*-

import copy
import time

import numpy as np
from scipy.stats import norm

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IBDT(data_set, config):
    if config["verbose"]:
        print("Processing BDT Identification...")
        start_time = time.time()

    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    d_t = int(np.ceil(config["IBDT_duration_threshold"] * s_f))

    fix_t = config["IBDT_fixation_threshold"]
    sac_t = config["IBDT_saccade_threshold"]
    pur_t = config["IBDT_pursuit_threshold"]

    fix_sd = config["IBDT_fixation_sd"]
    sac_sd = config["IBDT_saccade_sd"]

    a_s = data_set["absolute_speed"]
    print(n_s)
    print(a_s.shape)
    priors = dict({"fix": np.zeros(n_s), "sac": np.zeros(n_s), "pur": np.zeros(n_s)})

    likelihoods = dict({"fix": None, "sac": None, "pur": np.zeros(n_s)})

    posteriors = dict({"fix": None, "sac": None, "pur": None})

    for i in range(d_t):
        likelihoods["pur"][i] = np.sum(a_s[: i + 1] > pur_t) / (i + 1)

        if i == 0:
            priors["pur"][i] = likelihoods["pur"][0]

        else:
            priors["pur"][i] = np.mean(likelihoods["pur"][:i])

        priors["fix"][i] = priors["sac"][i] = (1 - priors["pur"][i]) / 2

    for i in range(d_t, n_s):
        likelihoods["pur"][i] = np.sum(a_s[i - d_t + 1 : i + 1] > pur_t) / (d_t)
        priors["pur"][i] = np.mean(likelihoods["pur"][i - d_t + 1 : i])

        priors["fix"][i] = priors["sac"][i] = (1 - priors["pur"][i]) / 2

    # Compute fixation likelihoods
    lk_f = copy.deepcopy(a_s)
    lk_f[lk_f < fix_t] = fix_t
    likelihoods["fix"] = norm.pdf(lk_f, loc=fix_t, scale=fix_sd)

    # Compute saccade likelihoods
    lk_s = copy.deepcopy(a_s)
    lk_s[lk_s > sac_t] = sac_t
    likelihoods["sac"] = norm.pdf(lk_s, loc=sac_t, scale=sac_sd)

    # Compute poseteriors
    for _ev in ["fix", "sac", "pur"]:
        posteriors[_ev] = priors[_ev] * likelihoods[_ev]

    a_m = np.argmax(
        np.concatenate(
            (
                posteriors["fix"].reshape(1, n_s),
                posteriors["sac"].reshape(1, n_s),
                posteriors["pur"].reshape(1, n_s),
            ),
            axis=0,
        ),
        axis=0,
    )

    if config["verbose"]:
        print("\n...BDT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return dict(
        {
            "is_saccade": a_m == 1,
            "saccade_intervals": interval_merging(
                np.where((a_m == 1) == True)[0], min_int_size=config["min_int_size"]
            ),
            "is_pursuit": a_m == 2,
            "pursuit_intervals": interval_merging(
                np.where((a_m == 2) == True)[0], min_int_size=config["min_int_size"]
            ),
            "is_fixation": a_m == 0,
            "fixation_intervals": interval_merging(
                np.where((a_m == 0) == True)[0], min_int_size=config["min_int_size"]
            ),
        }
    )
