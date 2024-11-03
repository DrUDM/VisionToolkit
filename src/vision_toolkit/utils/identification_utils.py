# -*- coding: utf-8 -*-

import itertools
from itertools import groupby

import numpy as np


def compute_aoi_sequence(seq_, dur_, config):
    """


    Parameters
    ----------
    seq_ : TYPE
        DESCRIPTION.
    dur_ : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    seq_ : TYPE
        DESCRIPTION.

    """

    n_s = len(seq_)
    aoi_temporal_binning = config["AoI_temporal_binning"]

    if aoi_temporal_binning == False:
        ## Convert to capital letters
        seq_ = [chr(seq_[i] + 65) for i in range(n_s)]
        seq_dur = [dur_[i] for i in range((n_s))]

    elif aoi_temporal_binning == True:
        ## Convert to capital letters with repetitions depending on fixation duration
        temp_bin = config["AoI_temporal_binning_length"]

        seq__ = []
        for i in range(n_s):
            [
                seq__.append(chr(seq_[i] + 65))
                for _ in range(int(np.ceil(dur_[i] / temp_bin)))
            ]
        seq_ = seq__

        seq_dur = []
        for i in range(n_s):
            if dur_[i] // temp_bin > 0:
                [seq_dur.append(temp_bin) for _ in range(int(dur_[i] // temp_bin))]
            if dur_[i] % temp_bin > 0:
                seq_dur.append(dur_[i] % temp_bin)

    elif aoi_temporal_binning == "collapse":
        ## Convert to capital letters removing elements that have consecutive duplicates
        seq_ = [chr(seq_[i] + 65) for i in range(n_s)]

        seq_dur = []
        dur__ = list(dur_)
        for key, _group in groupby(seq_):
            g_ = list(_group)
            l_ = sum(1 for x in g_)
            seq_dur.append(np.sum(np.array(dur__[:l_])))
            dur__ = dur__[l_:]

        seq_ = [key for key, _group in groupby(seq_)]

    else:
        raise ValueError(
            "'AoI_temporal_binning' must be set to True, or False, or 'collapse'"
        )

    assert len(seq_) == len(
        seq_dur
    ), "AoI sequences and duration sequences must have same length"

    return seq_, np.array(seq_dur)


def temporal_binning_AoI(seq_, dur_, config):
    n_s = len(seq_)

    assert (
        dur_ is not None
    ), "AoI_durations must be provided to perform temporal binning"

    temp_bin = config["AoI_temporal_binning_length"]

    seq__ = []
    for i in range(n_s):
        [seq__.append(seq_[i]) for _ in range(int(np.ceil(dur_[i] / temp_bin)))]
    seq_ = seq__

    seq_dur = []
    for i in range(n_s):
        if dur_[i] // temp_bin > 0:
            [seq_dur.append(temp_bin) for _ in range(int(dur_[i] // temp_bin))]
        if dur_[i] % temp_bin > 0:
            seq_dur.append(dur_[i] % temp_bin)

    return seq_, np.array(seq_dur)


def collapse_AoI(seq_, dur_):
    if dur_ is not None:
        seq_dur = []
        dur__ = list(dur_)
        for key, _group in groupby(seq_):
            g_ = list(_group)
            l_ = sum(1 for x in g_)
            seq_dur.append(np.sum(np.array(dur__[:l_])))
            dur__ = dur__[l_:]

    else:
        seq_dur = None

    seq_ = [key for key, _group in groupby(seq_)]

    return seq_, np.array(seq_dur)
