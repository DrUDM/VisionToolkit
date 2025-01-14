# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np

from vision_toolkit.aoi.aoi_base import AoISequence
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class NGram:
    def __init__(self, input, n_w):
        """


        Parameters
        ----------
        sequence : list
            Input sequence is a list of strings.
        n_w : int
            Defines the length of each pattern.

        Returns
        -------
        None.

        """

        self.sequence = np.array(input)
        self.n_w = n_w
        self.l_ = len(input)

        ## Compute histogram
        self.table = self.get_frequency()

    def get_frequency(self):
        """
        Given a list of symbols, store the
        frequency of each subsequence

        Returns
        -------
        freq : TYPE
            DESCRIPTION.

        """

        n_w = self.n_w
        l_ = self.l_

        ## Counter object
        freq = Counter()

        ## Indexer matrix to extract all subsequences of
        ## length n with a stride of 1
        window_indexer = np.array(
            np.expand_dims(np.arange(n_w), 0)
            + np.expand_dims(np.arange(l_ - n_w + 1), 0).T
        )

        ## Iterate over all subsequences
        for subsequence in self.sequence[window_indexer]:
            name = "".join(s + "," for s in subsequence)
            freq[name] += 1 / (l_ - n_w + 1)

        return freq


def AoI_NGram(input, **kwargs):
    verbose = kwargs.get("verbose", True)

    if verbose:
        print("Processing NGram Analysis...\n")

    if isinstance(input, str):
        aoi_sequence = AoISequence.generate(input, **kwargs)

    elif isinstance(input, BinarySegmentation):
        aoi_sequence = AoISequence.generate(input, **kwargs)

    elif isinstance(input, AoISequence):
        aoi_sequence = input

    elif isinstance(input, Scanpath):
        aoi_sequence = AoISequence.generate(input, **kwargs)

    n_w = kwargs.get("AoI_NGram_length", 3)
    n_g = NGram(aoi_sequence.sequence, n_w)
    results = dict({"AoI_NGram": n_g.table})

    if verbose:
        print("...NGram Analysis done\n")

    return results
