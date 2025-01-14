# -*- coding: utf-8 -*-


import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.aoi.common_subsequence.local_alignment.longest_common_subsequence import LongestCommonSubsequence
from vision_toolkit.aoi.global_alignment.levenshtein_distance import LevenshteinDistance
from vision_toolkit.aoi.global_alignment.string_edit_distance import AoI_levenshtein_distance


class eMine:
    def __init__(self, input, config):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = config["verbose"]

        self.aoi_sequences = input
        self.config = config

        self.n_sp = len(input)

        d_m = AoI_levenshtein_distance(
            self.aoi_sequences, display_results=False, verbose=False
        )["AoI_levenshtein_distance_matrix"]
        d_m += np.diag(np.ones(self.n_sp) * (np.max(d_m) + 1))
        self.d_m = d_m

        self.common_subsequence = self.process_emine()

    def process_emine(self):
        """


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        d_m = self.d_m
        m_ = np.max(d_m)
        aoi_sequences = self.aoi_sequences

        centers = aoi_sequences[0].centers
        nb_aoi = aoi_sequences[0].nb_aoi

        config = copy.deepcopy(self.config)
        config.update({"display_results": False})

        while len(aoi_sequences) > 1:
            ## Find the two most similar AoI sequences in aoi_sequences
            i, j = np.unravel_index(np.argmin(d_m), d_m.shape)

            s_1 = aoi_sequences[i]
            s_2 = aoi_sequences[j]

            ## Compute the longest common subsequence from these two sequences
            lcs = LongestCommonSubsequence([s_1, s_2], config).common_subsequence

            ## Generate a new AoISequence object from the lcs
            dict_ = dict(
                {
                    "sequence": lcs,
                    "durations": None,
                    "centers": centers,
                    "nb_aoi": nb_aoi,
                    "config": config,
                }
            )
            n_aoi_seq = AoISequence(dict_)

            # Remove these two AoI sequences from the set of AoI sequences
            aoi_sequences.remove(s_1)
            aoi_sequences.remove(s_2)

            ##...and from the dissimilarity matrix d_m
            d_m = np.delete(d_m, (i, j), 0)
            d_m = np.delete(d_m, (i, j), 1)

            ## Compute the new dissimilarity matrix d_m
            k = len(aoi_sequences)
            n_d_m = np.zeros((k + 1, k + 1))

            for i in range(k):
                n_d_m[i, k] = LevenshteinDistance(
                    [aoi_sequences[i], n_aoi_seq], config
                ).dist_

            n_d_m = n_d_m + n_d_m.T
            n_d_m[k, k] = m_

            n_d_m[:-1, :-1] = d_m
            d_m = copy.deepcopy(n_d_m)

            ## Add the longest common subsequence to the set of AoI sequences
            aoi_sequences.append(n_aoi_seq)

        return aoi_sequences[0].sequence
