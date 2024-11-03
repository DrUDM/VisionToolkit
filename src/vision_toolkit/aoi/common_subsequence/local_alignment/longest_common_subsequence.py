# -*- coding: utf-8 -*-

from vision_toolkit.aoi.common_subsequence.local_alignment.c_alignment_algorithms import c_alignment_algorithms as c_alignment


class LongestCommonSubsequence:
    def __init__(self, input, config, id_1="0", id_2="1"):
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

        self.s_1, self.s_2 = input[0].sequence, input[1].sequence
        self.n_1, self.n_2 = len(self.s_1), len(self.s_2)

        (
            common_subsequence,
            self.opt_align,
            dist_,
        ) = c_alignment.longest_common_subsequence(self.s_1, self.s_2)
        self.common_subsequence = [
            common_subsequence[i][0] for i in range(len(common_subsequence))
        ]
        norm_ = config["AoI_longest_common_subsequence_normalization"]
        if norm_ == "max":
            self.dist_ = dist_ / max(self.n_1, self.n_2)
        else:
            self.dist_ = dist_ / min(self.n_1, self.n_2)
