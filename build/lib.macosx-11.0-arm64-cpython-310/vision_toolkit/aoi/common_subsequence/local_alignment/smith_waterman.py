# -*- coding: utf-8 -*-
import numpy as np

from vision_toolkit.aoi.common_subsequence.local_alignment.c_alignment_algorithms import c_alignment_algorithms as c_alignment
from vision_toolkit.utils.binning import aoi_dict_dist_mat


class SmithWaterman:
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
        assert input[0].centers != None, "AoI centers must be specified"
        assert input[0].centers == input[1].centers, "AoI centers must be the same"

        self.centers = input[0].centers
        self.config = config

        self.s_1, self.s_2 = input[0].sequence, input[1].sequence
        self.n_1, self.n_2 = len(self.s_1), len(self.s_2)

        d_m, i_dict = aoi_dict_dist_mat(self.centers, normalize=False)
        ## Compute -distance_matrix/similarity_matrix and a character to index dictionary
        s_m = self.compute_sw_dist_mat(d_m)

        del_cost_base = config["AoI_smith_waterman_base_deletion_cost"]
        del_cost = config["AoI_smith_waterman_iterative_deletion_cost"]
        similarity_weight = config["AoI_smith_waterman_similarity_weight"]

        ## Call C implementation of the smith waterman algorithm
        self.common_subsequence, self.opt_align, dist_ = c_alignment.smith_waterman(
            self.s_1, self.s_2, i_dict, s_m, del_cost_base, del_cost, similarity_weight
        )
        self.dist_ = dist_ / (
            self.config["AoI_smith_waterman_similarity_threshold"]
            * max(self.n_1, self.n_2)
        )

    def compute_sw_dist_mat(self, d_m):
        """


        Returns
        -------
        None.

        """

        ## Compute the maximal distance between two points of the visual field if its dimension is known
        if "size_plan_x" and "size_plan_y" in self.config.keys():
            d_m_max = np.linalg.norm(
                np.array([self.config["size_plan_x"], self.config["size_plan_y"]])
            )
        ## Else normalize by the maximal distance between two AoI centers
        else:
            d_m_max = np.max(d_m)

        s_t = self.config["AoI_smith_waterman_similarity_threshold"]
        if s_t == None:
            s_t = 0.2 * d_m_max
            self.config.update({"AoI_smith_waterman_similarity_threshold": 0.2})

        s_m = (-d_m + s_t) / d_m_max

        return s_m
