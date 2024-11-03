# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
from vision_toolkit.visualization.scanpath.similarity.distance_based.elastic import plot_DTW_frechet


class DTWDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input, dist_to=False)

        c_m = cdist(self.s_1[0:2].T, self.s_2[0:2].T, metric="euclidean")

        self.opt_links, self.dist_ = c_comparison.DTW(self.s_1[0:2], self.s_2[0:2], c_m)

        if config["display_results"]:
            plot_DTW_frechet(self.s_1[0:2], self.s_2[0:2], self.opt_links, id_1, id_2)

    ## Not used -> C implementation
    def compute_dtw(self):
        s_1 = self.s_1[0:2]
        s_2 = self.s_2[0:2]

        n_1 = self.n_1
        n_2 = self.n_2

        d_m = np.zeros((n_1 + 1, n_2 + 1))
        d_m[1:, 0] = np.inf
        d_m[0, 1:] = np.inf

        c_m = cdist(s_1.T, s_2.T, metric="euclidean")

        b_dict = dict()

        for i in range(1, n_1 + 1):
            for j in range(1, n_2 + 1):
                c = c_m[i - 1, j - 1]

                w = np.array(
                    [c + d_m[i - 1, j], c + d_m[i, j - 1], c + d_m[i - 1, j - 1]]
                )

                b_i = np.argmin(w)

                if b_i == 0:
                    b_dict.update({(i, j): (i - 1, j)})

                if b_i == 1:
                    b_dict.update({(i, j): (i, j - 1)})

                if b_i == 2:
                    b_dict.update({(i, j): (i - 1, j - 1)})

                d_m[i, j] = w[b_i]

        def bactrack_links(d_mat, b_dict, o_l, i, j):
            (i_n, j_n) = b_dict[(i, j)]

            o_l.insert(0, [s_1[:, i_n - 1], s_2[:, j_n - 1], [i_n - 1, j_n - 1]])

            if i_n == 1 and j_n == 1:
                o_l.insert(0, [s_1[:, 0], s_2[:, 0], [0, 0]])

                return o_l

            return bactrack_links(d_mat, b_dict, o_l, i_n, j_n)

        if self.display:
            o_l = list()
            o_l.insert(0, [s_1[:, n_1 - 1], s_2[:, n_2 - 1], [n_1 - 1, n_2 - 1]])

            opt_links = bactrack_links(d_m, b_dict, o_l=o_l, i=n_1, j=n_2)

            self.opt_links = np.array(opt_links)

        return d_m[n_1, n_2]
