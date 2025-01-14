# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
from vision_toolkit.visualization.scanpath.similarity.distance_based.point_mapping import plot_mannan_eye_analysis


class MannanDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input, dist_to=True)

        nb_r = config["mannan_distance_nb_random_scanpaths"]

        ## Find max and min scanpath (x,y) values to generate random vectors
        max_ = np.max(
            np.array([np.max(self.s_1[0:2], axis=1), np.max(self.s_2[0:2], axis=1)]),
            axis=0,
        )

        min_ = np.min(
            np.array([np.min(self.s_1[0:2], axis=1), np.min(self.s_2[0:2], axis=1)]),
            axis=0,
        )

        ## Keep visual field dimensions
        self.p_x = max_[0] - min_[0]
        self.p_y = max_[1] - min_[1]

        self.ol_1 = None
        self.ol_2 = None

        ## Compute Mannan distance for scanpaths seq_1 and seq_2
        d_u = self.mannan(self.s_1[0:2], self.s_2[0:2], keep=True)

        ## Compute Mannan distance for random scanpath vector
        ## of respective sizes n_1 and n_2
        d_r = np.zeros(nb_r)

        for i in range(nb_r):
            d_r[i] = self.mannan(
                np.random.uniform(low=min_, high=max_, size=(self.n_1, 2)).T,
                np.random.uniform(low=min_, high=max_, size=(self.n_2, 2)).T,
            )

        ## Compute the Mannan Index of Similarity
        self.dist_ = 100 * (d_u / np.mean(d_r))

        if config["display_results"]:
            plot_mannan_eye_analysis(
                self.s_1, self.s_2, self.ol_1, self.ol_2, id_1, id_2
            )

    def mannan(self, s_1, s_2, keep=False):
        n_1 = self.n_1
        n_2 = self.n_2

        d_1, ol_1 = self.compute_mapping(s_1, s_2, n_1, n_2)
        d_2, ol_2 = self.compute_mapping(s_2, s_1, n_2, n_1)

        if keep:
            self.ol_1 = ol_1
            self.ol_2 = ol_2

        denom = np.sqrt(2 * n_1 * n_2 * (self.p_x**2 + self.p_y**2))
        num = np.sqrt(n_1 * np.sum(d_1**2) + n_2 * np.sum(d_2**2))

        return num / denom
