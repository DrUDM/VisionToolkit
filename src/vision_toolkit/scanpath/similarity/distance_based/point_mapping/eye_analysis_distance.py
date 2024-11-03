# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
from vision_toolkit.visualization.scanpath.similarity.distance_based.point_mapping import plot_mannan_eye_analysis


class EyeAnalysisDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input, dist_to=True)

        d_1, ol_1 = self.compute_mapping(
            self.s_1[0:2], self.s_2[0:2], self.n_1, self.n_2
        )

        d_2, ol_2 = self.compute_mapping(
            self.s_2[0:2], self.s_1[0:2], self.n_2, self.n_1
        )

        self.dist_ = self.eye_analysis(d_1, d_2)

        if config["display_results"]:
            plot_mannan_eye_analysis(self.s_1, self.s_2, ol_1, ol_2, id_1, id_2)

    def eye_analysis(self, d_1, d_2):
        denom = max(self.n_1, self.n_2)
        num = np.sum(d_1) + np.sum(d_2)

        return num / denom
