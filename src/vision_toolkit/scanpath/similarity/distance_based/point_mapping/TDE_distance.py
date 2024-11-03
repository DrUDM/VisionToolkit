# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased


class TDEDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input, dist_to=False)

        self.method = config["TDE_distance_method"]
        k = config["TDE_distance_subsequence_length"]
        scaling = config["TDE_distance_scaling"]

        if scaling:
            l = max(
                np.max(self.s_1[0]),
                np.max(self.s_1[1]),
                np.max(self.s_2[0]),
                np.max(self.s_2[1]),
            )

            s_1_n = self.s_1[0:2] / l
            s_2_n = self.s_2[0:2] / l

            n = min(self.n_1, self.n_2)
            tde = np.zeros(n)

            for _k in range(1, n + 1):
                u_1_cal = np.array(
                    [list(s_1_n[:, i : i + _k]) for i in range(self.n_1 - _k + 1)]
                )
                u_2_cal = np.array(
                    [list(s_2_n[:, i : i + _k]) for i in range(self.n_2 - _k + 1)]
                )
                tde[_k - 1] = self.TDE(_k, u_1_cal, u_2_cal)

            self.dist_ = np.exp(-np.sum(tde) / n)

        else:
            u_1_cal = np.array(
                [list(self.s_1[0:2, i : i + k]) for i in range(self.n_1 - k + 1)]
            )
            u_2_cal = np.array(
                [list(self.s_2[0:2, i : i + k]) for i in range(self.n_2 - k + 1)]
            )

            self.dist_ = self.TDE(k, u_1_cal, u_2_cal)

    def TDE(self, k, u_1_cal, u_2_cal):
        d = np.zeros(self.n_1 - k + 1)

        n_1 = self.n_1
        n_2 = self.n_2

        for i in range(n_1 - k + 1):
            w_v = [np.linalg.norm(u_1_cal[i] - u_2_cal[j]) for j in range(n_2 - k + 1)]

            d[i] = min(w_v)

        if self.method == "mean_minimal":
            return np.sum(d) / self.n_1

        elif self.method == "hausdorff":
            return np.max(d)
