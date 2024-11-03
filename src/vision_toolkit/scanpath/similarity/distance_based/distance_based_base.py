# -*- coding: utf-8 -*-


import numpy as np
from scipy.spatial.distance import cdist


class DistanceBased:
    def __init__(self, input, dist_to=False):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        dist_to : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        self.s_1 = input[0].values
        self.s_2 = input[1].values

        self.n_1 = len(self.s_1[0])
        self.n_2 = len(self.s_2[0])

        self.opt_links = None

    def compute_mapping(self, s_1, s_2, n_1, n_2):
        """


        Parameters
        ----------
        s_1 : TYPE
            DESCRIPTION.
        s_2 : TYPE
            DESCRIPTION.
        n_1 : TYPE
            DESCRIPTION.
        n_2 : TYPE
            DESCRIPTION.

        Returns
        -------
        d_t : TYPE
            DESCRIPTION.
        opt_links : TYPE
            DESCRIPTION.

        """

        d_t = np.zeros(n_2)
        opt_links = np.zeros((n_2, 3, 2))

        for j in range(n_2):
            w_v = cdist(s_2[:, j].reshape((1, 2)), s_1.T, metric="euclidean")

            b_i = np.argmin(w_v[0])
            d_t[j] = w_v[0, b_i]

            opt_links[j] = np.array([s_1[:, b_i], s_2[:, j], [b_i, j]])

        return d_t, opt_links
