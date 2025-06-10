# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter

import numpy as np


class RecurrenceBase:
    def __init__(self, input):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if isinstance(input, list):
            self.s_1, self.s_2 = input[0], input[1]

            if isinstance(input[0], np.ndarray):
                self.n_1, self.n_2 = self.s_1.shape[1], self.s_2.shape[1]

            else:
                self.n_1, self.n_2 = len(self.s_1), len(self.s_2)

        elif isinstance(input, np.ndarray):
            self.s_ = input
            self.n = input.shape[1]

    def find_lines(self, r_m, m_l, direction):
        l_set = []
        ## get indices where value is not zero
        t_a = np.argwhere(r_m != 0)

        t_a = t_a.T
        u = np.unique(t_a[1])

        for c in u:
            l_ = list(np.where(t_a[1] == c)[0])
            o_c = t_a[0][l_]

            if len(o_c) >= m_l:
                for k, g in groupby(enumerate(list(o_c)), lambda ix: ix[0] - ix[1]):
                    c_i = list(map(itemgetter(1), g))
                    l = len(c_i)

                    if l >= m_l:
                        ## Add vertical line as an array of (i,j) matrix coordinates
                        if direction == "vertical":
                            l_set.append(np.array([c_i, [c] * l]).T)

                        ## Add horizontal line as an array of (i,j) matrix coordinates
                        if direction == "horizontal":
                            l_set.append(np.array([[c] * l, c_i]).T)

        return l_set

    def find_diags(self, r_m, m_l, full=False):
        d_set = []
        ## get indices where value is not zero
        t_a = np.argwhere(r_m != 0)

        ## Iteratively shift x values to simplify the problem as a vertical line finding problem
        t_a[:, 1] = t_a[:, 1] - t_a[:, 0]

        t_a = t_a.T
        u = np.unique(t_a[1])

        for c in u:
            l_ = list(np.where(t_a[1] == c)[0])
            o_c = t_a[0][l_]

            if len(o_c) >= m_l:
                for k, g in groupby(enumerate(list(o_c)), lambda ix: ix[0] - ix[1]):
                    c_i = list(map(itemgetter(1), g))
                    l = len(c_i)

                    if l >= m_l:
                        if full:
                            ## Add diagonal line as an array of (i,j) matrix coordinates
                            ## after unshifting x-values and extension for the full recurrence matrix case
                            d_set.append(
                                np.array(
                                    [
                                        c_i,
                                        [
                                            c + c_i[i] - self.n_2
                                            for i in range(len(c_i))
                                        ],
                                    ]
                                ).T
                            )

                        else:
                            ## Add diagonal line as an array of (i,j) matrix coordinates
                            ## after unshifting x-values
                            d_set.append(
                                np.array([c_i, [c + c_i[i] for i in range(len(c_i))]]).T
                            ) 
        return d_set
