# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.visualization.scanpath.similarity.character_based.character_based import plot_character_based


class GlobalAlignment:
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
        assert input[0].centers != None, "AoI centers must be specified"
        assert all(
            all(x == y)
            for x, y in zip(
                list(input[0].centers.values()), list(input[1].centers.values())
            )
        ), "AoI centers must be the same"

        self.centers = input[0].centers

        self.s_1, self.s_2 = input[0].sequence, input[1].sequence
        self.n_1 = len(self.s_1)
        self.n_2 = len(self.s_2)

        self.opt_align = None

    def compute_visualization(self, id_1, id_2):
        o_a = self.opt_align
        d_b = self.centers

        s_1 = self.s_1
        s_2 = self.s_2

        i_1 = 0
        i_2 = 0
        o_l = []

        for a_ in o_a:
            if "__" not in a_:
                o_l.append([d_b[s_1[i_1]], d_b[s_2[i_2]], [i_1, i_2]])

                i_1 += 1
                i_2 += 1

            elif a_[0] == "__":
                i_2 += 1

            elif a_[1] == "__":
                i_1 += 1

        o_l = np.array(o_l)

        s_1_b = np.array([d_b[s_1[i]] for i in range(len(s_1))]).T

        s_2_b = np.array([d_b[s_2[j]] for j in range(len(s_2))]).T

        plot_character_based(s_1_b, s_2_b, o_l, id_1, id_2)
