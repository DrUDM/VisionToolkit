# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.utils.binning import dict_bin, spatial_bin_d, spatial_temp_bin_d
from vision_toolkit.visualization.scanpath.similarity.character_based.character_based import plot_character_based


class CharacterBased:
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

        self.x_nb_pixels = config["scanpath_spatial_binning_nb_pixels_x"]
        self.y_nb_pixels = config["scanpath_spatial_binning_nb_pixels_y"]

        x_size = config["size_plan_x"]
        y_size = config["size_plan_y"]

        temporal_binning = config["scanpath_temporal_binning"]

        ## Compute spayial and temporal binning if required
        if temporal_binning:
            temp_bin_l = config["scanpath_temporal_binning_length"]

            s_1 = input[0].values
            s_2 = input[1].values

            self.s_1, self.s_2, self.elem_sizes = spatial_temp_bin_d(
                s_1, s_2, temp_bin_l, self.x_nb_pixels, self.y_nb_pixels, x_size, y_size
            )
        ## Compute only spatial binning else
        else:
            s_1 = input[0].values[0:2, :]
            s_2 = input[1].values[0:2, :]

            self.s_1, self.s_2, self.elem_sizes = spatial_bin_d(
                s_1, s_2, self.x_nb_pixels, self.y_nb_pixels, x_size, y_size
            )

        self.n_1 = len(self.s_1)
        self.n_2 = len(self.s_2)

        self.opt_align = None

        ## Useful to plot results
        if config["display_results"]:
            self.d_b = dict_bin(self.x_nb_pixels, self.y_nb_pixels, self.elem_sizes)

    def compute_visualization(self, id_1, id_2):
        o_a = self.opt_align
        d_b = self.d_b

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
