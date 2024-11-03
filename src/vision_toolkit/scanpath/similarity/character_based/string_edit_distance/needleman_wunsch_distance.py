# -*- coding: utf-8 -*-

from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.scanpath.similarity.character_based.character_based_base import CharacterBased
from vision_toolkit.utils.binning import dist_mat


class NeedlemanWunschDistance(CharacterBased):
    def __init__(self, input, config, id_1="0", id_2="1"):
        super().__init__(input, config)

        c_b = config["scanpath_needleman_wunsch_distance_concordance_bonus"]
        g_c = config["scanpath_needleman_wunsch_distance_gap_cost"]

        norm_ = config["scanpath_needleman_wunsch_distance_normalization"]

        d_m, i_dict = dist_mat(self.x_nb_pixels, self.y_nb_pixels, self.elem_sizes)

        self.opt_align, dist_ = c_comparison.needleman_wunsch(
            self.s_1, self.s_2, g_c, c_b, i_dict, d_m
        )

        if norm_ == "max":
            self.dist_ = -dist_ / (max(self.n_1, self.n_2))

        else:
            self.dist_ = -dist_ / (min(self.n_1, self.n_2))

        if config["display_results"]:
            self.compute_visualization(id_1, id_2)
