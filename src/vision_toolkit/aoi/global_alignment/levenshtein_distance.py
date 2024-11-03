# -*- coding: utf-8 -*-


from vision_toolkit.aoi.global_alignment.global_alignment_base import GlobalAlignment
from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison


class LevenshteinDistance(GlobalAlignment):
    def __init__(self, input, config, id_1="0", id_2="1"):
        super().__init__(input, config)

        c_del = config["AoI_levenshtein_distance_deletion_cost"]
        c_ins = config["AoI_levenshtein_distance_insertion_cost"]
        c_sub = config["AoI_levenshtein_distance_substitution_cost"]

        norm_ = config["AoI_levenshtein_distance_normalization"]

        self.opt_align, dist_ = c_comparison.levenshtein(
            self.s_1, self.s_2, c_del, c_ins, c_sub
        )

        if norm_ == "max":
            self.dist_ = dist_ / max(self.n_1, self.n_2)

        else:
            self.dist_ = dist_ / min(self.n_1, self.n_2)

        if config["display_results"]:
            self.compute_visualization(id_1, id_2)
