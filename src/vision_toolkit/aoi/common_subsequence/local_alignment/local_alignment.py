# -*- coding: utf-8 -*-

import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import (AoI_sequences, AoIMultipleSequences,
                                 AoISequence)
from vision_toolkit.aoi.common_subsequence.local_alignment.e_mine import eMine
from vision_toolkit.aoi.common_subsequence.local_alignment.longest_common_subsequence import LongestCommonSubsequence
from vision_toolkit.aoi.common_subsequence.local_alignment.smith_waterman import SmithWaterman


class LocalAlignment:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Local Alignment...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a LocalAlignment instance or a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.config = aoi_sequences[0].config
        self.config.update({"verbose": verbose})

        if "nb_samples" in self.config.keys():
            del self.config["nb_samples"]

        self.aoi_sequences = aoi_sequences
        self.n_sp = len(aoi_sequences)

        self.dict_methods = dict(
            {
                "longest_common_subsequence": LongestCommonSubsequence,
                "smith_waterman": SmithWaterman,
            }
        )
        if verbose:
            print("...Local Alignment done\n")

    def verbose(self, add_=None):
        if self.config["verbose"]:
            print("\n --- Config used: ---\n")

            for it in self.config.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.config[it]
                    )
                )

            if add_ is not None:
                for it in add_.keys():
                    print(
                        "# {it}:{esp}{val}".format(
                            it=it, esp=" " * (50 - len(it)), val=add_[it]
                        )
                    )
            print("\n")

    def la_dist_mat(self, distance, config):
        aoi_sequences = self.aoi_sequences
        n_sp = self.n_sp

        dist_method = self.dict_methods[distance]
        d_m = np.zeros((n_sp, n_sp))
        p_m = dict()

        for j in range(1, n_sp):
            for i in range(j):
                e_a = dist_method(
                    [aoi_sequences[i], aoi_sequences[j]],
                    config,
                    id_1=str(i),
                    id_2=str(j),
                )
                d_m[i, j] = e_a.dist_
                p_m.update({(i, j): e_a.common_subsequence})
        d_m += d_m.T

        return d_m, p_m

    def AoI_longest_common_subsequence(self, lcs_normalization, display_results):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {"AoI_longest_common_subsequence_normalization": lcs_normalization}
        )

        d_m, p_m = self.la_dist_mat("longest_common_subsequence", config)
        results = dict(
            {
                "AoI_longest_common_subsequence_matrix": d_m,
                "AoI_longest_common_subsequence_pairs": p_m,
            }
        )

        self.verbose(
            dict({"AoI_longest_common_subsequence_normalization": lcs_normalization})
        )

        return results

    def AoI_smith_waterman(
        self,
        sw_base_deletion_cost,
        sw_iterative_deletion_cost,
        sw_similarity_weight,
        sw_similarity_threshold,
        display_results,
    ):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {
                "AoI_smith_waterman_base_deletion_cost": sw_base_deletion_cost,
                "AoI_smith_waterman_iterative_deletion_cost": sw_iterative_deletion_cost,
                "AoI_smith_waterman_similarity_weight": sw_similarity_weight,
                "AoI_smith_waterman_similarity_threshold": sw_similarity_threshold,
            }
        )

        d_m, p_m = self.la_dist_mat("smith_waterman", config)
        results = dict(
            {"AoI_smith_waterman_matrix": d_m, "AoI_smith_waterman_pairs": p_m}
        )

        self.verbose(
            dict(
                {
                    "AoI_smith_waterman_base_deletion_cost": sw_base_deletion_cost,
                    "AoI_smith_waterman_iterative_deletion_cost": sw_iterative_deletion_cost,
                    "AoI_smith_waterman_similarity_weight": sw_similarity_weight,
                    "AoI_smith_waterman_similarity_threshold": sw_similarity_threshold,
                }
            )
        )

        return results

    def AoI_eMine(
        self,
        levenshtein_deletion_cost,
        levenshtein_insertion_cost,
        levenshtein_substitution_cost,
        levenstein_normalization,
        lcs_normalization,
        display_results,
    ):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {
                "AoI_longest_common_subsequence_normalization": lcs_normalization,
                "AoI_levenshtein_distance_normalization": levenstein_normalization,
                "AoI_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                "AoI_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                "AoI_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
            }
        )

        em = eMine(self.aoi_sequences, config)
        results = dict({"AoI_eMine_common_subsequence": em.common_subsequence})

        self.verbose(
            dict(
                {
                    "AoI_longest_common_subsequence_normalization": lcs_normalization,
                    "AoI_levenshtein_distance_normalization": levenstein_normalization,
                    "AoI_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                    "AoI_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                    "AoI_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
                }
            )
        )

        return results


def AoI_longest_common_subsequence(input, **kwargs):
    lcs_normalization = kwargs.get(
        "AoI_longest_common_subsequence_normalization", "max"
    )
    display_results = kwargs.get("display_results", True)

    if isinstance(input, LocalAlignment):
        results = input.AoI_longest_common_subsequence(
            lcs_normalization, display_results
        )
    else:
        la_distance = LocalAlignment(input, **kwargs)
        results = la_distance.AoI_longest_common_subsequence(
            lcs_normalization, display_results
        )
    return results


def AoI_smith_waterman(input, **kwargs):
    sw_base_deletion_cost = kwargs.get("AoI_smith_waterman_base_deletion_cost", 0.1)
    sw_iterative_deletion_cost = kwargs.get(
        "AoI_smith_waterman_iterative_deletion_cost", 0.05
    )
    sw_similarity_weight = kwargs.get("AoI_smith_waterman_similarity_weight", 1.0)
    sw_similarity_threshold = kwargs.get(
        "AoI_smith_waterman_similarity_threshold", None
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, LocalAlignment):
        results = input.AoI_smith_waterman(
            sw_base_deletion_cost,
            sw_iterative_deletion_cost,
            sw_similarity_weight,
            sw_similarity_threshold,
            display_results,
        )
    else:
        la_distance = LocalAlignment(input, **kwargs)
        results = la_distance.AoI_smith_waterman(
            sw_base_deletion_cost,
            sw_iterative_deletion_cost,
            sw_similarity_weight,
            sw_similarity_threshold,
            display_results,
        )
    return results


def AoI_eMine(input, **kwargs):
    lcs_normalization = kwargs.get(
        "AoI_longest_common_subsequence_normalization", "max"
    )
    levenstein_normalization = kwargs.get(
        "AoI_levenshtein_distance_normalization", "max"
    )
    levenshtein_deletion_cost = kwargs.get(
        "AoI_levenshtein_distance_deletion_cost", 1.0
    )
    levenshtein_insertion_cost = kwargs.get(
        "AoI_levenshtein_distance_insertion_cost", 1.0
    )
    levenshtein_substitution_cost = kwargs.get(
        "AoI_levenshtein_distance_substitution_cost", 1.0
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, LocalAlignment):
        results = input.AoI_eMine(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenstein_normalization,
            lcs_normalization,
            display_results,
        )
    else:
        la = LocalAlignment(input, **kwargs)
        results = la.AoI_eMine(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenstein_normalization,
            lcs_normalization,
            display_results,
        )

    return results
