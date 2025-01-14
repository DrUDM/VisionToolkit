# -*- coding: utf-8 -*-
import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.aoi.global_alignment.generalized_edit_distance import GeneralizedEditDistance
from vision_toolkit.aoi.global_alignment.levenshtein_distance import LevenshteinDistance
from vision_toolkit.aoi.global_alignment.needleman_wunsch_distance import NeedlemanWunschDistance


class AoIStringEditDistance:
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
            print("Processing AoI String Edit Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a AoIStringEditDistance instance or a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

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
                "levenshtein_distance": LevenshteinDistance,
                "generalized_edit_distance": GeneralizedEditDistance,
                "needleman_wunsch_distance": NeedlemanWunschDistance,
            }
        )

        if verbose:
            print("...AoI String Edit Distance done\n")

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

    def se_dist_mat(self, distance, config):
        """


        Parameters
        ----------
        distance : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        d_m : TYPE
            DESCRIPTION.

        """

        aoi_sequences = self.aoi_sequences
        n_sp = self.n_sp

        dist_method = self.dict_methods[distance]
        d_m = np.zeros((n_sp, n_sp))

        for j in range(1, n_sp):
            for i in range(j):
                e_a = dist_method(
                    [aoi_sequences[i], aoi_sequences[j]],
                    config,
                    id_1=str(i),
                    id_2=str(j),
                )
                d_m[i, j] = e_a.dist_

        d_m += d_m.T

        return d_m

    def AoI_levenshtein_distance(
        self,
        levenshtein_deletion_cost,
        levenshtein_insertion_cost,
        levenshtein_substitution_cost,
        levenshtein_normalization,
        display_results,
    ):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {
                "AoI_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                "AoI_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                "AoI_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
                "AoI_levenshtein_distance_normalization": levenshtein_normalization,
            }
        )

        d_m = self.se_dist_mat("levenshtein_distance", config)
        results = dict({"AoI_levenshtein_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "AoI_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                    "AoI_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                    "AoI_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
                    "AoI_levenshtein_distance_normalization": levenshtein_normalization,
                }
            )
        )
        return results

    def AoI_generalized_edit_distance(
        self,
        generalized_edit_deletion_cost,
        generalized_edit_insertion_cost,
        generalized_edit_normalization,
        display_results,
    ):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {
                "AoI_generalized_edit_distance_deletion_cost": generalized_edit_deletion_cost,
                "AoI_generalized_edit_distance_insertion_cost": generalized_edit_insertion_cost,
                "AoI_generalized_edit_distance_normalization": generalized_edit_normalization,
            }
        )

        d_m = self.se_dist_mat("generalized_edit_distance", config)
        results = dict({"AoI_generalized_edit_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "AoI_generalized_edit_distance_deletion_cost": generalized_edit_deletion_cost,
                    "AoI_generalized_edit_distance_insertion_cost": generalized_edit_insertion_cost,
                    "AoI_generalized_edit_distance_normalization": generalized_edit_normalization,
                }
            )
        )
        return results

    def AoI_needleman_wunsch_distance(
        self,
        needleman_wunsch_concordance_bonus,
        needleman_wunsch_gap_cost,
        needleman_wunsch_normalization,
        display_results,
    ):
        self.config.update({"display_results": display_results})
        config = copy.deepcopy(self.config)
        config.update(
            {
                "AoI_needleman_wunsch_distance_concordance_bonus": needleman_wunsch_concordance_bonus,
                "AoI_needleman_wunsch_distance_gap_cost": needleman_wunsch_gap_cost,
                "AoI_needleman_wunsch_distance_normalization": needleman_wunsch_normalization,
            }
        )

        d_m = self.se_dist_mat("needleman_wunsch_distance", config)
        results = dict({"AoI_needleman_wunsch_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "AoI_needleman_wunsch_distance_concordance_bonus": needleman_wunsch_concordance_bonus,
                    "AoI_needleman_wunsch_distance_gap_cost": needleman_wunsch_gap_cost,
                    "AoI_needleman_wunsch_distance_normalization": needleman_wunsch_normalization,
                }
            )
        )

        return results


def AoI_levenshtein_distance(input, **kwargs):
    levenshtein_deletion_cost = kwargs.get(
        "AoI_levenshtein_distance_deletion_cost", 1.0
    )
    levenshtein_insertion_cost = kwargs.get(
        "AoI_levenshtein_distance_insertion_cost", 1.0
    )
    levenshtein_substitution_cost = kwargs.get(
        "AoI_levenshtein_distance_substitution_cost", 1.0
    )
    levenshtein_normalization = kwargs.get(
        "AoI_levenshtein_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, AoIStringEditDistance):
        results = input.AoI_levenshtein_distance(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenshtein_normalization,
            display_results,
        )
    else:
        pm_distance = AoIStringEditDistance(input, **kwargs)
        results = pm_distance.AoI_levenshtein_distance(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenshtein_normalization,
            display_results,
        )

    return results


def AoI_generalized_edit_distance(input, **kwargs):
    generalized_edit_deletion_cost = kwargs.get(
        "AoI_generalized_edit_distance_deletion_cost", 0.3
    )
    generalized_edit_insertion_cost = kwargs.get(
        "AoI_generalized_edit_distance_insertion_cost", 0.3
    )
    generalized_edit_normalization = kwargs.get(
        "AoI_generalized_edit_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, AoIStringEditDistance):
        results = input.AoI_generalized_edit_distance(
            generalized_edit_deletion_cost,
            generalized_edit_insertion_cost,
            generalized_edit_normalization,
            display_results,
        )
    else:
        pm_distance = AoIStringEditDistance(input, **kwargs)
        results = pm_distance.AoI_generalized_edit_distance(
            generalized_edit_deletion_cost,
            generalized_edit_insertion_cost,
            generalized_edit_normalization,
            display_results,
        )

    return results


def AoI_needleman_wunsch_distance(input, **kwargs):
    needleman_wunsch_concordance_bonus = kwargs.get(
        "AoI_needleman_wunsch_distance_concordance_bonus", 0.1
    )
    needleman_wunsch_gap_cost = kwargs.get(
        "AoI_needleman_wunsch_distance_gap_cost", 0.25
    )
    needleman_wunsch_normalization = kwargs.get(
        "AoI_needleman_wunsch_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, AoIStringEditDistance):
        results = input.AoI_needleman_wunsch_distance(
            needleman_wunsch_concordance_bonus,
            needleman_wunsch_gap_cost,
            needleman_wunsch_normalization,
            display_results,
        )
    else:
        pm_distance = AoIStringEditDistance(input, **kwargs)
        results = pm_distance.AoI_needleman_wunsch_distance(
            needleman_wunsch_concordance_bonus,
            needleman_wunsch_gap_cost,
            needleman_wunsch_normalization,
            display_results,
        )

    return results
