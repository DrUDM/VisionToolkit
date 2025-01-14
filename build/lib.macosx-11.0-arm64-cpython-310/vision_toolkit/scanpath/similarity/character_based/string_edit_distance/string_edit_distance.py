# -*- coding: utf-8 -*-


import copy

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.similarity.character_based.string_edit_distance.generalized_edit_distance import GeneralizedEditDistance
from vision_toolkit.scanpath.similarity.character_based.string_edit_distance.levenshtein_distance import LevenshteinDistance
from vision_toolkit.scanpath.similarity.character_based.string_edit_distance.needleman_wunsch_distance import NeedlemanWunschDistance
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class ScanpathStringEditDistance:
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
            print("Processing String Edit Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a StringEditDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        else:
            raise ValueError(
                "Input must be a StringEditDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
            )

        self.config = scanpaths[0].config
        self.config.update({"verbose": verbose})

        ## Keep max x and y sizes for gaze visual field
        x_size_max = np.max(
            np.array([scanpath.config["size_plan_x"] for scanpath in scanpaths])
        )
        y_size_max = np.max(
            np.array([scanpath.config["size_plan_y"] for scanpath in scanpaths])
        )

        self.config.update({"size_plan_x": x_size_max, "size_plan_y": y_size_max})

        ## Update parameters for temporal and spatial binning
        self.config.update(
            {
                "scanpath_spatial_binning_nb_pixels_x": kwargs.get(
                    "scanpath_spatial_binning_nb_pixels_x", 10
                ),
                "scanpath_spatial_binning_nb_pixels_y": kwargs.get(
                    "scanpath_spatial_binning_nb_pixels_y", 10
                ),
                "scanpath_temporal_binning": kwargs.get(
                    "scanpath_temporal_binning", False
                ),
            }
        )

        if self.config["scanpath_temporal_binning"]:
            self.config.update(
                {
                    "scanpath_temporal_binning_length": kwargs.get(
                        "scanpath_temporal_binning_length", 0.250
                    )
                }
            )

        if "nb_samples" in self.config.keys():
            del self.config["nb_samples"]

        self.scanpaths = scanpaths
        self.n_sp = len(scanpaths)

        self.dict_methods = dict(
            {
                "levenshtein_distance": LevenshteinDistance,
                "generalized_edit_distance": GeneralizedEditDistance,
                "needleman_wunsch_distance": NeedlemanWunschDistance,
            }
        )
        if verbose:
            print("...String Edit Distance done\n")

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

        scanpaths = self.scanpaths
        n_sp = self.n_sp

        dist_method = self.dict_methods[distance]
        d_m = np.zeros((n_sp, n_sp))

        for j in range(1, n_sp):
            for i in range(j):
                e_a = dist_method(
                    [scanpaths[i], scanpaths[j]], config, id_1=str(i), id_2=str(j)
                )
                d_m[i, j] = e_a.dist_

        d_m += d_m.T

        return d_m

    def scanpath_levenshtein_distance(
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
                "scanpath_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                "scanpath_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                "scanpath_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
                "scanpath_levenshtein_distance_normalization": levenshtein_normalization,
            }
        )

        d_m = self.se_dist_mat("levenshtein_distance", config)
        results = dict({"scanpath_levenshtein_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "scanpath_levenshtein_distance_deletion_cost": levenshtein_deletion_cost,
                    "scanpath_levenshtein_distance_insertion_cost": levenshtein_insertion_cost,
                    "scanpath_levenshtein_distance_substitution_cost": levenshtein_substitution_cost,
                    "scanpath_levenshtein_distance_normalization": levenshtein_normalization,
                }
            )
        )
        return results

    def scanpath_generalized_edit_distance(
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
                "scanpath_generalized_edit_distance_deletion_cost": generalized_edit_deletion_cost,
                "scanpath_generalized_edit_distance_insertion_cost": generalized_edit_insertion_cost,
                "scanpath_generalized_edit_distance_normalization": generalized_edit_normalization,
            }
        )

        d_m = self.se_dist_mat("generalized_edit_distance", config)
        results = dict({"scanpath_generalized_edit_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "scanpath_generalized_edit_distance_deletion_cost": generalized_edit_deletion_cost,
                    "scanpath_generalized_edit_distance_insertion_cost": generalized_edit_insertion_cost,
                    "scanpath_generalized_edit_distance_normalization": generalized_edit_normalization,
                }
            )
        )
        return results

    def scanpath_needleman_wunsch_distance(
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
                "scanpath_needleman_wunsch_distance_concordance_bonus": needleman_wunsch_concordance_bonus,
                "scanpath_needleman_wunsch_distance_gap_cost": needleman_wunsch_gap_cost,
                "scanpath_needleman_wunsch_distance_normalization": needleman_wunsch_normalization,
            }
        )

        d_m = self.se_dist_mat("needleman_wunsch_distance", config)
        results = dict({"scanpath_needleman_wunsch_distance_matrix": d_m})

        self.verbose(
            dict(
                {
                    "scanpath_needleman_wunsch_distance_concordance_bonus": needleman_wunsch_concordance_bonus,
                    "scanpath_needleman_wunsch_distance_gap_cost": needleman_wunsch_gap_cost,
                    "scanpath_needleman_wunsch_distance_normalization": needleman_wunsch_normalization,
                }
            )
        )
        return results


def scanpath_levenshtein_distance(input, **kwargs):
    levenshtein_deletion_cost = kwargs.get(
        "scanpath_levenshtein_distance_deletion_cost", 1.0
    )
    levenshtein_insertion_cost = kwargs.get(
        "scanpath_levenshtein_distance_insertion_cost", 1.0
    )
    levenshtein_substitution_cost = kwargs.get(
        "scanpath_levenshtein_distance_substitution_cost", 1.0
    )
    levenshtein_normalization = kwargs.get(
        "scanpath_levenshtein_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, ScanpathStringEditDistance):
        results = input.scanpath_levenshtein_distance(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenshtein_normalization,
            display_results,
        )
    else:
        pm_distance = ScanpathStringEditDistance(input, **kwargs)
        results = pm_distance.scanpath_levenshtein_distance(
            levenshtein_deletion_cost,
            levenshtein_insertion_cost,
            levenshtein_substitution_cost,
            levenshtein_normalization,
            display_results,
        )
    return results


def scanpath_generalized_edit_distance(input, **kwargs):
    generalized_edit_deletion_cost = kwargs.get(
        "scanpath_generalized_edit_distance_deletion_cost", 0.2
    )
    generalized_edit_insertion_cost = kwargs.get(
        "scanpath_generalized_edit_distance_insertion_cost", 0.2
    )
    generalized_edit_normalization = kwargs.get(
        "scanpath_generalized_edit_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, ScanpathStringEditDistance):
        results = input.scanpath_generalized_edit_distance(
            generalized_edit_deletion_cost,
            generalized_edit_insertion_cost,
            generalized_edit_normalization,
            display_results,
        )
    else:
        pm_distance = ScanpathStringEditDistance(input, **kwargs)
        results = pm_distance.scanpath_generalized_edit_distance(
            generalized_edit_deletion_cost,
            generalized_edit_insertion_cost,
            generalized_edit_normalization,
            display_results,
        )
    return results


def scanpath_needleman_wunsch_distance(input, **kwargs):
    needleman_wunsch_concordance_bonus = kwargs.get(
        "scanpath_needleman_wunsch_distance_concordance_bonus", 0.1
    )
    needleman_wunsch_gap_cost = kwargs.get(
        "scanpath_needleman_wunsch_distance_gap_cost", 0.25
    )
    needleman_wunsch_normalization = kwargs.get(
        "scanpath_needleman_wunsch_distance_normalization", "max"
    )

    display_results = kwargs.get("display_results", True)

    if isinstance(input, ScanpathStringEditDistance):
        results = input.scanpath_needleman_wunsch_distance(
            needleman_wunsch_concordance_bonus,
            needleman_wunsch_gap_cost,
            needleman_wunsch_normalization,
            display_results,
        )
    else:
        pm_distance = ScanpathStringEditDistance(input, **kwargs)
        results = pm_distance.scanpath_needleman_wunsch_distance(
            needleman_wunsch_concordance_bonus,
            needleman_wunsch_gap_cost,
            needleman_wunsch_normalization,
            display_results,
        )
    return results
