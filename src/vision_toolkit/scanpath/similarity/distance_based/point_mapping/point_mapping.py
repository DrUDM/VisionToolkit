# -*- coding: utf-8 -*-

import copy

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.similarity.distance_based.point_mapping.eye_analysis_distance import EyeAnalysisDistance
from vision_toolkit.scanpath.similarity.distance_based.point_mapping.mannan_distance import MannanDistance
from vision_toolkit.scanpath.similarity.distance_based.point_mapping.TDE_distance import TDEDistance
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class PointMappingDistance:
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
            print("Processing Point Mapping Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a PointMappingDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        else:
            raise ValueError(
                "Input must be a PointMappingDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
            )

        self.config = scanpaths[0].config
        self.config.update({"verbose": verbose})

        if "nb_samples" in self.config.keys():
            del self.config["nb_samples"]

        self.scanpaths = scanpaths
        self.n_sp = len(scanpaths)

        self.dict_methods = dict(
            {
                "eye_analysis_distance": EyeAnalysisDistance,
                "mannan_distance": MannanDistance,
                "TDE_distance": TDEDistance,
            }
        )

        if verbose:
            print("...Point Mapping Distance done\n")

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

    def pm_dist_mat(self, distance, config):
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

    def eye_analysis_distance(self, display_results):
        """


        Parameters
        ----------
        display_results : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.config.update({"display_results": display_results})

        config = copy.deepcopy(self.config)
        d_m = self.pm_dist_mat("eye_analysis_distance", config)

        results = dict({"eye_analysis_distance_matrix": d_m})
        self.verbose()

        return results

    def mannan_distance(self, mannan_nb_random_scanpaths, display_results):
        """


        Parameters
        ----------
        mannan_nb_random_scanpaths : TYPE
            DESCRIPTION.
        display_results : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.config.update({"display_results": display_results})

        config = copy.deepcopy(self.config)
        config.update(
            {"mannan_distance_nb_random_scanpaths": mannan_nb_random_scanpaths}
        )

        d_m = self.pm_dist_mat("mannan_distance", config)

        results = dict({"mannan_distance_matrix": d_m})
        self.verbose(
            dict({"mannan_distance_nb_random_scanpaths": mannan_nb_random_scanpaths})
        )

        return results

    def TDE_distance(
        self, TDED_method, TDED_subsequence_length, TDED_scaling, display_results
    ):
        """


        Parameters
        ----------
        TDED_method : TYPE
            DESCRIPTION.
        TDED_subsequence_length : TYPE
            DESCRIPTION.
        TDED_scaling : TYPE
            DESCRIPTION.
        display_results : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.config.update({"display_results": display_results})

        config = copy.deepcopy(self.config)
        config.update(
            {
                "TDE_distance_method": TDED_method,
                "TDE_distance_subsequence_length": TDED_subsequence_length,
                "TDE_distance_scaling": TDED_scaling,
            }
        )

        d_m = self.pm_dist_mat("TDE_distance", config)

        results = dict({"TDE_distance_matrix": d_m})
        self.verbose(
            dict(
                {
                    "TDE_distance_method": TDED_method,
                    "TDE_distance_subsequence_length": TDED_subsequence_length,
                    "TDE_distance_scaling": TDED_scaling,
                }
            )
        )

        return results


def scanpath_eye_analysis_distance(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, PointMappingDistance):
        results = input.eye_analysis_distance(display_results)

    else:
        pm_distance = PointMappingDistance(input, **kwargs)
        results = pm_distance.eye_analysis_distance(display_results)

    return results


def scanpath_mannan_distance(input, **kwargs):
    mannan_nb_random_scanpaths = kwargs.get("mannan_distance_nb_random_scanpaths", 1000)
    display_results = kwargs.get("display_results", True)

    if isinstance(input, PointMappingDistance):
        results = input.mannan_distance(mannan_nb_random_scanpaths, display_results)

    else:
        pm_distance = PointMappingDistance(input, **kwargs)
        results = pm_distance.mannan_distance(
            mannan_nb_random_scanpaths, display_results
        )

    return results


def scanpath_TDE_distance(input, **kwargs):
    TDED_method = kwargs.get("TDE_distance_method", "mean_minimal")
    TDED_subsequence_length = kwargs.get("TDE_distance_subsequence_length", 5)
    TDED_scaling = kwargs.get("TDE_distance_scaling", True)

    display_results = kwargs.get("display_results", True)

    if isinstance(input, PointMappingDistance):
        results = input.TDE_distance(
            TDED_method, TDED_subsequence_length, TDED_scaling, display_results
        )

    else:
        pm_distance = PointMappingDistance(input, **kwargs)
        results = pm_distance.TDE_distance(
            TDED_method, TDED_subsequence_length, TDED_scaling, display_results
        )

    return results
