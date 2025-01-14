# -*- coding: utf-8 -*-

import copy

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.similarity.distance_based.elastic_distance.DTW_distance import DTWDistance
from vision_toolkit.scanpath.similarity.distance_based.elastic_distance.frechet_distance import FrechetDistance
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class ElasticDistance:
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
            print("Processing Elastic Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a ElasticDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        else:
            raise ValueError(
                "Input must be a ElasticDistance instance or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
            )

        self.config = scanpaths[0].config
        self.config.update({"verbose": verbose})

        if "nb_samples" in self.config.keys():
            del self.config["nb_samples"]

        self.scanpaths = scanpaths
        self.n_sp = len(scanpaths)

        self.dict_methods = dict(
            {"DTW_distance": DTWDistance, "frechet_distance": FrechetDistance}
        )

        if verbose:
            print("...Elastic Distance done\n")

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

    def el_dist_mat(self, distance, config):
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

    def DTW_distance(self, display_results):
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
        d_m = self.el_dist_mat("DTW_distance", config)

        results = dict({"DTW_distance_matrix": d_m})
        self.verbose()

        return results

    def frechet_distance(self, display_results):
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
        d_m = self.el_dist_mat("frechet_distance", config)

        results = dict({"frechet_distance_matrix": d_m})
        self.verbose()

        return results


def scanpath_DTW_distance(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, ElasticDistance):
        results = input.DTW_distance(display_results)

    else:
        pm_distance = ElasticDistance(input, **kwargs)
        results = pm_distance.DTW_distance(display_results)

    return results


def scanpath_frechet_distance(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, ElasticDistance):
        results = input.frechet_distance(display_results)

    else:
        pm_distance = ElasticDistance(input, **kwargs)
        results = pm_distance.frechet_distance(display_results)

    return results
