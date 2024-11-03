# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.aoi.pattern_mining.n_gram import NGram
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.similarity.character_based.character_based_base import CharacterBased
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class SubsMatch(CharacterBased):
    def __init__(self, input, config, id_1, id_2):
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

        super().__init__(input, config)

        n_w = config["subsmatch_ngram_length"]

        self.h_1 = NGram(self.s_1, n_w)
        self.h_2 = NGram(self.s_2, n_w)

        dist_ = self.compute_distance()
        self.sim_ = 1 - dist_

        if config["display_results"]:
            print(
                "\nComparing sequences {id_1} and {id_2}: ".format(id_1=id_1, id_2=id_2)
            )
            print("\n --- Subsequences from sequence {id_1}: ---".format(id_1=id_1))
            for it in self.h_1.table.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.h_1.table[it]
                    )
                )
            print("\n --- Subsequences from sequence {id_2}: ---".format(id_2=id_2))
            for it in self.h_2.table.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.h_2.table[it]
                    )
                )
            print("\n")

    def compute_distance(self):
        """


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        t1 = self.h_1.table
        t2 = self.h_2.table

        keys = set(list(t1.keys()) + list(t2.keys()))
        dist = 0

        for key in keys:
            dist += abs(t1.get(key, 0.0) - t2.get(key, 0.0))

        return dist / 2


class SubsMatchSimilarity:
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
        display_results = kwargs.get("display_results", True)

        if verbose:
            print("Processing SubsMatch Similarity...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        else:
            raise ValueError(
                "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
            )

        self.config = scanpaths[0].config
        self.config.update({"verbose": verbose})
        self.config.update({"display_results": display_results})

        ## Keep max x and y sizes for gaze visual field
        x_size_max = np.max(
            np.array([scanpath.config["size_plan_x"] for scanpath in scanpaths])
        )
        y_size_max = np.max(
            np.array([scanpath.config["size_plan_y"] for scanpath in scanpaths])
        )

        vf_diag = np.linalg.norm(
            np.array([self.config["size_plan_x"], self.config["size_plan_y"]])
        )

        self.config.update(
            {
                "size_plan_x": x_size_max,
                "size_plan_y": y_size_max,
                "screen_diagonal": vf_diag,
            }
        )

        ## Update parameters for temporal and spatial binning
        self.config.update(
            {
                "scanpath_spatial_binning_nb_pixels_x": kwargs.get(
                    "scanpath_spatial_binning_nb_pixels_x", 8
                ),
                "scanpath_spatial_binning_nb_pixels_y": kwargs.get(
                    "scanpath_spatial_binning_nb_pixels_y", 8
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

        self.config.update(
            {"subsmatch_ngram_length": kwargs.get("subsmatch_ngram_length", 3)}
        )

        scanpaths = scanpaths
        n_sp = len(scanpaths)

        s_m = np.zeros((n_sp, n_sp))

        for j in range(1, n_sp):
            for i in range(j):
                e_a = SubsMatch(
                    [scanpaths[i], scanpaths[j]], self.config, id_1=str(i), id_2=str(j)
                )
                s_m[i, j] = e_a.sim_

        s_m += s_m.T

        self.results = dict({"subsmatch_similarity_matrix": s_m})
        self.verbose()

        if verbose:
            print("...SubsMatch Similarity done\n")

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


def scanpath_subsmatch_similarity(input, **kwargs):
    mm = SubsMatchSimilarity(input, **kwargs)
    results = mm.results

    return results
