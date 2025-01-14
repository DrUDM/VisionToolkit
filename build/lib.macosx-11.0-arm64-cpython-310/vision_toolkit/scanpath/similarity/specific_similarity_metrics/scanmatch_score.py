# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.scanpath.similarity.character_based.character_based_base import CharacterBased
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.utils.binning import dist_mat


class ScanMatch(CharacterBased):
    def __init__(self, input, config, id_1, id_2):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        id_1 : TYPE
            DESCRIPTION.
        id_2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super().__init__(input, config)

        c_b = config["scanmatch_score_concordance_bonus"]
        g_c = config["scanmatch_score_gap_cost"]

        s_t = config["scanmatch_score_substitution_threshold"]

        d_m, i_dict = dist_mat(
            self.x_nb_pixels, self.y_nb_pixels, self.elem_sizes, normalize=False
        )

        ## Apply cutoff point at which values in the substitution matrix are positive
        ## corresponding to negative distance values
        s_dm = d_m - s_t

        ## Minimal shifted distance set to -substitution_threshold
        min_ = -s_t
        ## Maximal distance between pixel centers
        max_ = np.linalg.norm(
            np.array(
                [
                    config["size_plan_x"] - self.elem_sizes[0, 0],
                    config["size_plan_y"] - self.elem_sizes[1, 0],
                ]
            )
        )

        ## Normalize negative distance values between [-c_b, 0]
        s_dm[s_dm < 0] = c_b * ((s_dm[s_dm < 0] - min_) / (-min_)) - c_b

        ## Normalize positive distance values between [0, 1]
        s_dm[s_dm > 0] = 1 * ((s_dm[s_dm > 0]) / (max_))

        self.opt_align, score_ = c_comparison.needleman_wunsch(
            self.s_1, self.s_2, g_c, c_b, i_dict, s_dm
        )
        self.score_ = score_ / (c_b * max(self.n_1, self.n_2))

        if config["display_results"]:
            self.compute_visualization(id_1, id_2)


class ScanMatchScore:
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
            print("Processing ScanMatch Score...\n")

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
                    "scanpath_spatial_binning_nb_pixels_x", 20
                ),
                "scanpath_spatial_binning_nb_pixels_y": kwargs.get(
                    "scanpath_spatial_binning_nb_pixels_y", 20
                ),
                "scanpath_temporal_binning": True,
            }
        )

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
            {
                "scanmatch_score_concordance_bonus": kwargs.get(
                    "scanmatch_score_concordance_bonus", 0.2
                ),
                "scanmatch_score_gap_cost": kwargs.get("scanmatch_score_gap_cost", 0.0),
                "scanmatch_score_substitution_threshold": kwargs.get(
                    "scanmatch_score_substitution_threshold", 0.15 * vf_diag
                ),
            }
        )

        scanpaths = scanpaths
        n_sp = len(scanpaths)

        s_m = np.zeros((n_sp, n_sp))

        for j in range(1, n_sp):
            for i in range(j):
                e_a = ScanMatch(
                    [scanpaths[i], scanpaths[j]], self.config, id_1=str(i), id_2=str(j)
                )

                s_m[i, j] = e_a.score_

        s_m += s_m.T

        self.results = dict({"scanmatch_score_matrix": s_m})
        self.verbose()

        if verbose:
            print("...ScanMatch Score done\n")

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


def scanpath_scanmatch_score(input, **kwargs):
    sm_s = ScanMatchScore(input, **kwargs)
    results = sm_s.results

    return results
