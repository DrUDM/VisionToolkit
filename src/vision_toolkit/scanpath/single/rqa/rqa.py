# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.single.rqa.rqa_base import RecurrenceBase
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.visualization.scanpath.single.rqa import (
    plot_RQA,
    plot_RQA_determinism,
    plot_RQA_laminarity)

 
## See https://drive.google.com/file/d/14mKqSnkGl08jzpgX_aRl5qsT4TqKDjk6/view

class RQAAnalysis(RecurrenceBase):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : str or BinarySegmentation or Scanpath
            DESCRIPTION.
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE, optional
            DESCRIPTION. The default is 'I_HMM'.
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
            print("Processing RQA Analysis...\n")

        if isinstance(input, str):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, Scanpath):
            self.scanpath = input

        else:
            raise ValueError(
                "Input must be a csv, a BinarySegmentation or a Scanpath object"
            )

        d_thrs = (
            np.linalg.norm(
                np.array(
                    [
                        self.scanpath.config["size_plan_x"],
                        self.scanpath.config["size_plan_y"],
                    ]
                )
            )
            * 0.015
        )
    
        self.scanpath.config.update(
            {
                "scanpath_RQA_distance_threshold": kwargs.get(
                    "scanpath_RQA_distance_threshold", d_thrs
                ),
                "scanpath_RQA_minimum_length": kwargs.get(
                    "scanpath_RQA_minimum_length", 3
                ),
                "verbose": verbose
            }
        )

        super().__init__(self.scanpath.values)

        ## Compute the recurrence matrix r_m and the sum r_p of
        ## recurrences in the upper triangle of the recurrence matrix
        self.r_m, self.r_p = self.comp_recurrence_matrix()

        ## Compute the upper trinagle matrix
        self.r_u = np.zeros((self.n, self.n))
        r_c = np.triu_indices(self.n, 1)
        self.r_u[r_c] = self.r_m[r_c]

        ## Compute the set of horizontal lines
        self.h_set = self.find_lines(
            self.r_u.T,
            self.scanpath.config["scanpath_RQA_minimum_length"],
            "horizontal",
        )

        ## Compute the set of vertical lines
        self.v_set = self.find_lines(
            self.r_u, self.scanpath.config["scanpath_RQA_minimum_length"], "vertical"
        )

        ## Compute the set of diagonal lines
        self.d_set = self.find_diags(
            self.r_u, self.scanpath.config["scanpath_RQA_minimum_length"]
        )

        if display_results:
            plot_RQA(self.r_m)
 
        if verbose:
            print("...RQA Analysis done\n")

    def comp_recurrence_matrix(self):
        """


        Returns
        -------
        r_m : TYPE
            DESCRIPTION.
        r_p : TYPE
            DESCRIPTION.

        """

        s_ = self.s_[0:2]
        n = self.n

        d_m = cdist(s_.T, s_.T, metric="euclidean")

        r_m = (d_m < self.scanpath.config["scanpath_RQA_distance_threshold"]).astype(
            int
        )
        r_p = (np.sum(r_m) - n) / 2

        return r_m, r_p

    def scanapath_RQA_recurrence_rate(self, display_results):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update({"display_results": display_results})
        r_r = (200 * self.r_p) / ((self.n - 1) * self.n)

        if np.isnan(r_r):
            results = dict({"RQA_recurrence_rate": 0})

        else:
            results = dict({"RQA_recurrence_rate": r_r})

        self.scanpath.verbose()

        return results

    def scanpath_RQA_laminarity(self, display_results):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update({"display_results": display_results})
        s_l = 0

        v_set = self.v_set
        h_set = self.h_set

        for v in v_set:
            s_l += len(v)

        for h in h_set:
            s_l += len(h)

        lam = (50 / self.r_p) * s_l

        if self.scanpath.config["display_results"]:
            plot_RQA_laminarity(self.r_m, self.h_set, self.v_set)

        if np.isnan(lam):
            results = dict({"RQA_laminarity": 0})

        else:
            results = dict({"RQA_laminarity": lam})

        self.scanpath.verbose()

        return results

    def scanpath_RQA_determinism(self, display_results):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update({"display_results": display_results})
        s_l = 0

        d_set = self.d_set

        for d in d_set:
            s_l += len(d)
      
        det = (100 / self.r_p) * s_l

        if self.scanpath.config["display_results"]:
            plot_RQA_determinism(self.r_m, self.d_set)

        if np.isnan(det):
            results = dict({"RQA_determinism": 0})

        else:
            results = dict({"RQA_determinism": det})

        self.scanpath.verbose()

        return results

    def scanpath_RQA_CORM(self, display_results):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update({"display_results": display_results})
        n = self.n
        r_m = self.r_m

        corm = 100 / ((n - 1) * self.r_p)

        r_ = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                r_ += r_m[i, j] * (j - i)

        corm *= r_

        if np.isnan(corm):
            results = dict({"RQA_CORM": 0})

        else:
            results = dict({"RQA_CORM": corm})

        self.scanpath.verbose()

        return results

    def scanpath_RQA_entropy(self, display_results):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update({"display_results": display_results})
        d_set = self.d_set

        l_s = np.array([len(d) for d in d_set])
        u_, c_ = np.unique(l_s, return_counts=True)
        p_ = c_ / len(l_s)
        entropy = 0

        for p in list(p_):
            entropy -= p * np.log(p)

        if np.isnan(entropy):
            results = dict({"RQA_entropy": 0})

        else:
            results = dict({"RQA_entropy": entropy})

        self.scanpath.verbose()

        return results


def scanpath_RQA_recurrence_rate(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_recurrence_rate(display_results)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_recurrence_rate(display_results)

    return results


def scanpath_RQA_laminarity(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_laminarity(display_results)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_laminarity(display_results)

    return results


def scanpath_RQA_determinism(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_determinism(display_results)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_determinism(display_results)

    return results


def scanpath_RQA_CORM(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_CORM(display_results)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_CORM(display_results)

    return results


def scanpath_RQA_entropy(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_entropy(display_results)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_entropy(display_results)

    return results
