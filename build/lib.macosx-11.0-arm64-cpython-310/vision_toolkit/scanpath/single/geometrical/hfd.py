# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

from vision_toolkit.utils.binning import spatial_bin
from vision_toolkit.visualization.scanpath.single.geometrical import plot_HFD


class HiguchiFractalDimension:
    def __init__(self, scanpath, HFD_hilbert_iterations, HFD_k_max):
        """


        Parameters
        ----------
        scanpath : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        ## Initialize parameters
        h_i = HFD_hilbert_iterations
        self.k_m = HFD_k_max
        x_grid_size = y_grid_size = 2**h_i
        x_size = scanpath.config["size_plan_x"]
        y_size = scanpath.config["size_plan_y"]

        ## Spatial binning of the input scanpath
        self.sc_b = spatial_bin(
            scanpath.values[0:2], x_grid_size, y_grid_size, x_size, y_size
        )
        ## Initialize Hilber curve
        h_c = HilbertCurve(h_i, 2)
        ## Compute distances from the Hilbert curve
        self.dist_ = h_c.distances_from_points(self.sc_b.T)
        ## Compute Higuchi fractal dimension
        s_, x_, l_ = self.compute_hfd(np.array(self.dist_))

        self.results = dict(
            {
                "fractal_dimension": s_[0],
                "log_lengths": l_,
                "log_inverse_time_intervals": x_,
            }
        )

        if scanpath.config["display_results"]:
            dist_h = list(range(0, int(2 ** (h_i * 2) - 1)))
            h_pts = h_c.points_from_distances(dist_h)

            plot_HFD(
                self.sc_b,
                self.dist_,
                np.array(h_pts),
                s_,
                x_,
                l_,
                scanpath.config["display_path"],
            )

    def compute_hfd(self, dist_):
        """


        Parameters
        ----------
        dist_ : TYPE
            DESCRIPTION.

        Returns
        -------
        s_ : TYPE
            DESCRIPTION.
        x_ : TYPE
            DESCRIPTION.
        l_ : TYPE
            DESCRIPTION.

        """
        l_, x_ = list(), list()
        n = len(dist_)

        for k in range(1, self.k_m + 1):
            l_k = 0
            for m in range(0, k):
                idxs = np.arange(
                    1, int(np.floor((n - (m + 1)) / k) + 1), dtype=np.int32
                )
                l_mk = np.sum(np.abs(dist_[m + idxs * k] - dist_[m + (idxs - 1) * k]))
                if ((n - (m + 1)) / k) > 0:
                    lmk = l_mk * (n - 1) / (((n - (m + 1)) / k) * k**2)
                    l_k += lmk
            if l_k == 0:
                l_.append(np.nan)
            else:
                l_.append(np.log(l_k / k))
            x_.append(np.log(1.0 / k))

        x_ = np.array(x_)
        l_ = np.array(l_)

        idx = np.isfinite(x_) & np.isfinite(l_)
        s_ = np.polyfit(x_[idx], l_[idx], 1)

        return s_, x_, l_
