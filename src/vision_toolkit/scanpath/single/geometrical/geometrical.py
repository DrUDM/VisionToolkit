# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.single.geometrical.bcea import bcea
from vision_toolkit.scanpath.single.geometrical.convex_hull import ConvexHull
from vision_toolkit.scanpath.single.geometrical.hfd import HiguchiFractalDimension
from vision_toolkit.scanpath.single.geometrical.k_coefficient import modified_k_coefficient
from vision_toolkit.scanpath.single.geometrical.voronoi import VoronoiCells
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class GeometricalAnalysis:
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

        if verbose:
            print("Processing Geometrical Analysis...\n")

        if isinstance(input, str):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, Scanpath):
            self.scanpath = input

        else:
            raise ValueError(
                "Input must be a csv, or a BinarySegmentation, or a Scanpath object"
            )

        if verbose:
            print("...Geometrical Analysis done\n")

    def scanpath_length(self):
        x_ = self.scanpath.values[:2]
        d_ = np.sum(np.linalg.norm(x_[:, 1:] - x_[:, :-1], axis=0))
        results = dict({"length": d_})

        return results

    def scanpath_BCEA(self, BCEA_probability, display_results, display_path):
        """


        Parameters
        ----------
        BCEA_probability : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )
        bcea_ = bcea(self.scanpath, BCEA_probability)

        ## Get results
        results = dict({"BCEA": bcea_})

        self.scanpath.verbose(dict({"scanpath_BCEA_probability": BCEA_probability}))

        return results

    def scanpath_k_coefficient(self, display_results, display_path):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )
        k_c = modified_k_coefficient(self.scanpath)

        ## Get results
        results = dict({"k_coefficient": k_c})

        self.scanpath.verbose()

        return results

    def scanpath_voronoi_cells(self, display_results, display_path, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        ## Compute Voronoi areas
        v_a = VoronoiCells(self.scanpath)

        ## Get results
        results = v_a.results

        self.scanpath.verbose(dict({"get_raw": get_raw}))

        if not get_raw:
            del results["voronoi_areas"]

        return results

    def scanpath_convex_hull(self, display_results, display_path, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        ## Compute convex hull
        c_h = ConvexHull(self.scanpath)

        ## Get results
        results = c_h.results

        self.scanpath.verbose(dict({"get_raw": get_raw}))

        if not get_raw:
            del results["hull_apex"]

        return results

    def scanpath_HFD(
        self, HFD_hilbert_iterations, HFD_k_max, display_results, display_path, get_raw
    ):
        """


        Parameters
        ----------
        HFD_hilbert_iterations : TYPE
            DESCRIPTION.
        HFD_k_max : TYPE
            DESCRIPTION.
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )
        ## Compute fractal dimension
        h_fd = HiguchiFractalDimension(self.scanpath, HFD_hilbert_iterations, HFD_k_max)
        ## Get results
        results = h_fd.results

        self.scanpath.verbose(
            dict(
                {
                    "scanpath_HFD_hilbert_iterations": HFD_hilbert_iterations,
                    "scanpath_HFD_k_max": HFD_k_max,
                    "get_raw": get_raw,
                }
            )
        )

        if not get_raw:
            del results["log_lengths"]
            del results["log_inverse_time_intervals"]

        return results


def scanpath_length(input, **kwargs):
    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_length()

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_length()

    return results


def scanpath_BCEA(input, **kwargs):
    BCEA_probability = kwargs.get("scanpath_BCEA_probability", 0.68)
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_BCEA(BCEA_probability, display_results, display_path)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_BCEA(
            BCEA_probability, display_results, display_path
        )

    return results


def scanpath_k_coefficient(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_k_coefficient(display_results, display_path)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_k_coefficient(
            display_results, display_path
        )

    return results


def scanpath_voronoi_cells(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_voronoi_cells(display_results, display_path, get_raw)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_voronoi_cells(
            display_results, display_path, get_raw
        )

    return results


def scanpath_convex_hull(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_convex_hull(display_results, display_path, get_raw)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_convex_hull(
            display_results, display_path, get_raw
        )

    return results


def scanpath_HFD(input, **kwargs):
    HFD_hilbert_iterations = kwargs.get("scanpath_HFD_hilbert_iterations", 4)
    HFD_k_max = kwargs.get("scanpath_HFD_k_max", 10)

    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_HFD(
            HFD_hilbert_iterations, HFD_k_max, display_results, display_path, get_raw
        )

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_HFD(
            HFD_hilbert_iterations, HFD_k_max, display_results, display_path, get_raw
        )

    return results
