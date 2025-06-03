# -*- coding: utf-8 -*-

import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoISequence
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class AoIBasicAnalysis:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : str | BinarySegmentation | Scanpath | AoISequence
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Markov Based Analysis...\n")

        if isinstance(input, str):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        elif isinstance(input, AoISequence):
            self.aoi_sequence = input

        elif isinstance(input, Scanpath):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)
        else:
            raise ValueError(
                "Input must be a csv, or a BinarySegmentation, or a Scanpath, or an AoISequence object"
            )

    def AoI_count(self):
        """


        Returns
        -------
        None.

        """

        ct = len(list(self.aoi_sequence.centers.keys()))
        result = dict({"count": ct})

        return result

    def AoI_duration(self, get_raw):
        durations = self.aoi_sequence.values[2]
        t_dur = []
        identification_results = self.aoi_sequence.identification_results

        for aoi in identification_results["clustered_fixations"].keys():
            l_dur = durations[identification_results["clustered_fixations"][aoi]]
            t_dur.append(np.sum(l_dur))
        prop_ = np.array(t_dur) / np.sum(t_dur)

        results = dict(
            {
                "average_duration": np.nanmean(np.array(t_dur)),
                "variance_duration": np.nanstd(np.array(t_dur)),
                "raw": np.array(t_dur),
                "proportion": prop_,
            }
        )

        if not get_raw:
            del results["raw"]

        return results


    def AoI_BCEA(self, BCEA_probability, get_raw):
        """


        Parameters
        ----------
        BCEA_probability : TYPE
            DESCRIPTION.
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        identification_results = self.aoi_sequence.identification_results
        positions = self.aoi_sequence.values[:2]
        bcea_s = []

        for aoi in identification_results["clustered_fixations"].keys():
            l_pos = positions[:, identification_results["clustered_fixations"][aoi]]
            l_bcea = self.BCEA(l_pos[0], l_pos[1], BCEA_probability)
            bcea_s.append(l_bcea)
        ## Compute absolute distance to median value
        med_ = np.nanmedian(np.array(bcea_s))
        e_med = np.nansum([np.abs(med_ - bcea) for bcea in bcea_s]) / len(bcea_s)

        results = dict(
            {
                "average_BCEA": np.nanmean(np.array(bcea_s)),
                "disp_BCEA": e_med,
                "raw": np.array(bcea_s),
            }
        )

        if not get_raw:
            del results["raw"]

        return results


    def AoI_weighted_BCEA(self, BCEA_probability):
        """


        Parameters
        ----------
        BCEA_probability : TYPE
            DESCRIPTION.
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        identification_results = self.aoi_sequence.identification_results
        positions = self.aoi_sequence.values[:2]
        durations = self.aoi_sequence.values[2]
        bcea_s = []
        t_dur = []

        for aoi in identification_results["clustered_fixations"].keys():
            l_pos = positions[:, identification_results["clustered_fixations"][aoi]]
            l_dur = np.sum(
                durations[identification_results["clustered_fixations"][aoi]]
            )
            l_bcea = self.BCEA(l_pos[0], l_pos[1], BCEA_probability)
            bcea_s.append(l_bcea * l_dur)
            t_dur.append(l_dur)

        results = dict(
            {
                "average_weighted_BCEA": np.sum(bcea_s / np.sum(t_dur)),
            }
        )

        return results


    def BCEA(self, x_a, y_a, probability):
        """


        Parameters
        ----------
        scanpath : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.

        Returns
        -------
        p_c : TYPE
            DESCRIPTION.

        """

        def pearson_corr(x, y):
            x = np.asarray(x)
            y = np.asarray(y)

            mx = x.mean()
            my = y.mean()

            xm, ym = x - mx, y - my

            _num = np.sum(xm * ym)
            _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
            if _den < 1e-20:
                p_c = 0
            else:
                p_c = _num / _den

            ## For some small artifact of floating point arithmetic.
            p_c = max(min(p_c, 1.0), -1.0)

            return p_c

        k = -np.log(1 - probability)
        p_c = pearson_corr(x_a, y_a)
        sd_x = np.std(x_a, ddof=1)
        sd_y = np.std(y_a, ddof=1)

        bcea = 2 * np.pi * k * sd_x * sd_y * np.sqrt(1 - p_c**2)

        return bcea
