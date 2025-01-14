# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.visualization.scanpath.single.geometrical import plot_BCEA


def bcea(scanpath, probability):
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
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        p_c : TYPE
            DESCRIPTION.

        """

        x = np.asarray(x)
        y = np.asarray(y)

        mx = x.mean()
        my = y.mean()

        xm, ym = x - mx, y - my

        _num = np.sum(xm * ym)
        _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))

        p_c = _num / _den

        ## For some small artifact of floating point arithmetic.
        p_c = max(min(p_c, 1.0), -1.0)

        return p_c

    k = -np.log(1 - probability)

    x_a = scanpath.values[0]
    y_a = scanpath.values[1]

    p_c = pearson_corr(x_a, y_a)
    sd_x = np.std(x_a, ddof=1)
    sd_y = np.std(y_a, ddof=1)

    bcea = 2 * np.pi * k * sd_x * sd_y * np.sqrt(1 - p_c**2)

    if scanpath.config["display_results"]:
        plot_BCEA(scanpath.values, probability, scanpath.config["display_path"])

    return bcea
