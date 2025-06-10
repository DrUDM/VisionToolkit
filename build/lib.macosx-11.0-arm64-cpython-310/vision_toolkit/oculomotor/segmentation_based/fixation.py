# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class FixationAnalysis(BinarySegmentation):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        sampling_frequency : TYPE, optional
            DESCRIPTION. The default is None.
        segmentation_method : TYPE, optional
            DESCRIPTION. The default is 'I_HMM'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Fixation Analysis...\n")

        if isinstance(input, BinarySegmentation):
            self.__dict__ = input.__dict__.copy()

        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert (
                sampling_frequency is not None
            ), "Sampling frequency must be specified"

            super().__init__(input, **kwargs)

        self.s_f = self.config["sampling_frequency"]
        assert (
            len(self.segmentation_results["fixation_intervals"]) > 0
        ), "No fixation identified"
       
        if verbose:
            print("...Fixation Analysis done\n")

    def fixation_count(self):
        """


        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["fixation_intervals"])
        result = dict({"count": ct})

        return result

    def fixation_frequency(self):
        """


        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["fixation_intervals"])
        f = ct / (self.config["nb_samples"] / self.s_f)

        result = dict({"frequency": f})

        return result

    def fixation_frequency_wrt_labels(self):
        """


        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["fixation_intervals"])
        f = ct / (np.sum(self.segmentation_results["is_labeled"]) / self.s_f)

        result = dict({"frequency": f})

        return result

    def fixation_durations(self, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        a_i = np.array(self.segmentation_results["fixation_intervals"])
        a_d = (a_i[:, 1] - a_i[:, 0] + 1) / self.s_f

        results = dict(
            {
                "duration_mean": np.nanmean(a_d),
                "duration sd": np.nanstd(a_d, ddof=1),
                "raw": a_d,
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def fixation_centroids(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        ctrds = self.segmentation_results["centroids"]

        results = dict({"centroids": np.array(ctrds)})

        return results

    def fixation_mean_velocities(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["fixation_intervals"]
        a_sp = self.data_set["absolute_speed"]

        m_sp = []
        sd_sp = []

        for _int in _ints:
            m_sp.append(np.nanmean(a_sp[_int[0] : _int[1]]))
            sd_sp.append(np.nanstd(a_sp[_int[0] : _int[1]], ddof=1))

        results = dict(
            {"velocity_means": np.array(m_sp), "velocity_sd": np.array(sd_sp)}
        )

        return results

    def fixation_average_velocity_means(self, weighted, get_raw):
        """


        Parameters
        ----------
        weighted : TYPE, optional
            DESCRIPTION. The default is False.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        m_sp = self.fixation_mean_velocities()["velocity_means"]

        if not weighted:
            results = dict({"average_velocity_means": np.nanmean(m_sp), "raw": m_sp})

            if not get_raw:
                del results["raw"]

            return results

        else:
            i_d = self.fixation_durations(get_raw=True)["raw"] * self.s_f - 1
            w_v = np.sum(i_d * m_sp) / np.sum(i_d)

            results = dict({"weighted_average_velocity_means": w_v, "raw": m_sp})

            if not get_raw:
                del results["raw"]

            return results

    def fixation_average_velocity_deviations(self, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        sd_sp = self.fixation_mean_velocities()["velocity_sd"]
        i_d = self.fixation_durations(get_raw=True)["raw"] * self.s_f - 2

        a_sd = np.sqrt(np.sum(i_d * (sd_sp**2)) / np.sum(i_d))

        results = dict({"average_velocity_sd": a_sd, "raw": sd_sp})

        if not get_raw:
            del results["raw"]

        return results

    def fixation_drift_displacements(self, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]

        _ints = self.segmentation_results["fixation_intervals"]
        dist_ = self.distances[self.config["distance_type"]]

        dsp = []

        for _int in _ints:
            l_d = dist_(
                np.array([x_a[_int[0]], y_a[_int[0]], z_a[_int[0]]]),
                np.array([x_a[_int[1]], y_a[_int[1]], z_a[_int[1]]]),
            )

            dsp.append(l_d)

        results = dict(
            {
                "drift_displacement_mean": np.nanmean(np.array(dsp)),
                "drift_displacement_sd": np.nanstd(np.array(dsp), ddof=1),
                "raw": np.array(dsp),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def fixation_drift_distances(self, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]
        n_s = len(x_a)
        _ints = self.segmentation_results["fixation_intervals"]
        dist_ = self.distances[self.config["distance_type"]]

        t_cum = []
        stack_ = np.concatenate(
            (x_a.reshape((1, n_s)), y_a.reshape((1, n_s)), z_a.reshape((1, n_s))),
            axis=0,
        )

        for _int in _ints:
            if self.config["distance_type"] == "euclidean":
                l_a = stack_[:, _int[0] : _int[1] + 1]
                l_c = np.sum(norm(l_a[:, 1:] - l_a[:, :-1], axis=0))
                t_cum.append(l_c)
            else:
                l_c = np.sum(
                    np.array(
                        [
                            dist_(
                                np.array([x_a[k], y_a[k], z_a[k]]),
                                np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]]),
                            )
                            for k in range(_int[0], _int[1])
                        ]
                    )
                )
                t_cum.append(l_c)

        results = dict(
            {
                "drift_cumul_distance_mean": np.nanmean(np.array(t_cum)),
                "drift_cumul_distance_sd": np.nanstd(np.array(t_cum), ddof=1),
                "raw": np.array(t_cum),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def fixation_drift_velocities(self, get_raw):
        """


        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        d_d = self.fixation_drift_displacements(get_raw=True)["raw"]
        i_d = self.fixation_durations(get_raw=True)["raw"] - 1 / self.s_f

        d_vel = [d_d[i] / i_d[i] for i in range(len(i_d)) if i_d[i] > 0]

        results = dict(
            {
                "drift_velocity_mean": np.nanmean(np.array(d_vel)),
                "drift_velocity_sd": np.nanstd(np.array(d_vel), ddof=1),
                "raw": np.array(d_vel),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def fixation_BCEA(self, BCEA_probability, get_raw):
        """


        Parameters
        ----------
        p : TYPE, optional
            DESCRIPTION. The default is 0.68.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
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

            mx = np.nanmean(x)
            my = np.nanmean(y)

            xm, ym = x - mx, y - my

            _num = np.sum(xm * ym)
            _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))

            p_c = _num / _den

            ## For some small artifact of floating point arithmetic.
            p_c = max(min(p_c, 1.0), -1.0)

            return p_c

        k = -np.log(1 - BCEA_probability)
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]

        _ints = self.segmentation_results["fixation_intervals"]
        bcea_s = []

        for _int in _ints:
            l_x = x_a[_int[0] : _int[1] + 1]
            l_y = y_a[_int[0] : _int[1] + 1]

            p_c = pearson_corr(l_x, l_y)
            sd_x = np.nanstd(l_x, ddof=1)
            sd_y = np.nanstd(l_y, ddof=1)

            l_bcea = 2 * np.pi * k * sd_x * sd_y * np.sqrt(1 - p_c**2)
            bcea_s.append(l_bcea)

        results = dict(
            {"average_BCEA": np.nanmean(np.array(bcea_s)), "raw": np.array(bcea_s)}
        )

        if not get_raw:
            del results["raw"]

        return results


## Some access functions
def fixation_count(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_count()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_count()
        fixation_analysis.verbose()

    return results


def fixation_frequency(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_frequency()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_frequency()
        fixation_analysis.verbose()

    return results


def fixation_frequency_wrt_labels(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_frequency_wrt_labels()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_frequency_wrt_labels()
        fixation_analysis.verbose()

    return results


def fixation_durations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_durations(get_raw)
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_durations(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_centroids(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_centroids()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_centroids()
        fixation_analysis.verbose()

    return results


def fixation_mean_velocities(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_mean_velocities()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_mean_velocities()
        fixation_analysis.verbose()

    return results


def fixation_average_velocity_means(input, **kwargs):
    weighted = kwargs.get("fixation_weighted_average_velocity_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_average_velocity_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "fixation_weighted_average_velocity_means": weighted,
                }
            )
        )

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_average_velocity_means(weighted, get_raw)
        fixation_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "fixation_weighted_average_velocity_means": weighted,
                }
            )
        )

    return results


def fixation_average_velocity_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_average_velocity_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_average_velocity_deviations(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_displacements(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_displacements(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_displacements(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_distances(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_distances(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_distances(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_velocities(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_velocities(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_velocities(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_BCEA(input, **kwargs):
    BCEA_probability = kwargs.get("fixation_BCEA_probability", 0.68)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_BCEA(BCEA_probability, get_raw)
        input.verbose(
            dict({"get_raw": get_raw, "fixation_BCEA_probability": BCEA_probability})
        )

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_BCEA(BCEA_probability, get_raw)
        fixation_analysis.verbose(
            dict({"get_raw": get_raw, "fixation_BCEA_probability": BCEA_probability})
        )

    return results
