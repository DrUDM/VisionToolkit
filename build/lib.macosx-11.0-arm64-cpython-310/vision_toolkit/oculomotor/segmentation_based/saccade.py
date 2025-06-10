import numpy as np
from scipy.stats import gamma 

from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.visualization.oculomotor.main_sequence import plot_main_sequence


class SaccadeAnalysis(BinarySegmentation):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input_ : TYPE
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
            print("Processing Saccade Analysis...\n")

        if isinstance(input, BinarySegmentation):
            self.__dict__ = input.__dict__.copy()
            self.config.update({"verbose": verbose})

        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert (
                sampling_frequency is not None
            ), "Sampling frequency must be specified"

            super().__init__(input, **kwargs)

        self.s_f = self.config["sampling_frequency"]
        assert (
            len(self.segmentation_results["saccade_intervals"]) > 0
        ), "No saccade identified"

        if verbose:
            print("...Saccade Analysis done\n")

    def saccade_count(self):
        """


        Returns
        -------
        result : dict
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["saccade_intervals"])
        result = dict({"count": ct})
  
        return result

    def saccade_frequency(self):
        """


        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["saccade_intervals"])
        f = ct / (self.config["nb_samples"] / self.s_f)

        result = dict({"frequency": f})

        return result

    def saccade_frequency_wrt_labels(self):
        """


        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        ct = len(self.segmentation_results["saccade_intervals"])
        f = ct / (np.sum(self.segmentation_results["is_labeled"]) / self.s_f)
        result = dict({"frequency": f})

        return result

    def saccade_durations(self, get_raw):
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

        a_i = np.array(self.segmentation_results["saccade_intervals"]) + np.array(
            [[0, 1]]
        )
        a_d = (a_i[:, 1] - a_i[:, 0]) / self.s_f

        results = dict(
            {
                "duration_mean": np.nanmean(a_d),
                "duration_sd": np.nanstd(a_d, ddof=1),
                "raw": a_d,
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_amplitudes(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        dist_ = self.distances[self.config["distance_type"]]

        s_a = []

        for _int in _ints:
            l_a = dist_(
                np.array([x_a[_int[0]], y_a[_int[0]], z_a[_int[0]]]),
                np.array([x_a[_int[1]], y_a[_int[1]], z_a[_int[1]]]),
            )
            s_a.append(l_a)

        results = dict(
            {
                "amplitude_mean": np.nanmean(np.array(s_a)),
                "amplitude_sd": np.nanstd(np.array(s_a), ddof=1),
                "raw": np.round(np.array(s_a), 3),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_travel_distances(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        dist_ = self.distances[self.config["distance_type"]]

        d_cum = []

        for _int in _ints:
            l_d = np.sum(
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
            d_cum.append(l_d)

        results = dict(
            {
                "distance_mean": np.nanmean(np.array(d_cum)),
                "distance_sd": np.nanstd(np.array(d_cum), ddof=1),
                "raw": np.array(d_cum),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_efficiencies(self, get_raw):
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

        s_a = self.saccade_amplitudes(get_raw=True)["raw"]
        d_cum = self.saccade_travel_distances(get_raw=True)["raw"]

        eff = [s_a[i] / d_cum[i] for i in range(len(s_a)) if d_cum[i] > 0]

        results = dict(
            {
                "efficiency_mean": np.nanmean(np.array(eff)),
                "efficiency_sd": np.nanstd(np.array(eff), ddof=1),
                "raw": np.array(eff),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def comp_dir(self, v_i):
        """


        Parameters
        ----------
        v_i : TYPE
            DESCRIPTION.

        Returns
        -------
        dir_ : TYPE
            DESCRIPTION.

        """

        ## To avoid numerical instability
        v_i += 1e-10

        _m = v_i[:, 1] < 0
        _p = v_i[:, 1] >= 0

        dir_ = np.zeros(len(v_i))

        n_p = np.linalg.norm(v_i[_p], axis=1)
        dir_[_p] = (180 / np.pi) * np.arccos(
            np.minimum(1, np.maximum(-1, np.divide(v_i[:, 0][_p], n_p, where=n_p > 0)))
        )

        n_m = np.linalg.norm(v_i[_m], axis=1)
        dir_[_m] = (180 / np.pi) * (
            2 * np.pi
            - np.arccos(
                np.minimum(
                    1, np.maximum(-1, np.divide(v_i[:, 0][_m], n_m, where=n_m > 0))
                )
            )
        )
        return dir_

    def saccade_directions(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]

        v_i = []

        for _int in _ints:
            v_i.append(
                np.array([x_a[_int[1]] - x_a[_int[0]], y_a[_int[1]] - y_a[_int[0]]])
            )
        dir_ = self.comp_dir(np.array(v_i))

        results = dict(
            {
                "direction_mean": np.nanmean(dir_),
                "direction_sd": np.nanstd(dir_, ddof=1),
                "raw": dir_,
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_horizontal_deviations(self, absolute, get_raw):
        """


        Parameters
        ----------
        absolute : TYPE, optional
            DESCRIPTION. The default is True.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]

        _ints = self.segmentation_results["saccade_intervals"]

        devs = []

        for _int in _ints:
            v_t = np.array([x_a[_int[1]] - x_a[_int[0]], y_a[_int[1]] - y_a[_int[0]]])

            dev = (180 / np.pi) * np.arccos(
                np.minimum(
                    1,
                    np.maximum(-1, np.dot(v_t / np.linalg.norm(v_t), np.array([1, 0]))),
                )
            )
            if absolute:
                ## We want the absolute deviation betwenn the horizontal axis and the saccade
                if dev > 90:
                    dev = 180 - dev
            devs.append(dev)

        results = dict(
            {
                "horizontal_deviation_mean": np.nanmean(np.array(devs)),
                "horizontal_deviation_sd": np.nanstd(np.array(devs), ddof=1),
                "raw": np.array(devs),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_successive_deviations(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]

        devs = []

        for i in range(len(_ints)):
            if i > 0:
                int_b = _ints[i - 1]
                int_a = _ints[i]

                v_b = np.array(
                    [x_a[int_b[1]] - x_a[int_b[0]], y_a[int_b[1]] - y_a[int_b[0]]]
                )

                v_a = np.array(
                    [x_a[int_a[1]] - x_a[int_a[0]], y_a[int_a[1]] - y_a[int_a[0]]]
                )

                dev = (180 / np.pi) * np.arccos(
                    np.minimum(
                        1,
                        np.maximum(
                            -1,
                            np.dot(
                                v_b / np.linalg.norm(v_b), v_a / np.linalg.norm(v_a)
                            ),
                        ),
                    )
                )
                devs.append(dev)

        results = dict(
            {
                "successive_deviation_mean": np.nanmean(np.array(devs)),
                "successive_deviation_sd": np.nanstd(np.array(devs), ddof=1),
                "raw": np.array(devs),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_initial_directions(self, duration_threshold, get_raw):
        """


        Parameters
        ----------
        duration_threshold : TYPE, optional
            DESCRIPTION. The default is 0.020.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        t_du = int(duration_threshold * self.s_f) + 1

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]

        _ints = self.segmentation_results["saccade_intervals"]
        v_i = []

        for _int in _ints:
            ## For small saccades
            t_s = min(t_du, (_int[1] - _int[0]))

            ## Compute saccadic vector corresponding to the initial saccadic displacement
            v_i.append(
                np.array(
                    [
                        x_a[_int[0] + t_s] - x_a[_int[0]],
                        y_a[_int[0] + t_s] - y_a[_int[0]],
                    ]
                )
            )

        ## Compute directions from the initial saccadic vectors
        dir_ = self.comp_dir(np.array(v_i))

        results = dict(
            {
                "initial_direction_mean": np.nanmean(dir_),
                "initial_direction_sd": np.nanstd(dir_, ddof=1),
                "raw": dir_,
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_initial_deviations(self, duration_threshold, get_raw):
        """


        Parameters
        ----------
        duration_threshold : TYPE, optional
            DESCRIPTION. The default is 0.020.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        t_du = int(duration_threshold * self.s_f) + 1

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]

        _ints = self.segmentation_results["saccade_intervals"]

        devs = []

        for _int in _ints:
            ## For small saccades
            t_s = min(t_du, (_int[1] - _int[0]))

            ## Compute saccadic vector corresponding to the initial saccadic displacement
            v_i = np.array(
                [x_a[_int[0] + t_s] - x_a[_int[0]], y_a[_int[0] + t_s] - y_a[_int[0]]]
            )

            v_t = np.array([x_a[_int[1]] - x_a[_int[0]], y_a[_int[1]] - y_a[_int[0]]])

            dev = (180 / np.pi) * np.arccos(
                np.minimum(
                    1,
                    np.maximum(
                        -1, np.dot(v_i / np.linalg.norm(v_i), v_t / np.linalg.norm(v_t))
                    ),
                )
            )
            devs.append(dev)

        results = dict(
            {
                "initial_deviation_mean": np.nanmean(np.array(devs)),
                "initial_deviation_sd": np.nanstd(np.array(devs), ddof=1),
                "raw": np.array(devs),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def shortest_distance(self, p_i, p_b=None, p_e=None):
        """
        Util function to compute the perpendcular distance between each saccadic
        # time-stamp and the total saccadic vector

        Parameters
        ----------
        p_i : TYPE
            DESCRIPTION.
        p_b : TYPE, optional
            DESCRIPTION. The default is None.
        p_e : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dist_ : TYPE
            DESCRIPTION.

        """

        if p_b == None:
            p_b = p_i[0]

        if p_e == None:
            p_e = p_i[-1]

        h_e = (p_e - p_b) / np.linalg.norm(p_e - p_b)

        n_i = len(p_i)

        d_ = p_i - p_b
        n_ = np.linalg.norm(d_, axis=1)
        h_v = np.divide(d_.T, n_, where=n_ > 0).T

        alpha_i = (h_v @ h_e.reshape(2, 1)).T[0]
        alpha_i = np.maximum(np.minimum(alpha_i, np.ones(n_i)), -np.ones(n_i))

        alpha_i = np.arccos(alpha_i)
        dist_ = np.sin(alpha_i) * np.linalg.norm(d_, axis=1)

        return dist_

    def linear_distance(self, p_i, p_b=None, p_e=None):
        """
        Util function to compute the linear distance along the total saccadic vector
        between two successive saccadic time-stamps

        Parameters
        ----------
        p_i : TYPE
            DESCRIPTION.
        p_b : TYPE, optional
            DESCRIPTION. The default is None.
        p_e : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        h_d_d : TYPE
            DESCRIPTION.

        """

        if p_b == None:
            p_b = p_i[0]

        if p_e == None:
            p_e = p_i[-1]

        h_e = (p_e - p_b) / np.linalg.norm(p_e - p_b)
        n_i = len(p_i)

        d_ = p_i - p_b
        n_ = np.linalg.norm(d_, axis=1)
        h_v = np.divide(d_.T, n_, where=n_ > 0).T

        alpha_i = (h_v @ h_e.reshape(2, 1)).T[0]
        alpha_i = np.maximum(np.minimum(alpha_i, np.ones(n_i)), -np.ones(n_i))

        alpha_i = np.arccos(alpha_i)
        h_d_ = np.cos(alpha_i) * np.linalg.norm(d_, axis=1)

        h_d_d = np.zeros(n_i)
        h_d_d[1:] = h_d_[1:] - h_d_[:-1]

        return h_d_d

    def saccade_max_curvatures(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        m_curv = []

        for _int in _ints:
            p_i = np.stack(
                [x_a[_int[0] : _int[1] + 1], y_a[_int[0] : _int[1] + 1]], axis=1
            )

            p_d = self.shortest_distance(p_i)
            m_c = np.max(p_d)
            m_curv.append(m_c)

        results = dict(
            {
                "max_curvature_mean": np.nanmean(np.array(m_curv)),
                "max_curvature_sd": np.nanstd(np.array(m_curv), ddof=1),
                "raw": np.array(m_curv),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_area_curvatures(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        area = []

        for _int in _ints:
            p_i = np.stack(
                [x_a[_int[0] : _int[1] + 1], y_a[_int[0] : _int[1] + 1]], axis=1
            )

            p_d = self.shortest_distance(p_i)
            l_d = self.linear_distance(p_i)

            l_area = np.sum(p_d * l_d)
            area.append(l_area)

        results = dict(
            {
                "curvature_area_mean": np.nanmean(np.array(area)),
                "curvature_area_sd": np.nanstd(np.array(area), ddof=1),
                "raw": np.array(area),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_mean_velocities(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["saccade_intervals"]
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

    def saccade_average_velocity_means(self, weighted, get_raw):
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

        m_sp = self.saccade_mean_velocities()["velocity_means"]

        if not weighted:
            results = dict({"average_velocity_means": np.nanmean(m_sp), "raw": m_sp})

            if not get_raw:
                del results["raw"]

            return results

        else:
            i_d = self.saccade_durations(get_raw=True)["raw"] * self.s_f - 1
            w_v = np.sum(i_d * m_sp) / np.sum(i_d)

            results = dict({"weighted_average_velocity_means": w_v, "raw": m_sp})

            if not get_raw:
                del results["raw"]

            return results

    def saccade_average_velocity_deviations(self, get_raw):
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

        sd_sp = self.saccade_mean_velocities()["velocity_sd"]
        i_d = self.saccade_durations(get_raw=True)["raw"] * self.s_f - 2

        a_sd = np.sqrt(np.sum(i_d * (sd_sp**2)) / np.sum(i_d))

        results = dict({"average_velocity _d": a_sd, "raw": sd_sp})

        if not get_raw:
            del results["raw"]

        return results

    def saccade_peak_velocities(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        a_sp = self.data_set["absolute_speed"]
        p_sp = []

        for _int in _ints:
            p_sp.append(np.max(a_sp[_int[0] : _int[1]]))

        results = dict(
            {
                "velocity_peak_mean": np.nanmean(np.array(p_sp)),
                "velocity_peak_sd": np.nanstd(np.array(p_sp), ddof=1),
                "raw": np.array(p_sp),
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def get_pk_vel_idx(self):
        """


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["saccade_intervals"]
        a_sp = self.data_set["absolute_speed"]
        p_sp_i = []

        for _int in _ints:
            l_idx = np.argmax(a_sp[_int[0] : _int[1]])
            p_sp_i.append(_int[0] + l_idx)

        return np.array(p_sp_i)

    def comp_abs_acc(self):
        """
        Util function to compute absolute acceleration

        Returns
        -------
        ac_v : TYPE
            DESCRIPTION.

        """

        a_sp = self.data_set["absolute_speed"]

        ac_v = np.zeros_like(a_sp)
        ac_v[:-1] = np.abs(a_sp[1:] - a_sp[:-1]) * self.s_f

        return ac_v

    def saccade_mean_acceleration_profiles(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["saccade_intervals"]
        ac_v = self.comp_abs_acc()

        m_ac = []
        sd_ac = []

        for _int in _ints:
            n_ = _int[1] - _int[0] - 1

            if n_ > 0:
                m_ac.append(np.nanmean(ac_v[_int[0] : _int[1] - 1]))
                sd_ac.append(
                    np.nanstd(ac_v[_int[0] : _int[1] - 1], ddof=min(n_ - 1, 1))
                )

            else:
                m_ac.append(np.nan)
                sd_ac.append(np.nan)

        results = dict(
            {
                "acceleration_profile_means": np.array(m_ac),
                "acceleration_profile_sd": np.array(sd_ac),
            }
        )

        return results

    def saccade_mean_accelerations(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["saccade_intervals"]
        p_sp_i = self.get_pk_vel_idx()
        ac_v = self.comp_abs_acc()

        m_ac = []
        sd_ac = []

        for i, _int in enumerate(_ints):
            n_ = p_sp_i[i] - _int[0]

            if n_ > 0:
                m_ac.append(np.nanmean(ac_v[_int[0] : p_sp_i[i]]))
                sd_ac.append(np.nanstd(ac_v[_int[0] : p_sp_i[i]], ddof=min(n_ - 1, 1)))

            else:
                m_ac.append(np.nan)
                sd_ac.append(np.nan)

        results = dict(
            {"acceleration_means": np.array(m_ac), "acceleration_sd": np.array(sd_ac)}
        )

        return results

    def saccade_mean_decelerations(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        _ints = self.segmentation_results["saccade_intervals"]
        p_sp_i = self.get_pk_vel_idx()
        ac_v = self.comp_abs_acc()

        m_dc = []
        sd_dc = []

        for i, _int in enumerate(_ints):
            n_ = _int[1] - p_sp_i[i] - 1

            if n_ > 0:
                m_dc.append(np.nanmean(ac_v[p_sp_i[i] : _int[1] - 1]))
                sd_dc.append(
                    np.nanstd(ac_v[p_sp_i[i] : _int[1] - 1], ddof=min(n_ - 1, 1))
                )

            else:
                m_dc.append(np.nan)
                sd_dc.append(np.nan)

        results = dict(
            {"deceleration_means": np.array(m_dc), "deceleration_sd": np.array(sd_dc)}
        )

        return results

    def acc_average(self, data, weighted, get_raw):
        """
        Util function for acceleration averaging

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        weighted : TYPE
            DESCRIPTION.
        get_raw : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        if not weighted:
            data_s = data[~np.isnan(data)]

            results = dict({"average_means": np.nanmean(data_s), "raw": data})

            if not get_raw:
                del results["raw"]

            return results

        else:
            i_d = self.saccade_durations(get_raw=True)["raw"] * self.s_f - 2

            tmp = i_d * data
            nan_i = np.argwhere(np.isnan(tmp)).T[0]

            w_v = np.sum(np.delete(tmp, nan_i)) / np.sum(np.delete(i_d, nan_i))

            results = dict({"weighted_average_means": w_v, "raw": data})

            if not get_raw:
                del results["raw"]

            return results

    def saccade_average_acceleration_profiles(self, weighted, get_raw):
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

        m_ac = self.saccade_mean_acceleration_profiles()["acceleration_profile_means"]

        results = self.acc_average(m_ac, weighted, get_raw)

        return results

    def saccade_average_acceleration_means(self, weighted, get_raw):
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

        m_ac = self.saccade_mean_accelerations()["acceleration_means"]

        results = self.acc_average(m_ac, weighted, get_raw)

        return results

    def saccade_average_deceleration_means(self, weighted, get_raw):
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

        m_dc = self.saccade_mean_decelerations()["deceleration_means"]

        results = self.acc_average(m_dc, weighted, get_raw)

        return results

    def saccade_peak_accelerations(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        p_sp_i = self.get_pk_vel_idx()

        ac_v = self.comp_abs_acc()
        p_ac = []

        for i, _int in enumerate(_ints):
            n_ = p_sp_i[i] - _int[0]

            if n_ > 0:
                p_ac.append(np.max(ac_v[_int[0] : p_sp_i[i]]))

            else:
                p_ac.append(np.nan)

        p_ac = np.array(p_ac)
        p_ac_s = p_ac[~np.isnan(p_ac)]

        results = dict({"peak_acceleration_mean": np.nanmean(p_ac_s), "raw": p_ac})

        if not get_raw:
            del results["raw"]

        return results

    def saccade_peak_decelerations(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        p_sp_i = self.get_pk_vel_idx()

        ac_v = self.comp_abs_acc()
        p_dc = []

        for i, _int in enumerate(_ints):
            n_ = _int[1] - p_sp_i[i] - 1

            if n_ > 0:
                p_dc.append(np.max(ac_v[p_sp_i[i] : _int[1] - 1]))

            else:
                p_dc.append(np.nan)

        p_dc = np.array(p_dc)
        p_dc_s = p_dc[~np.isnan(p_dc)]

        results = dict({"peak_deceleration_mean": np.nanmean(p_dc_s), "raw": p_dc})

        if not get_raw:
            del results["raw"]

        return results


    def saccade_skewness_exponents(self, get_raw):
        '''
        

        Parameters
        ----------
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        '''
        
        p_sp_i = self.get_pk_vel_idx()
        _ints = self.segmentation_results['saccade_intervals']  
        
        b_i = np.array(
            [_int[0] for _int in _ints]
                )
        
        s_l = np.array(
            [_int[1] - _int[0] for _int in _ints]
                )
   
        skw = (p_sp_i - b_i)/(s_l - 1) 
        results = dict({
            'skewness_exponent_mean': np.nanmean(skw),    
            'skewness_exponent_sd': np.nanstd(skw, ddof = 1), 
            'raw': skw
                })
        
        if not get_raw:
            del results['raw']
         
        return results
    
    
    def saccade_gamma_skewness_exponents(self, get_raw):
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

        _ints = self.segmentation_results["saccade_intervals"]
        a_sp = self.data_set["absolute_speed"]
        skw = []
        
        for _int in _ints:
            l_a_sp = a_sp[_int[0] : _int[1]]
            fit_shape, fit_loc, fit_scale=gamma.fit(l_a_sp)
            skw.append(fit_shape)

         
        results = dict(
            {
                "skewness_exponent_mean": np.nanmean(skw),
                "skewness_exponent_sd": np.nanstd(skw, ddof=1),
                "raw": np.array(skw),
            }
        )

        if not get_raw:
            del results["raw"]

        return results



    def saccade_amplitude_duration_ratios(self, get_raw):
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

        a_s = self.saccade_amplitudes(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]

        r_ = a_s / d_s
        results = dict(
            {"ratio_mean": np.nanmean(r_), "ratio_sd": np.nanstd(r_, ddof=1), "raw": r_}
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_peak_velocity_amplitude_ratios(self, get_raw):
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

        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        a_s = self.saccade_amplitudes(get_raw=True)["raw"]

        r_ = p_v / a_s
        results = dict(
            {"ratio_mean": np.nanmean(r_), "ratio_sd": np.nanstd(r_, ddof=1), "raw": r_}
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_peak_velocity_duration_ratios(self, get_raw):
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

        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]

        r_ = p_v / d_s
        results = dict(
            {"ratio_mean": np.nanmean(r_), "ratio_sd": np.nanstd(r_, ddof=1), "raw": r_}
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_peak_velocity_velocity_ratios(self, get_raw):
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

        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        a_d_r = self.saccade_amplitude_duration_ratios(get_raw=True)["raw"]

        r_ = p_v / a_d_r
        results = dict(
            {"ratio_mean": np.nanmean(r_), "ratio_sd": np.nanstd(r_, ddof=1), "raw": r_}
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_acceleration_deceleration_ratios(self, get_raw):
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

        a_c = self.saccade_peak_accelerations(get_raw=True)["raw"]
        d_c = self.saccade_peak_decelerations(get_raw=True)["raw"]

        r_ = a_c / d_c
        r_s = r_[~np.isnan(r_)]

        results = dict(
            {
                "ratio_mean": np.nanmean(r_s),
                "ratio_sd": np.nanstd(r_s, ddof=1),
                "raw": r_,
            }
        )

        if not get_raw:
            del results["raw"]

        return results

    def saccade_main_sequence(self, get_raw):
        a_s = self.saccade_amplitudes(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]

        l_p_v = np.log(self.saccade_peak_velocities(get_raw=True)["raw"])
        l_a = np.log(a_s)

        coefs_ad = np.polyfit(d_s, a_s, 1)
        coefs_pa = np.polyfit(l_a, l_p_v, 1)

        if self.config["display_results"]:
            plot_main_sequence(d_s, a_s, coefs_ad, l_p_v, l_a, coefs_pa, self.config)

        results = dict(
            {
                "slope_amplitude_duration": coefs_ad[0],
                "slope_log_peak_velocity_log_amplitude": coefs_pa[0],
                "raw_amplitude_duration": np.vstack((a_s, d_s)),
                "raw_log_peak_velocity_log_amplitude": np.vstack((l_p_v, l_a)),
            }
        )

        if not get_raw:
            del results["raw_amplitude_duration"]
            del results["raw_log_peak_velocity_log_amplitude"]

        return results


def saccade_count(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_count()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_count()
        saccade_analysis.verbose()

    return results


def saccade_frequency(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_frequency()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_frequency()
        saccade_analysis.verbose()

    return results


def saccade_frequency_wrt_labels(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_frequency_wrt_labels()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_frequency_wrt_labels()
        saccade_analysis.verbose()

    return results


def saccade_durations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_durations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_durations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_amplitudes(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_amplitudes(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_amplitudes(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_travel_distances(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_travel_distances(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_travel_distances(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_efficiencies(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_efficiencies(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_efficiencies(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_directions(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_directions(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_directions(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_horizontal_deviations(input, **kwargs):
    absolute = kwargs.get("saccade_absolute_horizontal_deviations", True)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_horizontal_deviations(absolute, get_raw)
        input.verbose(
            dict(
                {"get_raw": get_raw, "saccade_absolute_horizontal_deviations": absolute}
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_horizontal_deviations(absolute, get_raw)
        saccade_analysis.verbose(
            dict(
                {"get_raw": get_raw, "saccade_absolute_horizontal_deviations": absolute}
            )
        )

    return results


def saccade_successive_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_successive_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_successive_deviations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_initial_directions(input, **kwargs):
    duration_threshold = kwargs.get("saccade_init_direction_duration_threshold", 0.020)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_initial_directions(duration_threshold, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_direction_duration_threshold": duration_threshold,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_initial_directions(
            duration_threshold, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_direction_duration_threshold": duration_threshold,
                }
            )
        )

    return results


def saccade_initial_deviations(input, **kwargs):
    duration_threshold = kwargs.get("saccade_init_deviation_duration_threshold", 0.020)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_initial_deviations(duration_threshold, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_deviation_duration_threshold": duration_threshold,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_initial_deviations(
            duration_threshold, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_deviation_duration_threshold": duration_threshold,
                }
            )
        )

    return results


def saccade_max_curvatures(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_max_curvatures(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_max_curvatures(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_area_curvatures(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_area_curvatures(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_area_curvatures(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_mean_velocities(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_velocities()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_velocities()
        saccade_analysis.verbose()

    return results


def saccade_average_velocity_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_velocity_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_velocity_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_velocity_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_velocity_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_velocity_means": weighted,
                }
            )
        )

    return results


def saccade_average_velocity_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_velocity_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_velocity_deviations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocities(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocities(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocities(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_mean_acceleration_profiles(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_acceleration_profiles()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_acceleration_profiles()
        saccade_analysis.verbose()

    return results


def saccade_mean_accelerations(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_accelerations()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_accelerations()
        saccade_analysis.verbose()

    return results


def saccade_mean_decelerations(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_decelerations()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_decelerations()
        saccade_analysis.verbose()

    return results


def saccade_average_acceleration_profiles(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_acceleration_profiles", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_acceleration_profiles(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_profiles": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_acceleration_profiles(
            weighted, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_profiles": weighted,
                }
            )
        )

    return results


def saccade_average_acceleration_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_acceleration_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_acceleration_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_acceleration_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_means": weighted,
                }
            )
        )

    return results


def saccade_average_deceleration_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_deceleration_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_deceleration_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_deceleration_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_deceleration_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_deceleration_means": weighted,
                }
            )
        )

    return results


def saccade_peak_accelerations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_accelerations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_accelerations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_decelerations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_decelerations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_decelerations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_skewness_exponents(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_skewness_exponents(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_skewness_exponents(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_gamma_skewness_exponents(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_gamma_skewness_exponents(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_gamma_skewness_exponents(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_amplitude_duration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_amplitude_duration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_amplitude_duration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_amplitude_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_amplitude_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_amplitude_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_duration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_duration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_duration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_velocity_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_velocity_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_velocity_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_acceleration_deceleration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_acceleration_deceleration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_acceleration_deceleration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_main_sequence(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_main_sequence(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_main_sequence(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results
