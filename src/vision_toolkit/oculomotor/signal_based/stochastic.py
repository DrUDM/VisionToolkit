# -*- coding: utf-8 -*-
import numpy as np

from vision_toolkit.oculomotor.signal_based.signal_based_base import SignalBased
from vision_toolkit.visualization.oculomotor.stochastic import (plot_dacf, plot_dfa,
                                                        plot_msd)


class StochasticAnalysis(SignalBased):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Stochastic Analysis...")

        super().__init__(input, **kwargs)

        if verbose:
            print("...Stochastic Analysis done")

    def msd_computation(self, x, lag):
        n_s = self.config["nb_samples"]

        msd = np.sum((x[:, lag:] - x[:, :-lag]) ** 2, axis=1)
        msd = msd / (n_s - lag)

        return msd

    def msd_generate(self, _inputs, MSD_min_lag, MSD_max_lag, MSD_nb_lags):
        log_min = np.round(np.log10(MSD_min_lag))
        log_max = np.round(np.log10(MSD_max_lag))

        log_lags = np.linspace(log_min, log_max, MSD_nb_lags)
        lags = np.round(10**log_lags)

        new_lags = list(set(lags))
        new_lags.sort()

        lags_msd = [int(lag) for lag in new_lags]
        msds = np.zeros((2, len(lags_msd)))

        for k, lag in enumerate(lags_msd):
            msds[:, k] = self.msd_computation(_inputs, lag)

        return lags_msd, msds

    def comp_scaling_exponents(self, vals, time_lags, short_i, long_i):
        lg_msds = np.log10(vals)
        lg_lag = np.log10(time_lags)

        coefs = dict()

        for k, _dir in enumerate(["x", "y"]):
            lg_msd = lg_msds[k, :]

            s_coefs = np.polyfit(lg_lag[short_i], lg_msd[short_i], 1)
            l_coefs = np.polyfit(lg_lag[long_i], lg_msd[long_i], 1)

            coefs.update({_dir: [s_coefs, l_coefs]})

        return coefs

    def MSD(
        self,
        short_time_scale,
        long_time_scale,
        MSD_min_lag,
        MSD_max_lag,
        MSD_nb_lags,
        get_raw,
    ):
        if MSD_max_lag is None:
            MSD_max_lag = int(self.config["nb_samples"] / 4)

        if MSD_max_lag > self.config["nb_samples"] / 4:
            MSD_max_lag = int(self.config["nb_samples"] / 4)
            print("'MSD_max_lag' updated to new value: {n_}".format(n_=MSD_max_lag))

        n_s = self.config["nb_samples"]
        pos = np.concatenate(
            (
                self.data_set["x_array"].reshape(1, n_s),
                self.data_set["y_array"].reshape(1, n_s),
            ),
            axis=0,
        )

        lags, msds = self.msd_generate(pos, MSD_min_lag, MSD_max_lag, MSD_nb_lags)
        time_lags = np.array(lags) / self.config["sampling_frequency"]

        short_i = np.where(
            np.logical_and(
                time_lags > short_time_scale[0], time_lags < short_time_scale[1]
            )
        )

        long_i = np.where(
            np.logical_and(
                time_lags > long_time_scale[0],
                time_lags < min(long_time_scale[1], max(time_lags)),
            )
        )

        coefs = self.comp_scaling_exponents(msds, time_lags, short_i, long_i)

        alphas = {
            "x_scaling_exponents": {
                "short_time_scale": coefs["x"][0][0],
                "long_time_scale": coefs["x"][1][0],
            },
            "y_scaling_exponents": {
                "short_time_scale": coefs["y"][0][0],
                "long_time_scale": coefs["y"][1][0],
            },
        }

        if self.config["display_results"]:
            plot_msd(time_lags, msds, coefs, short_time_scale, long_time_scale)

        self.verbose(
            dict(
                {
                    "MSD_min_lag": MSD_min_lag,
                    "MSD_max_lag": MSD_max_lag,
                    "MSD_nb_lags": MSD_nb_lags,
                    "stochastic_analysis_short_time_scale": short_time_scale,
                    "stochastic_analysis_long_time_scale": long_time_scale,
                    "get_raw": get_raw,
                }
            )
        )

        results = dict(
            {
                "scaling_exponents": alphas,
                "lags_MSD": np.array(lags),
                "time_lags_MSD": time_lags,
                "MSDs": msds,
            }
        )

        if not get_raw:
            return alphas

        else:
            return results

    def dacf_computation(self, x, lag, DACF_order):
        n_s = self.config["nb_samples"]

        dacf = np.sum(
            (x[:, lag + DACF_order :] - x[:, lag:-DACF_order])
            * (x[:, DACF_order:-lag] - x[:, : -lag - DACF_order]),
            axis=1,
        )

        dacf /= DACF_order**2 * (n_s - lag - DACF_order)

        return dacf

    def dacf_generate(
        self,
        _inputs,
        DACF_min_lag,
        DACF_max_lag,
        DACF_nb_lags,
        DACF_order,
        DACF_normalization,
    ):
        lags = np.linspace(DACF_min_lag, DACF_max_lag, DACF_nb_lags)

        lags_dacf = [int(lag) for lag in lags]

        dacfs = np.zeros((2, DACF_nb_lags))

        for k, lag in enumerate(lags_dacf):
            dacfs[:, k] = self.dacf_computation(_inputs, lag, DACF_order)

        if DACF_normalization:
            dacfs = dacfs / dacfs[:, 0].reshape(len(_inputs), 1)

        return lags_dacf, dacfs

    def DACF(
        self, DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
    ):
        if DACF_max_lag is None:
            DACF_max_lag = int(self.config["nb_samples"] / 4)

        if DACF_max_lag > self.config["nb_samples"] / 4:
            DACF_max_lag = int(self.config["nb_samples"] / 4)
            print("'DACF_max_lag' updated to new value: {n_}".format(n_=DACF_max_lag))

        n_s = self.config["nb_samples"]
        pos = np.concatenate(
            (
                self.data_set["x_array"].reshape(1, n_s),
                self.data_set["y_array"].reshape(1, n_s),
            ),
            axis=0,
        )

        lags, dacfs = self.dacf_generate(
            pos,
            DACF_min_lag,
            DACF_max_lag,
            DACF_nb_lags,
            DACF_order,
            DACF_normalization,
        )
        time_lags = np.array(lags) / self.config["sampling_frequency"]

        if self.config["display_results"]:
            plot_dacf(DACF_order, time_lags, dacfs)

        self.verbose(
            dict(
                {
                    "DACF_min_lag": DACF_min_lag,
                    "DACF_max_lag": DACF_max_lag,
                    "DACF_nb_lags": DACF_nb_lags,
                    "DACF_order": DACF_order,
                    "DACF_normalization": DACF_normalization,
                }
            )
        )

        results = dict(
            {"lags_DACF": np.array(lags), "time_lags_DACF": time_lags, "DACFs": dacfs}
        )

        return results

    def dfa_find_segments(self, _input, w_l, DFA_overlap):
        n_s = self.config["nb_samples"]

        if DFA_overlap:
            segs = np.array(
                [_input[i : i + w_l] for i in np.arange(0, n_s - w_l, w_l // 2)]
            )

        else:
            segs = _input[: n_s - (n_s % w_l)]
            segs = segs.reshape((_input.shape[0] // w_l, w_l))

        return segs

    def dfa_fractal_trends(self, segs, w_l, DFA_order):
        x = np.arange(w_l)
        coefs = np.polyfit(x, segs.T, DFA_order).T

        trnds = np.array([np.polyval(coefs[j], x) for j in np.arange(len(segs))])

        return trnds

    def dfa_fractal_fluctuation(self, segs_f, segs_r, trnds_f, trnds_r):
        detrnds_f = segs_f - trnds_f
        detrnds_r = segs_r - trnds_r

        fluct = np.concatenate(
            (
                np.sum(detrnds_f**2, axis=1) / detrnds_f.shape[1],
                np.sum(detrnds_r**2, axis=1) / detrnds_r.shape[1],
            )
        )

        fluct = np.power(np.sum(fluct) / len(fluct), 1 / 2)

        return fluct

    def dfa_fractal(
        self,
        short_time_scale,
        long_time_scale,
        DFA_min_lag,
        DFA_max_lag,
        DFA_nb_lags,
        DFA_order,
        DFA_overlap,
    ):
        lags = np.zeros(DFA_nb_lags)

        fluct_s = np.zeros((2, DFA_nb_lags))
        h_exps = dict({})

        for k, _dir in enumerate(["x_array", "y_array"]):
            _input = self.data_set[_dir]
            fluct = np.zeros(DFA_nb_lags)

            for i, w_l in enumerate(
                10
                ** np.linspace(
                    np.log10(DFA_min_lag), np.log10(DFA_max_lag), DFA_nb_lags
                )
            ):
                i_w_l = int(w_l)
                lags[i] = i_w_l

                segs = dict({})
                trnds = dict({})

                segs.update(
                    {"forward": self.dfa_find_segments(_input, i_w_l, DFA_overlap)}
                )
                segs.update(
                    {
                        "reverse": self.dfa_find_segments(
                            np.flip(_input), i_w_l, DFA_overlap
                        )
                    }
                )

                for key in segs.keys():
                    trnds.update(
                        {key: self.dfa_fractal_trends(segs[key], i_w_l, DFA_order)}
                    )

                fluct[i] = self.dfa_fractal_fluctuation(
                    segs["forward"], segs["reverse"], trnds["forward"], trnds["reverse"]
                )

            fluct_s[k, :] = fluct
            h_exps.update({_dir[0]: np.polyfit(np.log10(lags), np.log10(fluct), 1)[0]})

        time_lags = lags / self.config["sampling_frequency"]

        short_i = np.where(
            np.logical_and(
                time_lags > short_time_scale[0], time_lags < short_time_scale[1]
            )
        )

        long_i = np.where(
            np.logical_and(
                time_lags > long_time_scale[0],
                time_lags < min(long_time_scale[1], max(time_lags)),
            )
        )

        coefs = self.comp_scaling_exponents(fluct_s, time_lags, short_i, long_i)

        results = dict(
            {"lags_DFA": lags, "time_lags_DFA": time_lags, "fluctuations_DFA": fluct_s}
        )

        return results, h_exps, coefs

    def DFA(
        self,
        short_time_scale,
        long_time_scale,
        DFA_min_lag,
        DFA_max_lag,
        DFA_nb_lags,
        DFA_order,
        DFA_overlap,
        get_raw,
    ):
        if DFA_max_lag is None:
            DFA_max_lag = int(self.config["nb_samples"] / 4)

        if DFA_max_lag > self.config["nb_samples"] / 4:
            DFA_max_lag = int(self.config["nb_samples"] / 4)
            print("'DFA_max_lag' updated to new value: {n_}".format(n_=DFA_max_lag))

        results, h_exps, coefs = self.dfa_fractal(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
        )

        alphas = {
            "x_scaling_exponents": {
                "short_time_scale": coefs["x"][0][0],
                "long_time_scale": coefs["x"][1][0],
            },
            "y_scaling_exponents": {
                "short_time_scale": coefs["y"][0][0],
                "long_time_scale": coefs["y"][1][0],
            },
        }

        if self.config["display_results"]:
            plot_dfa(
                results["time_lags_DFA"],
                results["fluctuations_DFA"],
                coefs,
                short_time_scale,
                long_time_scale,
            )

        self.verbose(
            dict(
                {
                    "DFA_min_lag": DFA_min_lag,
                    "DFA_max_lag": DFA_max_lag,
                    "DFA_nb_lags": DFA_nb_lags,
                    "DFA_order": DFA_order,
                    "DFA_overlap": DFA_overlap,
                    "stochastic_analysis_short_time_scale": short_time_scale,
                    "stochastic_analysis_long_time_scale": long_time_scale,
                    "get_raw": get_raw,
                }
            )
        )

        if not get_raw:
            return dict({"hurst_exponents": h_exps, "local_scaling_exponents": alphas})

        else:
            return results.update(
                {"hurst_exponents": h_exps, "local_scaling_exponents": alphas}
            )


def MSD(input, **kwargs):
    short_time_scale = kwargs.get(
        "stochastic_analysis_short_time_scale", [0.012, 0.120]
    )
    long_time_scale = kwargs.get("stochastic_analysis_long_time_scale", [0.120, 1.200])

    MSD_min_lag = kwargs.get("MSD_min_lag", 1)
    MSD_max_lag = kwargs.get("MSD_max_lag", None)
    MSD_nb_lags = kwargs.get("MSD_nb_lags", 50)

    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, StochasticAnalysis):
        results = input.MSD(
            short_time_scale,
            long_time_scale,
            MSD_min_lag,
            MSD_max_lag,
            MSD_nb_lags,
            get_raw,
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)

        results = stochastic_analysis.MSD(
            short_time_scale,
            long_time_scale,
            MSD_min_lag,
            MSD_max_lag,
            MSD_nb_lags,
            get_raw,
        )

    return results


def DACF(input, **kwargs):
    DACF_min_lag = kwargs.get("DACF_min_lag", 1)
    DACF_max_lag = kwargs.get("DACF_max_lag", None)
    DACF_nb_lags = kwargs.get("DACF_nb_lags", 50)

    DACF_order = kwargs.get("DACF_order", 20)
    DACF_normalization = kwargs.get("DACF_normalization", True)

    if isinstance(input, StochasticAnalysis):
        results = input.DACF(
            DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)
        results = stochastic_analysis.DACF(
            DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
        )

    return results


def DFA(input, **kwargs):
    short_time_scale = kwargs.get(
        "stochastic_analysis_short_time_scale", [0.012, 0.120]
    )
    long_time_scale = kwargs.get("stochastic_analysis_long_time_scale", [0.120, 1.200])

    DFA_order = kwargs.get("DFA_order", 2)
    DFA_overlap = kwargs.get("DFA_overlap", False)

    DFA_min_lag = kwargs.get("DFA_min_lag", DFA_order + 2)
    DFA_max_lag = kwargs.get("DFA_max_lag", None)
    DFA_nb_lags = kwargs.get("DFA_nb_lags", 50)

    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, StochasticAnalysis):
        results = input.DFA(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
            get_raw,
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)
        results = stochastic_analysis.DFA(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
            get_raw,
        )

    return results
