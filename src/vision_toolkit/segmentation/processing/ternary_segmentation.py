# -*- coding: utf-8 -*-

import copy

import numpy as np
import pandas as pd

from vision_toolkit.segmentation.basic_processing import oculomotor_series as ocs
from vision_toolkit.segmentation.segmentation_algorithms.I_BDT import process_IBDT
from vision_toolkit.segmentation.segmentation_algorithms.I_VDT import process_IVDT
from vision_toolkit.segmentation.segmentation_algorithms.I_VMP import process_IVMP
from vision_toolkit.segmentation.segmentation_algorithms.I_VVT import process_IVVT
from vision_toolkit.utils.velocity_distance_factory import (
    absolute_angular_distance, absolute_euclidian_distance)
from vision_toolkit.visualization.segmentation import display_ternary_segmentation


class TernarySegmentation:
    def __init__(self, input_df, sampling_frequency, segmentation_method, **kwargs):
        """

        Parameters
        ----------
        input_df : TYPE
            DESCRIPTION.
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        df = pd.read_csv(input_df)

        config = dict(
            {
                "sampling_frequency": sampling_frequency,
                "segmentation_method": segmentation_method,
                "distance_projection": kwargs.get("distance_projection"),
                "size_plan_x": kwargs.get("size_plan_x"),
                "size_plan_y": kwargs.get("size_plan_y"),
                "smoothing": kwargs.get("smoothing", "savgol"),
                "distance_type": kwargs.get("distance_type", "angular"),
                "min_int_size": kwargs.get("min_int_size", 2),
                "display_results": kwargs.get("display_results", True),
                "display_segmentation": kwargs.get("display_segmentation", False),
                "display_true_segmentation": kwargs.get(
                    "display_true_segmentation", False
                ),
                "verbose": kwargs.get("verbose", True),
            }
        )

        if (
            config["smoothing"] == "moving_average"
            or config["smoothing"] == "speed_moving_average"
        ):
            config.update(
                {"moving_average_window": kwargs.get("moving_average_window", 5)}
            )

        elif config["smoothing"] == "savgol":
            config.update(
                {
                    "savgol_window_length": kwargs.get("savgol_window_length", 15),
                    "savgol_polyorder": kwargs.get("savgol_polyorder", 3),
                }
            )

        basic_processed = ocs.OcculomotorSeries.generate(df, copy.deepcopy(config))

        self.data_set = basic_processed.get_data_set()
        config = basic_processed.get_config()

        vf_diag = np.linalg.norm(
            np.array([config["size_plan_x"], config["size_plan_y"]])
        )

        if segmentation_method == "I_VVT":
            if config["distance_type"] == "euclidean":
                s_t = vf_diag * 0.5
                p_t = vf_diag * 0.15

                config.update(
                    {
                        "IVVT_saccade_threshold": kwargs.get(
                            "IVVT_saccade_threshold", s_t
                        ),
                        "IVVT_pursuit_threshold": kwargs.get(
                            "IVVT_pursuit_threshold", p_t
                        ),
                    }
                )

            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "IVVT_saccade_threshold": kwargs.get(
                            "IVVT_saccade_threshold", 40
                        ),
                        "IVVT_pursuit_threshold": kwargs.get(
                            "IVVT_pursuit_threshold", 7
                        ),
                    }
                )

        elif segmentation_method == "I_VDT":
            if config["distance_type"] == "euclidean":
                s_t = vf_diag * 0.5
                di_t = 0.02 * vf_diag

                config.update(
                    {
                        "IVDT_saccade_threshold": kwargs.get(
                            "IVDT_saccade_threshold", s_t
                        ),
                        "IVDT_dispersion_threshold": kwargs.get(
                            "IVDT_dispersion_threshold", di_t
                        ),
                        "IVDT_window_duration": kwargs.get(
                            "IVDT_window_duration", 0.040
                        ),
                    }
                )

            if config["distance_type"] == "angular":
                config.update(
                    {
                        "IVDT_saccade_threshold": kwargs.get(
                            "IVDT_saccade_threshold", 40
                        ),
                        "IVDT_dispersion_threshold": kwargs.get(
                            "IVDT_dispersion_threshold", 0.20
                        ),
                        "IVDT_window_duration": kwargs.get(
                            "IVDT_window_duration", 0.040
                        ),
                    }
                )

        elif segmentation_method == "I_VMP":
            if config["distance_type"] == "euclidean":
                s_t = vf_diag * 0.5

                config.update(
                    {
                        "IVMP_saccade_threshold": kwargs.get(
                            "IVMP_saccade_threshold", s_t
                        ),
                        "IVMP_rayleigh_threshold": kwargs.get(
                            "IVMP_rayleigh_threshold", 0.50
                        ),
                        "IVMP_window_duration": kwargs.get(
                            "IVMP_window_duration", 0.050
                        ),
                    }
                )

            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "IVMP_saccade_threshold": kwargs.get(
                            "IVMP_saccade_threshold", 40
                        ),
                        "IVMP_rayleigh_threshold": kwargs.get(
                            "IVMP_rayleigh_threshold", 0.50
                        ),
                        "IVMP_window_duration": kwargs.get(
                            "IVMP_window_duration", 0.050
                        ),
                    }
                )

        elif segmentation_method == "I_BDT":
            if config["distance_type"] == "euclidean":
                fix_t = 0.1 * vf_diag
                pur_t = 0.15 * vf_diag
                sac_t = 1.0 * vf_diag

                config.update(
                    {
                        "IBDT_duration_threshold": kwargs.get(
                            "IBDT_duration_threshold", 0.050
                        ),
                        "IBDT_fixation_threshold": kwargs.get(
                            "IBDT_fixation_threshold", fix_t
                        ),
                        "IBDT_saccade_threshold": kwargs.get(
                            "IBDT_saccade_threshold", sac_t
                        ),
                        "IBDT_pursuit_threshold": kwargs.get(
                            "IBDT_pursuit_threshold", pur_t
                        ),
                        "IBDT_fixation_sd": kwargs.get("IBDT_fixation_sd", 0.01),
                        "IBDT_saccade_sd": kwargs.get("IBDT_saccade_sd", 0.01),
                    }
                )

            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "IBDT_duration_threshold": kwargs.get(
                            "IBDT_duration_threshold", 0.050
                        ),
                        "IBDT_fixation_threshold": kwargs.get(
                            "IBDT_fixation_threshold", 5
                        ),
                        "IBDT_pursuit_threshold": kwargs.get(
                            "IBDT_pursuit_threshold", 8
                        ),
                        "IBDT_saccade_threshold": kwargs.get(
                            "IBDT_saccade_threshold", 50
                        ),
                        "IBDT_fixation_sd": kwargs.get("IBDT_fixation_sd", 0.01),
                        "IBDT_saccade_sd": kwargs.get("IBDT_saccade_sd", 0.01),
                    }
                )

        self.config = config

        self.distances = dict(
            {
                "euclidian": absolute_euclidian_distance,
                "angular": absolute_angular_distance,
            }
        )

        self.dict_methods = dict(
            {
                "I_VVT": process_IVVT,
                "I_VMP": process_IVMP,
                "I_VDT": process_IVDT,
                "I_BDT": process_IBDT,
            }
        )

        self.verbose = config["verbose"]

        self.segmentation_results = None
        self.events = None

    def process(self, labels=True):
        """

        Returns
        -------
        None.

        """
        self.segmentation_results = self.dict_methods[
            self.config["segmentation_method"]
        ](self.data_set, self.config)

        path = "output/figs/ternary_segmentation_{sm}_2D".format(
            sm=self.config["segmentation_method"]
        )

        display_ternary_segmentation(
            self.data_set,
            self.config,
            self.segmentation_results["pursuit_intervals"],
            _color="seagreen",
            path=path,
        )

        self.events = self.get_events(labels)

        if self.config["verbose"]:
            print("\n --- Config used: ---\n")

            for it in self.config.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (30 - len(it)), val=self.config[it]
                    )
                )
            print("\n")

    @classmethod
    def generate(cls, input_df, sampling_frequency, segmentation_method, **kwargs):
        """

        Parameters
        ----------
        input_df : TYPE
            DESCRIPTION.
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        segmentation_analysis : TYPE
            DESCRIPTION.

        """
        segmentation_analysis = cls(
            input_df, sampling_frequency, segmentation_method, **kwargs
        )

        return segmentation_analysis

    def get_events(self, labels):
        """

        Returns
        -------
        cl : TYPE
            DESCRIPTION.

        """
        n_s = self.config["nb_samples"]
        cl = np.array(["None"] * n_s)

        i_fix = self.segmentation_results["is_fixation"]
        wi_fix = np.where(i_fix == True)[0]

        if labels:
            cl[wi_fix] = "fix"
        else:
            cl[wi_fix] = 1

        i_sac = self.segmentation_results["is_saccade"]
        wi_sac = np.where(i_sac == True)[0]

        if labels:
            cl[wi_sac] = "sac"
        else:
            cl[wi_sac] = 2

        i_pur = self.segmentation_results["is_pursuit"]
        wi_pur = np.where(i_pur == True)[0]

        if labels:
            cl[wi_pur] = "pur"
        else:
            cl[wi_pur] = 3

        return cl
