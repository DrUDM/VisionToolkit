# -*- coding: utf-8 -*-

import copy

import numpy as np
import pandas as pd

from vision_toolkit.segmentation.basic_processing import oculomotor_series as ocs
from vision_toolkit.segmentation.segmentation_algorithms.I_2MC import process_I2MC
from vision_toolkit.segmentation.segmentation_algorithms.I_DeT import process_IDeT
from vision_toolkit.segmentation.segmentation_algorithms.I_DiT import process_IDiT
from vision_toolkit.segmentation.segmentation_algorithms.I_HMM import process_IHMM
from vision_toolkit.segmentation.segmentation_algorithms.I_KF import process_IKF
from vision_toolkit.segmentation.segmentation_algorithms.I_MST import process_IMST
from vision_toolkit.segmentation.segmentation_algorithms.I_VT import process_IVT
from vision_toolkit.segmentation.segmentation_algorithms.I_RF import process_IRF
from vision_toolkit.utils.velocity_distance_factory import (
    absolute_angular_distance, absolute_euclidian_distance)
from vision_toolkit.visualization.segmentation import display_binary_segmentation


class BinarySegmentation:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
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
        if isinstance(input, pd.DataFrame):
            df = input
        else:
            df = pd.read_csv(input)

        sampling_frequency = kwargs.get("sampling_frequency", None)
        segmentation_method = kwargs.get("segmentation_method", "I_HMM")
        assert sampling_frequency is not None, "Sampling frequency must be specified"

        config = dict(
            {
                "sampling_frequency": sampling_frequency,
                "segmentation_method": segmentation_method,
                "distance_projection": kwargs.get("distance_projection"),
                "size_plan_x": kwargs.get("size_plan_x"),
                "size_plan_y": kwargs.get("size_plan_y"),
                "smoothing": kwargs.get("smoothing", "savgol"),
                "distance_type": kwargs.get("distance_type", "angular"),
                "min_fix_duration": kwargs.get("min_fix_duration", 7e-2),
                "max_fix_duration": kwargs.get("max_fix_duration", 2.0),
                "min_sac_duration": kwargs.get("min_sac_duration", 1.5e-2),
                "status_threshold": kwargs.get("status_threshold", 0.5),
                "display_segmentation": kwargs.get("display_segmentation", False),
                "display_segmentation_path": kwargs.get(
                    "display_segmentation_path", None
                ),
                "display_results": kwargs.get("display_results", True),
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
                    "savgol_window_length": kwargs.get("savgol_window_length", 31),
                    "savgol_polyorder": kwargs.get("savgol_polyorder", 3),
                }
            )
        basic_processed = ocs.OcculomotorSeries.generate(df, copy.deepcopy(config))
        self.data_set = basic_processed.get_data_set()
        config = basic_processed.get_config()
        vf_diag = np.linalg.norm(
            np.array([config["size_plan_x"], config["size_plan_y"]])
        )

        if segmentation_method == "I_VT":
            if config["distance_type"] == "euclidean":
                v_t = vf_diag * 0.2
                config.update(
                    {
                        "IVT_velocity_threshold": kwargs.get(
                            "IVT_velocity_threshold", v_t
                        ),
                    }
                )
            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "IVT_velocity_threshold": kwargs.get(
                            "IVT_velocity_threshold", 50
                        ),
                    }
                )

        elif segmentation_method == "I_DiT":
            if config["distance_type"] == "euclidean":
                di_t = 0.01 * vf_diag
                config.update(
                    {
                        "IDiT_window_duration": kwargs.get(
                            "IDiT_window_duration", 0.040
                        ),
                        "IDiT_dispersion_threshold": kwargs.get(
                            "IDiT_dispersion_threshold", di_t
                        ),
                    }
                )
            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "IDiT_window_duration": kwargs.get(
                            "IDiT_window_duration", 0.040
                        ),
                        "IDiT_dispersion_threshold": kwargs.get(
                            "IDiT_dispersion_threshold", 0.3
                        ),
                    }
                )

        elif segmentation_method == "I_DeT":
            ## To accelerate computation, duration threshold should be equal to 5 time stamps
            du_t = 5 / sampling_frequency
            if config["distance_type"] == "euclidean":
                ## The default density threshold is thus defined from the sampling frequency
                de_t = vf_diag / sampling_frequency
                config.update(
                    {
                        "IDeT_duration_threshold": kwargs.get(
                            "IDeT_duration_threshold", du_t
                        ),
                        "IDeT_density_threshold": kwargs.get(
                            "IDeT_density_threshold", de_t
                        ),
                    }
                )
            elif config["distance_type"] == "angular":
                ## The default density threshold is thus defined from the sampling frequency
                de_t = 30 / sampling_frequency
                config.update(
                    {
                        "IDeT_duration_threshold": kwargs.get(
                            "IDeT_duration_threshold", du_t
                        ),
                        "IDeT_density_threshold": kwargs.get(
                            "IDeT_density_threshold", de_t
                        ),
                    }
                )

        elif segmentation_method == "I_HMM":
            i_l = 0.001 * vf_diag
            i_h = 10.0 * vf_diag
            i_v = 100 * vf_diag**2
            config.update(
                {
                    "HMM_init_low_velocity": kwargs.get("HMM_init_low_velocity", i_l),
                    "HMM_init_high_velocity": kwargs.get("HMM_init_high_velocity", i_h),
                    "HMM_init_variance": kwargs.get("HMM_init_variance", i_v),
                    "HMM_nb_iters": kwargs.get("HMM_nb_iters", 10),
                }
            )

        elif segmentation_method == "I_KF":
            si_1 = (4 * vf_diag) ** 2
            si_2 = (5 * vf_diag) ** 2
            c_t = (1 * vf_diag) ** 2
            config.update(
                {
                    "distance_type": "euclidean",
                    "IKF_sigma_1": kwargs.get("IKF_sigma_1", si_1),
                    "IKF_sigma_2": kwargs.get("IKF_sigma_2", si_2),
                    "IKF_chi2_window": kwargs.get("IKF_chi2_window", 10),
                    "IKF_chi2_threshold": kwargs.get("IKF_chi2_threshold", c_t),
                }
            )

        elif segmentation_method == "I_MST":
            du_t = 40 / sampling_frequency
            s_ = 0.001 * vf_diag
            config.update(
                {
                    "distance_type": "euclidean",
                    "IMST_window_duration": kwargs.get("IMST_window_duration", du_t),
                    "IMST_distance_threshold": kwargs.get(
                        "IMST_distance_threshold", s_
                    ),
                }
            )

        elif segmentation_method == "I_2MC":
            if config["distance_type"] == "euclidean":
                di_t = 0.025 * vf_diag
                config.update(
                    {
                        "I2MC_window_duration": kwargs.get(
                            "I2MC_window_duration", 0.300
                        ),
                        "I2MC_moving_threshold": kwargs.get(
                            "I2MC_moving_threshold", 0.020
                        ),
                        "I2MC_merging_duration_threshold": kwargs.get(
                            "I2MC_merging_duration_threshold", 0.020
                        ),
                        "I2MC_merging_distance_threshold": kwargs.get(
                            "I2MC_merging_distance_threshold", di_t
                        ),
                    }
                )

            elif config["distance_type"] == "angular":
                config.update(
                    {
                        "I2MC_window_duration": kwargs.get(
                            "I2MC_window_duration", 0.300
                        ),
                        "I2MC_moving_threshold": kwargs.get(
                            "I2MC_moving_threshold", 0.020
                        ),
                        "I2MC_merging_duration_threshold": kwargs.get(
                            "I2MC_merging_duration_threshold", 0.020
                        ),
                        "I2MC_merging_distance_threshold": kwargs.get(
                            "I2MC_merging_distance_threshold", 0.5
                        ),
                    }
                )

        self.config = config
        self.distances = dict(
            {
                "euclidean": absolute_euclidian_distance,
                "angular": absolute_angular_distance,
            }
        )

        self.dict_methods = dict(
            {
                "I_VT": process_IVT,
                "I_DiT": process_IDiT,
                "I_DeT": process_IDeT,
                "I_KF": process_IKF,
                "I_MST": process_IMST,
                "I_HMM": process_IHMM,
                "I_2MC": process_I2MC,
                "I_RF": process_IRF
            }
        )

        self.segmentation_results = None
        self.process()

    def new_config(self, new_config):
        self.config = new_config

    def new_dataset(self, new_dataset):
        self.data_set = new_dataset

    def new_segmentation_results(self, new_segmentation_results):
        self.segmentation_results = new_segmentation_results

    def process(self):
        """

        Returns
        -------
        None.

        """
        self.segmentation_results = self.dict_methods[
            self.config["segmentation_method"]
        ](self.data_set, self.config)
        if self.config["display_segmentation"]:
            display_binary_segmentation(
                self.data_set,
                self.config,
                self.segmentation_results["fixation_intervals"],
                _color="darkblue",
            )
        self.verbose()

    def display_fixations(self):
        display_binary_segmentation(
            self.data_set,
            self.config,
            self.segmentation_results["fixation_intervals"],
            _color="darkblue",
        )

    def verbose(self, add_=None):
        """


        Parameters
        ----------
        add_ : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

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

    @classmethod
    def generate(cls, input, **kwargs):
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
        segmentation_analysis = cls(input, **kwargs)
        return segmentation_analysis
