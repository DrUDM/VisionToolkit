# -*- coding: utf-8 -*-


import pandas as pd

from vision_toolkit.segmentation.basic_processing import oculomotor_series as ocs


class SignalBased:
    def __init__(self, input_df, **kwargs):
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Signal-Based Analysis...")

        df = pd.read_csv(input_df)

        sampling_frequency = kwargs.get("sampling_frequency", None)
        assert sampling_frequency is not None, "Sampling frequency must be specified"

        config = dict(
            {
                "sampling_frequency": sampling_frequency,
                "distance_projection": kwargs.get("distance_projection"),
                "size_plan_x": kwargs.get("size_plan_x"),
                "size_plan_y": kwargs.get("size_plan_y"),
                "smoothing": kwargs.get("smoothing", "savgol"),
                "distance_type": kwargs.get("distance_type", "euclidian"),
                "display_results": kwargs.get("display", True),
                "verbose": verbose,
            }
        )

        if (
            config["smoothing"] == "moving_average"
            or config["smoothing"] == "speed_moving_average"
        ):
            config.update(
                {"moving_average_window": kwargs.get("moving_average_window", 3)}
            )

        elif config["smoothing"] == "savgol":
            config.update(
                {
                    "savgol_window_length": kwargs.get("savgol_window_length", 5),
                    "savgol_polyorder": kwargs.get("savgol_polyorder", 3),
                }
            )

        basic_processed = ocs.OcculomotorSeries.generate(df, config)

        self.data_set = basic_processed.get_data_set()
        self.config = basic_processed.get_config()

        if verbose:
            print("...Signal-Based Analysis done")

    def verbose(self, add_=None):
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
    def generate(cls, input_df, **kwargs):
        signal_based_analysis = cls(input_df, **kwargs)

        return signal_based_analysis

    def get_config(self):
        return self.config
