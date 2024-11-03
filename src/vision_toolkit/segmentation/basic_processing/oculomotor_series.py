# -*- coding: utf-8 -*-


import numpy as np

from vision_toolkit.segmentation.basic_processing import basic_processing as bp


class OcculomotorSeries:
    def __init__(self, data_set, config):
        self.data_set = data_set
        self.config = config

    @classmethod
    def generate(cls, df, config):
        """

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if config["distance_projection"] == None:
            d_p = 1000

        else:
            d_p = config["distance_projection"]

        x_a = df["gazeX"].values
        y_a = df["gazeY"].values

        if "gazeZ" in df.columns:
            z_a = df["gazeZ"].values

        else:
            z_a = np.ones(len(df)) * d_p

        if "status" in df.columns:
            stat_a = df["status"].values

        else:
            stat_a = np.ones(len(df))

        if config["size_plan_x"] == None:
            config["size_plan_x"] = max(x_a) + 0.001

        # if not config['size_plan_x'] > max(x_a):
        #    raise ValueError("size_plan_x must be greater than horizontal maximum value")

        if config["size_plan_y"] == None:
            config["size_plan_y"] = max(y_a) + 0.001

        # if not config['size_plan_y'] > max(y_a):
        #    raise ValueError("size_plan_y must be greater than vertical maximum value")

        config.update({"nb_samples": len(df)})

        data_set = dict(
            {"x_array": x_a, "y_array": y_a, "z_array": z_a, "status": stat_a}
        )

        basic_processed = bp.Basic_Processing.generate(data_set, config)

        return cls(basic_processed.get_data_set(), config)

    def get_data_set(self):
        return self.data_set

    def get_config(self):
        return self.config
