# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.oculomotor.segmentation_based.fixation import FixationAnalysis
from vision_toolkit.visualization.scanpath.basic_representation import (
    display_scanpath, display_scanpath_reference_image)


class Scanpath:
    def __init__(self, input, gaze_df=None, ref_image=None, **kwargs):
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

        if isinstance(input, dict):
            self.generate_from_dict(input)

        else:
            verbose = kwargs.get("verbose", True)

            if verbose:
                print("Processing Scanpath...\n")

            self.fixation_analysis = FixationAnalysis(input, **kwargs)
            self.config = self.fixation_analysis.config

            if gaze_df is None:
                c_ = self.fixation_analysis.fixation_centroids()["centroids"].T
                d_ = self.fixation_analysis.fixation_durations(get_raw=True)["raw"]
            else:
                d_, c_x, c_y = [], [], []
                self.config.update(
                    {
                        "size_plan_x": kwargs.get(
                            "size_plan_x_gaze", self.config["size_plan_x"]
                        ),
                        "size_plan_y": kwargs.get(
                            "size_plan_y_gaze", self.config["size_plan_y"]
                        ),
                    }
                )

                fix_i = np.array(
                    self.fixation_analysis.segmentation_results["fixation_intervals"]
                )
                gaze_x = gaze_df["ref_gazeX"].values
                gaze_y = gaze_df["ref_gazeY"].values

                if "status" in gaze_df.columns:
                    gaze_status = gaze_df["status"].values
                else:
                    gaze_status = np.ones(len(gaze_df))

                gaze_status[gaze_x < 0] = 0
                gaze_status[gaze_x > self.config["size_plan_x"]] = 0
                gaze_status[gaze_y < 0] = 0
                gaze_status[gaze_y > self.config["size_plan_y"]] = 0

                s_f = self.fixation_analysis.config["sampling_frequency"]
                for fix in fix_i:
                    coord_x = gaze_x[fix[0] : fix[1] + 1]
                    coord_y = gaze_y[fix[0] : fix[1] + 1]
                    status = gaze_status[fix[0] : fix[1] + 1]
                    if np.sum(status) > 0:
                        c_x.append(np.nanmean(coord_x[status > 0]))
                        c_y.append(np.nanmean(coord_y[status > 0]))
                        d_.append((fix[1] - fix[0] + 1) / s_f)
                c_ = np.stack((c_x, c_y), axis=0)

            self.values = np.stack((c_[0], c_[1], d_), axis=0)
            self.config.update(
                {
                    "verbose": verbose,
                    "display_scanpath": kwargs.get("display_scanpath", False),
                    "display_scanpath_path": kwargs.get("display_scanpath_path", None),
                }
            )

        if self.config["display_scanpath"]:
            if ref_image is None:
                display_scanpath(self.values, self.config)
            else:
                display_scanpath_reference_image(self.values, self.config, ref_image)

        self.verbose()

        if self.config["verbose"]:
            print("...Scanpath done\n")

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
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        sequence_analysis : TYPE
            DESCRIPTION.

        """

        sequence_analysis = cls(input, **kwargs)
        return sequence_analysis

    def generate_from_dict(self, input):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.values = input["values"]
        self.config = input["config"]

        self.fixation_analysis = input.get("fixation_analysis", None)
