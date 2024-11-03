# -*- coding: utf-8 -*-

import copy

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class Smoothing:
    def __init__(self, data_set, config):
        assert config["smoothing"] in [
            None,
            "moving_average",
            "speed_moving_average",
            "savgol",
        ], f"Smoothing method must take value None, 'moving_average' or 'savgol'"

        self.data_set = data_set
        self.config = config

        self.dict_methods = dict(
            {
                None: self.process_no_smoothing,
                "moving_average": self.process_moving_average,
                "speed_moving_average": self.process_speed_moving_average,
                "savgol": self.process_savgol_filtering,
            }
        )

    def process(self):
        smoothed_dataset = self.dict_methods[self.config["smoothing"]]()

        return smoothed_dataset

    def process_no_smoothing(self):
        return self.data_set

    def process_moving_average(self):
        _len = self.config["nb_samples"]
        _w = self.config["moving_average_window"]

        assert _w % 2 == 1, "Moving average window must be odd"

        for _dir in ["x_array", "y_array", "z_array"]:
            conv = np.convolve(self.data_set[_dir], np.ones(_w)) / _w
            smoothed_array = conv[:_len]

            for i in range(_w):
                smoothed_array[i] = smoothed_array[_w]

            self.data_set[_dir] = smoothed_array

        return self.data_set

    def process_speed_moving_average(self):
        _delta_t = 1 / self.config["sampling_frequency"]
        _len = self.config["nb_samples"]
        _w = self.config["moving_average_window"]

        assert _w % 2 == 1, "Moving average window must be odd"

        for _dir in ["x_array", "y_array", "z_array"]:
            smoothed_array = copy.deepcopy(self.data_set[_dir])
            smoothed_speed_vector = self.generate_smoothed_speed_vector(
                self.data_set[_dir], _len, _w, _delta_t
            )

            for i in range(1, self.config["nb_samples"]):
                smoothed_array[i] = (
                    smoothed_array[i - 1] + smoothed_speed_vector[i - 1] * _delta_t
                )

            self.data_set[_dir] = smoothed_array

        return self.data_set

    def generate_smoothed_speed_vector(self, position_array, _len, _w, _delta_t):
        speed_vector = np.zeros(_len)
        _v = int((_w - 1) / 2)

        conv = np.convolve(position_array, np.ones(_v)) / _v

        for i in range(_v, _len - _v):
            speed_vector[i] = (conv[i + _v] - conv[i - 1]) / ((_v + 1) * _delta_t)

        for i in range(_v):
            speed_vector[i] = speed_vector[_v]
            speed_vector[_len - (i + 1)] = speed_vector[_len - (_v + 1)]

        return speed_vector

    def process_savgol_filtering(self):
        _w = self.config["savgol_window_length"]
        _order = self.config["savgol_polyorder"]

        assert _w % 2 == 1, "Savgol window must be odd"

        for _dir in ["x_array", "y_array", "z_array"]:
            smoothed_array = savgol_filter(self.data_set[_dir], _w, _order)
            self.data_set[_dir] = smoothed_array

        return self.data_set
