# -*- coding: utf-8 -*-


from vision_toolkit.utils import smoothing as smg
from vision_toolkit.utils.velocity_distance_factory import (
    process_angular_absolute_speeds, process_angular_coord,
    process_euclidian_absolute_speeds, process_unitary_gaze_vectors)


class Basic_Processing:
    def __init__(self, data_set, config):
        self.data_set = data_set
        self.config = config

    @classmethod
    def generate(cls, data_set, config):
        """

        Parameters
        ----------
        data_set : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        smoothing = smg.Smoothing(data_set, config)
        data_set = smoothing.process()

        print(config)
        if config["distance_type"] == "euclidean":
            data_set.update(
                {"absolute_speed": process_euclidian_absolute_speeds(data_set, config)}
            )

        elif config["distance_type"] == "angular":
            data_set.update({"theta_coord": process_angular_coord(data_set, config)})

            # print('Theta coordinates generated')
            data_set.update(
                {"unitary_gaze_vectors": process_unitary_gaze_vectors(data_set, config)}
            )

            # print('Unitary gaze vectors generated')
            data_set.update(
                {"absolute_speed": process_angular_absolute_speeds(data_set, config)}
            )
            # print('Absolute speeds generated')

        return cls(data_set, config)

    def get_data_set(self):
        return self.data_set

    def get_congig(self):
        return self.config
