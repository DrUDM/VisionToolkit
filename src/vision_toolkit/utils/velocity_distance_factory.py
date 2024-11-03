# -*- coding: utf-8 -*-

import numpy as np


def process_unitary_gaze_vectors(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    unitary_gaze_vectors : TYPE
        DESCRIPTION.

    """
    x = data_set["x_array"] - config["size_plan_x"] / 2
    y = data_set["y_array"] - config["size_plan_y"] / 2
    z = data_set["z_array"]

    gaze_vectors = np.concatenate(
        (
            x.reshape(1, config["nb_samples"]),
            y.reshape(1, config["nb_samples"]),
            z.reshape(1, config["nb_samples"]),
        ),
        axis=0,
    )

    unitary_gaze_vectors = gaze_vectors / np.linalg.norm(gaze_vectors, axis=0)

    return unitary_gaze_vectors


def process_angular_coord(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    theta_coord : TYPE
        DESCRIPTION.

    """
    theta_x = (180 / np.pi) * np.arctan(
        (data_set["x_array"] - (config["size_plan_x"] / 2)) / data_set["z_array"]
    )
    theta_y = (180 / np.pi) * np.arctan(
        (data_set["y_array"] - (config["size_plan_y"] / 2)) / data_set["z_array"]
    )

    theta_coord = np.concatenate(
        (
            theta_x.reshape(1, config["nb_samples"]),
            theta_y.reshape(1, config["nb_samples"]),
        ),
        axis=0,
    )

    return theta_coord


def process_angular_absolute_speeds(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    absolute_speeds : TYPE
        DESCRIPTION.

    """
    unitary_gaze_vectors = data_set["unitary_gaze_vectors"]

    dot_ = np.array(
        [
            unitary_gaze_vectors[:, i - 1] @ unitary_gaze_vectors[:, i]
            for i in range(1, config["nb_samples"])
        ]
    )

    absolute_angular_distances_rad = np.arccos(dot_)
    absolute_angular_distances_deg = absolute_angular_distances_rad / (np.pi) * 180

    absolute_speeds = np.zeros(config["nb_samples"])
    absolute_speeds[:-1] = absolute_angular_distances_deg * config["sampling_frequency"]

    absolute_speeds[-1] = absolute_speeds[-2]

    return absolute_speeds


def process_euclidian_absolute_speeds(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    absolute_speeds : TYPE
        DESCRIPTION.

    """
    nb_samples = config["nb_samples"]

    gaze_points = np.concatenate(
        (
            data_set["x_array"].reshape(1, nb_samples),
            data_set["y_array"].reshape(1, nb_samples),
            data_set["z_array"].reshape(1, nb_samples),
        ),
        axis=0,
    )

    absolute_speeds = np.zeros(nb_samples)
    absolute_speeds[:-1] = (
        np.linalg.norm(gaze_points[:, 1:] - gaze_points[:, :-1], axis=0)
        * config["sampling_frequency"]
    )

    return absolute_speeds


def process_speed_components(data_set, config):
    """

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    sp : TYPE
        DESCRIPTION.

    """
    nb_s = config["nb_samples"]

    g_p = np.concatenate(
        (
            data_set["x_array"].reshape(1, nb_s),
            data_set["y_array"].reshape(1, nb_s),
            data_set["z_array"].reshape(1, nb_s),
        ),
        axis=0,
    )

    sp = np.zeros_like(g_p)
    sp[:, :-1] = (g_p[:, 1:] - g_p[:, :-1]) * config["sampling_frequency"]

    return sp


def absolute_angular_distance(gaze_vect_1, gaze_vect_2):
    """


    Parameters
    ----------
    gaze_vect_1 : TYPE
        DESCRIPTION.
    gaze_vect_2 : TYPE
        DESCRIPTION.
    rad : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    absolute_angular_distance_rad = np.arccos(
        gaze_vect_1
        @ gaze_vect_2
        / (np.linalg.norm(gaze_vect_1) * np.linalg.norm(gaze_vect_2))
    )
    absolute_angular_distance_deg = absolute_angular_distance_rad / (np.pi) * 180

    return absolute_angular_distance_deg


def absolute_euclidian_distance(gaze_point_1, gaze_point_2):
    """

    Parameters
    ----------
    gaze_point_1 : TYPE
        DESCRIPTION.
    gaze_point_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    distance : TYPE
        DESCRIPTION.

    """
    distance = np.linalg.norm(gaze_point_1 - gaze_point_2)

    return distance
