# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_predefined_reference_image)


def process_predefined(values, config, ref_image=None):
    """


    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pos_ = values[0:2]
    dur_ = values[2]

    aoi_coords = np.array(config["AoI_predefined_coordinates"])
    all_ = config["AoI_predefined_all"]
    x_ = pos_[0]
    y_ = pos_[1]

    centers_ = dict({})
    clus_ = dict({})
    n_aoi_coords = []
    seq_ = np.array([np.nan] * pos_.shape[1])
    i_ = 0

    for i, aoi_coord in enumerate(aoi_coords):
        x_max, x_min = aoi_coord[1, 0], aoi_coord[0, 0]
        y_max, y_min = aoi_coord[1, 1], aoi_coord[0, 1]
        idx = (x_ <= x_max) * (x_ >= x_min) * (y_ <= y_max) * (y_ >= y_min)
        if all_:
            seq_[idx] = i_
            n_aoi_coords.append(aoi_coord)
            centers_.update(
                {chr(i_ + 65): np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])}
            )
            i_ += 1
        else:
            if np.sum(idx) >= 2:
                seq_[idx] = i_
                n_aoi_coords.append(aoi_coord)
                centers_.update(
                    {chr(i_ + 65): np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])}
                )
                i_ += 1
    ## Delete fixation not belonging to any AoI
    nan_idx = np.argwhere(np.isnan(seq_)).flatten()
    seq_ = seq_[~np.isnan(seq_)]
    ## Update clustering
    for k in centers_.keys():
        j = ord(k) - 65
        vals_ = np.argwhere(seq_ == j).flatten()
        clus_.update({k: vals_})

    dur_ = np.delete(dur_, nan_idx)
    pos_ = np.vstack((np.delete(x_, nan_idx), np.delete(y_, nan_idx)))
    config.update({"AoI_coordinates": n_aoi_coords})
    ## Compute final AoI sequence
    seq_, seq_dur = compute_aoi_sequence(seq_.astype(int), dur_, config)
    if config["display_AoI_identification"]:
        ## Plot clusters
        if ref_image is None:
            display_aoi_identification(pos_, clus_, config)
        else:
            display_aoi_predefined_reference_image(pos_, clus_, config, ref_image)
    results = dict(
        {
            "AoI_sequence": seq_,
            "AoI_durations": seq_dur,
            "centers": centers_,
            "clustered_fixations": clus_,
        }
    )

    return results


def define(image):
    ref_point = []
    l_ref_point = []

    if isinstance(image, str):
        image = cv2.imread(image)

    y_init, x_init, _ = image.shape
    x_r = x_init / 800
    y_r = y_init / 600
    image = cv2.resize(image, (800, 600))
    clone = image.copy()
    cv2.namedWindow("image")

    def shape_selection(event, x, y, flags, param):
        ## grab references to the global variables
        global l_ref_point, crop

        ## If the left mouse button was clicked, record the starting
        ## (x, y) coordinates and indicate that cropping is being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            l_ref_point = [[x, y]]

        ## Check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            ## Record the ending (x, y) coordinates and indicate that
            ## the cropping operation is finished
            l_ref_point.append([x, y])

            # draw a rectangle around the region of interest
            cv2.rectangle(image, l_ref_point[0], l_ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            ref_point.append(
                [
                    [l_ref_point[0][0] * x_r, l_ref_point[0][1] * y_r],
                    [l_ref_point[1][0] * x_r, l_ref_point[1][1] * y_r],
                ]
            )

    cv2.setMouseCallback("image", shape_selection)
    ## Keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        if key == ord("r"):
            image = clone.copy()
            ref_point = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    print(ref_point)

    # close all open windows
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return ref_point


if __name__ == "__main__":
    path = "dataset/2023_09_11_10_41_15/"
    ref_im = cv2.imread(path + "image_ref.jpg")

    define(ref_im)
