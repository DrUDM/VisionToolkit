# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 

v.processing_rim(gaze_data = 'input_rim/_gaze.csv',
                 time_stamps = 'input_rim/_world_timestamps.csv',
                 reference_image = 'input_rim/_reference_image.jpg',
                 world_camera = 'input_rim/_worldCamera.mp4',
                 output_name = '_gaze',
                 output_dir = 'mappedGazeOutput')