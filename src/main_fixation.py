# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)

fa = v.FixationAnalysis(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)

print(v.fixation_count(fa))
print(v.fixation_frequency(fa))
print(v.fixation_frequency_wrt_labels(fa))
print(v.fixation_durations(fa))
print(v.fixation_centroids(fa))
print(v.fixation_mean_velocities(fa))
print(v.fixation_average_velocity_means(fa))
print(v.fixation_average_velocity_deviations(fa))
print(v.fixation_drift_displacements(fa))
print(v.fixation_drift_distances(fa))
print(v.fixation_drift_velocities(fa))
print(v.fixation_BCEA(fa))
 
print(fa.fixation_count())
print(fa.fixation_frequency())
print(fa.fixation_frequency_wrt_labels())
print(fa.fixation_durations(get_raw=True))
print(fa.fixation_centroids())
print(fa.fixation_mean_velocities())
print(fa.fixation_average_velocity_means(weighted=True, 
                                         get_raw=True))
print(fa.fixation_average_velocity_deviations(get_raw=True))
print(fa.fixation_drift_displacements(get_raw=True))
print(fa.fixation_drift_distances(get_raw=True))
print(fa.fixation_drift_velocities(get_raw=True))
print(fa.fixation_BCEA(BCEA_probability=0.68, 
                       get_raw=True))
