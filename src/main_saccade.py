# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)


bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_VT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)

## For Saccades
sa = v.SaccadeAnalysis(bs)


sa = v.SaccadeAnalysis(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)



print(sa.saccade_count())
print(sa.saccade_frequency())
print(sa.saccade_frequency_wrt_labels())
print(sa.saccade_durations(get_raw=False))
print(sa.saccade_amplitudes(get_raw=False))
print(sa.saccade_travel_distances(get_raw=False))



print(v.saccade_amplitude_duration_ratios(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800))

print(v.saccade_amplitudes(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800))

print(v.saccade_area_curvatures(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800))

print(v.saccade_average_acceleration_means(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800))

print(v.saccade_average_acceleration_profiles(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800))


print(v.saccade_amplitude_duration_ratios(sa))
print(v.saccade_amplitudes(sa))
print(v.saccade_area_curvatures(sa))
print(v.saccade_average_acceleration_means(sa))
print(v.saccade_average_acceleration_profiles(sa))
print(v.saccade_average_deceleration_means(sa))
print(v.saccade_average_velocity_deviations(sa)) 
print(v.saccade_average_velocity_means(sa))
print(v.saccade_count(sa)) 
print(v.saccade_directions(sa)) 
print(v.saccade_durations(sa))  
print(v.saccade_efficiencies(sa))
print(v.saccade_frequency(sa))
print(v.saccade_frequency_wrt_labels(sa))
print(v.saccade_horizontal_deviations(sa)) 
print(v.saccade_initial_deviations(sa))
print(v.saccade_initial_directions(sa)) 
print(v.saccade_main_sequence(sa))
print(v.saccade_max_curvatures(sa))
print(v.saccade_mean_acceleration_profiles(sa)) 
print(v.saccade_mean_accelerations(sa))
print(v.saccade_mean_decelerations(sa)) 
print(v.saccade_mean_velocities(sa))
print(v.saccade_peak_accelerations(sa))
print(v.saccade_peak_decelerations(sa))
print(v.saccade_peak_velocities(sa))
print(v.saccade_peak_velocity_amplitude_ratios(sa))
print(v.saccade_peak_velocity_duration_ratios(sa))
print(v.saccade_peak_velocity_velocity_ratios(sa)) 
print(v.saccade_skewness_exponents(sa))
print(v.saccade_gamma_skewness_exponents(sa))
print(v.saccade_successive_deviations(sa))
print(v.saccade_travel_distances(sa))
print(v.saccade_main_sequence(sa))
print(v.saccade_average_acceleration_profiles(bs,
                                      get_raw = True, 
                                      saccade_weighted_average_acceleration_profiles = True))

 