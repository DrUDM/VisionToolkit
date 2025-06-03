# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)
 

sb = v.periodogram(root + 'data_1.csv',
                   sampling_frequency=256,
                   periodogram_data_type = 'position', 
                   distance_type = 'angular',    
                   size_plan_x = 1200,
                   size_plan_y = 800)

sb = v.periodogram(root + 'data_1.csv',
                   sampling_frequency=256,
                   periodogram_data_type = 'velocity', 
                   distance_type = 'angular',    
                   size_plan_x = 1200,
                   size_plan_y = 800)


sb = v.welch_periodogram(root + 'data_1.csv',
                   sampling_frequency=256,
                   periodogram_data_type = 'position', 
                   distance_type = 'angular',    
                   size_plan_x = 1200,
                   size_plan_y = 800)

sb = v.welch_periodogram(root + 'data_1.csv',
                   sampling_frequency=256,
                   periodogram_data_type = 'velocity', 
                   distance_type = 'angular',   
                   size_plan_x = 1200,
                   size_plan_y = 800)



fa = v.FrequencyAnalysis(root + 'data_1.csv',
                   sampling_frequency=256,
                   periodogram_data_type = 'velocity', 
                   distance_type = 'angular',   
                   size_plan_x = 1200,
                   size_plan_y = 80)

print(fa.periodogram(periodogram_data_type='velocity'))
print(fa.welch_periodogram(periodogram_data_type='velocity',
                           Welch_samples_per_segment=256))


print(v.periodogram(fa, periodogram_data_type='velocity'))
print(v.periodogram(fa, periodogram_data_type='position'))
print(v.welch_periodogram(fa, periodogram_data_type='velocity'))
print(v.welch_periodogram(fa, periodogram_data_type='position'))



cfa = v.CrossFrequencyAnalysis([root + 'data_1.csv', root + 'data_2.csv'], 
                                      sampling_frequency = 256, 
                                      cross_data_type = 'velocity', 
                                      distance_type = 'angular',   
                                      display_segmentation = True,
                                      size_plan_x = 1200,
                                      size_plan_y = 800)


 
print(cfa.cross_spectral_density(cross_data_type='velocity'))
print(v.cross_spectral_density([root + 'data_1.csv', root + 'data_2.csv'], 
                              sampling_frequency=256, 
                              cross_data_type='velocity', 
                              distance_type='angular',   
                              display_segmentation=True,
                              size_plan_x=1200,
                              size_plan_y=800))
print(v.welch_cross_spectral_density([root + 'data_1.csv', root + 'data_2.csv'], 
                                    sampling_frequency=256, 
                                    cross_data_type='velocity', 
                                    distance_type='angular',   
                                    display_segmentation=True,
                                    size_plan_x=1200,
                                    size_plan_y=800))

print(v.signal_coherency([root + 'data_1.csv', root + 'data_2.csv'], 
                                    sampling_frequency=256, 
                                    cross_data_type='velocity', 
                                    distance_type='angular',   
                                    display_segmentation=True,
                                    size_plan_x=1200,
                                    size_plan_y=800))



