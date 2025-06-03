# -*- coding: utf-8 -*-
 
import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)


pa = v.PursuitAnalysis('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280)

# Zemblys:    565.0
#             533.0
#             301.0
# Hollywood2: 600.0
#             475.0
#             280.0



print(v.pursuit_count('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))
 
print(v.pursuit_frequency('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))

print(v.pursuit_velocity('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))
print(v.pursuit_duration('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))
print(v.pursuit_proportion('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))
print(v.pursuit_velocity('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))
print(v.pursuit_velocity_means('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))

print(v.pursuit_peak_velocity('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))

print(v.pursuit_amplitude('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))

print(v.pursuit_distance('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))

print(v.pursuit_efficiency('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_BDT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        distance_projection = 600.0, 
                        size_plan_x = 475,
                        size_plan_y = 280))


print(v.pursuit_count(pa))

print(v.pursuit_frequency(pa))
print(v.pursuit_duration(pa))
print(v.pursuit_proportion(pa))
print(v.pursuit_velocity(pa))
print(v.pursuit_velocity_means(pa))
print(v.pursuit_peak_velocity(pa))
print(v.pursuit_amplitude(pa))
print(v.pursuit_distance(pa))
print(v.pursuit_efficiency(pa))





 
print(pa.pursuit_count())
print(pa.pursuit_frequency())
print(pa.pursuit_duration(get_raw=True))
print(pa.pursuit_proportion())
print(pa.pursuit_velocity(get_raw=True))
print(pa.pursuit_velocity_means(get_raw=True))
print(pa.pursuit_peak_velocity(get_raw=True))
print(pa.pursuit_amplitude(get_raw=True))
print(pa.pursuit_distance(get_raw=True))
print(pa.pursuit_efficiency(get_raw=True)) 


