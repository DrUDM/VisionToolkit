# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)
 

## For Bineary Segmentation

bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_VT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_DiT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_DeT',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_KF',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_MST',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_2MC',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)
bs = v.BinarySegmentation(root + 'data_1.csv', 
                        sampling_frequency = 256,  
                        segmentation_method = 'I_HMM',
                        distance_type = 'angular',                        
                        display_segmentation = True,
                        size_plan_x = 1200,
                        size_plan_y = 800)


ts = v.TernarySegmentation('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                            sampling_frequency = 500, 
                            segmentation_method = 'I_VVT',  
                            display_segmentation = True,
                            size_plan_x = 475.0,
                            size_plan_y = 280.0)

ts = v.TernarySegmentation('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                            sampling_frequency = 500, 
                            segmentation_method = 'I_VDT',  
                            display_segmentation = True,
                            size_plan_x = 475.0,
                            size_plan_y = 280.0)

ts = v.TernarySegmentation('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                            sampling_frequency = 500, 
                            segmentation_method = 'I_VMP',  
                            display_segmentation = True,
                            size_plan_x = 475.0,
                            size_plan_y = 280.0)

ts = v.TernarySegmentation('dataset/DS_Hollywood2/' + 'gaze_s1.csv', 
                            sampling_frequency = 500, 
                            segmentation_method = 'I_BDT',  
                            display_segmentation = True,
                            size_plan_x = 475.0,
                            size_plan_y = 280.0)
 