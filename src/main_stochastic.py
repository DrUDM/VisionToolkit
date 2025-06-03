# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 
root = 'dataset/'
np.random.seed(1)
 

 
st = v.StochasticAnalysis(root + 'data_1.csv',
                   sampling_frequency=256, 
                   segmentation_method = 'I_HMM',
                   distance_type = 'angular',    
                   size_plan_x = 1200,
                   size_plan_y = 800)

 
print(v.MSD(st))





s = v.MSD(root + 'data_1.csv',
                   sampling_frequency=256, 
                   segmentation_method = 'I_HMM',
                   distance_type = 'angular',    
                   size_plan_x = 1200,
                   size_plan_y = 800)





