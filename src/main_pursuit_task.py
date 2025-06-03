# -*- coding: utf-8 -*-


 
import vision_toolkit as v
import numpy as np

root = 'dataset/'
np.random.seed(1)

# Create an instance of PursuitTask (optional, for comparison)
pt = v.PursuitTask(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=198,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)

print(pt.pursuit_task_count())
print(pt.pursuit_task_frequency())
print(pt.pursuit_task_duration())
print(pt.pursuit_task_proportion())
print(pt.pursuit_task_slope_ratios())




# Call pursuit_task_count directly via the module
count = v.pursuit_task_count(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=200,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)
freq = v.pursuit_task_frequency(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=200,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)
dur = v.pursuit_task_duration(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=200,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)
 

prop = v.pursuit_task_proportion(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=200,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)


slope = v.pursuit_task_slope_ratios(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=200,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)












 