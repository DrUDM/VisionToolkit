# -*- coding: utf-8 -*-


 
import vision_toolkit as v
import numpy as np

root = 'dataset/'
np.random.seed(1)

# Create an instance of PursuitTask (optional, for comparison)
pt = v.PursuitTask(
    root + 'example_pursuit.csv',
    root + 'example_pursuit_theo.csv',
    pursuit_start_idx=190,
    sampling_frequency=1000,
    segmentation_method='I_VMP',
    distance_type='angular',
    display_segmentation=True
)

# print(pt.pursuit_task_count())
# print(pt.pursuit_task_frequency())
# print(pt.pursuit_task_duration())
# print(pt.pursuit_task_proportion())
# print(pt.pursuit_task_velocity())
# print(pt.pursuit_task_velocity_means())
# print(pt.pursuit_task_peak_velocity())
# print(pt.pursuit_task_amplitude())
# print(pt.pursuit_task_distance())
# print(pt.pursuit_task_efficiency())
#print(pt.pursuit_task_slope_ratios())
# print(pt.pursuit_task_crossing_time(tolerance=1.0)) 
#print(pt.pursuit_task_overall_gain(get_raw=True))
#print(pt.pursuit_task_overall_gain_x(get_raw=True))
#print(pt.pursuit_task_slopel_gain(_type='weighted'))
#print(pt.pursuit_task_accuracy(.15, 'weighted'))
print(pt.pursuit_task_entropy(pursuit_entropy_window=25, 
                              pursuit_entropy_tolerance=.2))



# # Call pursuit_task_count directly via the module
# count = v.pursuit_task_count(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

# freq = v.pursuit_task_frequency(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

# dur = v.pursuit_task_duration(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )
 
# prop = v.pursuit_task_proportion(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

# vel = v.pursuit_task_velocity(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

 
# vel_m = v.pursuit_task_velocity_means(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )


# peak_vel = v.pursuit_task_peak_velocity(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )


# amp = v.pursuit_task_amplitude(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )


# dist = v.pursuit_task_distance(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )


# eff = v.pursuit_task_efficiency(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

 
# slope = v.pursuit_task_slope_ratios(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True
# )

# crossing = v.pursuit_task_crossing_time(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=200,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True,
#     tolerance=1.0
# )

# ptg = v.pursuit_task_overall_gain(
#     root + 'example_pursuit.csv',
#     root + 'example_pursuit_theo.csv',
#     pursuit_start_idx=201,
#     sampling_frequency=1000,
#     segmentation_method='I_VMP',
#     distance_type='angular',
#     display_segmentation=True,
#     get_raw=True
# )
 



 





 