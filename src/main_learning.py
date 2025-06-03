# -*- coding: utf-8 -*-

import segmentation_src as seg

root = 'dataset/DS_Hollywood2/'
#root = 'dataset/DS_Hollywood2/'

#idx = [1, 3, 4, 5, 6]
idx = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23,
       24, 26, 27, 29, 30, 31, 33, 34, 36, 38, 39, 40, 43, 45, 46, 47, 48, 49, 51]

training_set = [root + 'gaze_s{i}.csv'.format(i = i) for i in idx]
label_set = [root + 'labels_s{i}.csv'.format(i = i) for i in idx]

s = seg.DLTraining.fit_predict(training_set,
                               label_set,
                               sampling_frequency = 500, 
                               segmentation_method = 'I_VVT', 
                               task = 'ternary',
                               #classifier = 'rf',
                               distance_type = 'angular',
                               display_segmentation = True,
                               distance_projection = 600.0,
                               size_plan_x = 475.0,
                               size_plan_y = 280.0)

# Zemblys:    565.0
#             533.0
#             301.0
# Hollywood2: 600.0
#             475.0
#             280.0