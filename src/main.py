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
 


 


# root = 'dataset/DS_Hollywood2/'
# #root = 'dataset/DS_Hollywood2/'

# #idx = [1, 3, 4, 5, 6]
# idx = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23,
#         24, 26, 27, 29, 30, 31, 33, 34, 36, 38, 39, 40, 43, 45, 46, 47, 48, 49, 51]

# training_set = [root + 'gaze_s{i}.csv'.format(i = i) for i in idx]
# label_set = [root + 'labels_s{i}.csv'.format(i = i) for i in idx]

# s = v.MLTraining.hard_fit(training_set,
#                                 label_set,
#                                 sampling_frequency = 500, 
#                                 segmentation_method = 'I_FC', 
#                                 task = 'binary',
#                                 classifier = 'rf',
#                                 distance_type = 'angular',
#                                 display_segmentation = True,
#                                 distance_projection = 600.0,
#                                 size_plan_x = 475.0,
#                                 size_plan_y = 280.0)

# Zemblys:    565.0
#             533.0
#             301.0
# Hollywood2: 600.0
#             475.0
#             280.0




## For Saccades
sa = v.SaccadeAnalysis(bs)
print(sa.saccade_count())
print(sa.saccade_frequency())
print(sa.saccade_frequency_wrt_labels())
print(sa.saccade_durations(get_raw=False))
print(sa.saccade_amplitudes(get_raw=False))
print(sa.saccade_travel_distances(get_raw=False))







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

 
#

 


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



 

## For Frequency-based


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
                                      csd_data_type = 'velocity', 
                                      distance_type = 'angular',   
                                      display_segmentation = True,
                                      size_plan_x = 1200,
                                      size_plan_y = 800)


print(cfa.cross_spectral_density(cross_data_type='velocity'))







#print(v.signal_coherency(sb, 
#                          Welch_samples_per_segment = 100))
#print(v.signal_coherency([root + 'data_1.csv', root + 'data_2.csv'], 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800))

#sb = v.StochasticAnalysis(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800)
#print(v.DACF(sb))
#print(v.DACF(root + 'data_1.csv', 
#                                      sampling_frequency = 256, 
#                                      segmentation_method = 'I_HMM',
#                                      distance_type = 'angular',   
#                                      display_segmentation = True,
#                                      size_plan_x = 1200,
#                                      size_plan_y = 800))


## For Scanpath 
 

#sp = v.Scanpath(bs, 
#                display_scanpath=True)

#sp = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = True,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True)

 
## For Geometrical Analysis  
 
#ga = v.GeometricalAnalysis(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800)
 
#ga = v.GeometricalAnalysis(bs,
#                           display_scanpath=True)
 
#ga = v.GeometricalAnalysis(sp)
#
 
#print(v.scanpath_HFD(bs))
#                     display_results=False))
#print(v.scanpath_HFD(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800))


## For RQA Analysis  

#ga = v.RQAAnalysis(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800)
 
#ga = v.RQAAnalysis(bs,
#                   display_scanpath=True)

#ga = v.RQAAnalysis(sp)

 
#print(v.scanapath_RQA_entropy(bs))
#print('THEN')
#print(v.scanapath_RQA_entropy(sp))

#print(v.scanapath_RQA_recurrence_rate(root + 'data_1.csv', 
#                           sampling_frequency = 256, 
#                           segmentation_method = 'I_HMM',
#                           distance_type = 'angular',                        
#                           display_segmentation = True,
#                           display_scanpath=True,
#                           size_plan_x = 1200,
#                           size_plan_y = 800))


## For Point Mapping Distance

# sp1 = v.Scanpath(root + 'data_1.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=False,
#                 verbose=False)



#print(v.scanapath_RQA_determinism(sp1,
#                                  scanpath_RQA_distance_threshold=150))

# sp2 = v.Scanpath(root + 'data_2.csv', 
#                 sampling_frequency = 256,  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# ref_sm = v.scanpath_saliency_map(sp1, 
#                                  scanpath_saliency_pixel_number = 100, 
#                                  scanpath_saliency_gaussian_std = 5,
#                                    verbose=True)

#i_ref_sa = v.scanpath_saliency_nss(sp2, ref_sm)
#print(i_ref_sa)


#v.scanpath_CRQA_entropy([sp2, sp1], 
#              scanpath_CRQA_distance_threshold = 80, 
#             verbose = False)


# v.AoI_HMM(root + 'data_1.csv',
#           sampling_frequency = 256,
#           verbose=False)
# aoi = v.AoISequence(root + 'data_1.csv',
#           sampling_frequency = 256,
#           verbose=False)




#pm = v.PointMappingDistance([sp1, sp2])

#print(v.frechet_distance([sp1, sp2]))

## For Elastic Distance

#sp1 = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,                  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=False)

#sp2 = v.Scanpath(root + 'data_2.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=True)

#pm = v.ElasticDistance([sp1, sp2])

#print(v.DTW_distance([sp1, sp2]))
                                         
## For Edit Distance
    
#sp1 = v.Scanpath(root + 'data_1.csv', 
#                sampling_frequency = 256,                  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                size_plan_y = 800,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=False)

#sp2 = v.Scanpath(root + 'data_2.csv', 
#                sampling_frequency = 256,  
#                segmentation_method = 'I_HMM',
#                distance_type = 'angular',                        
#                display_segmentation = False,
#                size_plan_x = 1200,
#                size_plan_y = 800,
#                display_scanpath=True,
#                verbose=True)

# sp3 = v.Scanpath(root + 'data_3.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp4 = v.Scanpath(root + 'data_4.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp5 = v.Scanpath(root + 'data_5.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp6 = v.Scanpath(root + 'data_6.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

#pm = v.StringEditDistance(  [sp1, sp2])

#print(v.scanpath_generalized_edit_distance([sp1, sp2], 
#                                  scanpath_temporal_binning=False, 
#                                  display_results = False))

## For scanmatch
#print(v.scanmatch_score([sp1, sp2]))

## For subsmatch
#print(v.subsmatch_similarity([sp1, sp2]))

## For multimatch
#print(v.multimatch_alignment([sp1, sp2], 
#                             display_results = False)
#      )


## For AoI Sequence
#aoi_s = v.AoISequence(sp1, 
#                      AoI_identification_method='I_KM', 
#                      AoI_IKM_cluster_number='search', 
#                      )

#aoi_s = v.AoISequence(sp1, 
#                       AoI_identification_method='I_HMM')
#print(aoi_s.centers)
#mb_a = v.MarkovBasedAnalysis(sp1)

#print(v.AoI_HMM(sp1))
#model = v.AoI_HMM(sp1)['AoI_HMM_model_instance']


#val = sp1.values[:2]
#val += np.random.random(val.shape)*10

#print(v.AoI_HMM_fisher_vector(np.tile(val, 10), 
#      AoI_HMM_model = model))


#print(np.tile(val, 2).shape)
#print(v.AoI_transition_matrix(mb_a))
 


                
 
             
# seqs = [sp1, sp2, sp3, sp4, sp5, sp6]               
# aoi_seqs = v.AoI_sequences(seqs, 
#                       display_scanpath=True, 
#                       AoI_identification_method = 'I_KM', 
#                       AoI_IKM_cluster_number = 5,
#                       AoI_temporal_binning = False, 
#                       AoI_temporal_binning_length=.2,   ) 
 

# #print(aoi_seqs[0].fixation_analysis.segmentation_results)
# #print(v.AoI_eMine(aoi_seqs))
# print(v.AoI_trend_analysis(aoi_seqs, 
#                            verbose=False))
# #print(v.AoI_constrained_DTW_barycenter_averaging(aoi_seqs))
 
#aoi1 = v.AoISequence(sp1, 
#                     AoI_identification_method = 'I_HMM')
#print(aoi1.durations)
#[aoi1, aoi2] = v.AoI_sequences([sp1, sp2]) 
#print(v.AoI_smith_waterman([sp1, sp2]))
#print(v.AoI_longest_common_subsequence([aoi1, aoi2]))

#print(v.AoI_eMine([sp1, sp2, sp3]))


## For visualizations 

 



