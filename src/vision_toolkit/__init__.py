# -*- coding: utf-8 -*-



## For Oculomotor Processing
from .oculomotor.segmentation_based.fixation import (
    FixationAnalysis, 
    fixation_average_velocity_deviations,
    fixation_average_velocity_means, fixation_BCEA, fixation_centroids,
    fixation_count, fixation_drift_displacements, fixation_drift_distances,
    fixation_drift_velocities, fixation_durations, fixation_frequency,
    fixation_frequency_wrt_labels, fixation_mean_velocities)
from .oculomotor.segmentation_based.saccade import (
    SaccadeAnalysis, 
    saccade_acceleration_deceleration_ratios,
    saccade_amplitude_duration_ratios, saccade_amplitudes,
    saccade_area_curvatures, saccade_average_acceleration_means,
    saccade_average_acceleration_profiles, saccade_average_deceleration_means,
    saccade_average_velocity_deviations, saccade_average_velocity_means,
    saccade_count, saccade_directions, saccade_durations, saccade_efficiencies,
    saccade_frequency, saccade_frequency_wrt_labels,
    saccade_horizontal_deviations, saccade_initial_deviations,
    saccade_initial_directions, saccade_main_sequence, saccade_max_curvatures,
    saccade_mean_acceleration_profiles, saccade_mean_accelerations,
    saccade_mean_decelerations, saccade_mean_velocities,
    saccade_peak_accelerations, saccade_peak_decelerations,
    saccade_peak_velocities, saccade_peak_velocity_amplitude_ratios,
    saccade_peak_velocity_duration_ratios,
    saccade_peak_velocity_velocity_ratios, saccade_skewness_exponents,
    saccade_successive_deviations, saccade_travel_distances)
from .oculomotor.signal_based.frequency import (CrossFrequencyAnalysis,
                                                FrequencyAnalysis,
                                                cross_spectral_density,
                                                periodogram, signal_coherency,
                                                welch_cross_spectral_density,
                                                welch_periodogram)
from .oculomotor.signal_based.stochastic import (DACF, DFA, MSD,
                                                 StochasticAnalysis)

## For Scanpath Processing
from .scanpath.scanpath_base import Scanpath
from .scanpath.similarity.character_based.string_edit_distance.string_edit_distance import (
    ScanpathStringEditDistance, 
    scanpath_generalized_edit_distance,
    scanpath_levenshtein_distance, 
    scanpath_needleman_wunsch_distance)
from .scanpath.similarity.distance_based.elastic_distance.elastic_distance import ( 
    ElasticDistance, 
    scanpath_DTW_distance,
    scanpath_frechet_distance)
from .scanpath.similarity.distance_based.point_mapping.point_mapping import (
    PointMappingDistance, 
    scanpath_TDE_distance, 
    scanpath_eye_analysis_distance, 
    scanpath_mannan_distance)
from .scanpath.similarity.specific_similarity_metrics.multimatch_alignment import scanpath_multimatch_alignment
from .scanpath.similarity.specific_similarity_metrics.scanmatch_score import scanpath_scanmatch_score
from .scanpath.similarity.specific_similarity_metrics.subsmatch_similarity import scanpath_subsmatch_similarity
from .scanpath.single.geometrical.geometrical import (GeometricalAnalysis,
                                                      scanpath_BCEA,
                                                      scanpath_convex_hull,
                                                      scanpath_HFD,
                                                      scanpath_k_coefficient,
                                                      scanpath_length,
                                                      scanpath_voronoi_cells)
from .scanpath.single.rqa.rqa import (RQAAnalysis, scanapath_RQA_CORM,
                                      scanapath_RQA_determinism,
                                      scanapath_RQA_entropy,
                                      scanapath_RQA_laminarity,
                                      scanapath_RQA_recurrence_rate)

## For Segmentation Processing
from .segmentation.processing.binary_segmentation import BinarySegmentation
from .segmentation.processing.ternary_segmentation import TernarySegmentation
from .visualization.aoi.spatio_temporal_based.dwell_time import (
    AoI_predefined_dwell_time)
from .visualization.aoi.spatio_temporal_based.sankey_diagram import (
    AoI_sankey_diagram)
from .visualization.aoi.spatio_temporal_based.scarf_plot import AoI_scarf_plot
from .visualization.aoi.spatio_temporal_based.time_plot import AoI_time_plot
from .visualization.aoi.transition_based.chord_diagram import AoI_chord_diagram
from .visualization.aoi.transition_based.directed_graph import (
    AoI_directed_graph)
from .visualization.aoi.transition_based.transition_flow import (
    AoI_transition_flow)

## For AoI Processing
from .aoi.aoi_base import AoI_sequences, AoIMultipleSequences, AoISequence
from .aoi.basic.basic import AoIBasicAnalysis
from .aoi.common_subsequence.constrained_dtw_barycenter_averaging import AoI_CDBA
from .aoi.common_subsequence.local_alignment.local_alignment import (
    AoI_eMine, 
    AoI_longest_common_subsequence, 
    AoI_smith_waterman,
    LocalAlignment)
from .aoi.common_subsequence.trend_analysis import AoI_trend_analysis
from .aoi.global_alignment.string_edit_distance import (
    AoI_generalized_edit_distance, 
    AoI_levenshtein_distance,
    AoI_needleman_wunsch_distance, 
    AoIStringEditDistance)
from .aoi.markov_based.markov_based import (AoI_HMM, AoI_HMM_fisher_vector,
                                            AoI_HMM_transition_entropy,
                                            AoI_HMM_transition_matrix,
                                            AoI_successor_representation,
                                            AoI_transition_entropy,
                                            AoI_transition_matrix,
                                            MarkovBasedAnalysis)
from .aoi.pattern_mining.lempel_ziv import AoI_lempel_ziv
from .aoi.pattern_mining.n_gram import AoI_NGram
from .aoi.pattern_mining.spam import AoI_SPAM