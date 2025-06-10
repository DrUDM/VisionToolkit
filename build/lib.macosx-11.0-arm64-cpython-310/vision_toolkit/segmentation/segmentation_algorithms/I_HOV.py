# -*- coding: utf-8 -*-

import numpy as np  

from Vision.segmentation_src.utils.segmentation_utils import interval_merging, centroids_from_ints 
from Vision.segmentation_src.utils.velocity_distance_factory import absolute_angular_distance
 
import time 
import joblib


def pre_process_IHOV(data_set, config):
    """
     
    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    assert config['distance_type'] == 'euclidean', "'Distance type' must be set to 'euclidean"
 
    start_time = time.time()
    
    # Get generic data
    s_f = config['sampling_frequency']
    n_s = config['nb_samples'] 
    
    x_array = data_set['x_array']
    y_array = data_set['y_array'] 
    coord =np.concatenate((x_array.reshape(1, n_s),
                           y_array.reshape(1, n_s)), axis = 0)
    
    # Compute index intervals corresponding to the duration threshold
    t_du = np.ceil(config['IHOV_duration_threshold'] * s_f) 
    
    # Number of value to use arround index i for averaging step
    t_av = np.ceil(config['IHOV_averaging_threshold'] * s_f) 
    
    if (t_av % 2) == 0:
        t_av += 1
    
    t_av = int(t_av)
    
    nb_ang_bin = config['IHOV_angular_bin_nbr']
    ang_bin = 360 / nb_ang_bin
   

        
    
    # Compute vector matrix from window 1
    w_1_m = np.array([coord[:,i].reshape(2, 1) 
                      - coord[:, np.maximum(0, np.arange(i-t_du, 
                                                         i, dtype=int))] for i in range(n_s)]) 
  
    # Compute vector matrix from window 2
    w_2_m = np.array([coord[:, np.minimum(n_s-1, np.arange(i+1, 
                                                            i+t_du+1, dtype=int))] 
                      - coord[:,i].reshape(2, 1) for i in range(n_s)]) 
    
    # Compute vector matrix from window 3
    w_3_m = np.array([coord[:, np.minimum(n_s-1, np.flip(np.arange(i+1, 
                                                                    i+t_du+1, dtype=int)))] 
                       - coord[:, np.maximum(0, np.arange(i-t_du, 
                                                          i, dtype=int))] for i in range(n_s)]) 
    
    hists_w_1, hists_w_2, hists_w_3 = [], [], []
    
    
    def comp_hist(bin_, vect_):
        """
         
        Parameters
        ----------
        bin_ : TYPE
            DESCRIPTION.
        vect_ : TYPE
            DESCRIPTION.

        Returns
        -------
        n_hist : TYPE
            DESCRIPTION.

        """
        # Compute distances: the original implementation requires velocity but 
        # velocity histograms are re-normalized -> same result with distances
        dist_ = np.linalg.norm(vect_, axis=0) 
        
        # Initialize histogram
        hist, n_hist = np.zeros(nb_ang_bin), np.zeros(nb_ang_bin) 
  
        for k in range(nb_ang_bin):
        
            # Get indexes corresponding to the binned angular value 
            idx_ = np.where(bin_==k)[0] 
            
            # Add distances corresponding to indexes extracted above
            hist[k] = np.sum(dist_[idx_])
      
        # Rotation to have the maximum distance value in first position
        k = np.argmax(hist)
        t_ = nb_ang_bin-k
     
        n_hist[:t_] = hist[k:]
        n_hist[t_:] = hist[:k] 
        
        # Histogram normalization
        n_hist /= np.sum(n_hist)
        
        return n_hist
    
    
    for i in range(n_s):
        
        # Compute histogram for window 1 
        l_w1 = w_1_m[i]
        l_bin_1 = np.floor(comp_angles(l_w1) / ang_bin)
        
        hists_w_1.append(comp_hist(l_bin_1,
                                   l_w1)) 
    
        # Compute histogram for window 2 
        l_w2 = w_2_m[i]
        l_bin_2 = np.floor(comp_angles(l_w2) / ang_bin)
        
        hists_w_2.append(comp_hist(l_bin_2,
                                   l_w2))
        
        # Compute histogram for window 3
        l_w3 = w_3_m[i]
        l_bin_3 = np.floor(comp_angles(l_w3) / ang_bin)
         
        hists_w_3.append(comp_hist(l_bin_3,
                                   l_w3))
        
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 1 are re-normalized
    hists_w_1 = np.array(hists_w_1) 
    a_hists_w_1 = np.zeros_like(hists_w_1)
    
    for k in range(nb_ang_bin):
        a_hists_w_1[:,k] = averaging(hists_w_1[:,k], t_av)
  
    a_hists_w_1 /= np.sum(a_hists_w_1, axis = 1).reshape(n_s, 1) 
  
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 2 are re-normalized
    hists_w_2 = np.array(hists_w_2) 
    a_hists_w_2 = np.zeros_like(hists_w_2)
    
    for k in range(nb_ang_bin):
        a_hists_w_2[:,k] = averaging(hists_w_2[:,k], t_av)
  
    a_hists_w_2 /= np.sum(a_hists_w_2, axis = 1).reshape(n_s, 1) 
    
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 3 are re-normalized
    hists_w_3 = np.array(hists_w_3)
    a_hists_w_3 = np.zeros_like(hists_w_3)
    
    for k in range(nb_ang_bin):
        a_hists_w_3[:,k] = averaging(hists_w_3[:,k], t_av)
  
    a_hists_w_3 /= np.sum(a_hists_w_3, axis = 1).reshape(n_s, 1) 
    
    print("--- 1: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    # Averaged and normalized histograms are concatened for each time index
    feat_ = np.concatenate((a_hists_w_1, a_hists_w_2, a_hists_w_3), axis = 1)
 
    return feat_
    
 
def comp_angles(v_mat):
    """
     
    Parameters
    ----------
    v_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    dir_ : TYPE
        DESCRIPTION.

    """
    # to avoid numerical instability
    v_mat += 1e-10
    
    dir_ = np.zeros(v_mat.shape[1])
 
    # Find angles <= pi and > pi 
    _p = v_mat[1,:] >= 0  
    _m = v_mat[1,:] < 0
  
    # Compute norm of individual vectors for angles >= pi
    n_p = np.linalg.norm(v_mat[:,_p], axis = 0) 
      
    # Compute directions
    dir_[_p] = (180/np.pi)*np.arccos(np.divide(v_mat[0,:][_p], 
                                               n_p,
                                               where = n_p > 0))
 
    # Compute norm of individual vectors for angles < pi
    n_m = np.linalg.norm(v_mat[:,_m], axis = 0) 
 
    # Compute directions
    dir_[_m] = (180/np.pi)*(2 * np.pi - np.arccos(np.divide(v_mat[0,:][_m], 
                                                            n_m,
                                                            where = n_m > 0)))    
   
    return dir_


def averaging(v_, t_av):
    """
     
    Parameters
    ----------
    v_ : TYPE
        DESCRIPTION.
    t_av : TYPE
        DESCRIPTION.

    Returns
    -------
    averaged : TYPE
        DESCRIPTION.

    """
    n_ = len(v_)
    h_t_av = int(t_av/2)
 
    conv = np.convolve(v_, np.ones(t_av))/t_av     
    averaged = conv[h_t_av: n_ + h_t_av] 
     
    return averaged
    

def process_IHOV(data_set, config):
    
    if config['verbose']:
        
        print("Processing HOV Identification...")  
        start_time = time.time()
    
    task = config['task']
    classifier = config['IHOV_classifier']
    
    path = 'segmentation_src/segmentation_algorithms/trained_models/I_HOV/i_{cl}_{task}.joblib'.format(
        cl = classifier,
        task = task
        )
    
    clf = joblib.load(path)
    feature_mat = pre_process_IHOV(data_set, config) 
 
    preds = clf.predict(feature_mat)
    
    if config['verbose']:
    
        print("Done")  
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))   
  
    
    if task == 'binary':
        
        wi_fix = np.where(preds == 1)[0]
        i_fix = np.array([False]*config['nb_samples'])  
        i_fix[wi_fix] = True
	
        f_ints = interval_merging(wi_fix, 
                                  min_int_size = config['min_int_size'])
    
        x_a = data_set['x_array']
        y_a = data_set['y_array']
    
        ctrds = centroids_from_ints(f_ints,
                                    x_a, y_a)
 
        i_sac = i_fix == False
        wi_sac = np.where(i_sac == True)[0]

        s_ints = interval_merging(wi_sac,
                                  min_int_size = config['min_int_size'])
    

        return dict({
            'is_fixation': i_fix,
            'fixation_intervals': f_ints,
            'centroids': ctrds,
            'is_saccade': i_sac,
            'saccade_intervals': s_ints
                })
    
    elif task == "ternary":
        
        return dict({
            'is_fixation': preds == 1,
            'fixation_intervals': interval_merging(np.where((preds == 1) == True)[0],
                                                   min_int_size = config['min_int_size']),
            'is_saccade': preds == 2,
            'saccade_intervals': interval_merging(np.where((preds == 2) == True)[0],
                                                  min_int_size = config['min_int_size']),
            'is_pursuit': preds == 3,
            'pursuit_intervals': interval_merging(np.where((preds == 3) == True)[0],
                                                  min_int_size = config['min_int_size']),
            
            })  
    
    
   