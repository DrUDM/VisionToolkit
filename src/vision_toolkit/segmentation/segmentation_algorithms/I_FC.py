# -*- coding: utf-8 -*-

import numpy as np 
from scipy.cluster.vq import kmeans2

from Vision.segmentation_src.utils.segmentation_utils import interval_merging, centroids_from_ints
from Vision.segmentation_src.utils.velocity_distance_factory import absolute_angular_distance
from Vision.segmentation_src.utils.segmentation_utils import dict_vectorize, standard_normalization

 
import time
import joblib



def pre_process_IFC(data_set, config):
    """
    
    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    features : TYPE
        DESCRIPTION.

    """
    print('Processing feature extraction...')
    assert config['distance_type'] == 'angular', "'Distance type' must be set to 'angular"
  
    start_time = time.time()
    # Get generic data
    s_f = config['sampling_frequency']
    n_s = config['nb_samples'] 
    
    x_array = data_set['x_array']
    y_array = data_set['y_array']  
    
    theta_coord = data_set['theta_coord']
    unitary_gaze_vectors = data_set['unitary_gaze_vectors']
    
    bcea_p = config['IFC_bcea_prob'] 
    i2mc = config['IFC_i2mc']
    
    # Get absolute angular velocity vector
    a_sp = data_set['absolute_speed']
     
    # Compute position difference vectors
    diff_ = np.zeros((2, n_s))
    
    diff_[0,:-1] = x_array[1:] - x_array[:-1]
    diff_[1,:-1] = y_array[1:] - y_array[:-1]
    diff_[:,-1] = diff_[:,-2]
 
    # Compute angular acceleration vector
    acc = np.zeros_like(a_sp)
    
    acc[:-1] = a_sp[1:] - a_sp[:-1]
    acc[-1] = acc[-2]
     
    # Compute absolute angular distance beteen successive unitary gaze vectors
    ang_suc_dist = np.zeros_like(a_sp)
    
    dot_ = np.array([
        unitary_gaze_vectors[:,i-1] @ unitary_gaze_vectors[:,i] for i in range(1, config['nb_samples'])
        ])
    rad_suc_dist = np.arccos(dot_)
    
    ang_suc_dist[:-1] = np.abs(rad_suc_dist/(np.pi)*180)
    ang_suc_dist[-2] = ang_suc_dist[-1]
    
    # Compute successive vector directions relative to the horizontal axis
    # for the Rayleigh z-score
    suc_dir = np.zeros_like(a_sp)
    
    # to avoid numerical instability
    diff_ += 1e-10
    
    _m = diff_[1,:]<0
    _p = diff_[1,:]>=0
    
    n_p = np.linalg.norm(diff_[:,_p], axis = 0)
    suc_dir[_p] = np.arccos(np.divide(diff_[0,:][_p], 
                                      n_p,
                                      where = n_p > 0))
    
    n_m = np.linalg.norm(diff_[:,_m], axis = 0)
    suc_dir[_m] = (2 * np.pi - np.arccos(np.divide(diff_[0,:][_m], 
                                                   n_m,
                                                   where = n_m > 0)))    
    
    # Compute index intervals corresponding to 50 and 11 ms 
    n_50 = np.ceil(0.050 * s_f)
    n_11 = np.ceil(0.011 * s_f)
     
    # Initialize feature dictionary
    features = dict({'sf': np.ones(n_s)*s_f,
                     'rms': np.zeros(n_s),
                     'delta_rms': np.zeros(n_s),
                     'std_x': np.zeros(n_s),
                     'std_y': np.zeros(n_s),
                     'delta_std_x': np.zeros(n_s),
                     'delta_std_y': np.zeros(n_s),
                     'bcea': np.zeros(n_s),
                     'delta_bcea': np.zeros(n_s),
                     'dispersion': np.zeros(n_s),
                     'velocity': a_sp,
                     'acceleration': acc,
                     'med_dist': np.zeros(n_s),
                     'mean_dist': np.zeros(n_s),
                     'z_ray': np.zeros(n_s), 
                     })
    
    if i2mc:
        
        # Initialize utils to compute I2MC weigths
        features.update({'i2mc': np.zeros(n_s), })
        
        t_du = np.floor(config['IFC_i2mc_window_duration'] * s_f / 2) 
        t_mo = np.ceil(config['IFC_i2mc_moving_threshold'] * s_f) 
        
        n_wc = int(2*t_du + 1)
        n_wc_2 = int(np.ceil(n_wc/2))
        n_wc_4 = int(np.ceil(n_wc/4))
        
        f_w = np.zeros(n_s)
        norm_ = np.zeros(n_s)
        
    # Create vectors of descriptors 
    rms_v = np.zeros(n_s) 
    bcea_v = np.zeros(n_s)
     
    # Compute the matrix of angular position in 100ms windows centered on each time stamp i
    w_c_100_m = np.array([theta_coord[:, np.minimum(np.maximum(0, np.arange(i-n_50, 
                                                                            i+n_50+1, dtype=int)),
                                                    n_s-1)] for i in range(n_s)])
    
    # Compute the matrix of angular succesive distances in 100ms windows centered on each time stamp i
    w_c_100_asd = np.array([ang_suc_dist[np.minimum(np.maximum(0, np.arange(i-n_50, 
                                                                            i+n_50+1, dtype=int)),
                                                    n_s-1)] for i in range(n_s)])   
    
    # Compute the matrix of unitary gaze vectors in 100ms windows centered on each time stamp i
    w_c_100_ugv = np.array([unitary_gaze_vectors[:, np.minimum(np.maximum(0, np.arange(i-n_50, 
                                                                                       i+n_50+1, dtype=int)),
                                                               n_s-1)] for i in range(n_s)])   
    
    # Compute the matrix of successive vector directions in 22ms windows centered on each time stamp i
    w_c_22_sd = np.array([suc_dir[np.minimum(np.maximum(0, np.arange(i-n_11, 
                                                                     i+n_11+1, dtype=int)),
                                             n_s-1)] for i in range(n_s)])
    
    # Compute means of unitary gaze vectors in 100 ms windows
    mean_pos_v = np.mean(w_c_100_ugv, axis = 2)
    mean_pos_v *= 1/np.linalg.norm(mean_pos_v, axis = 1).reshape(n_s, 1)
   
    # Compute medians of unitary gaze vectors in 100 ms windows
    med_pos_v = np.median(w_c_100_ugv, axis = 2)
    med_pos_v *= 1/np.linalg.norm(med_pos_v, axis = 1).reshape(n_s, 1)
    
    # Compute dispersion of angular positions in a 100 ms window
    disp_v = (np.max(w_c_100_m[:,0,:], axis = 1) - np.min(w_c_100_m[:,0,:], axis = 1)
              + np.max(w_c_100_m[:,1,:], axis = 1) - np.min(w_c_100_m[:,1,:], axis = 1))
    
    # Compute horizontal and vertical standard deviations of angular positions in a 100 ms window
    std_v = np.std(w_c_100_m, axis = 2) 
 
    # Compute root mean square of angular succesive displacements
    rms_v = np.sqrt(np.sum(w_c_100_asd**2, axis = 1) / w_c_100_asd.shape[1])
    
    # Compute Rayleigh z-score
    r_mat = np.array([np.cos(w_c_22_sd),
                      np.sin(w_c_22_sd)])
    
    n_22 = r_mat.shape[2]
    
    rm_mat = np.sum(r_mat, axis = 2)/n_22
    z_ray_v = (np.linalg.norm(rm_mat, axis = 0)**2) * n_22
    
    for i in range(n_s):
    
        # Compute index windows of 100 and 200 ms length centered on i  
        w_c_100 = np.minimum(np.maximum(0, np.arange(i-n_50, 
                                                     i+n_50+1, dtype=int)),
                             n_s-1)
 
        # Compute BCEA in a 100 ms window
        bcea_v[i] = bcea(theta_coord[:, w_c_100], 
                         bcea_p, std_v[i])
        
    if i2mc:
 
        t_x = theta_coord[0,:]
        t_y = theta_coord[1,:]
        
        i = 0 
        while i + t_mo < n_s :
            
            i += t_mo
            
            # Compute index windows of two times t_du length centered on i
            w_c = np.minimum(np.maximum(0, np.arange(i-t_du, 
                                                     i+t_du+1, dtype=int)),
                             n_s-1)
            
            X_1 = np.concatenate((t_x[w_c].reshape(n_wc, 1),
                                  t_y[w_c].reshape(n_wc, 1)), axis = 1) 
            _, lbl_1 = kmeans2(X_1, 2, iter=15, minit='++')
          
            trs_1 = np.where(np.diff(lbl_1) != 0)[0] + 1
            
            # Compute weigths
            w_1 = np.zeros(n_wc)
            w_1[trs_1] = 1/len(trs_1)
           
            # Add weigths
            f_w[w_c] += w_1
            
            # Under-sample: 1 for 2
            X_2 = np.concatenate((t_x[w_c][::2].reshape(n_wc_2, 1),
                                  t_y[w_c][::2].reshape(n_wc_2, 1)), axis = 1)
            _, lbl_2 = kmeans2(X_2, 2, iter=15, minit='++')
            
            trs_2 = np.where(np.diff(lbl_2) != 0)[0] + 1
            
            # Compute weigths
            w_2 = np.zeros(n_wc_2)
            w_2[trs_2] = 1/len(trs_2)
            
            # Repeat results since sequence was under-sampled and add weigth
            f_w[w_c] += np.repeat(w_2, 2)[:n_wc]
            
            # Under-sample: 1 for 4
            X_4 = np.concatenate((t_x[w_c][::4].reshape(n_wc_4, 1),
                                  t_y[w_c][::4].reshape(n_wc_4, 1)), axis = 1)
            
            _, lbl_4 = kmeans2(X_4, 2, iter=15, minit='++')
            
            trs_4 = np.where(np.diff(lbl_4) != 0)[0] + 1
            
            # Compute weigths
            w_4 = np.zeros(n_wc_4)
            w_4[trs_4] = 1/len(trs_4)
            
            # Repeat results since sequence was under-sampled and add weigth
            f_w[w_c] += np.repeat(w_4, 4)[:n_wc]
            
            # Normalization factor to get the average 
            norm_[w_c] += np.ones(n_wc)  
            
        f_w /= norm_
        features['i2mc'] = f_w
   
    features['std_x'] = std_v[:,0]
    features['std_y'] = std_v[:,1]
    
    features['rms'] = rms_v
    
    features['dispersion'] = disp_v
    features['bcea'] = bcea_v
    
    features['z_ray'] = z_ray_v
      
    for i in range(n_s):
    
        # Compute center index of 100ms windows before and after i
        i_b_100 = int(max(0, i-n_50))
        i_a_100 = int(min(n_s-1, i+n_50))
       
        # Compute difference between root mean square of the sample-to-sample displacements calculated 
        # for 100 ms windows preceding and succeeding the sample i 
        features['delta_rms'][i] = rms_v[i_a_100] - rms_v[i_b_100]
     
        # Compute difference between horizontal and vertical standard deviations of angular positions
        # calculated for 100 ms windows preceding and succeeding the sample i 
        d_std = std_v[i_a_100] - std_v[i_b_100]
        features['delta_std_x'][i] = d_std[0]
        features['delta_std_y'][i] = d_std[1]
  
        # Compute difference between BCEAs calculated 
        # for 100 ms windows preceding and succeeding the sample i 
        features['delta_bcea'][i] = bcea_v[i_a_100] - bcea_v[i_b_100]
        
        # Compute absolute angular distance between mean positions 
        # computed in 100 ms windows preceding and succeeding the sample i 
        rad_mean_dist = np.arccos(np.matmul(mean_pos_v[i_a_100], mean_pos_v[i_b_100]))
        features['mean_dist'][i] = np.abs(rad_mean_dist/(np.pi)*180)
        
        # Compute absolute angular distance between median positions 
        # computed in 100 ms windows preceding and succeeding the sample i 
        rad_med_dist = np.arccos(np.matmul(med_pos_v[i_a_100], med_pos_v[i_b_100]))
        features['med_dist'][i] = np.abs(rad_med_dist/(np.pi)*180)
 
    print("--- 1: %s seconds ---" % (time.time() - start_time))
  
    return features
  
    
def bcea(theta_coord, 
         p, std_xy):
    """

    Parameters
    ----------
    theta_coord : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    bcea_ : TYPE
        DESCRIPTION.

    """
    
    theta_x = theta_coord[0]
    theta_y = theta_coord[1]
    
    k = -np.log(1 - p)
    
    def pearson_corr(t_x, t_y):
        """
     
        Parameters
        ----------
        t_x : TYPE
            DESCRIPTION.
        t_y : TYPE
            DESCRIPTION.

        Returns
        -------
        p_c : TYPE
            DESCRIPTION.

        """
 
        mx = np.mean(t_x)
        my = np.mean(t_y)
        
        xm, ym = t_x-mx, t_y-my
        
        _num = np.sum(xm * ym)
        _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
        
        p_c = _num / _den
    
        # For some small artifact of floating point arithmetic.
        p_c = max(min(p_c, 1.0), -1.0)      
    
        return p_c 
        
    p_c = pearson_corr(theta_x, theta_y)
    sd_x = std_xy[0]
    sd_y = std_xy[1]
    
    bcea_ = 2 * np.pi * k * sd_x * sd_y * np.sqrt(1 - p_c**2)
    
    return bcea_
            
            
def process_IFC(data_set, config):
    
    if config['verbose']:
        
        print("Processing FC Identification...")  
        start_time = time.time()
    
    task = config['task']
    classifier = config['IFC_classifier']
    
    path = 'segmentation_src/segmentation_algorithms/trained_models/I_FC/i_{cl}_{task}.joblib'.format(
        cl = classifier,
        task = task
        )
    
    clf = joblib.load(path)
    feature_dict = pre_process_IFC(data_set, config) 
    
    feature_mat = dict_vectorize([feature_dict])
    
    if classifier != 'rf':
        
        path = 'segmentation_src/segmentation_algorithms/trained_models/I_FC/normalization_{task}.npy'.format(
            task = task
            )
        norm_= np.load(path)
        feature_mat, _, _ = standard_normalization(feature_mat,
                                             mu = norm_[0], sigma = norm_[1])
  
    
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
    
    
     
        