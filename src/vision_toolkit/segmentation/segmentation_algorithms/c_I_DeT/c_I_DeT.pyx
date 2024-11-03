# distutils: language = c++
 
import numpy as np  

from vision_toolkit.utils.segmentation_utils import centroids_from_ints, interval_merging

from libc cimport math

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.pair cimport pair as cpair 
 

cdef map[int, int] dict_to_cmap(dict p_dict):
    
    cdef int map_key
    cdef int map_val
     
    cdef cpair[int, int] map_e 
    cdef map[int, int] c_map
    
    for key,val in p_dict.items():
        
        map_key = key
        map_val = val   
        map_e = (map_key, map_val)
        c_map.insert(map_e)
        
    return c_map

 
def process_IDeT(dict data_set, dict config):
    """
    I-DeT algorithm.
    """

    # Cython type declaration
    cdef int n_s = config['nb_samples']
    cdef int s_f = config['sampling_frequency'] 
    
    cdef bool euclidean 
    cdef double[:,:] g_npts
    
    if config['distance_type'] == 'euclidean':
        
        euclidean = True
        g_npts = np.concatenate((data_set['x_array'].reshape(1, n_s), 
                                 data_set['y_array'].reshape(1, n_s),
                                 data_set['z_array'].reshape(1, n_s)), axis = 0)
        
    else:
        
        euclidean = False  
        g_npts = data_set['unitary_gaze_vectors']
        print(g_npts)
        
    cdef double d_t = config['IDeT_density_threshold']   
    cdef int n_w = int(np.ceil(config['IDeT_duration_threshold'] 
                               *config['sampling_frequency']))
     
    cdef list C_clus = []
    cdef list neigh = []
    cdef dict avlb = {i: int(1) for i in range(0, n_s)}
    
    cdef int i = 0
    
    for i in range (n_s):
        
        #if i%10000==0:
        #    print('Processing {i}-th data sample'.format(i=i))
            
        if avlb[i] == True:
            
            neigh = vareps_neighborhood(g_npts, euclidean,
                                        n_s, i, 
                                        d_t, n_w)      
            
            if len(neigh)+1 >= n_w: 
               
                avlb[i] = False
                l_C_clus, avlb = expand_cluster(g_npts, euclidean,
                                                n_s, i, neigh, 
                                                d_t, n_w, 
                                                avlb)

                if len(l_C_clus) >= n_w: 
                    C_clus.append(l_C_clus)
 
    i_fix = np.zeros(n_s) 
    
    for clus in C_clus:
        
        l_fix = list()
        l_fix.append(min(clus))
        l_fix.append(max(clus)) 
        i_fix[l_fix[0]: l_fix[-1]+1] = 1
   
    wi_fix = np.where(i_fix == True)[0]
 
    x_a = data_set['x_array']
    y_a = data_set['y_array']
 
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size = np.ceil(config['min_sac_duration']*s_f), 
                              )
    
    if config['verbose']:
        print('   Saccadic intervals identified with minimum duration: {s_du} sec'.format(s_du=config['min_sac_duration']))
    
    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True]*config['nb_samples'])
    
    for s_int in s_ints:
        i_fix[s_int[0]: s_int[1]+1] = False
          
    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config['min_fix_duration']*s_f))
    
    for i in range(1, len(s_ints)):
        
        s_int = s_ints[i]
        o_s_int = s_ints[i-1]
        
        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1]: s_int[0]+1] = False
    
    if config['verbose']:
        print('   Close saccadic intervals merged with duration threshold: {f_du} sec'.format(f_du=config['min_fix_duration']))
     
    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]
    
    f_ints = interval_merging(
        wi_fix,   
        min_int_size = np.ceil(config['min_fix_duration']*s_f), 
        status = data_set['status'],
        proportion = config['status_threshold']
        )
    
    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints,
                                x_a, y_a)
    
    # Recompute saccadic intervals
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size = np.ceil(config['min_sac_duration']*s_f),
        status = data_set['status'],
        proportion = config['status_threshold']
        )
    
    if config['verbose']:
        print('   Fixations ans saccades identified using availability status threshold: {s_th}'.format(s_th=config['status_threshold']))
    
    assert len(f_ints) == len(ctrds), "Interval set and centroid set have different lengths"
    
    # Keep track of index that were effectively labeled
    i_lab = np.array([False]*config['nb_samples'])  
    
    for f_int in f_ints:
        i_lab[f_int[0]: f_int[1]+1] = True
        
    for s_int in s_ints:
        i_lab[s_int[0]: s_int[1]+1] = True
         
    return dict({
        'is_labeled': i_lab,
        'fixation_intervals': f_ints, 
        'saccade_intervals': s_ints,
        'centroids': ctrds, 
            })
  
    
cdef expand_cluster (double[:,:] g_npts, bool euclidean, 
                     int n_s, int idx, 
                     list neigh, double d_t, 
                     int n_w, dict avlb):
     
    cdef list l_C_clus = [idx]
    cdef list n_neigh = []
    
    for neigh_idx in neigh:

        if avlb[neigh_idx] == True:
            
            l_C_clus.append(neigh_idx)
            avlb[neigh_idx] = False
                
        n_neigh = vareps_neighborhood (g_npts, euclidean,
                                       n_s, neigh_idx, 
                                       d_t, n_w)    
        
        if len(n_neigh) > n_w:
             
            for key in n_neigh :
                
                if key not in neigh:
                    neigh.append(key)
     
    return l_C_clus, avlb


cdef list vareps_neighborhood (double[:,:] g_npts, 
                               bool euclidean, 
                               int n_s, int idx, 
                               double d_t, int n_w):
 
    cdef list neigh = []
    cdef double[:] ref_g_npts = g_npts[:,idx]

    #to the right
    cdef double d_r = 0.0
    cdef int r = idx
    
    cdef double d_l = 0.0
    cdef int l = idx
    
    cdef double n_1 = math.sqrt(ref_g_npts[0]**2 
                                + ref_g_npts[1]**2 
                                + ref_g_npts[2]**2)
    cdef double n_2 = 0.0
    
    cdef double ad_r = 0.0
    cdef double ad_d = 0.0
 
    if euclidean == True:
        
        with nogil:
            
            while r+1 < min(n_s, idx+n_w+1):  
                if d_r < d_t: 
                    
                    r = r+1  
                    d_r = math.sqrt((ref_g_npts[0]-g_npts[0,r])**2 
                                    + (ref_g_npts[1]-g_npts[1,r])**2 
                                    + (ref_g_npts[2]-g_npts[2,r])**2)
                else:
                    break
            
    else:
        
        with nogil:
             
            while r+1 < min(n_s, idx+n_w+1):   
                if d_r < d_t: 
            
                    r = r+1   
                    n_2 = math.sqrt(g_npts[0,r]**2 
                                    + g_npts[1,r]**2 
                                    + g_npts[2,r]**2)
                    
                    ad_r = math.acos((ref_g_npts[0]*g_npts[0,r]
                                      + ref_g_npts[1]*g_npts[1,r]
                                      + ref_g_npts[2]*g_npts[2,r]) 
                                     / (n_1*n_2))
                    d_r = math.fabs(ad_r/(2*math.pi)*360)
                    
                else:
                    break
    
    #to the left 
    if euclidean == True:
        
        with nogil:
            
            while l > max(0, idx-n_w-1):
                
                if d_l < d_t: 
         
                    l -= 1 
                    d_l = math.sqrt((ref_g_npts[0]-g_npts[0,l])**2 
                                    + (ref_g_npts[1]-g_npts[1,l])**2 
                                    + (ref_g_npts[2]-g_npts[2,l])**2)
                else:
                    break
            
    else:
        
        with nogil:
            
            while l > max(0, idx-n_w-1):
                
                if d_l < d_t: 
         
                    l -= 1 
                    n_2 = math.sqrt(g_npts[0,l]**2 
                                    + g_npts[1,l]**2 
                                    + g_npts[2,l]**2)
                    
                    ad_r = math.acos((ref_g_npts[0]*g_npts[0,l]
                                      + ref_g_npts[1]*g_npts[1,l]
                                      + ref_g_npts[2]*g_npts[2,l]) 
                                     / (n_1*n_2))
                    d_l = math.fabs(ad_r/(2*math.pi)*360)
                    
                else:
                    break
               
    neigh += [i for i in range(idx+1, r)]
    neigh += [i for i in range(l+1, idx)]
    neigh = sorted(neigh)

    return neigh


 
















