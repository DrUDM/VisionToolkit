# distutils: language = c++

from libcpp.map cimport map
from libcpp.pair cimport pair as cpair 
from libcpp cimport bool
 
import numpy as np



def longest_common_subsequence (list s_1, list s_2) :
     
    tmp_1, tmp_2 = int_convert(s_1, s_2) 
    cdef int[:] s_1_c = tmp_1 
    cdef int[:] s_2_c = tmp_2
      
    cdef int n_1 = len(s_1)
    cdef int n_2 = len(s_2)
 
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double) 
    ## Declare a path matrix for backtracking
    cdef int[:,:,:] b_map = np.zeros((n_1 + 1, n_2 + 1, 2), 
                                     dtype=np.intc) 
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0 
     
    cdef double w_s = 0.0
    cdef double w_i = 0.0
 
    cdef int i_ = 0
    cdef int j_ = 0
    
    cdef double tmp = 0.0
  
    ## Fill the D-matrix
    with nogil: 
        for i_ in range (1, n_1 + 1):    
            b_map[i_, 0, 0] = i_-1
            b_map[i_, 0, 1] = 0
            
            for j_ in range (1, n_2 + 1): 
                if i_ == 1: 
                    b_map[0, j_, 0] = 0
                    b_map[0, j_, 1] = j_-1
  
                if s_1_c[i_-1] == s_2_c[j_-1]:  
                    tmp = d_mat[i_-1, j_-1] + 1
                    d_mat[i_, j_] = tmp
                    
                    b_map[i_, j_, 0] = i_-1
                    b_map[i_, j_, 1] = j_-1
   
                else: 
                    w_s = d_mat[i_-1, j_]
                    s_s_0 = i_-1
                    s_s_1 = j_
                    
                    w_i = d_mat[i_, j_-1] 
                    if w_i > w_s: 
                        w_s = w_i
                        s_s_0 = i_
                        s_s_1 = j_-1
                      
                    d_mat[i_, j_] = w_s
                    b_map[i_, j_, 0] = s_s_0
                    b_map[i_, j_, 1] = s_s_1
             
    common_s, opt_align = character_generic_backtracking(s_1, s_2,
                                                         d_mat, b_map,
                                                         n_1, n_2) 
    ## To return
    cdef double lcs_s = d_mat[n_1, n_2]
                
    return common_s, opt_align, lcs_s 


def smith_waterman (list s_1, list s_2,  
                    dict dict_chr_idx, double[:,:] sim_mat,
                    double del_c_base, double del_c, 
                    double sim_weigth):
     
    tmp_1, tmp_2, n_dict = int_convert_from_dict(s_1, s_2,
                                                 dict_chr_idx) 
    cdef int[:] s_1_c = tmp_1
    cdef int[:] s_2_c = tmp_2
        
    ## Convert python dict to c dict for nogil loops
    cdef map[int, int] c_dict = dict_to_cmap(n_dict)
    
    ## Declare some constants
    cdef int n_1 = len(s_1)
    cdef int n_2 = len(s_2)
 
    ## Declare the D matrix
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double)
   
    # Declare a path matrix for backtracking
    cdef int[:,:,:] b_map = np.zeros((n_1 + 1, n_2 + 1, 2), 
                                     dtype=np.intc)
 
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef double w_s = 0.0
    cdef double w_t = 0.0
    
    cdef double w_k = 0.0
    cdef double w_k_s = 0.0
    
    cdef double w_l = 0.0
    cdef double w_l_s = 0.0
    
    cdef int i_ = 0
    cdef int j_ = 0 
    
    cdef int k_ = 0
    cdef int l_ = 0 
    
    cdef int k_s = 0
    cdef int l_s = 0 
    
    ## Fill the D-matrix
    with nogil: 
        for i_ in range (1, n_1+1):  
            for j_ in range (1, n_2+1): 
                ## Compute substitution cost
                w_s = d_mat[i_-1, j_-1] + sim_weigth * sim_mat[c_dict[s_1_c[i_-1]],
                                                               c_dict[s_2_c[j_-1]]]
                s_s_0 = i_-1
                s_s_1 = j_-1
                
                ## Explore column
                w_k_s = 0.0 
                for k_ in range (1, i_+1): 
                    w_k = d_mat[i_-k_, j_] - (del_c_base + k_*del_c) 
                    if w_k >= w_k_s:
                        w_k_s = w_k
                        k_s = k_ 
                if w_k_s > w_s: 
                    w_s = w_k_s
                    s_s_0 = i_-k_s
                    s_s_1 = j_      
                  
                ## Explore row
                w_l_s = 0.0
                for l_ in range (1, j_+1): 
                    w_l = d_mat[i_, j_-l_] - (del_c_base + l_*del_c)
                    if w_l >= w_l_s:
                        w_l_s = w_l
                        l_s = l_ 
                if w_l_s > w_s: 
                    w_s = w_l_s
                    s_s_0 = i_ 
                    s_s_1 = j_-l_s 
   
                ## If no similarity, terminate the process
                ## Usefull only if negative similarity values 
                if w_t > w_s: 
                    w_s = w_t
                    s_s_0 = 0
                    s_s_1 = 0  
                ## Add best score and keep track 
                d_mat[i_, j_] = w_s
                b_map[i_, j_, 0] = s_s_0
                b_map[i_, j_, 1] = s_s_1
     
    ## Fast enough using Python
    d_mat_p = np.asarray(d_mat)  
    ## Find best score value in the D-matrix
    i_m, j_m = np.unravel_index(np.argmax(d_mat_p), d_mat_p.shape)

    ## To return 
    cdef double sw_s = d_mat_p[i_m, j_m] 
    ## To manipulate
    cdef double local_v = d_mat_p[i_m, j_m]
           
    opt_align, common_s = [], []   
    i = np.copy(i_m)
    j = np.copy(j_m) 
    
    ## Compute non-generic backtracking using the B map
    while local_v > 0: 
        i_n = b_map[i, j, 0]
        j_n = b_map[i, j, 1]
        
        if i_n == i-1: 
            if j_n == j-1: 
                ## Compute alignement
                opt_align.insert(0, [s_1[i-1], s_2[j-1]])
                common_s.insert(0, [s_1[i-1], s_2[j-1]])
          
        if i_n == i: 
            k_m = j - j_n 
            for k in range(i, k_m+1):
                opt_align.insert(0, ['_', s_2[j-k]])
          
        if j_n == j: 
            k_m = i - i_n 
            for k in range(1, k_m):
                opt_align.insert(0, [s_1[i-k], '_']) 
        i = i_n
        j = j_n
        local_v = d_mat_p[i, j]
           
    return common_s, opt_align, sw_s


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


def int_convert(list s_1, list s_2):
    
    dict_str_int = dict() 
    for i, str_ in enumerate(sorted(set(s_1 + s_2))): 
        dict_str_int.update({str_: i})
    
    ## Convert input lists of str to list of int
    tmp_1 = np.array(
        [dict_str_int[s_1[i]] for i in range(len(s_1))], 
        dtype=np.int32
            ) 
    tmp_2 = np.array(
        [dict_str_int[s_2[j]] for j in range(len(s_2))], 
        dtype=np.int32
            ) 
    
    return tmp_1, tmp_2


def int_convert_from_dict(list s_1, list s_2,
                          dict dict_chr_idx):
    
    ## Create dict from str to int
    dict_str_int = dict()
    
    for i, str_ in enumerate(dict_chr_idx.keys()): 
        dict_str_int.update({str_: i})
     
    ## Convert input lists of str to list of int
    tmp_1 = np.array(
        [dict_str_int[s_1[i]] for i in range(len(s_1))], 
        dtype=np.int32
            ) 
    tmp_2 = np.array(
        [dict_str_int[s_2[j]] for j in range(len(s_2))], 
        dtype=np.int32
            ) 
    ## Create dict from int to indexes for the dist_mat
    n_dict = dict() 
    for key, val in dict_chr_idx.items(): 
        n_key = dict_str_int[key]
        n_dict.update({n_key: val})
        
    return tmp_1, tmp_2, n_dict
    
      
def character_generic_backtracking(list s_1, list s_2,
                                   double[:,:] d_mat, int[:,:,:] b_map,
                                   int n_1, int n_2):
    
    opt_align, common_s = [], []
    i, j = n_1, n_2 
    
    ## Fast enough using Python 
    while i != 0 or j != 0: 
        i_n = b_map[i, j, 0]
        j_n = b_map[i, j, 1] 
        if i_n == i-1: 
            if j_n == j-1:
                opt_align.insert(0, [s_1[i-1], s_2[j-1]]) 
                common_s.insert(0, [s_1[i-1], s_2[j-1]])
            else:
                opt_align.insert(0, [s_1[i-1], '_']) 
        else:
            opt_align.insert(0, ['_', s_2[j-1]])
        
        ## Iterate
        i, j = i_n, j_n 
        
    return common_s, opt_align


 