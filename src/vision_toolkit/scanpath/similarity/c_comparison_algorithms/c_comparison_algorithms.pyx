# distutils: language = c++

from libcpp.map cimport map
from libcpp.pair cimport pair as cpair 
from libcpp cimport bool
 
import numpy as np


 
def DTW (double[:,:] s_1, double[:,:] s_2,
         double[:,:] dist_mat):
    
    cdef int n_1 = len(s_1.T)
    cdef int n_2 = len(s_2.T) 
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double)
    
    d_mat[1:,0] = np.inf
    d_mat[0,1:] = np.inf
    
    cdef int[:,:,:] b_map = np.zeros((n_1 + 1, n_2 + 1, 2), 
                                     dtype=np.intc) 
    cdef int i_ = 0
    cdef int j_ = 0
    
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef double c = 0.0
    cdef double w_s = 0.0
    cdef double w_d = 0.0
    cdef double w_i = 0.0
    
    with nogil: 
        for i_ in range (1, n_1 + 1): 
            for j_ in range(1, n_2 + 1): 
                c = dist_mat[i_ - 1, j_ - 1]
                
                w_s = c + d_mat[i_ - 1, j_ - 1]
                s_s_0 = i_ - 1
                s_s_1 = j_ - 1
                
                w_d = c + d_mat[i_ - 1, j_] 
                if w_d < w_s: 
                    w_s = w_d
                    s_s_0 = i_ - 1
                    s_s_1 = j_
                    
                w_i = c + d_mat[i_, j_ - 1]   
                if w_i < w_s: 
                    w_s = w_i
                    s_s_0 = i_
                    s_s_1 = j_ - 1
                    
                d_mat[i_, j_] = w_s
                b_map[i_, j_, 0] = s_s_0
                b_map[i_, j_, 1] = s_s_1
       
   
    o_l = []
    o_l.insert(0, [s_1[:,n_1-1], 
                   s_2[:,n_2-1],
                   [n_1-1, n_2-1]])
    
    opt_links = dtw_links_backtracking(s_1, s_2, o_l, 
                                       d_mat, b_map,  
                                       i = n_1, j = n_2) 
    cdef double l_s = d_mat[n_1, n_2]  
    
    return opt_links, l_s
 
    
def discrete_frechet(double[:,:] s_1, double[:,:] s_2,
                     double[:,:] dist_mat):
    
    cdef int n_1 = len(s_1.T)
    cdef int n_2 = len(s_2.T) 
    cdef double[:,:] d_mat = np.zeros((n_1, n_2), dtype=np.double)
    
    d_mat[0,0] = dist_mat[0,0] 
    
    cdef int[:,:,:] b_map = np.zeros((n_1, n_2, 2), 
                                     dtype=np.intc) 
    cdef int i__ = 0
    cdef int j__ = 0
    
    with nogil:
        for i__ in range(1, n_1): 
            d_mat[i__, 0] = max(d_mat[i__ - 1, 0], 
                                dist_mat[i__, 0]) 
            b_map[i__, 0, 0] = i__-1
            b_map[i__, 0, 1] = 0
    
    with nogil:
        for j__ in range(1, n_2): 
            d_mat[0, j__] = max(d_mat[0, j__ - 1], 
                                dist_mat[0, j__]) 
            b_map[0, j__, 0] = 0
            b_map[0, j__, 1] = j__-1
    
    cdef int i_ = 0
    cdef int j_ = 0
    
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef double c = 0.0
    cdef double w_s = 0.0
    cdef double w_d = 0.0
    cdef double w_i = 0.0
    
    with nogil: 
        for i_ in range (1, n_1):  
            for j_ in range(1, n_2): 
                c = dist_mat[i_, j_]
                
                w_s = d_mat[i_ - 1, j_ - 1]
                s_s_0 = i_ - 1
                s_s_1 = j_ - 1
                
                w_d = d_mat[i_ - 1, j_] 
                if w_d < w_s: 
                    w_s = w_d
                    s_s_0 = i_ - 1
                    s_s_1 = j_
                    
                w_i = d_mat[i_, j_ - 1]   
                if w_i < w_s: 
                    w_s = w_i
                    s_s_0 = i_
                    s_s_1 = j_ - 1
                    
                d_mat[i_, j_] = max(w_s, c)
                b_map[i_, j_, 0] = s_s_0
                b_map[i_, j_, 1] = s_s_1
       
   
    o_l = []
    o_l.insert(0, [s_1[:,n_1-1], 
                   s_2[:,n_2-1],
                   [n_1-1, n_2-1]])
    
    opt_links = frechet_links_backtracking(s_1, s_2, o_l, 
                                           d_mat, b_map,  
                                           i = n_1-1, j = n_2-1)
    
    cdef double l_s = d_mat[n_1 - 1, n_2 - 1]  
    
    return opt_links, l_s

               
def levenshtein (list s_1, list s_2,
                 double del_c = 1.0, double ins_c = 1.0, double sub_c = 1.0) :
 
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
    
    cdef int i_ = 0
    cdef int j_ = 0
    
    ## Fill the first column and row
    with nogil: 
        for i_ in range(n_1+1):
            d_mat[i_, 0] = i_ * del_c   
        for j_ in range(n_2+1):
            d_mat[0, j_] = j_ * ins_c
    
    cdef double w_s = 0.0
    cdef double w_d = 0.0
    cdef double w_i = 0.0
    
    cdef int i__ = 0
    cdef int j__ = 0
    
    cdef double tmp = 0.0
  
    ## Fill the D-matrix
    with nogil: 
        for i__ in range (1, n_1 + 1):  
            b_map[i__, 0, 0] = i__-1
            b_map[i__, 0, 1] = 0
            
            for j__ in range (1, n_2 + 1): 
                if i__ == 1: 
                    b_map[0, j__, 0] = 0
                    b_map[0, j__, 1] = j__-1
  
                if s_1_c[i__-1] == s_2_c[j__-1]:  
                    tmp = d_mat[i__-1, j__-1]
                    d_mat[i__, j__] = tmp
                    
                    b_map[i__, j__, 0] = i__-1
                    b_map[i__, j__, 1] = j__-1
   
                else: 
                    w_s = d_mat[i__-1, j__-1] + sub_c
                    s_s_0 = i__-1
                    s_s_1 = j__-1
                    
                    w_d = d_mat[i__-1, j__] + del_c 
                    if w_d < w_s: 
                        w_s = w_d
                        s_s_0 = i__-1
                        s_s_1 = j__
                        
                    w_i = d_mat[i__, j__-1] + ins_c 
                    if w_i < w_s: 
                        w_s = w_i
                        s_s_0 = i__
                        s_s_1 = j__-1
                        
                    d_mat[i__, j__] = w_s
                    b_map[i__, j__, 0] = s_s_0
                    b_map[i__, j__, 1] = s_s_1
              
    opt_align = character_generic_backtracking(s_1, s_2,
                                               d_mat, b_map,
                                               n_1, n_2)
    
    cdef double l_s = d_mat[n_1, n_2]  
 
    return opt_align, l_s


def generalized_edit (list s_1, list s_2, 
                      double del_c, double ins_c,
                      dict dict_chr_idx, double[:,:] dist_mat) :
    
    tmp_1, tmp_2, n_dict = int_convert_from_dict(s_1, s_2,
                                                 dict_chr_idx)
    
    cdef int[:] s_1_c = tmp_1
    cdef int[:] s_2_c = tmp_2
 
    ## Convert python dict to c dict for nogil loops
    cdef map[int, int] c_dict = dict_to_cmap(n_dict)
    
    ## Declare some constants
    cdef int n_1 = len(s_1)
    cdef int n_2 = len(s_2)
    
    cdef int i_ = 0
    cdef int j_ = 0 
    
    ## Declare the D matrix
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double)
 
    ## Fill the first column and row
    with nogil: 
        for i_ in range(n_1+1):
            d_mat[i_][0] = i_ * del_c 
        for j_ in range(n_2+1):
            d_mat[0][j_] = j_ * ins_c 
 
    cdef double w_s = 0.0
    cdef double w_d = 0.0
    cdef double w_i = 0.0
     
    ## Declare a path matrix for backtracking
    cdef int[:,:,:] b_map = np.zeros((n_1 + 1, n_2 + 1, 2), 
                                     dtype=np.intc)
 
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef int i__ = 0
    cdef int j__ = 0 
    
    ## Fill the D-matrix
    with nogil: 
        for i__ in range (1, n_1+1):   
            b_map[i__, 0, 0] = i__-1
            b_map[i__, 0, 1] = 0
            
            for j__ in range (1, n_2+1): 
                if i__ == 1: 
                    b_map[0, j__, 0] = 0
                    b_map[0, j__, 1] = j__-1
                 
                w_s = d_mat[i__-1, j__-1] + dist_mat[c_dict[s_1_c[i__-1]],
                                                            c_dict[s_2_c[j__-1]]]
                s_s_0 = i__-1
                s_s_1 = j__-1 
                
                w_d = d_mat[i__-1, j__] + del_c 
                if w_d < w_s: 
                    w_s = w_d
                    s_s_0 = i__-1
                    s_s_1 = j__
                 
                w_i = d_mat[i__, j__-1] + ins_c 
                if w_i < w_s: 
                    w_s = w_i
                    s_s_0 = i__
                    s_s_1 = j__-1
                 
                d_mat[i__, j__] = w_s
                b_map[i__, j__, 0] = s_s_0
                b_map[i__, j__, 1] = s_s_1
     
    opt_align = character_generic_backtracking(s_1, s_2,
                                               d_mat, b_map,
                                               n_1, n_2)
     
    cdef double wf_s = d_mat[n_1, n_2] 
  
    return opt_align, wf_s


def needleman_wunsch (list s_1, list s_2, 
                      double gap_c, double conc_b,
                      dict dict_chr_idx, double[:,:] dist_mat) :
  
    tmp_1, tmp_2, n_dict = int_convert_from_dict(s_1, s_2,
                                                 dict_chr_idx)
    
    cdef int[:] s_1_c = tmp_1
    cdef int[:] s_2_c = tmp_2
    
    ## Convert python dict to c dict for nogil loops
    cdef map[int, int] c_dict = dict_to_cmap(n_dict)
    
    ## Declare some constants
    cdef int n_1 = len(s_1)
    cdef int n_2 = len(s_2)
    
    cdef int i_ = 0
    cdef int j_ = 0 
    
    ## Declare the D matrix
    cdef double[:,:] d_mat = np.zeros((n_1+1, n_2+1), dtype=np.double)
 
    ## Fill the first column and row
    with nogil: 
        for i_ in range(n_1+1):
            d_mat[i_][0] = - i_ * gap_c
            
        for j_ in range(n_2+1):
            d_mat[0][j_] = - j_ * gap_c 

    
    ## Declare a path matrix for backtracking
    cdef int[:,:,:] b_map = np.zeros((n_1 + 1, n_2 + 1, 2), 
                                     dtype=np.intc)
 
    cdef int s_s_0 = 0
    cdef int s_s_1 = 0
    
    cdef double w_s = 0.0
    cdef double w_d = 0.0
        
    cdef int i__ = 0
    cdef int j__ = 0 
    
    ## Fill the D-matrix
    with nogil: 
        for i__ in range (1, n_1+1):  
            b_map[i__, 0, 0] = i__-1
            b_map[i__, 0, 1] = 0
                
            for j__ in range (1, n_2+1): 
                if i__ == 1: 
                    b_map[0, j__, 0] = 0
                    b_map[0, j__, 1] = j__-1
                    
                if s_1_c[i__-1] == s_2_c[j__-1]: 
                    w_s = d_mat[i__-1, j__-1] + conc_b 
                else: 
                    w_s = d_mat[i__-1, j__-1] - dist_mat[c_dict[s_1_c[i__-1]],
                                                         c_dict[s_2_c[j__-1]]]
                s_s_0 = i__-1
                s_s_1 = j__-1
                
                w_d = d_mat[i__, j__-1] - gap_c 
                if w_d > w_s: 
                    w_s = w_d
                    s_s_0 = i__
                    s_s_1 = j__-1
                    
                w_d = d_mat[i__-1, j__] - gap_c 
                if w_d > w_s: 
                    w_s = w_d
                    s_s_0 = i__-1
                    s_s_1 = j__    
                    
                d_mat[i__, j__] = w_s
                b_map[i__, j__, 0] = s_s_0
                b_map[i__, j__, 1] = s_s_1
     
    opt_align = character_generic_backtracking(s_1, s_2,
                                               d_mat, b_map,
                                               n_1, n_2) 
    cdef double nw_s = d_mat[n_1, n_2]           
         
    return opt_align, nw_s
 

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
    
    opt_align = []
    i, j = n_1, n_2 
    
    ## Fast enough using Python 
    while i != 0 or j != 0: 
        i_n = b_map[i, j, 0]
        j_n = b_map[i, j, 1] 
        if i_n == i-1: 
            if j_n == j-1:
                opt_align.insert(0, [s_1[i-1], s_2[j-1]]) 
            else:
                opt_align.insert(0, [s_1[i-1], '__']) 
        else:
            opt_align.insert(0, ['__', s_2[j-1]])
        
        i, j = i_n, j_n
 
    return opt_align


def dtw_links_backtracking(double[:,:] s_1, double[:,:] s_2,
                           list opt_links,
                           double[:,:] d_mat, int[:,:,:] b_map,
                           int i, int j):
    
    i_n = b_map[i, j, 0]
    j_n = b_map[i, j, 1] 
    opt_links.insert(0, [s_1[:,i_n-1], 
                         s_2[:,j_n-1],
                         [i_n-1, j_n-1]])
    
    if i_n == 1 and j_n == 1: 
        opt_links.insert(0, [s_1[:,0], 
                             s_2[:,0],
                             [0, 0]])
        
        return np.asarray(opt_links) 
    
    return dtw_links_backtracking(s_1, s_2, opt_links,
                                  d_mat, b_map, 
                                  i_n, j_n)


def frechet_links_backtracking(double[:,:] s_1, double[:,:] s_2,
                               list opt_links,
                               double[:,:] d_mat, int[:,:,:] b_map,
                               int i, int j):
    
    i_n = b_map[i, j, 0]
    j_n = b_map[i, j, 1] 
    opt_links.insert(0, [s_1[:,i_n], 
                   s_2[:,j_n],
                   [i_n, j_n]])
    
    if i_n == 0 and j_n == 0: 
        opt_links.insert(0, [s_1[:,0], 
                       s_2[:,0],
                       [0, 0]])
        
        return np.asarray(opt_links)  
    
    return frechet_links_backtracking(s_1, s_2, opt_links,
                                  d_mat, b_map, 
                                  i_n, j_n)
        
 