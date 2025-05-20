# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

from scipy.ndimage import convolve  
from scipy.stats import multivariate_normal 

from Vision.scanpath_src.single.saliency.saliency_map_base import SaliencyMap

 

class SaliencyReference(SaliencyMap):
    
    def __init__(self, scanpath, 
                 size_plan_x, size_plan_y,
                 comp_saliency_map = True):
        
        super().__init__(scanpath, 
                         size_plan_x, size_plan_y, 
                         comp_saliency_map)
        
        self.delta = scanpath.config['normalized_scanpath_saliency_delta']
        
         
    def comp_percentile(self, ref_sm):
       
        plt.imshow(ref_sm)
        plt.show()
        plt.clf()
        
        r_sm_f = ref_sm.flatten()  
        k_b = len(r_sm_f) 
         
        p_ = 0
        
        # Get indexes in the reference saliency map of the new scanpath
        s_b = self.s_b[0].astype(int) 
        j_b = len(s_b[0])
        
        for i in range(j_b):
            
            # Get the saliency value of the reference saliency map in which 
            # the i-th fixation falls 
            h_i = ref_sm[s_b[1,i], s_b[0,i]] 
            inf_ = r_sm_f <= h_i 
            p_ += sum(inf_)/k_b
         
        perc_ = 100*p_/j_b
        
        return perc_
        
        
    def comp_nss(self, ref_sm):
        
        plt.imshow(ref_sm)
        plt.show()
        plt.clf()
        
        delta = self.delta
        # Delta parameter to compute a neighbourhood arround each fixation
        delta = int(np.ceil((self.p_n*delta)/2)) 
        
        r_sm_f = ref_sm.flatten()  
         
        mu = np.mean(r_sm_f) 
        sigma = np.std(r_sm_f)
        
        ref_sm = (ref_sm - mu)/sigma
         
        nss = 0
        s = delta/3
        cov_m = np.array([[s**2,0],
                          [0,s**2]])
        var = multivariate_normal(mean=[0,0], cov=cov_m)
 
        
        # Get indexes in the reference saliency map of the new scanpath
        s_b = self.s_b[0].astype(int) 
        j_b = len(s_b[0])
        
        for i in range(j_b):
            
            # Initialize local NSS
            l_nss = 0
            
            # Initialize mass
            m_ = 0
            
            # Get the coordinates of the reference saliency map in which 
            # the i-th fixation falls
            (x_i, y_i) = (s_b[0,i], s_b[1,i])
            
            for x in range(max(x_i-delta, 0), min(x_i+delta+1, self.p_n)):
                for y in range(max(y_i-delta, 0), min(y_i+delta+1, self.p_n)):
                    
                    g_p = var.pdf([x-x_i,
                                   y-y_i])
                    l_nss += ref_sm[y, x]*g_p
                    m_ += g_p
            
            nss += l_nss/m_
            
        return nss/j_b   
 
    
     




















