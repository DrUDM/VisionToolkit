# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

from scipy.ndimage import convolve  
import matplotlib.pyplot as plt 

from Vision.utils.binning import spatial_bin
from Vision.visualization.single_scanpath import plot_saliency_map

 
np.random.seed(141)


class SaliencyMap:

    def __init__(self, 
                 scanpaths, 
                 size_plan_x, size_plan_y,
                 comp_saliency_map = True):
        """
        Inputs:
            - scanpaths = scanpath or list of scanpaths to convert as a saliency map  
            - pixel_number = size of the square saliency map
        """
        
        if type(scanpaths) != list: 
            
            self.s_ = [scanpaths.values]
            config = scanpaths.config
            
        else: 
            self.s_ = [scanpath.values for scanpath in scanpaths]
            config = scanpaths[0].config
        
        self.display = config['display_results']
        self.std = config['saliency_gaussian_std']
        
        pixel_number = config['saliency_pixel_number']
        
        if (pixel_number % 2) == 0:
            self.p_n = pixel_number + 1
        
        else:
            self.p_n = pixel_number
            
        self.s_b = [] 
        
        for seq in self.s_: 
            
            self.s_b.append(spatial_bin(seq[0:2], 
                                        self.p_n, self.p_n,
                                        size_plan_x, size_plan_y))
        
        if comp_saliency_map:
            
            self.s_m = self.comp_saliency_map(self.s_b)
            
            if self.display:
                plot_saliency_map(self.s_m)
        
        
        
    def comp_saliency_map(self, s_b):
          
        f_m = np.zeros((self.p_n, self.p_n))
        
        for s in s_b:
            
            l_f_m = np.zeros((self.p_n, self.p_n))
            unique, counts = np.unique(s[0:2], return_counts=True, axis = 1)
           
            for i, coord in enumerate(unique.T):
    
                l_f_m[int(coord[1]), int(coord[0])] = counts[i]
                
            f_m = f_m + l_f_m
            
        f_m = f_m/len(s_b)
        
        def gkern(kernlen=self.p_n, std=self.std):
            """
            Returns a 2D Gaussian kernel array.
            """
            
            gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
            gkern2d = np.outer(gkern1d, gkern1d)
            
            return gkern2d
        
        
        s_m = convolve(f_m, gkern()) 
        s_m = s_m/np.sum(s_m.flatten())
        
        plt.imshow(s_m) 
        
        return s_m
    
 


       