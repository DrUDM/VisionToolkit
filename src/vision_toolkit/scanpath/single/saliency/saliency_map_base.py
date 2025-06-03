# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

from scipy.ndimage import convolve  
import matplotlib.pyplot as plt 

from vision_toolkit.utils.binning import spatial_bin
from vision_toolkit.visualization.scanpath.single.saliency import plot_saliency_map

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation

 
 

class SaliencyMap:

    def __init__(self, 
                 input,  
                 comp_saliency_map = True, 
                 **kwargs):
        """
        Inputs:
            - scanpaths = scanpath or list of scanpaths to convert as a saliency map  
            - pixel_number = size of the square saliency map
        """
        
        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)
        
      
        std = kwargs.get(
            "scanpath_saliency_gaussian_std", 5)
        pixel_number = kwargs.get(
            "scanpath_saliency_pixel_number", 100)
        
        if isinstance(input, list):
            
            if isinstance(input[0], str):
                scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

            elif isinstance(input[0], BinarySegmentation):
                scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

            elif isinstance(input[0], Scanpath):
                scanpaths = input
            
            else:
                raise ValueError(
                    "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
                )
        
        else:
            if isinstance(input, str):
                scanpaths = [Scanpath.generate(input, **kwargs)]

            elif isinstance(input, BinarySegmentation):
                scanpaths = [Scanpath.generate(input, **kwargs)]

            elif isinstance(input, Scanpath):
                scanpaths = [input]
            
            else:
                raise ValueError(
                    "Input must be a a Scanpath, or a BinarySegmentation, or a csv"
                )
             
        self.scanpaths = scanpaths
        
        self.scanpaths[0].config.update(
            {
                "scanpath_saliency_gaussian_std": std,
                "scanpath_saliency_pixel_number": pixel_number,
                "verbose": verbose,
                "display_results": display_results
            }
        )
        
        self.saliency_map = None 
        
        self.size_plan_x = self.scanpaths[0].config['size_plan_x']
        self.size_plan_y = self.scanpaths[0].config['size_plan_y']
        
        for scanpath in self.scanpaths:
            assert scanpath.config['size_plan_x'] == self.size_plan_x, 'All recordings must have the same "size_plan_x" and "size_plan_y" values'
            assert scanpath.config['size_plan_y'] == self.size_plan_y, 'All recordings must have the same "size_plan_x" and "size_plan_y" values'
        
          
        if (pixel_number % 2) == 0:
            self.p_n = pixel_number + 1
        
        else:
            self.p_n = pixel_number
        self.std = std
        
        self.s_b = [] 
        
        for seq in [scanpath.values for scanpath in self.scanpaths]: 
            self.s_b.append(spatial_bin(seq[0:2], 
                                        self.p_n, self.p_n,
                                        self.size_plan_x, self.size_plan_y))
        
        if comp_saliency_map: 
            self.saliency_map = self.comp_saliency_map(self.s_b)
             
            if display_results:
                plot_saliency_map(self.saliency_map)
         
        self.scanpaths[0].verbose()
        
        
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
    
    
  
    
 
def scanpath_saliency_map(input, **kwargs):
   
    saliency_map_i = SaliencyMap(input, **kwargs)
    results = dict({"salency_map": saliency_map_i.saliency_map})

    return results
       