# -*- coding: utf-8 -*-

import numpy as np 

from Vision.segmentation_src.processing.binary_segmentation import BinarySegmentation 
from Vision.segmentation_src.utils.segmentation_utils import interval_merging 


class SaccadeTask(BinarySegmentation):
    
    def __init__(self, input_df, 
                 sampling_frequency, segmentation_method, 
                 **kwargs
                 ):
        
        super().__init__(input_df, 
                         sampling_frequency, segmentation_method, 
                         **kwargs)
 
        
        self.process()
   
    
    def latencies(self):
     
        return    
    
    
    def latency_quantiles(self):
     
        return   
    
    
    def gain(self):
     
        return    
    
    
    
    
    
    
    