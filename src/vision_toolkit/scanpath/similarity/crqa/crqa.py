# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt  
from scipy.spatial.distance import cdist
  
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.scanpath.single.rqa.rqa_base import RecurrenceBase
from vision_toolkit.visualization.scanpath.similarity.crqa import ( 
    plot_CRQA_determinism,
    plot_CRQA_laminarity)
 
 

class CRQAAnalysis(RecurrenceBase):
    
    def __init__(self, input, **kwargs):
        """
        Inputs:
            - seq_1/seq_2 = pair of sequences to analyze  
            - dist_threshold = threshold used to define a recurrence between to points
            - min lengths = length threshold used to define a line  
        """
        
        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)

        if verbose:
            print("Processing CRQA Analysis...\n")

        
        assert isinstance(input, list), "Input must be a list of two .csv, BinarySegmentation, or Scanpath"
        
        if isinstance(input[0], str):
            self.scanpath_1 = Scanpath.generate(input[0], **kwargs)
            self.scanpath_2 = Scanpath.generate(input[1], **kwargs)

        elif isinstance(input[0], BinarySegmentation):
            self.scanpath_1 = Scanpath.generate(input[0], **kwargs)
            self.scanpath_2 = Scanpath.generate(input[1], **kwargs)

        elif isinstance(input[0], Scanpath):
            self.scanpath_1 = input[0]
            self.scanpath_2 = input[1]
        
        d_thrs = (
            np.linalg.norm(
                np.array(
                    [
                        self.scanpath_1.config["size_plan_x"],
                        self.scanpath_1.config["size_plan_y"],
                    ]
                )
            )
            * 0.015
        )
    
        self.scanpath_1.config.update(
            {
                "scanpath_CRQA_distance_threshold": kwargs.get(
                    "scanpath_CRQA_distance_threshold", d_thrs
                ),
                "scanpath_CRQA_minimum_length": kwargs.get(
                    "scanpath_CRQA_minimum_length", 3
                ),
                "verbose": verbose
            }
        )
        
        super().__init__([self.scanpath_1.values, 
                          self.scanpath_2.values])
    
       
        # Compute the recurrence matrix r_m
        self.r_m = self.comp_recurrence_matrix()
         
        # Compute the set of vertical lines
        self.v_set = self.find_lines(self.r_m, 
                                     self.scanpath_1.config["scanpath_CRQA_minimum_length"],
                                     'vertical')
        
        # Compute the set of horizontal lines
        self.h_set = self.find_lines(self.r_m.T, 
                                     self.scanpath_1.config["scanpath_CRQA_minimum_length"],
                                     'horizontal')
    
       
        # Compute the set of diagonal lines
        r_m2 = np.zeros((self.n_1, 2*self.n_2))
        r_m2[:,self.n_2:] = self.r_m
        
        self.d_set = self.find_diags(r_m2, 
                                     self.scanpath_1.config["scanpath_CRQA_minimum_length"],
                                     full = True)
     
        if display_results:
            self.plot_CRQA()
             
        if verbose:
            print("...CRQA Analysis done\n")
   
         
    def comp_recurrence_matrix(self):
         
        s_1 = self.s_1[0:2]
        s_2 = self.s_2[0:2] 
         
        d_m = cdist(s_1.T, s_2.T,
                    metric='euclidean')
    
        r_m = (d_m < self.scanpath_1.config["scanpath_CRQA_distance_threshold"]).astype(int) 
        
        return r_m 
    
     
    def scanpath_CRQA_recurrence_rate(self, display_results):
        
        self.scanpath_1.config.update({"display_results": display_results})
        r_r = 100*np.sum(self.r_m)/(self.n_1 * self.n_2)
        
        if np.isnan(r_r):
            results = dict({"RQA_recurrence_rate": 0})

        else:
            results = dict({"RQA_recurrence_rate": r_r})

        self.scanpath_1.verbose()

        return results 
    
 
    def scanpath_CRQA_laminarity(self, display_results, 
                                 direction='vertical'):
        
        assert direction in ['horizontal', 'vertical'], "Laminarity direction must be 'horizontal' or 'vertical'"
        
        if direction == 'vertical': 
            set_ = self.v_set
        
        elif direction == 'horizontal': 
            set_ = self.h_set
            
        s_l = 0
        
        for l_ in set_:    
            s_l += len(l_)
            
        lam = 100 * s_l/np.sum(self.r_m)
            
        if self.scanpath_1.config["display_results"]:
            plot_CRQA_laminarity(self.r_m, set_)

        if np.isnan(lam):
            results = dict({"CRQA_laminarity": 0})

        else:
            results = dict({"CRQA_laminarity": lam})

        self.scanpath_1.verbose()

        return results
    
    
    def scanpath_CRQA_determinism(self, display_results):
         
        set_ = self.d_set 
        s_d = 0
    
        for l_ in set_:    
            s_d += len(l_)
            
        det = (100 * s_d)/np.sum(self.r_m)
            
        if self.scanpath_1.config["display_results"]:
            plot_CRQA_determinism(self.r_m, self.d_set)

        if np.isnan(det):
            results = dict({"CRQA_determinism": 0})

        else:
            results = dict({"CRQA_determinism": det})

        self.scanpath_1.verbose()

        return results
    
      
    def scanpath_CRQA_entropy(self, display_results):
        
        l_s = np.array(
            [len(d) for d in self.d_set]
            )
 
        u_, c_ = np.unique(l_s, 
                           return_counts=True)
        p_ = c_/len(l_s) 
    
        entropy = 0
        
        for p in list(p_):
            entropy -= p * np.log(p) 
            
        if np.isnan(entropy):
            results = dict({"CRQA_entropy": 0})

        else:
            results = dict({"CRQA_entropy": entropy})

        self.scanpath_1.verbose()

        return results
    
    
    def plot_CRQA(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(self.r_m[:45,:45],  
                      cmap='GnBu', 
                      edgecolors='w', 
                      linewidth=0.5)
        
        ax.set_aspect('equal')
        
        ax.set_xlabel('Second sequence fixation index', fontsize=12)
        ax.set_ylabel('First sequence fixation index', fontsize=12)
        
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        fig.suptitle("Cross-recurrence matrix", fontsize=18)
        
        plt.show() 
        plt.clf()
        
        
        
def scanpath_CRQA_recurrence_rate(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, CRQAAnalysis):
        results = input.scanpath_CRQA_recurrence_rate(display_results)

    else:
        geometrical_analysis = CRQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_CRQA_recurrence_rate(display_results)

    return results


def scanpath_CRQA_laminarity(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, CRQAAnalysis):
        results = input.scanpath_CRQA_laminarity(display_results)

    else:
        geometrical_analysis = CRQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_CRQA_laminarity(display_results)

    return results


def scanpath_CRQA_determinism(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, CRQAAnalysis):
        results = input.scanpath_CRQA_determinism(display_results)

    else:
        geometrical_analysis = CRQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_CRQA_determinism(display_results)

    return results

 
def scanpath_CRQA_entropy(input, **kwargs):
    display_results = kwargs.get("display_results", True)

    if isinstance(input, CRQAAnalysis):
        results = input.scanpath_CRQA_entropy(display_results)

    else:
        geometrical_analysis = CRQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_CRQA_entropy(display_results)

    return results







