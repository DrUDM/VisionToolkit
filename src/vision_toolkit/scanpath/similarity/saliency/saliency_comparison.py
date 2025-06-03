# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from scipy.ndimage import convolve  
from scipy.stats import multivariate_normal 

from vision_toolkit.scanpath.single.saliency.saliency_map_base import SaliencyMap

 

class PairSaliencyMap:
    
    def __init__(self, 
                 scanpath_1, scanpath_2,  
                 x_size, y_size,
                 **kwargs):
       
        kwargs.update(size_plan_x=x_size,
                      size_plan_y=y_size)
        
        self.sm_1 = SaliencyMap(scanpath_1, comp_saliency_map = True,
                                **kwargs)
        
        self.sm_2 = SaliencyMap(scanpath_2, comp_saliency_map = True,
                                **kwargs)
        
      
    def comp_pearson_corr(self):
        
        s_m_1 = self.sm_1.s_m.flatten()
        s_m_2 = self.sm_2.s_m.flatten()
        
        S = np.stack((s_m_1, s_m_2), axis=0)
        cov_ = np.cov(S)[0,1]
     
        p_c = cov_/(np.std(s_m_1) * np.std(s_m_2))
        
        return p_c
    
       
    def comp_kl_divergence(self):
        
        s_m_1 = self.sm_1.s_m.flatten()
        s_m_2 = self.sm_2.s_m.flatten()
        
        kl_d = np.sum(s_m_1 * np.log(s_m_1/s_m_2)) 
        
        return kl_d
    
    
    

class SaliencyReference(SaliencyMap):
    
    def __init__(self, input, ref_saliency_map,  
                 comp_saliency_map = False, 
                 **kwargs):
        
        super().__init__(input,  
                         comp_saliency_map, 
                         **kwargs)
        
        if isinstance(ref_saliency_map, dict):
            self.ref_sm = ref_saliency_map['salency_map']
        else :
            self.ref_sm = ref_saliency_map
            
        
        assert self.p_n == self.ref_sm.shape[0], "The reference map and scanpath saliency map must have the same dimensions"
        #self.delta = scanpath.config['normalized_scanpath_saliency_delta']
        
         
    def scanpath_saliency_percentile(self):
       
        ref_sm = self.ref_sm 
        r_sm_f = ref_sm.flatten()  
        k_b = len(r_sm_f) 
         
        p_ = 0
        
        # Get indexes in the reference saliency map of the new scanpath
        s_b = self.s_b[0].astype(int) 
        j_b = len(s_b[0])
        
        for i in range(j_b): 
            h_i = ref_sm[s_b[1,i], s_b[0,i]] 
            inf_ = r_sm_f < h_i 
            
            p_ += sum(inf_)/k_b
         
        perc_ = 100*p_/j_b
        
        self.scanpaths[0].verbose()
             
        return dict({'percentile': perc_})   
        
        
    def scanpath_saliency_nss(self, 
                              sigma_kernel, delta_neighborhood):
        
        ref_sm = self.ref_sm 
        r_sm_f = ref_sm.flatten()  
         
        mu = np.mean(r_sm_f) 
        sigma = np.std(r_sm_f)
        
        ref_sm = (ref_sm - mu)/sigma
         
         
   
        cov_m = np.array([[sigma_kernel**2,0],
                          [0,sigma_kernel**2]])
        var = multivariate_normal(mean=[0,0], cov=cov_m)
 
        
        # Get indexes in the reference saliency map of the new scanpath
        s_b = self.s_b[0].astype(int) 
        j_b = len(s_b[0])
    
        nss = 0
        for i in range(j_b):
            
            # Initialize local NSS
            l_nss = 0
            
            # Initialize mass
            m_ = 0
            
            # Get the coordinates of the reference saliency map in which 
            # the i-th fixation falls
            (x_i, y_i) = (s_b[0,i], s_b[1,i])
            
            for x in range(max(x_i-delta_neighborhood, 0), 
                           min(x_i+delta_neighborhood+1, self.p_n)):
                for y in range(max(y_i-delta_neighborhood, 0), 
                               min(y_i+delta_neighborhood+1, self.p_n)):
                    
                    g_p = var.pdf([x-x_i,
                                   y-y_i])
                    l_nss += ref_sm[y, x]*g_p
                    m_ += g_p
            
            nss += l_nss/m_
            
        self.scanpaths[0].verbose(dict({'scanpath_saliency_nss_sigma_kernel': sigma_kernel,
                                    'scanpath_saliency_nss_delta': delta_neighborhood
                              }))
        
        return dict({'nss': nss/j_b})   
 
    
 
def infogain(s_map,gt,baseline_map):
	 
	# assuming s_map and baseline_map are normalized
	eps = 2.2204e-16

	s_map = s_map/(np.sum(s_map)*1.0)
	baseline_map = baseline_map/(np.sum(baseline_map)*1.0)

	# for all places where gt=1, calculate info gain
	temp = []
	x,y = np.where(gt==1)
	for i in zip(x,y):
		temp.append(np.log2(eps + s_map[i[0],i[1]]) - np.log2(eps + baseline_map[i[0],i[1]]))

	return np.mean(temp)



def auc_judd(s_map,gt):
	# ground truth is discrete, s_map is continous and normalized
	 
	# thresholds are calculated from the salience map, only at places where fixations are present
	thresholds = []
	for i in range(0,gt.shape[0]):
		for k in range(0,gt.shape[1]):
			if gt[i][k]>0:
				thresholds.append(s_map[i][k])

	
	num_fixations = np.sum(gt)
	# num fixations is no. of salience map values at gt >0


	thresholds = sorted(set(thresholds))
	
	#fp_list = []
	#tp_list = []
	area = []
	area.append((0.0,0.0))
	for thresh in thresholds:
		# in the salience map, keep only those pixels with values above threshold
		temp = np.zeros(s_map.shape)
		temp[s_map>=thresh] = 1.0
		assert np.max(gt)==1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
		assert np.max(s_map)==1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
		num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
		tp = num_overlap/(num_fixations*1.0)
		
		# total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
		# this becomes nan when gt is full of fixations..this won't happen
		fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
		
		area.append((round(tp,4),round(fp,4)))
		#tp_list.append(tp)
		#fp_list.append(fp)

	#tp_list.reverse()
	#fp_list.reverse()
	area.append((1.0,1.0))
	#tp_list.append(1.0)
	#fp_list.append(1.0)
	#print tp_list
	area.sort(key = lambda x:x[0])
	tp_list =  [x[0] for x in area]
	fp_list =  [x[1] for x in area]
	return np.trapz(np.array(tp_list),np.array(fp_list))



def auc_borji(s_map,gt,splits=100,stepsize=0.1):
	 
	num_fixations = np.sum(gt)

	num_pixels = s_map.shape[0]*s_map.shape[1]
	random_numbers = []
	for i in range(0,splits):
		temp_list = []
		for k in range(0,num_fixations):
			temp_list.append(np.random.randint(num_pixels))
		random_numbers.append(temp_list)

	aucs = []
	# for each split, calculate auc
	for i in random_numbers:
		r_sal_map = []
		for k in i:
			r_sal_map.append(s_map[k%s_map.shape[0]-1, k/s_map.shape[0]])
		# in these values, we need to find thresholds and calculate auc
		thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

		r_sal_map = np.array(r_sal_map)

		# once threshs are got
		thresholds = sorted(set(thresholds))
		area = []
		area.append((0.0,0.0))
		for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
			temp = np.zeros(s_map.shape)
			temp[s_map>=thresh] = 1.0
			num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
			tp = num_overlap/(num_fixations*1.0)
			
			#fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
			fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

			area.append((round(tp,4),round(fp,4)))
		
		area.append((1.0,1.0))
		area.sort(key = lambda x:x[0])
		tp_list =  [x[0] for x in area]
		fp_list =  [x[1] for x in area]

		aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))
	
	return np.mean(aucs)
 


def scanpath_saliency_percentile(input, reference_map,
                                 **kwargs):
      
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_percentile()
    else:
        perc_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = perc_i.scanpath_saliency_percentile()
        
    return results


def scanpath_saliency_nss(input, reference_map,
                          **kwargs):
    
    scanpath_saliency_nss_sigma_kernel = kwargs.get(
        "scanpath_saliency_nss_sigma_kernel", .5
    )
    scanpath_saliency_nss_delta = kwargs.get(
        "scanpath_saliency_nss_delta", 1
    )
   

    display_results = kwargs.get("display_results", True)

    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_nss(
            scanpath_saliency_nss_sigma_kernel, 
            scanpath_saliency_nss_delta
        )
    else:
        nss_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = nss_i.scanpath_saliency_nss(
            scanpath_saliency_nss_sigma_kernel, 
            scanpath_saliency_nss_delta
        )
    return results











