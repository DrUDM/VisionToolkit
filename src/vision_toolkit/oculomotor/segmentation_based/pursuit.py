# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.stats import norm, kurtosis, skew

from vision_toolkit.segmentation.processing.ternary_segmentation import TernarySegmentation 


class PursuitAnalysis(TernarySegmentation):
    
    def __init__(self, 
                 input_df, 
                 sampling_frequency, segmentation_method, 
                 **kwargs
                 ):
        
        
        super().__init__(input_df, 
                         sampling_frequency, segmentation_method, 
                         **kwargs)
        
        self.process() 
   
    
    @classmethod
    def generate(cls, input, **kwargs):
        if isinstance(input, PursuitAnalysis):
            return input
        kwargs_copy = kwargs.copy()
        sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
        segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
        return cls(input, sampling_frequency, segmentation_method, **kwargs_copy)
    
    
        
    def pursuit_count(self):
        
        ct = len(self.segmentation_results['pursuit_intervals'])
        results = dict({"count": ct})
        
        return results
    
    
    def pursuit_frequency(self):

        ct = len(self.segmentation_results['pursuit_intervals'])
        f = ct/(self.config['nb_samples']/self.config['sampling_frequency'])

        result = dict({"frequency": f})

        return result
    
    
    def pursuit_duration(self, get_raw):
        
        a_i = np.array(self.segmentation_results['pursuit_intervals']) + np.array([[0, 1]])
        a_d = a_i[:,1] - a_i[:,0]
        
        results = dict({
            'duration mean': np.mean(a_d), 
            'duration sd': np.std(a_d),  
            'raw': a_d
                })
        
        if not get_raw:
            del results["raw"]

        return results
    
    
    def pursuit_proportion(self):
        
        a_i = np.array(self.segmentation_results['pursuit_intervals']) + np.array([[0, 1]])
        a_d = (a_i[:,1] - a_i[:,0]).sum()
        
        return a_d/self.config['nb_samples']
    
    
    def pursuit_velocity(self, get_raw):
        
        _ints = self.segmentation_results['pursuit_intervals']
        a_sp = self.data_set['absolute_speed']
        
        l_sp = [] 
        for _int in _ints:       
            l_sp.extend(list(a_sp[_int[0]: _int[1]+1])) 
       
        results =  dict({
            'velocity mean': np.mean(np.array(l_sp)), 
            'velocity sd': np.std(np.array(l_sp)),  
            'raw': np.array(l_sp)
                })
        
        if not get_raw:
            del results["raw"]

        return results
    
    
    def pursuit_velocity_means(self, get_raw):
        
        _ints = self.segmentation_results['pursuit_intervals']
        a_sp = self.data_set['absolute_speed']
        
        m_sp = []
        for _int in _ints:       
            m_sp.append(np.mean(a_sp[_int[0]: _int[1]+1])) 
       
        results =  dict({
            'velocity mean mean': np.mean(np.array(m_sp)), 
            'velocity mean sd': np.std(np.array(m_sp)),  
            'raw': np.array(m_sp)
                })
        
        if not get_raw:
            del results["raw"]

        return results
    
    
    def pursuit_peak_velocity(self, get_raw):
        
        _ints = self.segmentation_results['pursuit_intervals']
        a_sp = self.data_set['absolute_speed']
        
        p_sp = []
        
        for _int in _ints:       
            p_sp.append(np.max(a_sp[_int[0]: _int[1]+1])) 
       
        results =  dict({
            'velocity peak mean': np.mean(np.array(p_sp)), 
            'velocity peak sd': np.std(np.array(p_sp)),  
            'raw': np.array(p_sp)
                }) 
        
        if not get_raw:
            del results["raw"]

        return results
    

    def pursuit_amplitude(self, get_raw):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.segmentation_results['pursuit_intervals']
        dist_ = self.distances[self.config['distance_type']]
        
        dsp = []
        for _int in _ints:
            
            s_p = np.array([x_a[_int[0]], y_a[_int[0]], z_a[_int[0]]])
            e_p = np.array([x_a[_int[1]], y_a[_int[1]], z_a[_int[1]]])
            
            dsp.append(dist_(s_p, e_p))
        
        results =  dict({
            'pursuit amplitude mean': np.mean(np.array(dsp)), 
            'pursuit amplitude sd': np.std(np.array(dsp)),  
            'raw': np.array(dsp)
                })
        
        if not get_raw:
            del results["raw"]

        return results
    
    
    def pursuit_distance(self, get_raw):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.segmentation_results['pursuit_intervals']
        dist_ = self.distances[self.config['distance_type']]
        
        t_cum = []
        for _int in _ints:
            
            l_cum = 0
            for k in range (_int[0], _int[1]):
                
                s_p = np.array([x_a[k], y_a[k], z_a[k]])
                e_p = np.array([x_a[k+1], y_a[k+1], z_a[k+1]])
                
                l_cum += dist_(s_p, e_p)
                
            t_cum.append(l_cum)
            
        results = dict({
            'pursuit cumul. distance mean': np.mean(np.array(t_cum)), 
            'pursuit cumul. distance sd': np.std(np.array(t_cum)),  
            'raw': np.array(t_cum)
                })
        
        if not get_raw:
            del results["raw"]

        return results
    
    
    def pursuit_efficiency(self, get_raw):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.segmentation_results['pursuit_intervals']
        dist_ = self.distances[self.config['distance_type']]
        
        eff = []
        for _int in _ints:
            
            s_p = np.array([x_a[_int[0]], y_a[_int[0]], z_a[_int[0]]])
            e_p = np.array([x_a[_int[1]], y_a[_int[1]], z_a[_int[1]]])
            
            s_amp = dist_(s_p, e_p)
            l_cum = 0
            
            for k in range (_int[0], _int[1]):
                
                s_p = np.array([x_a[k], y_a[k], z_a[k]])
                e_point = np.array([x_a[k+1], y_a[k+1], z_a[k+1]])
                
                l_cum += dist_(s_p, e_point)
            
            if l_cum != 0:
                eff.append(s_amp/l_cum)
            
        results =  dict({
            'pursuit efficiency mean': np.mean(np.array(eff)), 
            'pursuit efficiency sd': np.std(np.array(eff)),  
            'raw': np.array(eff)
                }) 
        
        if not get_raw:
            del results["raw"]

        return results
    
    
        
        
def pursuit_count(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_count()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_count()
        pursuit_analysis.verbose()
    return results

 
def pursuit_frequency(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_frequency()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_frequency()
        pursuit_analysis.verbose()
    return results


def pursuit_duration(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_duration(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_duration(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_proportion(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_proportion()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_proportion()
        pursuit_analysis.verbose()
    return results


def pursuit_velocity(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_velocity(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_velocity(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results



def pursuit_velocity_means(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_velocity_means(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_velocity_means(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results



def pursuit_peak_velocity(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_peak_velocity(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_peak_velocity(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_amplitude(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_amplitude(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_amplitude(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_distance(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_distance(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_distance(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_efficiency(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_efficiency(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_efficiency(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results

 

 