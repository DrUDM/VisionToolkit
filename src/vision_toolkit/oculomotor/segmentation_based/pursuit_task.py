# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from vision_toolkit.segmentation.processing.ternary_segmentation import TernarySegmentation 
from vision_toolkit.utils.segmentation_utils import interval_merging 


from matplotlib import pyplot as plt 

 
class PursuitTask(TernarySegmentation):
    
    def __init__(self, 
                 input_df, theoretical_df, 
                 sampling_frequency, segmentation_method = 'I_VMP', 
                 **kwargs
                 ):
        
        super().__init__(input_df, 
                         sampling_frequency, segmentation_method, tasks = ['pursuit'],
                         **kwargs)
        
        
        self.process()
        
       
        events = self.get_events(labels=True)
        #print(events)
    
    
        s_idx = self.config['pursuit_start_idx']     
        e_idx = self.config['pursuit_end_idx']
        t_df = pd.read_csv(theoretical_df)
        
        nb_s_p = len(np.array(t_df.iloc[:,0]))
        self.config.update({'nb_samples_pursuit': nb_s_p})
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        
        self.data_set.update({
            'x_pursuit': x_a,#[s_idx: s_idx+nb_s_p],
            'y_pursuit': y_a,#[s_idx: s_idx+nb_s_p],
            'x_theo_pursuit': np.array(t_df.iloc[:,0]),
            'y_theo_pursuit': np.array(t_df.iloc[:,1])
                })
        
      
      
        self.pursuit_intervals = self.segmentation_results['pursuit_intervals']
         
        
        plt.plot(self.data_set['x_theo_pursuit'])
        plt.xlabel("Time (ms)", fontsize=14)
        plt.ylabel("Horizontal position (px)", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.show()
        plt.clf()
        
        plt.plot(self.data_set['y_theo_pursuit'])
        plt.xlabel("Time (ms)", fontsize=14)
        plt.ylabel("Horizontal position (px)", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.show()
        plt.clf()
 
     
    
    
    def pursuit_task_count(self):
        
        ct = len(self.pursuit_intervals)
        results = dict({"count": ct})
        
        return results
    

    def pursuit_task_frequency(self):

        ct = len(self.pursuit_intervals)
        f = ct/(self.config['nb_samples_pursuit']/self.config['sampling_frequency'])

        result = dict({"frequency": f})

        return result
    
    
    def pursuit_task_duration(self):
        
        a_i = np.array(self.pursuit_intervals) + np.array([[0, 1]])
        a_d = a_i[:,1] - a_i[:,0]
        
        result =  dict({
            'duration mean': np.mean(a_d), 
            'duration sd': np.std(a_d),  
            'raw': a_d
                })
        
        return result
    
      
    def pursuit_task_proportion(self):
        """
        Calculate the proportion of time spent in pursuit intervals relative to the total duration.
    
        Returns
        -------
        dict
            A dictionary containing the proportion of pursuit time (unitless, 0 to 1) with key 'task proportion'.
            Returns 0.0 if there are no valid pursuit intervals or if the total duration is zero.
    
        Notes
        -----
        - Pursuit intervals are adjusted to include the end sample (end index + 1).
        - Total duration is determined by pursuit_end_idx - pursuit_start_idx if provided,
          otherwise by the length of the theoretical pursuit data.
        """
        
        if not self.pursuit_intervals:
            return {'task proportion': 0.0}
    
        # Convert intervals to NumPy array and adjust end index to include last sample
        intervals = np.array(self.pursuit_intervals) + np.array([[0, 1]])
    
        # Filter out invalid intervals (end < start)
        valid_intervals = intervals[intervals[:, 1] > intervals[:, 0]]
        if valid_intervals.size == 0:
            return {'task proportion': 0.0}
    
        # Calculate total duration of pursuit intervals in samples
        total_pursuit_duration = (valid_intervals[:, 1] - valid_intervals[:, 0]).sum()
    
        # Determine total duration of the pursuit period
        if self.config.get('pursuit_end_idx') is not None:
            total_duration = self.config['pursuit_end_idx'] - self.config['pursuit_start_idx']
        else:
            total_duration = len(self.data_set['x_theo_pursuit'])
    
        # Validate total duration
        if total_duration <= 0:
            print("Warning: Total duration is zero or negative, returning proportion 0.0")
            return {'task proportion': 0.0}
    
        # Compute proportion
        proportion = total_pursuit_duration / total_duration
    
        result = dict({'task proportion': float(proportion)})
        
        return result
    
  
    
    def pursuit_task_velocity(self):
        
        _ints = self.pursuit_intervals 
        a_sp = self.data_set['absolute_speed']
        
        l_sp = [] 
        for _int in _ints:       
            l_sp.extend(list(a_sp[_int[0]: _int[1]+1])) 
       
        return dict({
            'velocity mean': np.mean(np.array(l_sp)), 
            'velocity sd': np.std(np.array(l_sp)),  
            'raw': np.array(l_sp)
                })
    
    
    def pursuit_task_velocity_means(self):
        
        _ints = self.pursuit_intervals
        a_sp = self.data_set['absolute_speed']
        
        m_sp = []
        for _int in _ints:       
            m_sp.append(np.mean(a_sp[_int[0]: _int[1]+1])) 
       
        return dict({
            'velocity mean mean': np.mean(np.array(m_sp)), 
            'velocity mean sd': np.std(np.array(m_sp)),  
            'raw': np.array(m_sp)
                })
    
    
    def pursuit_task_peak_velocity(self):
        
        _ints = self.pursuit_intervals  
        a_sp = self.data_set['absolute_speed']
        
        p_sp = []
        
        for _int in _ints:       
            p_sp.append(np.max(a_sp[_int[0]: _int[1]+1])) 
       
        return dict({
            'velocity peak mean': np.mean(np.array(p_sp)), 
            'velocity peak sd': np.std(np.array(p_sp)),  
            'raw': np.array(p_sp)
                }) 
    

    def pursuit_task_amplitude(self):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.pursuit_intervals
        dist_ = self.distances[self.config['distance_type']]
        
        dsp = []
        for _int in _ints:
            
            s_p = np.array([x_a[_int[0]], y_a[_int[0]], z_a[_int[0]]])
            e_p = np.array([x_a[_int[1]], y_a[_int[1]], z_a[_int[1]]])
            
            dsp.append(dist_(s_p, e_p))
        
        return dict({
            'pursuit amplitude mean': np.mean(np.array(dsp)), 
            'pursuit amplitude sd': np.std(np.array(dsp)),  
            'raw': np.array(dsp)
                })
    
    
    def pursuit_task_distance(self):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.pursuit_intervals
        dist_ = self.distances[self.config['distance_type']]
        
        t_cum = []
        for _int in _ints:
            
            l_cum = 0
            for k in range (_int[0], _int[1]):
                
                s_p = np.array([x_a[k], y_a[k], z_a[k]])
                e_p = np.array([x_a[k+1], y_a[k+1], z_a[k+1]])
                
                l_cum += dist_(s_p, e_p)
                
            t_cum.append(l_cum)
            
        return dict({
            'pursuit cumul. distance mean': np.mean(np.array(t_cum)), 
            'pursuit cumul. distance sd': np.std(np.array(t_cum)),  
            'raw': np.array(t_cum)
                })
    
    
    def pursuit_task_efficiency(self):
        
        x_a = self.data_set['x_array']
        y_a = self.data_set['y_array']
        z_a = self.data_set['z_array']
        
        _ints = self.pursuit_intervals 
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
            
        return dict({
            'pursuit efficiency mean': np.mean(np.array(eff)), 
            'pursuit efficiency sd': np.std(np.array(eff)),  
            'raw': np.array(eff)
                }) 
    
     
    def pursuit_task_slope_ratios(self):
        _ints = self.pursuit_intervals
        d_t = 1 / self.config['sampling_frequency']
        s_idx = self.config['pursuit_start_idx']
    
        pos = dict({
            'x': self.data_set['x_pursuit'],
            'y': self.data_set['y_pursuit'],
        })
        theo = dict({
            'x': self.data_set['x_theo_pursuit'],
            'y': self.data_set['y_theo_pursuit'],  # Fixed bug: was x_theo_pursuit
        })
    
        s_r = dict({'x': [], 'y': []})
    
        for _int in _ints:
            # Validate interval
            if _int[0] >= _int[1]:
                print(f"Skipping invalid interval: {_int}")
                continue
    
            # Ensure interval indices are within bounds
            if _int[0] < 0 or _int[1] >= len(pos['x']):
                print(f"Skipping out-of-bounds interval: {_int}")
                continue
    
            # Adjust theoretical data indices
            theo_start = max(0, _int[0] - s_idx)
            theo_end = _int[1] - s_idx + 1
    
            # Ensure theoretical indices are valid
            if theo_start >= theo_end or theo_end > len(theo['x']):
                print(f"Skipping invalid theoretical indices: [{theo_start}, {theo_end}]")
                continue
    
            for _dir in ['x', 'y']:
                l_p_e = pos[_dir][_int[0]: _int[1] + 1]
                l_p_t = theo[_dir][theo_start: theo_end]
    
                # Ensure equal lengths for polynomial fitting
                min_len = min(len(l_p_e), len(l_p_t))
                if min_len < 2:  # Need at least 2 points for polyfit
                    print(f"Skipping interval with insufficient length: {_int}, min_len={min_len}")
                    continue
    
                l_p_e = l_p_e[:min_len]
                l_p_t = l_p_t[:min_len]
                l_x = np.arange(min_len) * d_t
                
                plt.plot(l_p_e)
                plt.show()
                plt.clf()
                
                plt.plot(l_p_t)
                plt.show()
                plt.clf()
    
                try:
                    slope_e = np.polyfit(l_x, l_p_e, deg=1)[0]
                    slope_t = np.polyfit(l_x, l_p_t, deg=1)[0]
                    if slope_t != 0:  # Avoid division by zero
                        s_r[_dir].append(slope_e / slope_t)
                    else:
                        print(f"Skipping interval with zero theoretical slope: {_int}")
                except Exception as e:
                    print(f"Error in polyfit for interval {_int}, dir {_dir}: {str(e)}")
    
        # Convert lists to arrays, handle empty cases
        for _dir in ['x', 'y']:
            s_r[_dir] = np.array(s_r[_dir]) if s_r[_dir] else np.array([])
    
        result = dict({'slope ratio': s_r})
        return result
    
    
    def pursuit_task_gain(self, _type = 'weighted'):
        
        s_r = self.slope_ratios()
        
        a_i = np.array(self.pursuit_intervals) + np.array([[0, 1]])
        a_d = np.array(a_i[:,1] - a_i[:,0])
      
        if _type == 'adjusted':
            p_p = self.proportion()
            
        gs = dict({})
        for _dir in ['x', 'y']:
            
            if _type == 'basic':
                gs[_dir] = np.mean(s_r[_dir])
            
            else:
                
                l_g = np.sum(a_d * s_r[_dir])/np.sum(a_d)
                if _type == 'weighted':
                    gs[_dir] = l_g 
                      
                else:
                    gs[_dir] = l_g * p_p
 
        return gs
    
    
    def pursuit_task_accuracy(self, _type = 'weighted'):
        
        a_i = np.array(self.pursuit_intervals) + np.array([[0, 1]])
        a_d = np.array(a_i[:,1] - a_i[:,0])
        
        s_r = self.slope_ratios()
        ac_t = self.config['pursuit_accuracy_tolerance']
        
        acs = dict({})
        for _dir in ['x', 'y']:
            
            w_b = np.where(s_r[_dir] < 1+ac_t, 1, 0)*np.where(s_r[_dir] > 1-ac_t, 1, 0)      
            
            if _type == 'weighted':
                acs[_dir] = np.sum(w_b * a_d)/np.sum(a_d)
               
            else:
                acs[_dir] = np.mean(w_b)
            
        return acs
    
  
    def ap_entropy (self, diff_vec, w_s, t_eps):
        
        n_s = len(diff_vec)
        x_m = np.zeros((n_s-w_s+1, w_s))
        x_mp = np.zeros((n_s-w_s, w_s+1))
        
        for i in range (n_s - w_s + 1):
            
            x_m[i] = diff_vec[i: i + w_s]
            if i < n_s - w_s:
                x_mp[i] = diff_vec[i: i+ w_s +1]
    
        C_m = np.zeros(n_s - w_s + 1)
        C_mp = np.zeros(n_s - w_s)
        
        for i in range(n_s - w_s + 1):
            
            d = abs(x_m - x_m[i])
            d_m = np.sum(np.max(d, axis = 1) < t_eps) 
            C_m[i] = d_m/(n_s - w_s + 1)
    
        for i in range(n_s - w_s):
            
            d = abs(x_mp - x_mp[i])
            d_mp = np.sum(np.max(d, axis = 1) < t_eps) 
            C_mp[i] = d_mp/(n_s - w_s)
    
        entropy = np.sum(np.log(C_m))/len(C_m) - np.sum(np.log(C_mp))/len(C_mp)

        return entropy
    
    
    def pursuit_task_entropy(self):
        
        nb_s_p = self.config['nb_samples_pursuit']
        
        pos_p = np.concatenate((self.data_set['x_pursuit'].reshape(1, nb_s_p),
                                self.data_set['y_pursuit'].reshape(1, nb_s_p)), axis = 0)

        sp_p = np.zeros_like(pos_p)
        sp_p[:,:-1] = ((pos_p[:,1:] - pos_p[:,:-1])
                     *self.config['sampling_frequency'])
        
        theo_p = np.concatenate((self.data_set['x_theo_pursuit'].reshape(1, nb_s_p),
                                 self.data_set['y_theo_pursuit'].reshape(1, nb_s_p)), axis = 0)

        sp_t = np.zeros_like(theo_p)
        sp_t[:,:-1] = ((theo_p[:,1:] - theo_p[:,:-1])
                     *self.config['sampling_frequency']) 
        
        app_en=dict({})
        for k, _dir in enumerate(['x', 'y']):
            
            d_s_v = sp_p[k,:] - sp_t[k,:]
            app_en[_dir] = self.ap_entropy(d_s_v, 
                                           self.config['pursuit_entropy_window'],
                                           self.config['pursuit_entropy_tolerance'])
        
        return app_en
    
    
    def pursuit_task_cross_correlation(self):
        
        pos = dict({
            'x': self.data_set['x_pursuit'],
            'y': self.data_set['y_pursuit'],
                })
        
        theo = dict({
            'x': self.data_set['x_theo_pursuit'],
            'y': self.data_set['x_theo_pursuit'],
                })
        
        c_cr = dict({})
        for _dir in ['x', 'y']:
        
            n_p = (pos[_dir] - np.mean(pos[_dir])) / (np.std(pos[_dir]))
            n_t = (theo[_dir] - np.mean(theo[_dir])) / (np.std(theo[_dir]))
        
            c_cr[_dir] = np.correlate(n_p, 
                                      n_t) / max(len(n_p), 
                                                 len(n_t))
                                           
        return c_cr
    
    
    def pursuit_task_onset(self):
        
        o_bl = self.config['pursuit_onset_baseline_length']
        o_sl = self.config['pursuit_onset_slope_length']
        o_t = self.config['pursuit_onset_threshold']
        
        s_f = self.config['sampling_frequency']
        d_t = 1/s_f
        
        nb_s_p = self.config['nb_samples_pursuit']
        pos_p = np.concatenate((self.data_set['x_pursuit'].reshape(1, nb_s_p),
                                self.data_set['y_pursuit'].reshape(1, nb_s_p)), axis = 0)

        sp_p = np.zeros_like(pos_p)
        sp_p[:,:-1] = ((pos_p[:,1:] - pos_p[:,:-1])
                     *self.config['sampling_frequency'])
        
        #set number of points corresponding to baseline and slope lengths
        nb_o_bl = round(o_bl/1000 * s_f)
        nb_o_sl = round(o_sl/1000 * s_f)
        
        #create corresponding x points
        b_x = np.arange(nb_o_bl)*d_t
        s_x = np.arange(nb_o_sl)*d_t
        
        onsets=dict({})
        for k, _dir in enumerate(['x', 'y']):
            
            #value threshold wrt number of sd param
            o_t_v = o_t * np.std(sp_p[k,:nb_o_bl])
            
            #find first point above threshold
            start_s = np.argmax(sp_p[k] > o_t_v)
            
            #fit baseline and pursuit portion
            coefs_b = np.polyfit(b_x, sp_p[k,:nb_o_bl], deg=1)
            coefs_s = np.polyfit(s_x + start_s*d_t, sp_p[k, start_s: start_s+nb_o_sl], deg=1)
            
            #find crossing point as onset time
            onsets[_dir] = (coefs_s[1] - coefs_b[1])/(coefs_b[0] - coefs_s[0]) 

        return onsets
        

    
def pursuit_task_count(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_count()
    pursuit_analysis.verbose()
    return results


def pursuit_task_frequency(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_frequency()
    pursuit_analysis.verbose()
    return results



def pursuit_task_duration(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_duration()
    pursuit_analysis.verbose()
    return results


def pursuit_task_proportion(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_proportion()
    pursuit_analysis.verbose()
    return results 


def pursuit_task_slope_ratios(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_slope_ratios()
    pursuit_analysis.verbose()
    return results 









     