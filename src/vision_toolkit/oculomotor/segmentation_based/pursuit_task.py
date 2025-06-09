# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.optimize as optimize 
import scipy 

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
     
            # Adjust theoretical data indices
            theo_start = max(0, _int[0] - s_idx)
            theo_end = min(len(theo['x']), _int[1] - s_idx + 1)
     
            for _dir in ['x', 'y']:
                l_p_e = pos[_dir][_int[0]: _int[1] + 1]
                l_p_t = theo[_dir][theo_start: theo_end]
    
                # Ensure equal lengths for polynomial fitting
                min_len = min(len(l_p_e), len(l_p_t))
                if min_len < 2:  # Need at least 2 points for polyfit
                    print('4')
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
    
        results = dict({'slope ratios': s_r})
      
        return results
    
    
    
    def pursuit_task_crossing_time(self, tolerance):
        """
        Calculate the first time the eye position matches the theoretical target position within a tolerance.
    
        Parameters
        ----------
        tolerance : float, optional
            Maximum position difference (e.g., degrees or pixels) to consider a match. Default is 1.0.
    
        Returns
        -------
        dict
            Dictionary with keys 'x' and 'y', containing the crossing time (seconds) for each direction.
            Returns None for a direction if no crossing occurs within the pursuit period.
    
        Notes
        -----
        - Based on De Brouwer et al. (2002), position error thresholds of 0.5–2 degrees trigger catch-up saccades.
        - Crossing time is the earliest time the position error falls below the tolerance.
        """
        sampling_frequency = self.config['sampling_frequency']
        time = np.arange(len(self.data_set['x_pursuit'])) / sampling_frequency
        
        crossings = {'x': None, 'y': None}
        
        for direction in ['x', 'y']:
            eye_pos = self.data_set[f'{direction}_pursuit']
            target_pos = self.data_set[f'{direction}_theo_pursuit']
            
            # Ensure equal lengths
            min_len = min(len(eye_pos), len(target_pos))
            eye_pos = eye_pos[:min_len]
            target_pos = target_pos[:min_len]
            
            # Compute absolute position error
            error = np.abs(eye_pos - target_pos)
            
            # Find first index where error is below tolerance
            valid_indices = np.where(error < tolerance)[0]
            if valid_indices.size > 0:
                crossings[direction] = time[valid_indices[0]]
        
        results = dict({'crossing time': crossings})
        
        return results

 
    
    def pursuit_task_overall_gain(self, get_raw):
        
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
     
        gains = [] 
        for _int in _ints:
     
            # Adjust theoretical data indices
            theo_start = max(0, _int[0] - s_idx)
            theo_end = min(len(theo['x']), _int[1] - s_idx + 1)
      
            x_e = pos['x'][theo_start: theo_end]
            y_e = pos['y'][theo_start: theo_end]
        
            x_t = theo['x'][theo_start: theo_end]
            y_t = theo['y'][theo_start: theo_end]
          
            s_e_x = x_e[1:]- x_e[:-1]
            s_e_y = y_e[1:]- y_e[:-1]
            s_t_x = x_t[1:]- x_t[:-1]
            s_t_y = y_t[1:]- y_t[:-1]
            
            e_vel = np.sqrt((s_e_x/d_t)**2 + (s_e_y/d_t)**2)
            t_vel = np.sqrt((s_t_x/d_t)**2 + (s_t_y/d_t)**2)
            
            l_gains = e_vel/t_vel
            gains += list(l_gains)
            
        results = dict({'overall gain': np.mean(gains),
                       'raw': np.array(gains)})
        
        if not get_raw:
            del results["raw"]
            
        return results
           
  
    def pursuit_task_overall_gain_x(self, get_raw):
        
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
   
        gains = [] 
        for _int in _ints:
     
            # Adjust theoretical data indices
            theo_start = max(0, _int[0] - s_idx)
            theo_end = min(len(theo['x']), _int[1] - s_idx + 1)
    
            x_e = pos['x'][theo_start: theo_end]
            x_t = theo['x'][theo_start: theo_end] 
      
            s_e_x = x_e[1:]- x_e[:-1] 
            s_t_x = x_t[1:]- x_t[:-1]
            
            e_vel = (s_e_x/d_t) 
            t_vel = (s_t_x/d_t) 
            
            l_gains = e_vel/t_vel
            gains += list(l_gains)
            
        results = dict({'overall gain x': np.mean(gains),
                       'raw': np.array(gains)})
        
        if not get_raw:
            del results["raw"]
            
        return results
        
    
    def pursuit_task_overall_gain_y(self, get_raw):
        
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
     
        gains = [] 
        for _int in _ints:
     
            # Adjust theoretical data indices
            theo_start = max(0, _int[0] - s_idx)
            theo_end = min(len(theo['y']), _int[1] - s_idx + 1)
      
            
            y_e = pos['y'][theo_start: theo_end] 
            y_t = theo['y'][theo_start: theo_end]
           
            s_e_y = y_e[1:]- y_e[:-1] 
            s_t_y = y_t[1:]- y_t[:-1]
            
            e_vel = s_e_y/d_t 
            t_vel = s_t_y/d_t 
            
            l_gains = e_vel/t_vel
            gains += list(l_gains)
            
        results = dict({'overall gain y': np.mean(gains),
                       'raw': np.array(gains)})
        
        if not get_raw:
            del results["raw"]
            
        return results
    
    
    
    def pursuit_task_slopel_gain(self, _type):
        """
        Calculate the gain of smooth pursuit eye movements in x and y directions.
    
        Parameters
        ----------
        _type : str, optional
            Calculation type: 'mean' (mean of slope ratios), 'weighted' (duration-weighted mean).
            Default is 'weighted'.
    
        Returns
        -------
        dict
            Keys 'x' and 'y' with gain values (float). Returns 0.0 for invalid cases.
    
        Notes
        -----
        Gain is the ratio of eye velocity to target velocity, derived from slope ratios.
        """
        # Check if pursuit intervals are empty; return zero gains if so
        if not self.pursuit_intervals: 
            return {'x': 0.0, 'y': 0.0}
    
        # Retrieve slope ratios from pursuit_task_slope_ratios method
        slope_ratios = self.pursuit_task_slope_ratios()['slope ratios']
    
        # Compute interval durations by adjusting end indices (add 1 to include last point)
        intervals = np.array(self.pursuit_intervals) + np.array([[0, 1]])
        durations = intervals[:, 1] - intervals[:, 0]
        
        # Create mask for valid intervals (duration > 0)
        valid_mask = durations > 0
        if not valid_mask.any():
            return {'x': 0.0, 'y': 0.0}
        valid_durations = durations[valid_mask]
    
        # Initialize dictionary to store gain values for x and y directions
        gains = {}
        for direction in ['x', 'y']:
            # Get slope ratios for the current direction (x or y)
            ratios = slope_ratios.get(direction, np.array([]))
            
            # Validate ratios: check if length matches intervals and all values are finite
            if len(ratios) != len(self.pursuit_intervals) or not np.all(np.isfinite(ratios)):
                gains[direction] = 0.0
                continue
            
            # Filter ratios to only those corresponding to valid intervals
            valid_ratios = ratios[valid_mask] 
            if len(valid_ratios) == 0:
                gains[direction] = 0.0
                continue
            
            # Calculate gain based on specified type
            if _type == 'mean':
                # Basic mode: compute simple mean of valid slope ratios
                gains[direction] = np.mean(valid_ratios)
            elif _type == 'weighted': 
                # Weighted mode: compute duration-weighted mean of slope ratios
                total_duration = np.sum(valid_durations)
                if total_duration == 0:
                    gains[direction] = 0.0
                    continue
                gains['gain ' + direction] = np.sum(valid_durations * valid_ratios) / total_duration
    
        # Return dictionary with gain values for x and y directions
        results = dict({'slope gain': gains})
        
        return results
    
    
    
    def pursuit_task_sinusoidal_phase(self):
        
        def fit_sin(tt, yy):
            '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
            tt = np.array(tt)
            yy = np.array(yy)
            ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
            Fyy = abs(np.fft.fft(yy))
            guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
            guess_amp = np.std(yy) * 2.**0.5
            guess_offset = np.mean(yy)
            guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
        
            def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
            popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
            A, w, p, c = popt
            f = w/(2.*np.pi)
            fitfunc = lambda t: A * np.sin(w*t + p) + c
            return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, 
                    "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), 
                    "rawres": (guess,popt,pcov)}
        
        #theo_start = max(0, _int[0] - s_idx)
        #theo_end = min(len(theo['x']), _int[1] - s_idx + 1)
        
        d_t = 1 / self.config['sampling_frequency']
        len_ = len(self.data_set['x_theo_pursuit']) * d_t
        tt = np.linspace(0, len_ - d_t, len(self.data_set['x_theo_pursuit']))
        
        s_idx = self.config['pursuit_start_idx']
        end_idx = len(self.data_set['x_theo_pursuit']) + s_idx
        
        # Ensure the slice doesn't exceed the length of x_pursuit
        if end_idx > len(self.data_set['x_pursuit']):
            end_idx = len(self.data_set['x_pursuit'])
        
        res1 = fit_sin(tt, self.data_set['x_theo_pursuit'])
        res2 = fit_sin(tt, self.data_set['x_pursuit'][s_idx:end_idx])
           
         
               
        if len(self.data_set['x_pursuit'][s_idx:end_idx]) != len(tt):
            tt_pursuit = np.linspace(0, len(self.data_set['x_pursuit'][s_idx:end_idx]) * d_t - d_t, 
                                    len(self.data_set['x_pursuit'][s_idx:end_idx]))
        else:
            tt_pursuit = tt
        
        # Plotting
        plt.plot(tt_pursuit, self.data_set['x_pursuit'][s_idx:end_idx], "purple", label="Raw pursuit data")
        plt.plot(tt, res1["fitfunc"](tt), "r--", label="Theoretical fit", linewidth=2)
        plt.plot(tt, res2["fitfunc"](tt), "b--", label="Pursuit fit", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title("Pursuit Task Sinusoidal Fit")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.clf()
        
        results = dict({'phase difference': res1['phase'] - res2['phase']})
        
        return results
         
                
        
    def pursuit_task_accuracy(self,pursuit_accuracy_tolerance, _type ):
        
        a_i = np.array(self.pursuit_intervals) + np.array([[0, 1]])
        a_d = np.array(a_i[:,1] - a_i[:,0])
        
        s_r = self.pursuit_task_slope_ratios()
        ac_t = pursuit_accuracy_tolerance
      
        acs = dict({})
        for _dir in ['x', 'y']:
          
            w_b = np.where(s_r['slope ratios'][_dir] < 1+ac_t, 1, 0)*np.where(s_r['slope ratios'][_dir] > 1-ac_t, 1, 0)      
            print(w_b)
            if _type == 'weighted':
                acs[_dir] = np.sum(w_b * a_d)/np.sum(a_d)
               
            elif  _type == 'mean':
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
    
    
    def pursuit_task_entropy(self,
                             pursuit_entropy_window, 
                             pursuit_entropy_tolerance):
        
        
        print(len(self.data_set['x_pursuit']))
        print(len(self.data_set['x_theo_pursuit']))
        
        nb_s_p = self.config['nb_samples_pursuit']
        s_idx = self.config['pursuit_start_idx']
        
        
        
        
        pos_p = np.concatenate((self.data_set['x_pursuit'][s_idx:s_idx+nb_s_p].reshape(1, nb_s_p),
                                self.data_set['y_pursuit'][s_idx:s_idx+nb_s_p].reshape(1, nb_s_p)), axis = 0)
        print(len(pos_p))
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
                                           pursuit_entropy_window,
                                           pursuit_entropy_tolerance)
        
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
    
    return results 


def pursuit_task_velocity(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_velocity()
    
    return results 


def pursuit_task_velocity_means(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_velocity_means()
   
    return results 


def pursuit_task_peak_velocity(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_peak_velocity()
   
    return results 


def pursuit_task_amplitude(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_amplitude()
   
    return results 


def pursuit_task_distance(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_distance()
    
    return results 


def pursuit_task_efficiency(input_df, theoretical_df, 
                       **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_efficiency()
    
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
   
    return results 


def pursuit_task_crossing_time(input_df, theoretical_df, 
                               **kwargs): 
    
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method,
                                   **kwargs_copy)
    tolerance = kwargs.get("tolerance", 1.0)
    results = pursuit_analysis.pursuit_task_crossing_time(tolerance)
   
    return results 
     

def pursuit_task_overall_gain(input_df, theoretical_df, 
                               **kwargs): 
    
    get_raw = kwargs.get("get_raw", True)
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method, 
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_overall_gain(get_raw)
    
    return results 


def pursuit_task_overall_gain_x(input_df, theoretical_df, 
                               **kwargs): 
    
    get_raw = kwargs.get("get_raw", True)
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method, 
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_overall_gain_x(get_raw)
    
    return results 


def pursuit_task_overall_gain_y(input_df, theoretical_df, 
                               **kwargs): 
    
    get_raw = kwargs.get("get_raw", True)
    kwargs_copy = kwargs.copy()
    sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
    segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
    
    pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                   sampling_frequency, segmentation_method, 
                                   **kwargs_copy)
    results = pursuit_analysis.pursuit_task_overall_gain_y(get_raw)
   
    return results 



def pursuit_task_sinusoidal_phase(input_df, theoretical_df, 
                               **kwargs): 

   get_raw = kwargs.get("get_raw", True)
   kwargs_copy = kwargs.copy()
   sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
   segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
   
   pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                  sampling_frequency, segmentation_method, 
                                  **kwargs_copy)
   results = pursuit_analysis.pursuit_task_sinusoidal_phase()
    
   return results


def pursuit_task_accuracy(input_df, theoretical_df, 
                               **kwargs): 

   get_raw = kwargs.get("get_raw", True)
   pursuit_accuracy_tolerance = kwargs.get("pursuit_accuracy_tolerance", .15)
   _type = kwargs.get('_type', 'mean')
   
   kwargs_copy = kwargs.copy()
   sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
   segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
   
   pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                  sampling_frequency, segmentation_method, 
                                  **kwargs_copy)
   results = pursuit_analysis.pursuit_task_accuracy(pursuit_accuracy_tolerance=.15,
                                                    _type='mean')
    
   return results


 

def pursuit_task_entropy(input_df, theoretical_df, 
                               **kwargs): 

   get_raw = kwargs.get("get_raw", True)
   kwargs_copy = kwargs.copy()
   sampling_frequency = kwargs_copy.pop("sampling_frequency", 250)
   segmentation_method = kwargs_copy.pop("segmentation_method", 'I_VMP')
   
   pursuit_analysis = PursuitTask(input_df, theoretical_df, 
                                  sampling_frequency, segmentation_method, 
                                  **kwargs_copy)
   results = pursuit_analysis.pursuit_task_entropy()
    
   return results











