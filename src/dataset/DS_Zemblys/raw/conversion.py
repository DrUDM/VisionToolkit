# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

  
def Zemblys():
    
    for i in range(1,7):
        
        print('Pre-processing: lookAtPoint_EL_S{id_}...'.format(id_=i))
        
        name = 'dataset/DS_Zemblys/raw/lookAtPoint_EL_S{id_}.npy'.format(id_ = i)
        seq = np.load(name)
     
        theta_x = np.array([
            seq[i][1] for i in range(len(seq))
            ])
        
        theta_y = np.array([
            seq[i][2] for i in range(len(seq))
            ])
        
        e_ = np.array([
            seq[i][4] for i in range(len(seq))
            ])
        
        df_e = pd.DataFrame(e_, columns = ['event_label'])
     
        # Merge fixations and post-saccadic oscillations
        df_e = df_e.replace(3, 1)
        
        # Not interested in SP
        df_e = df_e.replace(4, 0)
    
        # Merge blinks and undefined
        df_e = df_e.replace(5, 0)
        
        # Here, screen dimensions do not change
        eye_distance = 565.0
        screen_width = 533.0
        screen_height = 301.0
        
        # Save as cartesian coordinates
        x = np.tan(theta_x * (np.pi/180)) * eye_distance + screen_width/2
        y = np.tan(theta_y * (np.pi/180)) * eye_distance + screen_height/2
       
        df_s = pd.DataFrame(np.array([x, y]).T, 
                            columns = ['sig_gaze_points_x','sig_gaze_points_y'])
        
        # Undefined and blinks are replaced by np.nan and interpolated
        df_s.loc[df_e['event_label'] == 0] = np.nan 
        df_s = df_s.interpolate()
       
        noise = np.where(df_e['event_label'] == 0)[0]
        ratio = len(noise)/len(df_e)
        
        if ratio <= 0.10:
            
            df_s.to_csv('dataset/DS_Zemblys/gaze_s{id_}.csv'.format(id_ = i),index=False)
            df_e.to_csv('dataset/DS_Zemblys/labels_s{id_}.csv'.format(id_ = i),index=False)
    
            print('...saved...')
            
        print('...done')
  



