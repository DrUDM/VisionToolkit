# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

from scipy.io import arff
import pandas as pd


 
    
    
def hollywood2():
        
    df_c = pd.read_csv("dataset/DS_Hollywood2/raw/train/config.txt", sep=",", header=None)
    names = df_c.iloc[:,0]
    
    width_n = 475.0
    height_n = 280.0
    
    for i, name in enumerate(names):
       
        print('Pre-processing: {name}...'.format(name=name))
        
        f_name = 'dataset/DS_Hollywood2/raw/train/{name}.arff'.format(name=name)
      
        width_px = df_c.iloc[i,1]
        height_px = df_c.iloc[i,2]
        width_mm = df_c.iloc[i,3]
        height_mm = df_c.iloc[i,4]  
        
        data = arff.loadarff(f_name)
        l_df = pd.DataFrame(data[0])
        
        # Linear conversion from px to mm
        x = np.array(l_df.iloc[:,1]) * (width_mm/width_px)
        y = np.array(l_df.iloc[:,2]) * (height_mm/height_px)
       
        e_ = np.array(l_df.iloc[:,5]) 
        
        df_e = pd.DataFrame(e_, columns = ['event_label'])
        df_e['event_label'] = df_e['event_label'].str.decode('utf-8') 
     
    
        # Replace NOISE, FIX, SACCADE and SP by 0, 1, 2 and 3
        df_e = df_e.replace('NOISE', 0)
        df_e = df_e.replace('FIX', 1)
        df_e = df_e.replace('SACCADE', 2)
        df_e = df_e.replace('SP', 3)
       
        # To be sure all undefined and out of bounds values were labelled as undefined
        t_rb = list(set(
            list(np.where(x<=0)[0]) 
            + list(np.where(y<=0)[0])
            + list(np.where(x>width_mm)[0])
            + list(np.where(y>height_mm)[0]) 
                ))
        x[t_rb] = np.nan
        y[t_rb] = np.nan
        
        t_r = np.argwhere(np.isnan(x)).flatten()
        df_e.iloc[t_r] = 0
         
        x = x - (width_mm/2) + (width_n/2)
        y = y - (height_mm/2) + (height_n/2)
        
        plt.plot(x,y)
        plt.show()
        plt.clf()
        
        df_s = pd.DataFrame(np.array([x, y]).T, 
                            columns = ['sig_gaze_points_x',
                                       'sig_gaze_points_y'])
         
        df_s.loc[df_e['event_label'] == 0] = np.nan 
        
        # Interpolation that is usefull for my visual features
        df_s = df_s.interpolate()
      
        
        noise = np.where(df_e['event_label'] == 0)[0]
        ratio = len(noise)/len(df_e)
        
        if ratio <= 0.05:
            
            df_s.to_csv('dataset/DS_Hollywood2/gaze_s{id_}.csv'.format(id_ = i+1),index=False)
            df_e.to_csv('dataset/DS_Hollywood2/labels_s{id_}.csv'.format(id_ = i+1),index=False)
            
            print('...saved...')
            
        print('...done')
  
    
 









